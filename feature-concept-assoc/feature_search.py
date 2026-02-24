#!/usr/bin/env python3
"""
Promoter feature search (InterPLM-style) for HyenaDNA + SAE (layer 5).

Goal
- Sample 500 positive 2kb windows centered on a promoter interval midpoint.
- Sample 500 negative 2kb windows with NO overlap with any promoter interval.
- Extract HyenaDNA layer-5 token embeddings (1 base = 1 token).
- Compute SAE feature activations for:
    * all bases within the promoter interval (positives)
    * matched-count random bases from negative windows (negatives)
- Compute feature–concept association (precision + domain-adjusted recall + best-F1 over thresholds),
  analogous to InterPLM’s “domain-aware” evaluation for region concepts.

Inputs
- hg38 FASTA
- promoter BED (0-based, half-open). You already have: annotations/gencode/promoter_TSS_1000up_100down.bed
- SAE checkpoint trained on HyenaDNA layer 5 embeddings
- HyenaDNA model id

Outputs
- out_dir/promoter_eval_summary.csv : per-feature best threshold + precision/recall/F1 + some diagnostics
- out_dir/promoter_eval_details.pt  : saved tensors for further analysis (optional)

Notes
- Assumes HyenaDNA tokenizer produces exactly seq_len tokens with add_special_tokens=False.
- Handles promoter intervals longer/shorter than window center; uses interval intersection per sampled window.
- Domain-adjusted recall: each sampled promoter interval counts as one domain; recalled if any base inside it is predicted positive.

Usage example
python promoter_feature_search.py \
  --fasta data/hg38.primary.fa \
  --promoter_bed data/annotations/gencode/promoter_TSS_1000up_100down.bed \
  --sae_ckpt runs/sae/<your_run>/ckpt_step_XXXXXXX.pt \
  --out_dir runs/promoter_feature_search \
  --model_id LongSafari/hyenadna-large-1m-seqlen-hf \
  --layer 5 \
  --seq_len 2000 \
  --n_pos 500 --n_neg 500 \
  --seed 42 \
  --device cuda
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from pyfaidx import Fasta
except ImportError as e:
    raise SystemExit("Missing dependency pyfaidx. Install: pip install pyfaidx") from e

try:
    from intervaltree import Interval, IntervalTree
except ImportError as e:
    raise SystemExit("Missing dependency intervaltree. Install: pip install intervaltree") from e


# ----------------------------
# Data types
# ----------------------------
BedRow = Tuple[str, int, int]  # (chrom, start, end)


@dataclass(frozen=True)
class WindowSample:
    chrom: str
    start: int
    end: int
    # For positive windows, this is the specific promoter interval used to center the window.
    prom_start: Optional[int] = None
    prom_end: Optional[int] = None


# ----------------------------
# SAE model (match your training code)
# ----------------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, use_relu: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.use_relu = use_relu

        self.W_e = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_e = nn.Parameter(torch.zeros(d_hidden))
        self.W_d = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_d = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_e + self.b_e
        return F.relu(pre) if self.use_relu else pre

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_d.T + self.b_d


def load_sae(ckpt_path: str, device: torch.device) -> SparseAutoencoder:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    d_in = int(ckpt["d_in"])
    d_hidden = int(ckpt["d_hidden"])
    model = SparseAutoencoder(d_in=d_in, d_hidden=d_hidden, use_relu=True)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return model


# ----------------------------
# BED + interval trees
# ----------------------------
def read_bed3(path: str) -> List[BedRow]:
    rows: List[BedRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            if end <= start:
                continue
            rows.append((chrom, start, end))
    return rows


def build_interval_trees(rows: List[BedRow]) -> Dict[str, IntervalTree]:
    trees: Dict[str, IntervalTree] = {}
    for chrom, s, e in rows:
        if chrom not in trees:
            trees[chrom] = IntervalTree()
        trees[chrom].add(Interval(s, e))
    # Normalize/merge overlaps for faster overlap checks
    for t in trees.values():
        t.merge_overlaps()
    return trees


def any_overlap(trees: Dict[str, IntervalTree], chrom: str, start: int, end: int) -> bool:
    t = trees.get(chrom)
    if t is None:
        return False
    return len(t.overlap(start, end)) > 0


def intersect_intervals(trees: Dict[str, IntervalTree], chrom: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Return list of (s,e) intersections with [start,end) in genomic coordinates."""
    t = trees.get(chrom)
    if t is None:
        return []
    hits = sorted(t.overlap(start, end), key=lambda iv: iv.begin)
    out: List[Tuple[int, int]] = []
    for iv in hits:
        s = max(start, int(iv.begin))
        e = min(end, int(iv.end))
        if e > s:
            out.append((s, e))
    return out


# ----------------------------
# Sampling windows
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chrom_sizes_from_fasta(genome: Fasta) -> Dict[str, int]:
    # pyfaidx exposes keys and lengths
    sizes: Dict[str, int] = {}
    for chrom in genome.keys():
        sizes[chrom] = len(genome[chrom])
    return sizes


def choose_weighted_chrom(chrom_sizes: Dict[str, int]) -> str:
    # Weighted by length
    chroms = list(chrom_sizes.keys())
    weights = [chrom_sizes[c] for c in chroms]
    total = float(sum(weights))
    r = random.random() * total
    acc = 0.0
    for c, w in zip(chroms, weights):
        acc += float(w)
        if r <= acc:
            return c
    return chroms[-1]


def sample_positive_windows_centered(
    promoter_rows: List[BedRow],
    chrom_sizes: Dict[str, int],
    seq_len: int,
    n_pos: int,
) -> List[WindowSample]:
    half = seq_len // 2
    out: List[WindowSample] = []

    # Shuffle promoters and cycle if needed
    pool = promoter_rows[:]
    random.shuffle(pool)
    i = 0
    attempts = 0
    max_attempts = n_pos * 50

    while len(out) < n_pos and attempts < max_attempts:
        attempts += 1
        if i >= len(pool):
            random.shuffle(pool)
            i = 0
        chrom, ps, pe = pool[i]
        i += 1

        mid = (ps + pe) // 2
        ws = mid - half
        we = ws + seq_len

        size = chrom_sizes.get(chrom)
        if size is None:
            continue
        if ws < 0 or we > size:
            continue

        out.append(WindowSample(chrom=chrom, start=ws, end=we, prom_start=ps, prom_end=pe))

    if len(out) < n_pos:
        raise RuntimeError(f"Could only sample {len(out)}/{n_pos} positive windows (check promoter BED / contigs).")
    return out


def sample_negative_windows_no_promoter(
    promoter_trees: Dict[str, IntervalTree],
    chrom_sizes: Dict[str, int],
    seq_len: int,
    n_neg: int,
) -> List[WindowSample]:
    out: List[WindowSample] = []
    attempts = 0
    max_attempts = n_neg * 200

    while len(out) < n_neg and attempts < max_attempts:
        attempts += 1
        chrom = choose_weighted_chrom(chrom_sizes)
        size = chrom_sizes[chrom]
        if size <= seq_len:
            continue
        ws = random.randrange(0, size - seq_len)
        we = ws + seq_len

        if any_overlap(promoter_trees, chrom, ws, we):
            continue

        out.append(WindowSample(chrom=chrom, start=ws, end=we))

    if len(out) < n_neg:
        raise RuntimeError(f"Could only sample {len(out)}/{n_neg} negative windows after {attempts} attempts.")
    return out


# ----------------------------
# HyenaDNA embedding extraction for selected windows
# ----------------------------
def sanitize_dna(seq: str) -> str:
    seq = seq.upper()
    allowed = {"A", "C", "G", "T", "N"}
    if all(c in allowed for c in seq):
        return seq
    return "".join(c if c in allowed else "N" for c in seq)


def fetch_seq(genome: Fasta, chrom: str, start: int, end: int) -> str:
    s = str(genome[chrom][start:end])  # pyfaidx slicing is 0-based [start:end)
    return sanitize_dna(s)


@torch.inference_mode()
def embed_windows_layer(
    genome: Fasta,
    windows: List[WindowSample],
    tok,
    model,
    layer: int,
    seq_len: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, List[WindowSample]]:
    """
    Returns:
      emb_all: [N, L, D] on CPU float32
      windows: aligned list
    """
    embs: List[torch.Tensor] = []
    aligned: List[WindowSample] = []

    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        seqs = [fetch_seq(genome, w.chrom, w.start, w.end) for w in batch]
        inputs = tok(seqs, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)
        if inputs["input_ids"].shape[1] != seq_len:
            raise RuntimeError(
                f"Tokenized length {inputs['input_ids'].shape[1]} != expected {seq_len}. "
                f"Check tokenizer behavior / special tokens."
            )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        if layer < 0 or layer >= len(hs):
            raise RuntimeError(f"Requested layer {layer}, but hidden_states has len={len(hs)}")

        emb = hs[layer].detach().to("cpu", dtype=torch.float32)  # [B,L,D]
        embs.append(emb)
        aligned.extend(batch)

    emb_all = torch.cat(embs, dim=0)
    return emb_all, aligned


# ----------------------------
# Collect activations for positives (promoter bases) and negatives (random bases)
# ----------------------------
@torch.inference_mode()
def encode_in_chunks(sae: SparseAutoencoder, x: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """
    x: [N, d_in] on device
    Returns a: [N, d_hidden] on CPU float32
    """
    out_chunks: List[torch.Tensor] = []
    for i in range(0, x.shape[0], chunk):
        a = sae.encode(x[i:i + chunk])
        out_chunks.append(a.detach().to("cpu", dtype=torch.float32))
    return torch.cat(out_chunks, dim=0)


def promoter_mask_for_window(
    promoter_trees: Dict[str, IntervalTree],
    w: WindowSample,
    seq_len: int,
) -> torch.Tensor:
    """
    Returns boolean mask [L] for bases in ANY promoter interval overlapping the window.
    """
    hits = intersect_intervals(promoter_trees, w.chrom, w.start, w.end)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for s, e in hits:
        ls = s - w.start
        le = e - w.start
        if le > ls:
            mask[ls:le] = True
    return mask


def collect_activation_sets(
    emb_pos: torch.Tensor,  # [Npos, L, D] CPU
    win_pos: List[WindowSample],
    emb_neg: torch.Tensor,  # [Nneg, L, D] CPU
    win_neg: List[WindowSample],
    promoter_trees: Dict[str, IntervalTree],
    sae: SparseAutoencoder,
    device: torch.device,
    seq_len: int,
    neg_per_pos_base: float = 1.0,
    neg_sample_mode: str = "matched_count",  # matched_count or per_window
    per_window_neg_tokens: int = 100,
) -> Dict[str, object]:
    """
    Build:
      - A_pos_all: activations for all promoter bases across positive windows (concatenated)  [NposBases, F]
      - A_neg_all: activations for sampled negative bases across negative windows            [NnegBases, F]
      - pos_domain_slices: for each positive window/domain, slice indices into A_pos_all to test "any-hit" for recall
    """
    # Compute promoter masks for each positive window
    pos_masks: List[torch.Tensor] = []
    pos_counts: List[int] = []
    for w in win_pos:
        m = promoter_mask_for_window(promoter_trees, w, seq_len)
        c = int(m.sum().item())
        if c == 0:
            # This can happen if the promoter interval used to center doesn't overlap after clipping/merging.
            # Skip such window by treating as 0; it will reduce effective domains slightly.
            pass
        pos_masks.append(m)
        pos_counts.append(c)

    # Gather positive embeddings (only promoter bases) + domain slices
    pos_slices: List[Tuple[int, int]] = []
    x_pos_list: List[torch.Tensor] = []
    cursor = 0
    for i in range(len(win_pos)):
        m = pos_masks[i]
        if int(m.sum().item()) == 0:
            pos_slices.append((cursor, cursor))
            continue
        x = emb_pos[i, m, :]  # [Mi, D]
        x_pos_list.append(x)
        pos_slices.append((cursor, cursor + x.shape[0]))
        cursor += x.shape[0]

    if cursor == 0:
        raise RuntimeError("No promoter-labeled bases found in sampled positive windows.")

    X_pos = torch.cat(x_pos_list, dim=0)  # CPU [NposBases, D]

    # Decide negative sample count
    if neg_sample_mode == "matched_count":
        n_neg = int(math.ceil(cursor * neg_per_pos_base))
    elif neg_sample_mode == "per_window":
        n_neg = int(len(win_neg) * per_window_neg_tokens)
    else:
        raise ValueError("neg_sample_mode must be matched_count or per_window")

    # Sample negative base positions uniformly across negative windows
    nwin_neg = emb_neg.shape[0]
    neg_win_ids = [random.randrange(nwin_neg) for _ in range(n_neg)]
    neg_pos_ids = [random.randrange(seq_len) for _ in range(n_neg)]

    X_neg = emb_neg[torch.tensor(neg_win_ids), torch.tensor(neg_pos_ids), :]  # CPU [NnegBases, D]

    # Move to device and encode in chunks
    X_pos_dev = X_pos.to(device=device, dtype=torch.float32, non_blocking=True)
    X_neg_dev = X_neg.to(device=device, dtype=torch.float32, non_blocking=True)

    A_pos = encode_in_chunks(sae, X_pos_dev)  # CPU [NposBases, F]
    A_neg = encode_in_chunks(sae, X_neg_dev)  # CPU [NnegBases, F]

    return {
        "A_pos": A_pos,
        "A_neg": A_neg,
        "pos_domain_slices": pos_slices,  # list of (start,end) into A_pos
        "pos_counts": pos_counts,
        "neg_win_ids": neg_win_ids,
        "neg_pos_ids": neg_pos_ids,
    }


# ----------------------------
# InterPLM-style association: threshold sweep, precision, domain-recall, best-F1
# ----------------------------
def quantile_thresholds(x: torch.Tensor, qs: List[float]) -> List[float]:
    # x: [N] CPU float32
    # Use torch.quantile for stable thresholds; clamp qs within [0,1]
    qq = torch.tensor([min(max(q, 0.0), 1.0) for q in qs], dtype=torch.float32)
    return [float(v.item()) for v in torch.quantile(x, qq)]


def evaluate_feature_for_promoter(
    a_pos: torch.Tensor,  # [NposBases]
    a_neg: torch.Tensor,  # [NnegBases]
    pos_domain_slices: List[Tuple[int, int]],
    thresholds: List[float],
) -> Dict[str, float]:
    best = {
        "best_f1": 0.0,
        "best_t": float("nan"),
        "best_precision": 0.0,
        "best_domain_recall": 0.0,
        "best_tp": 0.0,
        "best_fp": 0.0,
    }

    # Precompute for speed
    a_pos = a_pos.contiguous()
    a_neg = a_neg.contiguous()

    for t in thresholds:
        pred_pos = a_pos >= t
        pred_neg = a_neg >= t

        tp = int(pred_pos.sum().item())
        fp = int(pred_neg.sum().item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Domain-adjusted recall: each promoter domain is recalled if any token within domain is predicted positive
        recalled = 0
        total_domains = 0
        for s, e in pos_domain_slices:
            total_domains += 1
            if e <= s:
                continue
            if bool(pred_pos[s:e].any().item()):
                recalled += 1
        domain_recall = recalled / total_domains if total_domains > 0 else 0.0

        if precision + domain_recall > 0:
            f1 = 2.0 * precision * domain_recall / (precision + domain_recall)
        else:
            f1 = 0.0

        if f1 > best["best_f1"]:
            best.update(
                best_f1=float(f1),
                best_t=float(t),
                best_precision=float(precision),
                best_domain_recall=float(domain_recall),
                best_tp=float(tp),
                best_fp=float(fp),
            )

    return best


def run_promoter_feature_search(
    A_pos: torch.Tensor,  # [NposBases, F] CPU
    A_neg: torch.Tensor,  # [NnegBases, F] CPU
    pos_domain_slices: List[Tuple[int, int]],
    out_csv: str,
    q_thresholds: List[float],
    max_features: Optional[int] = None,
) -> None:
    npos, Fdim = A_pos.shape
    nneg, Fdim2 = A_neg.shape
    if Fdim != Fdim2:
        raise ValueError("A_pos/A_neg feature dims mismatch")

    F_use = Fdim if max_features is None else min(Fdim, max_features)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "feature_id",
                "best_t",
                "best_f1",
                "best_precision",
                "best_domain_recall",
                "tp",
                "fp",
                "npos_bases",
                "nneg_bases",
            ],
        )
        w.writeheader()

        for j in range(F_use):
            a_pos_j = A_pos[:, j]
            a_neg_j = A_neg[:, j]

            # Choose thresholds from combined distribution (pos+neg) quantiles
            combined = torch.cat([a_pos_j, a_neg_j], dim=0)
            thresholds = quantile_thresholds(combined, q_thresholds)
            thresholds = sorted(set(thresholds))  # unique & sorted

            best = evaluate_feature_for_promoter(
                a_pos=a_pos_j,
                a_neg=a_neg_j,
                pos_domain_slices=pos_domain_slices,
                thresholds=thresholds,
            )

            w.writerow(
                {
                    "feature_id": j,
                    "best_t": best["best_t"],
                    "best_f1": best["best_f1"],
                    "best_precision": best["best_precision"],
                    "best_domain_recall": best["best_domain_recall"],
                    "tp": int(best["best_tp"]),
                    "fp": int(best["best_fp"]),
                    "npos_bases": npos,
                    "nneg_bases": nneg,
                }
            )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--promoter_bed", required=True)
    ap.add_argument("--sae_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--model_id", default="LongSafari/hyenadna-large-1m-seqlen-hf")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--seq_len", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--n_pos", type=int, default=500)
    ap.add_argument("--n_neg", type=int, default=500)

    ap.add_argument("--neg_per_pos_base", type=float, default=1.0, help="If matched_count, negatives = this * #pos_bases")
    ap.add_argument("--neg_sample_mode", choices=["matched_count", "per_window"], default="matched_count")
    ap.add_argument("--per_window_neg_tokens", type=int, default=100, help="If per_window, negatives per neg window")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)

    ap.add_argument(
        "--q_thresholds",
        default="0.90,0.95,0.97,0.99,0.995,0.999",
        help="Comma-separated quantiles for threshold sweep",
    )
    ap.add_argument("--max_features", type=int, default=0, help="0 = all features; else cap for quick runs")
    ap.add_argument("--save_tensors", action="store_true", help="Save activations and slices to a .pt file")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load genome + sizes
    genome = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    chrom_sizes = chrom_sizes_from_fasta(genome)

    # Load promoter intervals
    promoter_rows = read_bed3(args.promoter_bed)
    if not promoter_rows:
        raise SystemExit(f"No rows in {args.promoter_bed}")
    promoter_trees = build_interval_trees(promoter_rows)

    # Sample windows
    pos_windows = sample_positive_windows_centered(
        promoter_rows=promoter_rows,
        chrom_sizes=chrom_sizes,
        seq_len=args.seq_len,
        n_pos=args.n_pos,
    )
    neg_windows = sample_negative_windows_no_promoter(
        promoter_trees=promoter_trees,
        chrom_sizes=chrom_sizes,
        seq_len=args.seq_len,
        n_neg=args.n_neg,
    )

    # Load HyenaDNA
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).eval()

    # Embed sampled windows
    emb_pos, pos_windows = embed_windows_layer(
        genome=genome,
        windows=pos_windows,
        tok=tok,
        model=model,
        layer=args.layer,
        seq_len=args.seq_len,
        device=device,
        batch_size=args.batch_size,
    )
    emb_neg, neg_windows = embed_windows_layer(
        genome=genome,
        windows=neg_windows,
        tok=tok,
        model=model,
        layer=args.layer,
        seq_len=args.seq_len,
        device=device,
        batch_size=args.batch_size,
    )

    # Load SAE
    sae = load_sae(args.sae_ckpt, device=device)

    # Collect activation sets
    data = collect_activation_sets(
        emb_pos=emb_pos,
        win_pos=pos_windows,
        emb_neg=emb_neg,
        win_neg=neg_windows,
        promoter_trees=promoter_trees,
        sae=sae,
        device=device,
        seq_len=args.seq_len,
        neg_per_pos_base=args.neg_per_pos_base,
        neg_sample_mode=args.neg_sample_mode,
        per_window_neg_tokens=args.per_window_neg_tokens,
    )

    A_pos: torch.Tensor = data["A_pos"]
    A_neg: torch.Tensor = data["A_neg"]
    pos_domain_slices: List[Tuple[int, int]] = data["pos_domain_slices"]

    # Association evaluation
    q_thresholds = [float(x) for x in args.q_thresholds.split(",") if x.strip() != ""]
    out_csv = os.path.join(args.out_dir, "promoter_eval_summary.csv")

    max_features = None if args.max_features == 0 else int(args.max_features)
    run_promoter_feature_search(
        A_pos=A_pos,
        A_neg=A_neg,
        pos_domain_slices=pos_domain_slices,
        out_csv=out_csv,
        q_thresholds=q_thresholds,
        max_features=max_features,
    )

    if args.save_tensors:
        torch.save(
            {
                "A_pos": A_pos,
                "A_neg": A_neg,
                "pos_domain_slices": pos_domain_slices,
                "pos_windows": [(w.chrom, w.start, w.end, w.prom_start, w.prom_end) for w in pos_windows],
                "neg_windows": [(w.chrom, w.start, w.end) for w in neg_windows],
                "neg_win_ids": data["neg_win_ids"],
                "neg_pos_ids": data["neg_pos_ids"],
                "seq_len": args.seq_len,
                "layer": args.layer,
                "model_id": args.model_id,
                "sae_ckpt": args.sae_ckpt,
            },
            os.path.join(args.out_dir, "promoter_eval_details.pt"),
        )

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()