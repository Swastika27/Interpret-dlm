#!/usr/bin/env python3
"""
Evaluate concept–feature association for a BatchTopK SAE using stratified sampling
from disjoint windows, with a cached window index built from coords shards.

Key features:
- Uses coords shards (split/coords/shard_*.pt) to build an index of:
    * pos_windows: windows that overlap the concept BED (stores overlap segments in window coords)
    * neg_windows: windows that do not overlap
    * baseline_prevalence: fraction of bases overlapping concept within THIS split's window subset
- Saves/loads the index (.pt) so you build it once and reuse.
- Builds a labeled token pool (size N) by sampling:
    * positives from pos_windows (guaranteed overlap)
    * negatives from:
        - matched (same pos window but outside overlap), and/or
        - background (from neg_windows)
- Runs BatchTopK SAE forward in batches matching training batch_tokens and k_per_token.
  Batch composition can be mixed (e.g., 10% pos per batch) to reduce BatchTopK regime shift.
- Stores only nonzero feature events (value, label) per feature, then sweeps thresholds to get best F1.
- Reports enrichment at best-F1 using baseline_prevalence (not pool prevalence).

Assumptions:
- coords shard: split/coords/shard_XXXXX.pt contains {"coords": List[(chrom,start,end)], ...}
- emb shard:   split/{layer_dir_name}/shard_XXXXX.pt contains {"emb": FloatTensor[B,L,D], ...}
  and is row-aligned with coords shard of the same shard_XXXXX.pt.

Example:
python3 eval_concept_batchtopk_final.py \
  --ckpt ../runs/sae/layer8_bt8/checkpoints/final.pt \
  --data_root ../data/embeddings \
  --split train \
  --layer_dir_name layer_8 \
  --seq_len 2000 \
  --k_per_token 8 \
  --batch_tokens 512 \
  --concept_bed ../data/annotations/encode_ccres/PLS.bed \
  --n_tokens 1000000 \
  --pos_frac_in_pool 0.5 \
  --pos_frac_in_batch 0.1 \
  --neg_mode mixed \
  --index_path pls_window_index_train_layer8.pt \
  --out_csv pls_feature_assoc_final.csv \
  --cache_emb 16
"""

import argparse
import bisect
import csv
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch

from train_batchtopk import BatchTopKSAE


# ----------------------------
# Simple LRU cache (avoid disk IO)
# ----------------------------
class LRUCache:
    def __init__(self, max_items: int = 8):
        from collections import OrderedDict
        self.max_items = max_items
        self._cache = OrderedDict()

    def get(self, path: str):
        if path not in self._cache:
            return None
        self._cache.move_to_end(path)
        return self._cache[path]

    def put(self, path: str, obj):
        self._cache[path] = obj
        self._cache.move_to_end(path)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


# ----------------------------
# BED loading and overlap within a window
# ----------------------------
def load_bed_intervals(path: str) -> Dict[str, Tuple[List[int], List[int]]]:
    intervals: Dict[str, List[Tuple[int, int]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                s = int(parts[1]); e = int(parts[2])
            except ValueError:
                continue
            if e <= s:
                continue
            intervals.setdefault(chrom, []).append((s, e))

    out: Dict[str, Tuple[List[int], List[int]]] = {}
    for chrom, ivs in intervals.items():
        ivs.sort(key=lambda x: x[0])
        out[chrom] = ([s for s, _ in ivs], [e for _, e in ivs])
    return out


def window_overlap_segments(
    bed: Dict[str, Tuple[List[int], List[int]]],
    chrom: str,
    wstart: int,
    wend: int,
) -> List[Tuple[int, int]]:
    """
    Returns merged overlap segments in window-relative coords [0, L).
    Each segment is [a,b) with 0<=a<b<=L.
    """
    x = bed.get(chrom)
    if x is None:
        return []
    starts, ends = x
    L = wend - wstart
    if L <= 0:
        return []

    hi = bisect.bisect_left(starts, wend)
    lo = max(0, bisect.bisect_right(starts, wstart) - 1)

    segs: List[Tuple[int, int]] = []
    for k in range(lo, hi):
        s = starts[k]; e = ends[k]
        if e <= wstart or s >= wend:
            continue
        a = max(s, wstart) - wstart
        b = min(e, wend) - wstart
        if a < b:
            segs.append((a, b))
    if not segs:
        return []

    segs.sort()
    merged = [segs[0]]
    for a, b in segs[1:]:
        la, lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def seg_total_len(segs: List[Tuple[int, int]]) -> int:
    return sum(b - a for a, b in segs)


def sample_from_segments(segs: List[Tuple[int, int]]) -> int:
    tot = seg_total_len(segs)
    r = random.randrange(tot)
    for a, b in segs:
        l = b - a
        if r < l:
            return a + r
        r -= l
    raise RuntimeError("unreachable")


def point_in_segs(pos: int, segs: List[Tuple[int, int]]) -> bool:
    for a, b in segs:
        if a <= pos < b:
            return True
    return False


def sample_outside_segments(L: int, segs: List[Tuple[int, int]]) -> int:
    if seg_total_len(segs) >= L:
        raise ValueError("window fully covered; no outside positions")
    while True:
        p = random.randrange(L)
        if not point_in_segs(p, segs):
            return p


# ----------------------------
# Index entries and token refs
# ----------------------------
@dataclass(frozen=True)
class PosWindow:
    emb_shard_path: str
    row_idx: int
    segs: List[Tuple[int, int]]


@dataclass(frozen=True)
class NegWindow:
    emb_shard_path: str
    row_idx: int


@dataclass(frozen=True)
class TokenRef:
    emb_shard_path: str
    row_idx: int
    pos: int
    label: int  # 0/1


# ----------------------------
# Build/load/save window index (coords-only)
# ----------------------------
def build_window_index_from_coords(
    split_root: str,        # e.g. data_root/train
    layer_dir_name: str,    # e.g. layer_8
    bed: Dict[str, Tuple[List[int], List[int]]],
    seq_len: int,
    cache_coords: LRUCache,
) -> Tuple[List[PosWindow], List[NegWindow], float]:
    coords_dir = os.path.join(split_root, "coords")
    coords_paths = sorted(glob.glob(os.path.join(coords_dir, "shard_*.pt")))
    if not coords_paths:
        raise FileNotFoundError(f"No coords shards found in {coords_dir}")

    pos_windows: List[PosWindow] = []
    neg_windows: List[NegWindow] = []

    pos_bases = 0
    total_bases = 0

    for cp in coords_paths:
        cobj = cache_coords.get(cp)
        if cobj is None:
            cobj = torch.load(cp, map_location="cpu")
            cache_coords.put(cp, cobj)

        coords = cobj["coords"]
        shard_name = os.path.basename(cp)  # shard_00000.pt
        emb_shard_path = os.path.join(split_root, layer_dir_name, shard_name)

        for r, (chrom, start, end) in enumerate(coords):
            chrom = str(chrom); start = int(start); end = int(end)
            if end - start != seq_len:
                continue

            segs = window_overlap_segments(bed, chrom, start, end)
            if segs:
                pos_windows.append(PosWindow(emb_shard_path=emb_shard_path, row_idx=r, segs=segs))
                pos_bases += seg_total_len(segs)
            else:
                neg_windows.append(NegWindow(emb_shard_path=emb_shard_path, row_idx=r))

            total_bases += seq_len

    base_rate = (pos_bases / total_bases) if total_bases > 0 else 0.0
    return pos_windows, neg_windows, base_rate


def save_window_index(
    path: str,
    pos_windows: List[PosWindow],
    neg_windows: List[NegWindow],
    baseline_prevalence: float,
    metadata: dict,
) -> None:
    obj = {
        "pos_windows": [
            {"emb_shard_path": w.emb_shard_path, "row_idx": w.row_idx, "segs": w.segs}
            for w in pos_windows
        ],
        "neg_windows": [
            {"emb_shard_path": w.emb_shard_path, "row_idx": w.row_idx}
            for w in neg_windows
        ],
        "baseline_prevalence": float(baseline_prevalence),
        "metadata": metadata,
    }
    torch.save(obj, path)


def load_window_index(path: str) -> Tuple[List[PosWindow], List[NegWindow], float, dict]:
    obj = torch.load(path, map_location="cpu")
    pos_windows = [
        PosWindow(d["emb_shard_path"], int(d["row_idx"]), d["segs"])
        for d in obj["pos_windows"]
    ]
    neg_windows = [
        NegWindow(d["emb_shard_path"], int(d["row_idx"]))
        for d in obj["neg_windows"]
    ]
    base_rate = float(obj["baseline_prevalence"])
    meta = obj.get("metadata", {})
    return pos_windows, neg_windows, base_rate, meta


# ----------------------------
# Token pool builder
# ----------------------------
def build_token_pool_from_index(
    pos_windows: List[PosWindow],
    neg_windows: List[NegWindow],
    seq_len: int,
    n_pos: int,
    n_neg: int,
    neg_mode: str,  # matched|background|mixed
) -> List[TokenRef]:
    if n_pos > 0 and not pos_windows:
        raise RuntimeError("No positive windows in index; cannot sample positives.")
    if n_neg > 0 and (neg_mode in ("background", "mixed")) and not neg_windows:
        raise RuntimeError("No negative windows in index; cannot sample background negatives.")

    pool: List[TokenRef] = []

    # positives
    for _ in range(n_pos):
        pw = random.choice(pos_windows)
        pos = sample_from_segments(pw.segs)
        pool.append(TokenRef(pw.emb_shard_path, pw.row_idx, pos, 1))

    # negatives
    for _ in range(n_neg):
        if neg_mode == "matched":
            pw = random.choice(pos_windows)
            pos = sample_outside_segments(seq_len, pw.segs)
            pool.append(TokenRef(pw.emb_shard_path, pw.row_idx, pos, 0))
        elif neg_mode == "background":
            nw = random.choice(neg_windows)
            pos = random.randrange(seq_len)
            pool.append(TokenRef(nw.emb_shard_path, nw.row_idx, pos, 0))
        elif neg_mode == "mixed":
            # 50% matched, 50% background (if possible)
            if pos_windows and (not neg_windows or random.random() < 0.5):
                pw = random.choice(pos_windows)
                pos = sample_outside_segments(seq_len, pw.segs)
                pool.append(TokenRef(pw.emb_shard_path, pw.row_idx, pos, 0))
            else:
                nw = random.choice(neg_windows)
                pos = random.randrange(seq_len)
                pool.append(TokenRef(nw.emb_shard_path, nw.row_idx, pos, 0))
        else:
            raise ValueError(f"Unknown neg_mode: {neg_mode}")

    random.shuffle(pool)
    return pool


# ----------------------------
# Load embeddings for a batch of TokenRef efficiently
# ----------------------------
@torch.no_grad()
def load_token_batch(
    token_refs: List[TokenRef],
    cache_emb: LRUCache,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    by_shard: Dict[str, List[TokenRef]] = {}
    for tr in token_refs:
        by_shard.setdefault(tr.emb_shard_path, []).append(tr)

    xs = []
    ys = []
    for sp, trs in by_shard.items():
        obj = cache_emb.get(sp)
        if obj is None:
            obj = torch.load(sp, map_location="cpu")
            cache_emb.put(sp, obj)

        emb = obj["emb"]  # [B,L,D]
        if int(emb.shape[1]) != seq_len:
            raise ValueError(f"{sp}: seq_len mismatch: expected {seq_len}, got {int(emb.shape[1])}")

        for tr in trs:
            xs.append(emb[tr.row_idx, tr.pos, :])
            ys.append(tr.label)

    x = torch.stack(xs, dim=0).to(device=device, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.uint8)  # keep labels on CPU
    return x, y


# ----------------------------
# Main evaluation
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="BatchTopK SAE checkpoint (has ckpt['args'] and ckpt['model'])")
    ap.add_argument("--data_root", required=True, help="Root containing split folders")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--layer_dir_name", required=True, help="e.g. layer_8")
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--k_per_token", type=int, required=True)
    ap.add_argument("--batch_tokens", type=int, required=True, help="Use the SAME value you trained with (e.g. 512)")
    ap.add_argument("--concept_bed", required=True)

    ap.add_argument("--n_tokens", type=int, default=1_000_000)
    ap.add_argument("--pos_frac_in_pool", type=float, default=0.5, help="Fraction positives in the token pool")
    ap.add_argument("--pos_frac_in_batch", type=float, default=0.1, help="Fraction positives per forward batch")
    ap.add_argument("--neg_mode", choices=["matched", "background", "mixed"], default="mixed")

    ap.add_argument("--index_path", required=True, help="Path to save/load window index .pt")
    ap.add_argument("--rebuild_index", action="store_true", help="Force rebuild window index")
    ap.add_argument("--cache_coords", type=int, default=16)
    ap.add_argument("--cache_emb", type=int, default=16)

    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--min_active", type=int, default=50, help="Report best-F1 even if low, but you can filter later.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load concept
    print("Loading concept BED...")
    concept = load_bed_intervals(args.concept_bed)

    # Load SAE
    print("Loading SAE checkpoint/model...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    margs = ckpt["args"]
    d_in = int(margs["d_in"])
    d_sae = int(margs["d_sae"])

    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # decoder norm scaling
    print("Decoder norm")
    Wdec = model.W_dec.weight.detach()  # [d_in, d_sae]
    dec_norm = torch.norm(Wdec, dim=0).to(device).clamp_min(1e-8)

    # Build/load index
    split_root = os.path.join(args.data_root, args.split)
    cache_coords = LRUCache(max_items=args.cache_coords)

    if (not args.rebuild_index) and os.path.exists(args.index_path):
        print(f"Loading window index: {args.index_path}")
        pos_windows, neg_windows, base_rate, meta = load_window_index(args.index_path)
    else:
        print("Building window index from coords shards (coords-only)...")
        pos_windows, neg_windows, base_rate = build_window_index_from_coords(
            split_root=split_root,
            layer_dir_name=args.layer_dir_name,
            bed=concept,
            seq_len=args.seq_len,
            cache_coords=cache_coords,
        )
        meta = {
            "data_root": args.data_root,
            "split": args.split,
            "layer_dir_name": args.layer_dir_name,
            "seq_len": args.seq_len,
            "concept_bed": args.concept_bed,
        }
        save_window_index(args.index_path, pos_windows, neg_windows, base_rate, meta)
        print(f"Saved window index: {args.index_path}")

    print(f"Index summary: pos_windows={len(pos_windows)}, neg_windows={len(neg_windows)}")
    print(f"Baseline prevalence over THIS split subset: {base_rate:.10f}")

    # Build token pool
    n_pos = int(round(args.n_tokens * args.pos_frac_in_pool))
    n_pos = max(0, min(args.n_tokens, n_pos))
    n_neg = args.n_tokens - n_pos

    print(f"Building token pool: N={args.n_tokens} (pos={n_pos}, neg={n_neg}), neg_mode={args.neg_mode}")
    pool = build_token_pool_from_index(
        pos_windows=pos_windows,
        neg_windows=neg_windows,
        seq_len=args.seq_len,
        n_pos=n_pos,
        n_neg=n_neg,
        neg_mode=args.neg_mode,
    )

    pos_pool = [tr for tr in pool if tr.label == 1]
    neg_pool = [tr for tr in pool if tr.label == 0]
    random.shuffle(pos_pool)
    random.shuffle(neg_pool)

    # Prepare per-feature sparse storage
    feat_vals: List[List[float]] = [[] for _ in range(d_sae)]
    feat_lbls: List[List[int]] = [[] for _ in range(d_sae)]

    # Mixed batching for BatchTopK regime
    n_pos_batch = int(round(args.batch_tokens * args.pos_frac_in_batch))
    n_pos_batch = max(0, min(args.batch_tokens, n_pos_batch))
    n_neg_batch = args.batch_tokens - n_pos_batch
    if n_pos_batch > 0 and not pos_pool:
        raise RuntimeError("pos_frac_in_batch>0 but token pool has no positives.")
    if n_neg_batch > 0 and not neg_pool:
        raise RuntimeError("pos_frac_in_batch<1 but token pool has no negatives.")

    k_total = args.batch_tokens * args.k_per_token
    print(f"Forward settings: batch_tokens={args.batch_tokens}, k_per_token={args.k_per_token}, k_total={k_total}, "
          f"pos_per_batch={n_pos_batch}, neg_per_batch={n_neg_batch}")

    cache_emb = LRUCache(max_items=args.cache_emb)

    pos_ptr = 0
    neg_ptr = 0
    seen = 0

    with torch.no_grad():
        while seen < args.n_tokens:
            print(f"Seen {seen}/{args.n_tokens} tokens")
            # wrap
            if n_pos_batch > 0 and pos_ptr + n_pos_batch > len(pos_pool):
                random.shuffle(pos_pool)
                pos_ptr = 0
            if n_neg_batch > 0 and neg_ptr + n_neg_batch > len(neg_pool):
                random.shuffle(neg_pool)
                neg_ptr = 0

            batch = []
            if n_pos_batch > 0:
                batch.extend(pos_pool[pos_ptr:pos_ptr + n_pos_batch])
                pos_ptr += n_pos_batch
            if n_neg_batch > 0:
                batch.extend(neg_pool[neg_ptr:neg_ptr + n_neg_batch])
                neg_ptr += n_neg_batch
            random.shuffle(batch)

            x, y = load_token_batch(batch, cache_emb, args.seq_len, device)
            # y is CPU uint8
            seen += len(batch)

            _, z, _ = model(x, k_total=k_total)
            z_scaled = z * dec_norm.unsqueeze(0)

            idx = (z > 0).nonzero(as_tuple=False)  # on device
            if idx.numel() > 0:
                vals = z_scaled[idx[:, 0], idx[:, 1]].detach().cpu()
                i_cpu = idx[:, 0].detach().cpu()
                j_cpu = idx[:, 1].detach().cpu()
                lbl = y[i_cpu]  # CPU indexing

                # store events
                for t in range(idx.shape[0]):
                    jj = int(j_cpu[t].item())
                    feat_vals[jj].append(float(vals[t].item()))
                    feat_lbls[jj].append(int(lbl[t].item()))

            if (seen // args.batch_tokens) % 200 == 0:
                # lightweight progress
                pass

    # Compute best-F1 per feature (on POOL distribution)
    N = args.n_tokens
    Npos = n_pos
    print(f"Computing per-feature best-F1 on token pool: N={N}, Npos={Npos}, pool_pos_rate={Npos/max(1,N):.6f}")

    # Write CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "feature_id",
            "n_active",
            "active_rate",
            "best_t",
            "best_f1",
            "best_precision",
            "best_recall",
            "enrichment_at_best_f1",   # uses baseline prevalence over subset windows
            "tp",
            "fp",
            "fn",
            "Npos_pool",
            "N_pool",
            "baseline_prevalence",
        ])

        for fid in range(d_sae):
            print(f"Evaluating feature {fid}/{d_sae}")
            vals = feat_vals[fid]
            lbls = feat_lbls[fid]
            n_active = len(vals)
            active_rate = n_active / max(1, N)

            if n_active == 0:
                fn = Npos
                w.writerow([fid, 0, 0.0, "", 0.0, 0.0, 0.0, 0.0, 0, 0, fn, Npos, N, base_rate])
                continue

            # sort descending
            order = sorted(range(n_active), key=lambda k: vals[k], reverse=True)

            tp = 0
            fp = 0
            best_f1 = 0.0
            best_t = float("inf")
            best_p = 0.0
            best_r = 0.0
            best_tp = 0
            best_fp = 0
            best_enr = 0.0

            prev_v = None
            for k in order:
                v = vals[k]
                yk = lbls[k]
                if yk == 1:
                    tp += 1
                else:
                    fp += 1

                if prev_v is None or v != prev_v:
                    fn = Npos - tp
                    pp = tp + fp
                    p = (tp / pp) if pp > 0 else 0.0
                    r = (tp / Npos) if Npos > 0 else 0.0
                    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

                    cond_rate = (tp / pp) if pp > 0 else 0.0
                    enr = (cond_rate / base_rate) if base_rate > 0 else 0.0

                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = v
                        best_p = p
                        best_r = r
                        best_tp = tp
                        best_fp = fp
                        best_enr = enr

                    prev_v = v

            fn = Npos - best_tp

            # (optional) keep rows for low-activity features; user can filter later
            w.writerow([
                fid,
                n_active,
                active_rate,
                best_t,
                best_f1,
                best_p,
                best_r,
                best_enr,
                best_tp,
                best_fp,
                fn,
                Npos,
                N,
                base_rate,
            ])

    print(f"Wrote: {args.out_csv}")
    print(f"Tip: for stability, focus on features with n_active >= {args.min_active}.")


if __name__ == "__main__":
    main()