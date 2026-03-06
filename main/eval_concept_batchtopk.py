#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from train_batchtopk import BatchTopKSAE, LRUCache, build_sequence_index

# ----------------------------
# BED loading: fast overlap queries via sorted intervals + bisect
# ----------------------------
import bisect

def load_bed_intervals(path: str) -> Dict[str, Tuple[List[int], List[int]]]:
    """
    Returns: dict chrom -> (starts[], ends[]) sorted by starts
    """
    intervals: Dict[str, List[Tuple[int,int]]] = {}
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
        starts = [s for s,_ in ivs]
        ends = [e for _,e in ivs]
        out[chrom] = (starts, ends)
    return out

def overlaps_point(bed: Dict[str, Tuple[List[int], List[int]]], chrom: str, pos0: int) -> bool:
    """
    True if [pos0,pos0+1) overlaps any interval in bed[chrom].
    """
    x = bed.get(chrom)
    if x is None:
        return False
    starts, ends = x
    # rightmost interval with start <= pos0
    i = bisect.bisect_right(starts, pos0) - 1
    if i < 0:
        return False
    return pos0 < ends[i]

# ----------------------------
# Metadata for sampled tokens
# ----------------------------
@dataclass(frozen=True)
class TokMeta:
    chrom: str
    start: int
    end: int
    pos: int  # position within window [0, seq_len)
    shard_path: str
    row_idx: int

    @property
    def genome_pos0(self) -> int:
        return self.start + self.pos

# ----------------------------
# Sample one token per distinct window + keep meta
# ----------------------------
@torch.no_grad()
def make_token_batch_with_meta(seq_refs, cache: LRUCache, batch_seq_indices: List[int], seq_len: int, device):
    by_shard: Dict[str, List[int]] = {}
    for gi in batch_seq_indices:
        sr = seq_refs[gi]
        by_shard.setdefault(sr.shard_path, []).append(sr.row_idx)

    tokens = []
    metas: List[TokMeta] = []

    for sp, rows in by_shard.items():
        obj = cache.get(sp)
        if obj is None:
            obj = torch.load(sp, map_location="cpu")
            cache.put(sp, obj)

        emb = obj["emb"]  # [Bwin, L, D]
        coords = obj.get("coords", None)
        if coords is None:
            raise KeyError(f"{sp}: missing 'coords' key")

        L = int(emb.shape[1])
        if L != seq_len:
            raise ValueError(f"{sp}: seq_len mismatch: expected {seq_len}, got {L}")

        for r in rows:
            pos = random.randrange(seq_len)
            tok = emb[r, pos, :]
            chrom, start, end = coords[r]
            tokens.append(tok)
            metas.append(TokMeta(
                chrom=str(chrom),
                start=int(start),
                end=int(end),
                pos=int(pos),
                shard_path=sp,
                row_idx=int(r),
            ))

    x = torch.stack(tokens, dim=0).to(device=device, dtype=torch.float32)
    return x, metas

# ----------------------------
# Main evaluation
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="BatchTopK SAE checkpoint (has ckpt['args'] and ckpt['model'])")
    ap.add_argument("--data_root", required=True, help="Embeddings root containing split/layer dirs")
    ap.add_argument("--split", default="train")
    ap.add_argument("--layer_dir_name", required=True, help="e.g. layer_8")
    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--k_per_token", type=int, required=True)
    ap.add_argument("--n_tokens", type=int, default=1_000_000)
    ap.add_argument("--batch_tokens", type=int, default=256)
    ap.add_argument("--concept_bed", required=True, help="ENCODE concept BED file")
    ap.add_argument("--out_csv", default="concept_feature_assoc.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_shards", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load concept BED
    print("Loading concept BED intervals...")
    concept = load_bed_intervals(args.concept_bed)

    # Load SAE
    print("Loading SAE checkpoint...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    margs = ckpt["args"]
    d_in = int(margs["d_in"])
    d_sae = int(margs["d_sae"])

    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Decoder norm scaling (optional but recommended)
    Wdec = model.W_dec.weight.detach()  # [d_in, d_sae]
    dec_norm = torch.norm(Wdec, dim=0).to(device).clamp_min(1e-8)  # [d_sae]

    # Index windows
    layer_dir = os.path.join(args.data_root, args.split, args.layer_dir_name)
    seq_refs, n_seq = build_sequence_index(layer_dir)
    if n_seq < args.batch_tokens:
        raise ValueError(f"Need at least batch_tokens windows; have {n_seq}")

    cache = LRUCache(max_items=args.cache_shards)

    # Storage per feature: lists of (activation_value, label)
    # Only store for nonzero activations.
    feat_vals: List[List[float]] = [[] for _ in range(d_sae)]
    feat_lbls: List[List[int]] = [[] for _ in range(d_sae)]

    n_pos = 0
    n_seen = 0

    # Number of batches
    batches = (args.n_tokens + args.batch_tokens - 1) // args.batch_tokens
    k_total = args.batch_tokens * args.k_per_token

    print(f"Scanning {args.n_tokens} tokens in ~{batches} batches...")

    with torch.no_grad():
        for b in range(batches):
            print(f"Processing batch {b}/{batches}")
            # last batch may overshoot; we'll clip counts
            batch_seq_indices = random.sample(range(n_seq), k=args.batch_tokens)
            x, metas = make_token_batch_with_meta(seq_refs, cache, batch_seq_indices, args.seq_len, device)

            # labels for each token in this batch
            y = torch.zeros((args.batch_tokens,), dtype=torch.uint8)  # 0/1
            for i, meta in enumerate(metas):
                if overlaps_point(concept, meta.chrom, meta.genome_pos0):
                    y[i] = 1

            n_pos += int(y.sum().item())
            n_seen += args.batch_tokens

            # SAE forward
            _, z, _ = model(x, k_total=k_total)  # z: [B, d_sae], sparse-ish
            z_scaled = z * dec_norm.unsqueeze(0)  # scaled score for ranking/thresholds

            idx = (z > 0).nonzero(as_tuple=False)  # (i,j) on z device
            if idx.numel() > 0:
                i = idx[:, 0].detach().cpu()   # move indices to CPU for y
                j = idx[:, 1].detach().cpu()
                vals = z_scaled[idx[:, 0], idx[:, 1]].detach().cpu()  # gather using original device idx
                lbl = y[i].detach().cpu()

                for t in range(idx.shape[0]):
                    jj = int(j[t].item())
                    feat_vals[jj].append(float(vals[t].item()))
                    feat_lbls[jj].append(int(lbl[t].item()))

            if (b + 1) % 200 == 0:
                print(f"  batches={b+1} tokens_seen={n_seen} positives_seen={n_pos}")

            if n_seen >= args.n_tokens:
                break

    # Clip to exactly n_tokens if we overshot (we overshoot only in counting; stored events fine)
    n_tokens = min(n_seen, args.n_tokens)

    print(f"Done scanning. tokens={n_tokens}, positives={n_pos} ({(n_pos/max(1,n_tokens)):.6f})")

    # Evaluate best F1 per feature by sweeping thresholds over its observed nonzero values.
    # For threshold t > 0, predicted positives are exactly the stored events with value > t.
    # Sort descending values; cumulative counts give TP/FP as threshold decreases.
    print("Computing per-feature best F1...")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_id", "n_active", "best_t", "best_f1", "enrichment", "best_precision", "best_recall",
                    "tp", "fp", "fn", "n_pos_tokens", "n_tokens"])

        for fid in range(d_sae):
            print(f"Evaluating feature {fid}/{d_sae}")
            vals = feat_vals[fid]
            lbls = feat_lbls[fid]
            n_active = len(vals)
            if n_active == 0:
                # never active in sampled tokens
                w.writerow([fid, 0, "", 0.0, 0.0, 0.0, 0, 0, n_pos, n_pos, n_tokens])
                continue

            # sort by activation descending
            order = sorted(range(n_active), key=lambda k: vals[k], reverse=True)

            tp = 0
            fp = 0
            best_f1 = 0.0
            best_t = float("inf")
            best_p = 0.0
            best_r = 0.0
            best_tp = 0
            best_fp = 0
            best_f1_enrichment = 0.0

            # Sweep thresholds at each distinct value (predict >= current value)
            prev_v = None
            for idx_k in order:
                v = vals[idx_k]
                yk = lbls[idx_k]
                if yk == 1:
                    tp += 1
                else:
                    fp += 1

                # Only evaluate when value changes (new threshold)
                if prev_v is None or v != prev_v:
                    # predicted positives = all events with value >= v
                    # FN includes positives among tokens where feature is zero, plus active positives below threshold.
                    fn = n_pos - tp
                    denom_p = tp + fp
                    p = (tp / denom_p) if denom_p > 0 else 0.0
                    r = (tp / n_pos) if n_pos > 0 else 0.0
                    f1 = (2*p*r / (p+r)) if (p+r) > 0 else 0.0
                    pp = tp + fp
                    base_rate = n_pos / n_tokens
                    cond_rate = (tp / pp) if pp > 0 else 0.0
                    enrichment = (cond_rate / base_rate) if base_rate > 0 else 0.0

                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = v
                        best_p = p
                        best_r = r
                        best_tp = tp
                        best_fp = fp
                        best_f1_enrichment = enrichment

                    prev_v = v

            fn = n_pos - best_tp
            w.writerow([fid, n_active, best_t, best_f1, best_f1_enrichment, best_p, best_r,
                        best_tp, best_fp, fn, n_pos, n_tokens])

    print(f"Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()