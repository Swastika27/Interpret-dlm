#!/usr/bin/env python3
import argparse
import bisect
import csv
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from train_batchtopk import BatchTopKSAE, LRUCache, build_sequence_index

# ----------------------------
# BED as sorted intervals per chrom
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
    # segs are merged/sorted; linear scan is fine since seg count per window is small
    for a, b in segs:
        if a <= pos < b:
            return True
    return False

def sample_outside_segments(L: int, segs: List[Tuple[int, int]]) -> int:
    if not segs:
        return random.randrange(L)
    if seg_total_len(segs) >= L:
        raise ValueError("window fully covered; no negatives")
    while True:
        p = random.randrange(L)
        if not point_in_segs(p, segs):
            return p

# ----------------------------
# TokenRef for stratified pool
# ----------------------------
@dataclass(frozen=True)
class TokenRef:
    shard_path: str
    row_idx: int
    pos: int
    label: int  # 0/1

# ----------------------------
# Build tokens (pos/neg) from overlapping windows
# ----------------------------
def build_token_pool(
    seq_refs,
    cache: LRUCache,
    bed: Dict[str, Tuple[List[int], List[int]]],
    seq_len: int,
    n_pos: int,
    n_neg: int,
    max_tries: int = 50_000_000,
) -> Tuple[List[TokenRef], int, int]:
    """
    Returns token_refs list of length n_pos+n_neg.
    Also returns (n_windows_used, n_overlap_windows_seen).
    """
    tokens: List[TokenRef] = []
    pos_count = 0
    neg_count = 0
    n_windows_used = 0
    n_overlap_windows_seen = 0

    # We will randomly sample windows until we fill both quotas.
    tries = 0
    n_seq = len(seq_refs)

    while (pos_count < n_pos) or (neg_count < n_neg):
        tries += 1
        print(f"Building token pool. try {tries} | pos_count {pos_count} | neg_count {neg_count}")
        if tries > max_tries:
            raise RuntimeError(
                f"Exceeded max_tries={max_tries} while building token pool. "
                f"Got pos={pos_count}/{n_pos}, neg={neg_count}/{n_neg}. "
                f"Maybe concept is too rare in this subset."
            )

        gi = random.randrange(n_seq)
        sr = seq_refs[gi]

        obj = cache.get(sr.shard_path)
        if obj is None:
            obj = torch.load(sr.shard_path, map_location="cpu")
            cache.put(sr.shard_path, obj)

        coords = obj.get("coords", None)
        if coords is None:
            raise KeyError(f"{sr.shard_path}: missing 'coords'")

        chrom, start, end = coords[sr.row_idx]
        chrom = str(chrom); start = int(start); end = int(end)
        if end - start != seq_len:
            continue

        segs = window_overlap_segments(bed, chrom, start, end)
        if not segs:
            # pure negative window
            if neg_count < n_neg:
                pos = random.randrange(seq_len)
                tokens.append(TokenRef(sr.shard_path, sr.row_idx, pos, 0))
                neg_count += 1
                n_windows_used += 1
            continue

        # window overlaps concept
        n_overlap_windows_seen += 1

        # sample positives from overlap
        while pos_count < n_pos:
            pos = sample_from_segments(segs)
            tokens.append(TokenRef(sr.shard_path, sr.row_idx, pos, 1))
            pos_count += 1
            break

        # matched negative from same window outside overlap
        if neg_count < n_neg:
            posn = sample_outside_segments(seq_len, segs)
            tokens.append(TokenRef(sr.shard_path, sr.row_idx, posn, 0))
            neg_count += 1

        n_windows_used += 1

    return tokens, n_windows_used, n_overlap_windows_seen

# ----------------------------
# Load embeddings for a batch of TokenRef efficiently
# ----------------------------
@torch.no_grad()
def load_token_batch(token_refs: List[TokenRef], cache: LRUCache, seq_len: int, device: torch.device):
    by_shard: Dict[str, List[TokenRef]] = {}
    for tr in token_refs:
        by_shard.setdefault(tr.shard_path, []).append(tr)

    xs = []
    ys = []
    for sp, trs in by_shard.items():
        obj = cache.get(sp)
        if obj is None:
            obj = torch.load(sp, map_location="cpu")
            cache.put(sp, obj)

        emb = obj["emb"]  # [Bwin, L, D]
        if int(emb.shape[1]) != seq_len:
            raise ValueError(f"{sp}: seq_len mismatch: expected {seq_len}, got {int(emb.shape[1])}")

        for tr in trs:
            xs.append(emb[tr.row_idx, tr.pos, :])
            ys.append(tr.label)

    x = torch.stack(xs, dim=0).to(device=device, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.uint8)  # keep labels on CPU
    return x, y

# ----------------------------
# Prevalence over your subset windows
# ----------------------------
def compute_prevalence_over_subset(seq_refs, cache: LRUCache, bed, seq_len: int, max_windows: Optional[int] = None) -> float:
    """
    Computes P(y=1) over bases in the subset windows (or a subsample if max_windows set).
    """
    total_bases = 0
    pos_bases = 0

    n_seq = len(seq_refs)
    indices = list(range(n_seq))
    if max_windows is not None and max_windows < n_seq:
        indices = random.sample(indices, k=max_windows)

    for gi in indices:
        print(f"computing prevalence over subset {gi}/{len(indices)}")
        sr = seq_refs[gi]
        obj = cache.get(sr.shard_path)
        if obj is None:
            obj = torch.load(sr.shard_path, map_location="cpu")
            cache.put(sr.shard_path, obj)
        coords = obj["coords"]
        chrom, start, end = coords[sr.row_idx]
        chrom = str(chrom); start = int(start); end = int(end)
        if end - start != seq_len:
            continue
        segs = window_overlap_segments(bed, chrom, start, end)
        pos_bases += seg_total_len(segs)
        total_bases += seq_len

    return (pos_bases / total_bases) if total_bases > 0 else 0.0

# ----------------------------
# Main eval
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--layer_dir_name", required=True)
    ap.add_argument("--concept_bed", required=True)

    ap.add_argument("--seq_len", type=int, required=True)
    ap.add_argument("--k_per_token", type=int, required=True)
    ap.add_argument("--batch_tokens", type=int, default=512)  # match training
    ap.add_argument("--n_tokens", type=int, default=1_000_000)

    ap.add_argument("--pos_frac_in_pool", type=float, default=0.5, help="fraction positives in the token pool")
    ap.add_argument("--pos_frac_in_batch", type=float, default=0.1, help="fraction positives per forward batch")

    ap.add_argument("--cache_shards", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_csv", default="concept_feature_assoc.csv")

    ap.add_argument("--prevalence_max_windows", type=int, default=0,
                    help="0 => compute prevalence over all subset windows (may take time). "
                         "If >0, subsample this many windows for prevalence estimate.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("Loading concept BED...")
    concept = load_bed_intervals(args.concept_bed)

    print("Loading checkpoint/model...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    print("Checkpoint loading done")
    margs = ckpt["args"]
    d_in = int(margs["d_in"])
    d_sae = int(margs["d_sae"])

    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # decoder norm scaling (optional but recommended)
    Wdec = model.W_dec.weight.detach()
    dec_norm = torch.norm(Wdec, dim=0).to(device).clamp_min(1e-8)
    print("Decoder norm scaling done")

    # index subset windows
    layer_dir = os.path.join(args.data_root, args.split, args.layer_dir_name)
    seq_refs, n_seq = build_sequence_index(layer_dir)
    print("Built sequence subset index")
    if n_seq < args.batch_tokens:
        raise ValueError(f"Need at least batch_tokens windows; have {n_seq}")

    # caches
    cache_coords = LRUCache(max_items=args.cache_shards)
    cache_emb = LRUCache(max_items=args.cache_shards)

    # baseline prevalence over your subset
    print("Computing baseline prevalence over subset windows...")
    maxw = args.prevalence_max_windows if args.prevalence_max_windows > 0 else None
    base_rate = compute_prevalence_over_subset(seq_refs, cache_coords, concept, args.seq_len, max_windows=maxw)
    print(f"Baseline P(y=1) over subset = {base_rate:.8f} "
          f"(windows used = {maxw if maxw is not None else n_seq})")

    # Build token pool
    n_pos = int(args.n_tokens * args.pos_frac_in_pool)
    n_neg = args.n_tokens - n_pos
    print(f"Building token pool: n_pos={n_pos}, n_neg={n_neg} (disjoint windows, matched negatives)...")
    token_pool, n_windows_used, n_overlap_windows_seen = build_token_pool(
        seq_refs=seq_refs,
        cache=cache_coords,
        bed=concept,
        seq_len=args.seq_len,
        n_pos=n_pos,
        n_neg=n_neg,
    )
    print(f"Token pool built. windows_used={n_windows_used}, overlap_windows_seen={n_overlap_windows_seen}")

    # Split pool into pos and neg lists for controlled batching
    pos_refs = [tr for tr in token_pool if tr.label == 1]
    neg_refs = [tr for tr in token_pool if tr.label == 0]
    random.shuffle(pos_refs)
    random.shuffle(neg_refs)

    # Storage: only nonzero activations
    feat_vals: List[List[float]] = [[] for _ in range(d_sae)]
    feat_lbls: List[List[int]] = [[] for _ in range(d_sae)]

    # Forward passes in mixed batches
    k_total = args.batch_tokens * args.k_per_token
    n_pos_batch = int(round(args.batch_tokens * args.pos_frac_in_batch))
    n_pos_batch = max(0, min(args.batch_tokens, n_pos_batch))
    n_neg_batch = args.batch_tokens - n_pos_batch

    print(f"Running SAE forward: batch_tokens={args.batch_tokens}, k_total={k_total}, "
          f"pos_per_batch={n_pos_batch}, neg_per_batch={n_neg_batch}")

    pos_ptr = 0
    neg_ptr = 0
    n_seen = 0
    n_pos_seen = 0

    while n_seen < args.n_tokens:
        print(f"Calculating activations {n_seen}/{args.n_tokens}")
        # wrap pointers if needed
        if pos_ptr + n_pos_batch > len(pos_refs):
            random.shuffle(pos_refs)
            pos_ptr = 0
        if neg_ptr + n_neg_batch > len(neg_refs):
            random.shuffle(neg_refs)
            neg_ptr = 0

        batch = pos_refs[pos_ptr:pos_ptr+n_pos_batch] + neg_refs[neg_ptr:neg_ptr+n_neg_batch]
        pos_ptr += n_pos_batch
        neg_ptr += n_neg_batch
        random.shuffle(batch)

        x, y = load_token_batch(batch, cache_emb, args.seq_len, device)  # y on CPU
        n_seen += len(batch)
        n_pos_seen += int(y.sum().item())
        print(f"Loaded token batch n_seen={n_seen}, n_pos_seen={n_pos_seen}")

        with torch.no_grad():
            _, z, _ = model(x, k_total=k_total)
            z_scaled = z * dec_norm.unsqueeze(0)

            idx = (z > 0).nonzero(as_tuple=False)  # on device
            if idx.numel() > 0:
                vals = z_scaled[idx[:, 0], idx[:, 1]].detach().cpu()
                i_cpu = idx[:, 0].detach().cpu()
                j_cpu = idx[:, 1].detach().cpu()
                lbl = y[i_cpu]  # CPU indexing

                for t in range(idx.shape[0]):
                    jj = int(j_cpu[t].item())
                    feat_vals[jj].append(float(vals[t].item()))
                    feat_lbls[jj].append(int(lbl[t].item()))

        if (n_seen // args.batch_tokens) % 200 == 0:
            print(f"  tokens_seen={n_seen} pos_seen={n_pos_seen}")

    # Use the token pool's label count for F1 (since evaluation is on this dataset)
    N = args.n_tokens
    Npos = n_pos  # by construction of pool
    print(f"Computing best-enrichment per feature on the pool. N={N}, Npos={Npos}, pool_pos_rate={Npos/N:.4f}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "feature_id", "n_active",
            "best_t", "f1_at_best_enrichment", "best_precision", "best_recall",
            "enrichment",
            "tp", "fp", "fn",
            "Npos_pool", "N_pool",
            "baseline_prevalence"
        ])

        for fid in range(d_sae):
            print(f"Evalutating feature {fid}/{d_sae}")
            vals = feat_vals[fid]
            lbls = feat_lbls[fid]
            n_active = len(vals)
            if n_active == 0 or best_t == float("inf"):
                # no predictions ever
                # fn = Npos
                # w.writerow([fid, 0, "", 0.0, 0.0, 0.0, 0.0, 0, 0, fn, Npos, N, base_rate])
                continue

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
                    f1 = (2*p*r / (p+r)) if (p+r) > 0 else 0.0

                    # enrichment uses baseline prevalence over subset windows, not pool prevalence
                    cond_rate = (tp / pp) if pp > 0 else 0.0
                    enr = (cond_rate / base_rate) if base_rate > 0 else 0.0

                    if enr > best_enr:
                        best_f1 = f1
                        best_t = v
                        best_p = p
                        best_r = r
                        best_tp = tp
                        best_fp = fp
                        best_enr = enr

                    prev_v = v

            fn = Npos - best_tp
            w.writerow([fid, n_active, best_t, best_f1, best_p, best_r, best_enr,
                        best_tp, best_fp, fn, Npos, N, base_rate])

    print(f"Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()