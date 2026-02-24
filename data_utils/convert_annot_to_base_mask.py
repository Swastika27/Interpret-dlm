#!/usr/bin/env python3
"""
Convert ENCODE cCRE BED annotations into binary labels over a genomic sequence region.

Typical use (per-base labels for one window):
  python ccre_bed_to_labels.py \
    --fasta hg38.fa \
    --chrom chr1 --start 100000 --end 102000 \
    --bed pELS=data/ccre/pELS.bed \
    --bed PLS=data/ccre/PLS.bed \
    --bed dELS=data/ccre/dELS.bed \
    --bed CTCF_bound=data/ccre/CTCF-bound.bed \
    --bed CTCF_only=data/ccre/CTCF-only.bed \
    --bed DNase_H3K4me3=data/ccre/DNase-h3k4me3.bed \
    --out window_labels.npz

Optional (bin labels into tokens/bins, e.g., 1 label per 4 bp):
  --bin-size 4 --reduce any

Output .npz contains:
  - labels: uint8 array of shape (num_labels, num_positions_or_bins)
  - label_names: array of strings
  - chrom, start, end
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from pyfaidx import Fasta
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pyfaidx. Install with: pip install pyfaidx"
    ) from e


@dataclass(frozen=True)
class Interval:
    start: int
    end: int  # half-open [start, end)


def parse_bed_line(line: str) -> Optional[Tuple[str, int, int]]:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
        return None
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    chrom = parts[0]
    try:
        start = int(parts[1])
        end = int(parts[2])
    except ValueError:
        return None
    if end <= start:
        return None
    return chrom, start, end


def load_overlapping_intervals(
    bed_path: str, chrom: str, region_start: int, region_end: int
) -> List[Interval]:
    """
    Load only intervals on `chrom` that overlap [region_start, region_end).
    This is a simple scan (works well for reasonably sized BEDs; for huge BEDs, bgzip+tabix is faster).
    """
    out: List[Interval] = []
    with open(bed_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = parse_bed_line(line)
            if rec is None:
                continue
            c, s, e = rec
            if c != chrom:
                continue
            # no overlap
            if e <= region_start or s >= region_end:
                continue
            # clamp to region
            s2 = max(s, region_start)
            e2 = min(e, region_end)
            if e2 > s2:
                out.append(Interval(start=s2, end=e2))
    return out


def intervals_to_base_mask(
    intervals: Iterable[Interval],
    region_start: int,
    region_end: int,
) -> np.ndarray:
    """
    Produce per-base binary mask over [region_start, region_end).
    """
    L = region_end - region_start
    mask = np.zeros(L, dtype=np.uint8)
    for itv in intervals:
        a = itv.start - region_start
        b = itv.end - region_start
        # safe clamp (should already be clamped)
        a = max(a, 0)
        b = min(b, L)
        if b > a:
            mask[a:b] = 1
    return mask


def reduce_to_bins(base_mask: np.ndarray, bin_size: int, reduce: str) -> np.ndarray:
    """
    Convert per-base mask (0/1) to per-bin mask.

    reduce:
      - "any": bin is 1 if any base in bin is 1
      - "all": bin is 1 if all bases in bin are 1 (useful if you want full coverage)
      - "frac": returns float32 fraction in [0,1] instead of binary (still stored as float)
    """
    if bin_size <= 1:
        return base_mask

    L = base_mask.shape[0]
    n_bins = (L + bin_size - 1) // bin_size
    padded_len = n_bins * bin_size

    pad_width = padded_len - L
    if pad_width:
        base_mask = np.pad(base_mask, (0, pad_width), mode="constant", constant_values=0)

    x = base_mask.reshape(n_bins, bin_size)

    if reduce == "any":
        return (x.max(axis=1) > 0).astype(np.uint8)
    if reduce == "all":
        return (x.min(axis=1) > 0).astype(np.uint8)
    if reduce == "frac":
        return (x.mean(axis=1)).astype(np.float32)

    raise ValueError(f"Unknown reduce='{reduce}'. Expected one of: any, all, frac")


def parse_label_beds(bed_args: List[str]) -> List[Tuple[str, str]]:
    """
    Accept repeated --bed entries of the form LABEL=PATH or just PATH (label inferred from filename).
    """
    out: List[Tuple[str, str]] = []
    for b in bed_args:
        if "=" in b:
            label, path = b.split("=", 1)
            label = label.strip()
            path = path.strip()
            if not label or not path:
                raise ValueError(f"Bad --bed value: {b}")
            out.append((label, path))
        else:
            path = b.strip()
            if not path:
                raise ValueError(f"Bad --bed value: {b}")
            # infer label from filename
            label = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            out.append((label, path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert cCRE BEDs into binary labels over a sequence window.")
    ap.add_argument("--fasta", required=True, help="Reference FASTA (e.g., hg38.fa). Must be indexed or indexable by pyfaidx.")
    ap.add_argument("--chrom", required=True, help="Chromosome name (e.g., chr1).")
    ap.add_argument("--start", type=int, required=True, help="0-based start (inclusive).")
    ap.add_argument("--end", type=int, required=True, help="0-based end (exclusive).")
    ap.add_argument(
        "--bed",
        action="append",
        required=True,
        help="Repeatable. Either LABEL=PATH or PATH (label inferred). Example: --bed pELS=pELS.bed",
    )
    ap.add_argument("--out", required=True, help="Output .npz path.")
    ap.add_argument(
        "--bin-size",
        type=int,
        default=1,
        help="If >1, reduce per-base labels into bins of this size (e.g., token length). Default: 1 (per-base).",
    )
    ap.add_argument(
        "--reduce",
        choices=["any", "all", "frac"],
        default="any",
        help="How to reduce per-base labels into bins when --bin-size>1.",
    )
    ap.add_argument(
        "--no-seq",
        action="store_true",
        help="Do not store the sequence string in the output (labels only).",
    )

    args = ap.parse_args()

    if args.end <= args.start:
        raise SystemExit("--end must be > --start")

    label_beds = parse_label_beds(args.bed)

    # Load sequence (optional to store; still validates region exists)
    fasta = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    try:
        seq = fasta[args.chrom][args.start:args.end]
    except Exception as e:
        raise SystemExit(f"Failed to fetch {args.chrom}:{args.start}-{args.end} from FASTA: {e}") from e

    base_len = args.end - args.start

    label_names: List[str] = []
    label_arrays: List[np.ndarray] = []

    for label, bed_path in label_beds:
        intervals = load_overlapping_intervals(bed_path, args.chrom, args.start, args.end)
        base_mask = intervals_to_base_mask(intervals, args.start, args.end)

        if args.bin_size > 1:
            reduced = reduce_to_bins(base_mask, args.bin_size, args.reduce)
        else:
            reduced = base_mask

        label_names.append(label)
        label_arrays.append(reduced)

    # Stack: (num_labels, length_or_bins)
    # If reduce="frac", dtype may become float32; keep as-is
    labels = np.stack(label_arrays, axis=0)

    save_kwargs: Dict[str, object] = {
        "labels": labels,
        "label_names": np.array(label_names, dtype=object),
        "chrom": np.array([args.chrom], dtype=object),
        "start": np.int64(args.start),
        "end": np.int64(args.end),
        "bin_size": np.int64(args.bin_size),
        "reduce": np.array([args.reduce], dtype=object),
    }
    if not args.no_seq:
        save_kwargs["seq"] = np.array([str(seq)], dtype=object)

    np.savez_compressed(args.out, **save_kwargs)

    out_len = labels.shape[1]
    sys.stderr.write(
        f"Saved {len(label_names)} label tracks for {args.chrom}:{args.start}-{args.end} "
        f"({base_len} bp) -> length {out_len} (bin_size={args.bin_size}, reduce={args.reduce}) to {args.out}\n"
    )


if __name__ == "__main__":
    main()