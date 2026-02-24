#!/usr/bin/env python3
"""
make_windows.py

Generate fixed-length, non-overlapping genomic windows from GRCh38 primary assembly,
filtering out:
- Ns-heavy windows (fraction of N/n > threshold)
- ENCODE blacklist regions (BED)
- centromere regions (BED)

Then split windows by chromosome:
- Train: chr1–chr19
- Val: chr20–chr21
- Test: chr22 + chrX

Outputs BED files:
  out_dir/train_windows.bed
  out_dir/val_windows.bed
  out_dir/test_windows.bed

Notes:
- Coordinates are BED: 0-based, half-open [start, end)
- Requires: pyfaidx
    pip install pyfaidx
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from pyfaidx import Fasta


@dataclass
class Interval:
    start: int  # inclusive, 0-based
    end: int    # exclusive, 0-based


def read_bed_intervals(path: str) -> Dict[str, List[Interval]]:
    """
    Read a BED (or BED-like) file. Uses first 3 columns: chrom, start, end.
    Returns dict chrom -> list of intervals (not yet merged).
    Ignores header/comment lines starting with '#', 'track', 'browser'.
    """
    intervals: Dict[str, List[Interval]] = {}
    if not path:
        return intervals

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
            intervals.setdefault(chrom, []).append(Interval(start, end))
    return intervals


def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    """Merge overlapping/adjacent intervals. Assumes 0-based half-open."""
    if not intervals:
        return []
    intervals_sorted = sorted(intervals, key=lambda x: (x.start, x.end))
    merged: List[Interval] = [intervals_sorted[0]]
    for iv in intervals_sorted[1:]:
        last = merged[-1]
        if iv.start <= last.end:  # overlap or adjacency
            last.end = max(last.end, iv.end)
        else:
            merged.append(Interval(iv.start, iv.end))
    return merged


def merge_bed_dict(bed: Dict[str, List[Interval]]) -> Dict[str, List[Interval]]:
    return {chrom: merge_intervals(iv_list) for chrom, iv_list in bed.items()}


def interval_overlaps_any(iv: Interval, blocks: List[Interval]) -> bool:
    """
    Return True if iv overlaps any interval in blocks.
    blocks must be merged & sorted.
    """
    if not blocks:
        return False
    # binary search-ish scan
    lo, hi = 0, len(blocks) - 1
    # Find first block that could overlap (block.end > iv.start)
    while lo < hi:
        mid = (lo + hi) // 2
        if blocks[mid].end <= iv.start:
            lo = mid + 1
        else:
            hi = mid
    # Scan forward a bit
    i = lo
    while i < len(blocks) and blocks[i].start < iv.end:
        if blocks[i].end > iv.start and blocks[i].start < iv.end:
            return True
        i += 1
    return False


def n_fraction(seq: str) -> float:
    if not seq:
        return 1.0
    n_count = sum(1 for c in seq if c in ("N", "n"))
    return n_count / len(seq)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="GRCh38 primary assembly FASTA (indexed or will be indexed).")
    ap.add_argument("--out_dir", required=True, help="Output directory.")
    ap.add_argument("--window", type=int, default=2048, help="Window length in bp. Default 2048.")
    ap.add_argument("--stride", type=int, default=2048, help="Stride in bp. Default 2048 (no overlap).")
    ap.add_argument("--max_n_frac", type=float, default=0.01, help="Drop windows with N fraction > this. Default 0.01.")
    ap.add_argument("--blacklist_bed", default="", help="ENCODE blacklist BED (hg38). Optional.")
    ap.add_argument("--centromere_bed", default="", help="Centromere BED (hg38). Optional.")
    ap.add_argument(
        "--chroms",
        default="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX",
        help="Comma-separated chromosomes to consider. Default chr1-22,chrX.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Unused currently; reserved.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    train_chroms = {f"chr{i}" for i in range(1, 20)}
    val_chroms = {"chr20", "chr21"}
    test_chroms = {"chr22", "chrX"}

    # Load genome
    fa = Fasta(args.fasta, as_raw=True, sequence_always_upper=False)  # keep case; we count N/n anyway

    # Load exclusion intervals
    blacklist = merge_bed_dict(read_bed_intervals(args.blacklist_bed)) if args.blacklist_bed else {}
    centromeres = merge_bed_dict(read_bed_intervals(args.centromere_bed)) if args.centromere_bed else {}

    def excluded(chrom: str, iv: Interval) -> bool:
        if interval_overlaps_any(iv, blacklist.get(chrom, [])):
            return True
        if interval_overlaps_any(iv, centromeres.get(chrom, [])):
            return True
        return False

    # Output writers
    train_path = os.path.join(args.out_dir, "train_windows.bed")
    val_path = os.path.join(args.out_dir, "val_windows.bed")
    test_path = os.path.join(args.out_dir, "test_windows.bed")

    train_f = open(train_path, "w", encoding="utf-8")
    val_f = open(val_path, "w", encoding="utf-8")
    test_f = open(test_path, "w", encoding="utf-8")

    # Counters
    counts = {"train": 0, "val": 0, "test": 0, "skipped_n": 0, "skipped_excl": 0, "skipped_short": 0}

    for chrom in chroms:
        if chrom not in fa:
            continue

        chrom_len = len(fa[chrom])
        # iterate windows
        start = 0
        while start < chrom_len:
            end = start + args.window
            if end > chrom_len:
                counts["skipped_short"] += 1
                break  # drop last partial window
            iv = Interval(start, end)

            if excluded(chrom, iv):
                counts["skipped_excl"] += 1
                start += args.stride
                continue

            seq = fa[chrom][start:end]
            if n_fraction(seq) > args.max_n_frac:
                counts["skipped_n"] += 1
                start += args.stride
                continue

            line = f"{chrom}\t{start}\t{end}\n"
            if chrom in train_chroms:
                train_f.write(line)
                counts["train"] += 1
            elif chrom in val_chroms:
                val_f.write(line)
                counts["val"] += 1
            elif chrom in test_chroms:
                test_f.write(line)
                counts["test"] += 1
            # else: chromosome not in split set; ignore
            start += args.stride

    train_f.close()
    val_f.close()
    test_f.close()

    # Print summary
    print("Wrote:")
    print(f"  {train_path}  ({counts['train']} windows)")
    print(f"  {val_path}    ({counts['val']} windows)")
    print(f"  {test_path}   ({counts['test']} windows)")
    print("Skipped:")
    print(f"  excluded overlap: {counts['skipped_excl']}")
    print(f"  N-fraction:       {counts['skipped_n']}")
    print(f"  tail short:       {counts['skipped_short']}")


if __name__ == "__main__":
    main()
