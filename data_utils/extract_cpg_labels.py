#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd


def read_ucsc_csv_with_hash_header(path: str) -> pd.DataFrame:
    # UCSC exports often start with: #"bin","chrom",...
    header_cols: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                # remove leading '#', remove quotes, split by comma
                header = line[1:].strip()
                header = header.replace('"', "")
                header_cols = header.split(",")
                break
            if line == "":
                continue

    if not header_cols:
        raise ValueError("Could not find header line starting with '#'.")

    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=header_cols,
        dtype=str,
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="UCSC cpgIslandExt export (CSV)")
    ap.add_argument("--out_bed", required=True, help="Output BED path, e.g. cpg_islands.hg38.bed")
    ap.add_argument("--keep_extra", action="store_true", help="Keep name/metrics columns in BED")
    args = ap.parse_args()

    df = read_ucsc_csv_with_hash_header(args.in_csv)

    required = {"chrom", "chromStart", "chromEnd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Found {list(df.columns)}")

    df["chromStart"] = df["chromStart"].astype(int)
    df["chromEnd"] = df["chromEnd"].astype(int)
    df = df[df["chromEnd"] > df["chromStart"]].copy()

    os.makedirs(os.path.dirname(args.out_bed) or ".", exist_ok=True)

    if args.keep_extra:
        # BED6-ish: chrom start end name score strand (strand unknown -> ".")
        # Here we keep useful QC fields instead.
        out = df[["chrom", "chromStart", "chromEnd", "name", "length", "obsExp"]].copy()
        out.to_csv(args.out_bed, sep="\t", header=False, index=False)
    else:
        out = df[["chrom", "chromStart", "chromEnd"]].copy()
        out.to_csv(args.out_bed, sep="\t", header=False, index=False)

    print(f"Wrote: {args.out_bed}  (rows={len(out)})")


if __name__ == "__main__":
    main()
