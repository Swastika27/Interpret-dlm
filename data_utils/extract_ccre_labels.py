#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pandas as pd

def split_ccre_bed(in_bed: str, outdir: str) -> None:
    # Your file has 6 columns:
    # chrom, start, end, dhs_id, cre_id, class_tags
    df = pd.read_csv(in_bed, sep="\t", header=None, dtype=str)
    if df.shape[1] < 6:
        raise ValueError(f"Expected >=6 columns, got {df.shape[1]}")

    df.columns = ["chrom", "start", "end", "dhs_id", "ccre_id", "tags"]

    # Normalize tags into a list per row
    df["tag_list"] = df["tags"].fillna("").apply(lambda s: [t.strip() for t in s.split(",") if t.strip()])

    # Tags you likely want (add/remove as needed)
    wanted = [
        "PLS",          # promoter-like signature
        "pELS",         # proximal enhancer-like
        "dELS",         # distal enhancer-like
        "CTCF-only",    # insulator-like
        "CTCF-bound",   # CTCF-bound flag
        "DNase-H3K4me3" # appears in your sample; keep if present
    ]

    os.makedirs(outdir, exist_ok=True)

    # Write each tag as its own BED (keeping only first 3 columns by default)
    for tag in wanted:
        sub = df[df["tag_list"].apply(lambda xs: tag in xs)]
        if sub.empty:
            continue
        out_path = os.path.join(outdir, f"{tag}.bed")
        sub[["chrom", "start", "end"]].to_csv(out_path, sep="\t", header=False, index=False)

    # Optional: write “core regulatory” enhancers/promoters only
    core = df[df["tag_list"].apply(lambda xs: any(t in xs for t in ["PLS", "pELS", "dELS", "CTCF-only"]))]
    core[["chrom", "start", "end", "tags"]].to_csv(
        os.path.join(outdir, "cCRE_core_with_tags.bed"),
        sep="\t", header=False, index=False
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_bed", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    split_ccre_bed(args.in_bed, args.outdir)

if __name__ == "__main__":
    main()
