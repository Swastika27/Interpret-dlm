#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import List

import pandas as pd


def norm_label(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"[^\w\-\.]+", "_", x)
    return x


def read_ucsc_rmsk_csv(path: str) -> pd.DataFrame:
    """
    Read UCSC Table Browser CSV export for rmsk where the header line starts with '#'.
    Example:
      #bin,swScore,...,id
      "0","1892",...
    We must parse the header ourselves because comment='#' would drop it.
    """
    header_cols: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                # Header line: remove leading '#', split by comma
                header_cols = line[1:].split(",")
                break
            # Some exports may have blank lines before header
            if line.strip() == "":
                continue

    if not header_cols:
        raise ValueError(
            "Could not find UCSC header line starting with '#'. "
            "Open the file and confirm it contains a line like '#bin,swScore,...'."
        )

    # Now read the CSV, skipping comment lines but providing names
    # header=None ensures the first data row isn't treated as header.
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=header_cols,
        dtype=str,
    )
    return df


def write_bed(sub: pd.DataFrame, out_path: str, keep_name: bool) -> None:
    if keep_name:
        sub[["genoName", "genoStart", "genoEnd", "repName"]].to_csv(
            out_path, sep="\t", header=False, index=False
        )
    else:
        sub[["genoName", "genoStart", "genoEnd"]].to_csv(
            out_path, sep="\t", header=False, index=False
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="UCSC rmsk table export (CSV)")
    ap.add_argument("--outdir", required=True, help="Output directory for BED files")
    ap.add_argument(
        "--mode",
        choices=["class", "family", "both"],
        default="class",
        help="Split by repClass, repFamily, or repClass/repFamily pairs",
    )
    ap.add_argument(
        "--keep_name",
        action="store_true",
        help="Write 4th BED column (repName) for debugging",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = read_ucsc_rmsk_csv(args.in_csv)

    required = {"genoName", "genoStart", "genoEnd", "repName", "repClass", "repFamily"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Clean numeric columns
    df["genoStart"] = df["genoStart"].astype(int)
    df["genoEnd"] = df["genoEnd"].astype(int)

    # Basic sanity
    df = df[df["genoEnd"] > df["genoStart"]].copy()

    if args.mode == "class":
        for rep_class, sub in df.groupby("repClass", sort=True):
            label = norm_label(rep_class)
            out_path = os.path.join(args.outdir, f"{label}.bed")
            write_bed(sub, out_path, args.keep_name)

    elif args.mode == "family":
        for rep_family, sub in df.groupby("repFamily", sort=True):
            label = norm_label(rep_family)
            out_path = os.path.join(args.outdir, f"{label}.bed")
            write_bed(sub, out_path, args.keep_name)

    else:  # both
        for (rep_class, rep_family), sub in df.groupby(["repClass", "repFamily"], sort=True):
            c = norm_label(rep_class)
            f = norm_label(rep_family)
            out_path = os.path.join(args.outdir, f"{c}__{f}.bed")
            write_bed(sub, out_path, args.keep_name)

    # Convenience “core classes”
    core = ["LINE", "SINE", "LTR", "Satellite", "Simple_repeat"]
    for c in core:
        sub = df[df["repClass"] == c]
        if not sub.empty:
            out_path = os.path.join(args.outdir, f"CORE__{c}.bed")
            write_bed(sub, out_path, args.keep_name)

    print(f"Done. Wrote BEDs to {args.outdir}")


if __name__ == "__main__":
    main()
