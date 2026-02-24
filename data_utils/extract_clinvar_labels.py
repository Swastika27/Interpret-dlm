#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import os
import pandas as pd


def open_maybe_gzip(path: str):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.endswith(".gz") else open(path, "r", encoding="utf-8", errors="replace")


def detect_sep_and_header(path: str) -> tuple[str, int]:
    """
    ClinVar variant_summary is usually tab-delimited with a header line.
    Return (sep, header_row_index).
    """
    with open_maybe_gzip(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            # First non-empty line is typically header
            if "\t" in line:
                return "\t", 0
            if "," in line and line.lower().startswith("alleleid"):
                return ",", 0
            # sometimes comments precede header; keep scanning
            if line.startswith("#"):
                continue
            # fallback
            return "\t", 0
    return "\t", 0


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of these columns found: {candidates}. Found: {list(df.columns)[:50]}...")


def normalize_clinsig(s: str) -> str:
    # unify spacing/case; keep original semantics
    return " ".join(str(s).strip().split()).lower()


def write_bed(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df[["chrom", "bedStart", "bedEnd"]].to_csv(out_path, sep="\t", header=False, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="ClinVar variant_summary.txt or .txt.gz")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--assembly", default="GRCh38", help="GRCh38 (default) or GRCh37")
    ap.add_argument("--germline_only", action="store_true", help="Keep only OriginSimple == germline (recommended)")
    ap.add_argument("--exclude_conflicting", action="store_true", help="Drop 'conflicting classifications of pathogenicity'")
    ap.add_argument("--include_likely", action="store_true", help="Include 'likely pathogenic' and 'likely benign' in respective tracks")
    args = ap.parse_args()

    sep, header = detect_sep_and_header(args.infile)

    df = pd.read_csv(
        args.infile,
        sep=sep,
        header=header,
        dtype=str,
        low_memory=False,
        comment=None,  # ClinVar headers are not commented; keep everything
    )

    # Column names vary slightly across releases; locate what we need.
    col_assembly = find_col(df, ["Assembly", "assembly"])
    col_chr = find_col(df, ["Chromosome", "chromosome", "Chr"])
    col_start = find_col(df, ["Start", "start"])
    col_stop = find_col(df, ["Stop", "stop", "End", "end"])
    col_clinsig = find_col(df, ["ClinicalSignificance", "Clinical significance", "ClinicalSignificance (Last reviewed)", "ClinSigSimple", "ClinicalSignificanceSimple"])
    # OriginSimple exists in ClinVar variant_summary; use if present
    col_origin_simple = None
    for c in ["OriginSimple", "origin_simple", "Origin_simple", "Origin_simple "]:
        if c in df.columns:
            col_origin_simple = c
            break

    # Basic cleaning
    df = df.dropna(subset=[col_assembly, col_chr, col_start, col_stop, col_clinsig]).copy()

    # Filter assembly
    df = df[df[col_assembly] == args.assembly].copy()

    # Optional germline restriction
    if args.germline_only and col_origin_simple is not None:
        df = df[df[col_origin_simple].str.lower().fillna("") == "germline"].copy()

    # Clinical significance normalization
    df["_clinsig_norm"] = df[col_clinsig].map(normalize_clinsig)

    if args.exclude_conflicting:
        df = df[df["_clinsig_norm"] != "conflicting classifications of pathogenicity"].copy()

    # Convert to BED coordinates
    # ClinVar Start/Stop are 1-based inclusive.
    df["_start"] = df[col_start].astype(int)
    df["_stop"] = df[col_stop].astype(int)

    # BED: 0-based, end-exclusive
    df["chrom"] = df[col_chr].apply(lambda x: f"chr{x}" if str(x).isdigit() or str(x) in [str(i) for i in range(1,23)] else str(x))
    # ClinVar Chromosome often already like "1" or "X". Sometimes it's "chr1"; handle:
    df["chrom"] = df["chrom"].str.replace("^chrchr", "chr", regex=True)

    df["bedStart"] = df["_start"] - 1
    df["bedEnd"] = df["_stop"]  # inclusive -> exclusive by keeping same numeric value

    # Drop invalid
    df = df[df["bedEnd"] > df["bedStart"]].copy()
    df.loc[df["bedStart"] < 0, "bedStart"] = 0

    # Define which clinsig values map to labels
    pathogenic_set = {"pathogenic"}
    benign_set = {"benign"}

    if args.include_likely:
        pathogenic_set |= {"likely pathogenic"}
        benign_set |= {"likely benign"}

    patho = df[df["_clinsig_norm"].isin(pathogenic_set)].copy()
    benign = df[df["_clinsig_norm"].isin(benign_set)].copy()

    out_patho = os.path.join(args.outdir, f"clinvar_{args.assembly}_pathogenic.bed")
    out_benign = os.path.join(args.outdir, f"clinvar_{args.assembly}_benign.bed")

    write_bed(patho, out_patho)
    write_bed(benign, out_benign)

    # Also write a small QC summary
    qc_path = os.path.join(args.outdir, f"clinvar_{args.assembly}_qc.txt")
    with open(qc_path, "w", encoding="utf-8") as f:
        f.write(f"rows_total_filtered\t{len(df)}\n")
        f.write(f"rows_pathogenic\t{len(patho)}\n")
        f.write(f"rows_benign\t{len(benign)}\n")
        f.write("top_clinsig_counts\n")
        f.write(df["_clinsig_norm"].value_counts().head(30).to_string())
        f.write("\n")

    print("Wrote:")
    print(" ", out_patho)
    print(" ", out_benign)
    print(" ", qc_path)


if __name__ == "__main__":
    main()
