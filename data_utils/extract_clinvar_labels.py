#!/usr/bin/env python3
"""
extract_clinvar_labels.py
-------------------------
Convert ClinVar variant_summary.txt(.gz) into pathogenic and benign BED files
suitable for use as concept annotations in SAE training/evaluation pipelines.

Key correctness fixes vs. prior version
----------------------------------------
1. Header detection: ClinVar's header starts with '#AlleleID'. The old code
   skipped lines beginning with '#', so pandas received the first *data* row
   as column names — causing every subsequent find_col lookup to bind to the
   wrong field.  We now strip the leading '#' from the header row in-place.

2. Structural variant filtering: ClinVar contains CNVs and structural variants
   that span megabases.  Without a length cap the output BED covered ~66 % of
   the genome.  We cap variant length at MAX_VARIANT_BP (default 50 bp) so
   only SNVs and small indels are retained.  Pass --max_variant_bp 0 to keep
   all lengths (not recommended for sequence-model concept annotations).

3. Type filter: Optionally restrict to specific VariationType values (e.g.
   "single nucleotide variant") via --types.

4. Deduplication: Each AlleleID appears once per assembly in the file; after
   assembly filtering rows should already be unique per variant, but we
   deduplicate on (chrom, bedStart, bedEnd) to avoid double-counting variants
   submitted under multiple conditions.

Usage
-----
python data_utils/extract_clinvar_labels.py \\
    --infile data/clinvar/variant_summary.txt.gz \\
    --outdir data/concepts/ \\
    --assembly GRCh38 \\
    --germline_only \\
    --exclude_conflicting \\
    --include_likely \\
    --max_variant_bp 50

Outputs (in --outdir):
  clinvar_GRCh38_pathogenic.bed
  clinvar_GRCh38_benign.bed
  clinvar_GRCh38_qc.txt
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _open(path: str):
    """Return a text-mode file handle, transparently decompressing .gz."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _read_clinvar_tsv(path: str) -> pd.DataFrame:
    """
    Read variant_summary.txt robustly.

    ClinVar's header line begins with '#AlleleID\t...'.  pandas' comment=
    parameter would drop that line entirely, and the old code's line-scanner
    skipped it because it starts with '#'.  We read the file manually,
    strip the leading '#' from the header, then hand the corrected text to
    pandas so column names are resolved correctly.
    """
    lines: list[str] = []
    with _open(path) as fh:
        for line in fh:
            lines.append(line)

    if not lines:
        raise ValueError(f"Empty file: {path}")

    # Find the header line: first non-blank line (may start with '#')
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            header_idx = i
            break

    # Strip leading '#' from the header line only
    lines[header_idx] = lines[header_idx].lstrip("#")

    # Verify we have a tab-delimited header that looks right
    header_fields = lines[header_idx].split("\t")
    if "AlleleID" not in header_fields[0] and "AlleleID" not in header_fields:
        # Try stripping BOM or other artefacts
        lines[header_idx] = lines[header_idx].lstrip("\ufeff").lstrip("#").lstrip()

    text = "".join(lines[header_idx:])  # drop any preamble above the header
    df = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    # Strip stray whitespace from column names (seen in some releases)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"Required column not found. Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _norm_clinsig(s: str) -> str:
    """Lower-case and collapse whitespace."""
    return " ".join(str(s).strip().split()).lower()


def _norm_chrom(x: str) -> str:
    """
    Ensure chromosome name has 'chr' prefix.
    ClinVar Chromosome column typically contains bare '1', 'X', 'MT', etc.
    Guard against already-prefixed 'chr1' or malformed 'chrchr1'.
    """
    s = str(x).strip()
    if not s.startswith("chr"):
        s = "chr" + s
    # Remove accidental double prefix
    while s.startswith("chrchr"):
        s = s[3:]
    return s


# ---------------------------------------------------------------------------
# BED output
# ---------------------------------------------------------------------------

def _write_bed(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out = df[["chrom", "bedStart", "bedEnd"]].copy()
    out = out.drop_duplicates()
    out = out.sort_values(["chrom", "bedStart", "bedEnd"])
    out.to_csv(out_path, sep="\t", header=False, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract pathogenic/benign BED files from ClinVar variant_summary.txt"
    )
    ap.add_argument("--infile", required=True,
                    help="ClinVar variant_summary.txt or variant_summary.txt.gz")
    ap.add_argument("--outdir", required=True,
                    help="Output directory for BED files and QC report")
    ap.add_argument("--assembly", default="GRCh38",
                    help="Genome assembly to keep (default: GRCh38)")
    ap.add_argument("--germline_only", action="store_true",
                    help="Keep only rows where OriginSimple == 'germline' (recommended)")
    ap.add_argument("--exclude_conflicting", action="store_true",
                    help="Drop rows classified as 'conflicting classifications of pathogenicity'")
    ap.add_argument("--include_likely", action="store_true",
                    help="Include 'likely pathogenic' / 'likely benign' in respective tracks")
    ap.add_argument("--max_variant_bp", type=int, default=50,
                    help=(
                        "Drop variants whose (Stop - Start + 1) exceeds this length. "
                        "Prevents large CNVs/SVs from inflating coverage. "
                        "Set to 0 to disable (not recommended). Default: 50"
                    ))
    ap.add_argument("--types", nargs="*", default=None,
                    help=(
                        "Restrict to specific Type values (case-insensitive). "
                        "E.g. --types 'single nucleotide variant' deletion insertion indel. "
                        "Default: no type filter (length cap alone is usually sufficient)."
                    ))
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"Reading {args.infile} …")
    df = _read_clinvar_tsv(args.infile)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")

    # ------------------------------------------------------------------
    # Locate columns
    # ------------------------------------------------------------------
    col_assembly = _find_col(df, ["Assembly", "assembly"])
    col_chr      = _find_col(df, ["Chromosome", "chromosome", "Chr"])
    col_start    = _find_col(df, ["Start", "start", "PositionVCF"])
    col_stop     = _find_col(df, ["Stop", "stop", "End", "end"])
    col_clinsig  = _find_col(df, [
        "ClinicalSignificance",
        "Clinical significance",
        "ClinicalSignificance (Last reviewed)",
        "ClinSigSimple",
        "ClinicalSignificanceSimple",
    ])
    col_type     = _find_col(df, ["Type", "VariationType", "VariationClass"], required=False)
    col_origin   = _find_col(df, [
        "OriginSimple", "origin_simple", "Origin_simple", "Origin_simple "
    ], required=False)

    print(f"  Columns resolved: assembly={col_assembly!r}, chr={col_chr!r}, "
          f"start={col_start!r}, stop={col_stop!r}, clinsig={col_clinsig!r}, "
          f"type={col_type!r}, origin={col_origin!r}")

    # ------------------------------------------------------------------
    # Basic cleaning
    # ------------------------------------------------------------------
    required_cols = [col_assembly, col_chr, col_start, col_stop, col_clinsig]
    before = len(df)
    df = df.dropna(subset=required_cols).copy()
    print(f"  Dropped {before - len(df):,} rows with missing required fields → {len(df):,} remain.")

    # ------------------------------------------------------------------
    # Assembly filter
    # ------------------------------------------------------------------
    before = len(df)
    df = df[df[col_assembly].str.strip() == args.assembly].copy()
    print(f"  Assembly={args.assembly}: {len(df):,} rows (dropped {before - len(df):,}).")
    if len(df) == 0:
        sys.exit(f"ERROR: No rows remain after assembly filter. "
                 f"Available assemblies: {df[col_assembly].unique().tolist()}")

    # ------------------------------------------------------------------
    # Germline filter
    # ------------------------------------------------------------------
    if args.germline_only:
        if col_origin is not None:
            before = len(df)
            df = df[df[col_origin].str.strip().str.lower().fillna("") == "germline"].copy()
            print(f"  Germline filter: {len(df):,} rows (dropped {before - len(df):,}).")
        else:
            print("  WARNING: --germline_only requested but OriginSimple column not found; skipping.")

    # ------------------------------------------------------------------
    # Type filter
    # ------------------------------------------------------------------
    if args.types and col_type is not None:
        types_lower = {t.lower() for t in args.types}
        before = len(df)
        df = df[df[col_type].str.strip().str.lower().isin(types_lower)].copy()
        print(f"  Type filter {types_lower}: {len(df):,} rows (dropped {before - len(df):,}).")
    elif args.types:
        print("  WARNING: --types specified but Type column not found; skipping type filter.")

    # ------------------------------------------------------------------
    # Coordinate conversion: ClinVar uses 1-based inclusive coordinates.
    # BED uses 0-based half-open [start, end).
    #
    #   ClinVar SNV:   Start=100  Stop=100   → bedStart=99  bedEnd=100  (1 bp)
    #   ClinVar indel: Start=100  Stop=103   → bedStart=99  bedEnd=103  (4 bp)
    # ------------------------------------------------------------------
    df["_start_1based"] = pd.to_numeric(df[col_start], errors="coerce")
    df["_stop_1based"]  = pd.to_numeric(df[col_stop],  errors="coerce")

    before = len(df)
    df = df.dropna(subset=["_start_1based", "_stop_1based"]).copy()
    df["_start_1based"] = df["_start_1based"].astype(int)
    df["_stop_1based"]  = df["_stop_1based"].astype(int)
    print(f"  Dropped {before - len(df):,} rows with non-numeric coordinates.")

    df["bedStart"] = df["_start_1based"] - 1          # 0-based
    df["bedEnd"]   = df["_stop_1based"]               # inclusive → exclusive (same number)

    # ------------------------------------------------------------------
    # Length filter — this is the main guard against large SVs/CNVs
    # ------------------------------------------------------------------
    df["_variant_len"] = df["bedEnd"] - df["bedStart"]

    before = len(df)
    df = df[df["_variant_len"] > 0].copy()            # drop zero/negative-length rows
    print(f"  Dropped {before - len(df):,} zero/negative-length rows.")

    if args.max_variant_bp > 0:
        before = len(df)
        df = df[df["_variant_len"] <= args.max_variant_bp].copy()
        print(f"  Length cap ≤{args.max_variant_bp} bp: {len(df):,} rows (dropped {before - len(df):,} large variants).")

    # Clamp negative starts (shouldn't happen for well-formed ClinVar data)
    df.loc[df["bedStart"] < 0, "bedStart"] = 0

    # ------------------------------------------------------------------
    # Chromosome name normalisation
    # ------------------------------------------------------------------
    df["chrom"] = df[col_chr].map(_norm_chrom)

    # ------------------------------------------------------------------
    # Clinical significance
    # ------------------------------------------------------------------
    df["_clinsig_norm"] = df[col_clinsig].map(_norm_clinsig)

    if args.exclude_conflicting:
        before = len(df)
        df = df[df["_clinsig_norm"] != "conflicting classifications of pathogenicity"].copy()
        print(f"  Excluded conflicting: {len(df):,} rows (dropped {before - len(df):,}).")

    pathogenic_set = {"pathogenic"}
    benign_set     = {"benign"}
    if args.include_likely:
        pathogenic_set |= {"likely pathogenic"}
        benign_set     |= {"likely benign"}

    patho  = df[df["_clinsig_norm"].isin(pathogenic_set)].copy()
    benign = df[df["_clinsig_norm"].isin(benign_set)].copy()

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    out_patho  = os.path.join(args.outdir, f"clinvar_{args.assembly}_pathogenic.bed")
    out_benign = os.path.join(args.outdir, f"clinvar_{args.assembly}_benign.bed")
    qc_path    = os.path.join(args.outdir, f"clinvar_{args.assembly}_qc.txt")

    _write_bed(patho,  out_patho)
    _write_bed(benign, out_benign)

    # Genome coverage estimate (rough, ignores overlaps between intervals)
    genome_size_bp = 3_099_441_038  # GRCh38 primary assembly
    patho_bp  = int((patho["bedEnd"]  - patho["bedStart"]).sum())
    benign_bp = int((benign["bedEnd"] - benign["bedStart"]).sum())

    with open(qc_path, "w", encoding="utf-8") as fh:
        fh.write(f"assembly\t{args.assembly}\n")
        fh.write(f"rows_after_all_filters\t{len(df)}\n")
        fh.write(f"rows_pathogenic\t{len(patho)}\n")
        fh.write(f"rows_benign\t{len(benign)}\n")
        fh.write(f"patho_bp_raw\t{patho_bp}\n")
        fh.write(f"benign_bp_raw\t{benign_bp}\n")
        fh.write(f"patho_genome_fraction\t{patho_bp / genome_size_bp:.6f}\n")
        fh.write(f"benign_genome_fraction\t{benign_bp / genome_size_bp:.6f}\n")
        fh.write(f"max_variant_bp_cap\t{args.max_variant_bp}\n")
        fh.write(f"germline_only\t{args.germline_only}\n")
        fh.write(f"exclude_conflicting\t{args.exclude_conflicting}\n")
        fh.write(f"include_likely\t{args.include_likely}\n")
        fh.write("\ntop_clinsig_value_counts\n")
        fh.write(df["_clinsig_norm"].value_counts().head(30).to_string())
        fh.write("\n")
        if col_type is not None:
            fh.write("\ntop_type_value_counts\n")
            fh.write(df[col_type].str.strip().str.lower().value_counts().head(20).to_string())
            fh.write("\n")
        fh.write("\nvariant_len_percentiles\n")
        fh.write(df["_variant_len"].describe(percentiles=[.5, .9, .99, .999]).to_string())
        fh.write("\n")

    print("\nWrote:")
    print(f"  {out_patho}  ({len(patho):,} intervals, ~{patho_bp / genome_size_bp * 100:.4f}% of genome)")
    print(f"  {out_benign}  ({len(benign):,} intervals, ~{benign_bp / genome_size_bp * 100:.4f}% of genome)")
    print(f"  {qc_path}")


if __name__ == "__main__":
    main()