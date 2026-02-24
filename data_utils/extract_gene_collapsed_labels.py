#!/usr/bin/env python3
"""
Extract gene-collapsed (genome-level union) labels from a GENCODE GTF (GRCh38).

Outputs BED (0-based, half-open) for:
  - exon
  - CDS
  - five_prime_UTR
  - three_prime_UTR
  - intron (computed per transcript, then union)
  - splice_donor_2bp (±2bp around donor site)
  - splice_acceptor_2bp (±2bp around acceptor site)
  - promoter (TSS-based; default upstream=1000, downstream=100; strand-aware)

Gene-collapsed here means: we do NOT keep transcript/gene IDs in the labels; we take
the union across all transcripts/genes genome-wide for each label type.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


ATTR_GENE_ID_RE = re.compile(r'gene_id "([^"]+)"')
ATTR_TX_ID_RE = re.compile(r'transcript_id "([^"]+)"')


def parse_gene_id(attr: str) -> Optional[str]:
    m = ATTR_GENE_ID_RE.search(attr)
    return m.group(1) if m else None


def parse_transcript_id(attr: str) -> Optional[str]:
    m = ATTR_TX_ID_RE.search(attr)
    return m.group(1) if m else None


def read_gtf(gtf_path: str) -> pd.DataFrame:
    """
    Read GTF (plain or .gz) into a DataFrame.
    Converts GTF coordinates to BED coordinates:
      start: 1-based inclusive -> 0-based
      end:   1-based inclusive -> end-exclusive (same numeric value as GTF end)
    """
    df = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"],
        dtype={
            "chrom": "string",
            "source": "string",
            "feature": "string",
            "start": "int64",
            "end": "int64",
            "score": "string",
            "strand": "string",
            "frame": "string",
            "attributes": "string",
        },
    )

    # Convert to BED coords
    df["start"] = df["start"] - 1
    # df["end"] stays as-is (end-exclusive in BED after conversion)

    # Extract IDs (needed for intron computation and promoter/TSS)
    df["gene_id"] = df["attributes"].map(parse_gene_id)
    df["transcript_id"] = df["attributes"].map(parse_transcript_id)

    return df


def merge_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge overlapping/touching intervals per chromosome.

    Input df must have columns: chrom, start, end
    Output has same, merged, sorted.
    """
    if df.empty:
        return df[["chrom", "start", "end"]].copy()

    # Keep only required columns and drop NAs
    x = df[["chrom", "start", "end"]].dropna().copy()
    x["start"] = x["start"].astype("int64")
    x["end"] = x["end"].astype("int64")

    merged_rows: List[Tuple[str, int, int]] = []

    for chrom, g in x.groupby("chrom", sort=True):
        g = g.sort_values(["start", "end"], kind="mergesort")
        starts = g["start"].to_numpy()
        ends = g["end"].to_numpy()

        cur_s = int(starts[0])
        cur_e = int(ends[0])

        for s, e in zip(starts[1:], ends[1:]):
            s = int(s)
            e = int(e)
            if s <= cur_e:  # overlap or touch
                if e > cur_e:
                    cur_e = e
            else:
                merged_rows.append((str(chrom), cur_s, cur_e))
                cur_s, cur_e = s, e

        merged_rows.append((str(chrom), cur_s, cur_e))

    out = pd.DataFrame(merged_rows, columns=["chrom", "start", "end"])
    return out


def clip_starts_nonneg(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure start >= 0 for BED."""
    if df.empty:
        return df
    out = df.copy()
    out["start"] = out["start"].clip(lower=0)
    # Drop any intervals that became invalid (start >= end)
    out = out[out["start"] < out["end"]]
    return out


def write_bed(df: pd.DataFrame, path: str, label: Optional[str] = None) -> None:
    """
    Write BED:
      chrom  start  end  [label]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df[["chrom", "start", "end"]].copy()
    if label is not None:
        out["label"] = label
    out.to_csv(path, sep="\t", header=False, index=False)


def compute_introns_from_exons(exons: pd.DataFrame) -> pd.DataFrame:
    """
    Compute introns per transcript from exon intervals, then return all intron intervals (unmerged).
    Exons must include: chrom, start, end, strand, transcript_id
    """
    required = {"chrom", "start", "end", "strand", "transcript_id"}
    missing = required - set(exons.columns)
    if missing:
        raise ValueError(f"Exons missing required columns: {missing}")

    ex = exons.dropna(subset=["transcript_id"]).copy()
    if ex.empty:
        return pd.DataFrame(columns=["chrom", "start", "end"])

    intron_rows: List[Tuple[str, int, int]] = []

    # Group by transcript, chromosome, strand (strand not actually required for gaps,
    # but we keep it to avoid mixing if any transcript_id weirdness exists)
    for (tx, chrom, strand), g in ex.groupby(["transcript_id", "chrom", "strand"], sort=False):
        g = g.sort_values(["start", "end"], kind="mergesort")
        starts = g["start"].to_numpy()
        ends = g["end"].to_numpy()

        # Merge exons within transcript first (handles duplicated/overlapping exon entries)
        merged_tx: List[Tuple[int, int]] = []
        cur_s = int(starts[0])
        cur_e = int(ends[0])
        for s, e in zip(starts[1:], ends[1:]):
            s = int(s); e = int(e)
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                merged_tx.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged_tx.append((cur_s, cur_e))

        # Gaps between consecutive merged exons are introns
        for (s1, e1), (s2, e2) in zip(merged_tx[:-1], merged_tx[1:]):
            intr_s = e1
            intr_e = s2
            if intr_s < intr_e:
                intron_rows.append((str(chrom), int(intr_s), int(intr_e)))

    return pd.DataFrame(intron_rows, columns=["chrom", "start", "end"])


def compute_splice_sites_from_exons(exons: pd.DataFrame, radius_bp: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute splice donor and acceptor intervals (±radius_bp) around exon boundaries, strand-aware.

    Returns:
      donors_df, acceptors_df  (unmerged)
    """
    required = {"chrom", "start", "end", "strand"}
    missing = required - set(exons.columns)
    if missing:
        raise ValueError(f"Exons missing required columns: {missing}")

    ex = exons.copy()
    if ex.empty:
        empty = pd.DataFrame(columns=["chrom", "start", "end"])
        return empty, empty

    donor_rows: List[Tuple[str, int, int]] = []
    acceptor_rows: List[Tuple[str, int, int]] = []

    r = int(radius_bp)

    # For BED half-open, boundary at exon start is position = start
    # boundary at exon end is position = end (already exclusive)
    for chrom, start, end, strand in ex[["chrom", "start", "end", "strand"]].itertuples(index=False, name=None):
        chrom = str(chrom)
        start = int(start)
        end = int(end)
        strand = str(strand)

        if strand == "+":
            # acceptor at exon start; donor at exon end
            acc_pos = start
            don_pos = end
        else:
            # reverse for minus strand
            don_pos = start
            acc_pos = end

        don_s = don_pos - r
        don_e = don_pos + r
        acc_s = acc_pos - r
        acc_e = acc_pos + r

        # Convert to valid half-open (ensure non-neg later)
        if don_s < don_e:
            donor_rows.append((chrom, don_s, don_e))
        if acc_s < acc_e:
            acceptor_rows.append((chrom, acc_s, acc_e))

    donors = pd.DataFrame(donor_rows, columns=["chrom", "start", "end"])
    acceptors = pd.DataFrame(acceptor_rows, columns=["chrom", "start", "end"])
    donors = clip_starts_nonneg(donors)
    acceptors = clip_starts_nonneg(acceptors)
    return donors, acceptors


def compute_promoters_from_transcripts(
    transcripts: pd.DataFrame,
    upstream_bp: int = 1000,
    downstream_bp: int = 100,
) -> pd.DataFrame:
    """
    Compute promoter intervals from transcript TSS (strand-aware).
    Uses BED coords for transcript:
      transcript start: 0-based
      transcript end: end-exclusive
    TSS base (0-based):
      '+' : tss0 = start
      '-' : tss0 = end - 1
    Promoter interval (0-based half-open) includes the TSS base:
      '+' : [tss0 - upstream, tss0 + downstream + 1)
      '-' : [tss0 - downstream, tss0 + upstream + 1)
    """
    required = {"chrom", "start", "end", "strand"}
    missing = required - set(transcripts.columns)
    if missing:
        raise ValueError(f"Transcripts missing required columns: {missing}")

    tx = transcripts.copy()
    if tx.empty:
        return pd.DataFrame(columns=["chrom", "start", "end"])

    up = int(upstream_bp)
    down = int(downstream_bp)

    rows: List[Tuple[str, int, int]] = []

    for chrom, start, end, strand in tx[["chrom", "start", "end", "strand"]].itertuples(index=False, name=None):
        chrom = str(chrom)
        start = int(start)
        end = int(end)
        strand = str(strand)

        if strand == "+":
            tss0 = start
            prom_s = tss0 - up
            prom_e = tss0 + down + 1
        else:
            tss0 = end - 1
            prom_s = tss0 - down
            prom_e = tss0 + up + 1

        rows.append((chrom, prom_s, prom_e))

    prom = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    prom = clip_starts_nonneg(prom)
    return prom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", required=True, help="Path to GENCODE GTF (plain or .gz), e.g. gencode.v49.annotation.gtf.gz")
    ap.add_argument("--outdir", required=True, help="Output directory for BED files")
    ap.add_argument("--promoter_upstream", type=int, default=1000, help="Promoter upstream bp (default 1000)")
    ap.add_argument("--promoter_downstream", type=int, default=100, help="Promoter downstream bp (default 100)")
    ap.add_argument("--splice_radius", type=int, default=2, help="Splice site radius in bp (default ±2)")
    ap.add_argument("--write_combined", action="store_true", help="Also write a combined BED with a label column (not mutually exclusive labels).")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    gtf = read_gtf(args.gtf)

    # Basic feature extractions
    exons = gtf[gtf["feature"] == "exon"].copy()
    cds = gtf[gtf["feature"] == "CDS"].copy()
    utr5 = gtf[gtf["feature"] == "five_prime_UTR"].copy()
    utr3 = gtf[gtf["feature"] == "three_prime_UTR"].copy()
    transcripts = gtf[gtf["feature"] == "transcript"].copy()
    genes = gtf[gtf["feature"] == "gene"].copy()

    # Merge (gene-collapsed union)
    exon_merged = merge_intervals(exons)
    cds_merged = merge_intervals(cds)
    utr5_merged = merge_intervals(utr5)
    utr3_merged = merge_intervals(utr3)
    gene_merged = merge_intervals(genes)

    # Introns from exons (per transcript), then union
    introns_raw = compute_introns_from_exons(exons)
    intron_merged = merge_intervals(introns_raw)

    # Splice sites from exons, then union
    donors_raw, acceptors_raw = compute_splice_sites_from_exons(exons, radius_bp=args.splice_radius)
    donor_merged = merge_intervals(donors_raw)
    acceptor_merged = merge_intervals(acceptors_raw)

    # Promoters from transcripts, then union
    promoters_raw = compute_promoters_from_transcripts(
        transcripts,
        upstream_bp=args.promoter_upstream,
        downstream_bp=args.promoter_downstream,
    )
    promoter_merged = merge_intervals(promoters_raw)

    # Write per-label BEDs
    write_bed(exon_merged, os.path.join(outdir, "exon.bed"))
    write_bed(cds_merged, os.path.join(outdir, "CDS.bed"))
    write_bed(utr5_merged, os.path.join(outdir, "five_prime_UTR.bed"))
    write_bed(utr3_merged, os.path.join(outdir, "three_prime_UTR.bed"))
    write_bed(intron_merged, os.path.join(outdir, "intron.bed"))
    write_bed(donor_merged, os.path.join(outdir, f"splice_donor_pm{args.splice_radius}bp.bed"))
    write_bed(acceptor_merged, os.path.join(outdir, f"splice_acceptor_pm{args.splice_radius}bp.bed"))
    write_bed(promoter_merged, os.path.join(outdir, f"promoter_TSS_{args.promoter_upstream}up_{args.promoter_downstream}down.bed"))
    write_bed(gene_merged, os.path.join(outdir, "gene_span.bed"))

    if args.write_combined:
        combined_path = os.path.join(outdir, "combined_labels.bed")
        parts = [
            (exon_merged, "exon"),
            (cds_merged, "CDS"),
            (utr5_merged, "five_prime_UTR"),
            (utr3_merged, "three_prime_UTR"),
            (intron_merged, "intron"),
            (donor_merged, f"splice_donor_pm{args.splice_radius}bp"),
            (acceptor_merged, f"splice_acceptor_pm{args.splice_radius}bp"),
            (promoter_merged, f"promoter_TSS_{args.promoter_upstream}up_{args.promoter_downstream}down"),
            (gene_merged, "gene_span"),
        ]
        # Note: combined file is just concatenation; labels overlap.
        with open(combined_path, "w") as f:
            for df, lab in parts:
                if df.empty:
                    continue
                tmp = df.copy()
                tmp["label"] = lab
                tmp.to_csv(f, sep="\t", header=False, index=False)

    # Quick stats
    stats = {
        "exon": len(exon_merged),
        "CDS": len(cds_merged),
        "five_prime_UTR": len(utr5_merged),
        "three_prime_UTR": len(utr3_merged),
        "intron": len(intron_merged),
        f"splice_donor_pm{args.splice_radius}bp": len(donor_merged),
        f"splice_acceptor_pm{args.splice_radius}bp": len(acceptor_merged),
        f"promoter": len(promoter_merged),
        "gene_span": len(gene_merged),
    }
    print("Wrote BEDs to:", outdir)
    for k, v in stats.items():
        print(f"{k:30s} intervals: {v}")


if __name__ == "__main__":
    main()
