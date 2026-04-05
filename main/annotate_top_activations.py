"""
annotate_top_activations.py  (optimized)

Instead of one BEDTools call per (feature, concept) pair, we do:
  1. Build one BED file containing ALL top-activating tokens, labelled by feature_idx
  2. For each concept BED, run ONE intersect against the full token BED
  3. Parse the result and groupby feature_idx to get per-feature counts

This reduces BEDTools subprocess calls from O(n_features * n_concepts) → O(n_concepts).
"""

import argparse
import csv
import itertools
import os
from collections import defaultdict
from pathlib import Path

import torch
import pybedtools


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_top_activations(pt_path: str):
    data = torch.load(pt_path, map_location="cpu")
    return data["coords"], data["act_values"]

import re

def normalise_chrom_convention(bed: pybedtools.BedTool) -> pybedtools.BedTool:
    """
    Detect whether the majority of intervals use 'chr' prefix and
    normalise all intervals to match, then re-sort.
    """
    intervals = list(bed)
    if not intervals:
        return bed

    n_chr    = sum(1 for i in intervals if i.chrom.startswith("chr"))
    n_nochr  = len(intervals) - n_chr
    use_chr  = n_chr >= n_nochr   # majority convention wins

    lines = []
    for i in intervals:
        chrom = i.chrom
        if use_chr and not chrom.startswith("chr"):
            chrom = "chr" + chrom
        elif not use_chr and chrom.startswith("chr"):
            chrom = chrom[len("chr"):]
        # Rebuild the line preserving all other fields
        fields = [chrom] + list(i.fields[1:])
        lines.append("\t".join(fields))

    return pybedtools.BedTool("\n".join(lines) + "\n", from_string=True).sort()

def build_all_tokens_bed(coords: list) -> pybedtools.BedTool:
    """
    Build a single BED with ALL tokens from ALL features.
    Name field encodes feature_idx so we can groupby after intersect.

    chrom  start  end  feature_idx
    """
    lines = []
    for fi, coord_list in enumerate(coords):
        for coord in coord_list:
            if coord is None:
                continue
            chrom, start, end = coord
            if chrom == "":
                continue
            lines.append(f"{chrom}\t{start}\t{end}\t{fi}")

    if not lines:
        return pybedtools.BedTool("", from_string=True)

    return pybedtools.BedTool("\n".join(lines) + "\n", from_string=True).sort()


# ---------------------------------------------------------------------------
# Core: one intersect per concept, groupby feature_idx
# ---------------------------------------------------------------------------

def count_hits_per_feature(
    all_tokens_bed: pybedtools.BedTool,
    concept_bed: pybedtools.BedTool,
    n_features: int,
) -> list[int]:
    """
    Run one intersect and return hit counts indexed by feature_idx.
    -u: count each token at most once even if it overlaps multiple concept intervals.
    """
    hits = [0] * n_features
    if len(all_tokens_bed) == 0 or len(concept_bed) == 0:
        return hits

    result = all_tokens_bed.intersect(concept_bed, u=True)
    for interval in result:
        fi = int(interval.name)   # feature_idx stored in name field
        hits[fi] += 1
    return hits


def count_neither_per_feature(
    all_tokens_bed: pybedtools.BedTool,
    concept_beds: dict[str, pybedtools.BedTool],
    n_features: int,
) -> list[int]:
    """
    One intersect with the merged union of all concepts, inverted (-v).
    """
    neither = [0] * n_features
    if len(all_tokens_bed) == 0:
        return neither

    bed_list = list(concept_beds.values())
    universe = bed_list[0].cat(*bed_list[1:], postmerge=True)
    result   = all_tokens_bed.intersect(universe, v=True)
    for interval in result:
        fi = int(interval.name)
        neither[fi] += 1
    return neither


def count_pairwise_per_feature(
    all_tokens_bed: pybedtools.BedTool,
    concept_beds: dict[str, pybedtools.BedTool],
    n_features: int,
) -> dict[tuple, dict[str, list[int]]]:
    """
    For each pair (A, B), compute all four Venn cells independently:
        in_A_only  : overlaps A, does NOT overlap B   → intersect A, then -v B
        in_B_only  : overlaps B, does NOT overlap A   → intersect B, then -v A
        in_both    : overlaps A AND B                 → intersect A, then -u B
        in_neither : computed separately in count_neither_per_feature

    Returns {(nameA, nameB): {"A_only": [...], "B_only": [...], "both": [...]}}
    Each list is indexed by feature_idx.
    """
    names  = list(concept_beds)
    result = {}

    for a, b in itertools.combinations(names, 2):
        a_only  = [0] * n_features
        b_only  = [0] * n_features
        both    = [0] * n_features

        if len(all_tokens_bed) == 0:
            result[(a, b)] = {"A_only": a_only, "B_only": b_only, "both": both}
            continue

        bed_a = concept_beds[a]
        bed_b = concept_beds[b]

        # Tokens in both A and B
        in_a      = all_tokens_bed.intersect(bed_a, u=True)
        if len(in_a) > 0:
            in_ab = in_a.intersect(bed_b, u=True)
            for interval in in_ab:
                both[int(interval.name)] += 1

        # Tokens in A but NOT B
        if len(in_a) > 0:
            in_a_not_b = in_a.intersect(bed_b, v=True)
            for interval in in_a_not_b:
                a_only[int(interval.name)] += 1

        # Tokens in B but NOT A
        in_b = all_tokens_bed.intersect(bed_b, u=True)
        if len(in_b) > 0:
            in_b_not_a = in_b.intersect(bed_a, v=True)
            for interval in in_b_not_a:
                b_only[int(interval.name)] += 1

        result[(a, b)] = {"A_only": a_only, "B_only": b_only, "both": both}

    return result


# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------

def write_enrichment(
    path: str,
    n_features: int,
    n_totals: list[int],
    per_concept_hits: dict[str, list[int]],
    neither: list[int],
    concept_names: list[str],
):
    with open(path, "w", newline="") as f:
        fieldnames = (
            ["feature_idx", "n_total"]
            + [f"n_{c}" for c in concept_names]
            + [f"pct_{c}" for c in concept_names]
            + ["n_neither", "pct_neither"]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fi in range(n_features):
            nt  = n_totals[fi]
            row = {"feature_idx": fi, "n_total": nt}
            for c in concept_names:
                n = per_concept_hits[c][fi]
                row[f"n_{c}"]   = n
                row[f"pct_{c}"] = _pct(n, nt)
            row["n_neither"]   = neither[fi]
            row["pct_neither"] = _pct(neither[fi], nt)
            writer.writerow(row)
    print(f"Saved {path}")


def write_venn(
    path: str,
    n_features: int,
    n_totals: list[int],
    per_concept_hits: dict[str, list[int]],
    pairwise_hits: dict[tuple, list[int]],
    neither: list[int],
    pairs: list[tuple],
):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "feature_idx", "concept_A", "concept_B", "n_total",
            "n_A_only", "n_B_only", "n_both", "n_neither",
            "pct_A_only", "pct_B_only", "pct_both", "pct_neither",
        ])
        writer.writeheader()
        for fi in range(n_features):
            nt = n_totals[fi]
            for a, b in pairs:
                pw       = pairwise_hits[(a, b)]
                n_both   = pw["both"][fi]
                n_a_only = pw["A_only"][fi]
                n_b_only = pw["B_only"][fi]
                writer.writerow({
                    "feature_idx": fi, "concept_A": a, "concept_B": b,
                    "n_total":   nt,
                    "n_A_only":  n_a_only,  "pct_A_only":  _pct(n_a_only, nt),
                    "n_B_only":  n_b_only,  "pct_B_only":  _pct(n_b_only, nt),
                    "n_both":    n_both,    "pct_both":    _pct(n_both,   nt),
                    "n_neither": neither[fi], "pct_neither": _pct(neither[fi], nt),
                })
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--top_activations", required=True)
    p.add_argument("--bed_dir",         required=True)
    p.add_argument("--out_dir",         required=True)
    p.add_argument("--tmp_dir",         default="/tmp/pybedtools")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.chmod(args.out_dir, 0o777)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.chmod(args.tmp_dir, 0o777)
    pybedtools.set_tempdir(args.tmp_dir)

    print(f"Loading {args.top_activations} ...")
    coords, act_values = load_top_activations(args.top_activations)
    n_features = len(coords)

    # Per-feature total valid token counts
    n_totals = [
        sum(1 for c in coord_list if c is not None and c != ("", "", ""))
        for coord_list in coords
    ]

    # Load concept BEDs
    bed_paths = sorted(Path(args.bed_dir).glob("*.bed"))
    if not bed_paths:
        raise FileNotFoundError(f"No .bed files in {args.bed_dir}")
    print(f"Found {len(bed_paths)} concept BED files")
    concept_beds: dict[str, pybedtools.BedTool] = {
        bp.stem: pybedtools.BedTool(str(bp)).sort()
        for bp in bed_paths
    }
    concept_names = list(concept_beds)
    pairs = list(itertools.combinations(concept_names, 2))

    # Build the single combined token BED  (done once)
    print("Building combined token BED ...")
    all_tokens_bed = build_all_tokens_bed(coords)
    print(f"  {len(all_tokens_bed)} total token intervals across {n_features} features")

    # Detect convention from token bed (your data is the reference)
    sample = next(iter(all_tokens_bed))
    tokens_use_chr = sample.chrom.startswith("chr")

    def normalise_to(bed: pybedtools.BedTool, use_chr: bool) -> pybedtools.BedTool:
        intervals = list(bed)
        if not intervals:
            return bed
        lines = []
        for i in intervals:
            chrom = i.chrom
            if use_chr and not chrom.startswith("chr"):
                chrom = "chr" + chrom
            elif not use_chr and chrom.startswith("chr"):
                chrom = chrom[len("chr"):]
            lines.append("\t".join([chrom] + list(i.fields[1:])))
        return pybedtools.BedTool("\n".join(lines) + "\n", from_string=True).sort()

    # Apply when loading concept BEDs
    concept_beds: dict[str, pybedtools.BedTool] = {
        bp.stem: normalise_to(pybedtools.BedTool(str(bp)).sort(), tokens_use_chr)
        for bp in bed_paths
    }
    # One intersect per concept
    print("Intersecting with concept BEDs ...")
    per_concept_hits: dict[str, list[int]] = {}
    for name, bed in concept_beds.items():
        print(f"  {name} ...")
        per_concept_hits[name] = count_hits_per_feature(all_tokens_bed, bed, n_features)

    # Neither: one intersect against merged union
    print("Computing 'neither' ...")
    neither = count_neither_per_feature(all_tokens_bed, concept_beds, n_features)

    # Pairwise: two intersects per pair
    print(f"Computing {len(pairs)} pairwise intersections ...")
    pairwise_hits = count_pairwise_per_feature(all_tokens_bed, concept_beds, n_features)

    # Write outputs
    write_enrichment(
        os.path.join(args.out_dir, "enrichment.csv"),
        n_features, n_totals, per_concept_hits, neither, concept_names,
    )
    write_venn(
        os.path.join(args.out_dir, "venn.csv"),
        n_features, n_totals, per_concept_hits, pairwise_hits, neither, pairs,
    )

    print("Done.")
    pybedtools.cleanup()


def _pct(n: int, total: int) -> str:
    return "NA" if total == 0 else f"{100 * n / total:.2f}"


if __name__ == "__main__":
    main()