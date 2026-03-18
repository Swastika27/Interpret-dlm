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
) -> dict[tuple, list[int]]:
    """
    For each pair (A, B): intersect all_tokens with A, then intersect result with B.
    Returns {(nameA, nameB): [count_per_feature]}.
    Two BEDTools calls per pair regardless of n_features.
    """
    names  = list(concept_beds)
    result = {}
    for a, b in itertools.combinations(names, 2):
        in_a = all_tokens_bed.intersect(concept_beds[a], u=True)
        counts = [0] * n_features
        if len(in_a) > 0:
            in_ab = in_a.intersect(concept_beds[b], u=True)
            for interval in in_ab:
                fi = int(interval.name)
                counts[fi] += 1
        result[(a, b)] = counts
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
                n_both   = pairwise_hits[(a, b)][fi]
                n_a_only = per_concept_hits[a][fi] - n_both
                n_b_only = per_concept_hits[b][fi] - n_both
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
    os.makedirs(args.tmp_dir, exist_ok=True)
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