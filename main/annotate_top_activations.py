"""
annotate_top_activations.py  (optimized)

Instead of one BEDTools call per (feature, concept) pair, we do:
  1. Build one BED file containing ALL top-activating *tokens* (1 bp intervals),
     labelled by feature_idx — uses token_pos + window coords from top_activations.pt
  2. For each concept BED, run ONE intersect against the full token BED
  3. Parse the result and groupby feature_idx to get per-feature counts

This reduces BEDTools subprocess calls from O(n_features * n_concepts) → O(n_concepts).

Chromosome naming matches concept_feature_analysis / embedding shards via
utils.genomics_coords (same ``use_chr`` convention inferred from activation coords).
"""

import argparse
import csv
import itertools
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import pybedtools

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.genomics_coords import (  # noqa: E402
    infer_use_chr_from_top_activation_coords,
    token_one_bp_bed,
)


def normalise_to(bed: pybedtools.BedTool, use_chr: bool) -> pybedtools.BedTool:
    """Normalize concept BED chrom names to match embedding / token BED convention."""
    intervals = list(bed)
    if not intervals:
        return bed
    lines = []
    for i in intervals:
        chrom = i.chrom
        if use_chr and not chrom.startswith("chr"):
            chrom = "chr" + chrom
        elif not use_chr and chrom.startswith("chr"):
            chrom = chrom[len("chr") :]
        lines.append("\t".join([chrom] + list(i.fields[1:])))
    return pybedtools.BedTool("\n".join(lines) + "\n", from_string=True).sort()


def build_all_tokens_bed(
    coords: list,
    token_pos: torch.Tensor,
    act_values: torch.Tensor,
    use_chr: bool,
    seq_len: Optional[int],
) -> pybedtools.BedTool:
    """
    One 1 bp half-open interval per top activation (same genomic point as
    concept_feature_analysis token labelling).
    """
    lines: List[str] = []
    n_features = len(coords)
    for fi in range(n_features):
        row_coords = coords[fi]
        for rank, coord in enumerate(row_coords):
            if coord is None:
                continue
            chrom, start, end = coord[0], int(coord[1]), int(coord[2])
            if chrom == "":
                continue
            L = int(seq_len) if seq_len is not None else max(1, end - start)
            tp = int(token_pos[fi, rank].item())
            c, s, e = token_one_bp_bed(chrom, start, end, L, tp, use_chr)
            lines.append(f"{c}\t{s}\t{e}\t{fi}")

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


def pairwise_cells_one_pair(
    all_tokens_bed: pybedtools.BedTool,
    bed_a: pybedtools.BedTool,
    bed_b: pybedtools.BedTool,
    n_features: int,
) -> dict[str, list[int]]:
    """Venn-style counts for one concept pair (same semantics as count_pairwise loop body)."""
    a_only  = [0] * n_features
    b_only  = [0] * n_features
    both    = [0] * n_features

    if len(all_tokens_bed) == 0:
        return {"A_only": a_only, "B_only": b_only, "both": both}

    in_a = all_tokens_bed.intersect(bed_a, u=True)
    if len(in_a) > 0:
        in_ab = in_a.intersect(bed_b, u=True)
        for interval in in_ab:
            both[int(interval.name)] += 1

    if len(in_a) > 0:
        in_a_not_b = in_a.intersect(bed_b, v=True)
        for interval in in_a_not_b:
            a_only[int(interval.name)] += 1

    in_b = all_tokens_bed.intersect(bed_b, u=True)
    if len(in_b) > 0:
        in_b_not_a = in_b.intersect(bed_a, v=True)
        for interval in in_b_not_a:
            b_only[int(interval.name)] += 1

    return {"A_only": a_only, "B_only": b_only, "both": both}


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
    pairwise_hits: dict,
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
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cached per-concept / pairwise intersections in out_dir when present.",
    )
    return p.parse_args()


ANNOT_CACHE_DIR = ".annotate_resume"


def _safe_cache_token(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _annotate_cache_paths(out_dir: str) -> str:
    return os.path.join(out_dir, ANNOT_CACHE_DIR)


def _annotate_clear_cache(cache_dir: str) -> None:
    if not os.path.isdir(cache_dir):
        return
    for fn in os.listdir(cache_dir):
        try:
            os.remove(os.path.join(cache_dir, fn))
        except OSError:
            pass
    try:
        os.rmdir(cache_dir)
    except OSError:
        pass


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.chmod(args.out_dir, 0o777)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.chmod(args.tmp_dir, 0o777)
    pybedtools.set_tempdir(args.tmp_dir)

    enrich_path = os.path.join(args.out_dir, "enrichment.csv")
    venn_path   = os.path.join(args.out_dir, "venn.csv")
    cache_dir   = _annotate_cache_paths(args.out_dir)
    if args.resume and os.path.isfile(enrich_path) and os.path.isfile(venn_path):
        print("[resume] enrichment.csv and venn.csv already exist — exiting.")
        return

    print(f"Loading {args.top_activations} ...")
    data = torch.load(args.top_activations, map_location="cpu")
    coords = data["coords"]
    act_values = data["act_values"]
    token_pos = data.get("token_pos")
    seq_len = data.get("seq_len")
    if token_pos is None:
        raise ValueError(
            "top_activations.pt must contain 'token_pos' (re-run find_top_activations.py). "
            "Annotation uses per-token 1 bp intervals aligned with concept_feature_analysis."
        )
    n_features = len(coords)

    use_chr = infer_use_chr_from_top_activation_coords(coords)
    print(f"  Chromosome naming (match embeddings): use_chr={use_chr}")

    # Per-feature total valid token counts (non-empty window slots)
    n_totals = [
        sum(
            1
            for c in coord_list
            if c is not None and c != ("", "", "")
        )
        for coord_list in coords
    ]

    # Load concept BEDs (normalized to same chr style as token intervals)
    bed_paths = sorted(Path(args.bed_dir).glob("*.bed"))
    if not bed_paths:
        raise FileNotFoundError(f"No .bed files in {args.bed_dir}")
    print(f"Found {len(bed_paths)} concept BED files")
    concept_beds: dict[str, pybedtools.BedTool] = {
        bp.stem: normalise_to(pybedtools.BedTool(str(bp)).sort(), use_chr)
        for bp in bed_paths
    }
    concept_names = list(concept_beds)
    pairs = list(itertools.combinations(concept_names, 2))

    # Build the single combined token BED  (done once)
    print("Building combined token BED (1 bp per top activation) ...")
    all_tokens_bed = build_all_tokens_bed(coords, token_pos, act_values, use_chr, seq_len)
    print(f"  {len(all_tokens_bed)} total token intervals across {n_features} features")

    if args.resume:
        os.makedirs(cache_dir, exist_ok=True)
        os.chmod(cache_dir, 0o777)

    # One intersect per concept
    print("Intersecting with concept BEDs ...")
    per_concept_hits: dict[str, list[int]] = {}
    for name, bed in concept_beds.items():
        hit_np = os.path.join(cache_dir, f"hits_{_safe_cache_token(name)}.npy")
        if args.resume and os.path.isfile(hit_np):
            print(f"  {name} ... [resume: cached]")
            per_concept_hits[name] = np.load(hit_np).astype(int).tolist()
            continue
        print(f"  {name} ...")
        hits = count_hits_per_feature(all_tokens_bed, bed, n_features)
        per_concept_hits[name] = hits
        if args.resume:
            np.save(hit_np, np.asarray(hits, dtype=np.int64))

    # Neither: one intersect against merged union
    neither_np = os.path.join(cache_dir, "neither.npy")
    if args.resume and os.path.isfile(neither_np):
        print("Computing 'neither' ... [resume: cached]")
        neither = np.load(neither_np).astype(int).tolist()
    else:
        print("Computing 'neither' ...")
        neither = count_neither_per_feature(all_tokens_bed, concept_beds, n_features)
        if args.resume:
            np.save(neither_np, np.asarray(neither, dtype=np.int64))

    # Pairwise: two intersects per pair
    print(f"Computing {len(pairs)} pairwise intersections ...")
    pairwise_hits: dict[tuple, dict[str, list[int]]] = {}
    for a, b in pairs:
        pw_np = os.path.join(
            cache_dir, f"pair__{_safe_cache_token(a)}__{_safe_cache_token(b)}.npz"
        )
        if args.resume and os.path.isfile(pw_np):
            print(f"  pair ({a}, {b}) ... [resume: cached]")
            z = np.load(pw_np)
            pairwise_hits[(a, b)] = {
                "A_only": z["A_only"].astype(int).tolist(),
                "B_only": z["B_only"].astype(int).tolist(),
                "both":   z["both"].astype(int).tolist(),
            }
            continue
        print(f"  pair ({a}, {b}) ...")
        bed_a = concept_beds[a]
        bed_b = concept_beds[b]
        cells = pairwise_cells_one_pair(all_tokens_bed, bed_a, bed_b, n_features)
        pairwise_hits[(a, b)] = cells
        if args.resume:
            np.savez(
                pw_np,
                A_only=np.asarray(cells["A_only"], dtype=np.int64),
                B_only=np.asarray(cells["B_only"], dtype=np.int64),
                both=np.asarray(cells["both"], dtype=np.int64),
            )

    # Write outputs
    write_enrichment(
        enrich_path,
        n_features, n_totals, per_concept_hits, neither, concept_names,
    )
    write_venn(
        venn_path,
        n_features, n_totals, per_concept_hits, pairwise_hits, neither, pairs,
    )

    if args.resume:
        _annotate_clear_cache(cache_dir)

    print("Done.")
    pybedtools.cleanup()


def _pct(n: int, total: int) -> str:
    return "NA" if total == 0 else f"{100 * n / total:.2f}"


if __name__ == "__main__":
    main()
