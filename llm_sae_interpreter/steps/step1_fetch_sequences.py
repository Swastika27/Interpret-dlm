"""
steps/step1_fetch_sequences.py — Fetch DNA sequences for all activating loci.

Reads top_activations.csv, fetches the 512 bp genomic window for each row
from the hg38 reference FASTA using pyfaidx, and saves the result to
output/sequences.csv with an additional 'sequence' column.

Run:
    python steps/step1_fetch_sequences.py
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TOP_ACTIVATIONS_CSV,
    GENOME_FASTA,
    SEQUENCES_CSV,
)
from utils.io import load_top_activations
from utils.genome import fetch_sequence


def run() -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1: Fetching DNA sequences from reference genome")
    print("=" * 60)

    # ── Load input ──────────────────────────────────────────────────────────
    print(f"\nLoading {TOP_ACTIVATIONS_CSV} ...")
    df = load_top_activations(TOP_ACTIVATIONS_CSV)
    print(f"  {len(df)} rows | {df['feature_idx'].nunique()} features")

    # ── Resume support ──────────────────────────────────────────────────────
    if SEQUENCES_CSV.exists():
        existing = pd.read_csv(SEQUENCES_CSV, usecols=["feature_idx", "rank"])
        done_pairs = set(zip(existing["feature_idx"], existing["rank"]))
        n_todo = len(df) - len(done_pairs)
        print(f"  Resuming: {len(done_pairs)} rows already done, {n_todo} remaining")
        df_todo = df[
            ~df.apply(lambda r: (r["feature_idx"], r["rank"]) in done_pairs, axis=1)
        ]
        append_mode = True
    else:
        df_todo = df
        append_mode = False

    if df_todo.empty:
        print("  All rows already processed. Loading existing file.")
        return pd.read_csv(SEQUENCES_CSV, dtype={"coord_chrom": str})

    # ── Fetch sequences ─────────────────────────────────────────────────────
    print(f"\nFetching sequences from {GENOME_FASTA} ...")
    sequences = []
    errors    = []

    for _, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Fetching"):
        try:
            seq, actual_start, actual_end = fetch_sequence(
                fasta_path=str(GENOME_FASTA),
                chrom=row["coord_chrom"],
                start=int(row["coord_start"]),
                end=int(row["coord_end"]),
            )
            sequences.append(seq)
        except Exception as exc:
            # Store empty string on failure; log the error
            sequences.append("")
            errors.append({
                "feature_idx": row["feature_idx"],
                "rank":        row["rank"],
                "coord":       f"{row['coord_chrom']}:{row['coord_start']}-{row['coord_end']}",
                "error":       str(exc),
            })

    df_todo = df_todo.copy()
    df_todo["sequence"] = sequences

    # ── Save ────────────────────────────────────────────────────────────────
    if append_mode:
        df_todo.to_csv(SEQUENCES_CSV, mode="a", header=False, index=False)
        result = pd.read_csv(SEQUENCES_CSV, dtype={"coord_chrom": str})
    else:
        df_todo.to_csv(SEQUENCES_CSV, index=False)
        result = df_todo

    print(f"\n  Saved {len(result)} rows → {SEQUENCES_CSV}")

    if errors:
        import json
        err_path = SEQUENCES_CSV.parent / "step1_errors.jsonl"
        with open(err_path, "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        print(f"  WARNING: {len(errors)} fetch errors logged → {err_path}")
    else:
        print("  No fetch errors.")

    # ── Summary ─────────────────────────────────────────────────────────────
    n_empty = (result["sequence"] == "").sum()
    n_ok    = len(result) - n_empty
    print(f"\nSummary:")
    print(f"  Sequences fetched:  {n_ok}")
    print(f"  Failed (empty seq): {n_empty}")
    print(f"  Unique features:    {result['feature_idx'].nunique()}")
    if n_ok > 0:
        sample_len = result[result["sequence"] != ""]["sequence"].iloc[0]
        print(f"  Example sequence length: {len(sample_len)} bp")

    return result


if __name__ == "__main__":
    run()
