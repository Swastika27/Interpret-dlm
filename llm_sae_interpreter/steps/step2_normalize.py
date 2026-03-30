"""
steps/step2_normalize.py — Normalise activation values to integer 0–10.

For each feature, clamps negatives to 0, scales the max to 10, and
discretises to integers — matching the Bills et al. (2023) protocol.

Also adds a 'highlighted_sequence' column: the sequence with the activating
nucleotide wrapped in square brackets [X], ready for prompt formatting.

Input:  output/sequences.csv
Output: output/normalized.csv

Run:
    python steps/step2_normalize.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SEQUENCES_CSV, NORMALIZED_CSV
from utils.genome import highlight_position, truncate_for_display


def normalise_per_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'activation_norm' column: per-feature min-max normalisation to 0–10.

    Steps:
      1. Clamp activation_value to [0, inf)
      2. Scale so the feature's max activation → 10
      3. Round to nearest integer
    """
    def _norm_group(group):
        vals = group["activation_value"].clip(lower=0.0)
        max_val = vals.max()
        if max_val > 0:
            group = group.copy()
            group["activation_norm"] = (vals / max_val * 10).round().astype(int)
        else:
            group = group.copy()
            group["activation_norm"] = 0
        return group

    tqdm.pandas(desc="Normalising")
    df = df.groupby("feature_idx", group_keys=False).progress_apply(_norm_group)
    return df


def add_highlighted_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'highlighted_sequence' column: sequence with [X] at the activating
    nucleotide position, then truncated to 200 chars for display in prompts.

    The 'tok_pos' column is the character offset WITHIN the fetched window
    (the sequence starts at coord_start, so tok_pos directly indexes into
    the sequence string).
    """
    highlighted = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Highlighting"):
        seq = str(row.get("sequence", ""))
        if not seq:
            highlighted.append("")
            continue

        tok_pos = int(row["tok_pos"])
        # tok_pos is the position within the model's full sequence (seq_idx).
        # The fetched window corresponds to coord_start:coord_end within the
        # chromosome; tok_pos is already an offset into that window.
        # We clamp to avoid out-of-bounds.
        tok_pos = min(max(tok_pos, 0), len(seq) - 1)

        seq_hl = highlight_position(
            sequence=seq,
            tok_pos=tok_pos,
            seq_start=int(row["coord_start"]),
            window_start=int(row["coord_start"]),
        )
        seq_trunc = truncate_for_display(seq_hl, max_chars=200)
        highlighted.append(seq_trunc)

    df = df.copy()
    df["highlighted_sequence"] = highlighted
    return df


def run() -> pd.DataFrame:
    print("=" * 60)
    print("STEP 2: Normalising activations and highlighting sequences")
    print("=" * 60)

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"\nLoading {SEQUENCES_CSV} ...")
    df = pd.read_csv(SEQUENCES_CSV, dtype={"coord_chrom": str})
    print(f"  {len(df)} rows | {df['feature_idx'].nunique()} features")

    # ── Normalise ───────────────────────────────────────────────────────────
    print("\nNormalising activations (per-feature, 0–10) ...")
    df = normalise_per_feature(df)

    # ── Highlight ───────────────────────────────────────────────────────────
    print("\nAdding highlighted sequences ...")
    df = add_highlighted_sequences(df)

    # ── Save ────────────────────────────────────────────────────────────────
    df.to_csv(NORMALIZED_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {NORMALIZED_CSV}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\nSample normalised activations (feature 0, top 5):")
    sample = df[df["feature_idx"] == 0].nlargest(5, "activation_norm")[
        ["feature_idx", "rank", "activation_value", "activation_norm",
         "coord_chrom", "coord_start", "highlighted_sequence"]
    ]
    print(sample.to_string(index=False))

    # Per-feature stats
    stats = df.groupby("feature_idx")["activation_norm"].agg(["max", "mean", "std"])
    print(f"\nActivation norm stats across {len(stats)} features:")
    print(f"  Mean of per-feature max: {stats['max'].mean():.2f}")
    print(f"  All per-feature maxes == 10: {(stats['max'] == 10).all()}")

    return df


if __name__ == "__main__":
    run()
