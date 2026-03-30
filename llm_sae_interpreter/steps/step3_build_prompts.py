"""
steps/step3_build_prompts.py — Build explainer prompts for each feature.

For each feature, assembles:
  - An annotation enrichment block from enrichment.csv
  - A formatted block of the top N highest-activating examples
  - The final user prompt string

Saves one JSONL record per feature to output/prompts.jsonl.

Input:  output/normalized.csv, enrichment.csv
Output: output/prompts.jsonl

Run:
    python steps/step3_build_prompts.py
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    NORMALIZED_CSV,
    ENRICHMENT_CSV,
    PROMPTS_JSONL,
    N_EXPLAINER_EXAMPLES,
    SIMULATION_HOLDOUT_START,
    SIMULATION_HOLDOUT_END,
)
from utils.io import load_enrichment, save_jsonl
from utils.enrichment import format_enrichment_block
from prompts.templates import EXPLAINER_SYSTEM, EXPLAINER_USER


def build_examples_block(
    feature_df: pd.DataFrame,
    n: int = N_EXPLAINER_EXAMPLES,
) -> str:
    """
    Format the top N activating examples into a prompt block.

    Each example shows:
      - Rank and normalised activation score
      - Chromosome coordinates
      - Highlighted sequence (activating nucleotide in [brackets])
    """
    top_n = feature_df.nlargest(n, "activation_norm")
    lines = []

    for i, (_, row) in enumerate(top_n.iterrows(), start=1):
        coord = f"{row['coord_chrom']}:{int(row['coord_start'])}-{int(row['coord_end'])}"
        seq   = str(row.get("highlighted_sequence", ""))
        norm  = int(row["activation_norm"])
        raw   = float(row["activation_value"])

        lines.append(
            f"Example {i:02d} | activation: {norm}/10 (raw: {raw:.4f}) | {coord}"
        )
        if seq:
            lines.append(f"  {seq}")
        else:
            lines.append("  [sequence unavailable]")
        lines.append("")  # blank line between examples

    return "\n".join(lines).rstrip()


def build_holdout_block(
    feature_df: pd.DataFrame,
    start: int = SIMULATION_HOLDOUT_START,
    end: int = SIMULATION_HOLDOUT_END,
) -> tuple[list[str], list[int]]:
    """
    Build the held-out examples block used in Step 5 simulation scoring.

    Returns
    -------
    sequences : list of highlighted_sequence strings (for the simulator prompt)
    real_acts : list of integer activation_norm values (ground truth)
    """
    holdout = feature_df.iloc[start:end]
    sequences = []
    real_acts = []

    for _, row in holdout.iterrows():
        seq = str(row.get("highlighted_sequence", ""))
        sequences.append(seq if seq else "[unavailable]")
        real_acts.append(int(row["activation_norm"]))

    return sequences, real_acts


def build_prompt_record(
    feature_idx: int,
    feature_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
) -> dict:
    """
    Build the complete prompt record for one feature.

    Returns a dict that is saved as one line in prompts.jsonl.
    """
    # ── Annotation enrichment ────────────────────────────────────────────────
    if feature_idx in enrichment_df.index:
        enr_row = enrichment_df.loc[feature_idx]
        enrichment_block = format_enrichment_block(enr_row)
    else:
        enrichment_block = "  (no enrichment data available for this feature)"

    # ── Top examples ─────────────────────────────────────────────────────────
    # Sort by rank so we always take the true top-N
    feature_df_sorted = feature_df.sort_values("rank")
    examples_block = build_examples_block(feature_df_sorted, n=N_EXPLAINER_EXAMPLES)

    # ── Held-out block for simulation (step 5) ────────────────────────────────
    holdout_seqs, holdout_acts = build_holdout_block(feature_df_sorted)

    # ── Assemble user prompt ─────────────────────────────────────────────────
    user_prompt = EXPLAINER_USER.format(
        feature_idx=feature_idx,
        enrichment_block=enrichment_block,
        n_examples=N_EXPLAINER_EXAMPLES,
        examples_block=examples_block,
    )

    return {
        "feature_idx":       feature_idx,
        "system_prompt":     EXPLAINER_SYSTEM,
        "user_prompt":       user_prompt,
        "enrichment_block":  enrichment_block,  # stored for consistency check in step 7
        "holdout_sequences": holdout_seqs,
        "holdout_activations": holdout_acts,
        "n_examples_used":   min(N_EXPLAINER_EXAMPLES, len(feature_df)),
        "n_holdout":         len(holdout_seqs),
    }


def run() -> list[dict]:
    print("=" * 60)
    print("STEP 3: Building explainer prompts")
    print("=" * 60)

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"\nLoading {NORMALIZED_CSV} ...")
    df = pd.read_csv(NORMALIZED_CSV, dtype={"coord_chrom": str})
    print(f"  {len(df)} rows | {df['feature_idx'].nunique()} features")

    print(f"Loading {ENRICHMENT_CSV} ...")
    enrichment_df = load_enrichment(ENRICHMENT_CSV)
    print(f"  {len(enrichment_df)} features in enrichment data")

    # ── Build prompts ────────────────────────────────────────────────────────
    feature_ids = sorted(df["feature_idx"].unique())
    records = []

    for feat_idx in tqdm(feature_ids, desc="Building prompts"):
        feat_df  = df[df["feature_idx"] == feat_idx]
        record   = build_prompt_record(feat_idx, feat_df, enrichment_df)
        records.append(record)

    # ── Save ────────────────────────────────────────────────────────────────
    save_jsonl(records, PROMPTS_JSONL)

    # ── Preview ─────────────────────────────────────────────────────────────
    if records:
        print("\n--- SAMPLE PROMPT (feature 0) ---")
        r = records[0]
        print(f"System prompt (first 300 chars):\n{r['system_prompt'][:300]}...\n")
        print(f"User prompt (first 800 chars):\n{r['user_prompt'][:800]}...\n")
        print(f"Holdout examples: {r['n_holdout']}")
        print(f"Ground truth activations: {r['holdout_activations'][:10]}...")

    return records


if __name__ == "__main__":
    run()
