"""
steps/step5_score.py — Score explanations via LLM simulation.

For each feature with a valid explanation, the simulator LLM is given:
  - The generated hypothesis
  - A held-out set of 30 sequence examples (ranks 100–130)

The simulator predicts an activation score (0–10) for each example.
The Pearson r between predicted and real activations is the quality score.

Results are saved to output/scores.csv.

Input:  output/explanations.jsonl
Output: output/scores.csv

Run:
    python steps/step5_score.py
"""

import sys
import re
import json
import time
import traceback
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EXPLANATIONS_JSONL,
    SCORES_CSV,
    SIMULATOR_MODEL,
    MAX_TOKENS_SIMULATE,
    API_CALL_DELAY,
    ANTHROPIC_API_KEY,
    SCORE_ACCEPT_THRESHOLD,
    SCORE_MIN_THRESHOLD,
)
from utils.io import load_jsonl

import anthropic
from scipy.stats import pearsonr
from tqdm import tqdm


# ── Prompt formatting ──────────────────────────────────────────────────────────

from prompts.templates import SIMULATOR_SYSTEM, SIMULATOR_USER


def format_sequences_block(sequences: list[str]) -> str:
    """Format held-out sequences for the simulator prompt."""
    lines = []
    for i, seq in enumerate(sequences, start=1):
        lines.append(f"Seq {i:02d}: {seq}")
    return "\n".join(lines)


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_simulator_response(raw_text: str, n_expected: int) -> list[int] | None:
    """
    Parse the simulator's JSON array response.

    Tries strict JSON parsing first, then falls back to regex extraction.
    Returns a list of integers or None on failure.
    """
    raw_text = raw_text.strip()

    # Remove markdown fences if present
    raw_text = re.sub(r"```(?:json)?", "", raw_text).strip()

    # Try to find a JSON array anywhere in the response
    array_match = re.search(r"\[[\d,\s]+\]", raw_text)
    if array_match:
        try:
            values = json.loads(array_match.group())
            if len(values) == n_expected:
                return [max(0, min(10, int(v))) for v in values]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract all integers in order
    ints = re.findall(r"\b(\d|10)\b", raw_text)
    if len(ints) == n_expected:
        return [max(0, min(10, int(x))) for x in ints]

    return None


# ── Scoring ────────────────────────────────────────────────────────────────────

def compute_pearson(predicted: list[int], real: list[int]) -> float:
    """Compute Pearson r between predicted and real activations."""
    if len(predicted) < 3:
        return 0.0
    if np.std(predicted) == 0 or np.std(real) == 0:
        # Constant predictions → zero correlation
        return 0.0
    r, _ = pearsonr(predicted, real)
    return float(r)


# ── API call ───────────────────────────────────────────────────────────────────

def call_simulator(
    client: anthropic.Anthropic,
    hypothesis: str,
    sequences: list[str],
    feature_idx: int,
    max_retries: int = 3,
) -> dict:
    """
    Call the simulator LLM to predict activations.

    Returns a dict with predicted_activations (list[int] or None) and metadata.
    """
    n_examples = len(sequences)
    sequences_block = format_sequences_block(sequences)

    system_prompt = SIMULATOR_SYSTEM.format(n_examples=n_examples)
    user_prompt   = SIMULATOR_USER.format(
        hypothesis=hypothesis,
        n_examples=n_examples,
        sequences_block=sequences_block,
    )

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=SIMULATOR_MODEL,
                max_tokens=MAX_TOKENS_SIMULATE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw_text  = response.content[0].text
            predicted = parse_simulator_response(raw_text, n_expected=n_examples)

            return {
                "feature_idx":         feature_idx,
                "predicted_activations": predicted,
                "raw_response":        raw_text,
                "input_tokens":        response.usage.input_tokens,
                "output_tokens":       response.usage.output_tokens,
                "parse_ok":            predicted is not None,
                "api_error":           None,
                "attempt":             attempt,
            }

        except anthropic.RateLimitError as e:
            wait = 60 * attempt
            print(f"\n  [Feature {feature_idx}] Rate limit. Waiting {wait}s ...")
            time.sleep(wait)
            last_error = str(e)

        except anthropic.APIError as e:
            print(f"\n  [Feature {feature_idx}] API error attempt {attempt}: {e}")
            time.sleep(5 * attempt)
            last_error = str(e)

        except Exception as e:
            print(f"\n  [Feature {feature_idx}] Unexpected error: {e}")
            traceback.print_exc()
            last_error = str(e)
            break

    return {
        "feature_idx":           feature_idx,
        "predicted_activations": None,
        "raw_response":          "",
        "input_tokens":          0,
        "output_tokens":         0,
        "parse_ok":              False,
        "api_error":             last_error,
        "attempt":               max_retries,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    print("=" * 60)
    print("STEP 5: Scoring explanations via LLM simulation")
    print("=" * 60)

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")

    # ── Load explanations ─────────────────────────────────────────────────────
    print(f"\nLoading {EXPLANATIONS_JSONL} ...")
    explanations = load_jsonl(EXPLANATIONS_JSONL)
    valid = [e for e in explanations if e.get("parse_ok") and e.get("hypothesis")]
    print(f"  {len(explanations)} total | {len(valid)} with valid hypotheses")

    # ── Resume support ────────────────────────────────────────────────────────
    if SCORES_CSV.exists():
        existing_scores = pd.read_csv(SCORES_CSV)
        done_ids = set(existing_scores["feature_idx"].tolist())
    else:
        existing_scores = pd.DataFrame()
        done_ids = set()

    todo = [e for e in valid if e["feature_idx"] not in done_ids]
    print(f"  Already scored: {len(done_ids)} | Remaining: {len(todo)}")

    if not todo:
        print("  All features already scored.")
        return pd.read_csv(SCORES_CSV)

    # ── API client ─────────────────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Score loop ─────────────────────────────────────────────────────────────
    score_rows = []

    for expl in tqdm(todo, desc="Simulating"):
        feat_idx   = expl["feature_idx"]
        hypothesis = expl["hypothesis"]
        sequences  = expl.get("holdout_sequences", [])
        real_acts  = expl.get("holdout_activations", [])

        if not sequences or not real_acts:
            print(f"  [Feature {feat_idx}] No holdout data — skipping simulation.")
            score_rows.append({
                "feature_idx":           feat_idx,
                "pearson_r":             None,
                "n_holdout":             0,
                "predicted_activations": None,
                "real_activations":      None,
                "parse_ok":              False,
                "api_error":             "no holdout data",
                "score_category":        "unscored",
                "hypothesis":            hypothesis,
                "confidence":            expl.get("confidence", ""),
            })
            continue

        sim_result = call_simulator(
            client=client,
            hypothesis=hypothesis,
            sequences=sequences,
            feature_idx=feat_idx,
        )

        predicted = sim_result.get("predicted_activations")
        if predicted is not None and len(predicted) == len(real_acts):
            r = compute_pearson(predicted, real_acts)
        else:
            r = None

        # Categorise the score
        if r is None:
            category = "unscored"
        elif r >= SCORE_ACCEPT_THRESHOLD:
            category = "accept"
        elif r >= SCORE_MIN_THRESHOLD:
            category = "review"
        else:
            category = "recluster"

        score_rows.append({
            "feature_idx":           feat_idx,
            "pearson_r":             r,
            "n_holdout":             len(real_acts),
            "predicted_activations": json.dumps(predicted) if predicted else None,
            "real_activations":      json.dumps(real_acts),
            "parse_ok":              sim_result["parse_ok"],
            "api_error":             sim_result["api_error"],
            "score_category":        category,
            "hypothesis":            hypothesis,
            "confidence":            expl.get("confidence", ""),
            "evidence":              expl.get("evidence", ""),
        })

        time.sleep(API_CALL_DELAY)

    # ── Save ───────────────────────────────────────────────────────────────────
    new_df = pd.DataFrame(score_rows)
    if not existing_scores.empty:
        result_df = pd.concat([existing_scores, new_df], ignore_index=True)
    else:
        result_df = new_df

    result_df.to_csv(SCORES_CSV, index=False)
    print(f"\nSaved {len(result_df)} scored features → {SCORES_CSV}")

    # ── Summary ────────────────────────────────────────────────────────────────
    if not result_df.empty and "pearson_r" in result_df.columns:
        scored = result_df.dropna(subset=["pearson_r"])
        print(f"\nScore distribution (n={len(scored)}):")
        print(f"  Mean Pearson r:  {scored['pearson_r'].mean():.3f}")
        print(f"  Median Pearson r:{scored['pearson_r'].median():.3f}")

        if "score_category" in result_df.columns:
            cats = result_df["score_category"].value_counts()
            print("\nCategories:")
            for cat, count in cats.items():
                print(f"  {cat:<12}: {count}")

        # Show top and bottom examples
        top3 = scored.nlargest(3, "pearson_r")[["feature_idx","pearson_r","hypothesis"]]
        print(f"\nTop 3 features by Pearson r:")
        for _, row in top3.iterrows():
            hyp_short = row["hypothesis"][:80] + "..." if len(row["hypothesis"]) > 80 else row["hypothesis"]
            print(f"  [{row['feature_idx']}] r={row['pearson_r']:.3f} — {hyp_short}")

    return result_df


if __name__ == "__main__":
    run()
