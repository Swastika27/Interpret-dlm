"""
steps/step6_recluster.py — Revise low-scoring feature explanations.

For features with Pearson r < SCORE_ACCEPT_THRESHOLD, this step:
  1. Loads the top-200 activating examples for the feature
  2. Clusters them by annotation profile (K-Means on pct_* columns)
  3. Sends each sub-cluster to the LLM with an expanded context window
  4. Generates separate hypotheses per sub-cluster
  5. Runs simulation scoring on each sub-cluster hypothesis
  6. Saves the best-scoring hypothesis per feature to output/reclustered.jsonl

Input:  output/scores.csv, output/normalized.csv, enrichment.csv
Output: output/reclustered.jsonl

Run:
    python steps/step6_recluster.py
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
    SCORES_CSV,
    NORMALIZED_CSV,
    ENRICHMENT_CSV,
    RECLUSTERED_JSONL,
    GENOME_FASTA,
    EXPLAINER_MODEL,
    SIMULATOR_MODEL,
    MAX_TOKENS_EXPLAIN,
    MAX_TOKENS_SIMULATE,
    API_CALL_DELAY,
    ANTHROPIC_API_KEY,
    SCORE_ACCEPT_THRESHOLD,
    N_SUBCLUSTERS,
    FLANK_BP,
    N_EXPLAINER_EXAMPLES,
)
from utils.io import load_enrichment, load_jsonl, append_jsonl, load_existing_feature_ids
from utils.enrichment import format_enrichment_block, compute_feature_annotation_vector
from utils.genome import fetch_sequence, highlight_position, truncate_for_display
from prompts.templates import RECLUSTER_SYSTEM, RECLUSTER_USER, SIMULATOR_SYSTEM, SIMULATOR_USER

import anthropic
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ── Sub-clustering ─────────────────────────────────────────────────────────────

def cluster_feature_examples(
    feature_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    n_clusters: int = N_SUBCLUSTERS,
) -> list[dict]:
    """
    Cluster the top-200 examples of a feature by their per-example annotation
    profile. Since enrichment data is at feature level (not per-example), we
    cluster using the available per-row fields:
      - activation_norm
      - chromosome (encoded as int)
      - coord_start (scaled)

    Returns a list of sub-cluster dicts, each with:
      - cluster_id: int
      - rows: list of row dicts
      - annotation_summary: str (from the sub-cluster's mean annotation)
    """
    df = feature_df.sort_values("rank").copy()

    # Build feature matrix: activation + chromosomal position features
    chrom_map = {c: i for i, c in enumerate(df["coord_chrom"].unique())}
    X = np.column_stack([
        df["activation_norm"].fillna(0).values,
        df["coord_chrom"].map(chrom_map).fillna(0).values,
        df["coord_start"].values / 1e8,  # scale to ~[0,1]
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(n_clusters, len(df))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    df["cluster_label"] = labels

    subclusters = []
    for cid in range(n_clusters):
        cluster_rows = df[df["cluster_label"] == cid]
        if cluster_rows.empty:
            continue
        rows = cluster_rows.to_dict("records")

        # Summarise cluster
        top_chroms = cluster_rows["coord_chrom"].value_counts().head(3)
        chrom_str  = ", ".join(f"{c} ({n})" for c, n in top_chroms.items())
        mean_act   = cluster_rows["activation_norm"].mean()

        annotation_summary = (
            f"  Chromosomes: {chrom_str}\n"
            f"  Mean activation: {mean_act:.1f}/10\n"
            f"  n examples: {len(rows)}"
        )

        subclusters.append({
            "cluster_id":          cid + 1,
            "rows":                rows,
            "annotation_summary":  annotation_summary,
            "n":                   len(rows),
        })

    return subclusters


# ── Prompt building ────────────────────────────────────────────────────────────

def build_subcluster_examples_block(
    subcluster_rows: list[dict],
    n: int = 10,
) -> str:
    """Format examples for a single sub-cluster."""
    rows = sorted(subcluster_rows, key=lambda r: r.get("activation_norm", 0), reverse=True)
    lines = []
    for i, row in enumerate(rows[:n], start=1):
        seq  = str(row.get("highlighted_sequence", ""))
        norm = row.get("activation_norm", "?")
        coord= f"{row['coord_chrom']}:{int(row['coord_start'])}-{int(row['coord_end'])}"
        lines.append(f"  [{i}] activation {norm}/10 | {coord}")
        if seq:
            lines.append(f"      {seq}")
    return "\n".join(lines)


def build_recluster_prompt(
    feature_idx: int,
    original_hypothesis: str,
    score: float,
    subclusters: list[dict],
    enrichment_row,
) -> tuple[str, str]:
    """Build the system and user prompt for the reclustering call."""
    system = RECLUSTER_SYSTEM.format(n_clusters=len(subclusters))

    # Build per-subcluster annotation and example blocks
    enrich_block_parts = []
    examples_block_parts = []

    for sc in subclusters:
        enrich_block_parts.append(
            f"Sub-cluster {sc['cluster_id']} ({sc['n']} examples):\n"
            f"{sc['annotation_summary']}"
        )
        examples_block = build_subcluster_examples_block(sc["rows"])
        examples_block_parts.append(
            f"Sub-cluster {sc['cluster_id']}:\n{examples_block}"
        )

    user = RECLUSTER_USER.format(
        feature_idx=feature_idx,
        score=score,
        original_hypothesis=original_hypothesis,
        subcluster_enrichment_block="\n\n".join(enrich_block_parts),
        subcluster_examples_block="\n\n".join(examples_block_parts),
    )

    return system, user


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_recluster_response(raw_text: str, n_clusters: int) -> list[dict]:
    """
    Parse the multi-subcluster response into a list of hypothesis dicts.

    Expected format:
        SUBCLUSTER 1:
        HYPOTHESIS: ...
        EVIDENCE: ...
        CONFIDENCE: HIGH|MEDIUM|LOW

        SUBCLUSTER 2:
        ...
    """
    results = []

    for cid in range(1, n_clusters + 1):
        # Extract block for this sub-cluster
        pattern = rf"SUBCLUSTER\s+{cid}:\s*\nHYPOTHESIS:\s*(.+?)(?=\nEVIDENCE:|$)"
        m_hyp   = re.search(pattern, raw_text, re.S | re.I)

        pattern_ev = rf"SUBCLUSTER\s+{cid}.*?EVIDENCE:\s*(.+?)(?=\nCONFIDENCE:|$)"
        m_ev       = re.search(pattern_ev, raw_text, re.S | re.I)

        pattern_cf = rf"SUBCLUSTER\s+{cid}.*?CONFIDENCE:\s*(HIGH|MEDIUM|LOW)"
        m_cf       = re.search(pattern_cf, raw_text, re.S | re.I)

        results.append({
            "cluster_id": cid,
            "hypothesis": m_hyp.group(1).strip() if m_hyp else "",
            "evidence":   m_ev.group(1).strip()  if m_ev  else "",
            "confidence": m_cf.group(1).upper()  if m_cf  else "",
            "parse_ok":   bool(m_hyp),
        })

    return results


# ── Simulation scoring for sub-clusters ───────────────────────────────────────

def score_subcluster_hypothesis(
    client: anthropic.Anthropic,
    hypothesis: str,
    subcluster_rows: list[dict],
    feature_idx: int,
    cluster_id: int,
) -> float:
    """Score a single sub-cluster hypothesis via simulation."""
    # Use rows not in the top 10 (top 10 were shown in the prompt)
    holdout_rows = sorted(
        subcluster_rows, key=lambda r: r.get("activation_norm", 0), reverse=True
    )[10:25]  # rows 11-25 as holdout

    if len(holdout_rows) < 3:
        return 0.0

    sequences = [str(r.get("highlighted_sequence", "")) for r in holdout_rows]
    real_acts = [int(r.get("activation_norm", 0)) for r in holdout_rows]
    n = len(sequences)

    system = SIMULATOR_SYSTEM.format(n_examples=n)
    user   = SIMULATOR_USER.format(
        hypothesis=hypothesis,
        n_examples=n,
        sequences_block="\n".join(f"Seq {i+1:02d}: {s}" for i, s in enumerate(sequences)),
    )

    try:
        response = client.messages.create(
            model=SIMULATOR_MODEL,
            max_tokens=MAX_TOKENS_SIMULATE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = response.content[0].text

        # Parse JSON array
        array_match = re.search(r"\[[\d,\s]+\]", raw)
        if array_match:
            predicted = json.loads(array_match.group())
            if len(predicted) == n:
                predicted = [max(0, min(10, int(v))) for v in predicted]
                if np.std(predicted) == 0 or np.std(real_acts) == 0:
                    return 0.0
                r, _ = pearsonr(predicted, real_acts)
                return float(r)
    except Exception as e:
        print(f"  [Feature {feature_idx}, cluster {cluster_id}] Sim error: {e}")

    return 0.0


# ── API call ───────────────────────────────────────────────────────────────────

def call_reclustering_llm(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_prompt: str,
    feature_idx: int,
    n_clusters: int,
    max_retries: int = 3,
) -> dict:
    """Call the reclustering LLM with retry logic."""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=EXPLAINER_MODEL,
                max_tokens=MAX_TOKENS_EXPLAIN * n_clusters,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = response.content[0].text
            return {
                "raw_response":   raw_text,
                "input_tokens":   response.usage.input_tokens,
                "output_tokens":  response.usage.output_tokens,
                "api_error":      None,
            }
        except anthropic.RateLimitError as e:
            wait = 60 * attempt
            print(f"\n  [Feature {feature_idx}] Rate limit. Waiting {wait}s ...")
            time.sleep(wait)
            last_error = str(e)
        except Exception as e:
            print(f"\n  [Feature {feature_idx}] Error: {e}")
            last_error = str(e)
            break

    return {"raw_response": "", "input_tokens": 0, "output_tokens": 0, "api_error": last_error}


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> list[dict]:
    print("=" * 60)
    print("STEP 6: Reclustering low-scoring features")
    print("=" * 60)

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")

    # ── Load scores ───────────────────────────────────────────────────────────
    print(f"\nLoading {SCORES_CSV} ...")
    scores_df = pd.read_csv(SCORES_CSV)
    to_recluster = scores_df[
        scores_df["score_category"].isin(["recluster", "review"])
    ]
    print(f"  Features to recluster: {len(to_recluster)}")

    # ── Resume support ────────────────────────────────────────────────────────
    done_ids = load_existing_feature_ids(RECLUSTERED_JSONL)
    todo_df  = to_recluster[~to_recluster["feature_idx"].isin(done_ids)]
    print(f"  Already done: {len(done_ids)} | Remaining: {len(todo_df)}")

    if todo_df.empty:
        print("  All low-scoring features already reclustered.")
        return load_jsonl(RECLUSTERED_JSONL)

    # ── Load sequence data ────────────────────────────────────────────────────
    print(f"\nLoading {NORMALIZED_CSV} ...")
    norm_df = pd.read_csv(NORMALIZED_CSV, dtype={"coord_chrom": str})

    print(f"Loading {ENRICHMENT_CSV} ...")
    enrichment_df = load_enrichment(ENRICHMENT_CSV)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    results = []

    for _, score_row in tqdm(todo_df.iterrows(), total=len(todo_df), desc="Reclustering"):
        feat_idx = int(score_row["feature_idx"])
        orig_hyp = str(score_row.get("hypothesis", ""))
        orig_r   = float(score_row.get("pearson_r", 0.0) or 0.0)

        # ── Get feature examples ───────────────────────────────────────────────
        feat_df = norm_df[norm_df["feature_idx"] == feat_idx].sort_values("rank")
        if feat_df.empty:
            print(f"  [Feature {feat_idx}] No data found. Skipping.")
            continue

        # ── Cluster examples ───────────────────────────────────────────────────
        enr_row = enrichment_df.loc[feat_idx] if feat_idx in enrichment_df.index else None
        subclusters = cluster_feature_examples(feat_df, enrichment_df, n_clusters=N_SUBCLUSTERS)

        # ── Build and send prompt ─────────────────────────────────────────────
        system, user = build_recluster_prompt(
            feature_idx=feat_idx,
            original_hypothesis=orig_hyp,
            score=orig_r,
            subclusters=subclusters,
            enrichment_row=enr_row,
        )

        llm_result = call_reclustering_llm(
            client=client,
            system_prompt=system,
            user_prompt=user,
            feature_idx=feat_idx,
            n_clusters=len(subclusters),
        )

        if llm_result["api_error"]:
            print(f"  [Feature {feat_idx}] API error: {llm_result['api_error']}")
            continue

        # ── Parse sub-cluster hypotheses ──────────────────────────────────────
        sub_hyps = parse_recluster_response(
            llm_result["raw_response"], n_clusters=len(subclusters)
        )

        # ── Score each sub-cluster hypothesis ─────────────────────────────────
        best_r = orig_r
        best_hypothesis = orig_hyp
        best_cluster_id = None
        scored_subs = []

        for sc, hyp in zip(subclusters, sub_hyps):
            if not hyp["parse_ok"] or not hyp["hypothesis"]:
                scored_subs.append({**hyp, "pearson_r": 0.0})
                continue

            r = score_subcluster_hypothesis(
                client=client,
                hypothesis=hyp["hypothesis"],
                subcluster_rows=sc["rows"],
                feature_idx=feat_idx,
                cluster_id=sc["cluster_id"],
            )
            scored_subs.append({**hyp, "pearson_r": r})

            if r > best_r:
                best_r = r
                best_hypothesis = hyp["hypothesis"]
                best_cluster_id = sc["cluster_id"]

            time.sleep(API_CALL_DELAY)

        record = {
            "feature_idx":       feat_idx,
            "original_score":    orig_r,
            "original_hypothesis": orig_hyp,
            "best_score":        best_r,
            "best_hypothesis":   best_hypothesis,
            "best_cluster_id":   best_cluster_id,
            "n_subclusters":     len(subclusters),
            "subcluster_results": scored_subs,
            "input_tokens":      llm_result["input_tokens"],
            "output_tokens":     llm_result["output_tokens"],
            "improved":          best_r > orig_r,
        }

        append_jsonl(record, RECLUSTERED_JSONL)
        results.append(record)

        time.sleep(API_CALL_DELAY)

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        improved = sum(1 for r in results if r["improved"])
        orig_rs  = [r["original_score"] for r in results]
        best_rs  = [r["best_score"]     for r in results]
        print(f"\nReclustering results ({len(results)} features):")
        print(f"  Improved: {improved} / {len(results)}")
        print(f"  Mean original Pearson r: {np.mean(orig_rs):.3f}")
        print(f"  Mean best Pearson r:     {np.mean(best_rs):.3f}")

    return load_jsonl(RECLUSTERED_JSONL)


if __name__ == "__main__":
    run()
