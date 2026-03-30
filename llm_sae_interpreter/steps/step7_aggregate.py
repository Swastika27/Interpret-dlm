"""
steps/step7_aggregate.py — Build the final feature atlas.

This step:
  1. Merges all explanation sources (step 4 + step 6 reclustering)
  2. Runs a consistency check: does the hypothesis match the enrichment?
  3. Embeds all hypotheses (via LLM) and clusters by semantic similarity
  4. Labels each cluster with a 3–7 word biological theme
  5. Saves the final feature atlas to output/feature_atlas.csv

Input:  output/scores.csv, output/reclustered.jsonl, enrichment.csv
Output: output/feature_atlas.csv

Run:
    python steps/step7_aggregate.py
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
    RECLUSTERED_JSONL,
    ENRICHMENT_CSV,
    FEATURE_ATLAS_CSV,
    CLUSTER_LABEL_MODEL,
    EXPLAINER_MODEL,
    MAX_TOKENS_CLUSTER,
    API_CALL_DELAY,
    ANTHROPIC_API_KEY,
    SCORE_ACCEPT_THRESHOLD,
    SCORE_MIN_THRESHOLD,
)
from utils.io import load_enrichment, load_jsonl
from utils.enrichment import format_enrichment_block, compute_feature_annotation_vector
from prompts.templates import (
    CLUSTER_LABEL_SYSTEM, CLUSTER_LABEL_USER,
    CONSISTENCY_CHECK_SYSTEM, CONSISTENCY_CHECK_USER,
)

import anthropic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ── Step 7a: Merge all explanations ───────────────────────────────────────────

def merge_explanations(
    scores_df: pd.DataFrame,
    reclustered: list[dict],
) -> pd.DataFrame:
    """
    Build a unified DataFrame of best hypotheses.

    For features with reclustering results, use the best sub-cluster
    hypothesis if it scored higher than the original.
    """
    # Build lookup from reclustered results
    recluster_lookup = {r["feature_idx"]: r for r in reclustered}

    rows = []
    for _, score_row in scores_df.iterrows():
        feat_idx = int(score_row["feature_idx"])
        r_orig   = score_row.get("pearson_r")
        hyp_orig = str(score_row.get("hypothesis", ""))
        conf     = str(score_row.get("confidence", ""))
        evid     = str(score_row.get("evidence", ""))

        if feat_idx in recluster_lookup:
            rec = recluster_lookup[feat_idx]
            if rec["improved"] and rec["best_score"] > (r_orig or 0.0):
                rows.append({
                    "feature_idx":      feat_idx,
                    "hypothesis":       rec["best_hypothesis"],
                    "evidence":         evid,  # keep original evidence
                    "confidence":       conf,
                    "pearson_r":        rec["best_score"],
                    "original_pearson_r": r_orig,
                    "score_category":   score_row.get("score_category", ""),
                    "source":           "reclustered",
                    "cluster_id_used":  rec["best_cluster_id"],
                })
                continue

        rows.append({
            "feature_idx":      feat_idx,
            "hypothesis":       hyp_orig,
            "evidence":         evid,
            "confidence":       conf,
            "pearson_r":        r_orig,
            "original_pearson_r": r_orig,
            "score_category":   score_row.get("score_category", ""),
            "source":           "original",
            "cluster_id_used":  None,
        })

    return pd.DataFrame(rows)


# ── Step 7b: Consistency check ─────────────────────────────────────────────────

def run_consistency_check(
    client: anthropic.Anthropic,
    feature_idx: int,
    hypothesis: str,
    enrichment_block: str,
) -> dict:
    """
    Ask the LLM whether the hypothesis is consistent with the annotation stats.

    Returns a dict with: consistent (YES/NO/PARTIAL), issues, revised_hypothesis.
    """
    system = CONSISTENCY_CHECK_SYSTEM
    user   = CONSISTENCY_CHECK_USER.format(
        feature_idx=feature_idx,
        hypothesis=hypothesis,
        enrichment_block=enrichment_block,
    )

    try:
        response = client.messages.create(
            model=EXPLAINER_MODEL,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = response.content[0].text

        m_cons = re.search(r"CONSISTENT:\s*(YES|NO|PARTIAL)", raw, re.I)
        m_issue= re.search(r"ISSUES:\s*(.+?)(?=\nREVISED_HYPOTHESIS:|$)", raw, re.S)
        m_rev  = re.search(r"REVISED_HYPOTHESIS:\s*(.+)", raw, re.S)

        return {
            "consistent":         m_cons.group(1).upper() if m_cons else "UNKNOWN",
            "issues":             m_issue.group(1).strip() if m_issue else "",
            "revised_hypothesis": m_rev.group(1).strip()  if m_rev  else hypothesis,
            "api_error":          None,
        }
    except Exception as e:
        return {
            "consistent":         "UNKNOWN",
            "issues":             str(e),
            "revised_hypothesis": hypothesis,
            "api_error":          str(e),
        }


# ── Step 7c: Cluster by annotation profile ─────────────────────────────────────

def cluster_by_annotation(
    merged_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    n_clusters: int = 10,
) -> pd.DataFrame:
    """
    Cluster features by their annotation enrichment vectors.

    Uses K-Means on the pct_* columns from enrichment.csv.
    Adds a 'semantic_cluster' column to merged_df.
    """
    feature_ids = merged_df["feature_idx"].tolist()
    vectors = []

    for fid in feature_ids:
        if fid in enrichment_df.index:
            v = compute_feature_annotation_vector(enrichment_df.loc[fid])
        else:
            v = [0.0] * 20  # fallback zero vector
        vectors.append(v)

    X = np.array(vectors)
    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(n_clusters, len(feature_ids))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    merged_df = merged_df.copy()
    merged_df["semantic_cluster"] = labels
    return merged_df


# ── Step 7d: Label each cluster ────────────────────────────────────────────────

def label_cluster(
    client: anthropic.Anthropic,
    cluster_id: int,
    hypotheses: list[str],
) -> dict:
    """Ask the LLM to generate a 3–7 word label for a cluster."""
    hyp_block = "\n".join(
        f"  [{i+1}] {h}" for i, h in enumerate(hypotheses[:15])  # cap at 15 per call
    )
    system = CLUSTER_LABEL_SYSTEM
    user   = CLUSTER_LABEL_USER.format(
        cluster_id=cluster_id,
        n_features=len(hypotheses),
        hypotheses_block=hyp_block,
    )

    try:
        response = client.messages.create(
            model=CLUSTER_LABEL_MODEL,
            max_tokens=MAX_TOKENS_CLUSTER,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = response.content[0].text

        m_label = re.search(r"LABEL:\s*(.+)", raw)
        m_desc  = re.search(r"DESCRIPTION:\s*(.+)", raw)

        return {
            "cluster_label":       m_label.group(1).strip() if m_label else f"Cluster {cluster_id}",
            "cluster_description": m_desc.group(1).strip()  if m_desc  else "",
            "api_error":           None,
        }
    except Exception as e:
        return {
            "cluster_label":       f"Cluster {cluster_id}",
            "cluster_description": "",
            "api_error":           str(e),
        }


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    print("=" * 60)
    print("STEP 7: Aggregating and building feature atlas")
    print("=" * 60)

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Load all inputs ───────────────────────────────────────────────────────
    print(f"\nLoading {SCORES_CSV} ...")
    scores_df = pd.read_csv(SCORES_CSV)
    print(f"  {len(scores_df)} scored features")

    reclustered = load_jsonl(RECLUSTERED_JSONL) if RECLUSTERED_JSONL.exists() else []
    print(f"  {len(reclustered)} reclustered features")

    print(f"Loading {ENRICHMENT_CSV} ...")
    enrichment_df = load_enrichment(ENRICHMENT_CSV)

    # ── 7a: Merge ─────────────────────────────────────────────────────────────
    print("\n[7a] Merging explanations ...")
    merged_df = merge_explanations(scores_df, reclustered)
    print(f"  {len(merged_df)} features in merged set")
    print(f"  Source breakdown: {merged_df['source'].value_counts().to_dict()}")

    # ── 7b: Consistency check ─────────────────────────────────────────────────
    print("\n[7b] Running consistency checks ...")
    consistent_results = []

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Checking"):
        feat_idx = int(row["feature_idx"])
        hyp      = str(row["hypothesis"])

        if not hyp or hyp == "nan":
            consistent_results.append({
                "consistent": "UNKNOWN", "issues": "no hypothesis",
                "revised_hypothesis": hyp, "api_error": None,
            })
            continue

        # Get enrichment block
        if feat_idx in enrichment_df.index:
            enr_block = format_enrichment_block(enrichment_df.loc[feat_idx])
        else:
            enr_block = "(no enrichment data)"

        result = run_consistency_check(client, feat_idx, hyp, enr_block)
        consistent_results.append(result)

        time.sleep(API_CALL_DELAY)

    consistency_df = pd.DataFrame(consistent_results)
    merged_df["consistent"]         = consistency_df["consistent"].values
    merged_df["consistency_issues"] = consistency_df["issues"].values
    merged_df["final_hypothesis"]   = consistency_df["revised_hypothesis"].values

    print(f"  Consistency breakdown: {pd.Series(consistency_df['consistent']).value_counts().to_dict()}")

    # ── 7c: Cluster by annotation ─────────────────────────────────────────────
    print("\n[7c] Clustering features by annotation profile ...")
    # Choose n_clusters based on dataset size
    n_feat = len(merged_df)
    n_clust = max(3, min(20, n_feat // 5))
    merged_df = cluster_by_annotation(merged_df, enrichment_df, n_clusters=n_clust)
    print(f"  {n_clust} clusters created")

    # ── 7d: Label each cluster ────────────────────────────────────────────────
    print("\n[7d] Generating cluster labels ...")
    cluster_labels = {}

    for cid in tqdm(sorted(merged_df["semantic_cluster"].unique()), desc="Labelling clusters"):
        cluster_hyps = merged_df[merged_df["semantic_cluster"] == cid]["final_hypothesis"].dropna().tolist()
        if not cluster_hyps:
            cluster_labels[cid] = {"cluster_label": f"Cluster {cid}", "cluster_description": ""}
            continue

        label_result = label_cluster(client, cid, cluster_hyps)
        cluster_labels[cid] = label_result

        time.sleep(API_CALL_DELAY)

    merged_df["cluster_label"]       = merged_df["semantic_cluster"].map(
        lambda x: cluster_labels.get(x, {}).get("cluster_label", f"Cluster {x}")
    )
    merged_df["cluster_description"] = merged_df["semantic_cluster"].map(
        lambda x: cluster_labels.get(x, {}).get("cluster_description", "")
    )

    # ── Add quality flag ──────────────────────────────────────────────────────
    def quality_flag(row):
        r = row.get("pearson_r")
        cons = row.get("consistent", "UNKNOWN")
        if r is None:
            return "unscored"
        if r >= SCORE_ACCEPT_THRESHOLD and cons in ("YES", "PARTIAL"):
            return "high_quality"
        if r >= SCORE_MIN_THRESHOLD:
            return "medium_quality"
        return "low_quality"

    merged_df["quality_flag"] = merged_df.apply(quality_flag, axis=1)

    # ── Save ──────────────────────────────────────────────────────────────────
    col_order = [
        "feature_idx", "final_hypothesis", "hypothesis", "evidence",
        "confidence", "pearson_r", "original_pearson_r", "score_category",
        "consistent", "consistency_issues", "source", "cluster_id_used",
        "semantic_cluster", "cluster_label", "cluster_description", "quality_flag",
    ]
    # Only keep columns that exist
    col_order = [c for c in col_order if c in merged_df.columns]
    atlas_df = merged_df[col_order]
    atlas_df.to_csv(FEATURE_ATLAS_CSV, index=False)
    print(f"\nSaved feature atlas → {FEATURE_ATLAS_CSV}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE ATLAS SUMMARY")
    print("=" * 60)
    print(f"\nTotal features: {len(atlas_df)}")

    print("\nQuality breakdown:")
    print(atlas_df["quality_flag"].value_counts().to_string())

    print("\nTop clusters by size:")
    cluster_sizes = (
        atlas_df.groupby(["semantic_cluster", "cluster_label"])
        .size()
        .reset_index(name="n_features")
        .sort_values("n_features", ascending=False)
    )
    print(cluster_sizes.head(10).to_string(index=False))

    print("\nSample high-quality interpretations:")
    hq = atlas_df[atlas_df["quality_flag"] == "high_quality"].head(5)
    for _, row in hq.iterrows():
        hyp = str(row["final_hypothesis"])[:100]
        print(f"  [{row['feature_idx']}] (r={row.get('pearson_r', '?'):.3f}) {hyp}...")

    return atlas_df


if __name__ == "__main__":
    run()
