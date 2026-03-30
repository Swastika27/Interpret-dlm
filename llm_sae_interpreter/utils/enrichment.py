"""
utils/enrichment.py — Format annotation enrichment data for LLM prompts.

Converts the pct_* columns from enrichment.csv into human-readable blocks
used in explainer, simulator, and consistency-check prompts.
"""

from __future__ import annotations
import pandas as pd
from config import (
    ANNOTATION_PCT_COLS,
    ANNOTATION_LABELS,
    ENRICHMENT_DISPLAY_THRESHOLD,
)


def format_enrichment_block(
    enrichment_row: pd.Series,
    threshold: float = ENRICHMENT_DISPLAY_THRESHOLD,
    include_all: bool = False,
) -> str:
    """
    Build a human-readable annotation enrichment summary for a single feature.

    Parameters
    ----------
    enrichment_row : a row from enrichment.csv (indexed by feature_idx)
    threshold      : only show annotations above this percentage (default 5%)
    include_all    : if True, show all annotations regardless of threshold

    Returns a multi-line string like:
        - intron: 70.0%
        - SINE repeat: 57.0%
        - ClinVar pathogenic variant: 95.0%
        ...
    """
    lines = []
    for col in ANNOTATION_PCT_COLS:
        if col not in enrichment_row.index:
            continue
        pct = float(enrichment_row[col])
        if include_all or pct >= threshold:
            label = ANNOTATION_LABELS.get(col, col)
            lines.append(f"  - {label}: {pct:.1f}%")

    if not lines:
        lines = ["  (no annotations above threshold)"]

    # Add n_total for context
    if "n_total" in enrichment_row.index:
        header = f"(n = {int(enrichment_row['n_total'])} loci analysed)\n"
    else:
        header = ""

    return header + "\n".join(lines)


def get_top_annotations(
    enrichment_row: pd.Series,
    n: int = 5,
) -> list[tuple[str, float]]:
    """
    Return the top-n annotations by percentage for a feature.

    Returns a list of (label, pct) tuples sorted descending.
    """
    pairs = []
    for col in ANNOTATION_PCT_COLS:
        if col in enrichment_row.index:
            pct = float(enrichment_row[col])
            label = ANNOTATION_LABELS.get(col, col)
            pairs.append((label, pct))

    pairs.sort(key=lambda x: -x[1])
    return pairs[:n]


def format_subcluster_enrichment_block(
    subclusters: list[dict],
    enrichment_df: pd.DataFrame,
) -> str:
    """
    Format enrichment summaries for multiple sub-clusters side by side.

    Parameters
    ----------
    subclusters   : list of dicts with keys 'cluster_id' and 'feature_indices'
                    (list of feature_idx values or row indices in the sub-cluster)
    enrichment_df : full enrichment DataFrame indexed by feature_idx

    Returns a formatted multi-line string with one section per sub-cluster.
    """
    blocks = []
    for sc in subclusters:
        cid = sc["cluster_id"]
        # Average enrichment across examples in this sub-cluster
        # sc["rows"] are the actual DataFrame rows (from top_activations)
        rows = sc.get("rows", [])
        n = len(rows)

        blocks.append(f"Sub-cluster {cid} ({n} examples):")

        # Compute per-annotation averages across examples if we have feature rows
        if rows and hasattr(rows[0], "get"):
            # rows are dicts — compute mean activation
            mean_act = sum(r.get("activation_norm", 0) for r in rows) / max(n, 1)
            blocks.append(f"  Mean activation: {mean_act:.1f}/10")

        # Show annotation info if available
        annotation_summary = sc.get("annotation_summary", "")
        if annotation_summary:
            blocks.append(annotation_summary)
        blocks.append("")

    return "\n".join(blocks)


def compute_feature_annotation_vector(
    enrichment_row: pd.Series,
) -> list[float]:
    """
    Return the pct_ values as a numeric vector for clustering / similarity.
    Used in Step 6 to cluster features by annotation profile.
    """
    return [
        float(enrichment_row.get(col, 0.0))
        for col in ANNOTATION_PCT_COLS
    ]
