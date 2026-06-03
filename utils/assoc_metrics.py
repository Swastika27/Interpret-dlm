"""
assoc_metrics.py — single source of truth for concept↔feature association metrics.

Both main/concept_feature_analysis.py (new runs) and utils/recompute_metrics.py
(recompute from existing counts) import from here so the output schema is identical.

All metrics are computed from RAW (un-balanced) confusion-matrix counts:

    TP = pos_acts                  # positive tokens on which the feature fires
    FP = neg_acts                  # negative tokens on which the feature fires
    FN = n_pos - TP
    TN = n_neg - FP

Why raw, not class-balanced
---------------------------
The old pipeline scaled negatives down to n_pos. That makes an *always-on* feature
score F1 = 0.667 on every concept regardless of prevalence — a degenerate floor that
poisons thresholding. Computed on raw counts:

  - MCC has a true zero at chance (an always-on feature → MCC = 0), so it needs no
    arbitrary association threshold and is the recommended headline scalar.
  - enrichment / lift = P(fire|pos)/P(fire|neg) is the interpretable "how concept-
    specific" measure (baseline 1.0).
  - precision/recall/F1 are reported raw (F1 is prevalence-sensitive — always print
    prevalence beside it).
  - balanced_f1 is kept ONLY for continuity with the previous analysis; do not use
    it as the association test.

Note on metric magnitudes: every metric's *magnitude* is prevalence-sensitive on
imbalanced data; MCC's zero-point is not, which is the property we rely on. Use the
null model (main/concept_feature_null.py) to decide whether a value is surprising.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# Per-feature CSV (all_features.csv / top_features.csv)
CANONICAL_FEATURE_COLUMNS = [
    "feature_idx",
    "mcc",
    "enrichment",
    "log2_enrichment",
    "f1",
    "precision",
    "recall_tpr",
    "specificity_tnr",
    "fpr",
    "fnr",
    "tp", "fp", "tn", "fn",
    "n_positive_tokens", "n_negative_tokens",
    "prevalence",
    "balanced_f1",
]

# Best-feature-per-concept CSV (summary.csv)
CANONICAL_SUMMARY_COLUMNS = [
    "concept",
    "best_feature_idx",
    "mcc",
    "enrichment",
    "f1",
    "precision",
    "recall_tpr",
    "specificity_tnr",
    "fpr",
    "tp", "fp", "tn", "fn",
    "n_positive_tokens", "n_negative_tokens",
    "prevalence",
    "balanced_f1",
]

# Default ranking key (descending). MCC has a meaningful zero baseline.
RANK_KEY = "mcc"


def _safe_div(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)


def compute_raw_metrics(
    pos_acts: np.ndarray,   # (F,) firings on positive tokens  (= raw TP)
    neg_acts: np.ndarray,   # (F,) firings on negative tokens  (= raw FP)
    n_pos: int,
    n_neg: int,
) -> Dict[str, np.ndarray]:
    """
    Vectorised over features. Returns a dict of float64 arrays (length F) plus the
    raw integer confusion-matrix cells, keyed to CANONICAL_FEATURE_COLUMNS names
    (minus feature_idx / the two scalar token counts / prevalence, which callers add).
    """
    n_pos = float(n_pos)
    n_neg = float(n_neg)
    TP = np.asarray(pos_acts, dtype=np.float64)
    FP = np.asarray(neg_acts, dtype=np.float64)
    FN = n_pos - TP
    TN = n_neg - FP

    recall = _safe_div(TP, n_pos)                 # P(fire | positive) = TPR
    fpr = _safe_div(FP, n_neg)                    # P(fire | negative)
    precision = _safe_div(TP, TP + FP)
    specificity = _safe_div(TN, n_neg)            # TNR = 1 - fpr
    fnr = _safe_div(FN, n_pos)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # enrichment / lift: recall / fpr.  fpr==0 & recall>0 -> +inf (perfectly specific);
    # recall==0 -> 0.  log2 mirrors that.
    with np.errstate(divide="ignore", invalid="ignore"):
        enrichment = np.where(fpr > 0, recall / np.where(fpr > 0, fpr, 1.0),
                              np.where(recall > 0, np.inf, 0.0))
        log2_enrichment = np.where(
            np.isfinite(enrichment) & (enrichment > 0),
            np.log2(np.where(enrichment > 0, enrichment, 1.0)),
            np.where(enrichment == 0, -np.inf, np.inf),
        )

    # Matthews correlation coefficient (raw). Denominator 0 -> MCC 0 (no discriminative power).
    mcc_num = TP * TN - FP * FN
    mcc_den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = np.divide(mcc_num, mcc_den, out=np.zeros_like(mcc_num), where=mcc_den > 0)

    # balanced F1 (legacy: negatives scaled to n_pos) — comparability only.
    n_neg_eff = min(n_neg, n_pos)
    scale = (n_neg_eff / n_neg) if n_neg > 0 else 0.0
    FP_bal = FP * scale
    precision_bal = _safe_div(TP, TP + FP_bal)
    balanced_f1 = _safe_div(2 * precision_bal * recall, precision_bal + recall)

    return {
        "mcc": mcc,
        "enrichment": enrichment,
        "log2_enrichment": log2_enrichment,
        "f1": f1,
        "precision": precision,
        "recall_tpr": recall,
        "specificity_tnr": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "tp": np.rint(TP).astype(np.int64),
        "fp": np.rint(FP).astype(np.int64),
        "tn": np.rint(TN).astype(np.int64),
        "fn": np.rint(FN).astype(np.int64),
        "balanced_f1": balanced_f1,
    }


def rank_order(metrics: Dict[str, np.ndarray], key: str = RANK_KEY) -> np.ndarray:
    """Indices that sort features by `key` descending (NaN/-inf last)."""
    vals = np.asarray(metrics[key], dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    return np.argsort(vals)[::-1]


def format_value(k: str, v) -> str:
    """Consistent CSV formatting: ints bare, floats 6dp, inf/nan as-is."""
    if k == "concept":
        return str(v)
    if k in ("feature_idx", "best_feature_idx", "tp", "fp", "tn", "fn",
             "n_positive_tokens", "n_negative_tokens"):
        return str(int(v))
    f = float(v)
    if not np.isfinite(f):
        return "inf" if f > 0 else ("-inf" if f < 0 else "nan")
    return f"{f:.6f}"
