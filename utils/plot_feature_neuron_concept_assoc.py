#!/usr/bin/env python3
"""
Compare concept–association strength from SAE feature analysis vs raw neuron analysis.

Reads, for each training step directory under a results root (first match wins):
  <results_root>/<step_dir>/feature_*_analysis/summary.csv
  <results_root>/<step_dir>/neuron_*_analysis/summary.csv
  or the same filenames under <results_root>/<analysis_dir>/<step_dir>/

Produces:
  1) Per step: horizontal grouped bars (feature vs neuron) per concept; figure height grows with
     the number of concepts.
  2) Per concept: grouped bar chart of that metric vs training step (feature vs neuron per step).

Example:
  python utils/plot_feature_neuron_concept_assoc.py \
    --results_root results/layer6_8192_gated_l10.01_aux1.0_0.0003 \
    --out_dir results/layer6_8192_gated_l10.01_aux1.0_0.0003/concept_assoc_plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Trailing commas required: ("x") is a str in Python, not a 1-tuple.
FEATURE_SUBDIRS = (
    "feature_concept_analysis",
    "feature_concpet_analysis",
)
NEURON_SUBDIRS = (
    "neuron_concept_analysis",
    "neuron_concpet_analysis",
)

DEFAULT_METRICS = ("f1", "precision", "recall_tpr")


def _find_summary(
    step_dir: Path,
    results_root: Path,
    subdirs: Tuple[str, ...],
) -> Optional[Path]:
    """Prefer <step>/<analysis>/summary.csv; also try <root>/<analysis>/<step>/summary.csv."""
    for sd in subdirs:
        p = step_dir / sd / "summary.csv"
        if p.is_file():
            return p
    for sd in subdirs:
        p = results_root / sd / step_dir.name / "summary.csv"
        if p.is_file():
            return p
    return None


def parse_step_dir_name(name: str) -> Optional[int]:
    m = re.match(r"step_?(\d+)$", name.strip(), flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def list_step_dirs(results_root: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    if not results_root.is_dir():
        return out
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        step = parse_step_dir_name(child.name)
        if step is None:
            continue
        out.append((step, child))
    out.sort(key=lambda x: x[0])
    return out


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "concept" not in df.columns:
        raise ValueError(f"Missing 'concept' column in {path}")
    return df


def merge_feature_neuron(
    feat_df: pd.DataFrame,
    neur_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    c = "concept"
    for d in (feat_df, neur_df):
        if metric not in d.columns:
            raise ValueError(f"Missing metric column {metric!r} in summary CSV")
    a = feat_df[[c, metric]].copy()
    b = neur_df[[c, metric]].copy()
    a = a.rename(columns={metric: f"feature_{metric}"})
    b = b.rename(columns={metric: f"neuron_{metric}"})
    return a.merge(b, on=c, how="inner")


def plot_per_step(
    *,
    step: int,
    merged: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    n = len(merged)
    # Fixed width; height grows with one row per concept (horizontal bars).
    fig_w, row_h = 9.0, 0.38
    fig, ax = plt.subplots(figsize=(fig_w, max(4.5, row_h * n + 1.8)))
    y = np.arange(n, dtype=float)
    h = 0.34
    fk = f"feature_{metric}"
    nk = f"neuron_{metric}"
    ax.barh(y - h / 2, merged[fk], height=h, label=f"Feature ({metric})", align="center")
    ax.barh(y + h / 2, merged[nk], height=h, label=f"Neuron ({metric})", align="center")
    ax.set_yticks(y)
    ax.set_yticklabels(merged["concept"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_concept_over_steps(
    series: Dict[str, List[Tuple[int, float, float]]],
    metric: str,
    out_dir: Path,
) -> None:
    """series[concept] = list of (step, feature_val, neuron_val) sorted by step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for concept, rows in sorted(series.items()):
        if not rows:
            continue
        steps = [r[0] for r in rows]
        fv = [r[1] for r in rows]
        nv = [r[2] for r in rows]
        n_steps = len(steps)
        fig_w = max(8.0, 0.55 * n_steps + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 5.0))
        x = np.arange(n_steps, dtype=float)
        w = 0.36
        ax.bar(x - w / 2, fv, width=w, label=f"Feature ({metric})")
        ax.bar(x + w / 2, nv, width=w, label=f"Neuron ({metric})")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right")
        ax.set_xlabel("Training step")
        ax.set_ylabel(metric)
        safe = re.sub(r"[^\w.\-]+", "_", concept)[:120]
        ax.set_title(f"{concept} — feature vs neuron")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe}_over_steps.png", dpi=150)
        plt.close(fig)


def run(
    results_root: Path,
    out_dir: Path,
    metric: str,
) -> None:
    step_dirs = list_step_dirs(results_root)
    if not step_dirs:
        raise SystemExit(f"No step* directories under {results_root}")

    per_step_dir = out_dir / "per_step"
    over_steps_dir = out_dir / "per_concept_over_steps"

    # Accumulate per-concept series for part 2
    acc: Dict[str, List[Tuple[int, float, float]]] = {}

    for step, sdir in step_dirs:
        fp = _find_summary(sdir, results_root, FEATURE_SUBDIRS)
        np_ = _find_summary(sdir, results_root, NEURON_SUBDIRS)
        if fp is None or np_ is None:
            miss = []
            if fp is None:
                miss.append("feature summary")
            if np_ is None:
                miss.append("neuron summary")
            print(f"[skip] step {step} ({sdir.name}): missing {', '.join(miss)}")
            continue

        try:
            feat_df = load_summary(fp)
            neur_df = load_summary(np_)
            merged = merge_feature_neuron(feat_df, neur_df, metric)
        except Exception as e:
            print(f"[skip] step {step}: {e}")
            continue

        if merged.empty:
            print(f"[skip] step {step}: no overlapping concepts after merge")
            continue

        title = f"{results_root.name} — step {step} — feature vs neuron ({metric})"
        plot_per_step(
            step=step,
            merged=merged,
            metric=metric,
            out_path=per_step_dir / f"step_{step}_feature_vs_neuron_{metric}.png",
            title=title,
        )
        print(f"[ok] per-step plot {step} ({len(merged)} concepts)")

        fk = f"feature_{metric}"
        nk = f"neuron_{metric}"
        for _, row in merged.iterrows():
            cname = str(row["concept"])
            acc.setdefault(cname, []).append(
                (step, float(row[fk]), float(row[nk]))
            )

    for cname in acc:
        acc[cname].sort(key=lambda t: t[0])

    if not acc:
        raise SystemExit("No data to plot (check paths and CSV contents).")

    plot_per_concept_over_steps(acc, metric, over_steps_dir)
    print(f"[ok] wrote {len(acc)} per-concept over-steps figures under {over_steps_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results_root",
        type=str,
        default="results/layer6_8192_gated_l10.01_aux1.0_0.0003",
        help="Directory containing step* subfolders with feature/neuron analysis outputs.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: <results_root>/concept_assoc_plots).",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="f1",
        help=f"Column from summary.csv to compare (typical: {', '.join(DEFAULT_METRICS)}).",
    )
    args = ap.parse_args()

    root = Path(args.results_root)
    out = Path(args.out_dir) if args.out_dir else root / "concept_assoc_plots"
    out = out.resolve()

    m = args.metric.strip()
    if not m:
        raise SystemExit("--metric must be non-empty")
    run(results_root=root.resolve(), out_dir=out, metric=m)
    print(f"Done. Plots under {out}")


if __name__ == "__main__":
    main()
