"""
plot_null_concept.py — the thesis figure: per-concept observed best metric (e.g. MCC)
overlaid on the autocorrelation-preserving null band, with the raw-neuron baseline.

Reads null_concept_summary.csv (written by main/concept_feature_null.py), whose columns
are metric-generic: concept, metric, best_feature_idx, observed_best, p_value_bh,
significant_bh, null_max_mean, null_max_q95, exceeds_null_max_q95, prevalence, ...

For each concept it draws:
  - a shaded band  [null_max_mean , null_max_q95]  = the chance level of the BEST-of-
    candidates metric under the null (the selection-aware floor),
  - the observed best metric as a dot (green = BH-significant, grey = not),
  - the raw-neuron observed best as a hollow triangle (if --neuron_dir / sweep found it),
  - a dashed reference line at the metric's chance value (0 for MCC, 1 for enrichment).
A concept whose dot sits to the right of its band is a real association.

Usage
-----
  # one layer/checkpoint:
  python utils/plot_null_concept.py \
      --null_dir   results/layer4_.../step1600000/concept_feature_null_circular_mcc \
      --neuron_dir results/layer4_.../neuron_concept_null_circular_mcc \
      --out results/layer4_.../step1600000/null_overlay_circular_mcc.png

  # all layers at best epoch, one grid figure (called automatically by run_null_sweep --also_plot):
  python utils/plot_null_concept.py --sweep --results_root results \
      --mode circular --metric mcc --out results/null_concept_sweep_circular_mcc.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from run_null_sweep import find_best_step, layer_of  # noqa: E402

CHANCE = {"mcc": 0.0, "enrichment": 1.0, "f1": 0.0, "precision": 0.0, "balanced_f1": 0.0}


def _load(null_csv):
    if not null_csv or not os.path.isfile(null_csv):
        return None
    df = pd.read_csv(null_csv)
    if df.empty or "observed_best" not in df.columns:
        return None
    return df


def plot_one(ax, sae_df, neuron_df, metric, title):
    """Render one panel; returns False if no data."""
    if sae_df is None:
        ax.set_title(f"{title}\n(no data)", fontsize=9)
        ax.axis("off")
        return False
    df = sae_df.sort_values("observed_best", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    mean = df["null_max_mean"].to_numpy(float)
    q95 = df["null_max_q95"].to_numpy(float)
    obs = df["observed_best"].to_numpy(float)
    sig = df["significant_bh"].astype(str).str.lower().isin(["true", "1"]).to_numpy() \
        if "significant_bh" in df.columns else (obs > q95)

    # null band: mean -> q95
    ax.barh(y, width=np.maximum(q95 - mean, 0), left=mean, height=0.55,
            color="0.80", edgecolor="0.6", linewidth=0.3, zorder=1,
            label="null band (mean→q95)")
    # q95 ticks (selection-aware threshold)
    ax.scatter(q95, y, marker="|", color="0.45", s=60, zorder=2)
    # observed
    colors = np.where(sig, "tab:green", "tab:gray")
    ax.scatter(obs, y, color=colors, s=34, zorder=4, label="SAE observed (best)")

    # neuron baseline
    if neuron_df is not None:
        nmap = dict(zip(neuron_df["concept"], neuron_df["observed_best"]))
        nx = np.array([nmap.get(c, np.nan) for c in df["concept"]], float)
        ax.scatter(nx, y, marker="^", facecolors="none", edgecolors="tab:orange",
                   s=42, linewidths=1.1, zorder=3, label="neuron observed (best)")

    ax.axvline(CHANCE.get(metric, 0.0), ls="--", color="0.5", lw=0.8, zorder=0)
    if metric == "enrichment":
        ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(df["concept"].str.replace("_", " ").str.slice(0, 26), fontsize=7)
    ax.set_xlabel(metric)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", ls=":", alpha=0.4)
    return True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--null_dir", default=None, help="A concept_feature_null_* dir (single panel)")
    p.add_argument("--neuron_dir", default=None, help="Matching neuron_concept_null_* dir")
    p.add_argument("--sweep", action="store_true", help="Grid over all layers at best epoch")
    p.add_argument("--results_root", default=os.path.join(REPO, "results"))
    p.add_argument("--mode", default="circular", choices=["circular", "blockswap"])
    p.add_argument("--metric", default="mcc",
                   choices=["mcc", "enrichment", "f1", "precision", "balanced_f1"])
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    return p.parse_args()


def main():
    a = parse_args()

    if not a.sweep:
        if not a.null_dir:
            raise SystemExit("Provide --null_dir (single panel) or --sweep.")
        sae = _load(os.path.join(a.null_dir, "null_concept_summary.csv"))
        neu = _load(os.path.join(a.neuron_dir, "null_concept_summary.csv")) if a.neuron_dir else None
        metric = (sae["metric"].iloc[0] if sae is not None and "metric" in sae.columns
                  else a.metric)
        fig, ax = plt.subplots(figsize=(7.5, 0.34 * (len(sae) if sae is not None else 10) + 1.5))
        plot_one(ax, sae, neu, metric, a.title or os.path.basename(a.null_dir.rstrip("/\\")))
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        out = a.out or os.path.join(a.null_dir, f"null_overlay_{metric}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"Wrote {out}")
        return

    # ---- sweep: one panel per layer at its best epoch ----
    tags = []
    for name in sorted(os.listdir(a.results_root)):
        if re.match(r"layer\d+_.*batchtopk", name) and \
                os.path.isdir(os.path.join(a.results_root, name)):
            L = layer_of(name)
            if a.layers and L not in a.layers:
                continue
            tags.append((L, name))
    tags.sort()
    if not tags:
        raise SystemExit(f"No layer dirs under {a.results_root}")

    suffix = f"{a.mode}_{a.metric}"
    n = len(tags)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 0.34 * 20 + 1.6),
                             squeeze=False)
    drawn = 0
    for idx, (L, tag) in enumerate(tags):
        ax = axes[idx // ncols][idx % ncols]
        best, _, _ = find_best_step(os.path.join(a.results_root, tag))
        if best is None:
            ax.axis("off"); continue
        sdir = os.path.join(a.results_root, tag, f"step{best}", f"concept_feature_null_{suffix}")
        ndir = os.path.join(a.results_root, tag, f"neuron_concept_null_{suffix}")
        sae = _load(os.path.join(sdir, "null_concept_summary.csv"))
        neu = _load(os.path.join(ndir, "null_concept_summary.csv"))
        if plot_one(ax, sae, neu, a.metric, f"L{L} (step {best})"):
            drawn += 1
        if idx == 0 and sae is not None:
            ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    for j in range(len(tags), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Observed best {a.metric} vs autocorrelation null "
                 f"(null_mode={a.mode}) — SAE (dots) vs neurons (△)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = a.out or os.path.join(a.results_root, f"null_concept_sweep_{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}  ({drawn}/{len(tags)} layers had data)")


if __name__ == "__main__":
    main()
