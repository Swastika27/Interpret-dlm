"""Offline sensitivity sweep over interpretability thresholds.

Reads per-concept all_features.csv files produced by concept_feature_analysis.py
and sweeps:
  - F1 threshold tau used for the polysemanticity proxy
  - frequency threshold used to define the "dense" feature set

Activation frequency per feature is recovered from any concept row via:
    freq = tpr * prev + fpr * (1 - prev)
which is identical across concepts (modulo tiny rounding) because the
underlying acts>0 count is concept-agnostic.

Outputs CSV tables to stdout and writes plots next to results_root.

Usage:
    python utils/threshold_sensitivity.py \
        --results_root results/layer6_16384_batchtopk_64_0.0003/step8000000 \
        --neuron_root  results/layer6_16384_batchtopk_64_0.0003/neuron_concept_analysis \
        --out_dir      results/layer6_16384_batchtopk_64_0.0003/threshold_sensitivity
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Concepts known to be degenerate after rasterization (prevalence == 1.0 in
# the balanced sample, F1 collapses to the trivial 2/3). Exclude from
# aggregate stats but report separately.
DEGENERATE_HINT = ("clinvar_benign", "clinvar_pathogenic")


def load_feature_concept_matrix(fca_root: Path):
    concept_dirs = sorted([p for p in fca_root.iterdir()
                           if p.is_dir() and (p / "all_features.csv").exists()])
    concepts = [p.name for p in concept_dirs]
    f1_cols, freq_estimates, prevalences = {}, {}, {}
    n_features = None
    for cdir in concept_dirs:
        df = pd.read_csv(cdir / "all_features.csv")
        df = df.sort_values("feature_idx").reset_index(drop=True)
        if n_features is None:
            n_features = len(df)
        f1_cols[cdir.name] = df["f1"].to_numpy()
        prev = float(df["baseline_prevalence"].iloc[0])
        prevalences[cdir.name] = prev
        freq = df["recall_tpr"].to_numpy() * prev + df["fpr"].to_numpy() * (1.0 - prev)
        freq_estimates[cdir.name] = freq
    F1 = np.stack([f1_cols[c] for c in concepts], axis=1)  # [n_feat, n_concept]
    # Recover a single per-feature freq by median across non-degenerate concepts
    non_deg = [c for c in concepts if c not in DEGENERATE_HINT]
    freq_stack = np.stack([freq_estimates[c] for c in non_deg], axis=1)
    freq = np.median(freq_stack, axis=1)
    return concepts, F1, freq, prevalences


def polysemanticity_at(F1: np.ndarray, tau: float, mask=None) -> dict:
    """mean # concepts a feature passes; restricted to mask if given."""
    pass_mat = F1 >= tau
    if mask is not None:
        pass_mat = pass_mat[mask]
    per_feature_n_concepts = pass_mat.sum(axis=1)
    interpretable = per_feature_n_concepts >= 1
    return {
        "tau": tau,
        "n_features": int(pass_mat.shape[0]),
        "poly_proxy": float(per_feature_n_concepts.mean()),
        "poly_proxy_among_interp": float(
            per_feature_n_concepts[interpretable].mean()
        ) if interpretable.any() else 0.0,
        "n_interpretable": int(interpretable.sum()),
        "frac_interpretable": float(interpretable.mean()),
        "n_monosemantic": int((per_feature_n_concepts == 1).sum()),
        "n_polysemantic": int((per_feature_n_concepts >= 2).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True, type=Path,
                    help="…/<tag>/stepN  — must contain feature_concept_analysis/")
    ap.add_argument("--neuron_root", type=Path, default=None,
                    help="…/<tag>/neuron_concept_analysis (no step subdir)")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--tau_grid", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    ap.add_argument("--freq_grid", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10, 0.20, 0.30, 0.50])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fca_root = args.results_root / "feature_concept_analysis"
    concepts, F1, freq, prev = load_feature_concept_matrix(fca_root)
    print(f"Loaded {F1.shape[0]} features × {F1.shape[1]} concepts")
    print(f"Active features (freq>0): {(freq > 0).sum()}  "
          f"freq[max]={freq.max():.4f}  freq[median]={np.median(freq):.6f}")

    keep_concept_mask = np.array([c not in DEGENERATE_HINT for c in concepts])
    F1_clean = F1[:, keep_concept_mask]
    clean_concepts = [c for c in concepts if c not in DEGENERATE_HINT]
    print(f"Non-degenerate concepts: {len(clean_concepts)} (excluded: {DEGENERATE_HINT})")

    # ---- Sweep 1: F1 threshold tau over polysemanticity proxy ----
    rows = []
    for tau in args.tau_grid:
        rec = polysemanticity_at(F1_clean, tau)
        rec["scope"] = "all_features"
        rows.append(rec)
    tau_df = pd.DataFrame(rows)
    tau_df.to_csv(args.out_dir / "sweep_tau.csv", index=False)
    print("\n=== Sweep 1: F1 threshold (tau) ===")
    print(tau_df.to_string(index=False))

    # ---- Sweep 2: frequency threshold defining "dense" set ----
    rows = []
    for fthr in args.freq_grid:
        dense_mask = freq > fthr
        sparse_mask = ~dense_mask
        for tau in [0.10, 0.20]:
            rec = {
                "freq_thr": fthr,
                "tau": tau,
                "n_dense": int(dense_mask.sum()),
                "frac_dense": float(dense_mask.mean()),
                "poly_full": polysemanticity_at(F1_clean, tau)["poly_proxy"],
                "poly_sparse_only": polysemanticity_at(
                    F1_clean, tau, mask=sparse_mask
                )["poly_proxy"],
                "poly_dense_only": polysemanticity_at(
                    F1_clean, tau, mask=dense_mask
                )["poly_proxy"] if dense_mask.any() else float("nan"),
            }
            rows.append(rec)
    freq_df = pd.DataFrame(rows)
    freq_df.to_csv(args.out_dir / "sweep_freq.csv", index=False)
    print("\n=== Sweep 2: frequency threshold (dense definition) ===")
    print(freq_df.to_string(index=False))

    # ---- Sweep 3: per-concept best F1 vs neuron baseline ----
    best_sae = F1.max(axis=0)  # [n_concept]
    rows = []
    if args.neuron_root is not None:
        n_concepts, N1, _, _ = load_feature_concept_matrix(args.neuron_root)
        # Align by concept name
        n_lookup = {c: N1[:, i].max() for i, c in enumerate(n_concepts)}
    else:
        n_lookup = {}
    for i, c in enumerate(concepts):
        rows.append({
            "concept": c,
            "best_sae_f1": float(best_sae[i]),
            "best_neuron_f1": float(n_lookup.get(c, float("nan"))),
            "delta": float(best_sae[i] - n_lookup.get(c, float("nan")))
                     if c in n_lookup else float("nan"),
            "degenerate": c in DEGENERATE_HINT,
        })
    per_concept = pd.DataFrame(rows).sort_values("best_sae_f1", ascending=False)
    per_concept.to_csv(args.out_dir / "per_concept_best_f1.csv", index=False)
    print("\n=== Per-concept best F1 (SAE feature vs raw neuron) ===")
    print(per_concept.to_string(index=False))

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tau_df["tau"], tau_df["poly_proxy"], marker="o",
                label="mean # concepts per feature")
        ax.plot(tau_df["tau"], tau_df["frac_interpretable"], marker="s",
                label="fraction of features with ≥1 concept passing")
        ax.set_xlabel("F1 threshold τ")
        ax.set_ylabel("value")
        ax.set_title("Polysemanticity proxy vs F1 threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "tau_sensitivity.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sub = freq_df[freq_df["tau"] == 0.10]
        ax.plot(sub["freq_thr"], sub["poly_full"], marker="o", label="all features")
        ax.plot(sub["freq_thr"], sub["poly_sparse_only"], marker="s",
                label="excluding dense set")
        ax2 = ax.twinx()
        ax2.plot(sub["freq_thr"], sub["n_dense"], marker="x", color="gray",
                 alpha=0.6, label="|dense set| (right axis)")
        ax.set_xlabel("frequency threshold defining 'dense'")
        ax.set_ylabel("polysemanticity proxy @ τ=0.10")
        ax2.set_ylabel("# dense features")
        ax.set_xscale("log")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "freq_sensitivity.png", dpi=150)
        plt.close(fig)
        print(f"\nWrote plots and CSVs to {args.out_dir}")
    except Exception as e:
        print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
