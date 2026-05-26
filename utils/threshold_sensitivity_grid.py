"""Run the threshold sensitivity sweep across all layer x checkpoint cells.

Outputs a long-form CSV (one row per cell x tau or cell x freq_thr) plus
summary plots showing how the polysemanticity proxy and the SAE-vs-neuron
gap evolve over training, and how they differ across HyenaDNA layers.

Assumes the directory layout produced by experiment.sh:
    results/<layer_tag>/step<N>/feature_concept_analysis/<concept>/all_features.csv
    results/<layer_tag>/neuron_concept_analysis/<concept>/all_features.csv  (optional)
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEGENERATE = ("clinvar_benign", "clinvar_pathogenic")
TAU_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50] # f1 thresholds
FREQ_GRID = [0.02, 0.04, 0.05, 0.10, 0.20, 0.30, 0.50] # activation frequency thresholds
LAYER_RE = re.compile(r"layer(\d+)_")
STEP_RE = re.compile(r"step(\d+)")


def load_matrix(fca_root: Path):
    """Return (concepts, F1[n_feat,n_concept], freq[n_feat])."""
    concept_dirs = sorted(
        p for p in fca_root.iterdir()
        if p.is_dir() and (p / "all_features.csv").exists()
    )
    if not concept_dirs:
        return None, None, None
    # Determine canonical feature count = mode across concept files
    sizes = []
    for cdir in concept_dirs:
        try:
            sizes.append(sum(1 for _ in open(cdir / "all_features.csv")) - 1)
        except Exception:
            sizes.append(0)
    if not sizes:
        return None, None, None
    canonical = max(set(sizes), key=sizes.count)
    concepts, f1_cols, freq_cols, skipped = [], [], [], []
    for cdir, sz in zip(concept_dirs, sizes):
        if sz != canonical:
            skipped.append((cdir.name, sz))
            continue
        df = pd.read_csv(cdir / "all_features.csv")
        df = df.sort_values("feature_idx").reset_index(drop=True)
        concepts.append(cdir.name)
        f1_cols.append(df["f1"].to_numpy(dtype=np.float32))
        prev = float(df["baseline_prevalence"].iloc[0])
        freq_cols.append(
            df["recall_tpr"].to_numpy(dtype=np.float32) * prev
            + df["fpr"].to_numpy(dtype=np.float32) * (1.0 - prev)
        )
    if skipped:
        print(f"  WARN {fca_root}: skipped {len(skipped)} partial concepts (canonical={canonical}): {skipped[:3]}...")
    if not f1_cols:
        return None, None, None
    F1 = np.stack(f1_cols, axis=1)
    non_deg = [i for i, c in enumerate(concepts) if c not in DEGENERATE]
    freq_stack = np.stack([freq_cols[i] for i in non_deg], axis=1)
    freq = np.median(freq_stack, axis=1)
    return concepts, F1, freq


def cell_rows(layer: int, step: int, fca_root: Path, neuron_root: Path | None):
    concepts, F1, freq = load_matrix(fca_root)
    if concepts is None:
        return [], []
    keep = np.array([c not in DEGENERATE for c in concepts])
    F1c = F1[:, keep]
    n_concepts_clean = int(keep.sum())

    # Neuron baseline: best F1 per concept (raw HyenaDNA channels)
    neuron_best = None
    if neuron_root is not None and neuron_root.is_dir():
        n_concepts, N1, _ = load_matrix(neuron_root)
        if n_concepts is not None:
            lookup = dict(zip(n_concepts, N1.max(axis=0)))
            neuron_best = np.array(
                [lookup.get(c, np.nan) for c in concepts], dtype=np.float32
            )

    sae_best = F1.max(axis=0)

    tau_rows = []
    for tau in TAU_GRID:
        passing = F1c >= tau
        per_feat = passing.sum(axis=1)
        interp = per_feat >= 1
        # SAE-vs-neuron concept-level win rate at this tau:
        if neuron_best is not None:
            both_above = (sae_best >= tau) & (neuron_best >= tau)
            sae_wins = int(((sae_best > neuron_best + 1e-4) & both_above).sum())
            ties = int(((np.abs(sae_best - neuron_best) <= 1e-4)).sum())
        else:
            sae_wins = ties = -1
        tau_rows.append({
            "layer": layer, "step": step, "tau": tau,
            "poly_proxy": float(per_feat.mean()),
            "poly_proxy_among_interp": float(per_feat[interp].mean()) if interp.any() else 0.0,
            "n_interpretable": int(interp.sum()),
            "frac_interpretable": float(interp.mean()),
            "n_monosemantic": int((per_feat == 1).sum()),
            "n_polysemantic": int((per_feat >= 2).sum()),
            "n_concepts_clean": n_concepts_clean,
            "sae_wins_vs_neuron": sae_wins,
            "sae_ties_vs_neuron": ties,
        })

    freq_rows = []
    for fthr in FREQ_GRID:
        dense_mask = freq > fthr
        sparse_mask = ~dense_mask
        for tau in (0.10, 0.20):
            full = (F1c >= tau).sum(axis=1)
            sparse = (F1c[sparse_mask] >= tau).sum(axis=1) if sparse_mask.any() else np.array([0.0])
            dense = (F1c[dense_mask] >= tau).sum(axis=1) if dense_mask.any() else np.array([np.nan])
            freq_rows.append({
                "layer": layer, "step": step,
                "freq_thr": fthr, "tau": tau,
                "n_dense": int(dense_mask.sum()),
                "frac_dense": float(dense_mask.mean()),
                "poly_full": float(full.mean()),
                "poly_sparse_only": float(sparse.mean()),
                "poly_dense_only": float(np.nanmean(dense)),
            })

    # Cell-level SAE-vs-neuron summary (no tau axis): mean delta on non-trivial concepts
    summary = {
        "layer": layer, "step": step,
        "n_concepts_total": len(concepts),
        "n_concepts_clean": n_concepts_clean,
        "n_features_active": int((freq > 0).sum()),
        "mean_freq_active": float(freq[freq > 0].mean()) if (freq > 0).any() else 0.0,
        "best_sae_f1_mean": float(sae_best[keep].mean()),
        "best_sae_f1_max":  float(sae_best[keep].max()),
        "best_neuron_f1_mean": float(neuron_best[keep].mean()) if neuron_best is not None else np.nan,
        "delta_sae_minus_neuron_mean": float((sae_best[keep] - neuron_best[keep]).mean())
            if neuron_best is not None else np.nan,
        "n_concepts_at_trivial": int(np.isclose(sae_best[keep], 2/3, atol=1e-3).sum()),
    }
    return tau_rows, freq_rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, default=Path("results"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("results/_aggregate/threshold_sensitivity"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_tau, all_freq, all_summary = [], [], []
    cells = []
    for tag in sorted(args.results_root.iterdir()):
        m = LAYER_RE.search(tag.name)
        if not m or "batchtopk" not in tag.name:
            continue
        layer = int(m.group(1))
        neuron_root = tag / "neuron_concept_analysis"
        if not neuron_root.is_dir():
            neuron_root = None
        for step_dir in sorted(tag.iterdir()):
            sm = STEP_RE.match(step_dir.name)
            if not sm:
                continue
            fca_root = step_dir / "feature_concept_analysis"
            if not fca_root.is_dir():
                continue
            step = int(sm.group(1))
            cells.append((layer, step, fca_root, neuron_root))

    print(f"Running sweep over {len(cells)} layer x checkpoint cells...")
    for i, (layer, step, fca_root, nroot) in enumerate(cells, 1):
        tau_rows, freq_rows, summary = cell_rows(layer, step, fca_root, nroot)
        all_tau.extend(tau_rows)
        all_freq.extend(freq_rows)
        all_summary.append(summary)
        if i % 10 == 0 or i == len(cells):
            print(f"  [{i}/{len(cells)}] layer{layer} step{step}")

    tau_df = pd.DataFrame(all_tau)
    freq_df = pd.DataFrame(all_freq)
    summ_df = pd.DataFrame(all_summary).sort_values(["layer", "step"])

    tau_df.to_csv(args.out_dir / "grid_sweep_tau.csv", index=False)
    freq_df.to_csv(args.out_dir / "grid_sweep_freq.csv", index=False)
    summ_df.to_csv(args.out_dir / "grid_summary.csv", index=False)
    print(f"Wrote 3 CSVs to {args.out_dir}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Heatmap: poly_proxy at tau=0.10 over (layer, step)
        for tau in (0.10, 0.20):
            sub = tau_df[np.isclose(tau_df["tau"], tau)]
            pv = sub.pivot(index="layer", columns="step", values="poly_proxy")
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(pv.values, aspect="auto", cmap="viridis",
                           origin="lower", interpolation="nearest")
            ax.set_xticks(range(len(pv.columns)))
            ax.set_xticklabels([f"{c//1000}k" for c in pv.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(pv.index)))
            ax.set_yticklabels(pv.index)
            ax.set_xlabel("training step")
            ax.set_ylabel("HyenaDNA layer")
            ax.set_title(f"Polysemanticity proxy (tau={tau}) over layer x step")
            fig.colorbar(im, ax=ax, label="mean # concepts / feature")
            fig.tight_layout()
            fig.savefig(args.out_dir / f"heatmap_poly_tau{tau:.2f}.png", dpi=150)
            plt.close(fig)

        # Trajectory: poly_proxy vs step, one line per layer
        for tau in (0.10, 0.20):
            sub = tau_df[np.isclose(tau_df["tau"], tau)]
            fig, ax = plt.subplots(figsize=(7, 4))
            for layer, g in sub.groupby("layer"):
                g = g.sort_values("step")
                ax.plot(g["step"], g["poly_proxy"], marker="o", label=f"layer {layer}")
            ax.set_xlabel("training step")
            ax.set_ylabel(f"poly proxy @ tau={tau}")
            ax.set_title("Polysemanticity proxy across training")
            ax.legend(ncol=2, fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(args.out_dir / f"trajectory_poly_tau{tau:.2f}.png", dpi=150)
            plt.close(fig)

        # Trajectory: # interpretable features
        sub = tau_df[np.isclose(tau_df["tau"], 0.10)]
        fig, ax = plt.subplots(figsize=(7, 4))
        for layer, g in sub.groupby("layer"):
            g = g.sort_values("step")
            ax.plot(g["step"], g["n_interpretable"], marker="o", label=f"layer {layer}")
        ax.set_xlabel("training step")
        ax.set_ylabel("# features with F1>=0.10 on >=1 concept")
        ax.set_title("Interpretable feature count across training")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "trajectory_n_interpretable.png", dpi=150)
        plt.close(fig)

        # SAE-vs-neuron delta over training
        fig, ax = plt.subplots(figsize=(7, 4))
        for layer, g in summ_df.groupby("layer"):
            g = g.sort_values("step")
            ax.plot(g["step"], g["delta_sae_minus_neuron_mean"],
                    marker="o", label=f"layer {layer}")
        ax.axhline(0.0, color="black", lw=0.7)
        ax.set_xlabel("training step")
        ax.set_ylabel("mean (best SAE F1 - best neuron F1) across concepts")
        ax.set_title("SAE vs raw-neuron concept retrieval gap")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "trajectory_sae_vs_neuron.png", dpi=150)
        plt.close(fig)

        # Final-checkpoint cross-layer comparison
        finals = summ_df.sort_values("step").groupby("layer").tail(1)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = finals["layer"].to_numpy()
        ax.bar(x - 0.2, finals["best_sae_f1_mean"], width=0.4,
               label="SAE (best feature, mean across concepts)")
        ax.bar(x + 0.2, finals["best_neuron_f1_mean"], width=0.4,
               label="Raw neuron (best, mean across concepts)")
        ax.set_xlabel("HyenaDNA layer")
        ax.set_ylabel("mean best F1 across non-degenerate concepts")
        ax.set_title("Final-checkpoint concept retrieval by layer")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(args.out_dir / "final_layer_comparison.png", dpi=150)
        plt.close(fig)

        # Dense-feature sensitivity over layers at final ckpt
        finals_step = freq_df["step"].max()
        sub = freq_df[(freq_df["step"] == finals_step) & (freq_df["tau"] == 0.10)]
        fig, ax = plt.subplots(figsize=(7, 4))
        for layer, g in sub.groupby("layer"):
            g = g.sort_values("freq_thr")
            ax.plot(g["freq_thr"], g["poly_sparse_only"],
                    marker="o", label=f"layer {layer}")
        ax.set_xscale("log")
        ax.set_xlabel("freq threshold defining 'dense'")
        ax.set_ylabel("poly proxy (sparse-only) @ tau=0.10")
        ax.set_title(f"Dense-threshold sensitivity at step {finals_step}")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "final_freq_sensitivity_by_layer.png", dpi=150)
        plt.close(fig)

        print(f"Wrote plots to {args.out_dir}")
    except Exception as e:
        print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
