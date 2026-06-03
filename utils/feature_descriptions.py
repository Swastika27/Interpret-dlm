"""Generate per-feature descriptions for layer 6 SAE.

Joins:
  feature_concept_analysis/<concept>/top_features.csv  - which features are best for which concept
  feature_concept_analysis/<concept>/all_features.csv  - F1 per feature for activation_freq recovery
  activation_concept_assoc/enrichment.csv              - % of a feature's top-50 activations overlapping each concept
  activation_concept_assoc/venn.csv                    - pairwise concept overlap among top activations
  top_activations/top_activations.csv                  - genomic coords of top-activating tokens
  epoch_diagnostics/.../dense_feature_stats.csv        - position/norm correlations for dense features

Outputs a markdown report and a CSV.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEGENERATE = ("clinvar_benign", "clinvar_pathogenic", "encode_CTCF-only")
# encode_CTCF-only also collapses to 2/3 trivial in this run; treat as degenerate.


def feature_freq(fca_root: Path) -> pd.Series:
    """Recover per-feature activation frequency from a single non-degenerate concept."""
    for cdir in sorted(fca_root.iterdir()):
        if cdir.name in DEGENERATE:
            continue
        af = cdir / "all_features.csv"
        if not af.exists():
            continue
        df = pd.read_csv(af).sort_values("feature_idx")
        if len(df) != 16384:
            continue
        # New schema stores the (constant) concept token prevalence as "prevalence";
        # old schema stored per-feature recall mislabelled as "baseline_prevalence".
        prev_col = "prevalence" if "prevalence" in df.columns else "baseline_prevalence"
        prev = float(df[prev_col].iloc[0])
        freq = df["recall_tpr"].to_numpy() * prev + df["fpr"].to_numpy() * (1 - prev)
        return pd.Series(freq, index=df["feature_idx"].astype(int).to_numpy())
    raise RuntimeError("no usable concept in fca_root")


def best_concepts_per_feature(fca_root: Path) -> dict:
    """For each feature, return list of (concept, f1) where it's in concept's top-10."""
    feats: dict[int, list[tuple[str, float]]] = {}
    for cdir in sorted(fca_root.iterdir()):
        if not cdir.is_dir() or cdir.name in DEGENERATE:
            continue
        tf = cdir / "top_features.csv"
        if not tf.exists():
            continue
        df = pd.read_csv(tf)
        # Drop trivial rows (precision=0.5 exactly = always-positive baseline)
        df = df[~np.isclose(df["f1"], 2/3, atol=1e-3)]
        for _, r in df.iterrows():
            feats.setdefault(int(r["feature_idx"]), []).append((cdir.name, float(r["f1"])))
    return feats


def enrichment_topk(enr_row: pd.Series, k: int = 3) -> list[tuple[str, float, int]]:
    """Return top-k concepts by pct_<concept>, with (concept, pct, n)."""
    pct_cols = [c for c in enr_row.index if c.startswith("pct_") and c != "pct_neither"]
    items = []
    for c in pct_cols:
        concept = c[len("pct_"):]
        pct = float(enr_row[c])
        n = int(enr_row.get(f"n_{concept}", 0))
        items.append((concept, pct, n))
    items.sort(key=lambda x: -x[1])
    return items[:k]


def top_examples(top_act: pd.DataFrame, feat: int, k: int = 3) -> list[dict]:
    sub = top_act[top_act["feature_idx"] == feat].head(k)
    return sub[["activation_value", "coord_chrom", "coord_start", "coord_end", "tok_pos"]].to_dict("records")


def describe_feature(feat: int, freq: float, dense_thr: float,
                     fc_best: list[tuple[str, float]],
                     enr_row: pd.Series,
                     top_act: pd.DataFrame,
                     dense_stats: pd.DataFrame) -> dict:
    is_dense = freq > dense_thr
    top_concepts = enrichment_topk(enr_row, 3)
    examples = top_examples(top_act, feat, 3)

    fc_best_sorted = sorted(fc_best, key=lambda x: -x[1]) if fc_best else []
    best_fc = fc_best_sorted[0] if fc_best_sorted else (None, None)

    pos_summary = ""
    if is_dense and not dense_stats.empty:
        row = dense_stats[dense_stats["feature_idx"] == feat]
        if not row.empty:
            row = row.iloc[0]
            pos_summary = (f"corr(tok_pos)={row.corr_tok_pos:+.2f} "
                           f"corr(emb_norm)={row.corr_emb_norm:+.2f} "
                           f"corr(recon_err)={row.corr_recon_err:+.2f}")

    return {
        "feature_idx": feat,
        "freq": freq,
        "category": "DENSE" if is_dense else "sparse",
        "best_F1_concept": best_fc[0],
        "best_F1": best_fc[1],
        "n_concepts_in_top10": len(fc_best),
        "all_top10_concepts": "|".join(c for c, _ in fc_best_sorted),
        "enrich_top1": f"{top_concepts[0][0]}={top_concepts[0][1]:.0f}%" if top_concepts else "",
        "enrich_top2": f"{top_concepts[1][0]}={top_concepts[1][1]:.0f}%" if len(top_concepts) > 1 else "",
        "enrich_top3": f"{top_concepts[2][0]}={top_concepts[2][1]:.0f}%" if len(top_concepts) > 2 else "",
        "pct_neither": float(enr_row.get("pct_neither", 0.0)),
        "top_example": f"{examples[0]['coord_chrom']}:{examples[0]['coord_start']}-{examples[0]['coord_end']} (act={examples[0]['activation_value']:.2f}, tok={examples[0]['tok_pos']})"
            if examples else "",
        "dense_pos_corr": pos_summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path,
                    default=Path("results/layer6_16384_batchtopk_64_0.0003/step8000000"))
    ap.add_argument("--dense_thr", type=float, default=0.04)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("results/layer6_16384_batchtopk_64_0.0003/step8000000/feature_descriptions"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fca_root = args.results_root / "feature_concept_analysis"
    print(f"Loading inputs from {args.results_root}...")
    freq = feature_freq(fca_root)
    fc = best_concepts_per_feature(fca_root)

    enr = pd.read_csv(args.results_root / "activation_concept_assoc" / "enrichment.csv").set_index("feature_idx")
    top_act = pd.read_csv(args.results_root / "top_activations" / "top_activations.csv")
    diag = args.results_root / "epoch_diagnostics" / "step_8000000" / "dense_feature_stats.csv"
    dense_stats = pd.read_csv(diag) if diag.exists() else pd.DataFrame()

    feat_set = sorted(fc.keys())
    print(f"  {len(feat_set)} features appear in some concept's top-10 (after dropping degenerate concepts)")

    rows = [describe_feature(f, float(freq[f]), args.dense_thr,
                             fc[f], enr.loc[f], top_act, dense_stats)
            for f in feat_set]
    df = pd.DataFrame(rows).sort_values(["category", "best_F1"], ascending=[True, False])
    df.to_csv(args.out_dir / "all_top10_features.csv", index=False)

    sparse = df[df["category"] == "sparse"].copy()
    sparse.to_csv(args.out_dir / "sparse_features.csv", index=False)
    dense = df[df["category"] == "DENSE"].copy()
    dense.to_csv(args.out_dir / "dense_features.csv", index=False)

    # --- Markdown report ---
    md_lines = [
        "# Layer 6 SAE — feature descriptions",
        "",
        f"Source: `{args.results_root}`",
        f"Dense threshold: freq > **{args.dense_thr}** (≈10× the SAE's design sparsity of 64/16384 = 0.0039).",
        "",
        f"Universe: **{len(feat_set)} unique features** that appear in at least one concept's "
        f"`top_features.csv` (after excluding concepts that collapse to F1=2/3: clinvar_benign, "
        f"clinvar_pathogenic, encode_CTCF-only).",
        f"- **Sparse** (freq ≤ {args.dense_thr}): {len(sparse)} features",
        f"- **Dense**  (freq >  {args.dense_thr}): {len(dense)} features",
        "",
        "Columns:",
        "- `best_F1_concept`/`best_F1`: highest F1 the feature achieves on any non-degenerate concept",
        "- `n_concepts_in_top10`: how many concepts list this feature in their top-10 (polysemanticity)",
        "- `enrich_top1..3`: among this feature's top-50 activating tokens, % overlapping each concept (from `enrichment.csv`)",
        "- `pct_neither`: % of top-50 activations falling outside *every* concept (un-annotated regions)",
        "- `top_example`: highest-activating token (genomic coord)",
        "- `dense_pos_corr`: only for dense features — correlation with token position / embedding norm / reconstruction error",
        "",
        "---",
        "",
        "## Sparse features (the interpretable, non-positional set)",
        "",
    ]
    for _, r in sparse.iterrows():
        md_lines.append(_format_feature_block(r))

    md_lines += ["", "---", "", "## Dense features (excluded at freq > %.2f)" % args.dense_thr, ""]
    for _, r in dense.iterrows():
        md_lines.append(_format_feature_block(r))

    (args.out_dir / "feature_descriptions.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {args.out_dir/'feature_descriptions.md'} ({len(md_lines)} lines)")
    print(f"Sparse features: {len(sparse)}  /  Dense features: {len(dense)}")


def _format_feature_block(r: pd.Series) -> str:
    lines = [
        f"### feature {int(r['feature_idx'])} — {r['category']} (freq={r['freq']:.4f})",
        f"- Best concept by F1: **{r['best_F1_concept']}** (F1={r['best_F1']:.3f})",
        f"- Also in top-10 of: {r['all_top10_concepts']}  *(n={r['n_concepts_in_top10']})*",
        f"- Top-50 activations enriched in: {r['enrich_top1']}, {r['enrich_top2']}, {r['enrich_top3']}  "
        f"(unannotated: {r['pct_neither']:.0f}%)",
        f"- Top example: `{r['top_example']}`",
    ]
    if r["dense_pos_corr"]:
        lines.append(f"- Positional signature: {r['dense_pos_corr']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()