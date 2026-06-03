"""
recompute_metrics.py — recompute clean MCC / precision / recall / F1 / enrichment
from the association counts you ALREADY have, without re-running the GPU pass.

How it works
------------
The existing all_features.csv stores, per feature, two BALANCING-INVARIANT rates:
    recall_tpr = TP / n_pos = P(fire | positive)
    fpr        = FP / n_neg = P(fire | negative)      <- raw, NOT the balanced FP
together with n_positive_tokens (n_pos) and n_negative_tokens (n_neg). From those we
reconstruct the exact raw confusion matrix:
    TP = round(recall_tpr * n_pos)   (equals the stored `tp` column)
    FP = round(fpr        * n_neg)
    FN = n_pos - TP ;  TN = n_neg - FP
and feed them through utils/assoc_metrics.compute_raw_metrics — the SAME function the
updated concept_feature_analysis.py uses, so the output schema matches new runs exactly.
(Only limitation: fpr is stored to 6 decimals, so FP for very rare concepts has a tiny
rounding error; re-run concept_feature_analysis.py for bit-exact values.)

Default behaviour
-----------------
For each results/<layer_tag>: recompute the SAE features at that layer's BEST epoch
(fewest dead features) and the raw-neuron baseline, writing sibling dirs:
    .../step<best>/feature_concept_analysis_mcc/<concept>/{all,top}_features.csv + summary.csv
    .../neuron_concept_analysis_mcc/<concept>/...                      + summary.csv
and prints a per-layer SAE-vs-neuron comparison ranked by MCC.

Examples
--------
  python utils/recompute_metrics.py                       # all layers, best epoch, SAE + neurons
  python utils/recompute_metrics.py --layers 4 5 6 --all_steps
  python utils/recompute_metrics.py --rank_by enrichment
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from assoc_metrics import (  # noqa: E402
    CANONICAL_FEATURE_COLUMNS,
    CANONICAL_SUMMARY_COLUMNS,
    compute_raw_metrics,
    rank_order,
    format_value,
)
from run_null_sweep import find_best_step, layer_of  # noqa: E402


def read_existing_all_features(path: str):
    """Return (feature_idx[], pos_acts[], neg_acts[], n_pos, n_neg) reconstructed raw."""
    fidx, recall, fpr, tp_col = [], [], [], []
    n_pos = n_neg = None
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            fidx.append(int(float(r["feature_idx"])))
            recall.append(float(r["recall_tpr"]))
            fpr.append(float(r["fpr"]))
            tp_col.append(float(r["tp"]))
            if n_pos is None:
                n_pos = int(float(r["n_positive_tokens"]))
                # older CSVs always include n_negative_tokens in all_features.csv
                n_neg = int(float(r.get("n_negative_tokens", 0)))
    if not fidx or n_neg in (None, 0):
        return None
    fidx = np.array(fidx, dtype=np.int64)
    # raw TP is stored exactly in `tp`; raw FP reconstructed from the raw rate fpr
    pos_acts = np.rint(np.array(tp_col, dtype=np.float64)).astype(np.int64)
    neg_acts = np.rint(np.array(fpr, dtype=np.float64) * n_neg).astype(np.int64)
    return fidx, pos_acts, neg_acts, n_pos, n_neg


def build_rows(fidx, pos_acts, neg_acts, n_pos, n_neg, rank_by):
    m = compute_raw_metrics(pos_acts, neg_acts, n_pos, n_neg)
    prevalence = n_pos / (n_pos + n_neg)
    order = rank_order(m, key=rank_by)
    rows = []
    for j in order:
        row = {"feature_idx": int(fidx[j]),
               "n_positive_tokens": int(n_pos),
               "n_negative_tokens": int(n_neg),
               "prevalence": prevalence}
        for k in m:
            row[k] = m[k][j]
        rows.append(row)
    return rows


def write_feature_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CANONICAL_FEATURE_COLUMNS)
        for r in rows:
            w.writerow([format_value(k, r[k]) for k in CANONICAL_FEATURE_COLUMNS])


def summary_row_from_best(concept, best):
    out = {"concept": concept, "best_feature_idx": best["feature_idx"]}
    for k in CANONICAL_SUMMARY_COLUMNS:
        if k in ("concept", "best_feature_idx"):
            continue
        out[k] = best[k]
    return out


def write_summary_csv(path, summary_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CANONICAL_SUMMARY_COLUMNS)
        for r in summary_rows:
            w.writerow([format_value(k, r[k]) for k in CANONICAL_SUMMARY_COLUMNS])


def recompute_tree(src_dir: str, dst_dir: str, rank_by: str) -> Dict[str, dict]:
    """Recompute every <concept>/all_features.csv under src_dir into dst_dir.
    Returns {concept: best_row}."""
    best_by_concept: Dict[str, dict] = {}
    summary_rows = []
    if not os.path.isdir(src_dir):
        return best_by_concept
    for concept in sorted(os.listdir(src_dir)):
        cdir = os.path.join(src_dir, concept)
        af = os.path.join(cdir, "all_features.csv")
        if not os.path.isfile(af):
            continue
        parsed = read_existing_all_features(af)
        if parsed is None:
            print(f"    [skip] {concept}: missing n_negative_tokens or empty")
            continue
        rows = build_rows(*parsed, rank_by=rank_by)
        out_cdir = os.path.join(dst_dir, concept)
        write_feature_csv(os.path.join(out_cdir, "all_features.csv"), rows)
        write_feature_csv(os.path.join(out_cdir, "top_features.csv"), rows[:10])
        best_by_concept[concept] = rows[0]
        summary_rows.append(summary_row_from_best(concept, rows[0]))
    if summary_rows:
        write_summary_csv(os.path.join(dst_dir, "summary.csv"), summary_rows)
    return best_by_concept


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", default=os.path.join(REPO, "results"))
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--all_steps", action="store_true",
                   help="Recompute every step dir (default: best epoch per layer only)")
    p.add_argument("--rank_by", default="mcc",
                   choices=["mcc", "enrichment", "f1", "precision", "balanced_f1"])
    p.add_argument("--sae_subdir", default="feature_concept_analysis")
    p.add_argument("--out_suffix", default="_mcc")
    p.add_argument("--skip_neurons", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    tags = []
    for name in sorted(os.listdir(a.results_root)):
        if re.match(r"layer\d+_.*batchtopk", name) and \
                os.path.isdir(os.path.join(a.results_root, name)):
            L = layer_of(name)
            if a.layers and L not in a.layers:
                continue
            tags.append((L, name))
    tags.sort()

    for L, tag in tags:
        ldir = os.path.join(a.results_root, tag)
        if a.all_steps:
            steps = sorted(int(re.search(r"step(\d+)", d).group(1))
                           for d in os.listdir(ldir) if re.fullmatch(r"step\d+", d))
        else:
            best, _, _ = find_best_step(ldir)
            steps = [best] if best is not None else []

        # neuron baseline (epoch-independent, per layer)
        neuron_best = {}
        if not a.skip_neurons:
            nsrc = os.path.join(ldir, "neuron_concept_analysis")
            ndst = os.path.join(ldir, "neuron_concept_analysis" + a.out_suffix)
            neuron_best = recompute_tree(nsrc, ndst, a.rank_by)

        for step in steps:
            src = os.path.join(ldir, f"step{step}", a.sae_subdir)
            dst = os.path.join(ldir, f"step{step}", a.sae_subdir + a.out_suffix)
            best = recompute_tree(src, dst, a.rank_by)
            if not best:
                continue
            print(f"\n=== L{L} step{step} (ranked by {a.rank_by}) ===  -> {os.path.relpath(dst, a.results_root)}")
            print(f"{'concept':28s} {'feat':>6s} {'MCC':>7s} {'enrich':>7s} "
                  f"{'prec':>6s} {'recall':>6s} {'f1':>6s} {'balF1':>6s} {'prev%':>6s}"
                  + ("   | nMCC nEnrich" if neuron_best else ""))
            print("-" * (96 if neuron_best else 78))
            for concept in sorted(best, key=lambda c: -float(best[c]["mcc"])):
                b = best[concept]
                enr = b["enrichment"]
                enr_s = f"{enr:7.2f}" if np.isfinite(enr) else "    inf"
                line = (f"{concept[:28]:28s} {b['feature_idx']:>6d} {b['mcc']:>7.3f} {enr_s} "
                        f"{b['precision']:>6.3f} {b['recall_tpr']:>6.3f} {b['f1']:>6.3f} "
                        f"{b['balanced_f1']:>6.3f} {b['prevalence']*100:>6.2f}")
                if neuron_best and concept in neuron_best:
                    nb = neuron_best[concept]
                    nenr = nb["enrichment"]
                    nenr_s = f"{nenr:6.2f}" if np.isfinite(nenr) else "   inf"
                    line += f"   | {nb['mcc']:>5.3f} {nenr_s}"
                print(line)

    print("\nDone. Recomputed CSVs written alongside originals with suffix "
          f"'{a.out_suffix}'. Schema == utils/assoc_metrics.CANONICAL_FEATURE_COLUMNS.")


if __name__ == "__main__":
    main()
