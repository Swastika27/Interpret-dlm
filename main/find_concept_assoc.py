"""
concept_feature_analysis.py

For each concept BED file, label tokens inside BED intervals as positive and
sample an equal number of negatives. For each SAE feature, compute F1 and the
full confusion matrix. Report the top-10 features per concept.

Usage:
    python concept_feature_analysis.py \
        --sae_checkpoint  runs/my_run/checkpoint_step10000.pt \
        --sae_cfg         runs/my_run/cfg.json \
        --save_dir        /data/embeddings \
        --layer           2 \
        --splits          train val test \
        --bed_dir         concepts/ \
        --out_dir         results/concept_analysis \
        --device          cuda \
        --batch_size      2048 \
        --top_k_features  10 \
        --seed            42

BED file format expected (tab-separated, 0-based half-open):
    chrom   chromStart   chromEnd   [name   score   strand   ...]

Output layout:
    out_dir/
        <concept_name>/
            top_features.csv     – top-10 features ranked by F1
            all_features.csv     – full stats for every feature
        summary.csv              – best feature per concept across all concepts
"""

import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from BatchTopK.sae import BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE, JumpReLUInferenceSAE

def restore_cfg_types(cfg):
    if isinstance(cfg.get("dtype"), str):
        cfg["dtype"] = getattr(torch, cfg["dtype"].replace("torch.", ""))
    if isinstance(cfg.get("device"), str):
        cfg["device"] = torch.device(cfg["device"])
    return cfg

# ---------------------------------------------------------------------------
# BED loading & interval lookup
# ---------------------------------------------------------------------------

class BEDIndex:
    """
    Load a BED file and answer point-in-interval queries efficiently.
    Intervals are 0-based, half-open: [chromStart, chromEnd).
    """
    def __init__(self, bed_path: str):
        self.name = Path(bed_path).stem
        # chrom -> sorted list of (start, end)
        self._intervals: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._sorted: Dict[str, bool] = {}
        self._load(bed_path)

    def _load(self, path: str):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    parts = line.split()
                chrom = parts[0]
                start = int(parts[1])
                end   = int(parts[2])
                self._intervals[chrom].append((start, end))
        # Sort each chrom's intervals by start for binary search
        for chrom in self._intervals:
            self._intervals[chrom].sort()
        total = sum(len(v) for v in self._intervals.values())
        print(f"  Loaded BED '{self.name}': {total} intervals across {len(self._intervals)} chroms")

    def contains(self, chrom: str, pos: int) -> bool:
        """Return True if pos falls within any interval on chrom (0-based)."""
        ivs = self._intervals.get(chrom)
        if not ivs:
            return False
        # Binary search for last interval with start <= pos
        lo, hi = 0, len(ivs) - 1
        idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if ivs[mid][0] <= pos:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if idx == -1:
            return False
        # Check all overlapping intervals (handles nested / overlapping intervals)
        # Walk back while starts could still contain pos (rare but safe)
        while idx >= 0 and ivs[idx][0] <= pos:
            if ivs[idx][1] > pos:
                return True
            idx -= 1
        return False


# ---------------------------------------------------------------------------
# SAE loading (same as find_top_activations.py)
# ---------------------------------------------------------------------------

def load_sae(cfg: dict, checkpoint_path: str, device: str):
    state     = torch.load(checkpoint_path, map_location=device)
    sae_state = state.get("sae_state_dict") or state.get("model_state_dict") or state
    saved_cfg = state.get("cfg", cfg)
    theta     = state.get("theta") or saved_cfg.get("theta")

    arch = cfg.get("architecture", "batchtopk").lower()
    cls_map = {
        "batchtopk": BatchTopKSAE,
        "top_k":       TopKSAE,
        "vanilla":     VanillaSAE,
        "jumprelu":    JumpReLUSAE,
    }
    sae = cls_map[arch](cfg)
    sae.load_state_dict(sae_state, strict=False)

    if arch == "batchtopk":
        if theta is None:
            raise ValueError("BatchTopKSAE checkpoint has no 'theta'.")
        print(f"Wrapping BatchTopKSAE with JumpReLUInferenceSAE (theta={theta:.6f})")
        sae = JumpReLUInferenceSAE(sae, theta=theta)

    sae.eval().to(device)
    return sae


@torch.no_grad()
def get_activations(sae, x: torch.Tensor) -> torch.Tensor:
    """x: (N, D) → acts: (N, dict_size) on CPU."""
    dtype  = next(sae.parameters()).dtype
    device = next(sae.parameters()).device
    x = x.to(device).to(dtype)

    if isinstance(sae, JumpReLUInferenceSAE):
        _, acts = sae(x)
        return acts.float().cpu()

    x, x_mean, x_std = sae.preprocess_input(x)
    x_cent = x - sae.b_dec
    acts   = F.relu(x_cent @ sae.W_enc)
    if hasattr(sae, 'jumprelu'):
        acts = sae.jumprelu(acts)
    elif 'top_k' in sae.cfg:
        k    = sae.cfg['top_k']
        topk = torch.topk(acts, min(k, acts.shape[-1]), dim=-1)
        acts = torch.zeros_like(acts).scatter(-1, topk.indices, topk.values)
    return acts.float().cpu()


# ---------------------------------------------------------------------------
# Collect labels + activations across all shards
# ---------------------------------------------------------------------------

def collect_labeled_activations(
    sae,
    shard_paths: List[str],
    split_name: str,
    bed_indices: List[BEDIndex],
    act_size: int,
    batch_size: int,
    device: str,
) -> Tuple[
    np.ndarray,          # acts:   (N_total, dict_size)  float32
    np.ndarray,          # labels: (N_total, n_concepts) bool
]:
    """
    Stream shards, run SAE encoder, and assign per-concept binary labels.
    Returns arrays for ALL tokens (subsampling is done later per-concept).
    """
    n_concepts = len(bed_indices)
    all_acts_list   = []
    all_labels_list = []

    for shard_path in tqdm(shard_paths, desc=f"  {split_name}", leave=False):
        shard      = torch.load(shard_path, map_location="cpu")
        emb        = shard["emb"]           # (B, L, D)
        coords_raw = shard["coords"]        # List[Tuple[chrom, start, end]], length B

        B, L, D = emb.shape
        assert D == act_size

        # Flatten embeddings
        emb_flat = emb.reshape(B * L, D)

        # Compute per-token labels for all concepts
        # label[i * L + j] = True if token j of sequence i is positive for concept c
        token_labels = np.zeros((B * L, n_concepts), dtype=bool)
        for seq_i, coord in enumerate(coords_raw):
            chrom, seq_start, seq_end = coord
            seq_len_actual = L
            # Each token covers (seq_end - seq_start) / L bases
            bp_per_token = (seq_end - seq_start) / seq_len_actual
            for tok_j in range(seq_len_actual):
                tok_mid = int(seq_start + (tok_j + 0.5) * bp_per_token)
                flat_idx = seq_i * L + tok_j
                for ci, bed in enumerate(bed_indices):
                    token_labels[flat_idx, ci] = bed.contains(chrom, tok_mid)

        # Run SAE in sub-batches
        acts_chunks = []
        for start in range(0, B * L, batch_size):
            end   = min(start + batch_size, B * L)
            chunk = emb_flat[start:end]
            acts_chunks.append(get_activations(sae, chunk).numpy())
        acts_np = np.concatenate(acts_chunks, axis=0)  # (B*L, dict_size)

        all_acts_list.append(acts_np)
        all_labels_list.append(token_labels)

    all_acts   = np.concatenate(all_acts_list,   axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)
    return all_acts, all_labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_for_concept(
    acts: np.ndarray,         # (N, dict_size)
    labels: np.ndarray,       # (N,) bool
    rng: np.random.Generator,
    top_k_features: int,
) -> List[dict]:
    """
    For a single concept:
    1. Subsample negatives to match number of positives.
    2. Binarise each feature (active = activation > 0).
    3. Compute TP, TN, FP, FN, precision, recall, F1, baseline prevalence.
    4. Return stats for ALL features sorted by F1 descending.
    """
    pos_idx = np.where( labels)[0]
    neg_idx = np.where(~labels)[0]
    n_pos   = len(pos_idx)

    if n_pos == 0:
        return []

    # Subsample negatives
    if len(neg_idx) > n_pos:
        neg_idx = rng.choice(neg_idx, size=n_pos, replace=False)

    idx_balanced = np.concatenate([pos_idx, neg_idx])
    y_true = np.concatenate([np.ones(n_pos, dtype=bool),
                             np.zeros(len(neg_idx), dtype=bool)])

    acts_bal = acts[idx_balanced]          # (2*n_pos, dict_size)
    y_pred   = acts_bal > 0                # binary: active or not

    # Vectorised confusion matrix over all features at once
    TP = ( y_pred &  y_true[:, None]).sum(axis=0).astype(np.float64)
    TN = (~y_pred & ~y_true[:, None]).sum(axis=0).astype(np.float64)
    FP = ( y_pred & ~y_true[:, None]).sum(axis=0).astype(np.float64)
    FN = (~y_pred &  y_true[:, None]).sum(axis=0).astype(np.float64)

    N = len(y_true)
    precision = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
    recall    = np.where(TP + FN > 0, TP / (TP + FN), 0.0)
    f1        = np.where(precision + recall > 0,
                         2 * precision * recall / (precision + recall), 0.0)

    # Baseline prevalence = fraction of ALL (unbalanced) tokens where feature is active
    baseline_prevalence = (acts > 0).mean(axis=0)  # (dict_size,)

    # TPR / TNR / FPR / FNR
    tpr = recall                                                        # TP / (TP + FN)
    tnr = np.where(TN + FP > 0, TN / (TN + FP), 0.0)                 # TN / (TN + FP)
    fpr = np.where(FP + TN > 0, FP / (FP + TN), 0.0)                 # FP / (FP + TN)
    fnr = np.where(FN + TP > 0, FN / (FN + TP), 0.0)                 # FN / (FN + TP)

    dict_size = acts.shape[1]
    rows = []
    for fi in range(dict_size):
        rows.append({
            "feature_idx":         fi,
            "f1":                  float(f1[fi]),
            "precision":           float(precision[fi]),
            "recall_tpr":          float(tpr[fi]),
            "tnr":                 float(tnr[fi]),
            "fpr":                 float(fpr[fi]),
            "fnr":                 float(fnr[fi]),
            "tp":                  int(TP[fi]),
            "tn":                  int(TN[fi]),
            "fp":                  int(FP[fi]),
            "fn":                  int(FN[fi]),
            "n_positive_tokens":   n_pos,
            "n_negative_tokens":   int(len(neg_idx)),
            "baseline_prevalence": float(baseline_prevalence[fi]),
        })

    rows.sort(key=lambda r: r["f1"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

FEATURE_CSV_HEADER = [
    "feature_idx", "f1", "precision", "recall_tpr",
    "tnr", "fpr", "fnr",
    "tp", "tn", "fp", "fn",
    "n_positive_tokens", "n_negative_tokens",
    "baseline_prevalence",
]

def write_feature_csv(path: str, rows: List[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: f"{row[k]:.6f}" if isinstance(row[k], float) else row[k]
                        for k in FEATURE_CSV_HEADER})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sae_checkpoint", required=True)
    p.add_argument("--sae_cfg",        required=True)
    p.add_argument("--save_dir",       required=True,
                   help="Root dir with train/val/test splits of shards")
    p.add_argument("--layer",          type=int, required=True)
    p.add_argument("--splits",         nargs="+", default=["train", "val", "test"])
    p.add_argument("--bed_dir",        required=True,
                   help="Directory containing *.bed concept annotation files")
    p.add_argument("--out_dir",        required=True)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size",     type=int, default=2048)
    p.add_argument("--top_k_features", type=int, default=10)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # Load config & SAE
    with open(args.sae_cfg) as f:
        cfg = json.load(f)
    cfg = restore_cfg_types(cfg)
    cfg["device"] = args.device
    print(f"Loading SAE ...")
    sae      = load_sae(cfg, args.sae_checkpoint, args.device)
    act_size = cfg["act_size"]

    # Load all BED files
    bed_paths = sorted(glob.glob(os.path.join(args.bed_dir, "*.bed")))
    if not bed_paths:
        raise FileNotFoundError(f"No .bed files found in {args.bed_dir}")
    print(f"\nFound {len(bed_paths)} BED concept files:")
    bed_indices = [BEDIndex(p) for p in bed_paths]

    # Collect activations + labels across all splits
    all_acts_list   = []
    all_labels_list = []

    for split in args.splits:
        layer_dir   = os.path.join(args.save_dir, split, f"layer_{args.layer}")
        shard_paths = sorted(glob.glob(os.path.join(layer_dir, "shard_*.pt")))
        if not shard_paths:
            print(f"  [WARNING] No shards in {layer_dir}, skipping.")
            continue
        print(f"\nCollecting {split}: {len(shard_paths)} shards")
        acts, labels = collect_labeled_activations(
            sae         = sae,
            shard_paths = shard_paths,
            split_name  = split,
            bed_indices = bed_indices,
            act_size    = act_size,
            batch_size  = args.batch_size,
            device      = args.device,
        )
        all_acts_list.append(acts)
        all_labels_list.append(labels)

    all_acts   = np.concatenate(all_acts_list,   axis=0)  # (N_total, dict_size)
    all_labels = np.concatenate(all_labels_list, axis=0)  # (N_total, n_concepts)

    print(f"\nTotal tokens collected: {all_acts.shape[0]:,}")
    for ci, bed in enumerate(bed_indices):
        n_pos = all_labels[:, ci].sum()
        print(f"  {bed.name}: {n_pos:,} positive tokens "
              f"({100*n_pos/len(all_labels):.2f}% prevalence)")

    # Analyse each concept
    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []

    for ci, bed in enumerate(bed_indices):
        print(f"\nAnalysing concept '{bed.name}' ...")
        labels_ci = all_labels[:, ci]
        rows = compute_metrics_for_concept(all_acts, labels_ci, rng, args.top_k_features)

        if not rows:
            print(f"  [WARNING] No positive tokens for '{bed.name}', skipping.")
            continue

        concept_dir = os.path.join(args.out_dir, bed.name)
        os.makedirs(concept_dir, exist_ok=True)

        # All features
        write_feature_csv(os.path.join(concept_dir, "all_features.csv"), rows)

        # Top-k features
        top_rows = rows[:args.top_k_features]
        write_feature_csv(os.path.join(concept_dir, "top_features.csv"), top_rows)

        # Print table
        print(f"\n  Top {args.top_k_features} features for '{bed.name}':")
        print(f"  {'feat':>6}  {'F1':>6}  {'Prec':>6}  {'Rec/TPR':>8}  "
              f"{'TNR':>6}  {'FPR':>6}  {'FNR':>6}  "
              f"{'TP':>6}  {'TN':>6}  {'FP':>6}  {'FN':>6}  {'BasePrevalence':>14}")
        print("  " + "-" * 105)
        for r in top_rows:
            print(f"  {r['feature_idx']:>6}  {r['f1']:>6.3f}  {r['precision']:>6.3f}  "
                  f"{r['recall_tpr']:>8.3f}  {r['tnr']:>6.3f}  {r['fpr']:>6.3f}  "
                  f"{r['fnr']:>6.3f}  {r['tp']:>6}  {r['tn']:>6}  "
                  f"{r['fp']:>6}  {r['fn']:>6}  {r['baseline_prevalence']:>14.4f}")

        # Summary: best feature per concept
        best = top_rows[0]
        summary_rows.append({
            "concept":             bed.name,
            "best_feature_idx":    best["feature_idx"],
            "f1":                  best["f1"],
            "precision":           best["precision"],
            "recall_tpr":          best["recall_tpr"],
            "tnr":                 best["tnr"],
            "fpr":                 best["fpr"],
            "fnr":                 best["fnr"],
            "tp":                  best["tp"],
            "tn":                  best["tn"],
            "fp":                  best["fp"],
            "fn":                  best["fn"],
            "n_positive_tokens":   best["n_positive_tokens"],
            "baseline_prevalence": best["baseline_prevalence"],
        })

    # Write summary CSV
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_header = [
        "concept", "best_feature_idx", "f1", "precision", "recall_tpr",
        "tnr", "fpr", "fnr", "tp", "tn", "fp", "fn",
        "n_positive_tokens", "baseline_prevalence",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_header)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: f"{row[k]:.6f}" if isinstance(row[k], float) else row[k]
                        for k in summary_header})

    print(f"\nDone. Results saved to {args.out_dir}/")
    print(f"  summary.csv                     – best feature per concept")
    print(f"  <concept>/top_features.csv      – top {args.top_k_features} features per concept")
    print(f"  <concept>/all_features.csv      – full ranked feature list")


if __name__ == "__main__":
    main()