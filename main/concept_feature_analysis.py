"""
concept_feature_analysis.py  (optimized + streaming)

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

    Raw neurons (same pipeline; features = embedding dimensions, ReLU then acts > 0):

    python concept_feature_analysis.py --raw_neurons --sae_cfg runs/my_run/cfg.json ...

BED file format expected (tab-separated, 0-based half-open):
    chrom   chromStart   chromEnd   [name   score   strand   ...]

Output layout:
    out_dir/
        <concept_name>/
            top_features.csv     – top-10 features ranked by F1
            all_features.csv     – full stats for every feature
        summary.csv              – best feature per concept across all concepts

Memory model
------------
The original script accumulated all shard activations (float32) in RAM before
any analysis, which caused silent OOM kills on large datasets.

This version streams shards one at a time:
  1. Run the SAE encoder on the shard.
  2. Binarise activations immediately  (active = acts > 0)  — 4× smaller.
  3. Accumulate per-concept TP/FP/FN/TN *counts* into two small int64 matrices
     of shape (n_concepts, dict_size).
  4. Discard float activations and the bool matrix before loading the next shard.

Peak RAM is now O(n_concepts × dict_size) ≈ a few MB, independent of dataset
size, instead of O(N_tokens × dict_size).

Gated SAE: set "sae_type": "gated" in the SAE config JSON from training.

Because we never materialise all negatives simultaneously we correct for the
positive/negative class imbalance analytically in compute_metrics_from_counts()
by scaling the accumulated negative counts down to match n_pos.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from SAE_training.sae import (
    BatchTopKSAE,
    TopKSAE,
    VanillaSAE,
    JumpReLUSAE,
    JumpReLUInferenceSAE,
    GatedSAE,
    GatedInferenceSAE,
)


class RawNeuronSAE(torch.nn.Module):
    """
    Drop-in replacement for a trained SAE that returns raw model activations.
    The "features" are the neuron dimensions themselves (size = act_size).
    Compatible with get_activations() since it does not use preprocess_input
    or W_enc — the dedicated branch returns the input (after ReLU) as acts.
    """
    def __init__(self, act_size: int):
        super().__init__()
        self.act_size  = act_size
        self.dict_size = act_size
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return F.relu(x)


def restore_cfg_types(cfg: dict) -> dict:
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

    Storage: two numpy int64 arrays per chrom (starts, ends), sorted by start.

    Scalar query  – contains(chrom, pos)
    Batch query   – contains_batch(chrom, positions[])   <- hot path
    """

    def __init__(self, bed_path: str):
        self.name = Path(bed_path).stem
        self._starts: Dict[str, np.ndarray] = {}
        self._ends:   Dict[str, np.ndarray] = {}
        self._load(bed_path)

    def _load(self, path: str):
        raw: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(("#", "track", "browser")):
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                raw[parts[0]].append((int(parts[1]), int(parts[2])))

        total = 0
        for chrom, ivs in raw.items():
            ivs.sort()
            arr = np.array(ivs, dtype=np.int64)   # (M, 2)
            self._starts[chrom] = arr[:, 0]
            self._ends[chrom]   = arr[:, 1]
            total += len(ivs)
        print(f"  Loaded BED '{self.name}': {total} intervals "
              f"across {len(self._starts)} chroms")

    # ------------------------------------------------------------------
    # Scalar query
    # ------------------------------------------------------------------
    def contains(self, chrom: str, pos: int) -> bool:
        starts = self._starts.get(chrom)
        if starts is None:
            return False
        idx = int(np.searchsorted(starts, pos, side="right")) - 1
        if idx < 0:
            return False
        ends = self._ends[chrom]
        # Forward walk covers overlapping / abutting intervals (rare but correct)
        while idx < len(starts) and starts[idx] <= pos:
            if ends[idx] > pos:
                return True
            idx += 1
        return False

    # ------------------------------------------------------------------
    # Vectorised batch query  <- used in the labelling hot-path
    # ------------------------------------------------------------------
    def contains_batch(self, chrom: str, positions: np.ndarray) -> np.ndarray:
        """
        positions : 1-D int64 array of genomic positions (0-based).
        Returns   : bool array of the same length.
        """
        starts = self._starts.get(chrom)
        if starts is None:
            return np.zeros(len(positions), dtype=bool)

        ends   = self._ends[chrom]
        # For each position: index of the last interval whose start <= pos
        idx    = np.searchsorted(starts, positions, side="right") - 1   # (N,)
        result = np.zeros(len(positions), dtype=bool)

        valid = idx >= 0
        vi    = np.where(valid)[0]
        if vi.size == 0:
            return result

        # Primary hit: candidate interval end > pos
        candidate_ends = ends[idx[vi]]
        hit            = candidate_ends > positions[vi]
        result[vi[hit]] = True

        # Secondary pass for overlapping intervals (uncommon)
        maybe = vi[~hit]
        if maybe.size:
            next_idx = idx[maybe] + 1
            in_range = next_idx < len(starts)
            mr = maybe[in_range]
            ni = next_idx[in_range]
            more = (starts[ni] <= positions[mr]) & (ends[ni] > positions[mr])
            result[mr[more]] = True

        return result


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

def load_sae(cfg: dict, checkpoint_path: str, device: str):
    state     = torch.load(checkpoint_path, map_location=device)
    sae_state = state.get("state_dict") or state.get("model_state_dict") or state
    saved_cfg = state.get("cfg", cfg)
    theta     = state.get("theta") or saved_cfg.get("theta")

    arch    = cfg.get("sae_type", "batchtopk").lower()
    cls_map = {
        "batchtopk": BatchTopKSAE,
        "top_k":     TopKSAE,
        "vanilla":   VanillaSAE,
        "jumprelu":  JumpReLUSAE,
        "gated":     GatedSAE,
    }
    sae = cls_map[arch](cfg)
    sae.load_state_dict(sae_state, strict=False)

    if arch == "batchtopk":
        if theta is None:
            raise ValueError("BatchTopKSAE checkpoint has no 'theta'.")
        print(f"Wrapping BatchTopKSAE with JumpReLUInferenceSAE (theta={theta:.6f})")
        sae = JumpReLUInferenceSAE(sae, theta=theta)
    elif arch == "gated":
        print("Wrapping GatedSAE with GatedInferenceSAE")
        sae = GatedInferenceSAE(sae)

    sae.eval().to(device)
    return sae


@torch.no_grad()
def get_activations(sae, x: torch.Tensor) -> torch.Tensor:
    """x : (N, D)  ->  acts : (N, dict_size) float32 on CPU."""
    dtype  = next(sae.parameters()).dtype
    device = next(sae.parameters()).device
    x = x.to(device=device, dtype=dtype)

    if isinstance(sae, RawNeuronSAE):
        return F.relu(x).float().cpu()

    if isinstance(sae, (JumpReLUInferenceSAE, GatedInferenceSAE)):
        _, acts = sae(x)
        return acts.float().cpu()

    x, _, _ = sae.preprocess_input(x)
    x_cent  = x - sae.b_dec
    acts    = F.relu(x_cent @ sae.W_enc)

    if hasattr(sae, "jumprelu"):
        acts = sae.jumprelu(acts)
    elif "top_k" in sae.cfg:
        k    = sae.cfg["top_k"]
        topk = torch.topk(acts, min(k, acts.shape[-1]), dim=-1)
        acts = torch.zeros_like(acts).scatter_(-1, topk.indices, topk.values)

    return acts.float().cpu()


# ---------------------------------------------------------------------------
# Token position helper
# ---------------------------------------------------------------------------

def _build_token_positions(
    coords_raw: list,
    L: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Pre-compute the genomic mid-point for every (sequence, token) pair.

    Returns
    -------
    chroms    : list[str], length B*L
    positions : int64 ndarray, shape (B*L,)
    """
    B         = len(coords_raw)
    chroms    = []
    positions = np.empty(B * L, dtype=np.int64)

    for seq_i, (chrom, seq_start, seq_end) in enumerate(coords_raw):
        bp_per_token = (seq_end - seq_start) / L
        tok_indices  = np.arange(L, dtype=np.float64)
        mids         = (seq_start + (tok_indices + 0.5) * bp_per_token).astype(np.int64)
        base         = seq_i * L
        positions[base: base + L] = mids
        chroms.extend([chrom] * L)

    return chroms, positions


# ---------------------------------------------------------------------------
# Streaming accumulation
# ---------------------------------------------------------------------------

def _make_counts(n_concepts: int, dict_size: int) -> dict:
    """Allocate zero-initialised streaming accumulators."""
    return {
        "pos_acts": np.zeros((n_concepts, dict_size), dtype=np.int64),
        "neg_acts": np.zeros((n_concepts, dict_size), dtype=np.int64),
        "n_pos":    np.zeros(n_concepts,              dtype=np.int64),
        "n_neg":    np.zeros(n_concepts,              dtype=np.int64),
    }


def accumulate_shard(
    sae,
    shard_path: str,
    bed_indices: List[BEDIndex],
    act_size: int,
    batch_size: int,
    device: str,
    counts: dict,
) -> None:
    """
    Load one shard, assign labels, run SAE encoder, binarise activations,
    and accumulate per-concept active-count matrices.

    Float activations are freed before returning so peak RAM stays bounded
    to O(n_concepts x dict_size) across the full dataset.
    """
    print(f"    Processing shard {os.path.basename(shard_path)} ...")
    n_concepts = len(bed_indices)

    shard      = torch.load(shard_path, map_location="cpu")
    emb        = shard["emb"]       # (B, L, D)
    coords_raw = shard["coords"]    # list[(chrom, start, end)]
    B, L, D    = emb.shape
    assert D == act_size, f"Embedding dim mismatch: got {D}, expected {act_size}"

    # ---- 1. Vectorised token labelling --------------------------------
    print("      Labelling tokens ...")
    chroms, positions = _build_token_positions(coords_raw, L)

    token_labels = np.zeros((B * L, n_concepts), dtype=bool)
    chroms_arr = np.array(chroms)
    chrom_to_indices: Dict[str, np.ndarray] = {}
    for ch in np.unique(chroms_arr):
        chrom_to_indices[ch] = np.where(chroms_arr == ch)[0]

    for ch, flat_indices in chrom_to_indices.items():
        # print(f"      Processing chrom '{ch}' with {len(flat_indices)} tokens ...")
        flat_arr = np.array(flat_indices, dtype=np.int64)
        pos_arr  = positions[flat_arr]
        for ci, bed in enumerate(bed_indices):
            hits = bed.contains_batch(ch, pos_arr)
            token_labels[flat_arr[hits], ci] = True

    # ---- 2. SAE forward pass — binarise immediately -------------------
    emb_flat     = emb.reshape(B * L, D)
    active_parts = []
    for start in range(0, B * L, batch_size):
        # print(f"      Running SAE on tokens {start} to {min(start + batch_size, B * L)} ...")
        end   = min(start + batch_size, B * L)
        chunk = get_activations(sae, emb_flat[start:end]).numpy()
        active_parts.append(chunk > 0)          # bool, 4x smaller than float32
    active = np.concatenate(active_parts, axis=0)   # (B*L, dict_size) bool

    # ---- 3. Accumulate counts per concept -----------------------------
    for ci in range(n_concepts):
        # print(f"      Accumulating counts for concept {ci} ('{bed_indices[ci].name}') ...")
        pos_mask = token_labels[:, ci]
        neg_mask = ~pos_mask

        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        if n_pos > 0:
            counts["pos_acts"][ci] += active[pos_mask].sum(axis=0)
        counts["neg_acts"][ci] += active[neg_mask].sum(axis=0)
        counts["n_pos"][ci]    += n_pos
        counts["n_neg"][ci]    += n_neg

    # ---- 4. Explicit cleanup ------------------------------------------
    del shard, emb, emb_flat, active, token_labels


# ---------------------------------------------------------------------------
# Metrics from streaming counts
# ---------------------------------------------------------------------------

def compute_metrics_from_counts(
    counts: dict,
    top_k_features: int,
) -> List[List[dict]]:
    """
    Derive per-concept F1 and confusion matrix from the accumulated integer
    counts.

    Class-imbalance correction
    --------------------------
    Because negatives were never subsampled during streaming, the raw negative
    counts reflect the full (imbalanced) dataset.  We correct by scaling the
    accumulated negative counts so that the effective number of negatives equals
    min(n_neg, n_pos) — the same balance the original per-shard subsampling
    would have produced in expectation.

    Returns a list (one entry per concept) of row-lists sorted by F1 descending.
    """
    n_concepts = counts["n_pos"].shape[0]
    all_rows: List[List[dict]] = []

    for ci in range(n_concepts):
        n_pos = int(counts["n_pos"][ci])
        n_neg = int(counts["n_neg"][ci])

        if n_pos == 0:
            all_rows.append([])
            continue

        pos_acts = counts["pos_acts"][ci].astype(np.float64)
        neg_acts = counts["neg_acts"][ci].astype(np.float64)

        # Scale negatives down to n_pos to correct for class imbalance
        n_neg_eff       = min(n_neg, n_pos)
        scale           = n_neg_eff / n_neg if n_neg > 0 else 0.0
        neg_acts_scaled = neg_acts * scale

        TP = pos_acts
        FN = n_pos      - pos_acts
        FP = neg_acts_scaled
        TN = n_neg_eff  - neg_acts_scaled

        precision = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0)
        recall    = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) > 0)
        f1        = np.divide(2 * precision * recall, precision + recall,
                            out=np.zeros_like(precision), where=(precision + recall) > 0)

        tpr = recall
        tnr = np.divide(TN, TN + FP, out=np.zeros_like(TN), where=(TN + FP) > 0)
        fpr = np.divide(FP, FP + TN, out=np.zeros_like(FP), where=(FP + TN) > 0)
        fnr = np.divide(FN, FN + TP, out=np.zeros_like(FN), where=(FN + TP) > 0)

        # Fraction of positive tokens on which each feature fires
        baseline_prevalence = pos_acts / n_pos

        order = np.argsort(f1)[::-1]
        rows  = []
        for fi in order:
            fi = int(fi)
            rows.append({
                "feature_idx":         fi,
                "f1":                  float(f1[fi]),
                "precision":           float(precision[fi]),
                "recall_tpr":          float(tpr[fi]),
                "tnr":                 float(tnr[fi]),
                "fpr":                 float(fpr[fi]),
                "fnr":                 float(fnr[fi]),
                "tp":                  int(round(TP[fi])),
                "tn":                  int(round(TN[fi])),
                "fp":                  int(round(FP[fi])),
                "fn":                  int(round(FN[fi])),
                "n_positive_tokens":   n_pos,
                "n_negative_tokens":   n_neg_eff,
                "baseline_prevalence": float(baseline_prevalence[fi]),
            })
        all_rows.append(rows)

    return all_rows


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
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FEATURE_CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({
                k: f"{row[k]:.6f}" if isinstance(row[k], float) else row[k]
                for k in FEATURE_CSV_HEADER
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sae_checkpoint",
        default=None,
        help="Path to SAE .pt checkpoint (required unless --raw_neurons).",
    )
    p.add_argument("--sae_cfg",        required=True)
    p.add_argument("--save_dir",       required=True,
                   help="Root dir with train/val/test splits of shards")
    p.add_argument("--layer",          type=int, required=True)
    p.add_argument("--splits",         nargs="+", default=["train", "val", "test"])
    p.add_argument("--bed_dir",        required=True,
                   help="Directory containing *.bed concept annotation files")
    p.add_argument("--out_dir",        required=True)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size",     type=int, default=2048)
    p.add_argument("--top_k_features", type=int, default=10)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip shards already recorded in out_dir; skip writing concept CSVs that already exist.",
    )
    p.add_argument(
        "--raw_neurons",
        action="store_true",
        help="Analyse raw model neurons (embedding dims) instead of SAE features; "
             "uses ReLU so binarisation acts > 0 matches SAE convention.",
    )
    args = p.parse_args()
    if not args.raw_neurons and not args.sae_checkpoint:
        p.error("--sae_checkpoint is required unless --raw_neurons is set")
    return args


# ---------------------------------------------------------------------------
# Resume (shard checkpoints + optional per-concept output skip)
# ---------------------------------------------------------------------------

COFA_RESUME_JSON = ".concept_feature_analysis_resume.json"
COFA_COUNTS_NPZ  = ".concept_feature_analysis_counts.npz"


def _cofa_shard_key(save_dir: str, shard_path: str) -> str:
    save_dir = os.path.normpath(os.path.abspath(save_dir))
    shard_path = os.path.normpath(os.path.abspath(shard_path))
    try:
        return os.path.relpath(shard_path, save_dir)
    except ValueError:
        return shard_path


def _cofa_shard_plan_sha(expected_keys: List[str]) -> str:
    h = hashlib.sha256()
    h.update("\n".join(expected_keys).encode())
    return h.hexdigest()


def _cofa_fingerprint(
    sae_checkpoint: str,
    bed_basenames: List[str],
    splits: List[str],
    layer: int,
    dict_size: int,
    n_concepts: int,
    shard_plan_sha: str,
) -> dict:
    h = hashlib.sha256()
    h.update(os.path.normpath(os.path.abspath(sae_checkpoint)).encode())
    h.update("|".join(bed_basenames).encode())
    return {
        "sae_checkpoint_sha256": h.hexdigest(),
        "bed_basenames":        bed_basenames,
        "splits":               list(splits),
        "layer":                layer,
        "dict_size":            dict_size,
        "n_concepts":           n_concepts,
        "shard_plan_sha256":    shard_plan_sha,
    }


def _cofa_fingerprint_match(stored: dict, expected: dict) -> bool:
    keys = (
        "bed_basenames", "splits", "layer", "dict_size", "n_concepts",
        "sae_checkpoint_sha256", "shard_plan_sha256",
    )
    return all(stored.get(k) == expected.get(k) for k in keys)


def _cofa_load_counts_npz(path: str, n_concepts: int, dict_size: int) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    z = np.load(path)
    try:
        pos_acts = z["pos_acts"]
        neg_acts = z["neg_acts"]
        n_pos    = z["n_pos"]
        n_neg    = z["n_neg"]
    except KeyError:
        return None
    if pos_acts.shape != (n_concepts, dict_size) or neg_acts.shape != (n_concepts, dict_size):
        return None
    if n_pos.shape != (n_concepts,) or n_neg.shape != (n_concepts,):
        return None
    return {
        "pos_acts": pos_acts.astype(np.int64, copy=True),
        "neg_acts": neg_acts.astype(np.int64, copy=True),
        "n_pos":    n_pos.astype(np.int64, copy=True),
        "n_neg":    n_neg.astype(np.int64, copy=True),
    }


def _cofa_save_resume(out_dir: str, meta: dict, counts: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir,0o777)
    json_path = os.path.join(out_dir, COFA_RESUME_JSON)
    npz_path  = os.path.join(out_dir, COFA_COUNTS_NPZ)
    tmp_json  = json_path + ".tmp"
    # np.savez appends ".npz" when the filename does not end with ".npz",
    # so "*.npz.tmp" becomes "*.npz.tmp.npz" and os.replace fails (FileNotFoundError).
    tmp_npz = (npz_path[:-4] + ".tmp.npz") if npz_path.endswith(".npz") else npz_path + ".tmp.npz"
    with open(tmp_json, "w") as fh:
        json.dump(meta, fh, indent=2)
    np.savez(
        tmp_npz,
        pos_acts=counts["pos_acts"],
        neg_acts=counts["neg_acts"],
        n_pos=counts["n_pos"],
        n_neg=counts["n_neg"],
    )
    os.replace(tmp_json, json_path)
    os.replace(tmp_npz, npz_path)
    # #region agent log
    try:
        _logp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug-64cfd6.log"))
        with open(_logp, "a", encoding="utf-8") as _lf:
            _lf.write(
                json.dumps(
                    {
                        "sessionId": "64cfd6",
                        "hypothesisId": "H1",
                        "location": "concept_feature_analysis._cofa_save_resume",
                        "message": "resume npz atomic save ok",
                        "data": {
                            "tmp_npz_used": tmp_npz,
                            "final_npz_exists": os.path.isfile(npz_path),
                            "wrong_legacy_name_exists": os.path.isfile(npz_path + ".tmp"),
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion


def _cofa_clear_resume(out_dir: str) -> None:
    for name in (COFA_RESUME_JSON, COFA_COUNTS_NPZ):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            os.remove(p)


def _cofa_collect_shard_plan(save_dir: str, splits: List[str], layer: int) -> List[Tuple[str, str]]:
    """Ordered (split, shard_path) for all shards that exist."""
    plan: List[Tuple[str, str]] = []
    for split in splits:
        layer_dir   = os.path.join(save_dir, split, f"layer_{layer}")
        shard_paths = sorted(glob.glob(os.path.join(layer_dir, "shard_*.pt")))
        for sp in shard_paths:
            plan.append((split, sp))
    return plan


def _cofa_concept_outputs_exist(out_dir: str, bed_names: List[str]) -> bool:
    summary = os.path.join(out_dir, "summary.csv")
    if not os.path.isfile(summary):
        return False
    for name in bed_names:
        d = os.path.join(out_dir, name)
        if not os.path.isfile(os.path.join(d, "all_features.csv")):
            return False
        if not os.path.isfile(os.path.join(d, "top_features.csv")):
            return False
    return True


def _cofa_read_summary_row_from_top_csv(concept_dir: str) -> Optional[dict]:
    top_path = os.path.join(concept_dir, "top_features.csv")
    if not os.path.isfile(top_path):
        return None
    with open(top_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    best = rows[0]
    summary_header = [
        "concept", "best_feature_idx", "f1", "precision", "recall_tpr",
        "tnr", "fpr", "fnr", "tp", "tn", "fp", "fn",
        "n_positive_tokens", "baseline_prevalence",
    ]
    concept_name = os.path.basename(os.path.normpath(concept_dir))
    out: dict = {"concept": concept_name}
    for k in summary_header:
        if k == "concept":
            continue
        src = k
        if k == "best_feature_idx":
            src = "best_feature_idx" if "best_feature_idx" in best else "feature_idx"
        if src not in best:
            return None
        v = best[src]
        if k == "best_feature_idx":
            out[k] = int(float(v))
        elif k in ("tp", "tn", "fp", "fn", "n_positive_tokens"):
            out[k] = int(float(v))
        else:
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Load config (SAE loaded only when shards still need work) ---
    with open(args.sae_cfg) as fh:
        cfg = json.load(fh)
    cfg = restore_cfg_types(cfg)
    cfg["device"] = args.device

    act_size = cfg["act_size"]
    if args.raw_neurons:
        dict_size = act_size
    else:
        dict_size = cfg["dict_size"]

    ckpt_for_fingerprint = (
        args.sae_checkpoint
        if args.sae_checkpoint
        else os.path.abspath(args.sae_cfg) + "::RAW_NEURONS"
    )

    # ---- Load BED concept files --------------------------------------
    bed_paths = sorted(glob.glob(os.path.join(args.bed_dir, "*.bed")))
    if not bed_paths:
        raise FileNotFoundError(f"No .bed files found in {args.bed_dir}")
    print(f"\nFound {len(bed_paths)} BED concept files:")
    bed_indices = [BEDIndex(p) for p in bed_paths]
    n_concepts  = len(bed_indices)
    bed_basenames = [Path(p).name for p in bed_paths]

    shard_plan = _cofa_collect_shard_plan(args.save_dir, args.splits, args.layer)
    expected_keys = [_cofa_shard_key(args.save_dir, sp) for _, sp in shard_plan]
    plan_sha = _cofa_shard_plan_sha(expected_keys)

    fp_expected = _cofa_fingerprint(
        ckpt_for_fingerprint, bed_basenames, args.splits, args.layer, dict_size, n_concepts, plan_sha
    )

    resume_json_path = os.path.join(args.out_dir, COFA_RESUME_JSON)
    resume_npz_path  = os.path.join(args.out_dir, COFA_COUNTS_NPZ)
    done_keys: Set[str] = set()
    counts: dict = _make_counts(n_concepts, dict_size)

    if args.resume and os.path.isfile(resume_json_path):
        with open(resume_json_path) as fh:
            meta = json.load(fh)
        if not _cofa_fingerprint_match(meta.get("fingerprint", {}), fp_expected):
            print("[resume] Fingerprint mismatch — starting counts from scratch.")
        else:
            loaded = _cofa_load_counts_npz(resume_npz_path, n_concepts, dict_size)
            if loaded is None:
                print("[resume] Missing or invalid counts file — starting from scratch.")
            else:
                counts = loaded
                done_keys = set(meta.get("completed_shard_keys", [])) & set(expected_keys)
                print(f"[resume] Loaded partial state: {len(done_keys)}/{len(expected_keys)} shards done.")

    if (
        args.resume
        and expected_keys
        and len(done_keys) >= len(expected_keys)
        and _cofa_concept_outputs_exist(args.out_dir, [b.name for b in bed_indices])
    ):
        print("[resume] All shards and concept outputs already present — exiting.")
        return

    need_sae = False
    for key in expected_keys:
        if key not in done_keys:
            need_sae = True
            break
    if need_sae:
        if args.raw_neurons:
            print("Raw neuron mode — skipping SAE checkpoint, using ReLU passthrough on embeddings ...")
            sae = RawNeuronSAE(act_size).eval().to(args.device)
        else:
            print("Loading SAE ...")
            sae = load_sae(cfg, args.sae_checkpoint, args.device)
    else:
        sae = None
        print("[resume] All shards already accumulated — skipping SAE load.")

    for split in args.splits:
        layer_dir = os.path.join(args.save_dir, split, f"layer_{args.layer}")
        if not glob.glob(os.path.join(layer_dir, "shard_*.pt")):
            print(f"  [WARNING] No shards in {layer_dir}, skipping.")

    # ---- Streaming accumulation over all splits ----------------------
    total_shards = 0
    for split, shard_path in tqdm(shard_plan, desc="  shards"):
        sk = _cofa_shard_key(args.save_dir, shard_path)
        if sk in done_keys:
            continue
        if not shard_path or not os.path.isfile(shard_path):
            print(f"  [WARNING] Missing shard {shard_path}, skipping.")
            continue
        if sae is None:
            raise RuntimeError("Internal error: need SAE for unfinished shards.")
        print(f"\nStreaming {split}: {os.path.basename(shard_path)}")
        accumulate_shard(
            sae         = sae,
            shard_path  = shard_path,
            bed_indices = bed_indices,
            act_size    = act_size,
            batch_size  = args.batch_size,
            device      = args.device,
            counts      = counts,
        )
        total_shards += 1
        done_keys.add(sk)
        if args.resume:
            meta = {
                "fingerprint": fp_expected,
                "completed_shard_keys": sorted(done_keys),
            }
            _cofa_save_resume(args.out_dir, meta, counts)

    # ---- Token / concept statistics ----------------------------------
    total_tokens = int(counts["n_pos"].sum() + counts["n_neg"].sum())
    n_shards_accounted = len(expected_keys) if expected_keys else len(done_keys)
    extra = f", {total_shards} newly scanned this run" if total_shards else ""
    print(f"\nTotal tokens processed: {total_tokens:,}  ({n_shards_accounted} shards{extra})")
    for ci, bed in enumerate(bed_indices):
        n_pos   = int(counts["n_pos"][ci])
        n_total = n_pos + int(counts["n_neg"][ci])
        pct     = 100 * n_pos / n_total if n_total > 0 else 0.0
        print(f"  {bed.name}: {n_pos:,} positive tokens ({pct:.2f}% prevalence)")

    # ---- Compute metrics from accumulated counts ---------------------
    print("\nComputing metrics ...")
    all_concept_rows = compute_metrics_from_counts(counts, args.top_k_features)

    # ---- Write results -----------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    os.chmod(args.out_dir, 0o777)
    summary_rows = []

    for ci, bed in enumerate(bed_indices):
        rows = all_concept_rows[ci]
        if not rows:
            print(f"\n  [WARNING] No positive tokens for '{bed.name}', skipping.")
            continue

        concept_dir = os.path.join(args.out_dir, bed.name)
        all_csv = os.path.join(concept_dir, "all_features.csv")
        top_csv = os.path.join(concept_dir, "top_features.csv")
        if args.resume and os.path.isfile(all_csv) and os.path.isfile(top_csv):
            print(f"\n  [resume] Skipping '{bed.name}' — CSV outputs already exist.")
            prev = _cofa_read_summary_row_from_top_csv(concept_dir)
            if prev:
                summary_rows.append(prev)
            continue

        print(f"\n  Top {args.top_k_features} features for '{bed.name}':")
        print(f"  {'feat':>6}  {'F1':>6}  {'Prec':>6}  {'Rec/TPR':>8}  "
              f"{'TNR':>6}  {'FPR':>6}  {'FNR':>6}  "
              f"{'TP':>6}  {'TN':>6}  {'FP':>6}  {'FN':>6}  "
              f"{'BasePrevalence':>14}")
        print("  " + "-" * 105)
        top_rows = rows[:args.top_k_features]
        for r in top_rows:
            print(
                f"  {r['feature_idx']:>6}  {r['f1']:>6.3f}  {r['precision']:>6.3f}  "
                f"{r['recall_tpr']:>8.3f}  {r['tnr']:>6.3f}  {r['fpr']:>6.3f}  "
                f"{r['fnr']:>6.3f}  {r['tp']:>6}  {r['tn']:>6}  "
                f"{r['fp']:>6}  {r['fn']:>6}  {r['baseline_prevalence']:>14.4f}"
            )

        os.makedirs(concept_dir, exist_ok=True)
        os.chmod(concept_dir, 0o777)
        write_feature_csv(os.path.join(concept_dir, "all_features.csv"), rows)
        write_feature_csv(os.path.join(concept_dir, "top_features.csv"), top_rows)

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

    summary_path   = os.path.join(args.out_dir, "summary.csv")
    summary_header = [
        "concept", "best_feature_idx", "f1", "precision", "recall_tpr",
        "tnr", "fpr", "fnr", "tp", "tn", "fp", "fn",
        "n_positive_tokens", "baseline_prevalence",
    ]
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=summary_header)
        w.writeheader()
        for row in summary_rows:
            w.writerow({
                k: f"{row[k]:.6f}" if isinstance(row[k], float) else row[k]
                for k in summary_header
            })

    if args.resume:
        _cofa_clear_resume(args.out_dir)

    print(f"\nDone. Results saved to {args.out_dir}/")
    print(f"  summary.csv                    – best feature per concept")
    print(f"  <concept>/top_features.csv     – top {args.top_k_features} features")
    print(f"  <concept>/all_features.csv     – full ranked feature list")


if __name__ == "__main__":
    main()