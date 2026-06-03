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
        --splits          train test \
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

    Optional second tree (--out_dir_excluding_dense + --exclude_feature_indices_json):
        Same layout, but rankings exclude dense/highly-active features (union of index
        lists from e.g. evaluate_sae test_highly_active_features.json and
        sae_epoch_diagnostics summary.json dense_features_union).

Memory model
------------
The original script accumulated all shard activations (float32) in RAM before
any analysis, which caused silent OOM kills on large datasets.

This version streams shards one at a time (each ``shard_*.pt`` is ``torch.load``-ed
once per pass; the shard opened for chr-prefix inference is reused when that path
is processed):
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

sys.path.insert(0, os.path.dirname(__file__))
from utils.genomics_coords import infer_use_chr_from_chroms, normalize_chrom_name
from utils.gpu_setup import configure_cuda_performance, resolve_device_str
from utils.assoc_metrics import (
    CANONICAL_FEATURE_COLUMNS,
    CANONICAL_SUMMARY_COLUMNS,
    RANK_KEY,
    compute_raw_metrics,
    rank_order,
    format_value as _fmt_metric,
)
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

    Chromosome names are normalized with the same ``chr`` convention as embedding
    coords (``use_chr``). If ``use_chr`` is None, it is inferred from this BED file.

    Scalar query  – contains(chrom, pos)
    Batch query   – contains_batch(chrom, positions[])   <- hot path
    """

    def __init__(self, bed_path: str, use_chr: Optional[bool] = None):
        self.name = Path(bed_path).stem
        self._starts: Dict[str, np.ndarray] = {}
        self._ends:   Dict[str, np.ndarray] = {}
        self._use_chr: Optional[bool] = use_chr
        self._load(bed_path)

    def _load(self, path: str):
        rows: List[Tuple[str, int, int]] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(("#", "track", "browser")):
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                rows.append((parts[0], int(parts[1]), int(parts[2])))

        if self._use_chr is None:
            self._use_chr = infer_use_chr_from_chroms([c for c, _, _ in rows]) if rows else True

        raw: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for chrom, s, e in rows:
            k = normalize_chrom_name(chrom, self._use_chr)
            raw[k].append((s, e))

        total = 0
        for chrom, ivs in raw.items():
            ivs.sort()
            arr = np.array(ivs, dtype=np.int64)   # (M, 2)
            self._starts[chrom] = arr[:, 0]
            self._ends[chrom]   = arr[:, 1]
            total += len(ivs)
        print(f"  Loaded BED '{self.name}': {total} intervals "
              f"across {len(self._starts)} chroms (use_chr={self._use_chr})")

    # ------------------------------------------------------------------
    # Scalar query
    # ------------------------------------------------------------------
    def contains(self, chrom: str, pos: int) -> bool:
        chrom = normalize_chrom_name(chrom, self._use_chr)  # type: ignore[arg-type]
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
        chrom = normalize_chrom_name(chrom, self._use_chr)  # type: ignore[arg-type]
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
def get_activations(sae, x: torch.Tensor, return_cpu: bool = True) -> torch.Tensor:
    """
    x : (N, D) -> acts : (N, dict_size) float32.

    If return_cpu is True (default), returns activations on CPU for backward compatibility.
    If False, returns on the SAE device so callers can run top-k / thresholds on GPU
    and copy only smaller bool masks or reduced tensors back to host.
    """
    dtype  = next(sae.parameters()).dtype
    device = next(sae.parameters()).device
    x = x.to(
        device=device,
        dtype=dtype,
        non_blocking=(device.type == "cuda"),
    )

    if isinstance(sae, RawNeuronSAE):
        out = F.relu(x).float()
        return out.cpu() if return_cpu else out

    if isinstance(sae, (JumpReLUInferenceSAE, GatedInferenceSAE)):
        _, acts = sae(x)
        out = acts.float()
        return out.cpu() if return_cpu else out

    x, _, _ = sae.preprocess_input(x)
    x_cent  = x - sae.b_dec
    acts    = F.relu(x_cent @ sae.W_enc)

    if hasattr(sae, "jumprelu"):
        acts = sae.jumprelu(acts)
    elif "top_k" in sae.cfg:
        k    = sae.cfg["top_k"]
        topk = torch.topk(acts, min(k, acts.shape[-1]), dim=-1)
        acts = torch.zeros_like(acts).scatter_(-1, topk.indices, topk.values)

    out = acts.float()
    return out.cpu() if return_cpu else out


# ---------------------------------------------------------------------------
# Token position helper
# ---------------------------------------------------------------------------

def _build_token_positions(
    coords_raw: list,
    L: int,
    use_chr: bool,
) -> Tuple[List[str], np.ndarray]:
    """
    Pre-compute the genomic mid-point for every (sequence, token) pair.

    Chromosome strings are normalized with ``use_chr`` to match BEDIndex.

    Returns
    -------
    chroms    : list[str], length B*L
    positions : int64 ndarray, shape (B*L,)
    """
    B         = len(coords_raw)
    chroms    = []
    positions = np.empty(B * L, dtype=np.int64)

    for seq_i, (chrom, seq_start, seq_end) in enumerate(coords_raw):
        cnorm = normalize_chrom_name(str(chrom), use_chr)
        bp_per_token = (seq_end - seq_start) / L
        tok_indices  = np.arange(L, dtype=np.float64)
        mids         = (seq_start + (tok_indices + 0.5) * bp_per_token).astype(np.int64)
        base         = seq_i * L
        positions[base: base + L] = mids
        chroms.extend([cnorm] * L)

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
    use_chr: bool,
    preloaded_shard: Optional[dict] = None,
) -> None:
    """
    Load one shard (unless ``preloaded_shard`` is provided), assign labels, run SAE
    encoder, binarise activations, and accumulate per-concept active-count matrices.

    Float activations are freed before returning so peak RAM stays bounded
    to O(n_concepts x dict_size) across the full dataset.
    """
    print(f"    Processing shard {os.path.basename(shard_path)} ...")
    n_concepts = len(bed_indices)

    shard = preloaded_shard if preloaded_shard is not None else torch.load(
        shard_path, map_location="cpu"
    )
    emb        = shard["emb"]       # (B, L, D)
    coords_raw = shard["coords"]    # list[(chrom, start, end)]
    B, L, D    = emb.shape
    assert D == act_size, f"Embedding dim mismatch: got {D}, expected {act_size}"

    # ---- 1. Vectorised token labelling --------------------------------
    print("      Labelling tokens ...")
    chroms, positions = _build_token_positions(coords_raw, L, use_chr)

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
    emb_flat = emb.reshape(B * L, D)
    dev = torch.device(device)
    for start in range(0, B * L, batch_size):
        # print(f"      Running SAE on tokens {start} to {min(start + batch_size, B * L)} ...")
        end = min(start + batch_size, B * L)
        acts = get_activations(sae, emb_flat[start:end], return_cpu=False)  # [b, F] on SAE device
        active_t = acts > 0

        feat_sum_t = active_t.sum(dim=0, dtype=torch.int64)  # [F]
        feat_sum = feat_sum_t.detach().cpu().numpy()
        bsz = int(active_t.shape[0])

        tl = token_labels[start:end]  # numpy bool [b, C]
        for ci in range(n_concepts):
            pos_mask_np = tl[:, ci]
            n_pos = int(pos_mask_np.sum())
            n_neg = bsz - n_pos
            if n_pos > 0:
                pos_mask_t = torch.from_numpy(pos_mask_np).to(
                    dev, non_blocking=(dev.type == "cuda")
                )
                pos_sum_t = active_t[pos_mask_t].sum(dim=0, dtype=torch.int64)
                pos_sum = pos_sum_t.detach().cpu().numpy()
                counts["pos_acts"][ci] += pos_sum
                counts["neg_acts"][ci] += (feat_sum - pos_sum)
            else:
                counts["neg_acts"][ci] += feat_sum
            counts["n_pos"][ci] += n_pos
            counts["n_neg"][ci] += n_neg

    # ---- 4. Explicit cleanup ------------------------------------------
    del shard, emb, emb_flat, token_labels


# ---------------------------------------------------------------------------
# Metrics from streaming counts
# ---------------------------------------------------------------------------

def load_excluded_feature_indices_json(paths: List[str]) -> Set[int]:
    """
    Union feature indices from one or more JSON files.

    Accepts:
      - A bare JSON list of ints: [0, 1, 2]
      - Dicts with any of: excluded_indices, excluded_feature_indices,
        highly_active_feature_indices, dense_features_union, dense_features_top_n,
        dense_features_over_threshold, indices (evaluate_sae sidecar)
    """
    out: Set[int] = set()
    keys = (
        "excluded_indices",
        "excluded_feature_indices",
        "highly_active_feature_indices",
        "dense_features_union",
        "dense_features_top_n",
        "dense_features_over_threshold",
        "indices",
    )
    for path in paths:
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"exclude JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            out.update(int(x) for x in data)
            continue
        if isinstance(data, dict):
            for k in keys:
                if k in data and data[k] is not None:
                    seq = data[k]
                    if isinstance(seq, list):
                        out.update(int(x) for x in seq)
    return out


def compute_metrics_from_counts(
    counts: dict,
    top_k_features: int,
    exclude_feature_indices: Optional[Set[int]] = None,
    rank_by: str = RANK_KEY,
) -> List[List[dict]]:
    """
    Derive per-concept association metrics from the accumulated integer counts.

    Metrics are computed on the RAW (un-balanced) confusion matrix via
    utils.assoc_metrics.compute_raw_metrics — the same code path used by
    utils/recompute_metrics.py, so the schema (CANONICAL_FEATURE_COLUMNS) is shared.

    Headline scalar is MCC (true zero at chance); enrichment/precision/recall/F1 are
    reported raw, and balanced_f1 is kept for continuity only (see assoc_metrics docs).
    Features are ranked by ``rank_by`` (default MCC) descending.

    Returns a list (one entry per concept) of row-lists sorted by ``rank_by``.
    """
    n_concepts = counts["n_pos"].shape[0]
    all_rows: List[List[dict]] = []

    for ci in range(n_concepts):
        n_pos = int(counts["n_pos"][ci])
        n_neg = int(counts["n_neg"][ci])

        if n_pos == 0:
            all_rows.append([])
            continue

        pos_acts = counts["pos_acts"][ci]
        neg_acts = counts["neg_acts"][ci]
        m = compute_raw_metrics(pos_acts, neg_acts, n_pos, n_neg)
        prevalence = n_pos / (n_pos + n_neg)

        order = rank_order(m, key=rank_by)
        if exclude_feature_indices:
            order = np.array(
                [fi for fi in order if int(fi) not in exclude_feature_indices],
                dtype=np.int64,
            )
        rows = []
        for fi in order:
            fi = int(fi)
            row = {
                "feature_idx": fi,
                "n_positive_tokens": n_pos,
                "n_negative_tokens": n_neg,
                "prevalence": prevalence,
            }
            for k, arr in m.items():
                row[k] = arr[fi]
            rows.append(row)
        all_rows.append(rows)

    return all_rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

FEATURE_CSV_HEADER = list(CANONICAL_FEATURE_COLUMNS)


def write_feature_csv(path: str, rows: List[dict]):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FEATURE_CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt_metric(k, row[k]) for k in FEATURE_CSV_HEADER})


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
                   help="Root dir with train/test splits of shards")
    p.add_argument("--layer",          type=int, required=True)
    p.add_argument("--splits",         nargs="+", default=["train", "test"])
    p.add_argument("--bed_dir",        required=True,
                   help="Directory containing *.bed concept annotation files")
    p.add_argument("--out_dir",        required=True)
    p.add_argument(
        "--device",
        default=None,
        help="cuda (default if available), cpu, … — falls back to CPU if CUDA unavailable.",
    )
    p.add_argument("--batch_size",     type=int, default=2048)
    p.add_argument("--top_k_features", type=int, default=10)
    p.add_argument(
        "--rank_by",
        default=RANK_KEY,
        choices=["mcc", "enrichment", "f1", "precision", "balanced_f1"],
        help="Metric used to rank/select the best feature per concept (default: mcc).",
    )
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
    p.add_argument(
        "--exclude_feature_indices_json",
        nargs="*",
        default=[],
        metavar="PATH",
        help="Optional JSON file(s) whose feature indices are merged (union) and "
             "omitted from rankings when writing --out_dir_excluding_dense. "
             "Use sidecars from evaluate_sae (*_highly_active_features.json) and/or "
             "sae_epoch_diagnostics summary.json (dense_features_union).",
    )
    p.add_argument(
        "--out_dir_excluding_dense",
        default=None,
        help="If set with --exclude_feature_indices_json, write a second result tree "
             "here with dense/highly-active features excluded from F1 rankings.",
    )
    args = p.parse_args()
    if not args.raw_neurons and not args.sae_checkpoint:
        p.error("--sae_checkpoint is required unless --raw_neurons is set")
    if args.out_dir_excluding_dense and not args.exclude_feature_indices_json:
        p.error("--out_dir_excluding_dense requires at least one --exclude_feature_indices_json")
    if args.exclude_feature_indices_json and not args.out_dir_excluding_dense:
        p.error("--exclude_feature_indices_json requires --out_dir_excluding_dense")
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


def _cofa_bootstrap_use_chr_from_shard_plan(
    shard_plan: List[Tuple[str, str]],
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Infer ``use_chr`` from the first shard in ``shard_plan`` that has ``coords``.

    Returns ``(use_chr, path, shard_dict)`` so ``main()`` can skip a duplicate
    ``torch.load`` when that path is processed.
    """
    for _split, shard_path in shard_plan:
        if not shard_path or not os.path.isfile(shard_path):
            continue
        shard = torch.load(shard_path, map_location="cpu")
        coords = shard.get("coords")
        if not coords:
            del shard
            continue
        chroms: List[str] = []
        for row in coords:
            if isinstance(row, (list, tuple)) and len(row) >= 1 and row[0]:
                chroms.append(str(row[0]))
        if chroms:
            return infer_use_chr_from_chroms(chroms), shard_path, shard
        del shard
    return True, None, None


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
    concept_name = os.path.basename(os.path.normpath(concept_dir))
    out: dict = {"concept": concept_name}
    int_cols = ("best_feature_idx", "tp", "tn", "fp", "fn",
                "n_positive_tokens", "n_negative_tokens")
    for k in CANONICAL_SUMMARY_COLUMNS:
        if k == "concept":
            continue
        src = "feature_idx" if k == "best_feature_idx" else k
        if src not in best:
            return None
        v = best[src]
        if k in int_cols:
            out[k] = int(float(v))
        else:
            out[k] = float(v)   # tolerates 'inf'/'nan' (e.g. enrichment)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.device = resolve_device_str(args.device)
    configure_cuda_performance()

    exclude_set = (
        load_excluded_feature_indices_json(list(args.exclude_feature_indices_json))
        if args.exclude_feature_indices_json
        else set()
    )
    if args.out_dir_excluding_dense:
        if len(exclude_set) == 0:
            raise ValueError(
                "Merged exclusion set is empty. Populate JSON from evaluate_sae "
                "(*_highly_active_features.json), sae_epoch_diagnostics (dense_features_union), "
                "or a manual list."
            )
        print(f"\nExcluding {len(exclude_set)} feature index(es) from secondary output "
              f"({args.out_dir_excluding_dense}).")

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
    shard_plan = _cofa_collect_shard_plan(args.save_dir, args.splits, args.layer)
    use_chr, pending_shard_path, pending_shard = _cofa_bootstrap_use_chr_from_shard_plan(shard_plan)
    print(f"\nFound {len(bed_paths)} BED concept files:")
    print(f"  Chromosome naming (match embeddings): use_chr={use_chr}")
    bed_indices = [BEDIndex(p, use_chr=use_chr) for p in bed_paths]
    n_concepts  = len(bed_indices)
    bed_basenames = [Path(p).name for p in bed_paths]
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

    bed_names = [b.name for b in bed_indices]
    second_needed = bool(args.out_dir_excluding_dense and exclude_set)
    if (
        args.resume
        and expected_keys
        and len(done_keys) >= len(expected_keys)
        and _cofa_concept_outputs_exist(args.out_dir, bed_names)
        and (
            not second_needed
            or _cofa_concept_outputs_exist(args.out_dir_excluding_dense, bed_names)
        )
    ):
        print("[resume] All shards and concept outputs already present — exiting.")
        del pending_shard
        return

    if pending_shard_path is not None and pending_shard is not None:
        sk_boot = _cofa_shard_key(args.save_dir, pending_shard_path)
        if sk_boot in done_keys:
            pending_shard = None
            pending_shard_path = None

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
        shard_preloaded = None
        if pending_shard is not None and os.path.normpath(shard_path) == os.path.normpath(
            pending_shard_path or ""
        ):
            shard_preloaded = pending_shard
            pending_shard = None
            pending_shard_path = None
        accumulate_shard(
            sae         = sae,
            shard_path  = shard_path,
            bed_indices = bed_indices,
            act_size    = act_size,
            batch_size  = args.batch_size,
            device      = args.device,
            counts      = counts,
            use_chr     = use_chr,
            preloaded_shard = shard_preloaded,
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
    all_concept_rows = compute_metrics_from_counts(
        counts, args.top_k_features, rank_by=args.rank_by
    )
    excl_concept_rows: Optional[List[List[dict]]] = None
    if second_needed:
        excl_concept_rows = compute_metrics_from_counts(
            counts, args.top_k_features, exclude_feature_indices=exclude_set,
            rank_by=args.rank_by,
        )

    def _write_results_tree(
        out_root: str,
        concept_rows: List[List[dict]],
        summary_label: str,
    ) -> None:
        os.makedirs(out_root, exist_ok=True)
        os.chmod(out_root, 0o777)
        local_summary: List[dict] = []

        for ci, bed in enumerate(bed_indices):
            rows = concept_rows[ci]
            if not rows:
                n_pos_c = int(counts["n_pos"][ci])
                reason = (
                    "no positive tokens"
                    if n_pos_c == 0
                    else "no features left after exclusion (all indices filtered)"
                )
                print(f"\n  [WARNING] Skipping '{bed.name}' ({summary_label}) — {reason}.")
                continue

            concept_dir = os.path.join(out_root, bed.name)
            all_csv = os.path.join(concept_dir, "all_features.csv")
            top_csv = os.path.join(concept_dir, "top_features.csv")
            if args.resume and os.path.isfile(all_csv) and os.path.isfile(top_csv):
                print(f"\n  [resume] Skipping '{bed.name}' ({summary_label}) — CSV outputs already exist.")
                prev = _cofa_read_summary_row_from_top_csv(concept_dir)
                if prev:
                    local_summary.append(prev)
                continue

            print(f"\n  Top {args.top_k_features} features for '{bed.name}' ({summary_label}) "
                  f"[ranked by {args.rank_by}]:")
            print(f"  {'feat':>6}  {'MCC':>7}  {'enrich':>8}  {'Prec':>6}  "
                  f"{'Recall':>6}  {'F1':>6}  {'balF1':>6}")
            print("  " + "-" * 60)
            top_rows = rows[:args.top_k_features]
            for r in top_rows:
                enr = r["enrichment"]
                enr_s = f"{enr:8.2f}" if np.isfinite(enr) else f"{'inf':>8}"
                print(
                    f"  {r['feature_idx']:>6}  {r['mcc']:>7.3f}  {enr_s}  "
                    f"{r['precision']:>6.3f}  {r['recall_tpr']:>6.3f}  "
                    f"{r['f1']:>6.3f}  {r['balanced_f1']:>6.3f}"
                )

            os.makedirs(concept_dir, exist_ok=True)
            os.chmod(concept_dir, 0o777)
            write_feature_csv(os.path.join(concept_dir, "all_features.csv"), rows)
            write_feature_csv(os.path.join(concept_dir, "top_features.csv"), top_rows)

            best = top_rows[0]
            srow = {"concept": bed.name, "best_feature_idx": best["feature_idx"]}
            for k in CANONICAL_SUMMARY_COLUMNS:
                if k not in ("concept", "best_feature_idx"):
                    srow[k] = best[k]
            local_summary.append(srow)

        summary_path = os.path.join(out_root, "summary.csv")
        with open(summary_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CANONICAL_SUMMARY_COLUMNS)
            w.writeheader()
            for row in local_summary:
                w.writerow({k: _fmt_metric(k, row[k]) for k in CANONICAL_SUMMARY_COLUMNS})

    # ---- Write results -----------------------------------------------
    _write_results_tree(args.out_dir, all_concept_rows, "all features")
    if second_needed and excl_concept_rows is not None and args.out_dir_excluding_dense:
        os.makedirs(args.out_dir_excluding_dense, exist_ok=True)
        with open(
            os.path.join(args.out_dir_excluding_dense, "excluded_feature_indices.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(
                {
                    "excluded_indices": sorted(exclude_set),
                    "source_json_files": list(args.exclude_feature_indices_json),
                },
                fh,
                indent=2,
            )
        _write_results_tree(args.out_dir_excluding_dense, excl_concept_rows, "excluding dense list")

    if args.resume:
        _cofa_clear_resume(args.out_dir)

    print(f"\nDone. Results saved to {args.out_dir}/")
    if second_needed and args.out_dir_excluding_dense:
        print(f"     (excluding dense) also saved to {args.out_dir_excluding_dense}/")
    print(f"  summary.csv                    – best feature per concept")
    print(f"  <concept>/top_features.csv     – top {args.top_k_features} features")
    print(f"  <concept>/all_features.csv     – full ranked feature list")


if __name__ == "__main__":
    main()