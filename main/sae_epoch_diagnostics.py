#!/usr/bin/env python
"""
sae_epoch_diagnostics.py
------------------------
Run "sanity checks" for SAE feature/concept association across many checkpoints.

Why this exists
--------------
For SAEs trained on DNA language model activations, a small number of extremely
high-frequency (dense) features can dominate naive association metrics and make
everything look polysemantic. This script:

1) Identifies highly-active features (top-N and/or freq > threshold).
2) Reruns BED-concept association *excluding* those dense features and reports
   whether polysemanticity collapses.
3) Characterizes dense features by correlating activations with token-level
   statistics that matter for DNA data (token position, genomic coordinate,
   embedding norm, recon error).
4) Finds worst-reconstructed activation dimensions (per-dim MSE) and checks if
   dense features' decoder weights concentrate in those dimensions.

This is designed to work with this repo's existing data format:
  - Shards live under:  <save_dir>/<split>/layer_<layer>/shard_*.pt
  - Each shard is a torch dict with:
      "emb":    [B, L, D] float tensor (HyenaDNA embeddings/activations)
      "coords": list[(chrom, start, end)] length B
  - Concept annotations are BED files under --bed_dir (0-based, half-open).

Example
-------
python main/sae_epoch_diagnostics.py ^
  --sae_cfg runs/my_run/cfg.json ^
  --checkpoints_glob runs/my_run/checkpoint_*.pt ^
  --save_dir data/embeddings ^
  --layer 6 ^
  --splits val test ^
  --bed_dir concepts/ ^
  --out_dir results/epoch_diagnostics ^
  --device cuda ^
  --eval_batch_size 2048 ^
  --dense_top_n 8 ^
  --dense_freq_threshold 0.10 ^
  --assoc_f1_threshold 0.10 ^
  --high_mse_top_k 20
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(__file__))

from concept_feature_analysis import (  # type: ignore
    BEDIndex,
    get_activations,
    load_sae,
    restore_cfg_types,
    _build_token_positions,
)


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def _collect_shards(save_dir: str, splits: Sequence[str], layer: int) -> List[str]:
    out: List[str] = []
    for split in splits:
        layer_dir = os.path.join(save_dir, split, f"layer_{layer}")
        out.extend(sorted(glob.glob(os.path.join(layer_dir, "shard_*.pt"))))
    if not out:
        raise FileNotFoundError(
            f"No shard_*.pt found under save_dir={save_dir} for splits={list(splits)} layer={layer}"
        )
    return out


@dataclass
class OnlineCorr:
    """Online Pearson correlation for two scalar streams."""

    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.shape != y.shape:
            raise ValueError(f"corr update shape mismatch: {x.shape} vs {y.shape}")
        self.n += int(x.size)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float((x * x).sum())
        self.sum_y2 += float((y * y).sum())
        self.sum_xy += float((x * y).sum())

    def corr(self) -> float:
        if self.n <= 1:
            return float("nan")
        # cov(x,y) = E[xy] - E[x]E[y]
        ex = self.sum_x / self.n
        ey = self.sum_y / self.n
        ex2 = self.sum_x2 / self.n
        ey2 = self.sum_y2 / self.n
        exy = self.sum_xy / self.n
        vx = max(ex2 - ex * ex, 0.0)
        vy = max(ey2 - ey * ey, 0.0)
        if vx <= 0 or vy <= 0:
            return float("nan")
        cov = exy - ex * ey
        return float(cov / (vx ** 0.5 * vy ** 0.5))


@torch.no_grad()
def _iter_shard_token_batches(
    shard_paths: Sequence[str],
    batch_size: int,
) -> Iterable[Tuple[torch.Tensor, Dict[str, np.ndarray]]]:
    """
    Yield batches of token embeddings plus aligned per-token stats.

    Yields:
      x: [b, D] float32 CPU tensor
      stats dict with numpy arrays length b:
        - "tok_pos": 0..L-1 (position within sequence)
        - "genomic_mid": genomic midpoints (0-based) as int64
        - "emb_norm": ||x||_2 as float64
    """
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        emb: torch.Tensor = shard["emb"]  # [B, L, D]
        coords_raw = shard.get("coords")
        if emb.ndim != 3:
            raise ValueError(f"{os.path.basename(shard_path)}: expected emb [B,L,D], got {list(emb.shape)}")
        B, L, D = emb.shape

        # token position within each sequence
        tok_pos = np.tile(np.arange(L, dtype=np.int64), B)  # [B*L]

        # genomic midpoints per token
        if coords_raw is not None:
            _chroms, genomic_mid = _build_token_positions(coords_raw, L)
        else:
            genomic_mid = np.zeros(B * L, dtype=np.int64)

        x_flat = emb.reshape(B * L, D).float()
        emb_norm = torch.linalg.vector_norm(x_flat, ord=2, dim=-1).double().numpy()

        n = B * L
        start = 0
        while start < n:
            end = min(start + batch_size, n)
            x = x_flat[start:end]
            stats = {
                "tok_pos": tok_pos[start:end],
                "genomic_mid": genomic_mid[start:end],
                "emb_norm": emb_norm[start:end],
            }
            yield x, stats
            start = end


@torch.no_grad()
def _compute_activation_frequencies(
    sae,
    shard_paths: Sequence[str],
    batch_size: int,
) -> Tuple[np.ndarray, int]:
    """Return (freq_per_feature [dict_size], n_tokens_total)."""
    dict_size = int(getattr(sae, "dict_size"))
    act_sum = np.zeros(dict_size, dtype=np.int64)
    n_total = 0

    for x, _stats in _iter_shard_token_batches(shard_paths, batch_size):
        acts = get_activations(sae, x).numpy()  # [b, dict_size] float32 on CPU
        active = acts > 0
        act_sum += active.sum(axis=0).astype(np.int64)
        n_total += int(active.shape[0])

    freq = act_sum.astype(np.float64) / max(n_total, 1)
    return freq, n_total


def _compute_bed_counts(
    sae,
    shard_paths: Sequence[str],
    bed_indices: List[BEDIndex],
    batch_size: int,
    act_size: int,
) -> dict:
    """
    Streaming version of concept_feature_analysis accumulation but returns counts
    for *all* concepts and features.
    """
    n_concepts = len(bed_indices)
    dict_size = int(getattr(sae, "dict_size"))
    counts = {
        "pos_acts": np.zeros((n_concepts, dict_size), dtype=np.int64),
        "neg_acts": np.zeros((n_concepts, dict_size), dtype=np.int64),
        "n_pos": np.zeros(n_concepts, dtype=np.int64),
        "n_neg": np.zeros(n_concepts, dtype=np.int64),
    }

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        emb = shard["emb"]  # (B, L, D)
        coords_raw = shard.get("coords")
        B, L, D = emb.shape
        if D != act_size:
            raise ValueError(f"Embedding dim mismatch: got {D}, expected {act_size}")

        # label tokens per concept (vectorized by chrom)
        if coords_raw is None:
            raise KeyError(
                f"Shard {os.path.basename(shard_path)} missing 'coords' which is required for BED labeling."
            )
        chroms, positions = _build_token_positions(coords_raw, L)
        token_labels = np.zeros((B * L, n_concepts), dtype=bool)
        chroms_arr = np.asarray(chroms)
        for ch in np.unique(chroms_arr):
            flat_idx = np.where(chroms_arr == ch)[0].astype(np.int64)
            pos_arr = positions[flat_idx]
            for ci, bed in enumerate(bed_indices):
                hits = bed.contains_batch(ch, pos_arr)
                token_labels[flat_idx[hits], ci] = True

        # SAE acts (binarise immediately)
        emb_flat = emb.reshape(B * L, D).float()
        active_parts: List[np.ndarray] = []
        for start in range(0, B * L, batch_size):
            end = min(start + batch_size, B * L)
            chunk = get_activations(sae, emb_flat[start:end]).numpy()
            active_parts.append(chunk > 0)
        active = np.concatenate(active_parts, axis=0)  # (B*L, dict_size) bool

        for ci in range(n_concepts):
            pos_mask = token_labels[:, ci]
            neg_mask = ~pos_mask
            n_pos = int(pos_mask.sum())
            n_neg = int(neg_mask.sum())
            if n_pos > 0:
                counts["pos_acts"][ci] += active[pos_mask].sum(axis=0).astype(np.int64)
            counts["neg_acts"][ci] += active[neg_mask].sum(axis=0).astype(np.int64)
            counts["n_pos"][ci] += n_pos
            counts["n_neg"][ci] += n_neg

        del shard, emb, emb_flat, active, token_labels

    return counts


def _f1_matrix_from_counts(counts: dict) -> np.ndarray:
    """
    Return f1 matrix shaped [n_concepts, dict_size] with class-imbalance correction
    matching concept_feature_analysis.py (scale negatives down to n_pos).
    """
    pos_acts = counts["pos_acts"].astype(np.float64)  # [C,F]
    neg_acts = counts["neg_acts"].astype(np.float64)  # [C,F]
    n_pos = counts["n_pos"].astype(np.float64)        # [C]
    n_neg = counts["n_neg"].astype(np.float64)        # [C]

    # effective negatives per concept
    n_neg_eff = np.minimum(n_neg, n_pos)              # [C]
    scale = np.divide(n_neg_eff, n_neg, out=np.zeros_like(n_neg_eff), where=n_neg > 0)  # [C]
    neg_scaled = neg_acts * scale[:, None]

    TP = pos_acts
    FN = n_pos[:, None] - pos_acts
    FP = neg_scaled

    precision = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0)
    recall = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return f1


@torch.no_grad()
def _per_dim_mse(
    sae,
    shard_paths: Sequence[str],
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, int]:
    """
    Stream per-dimension MSE over shards for reconstruction: mean((x - recon)^2, dim=0).
    """
    per_dim_sum: Optional[torch.Tensor] = None
    n_total = 0

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        emb: torch.Tensor = shard["emb"]
        if emb.ndim != 3:
            raise ValueError(f"{os.path.basename(shard_path)}: expected emb [B,L,D], got {list(emb.shape)}")
        B, L, D = emb.shape
        x_flat = emb.reshape(B * L, D).float()

        for start in range(0, x_flat.shape[0], batch_size):
            end = min(start + batch_size, x_flat.shape[0])
            x = x_flat[start:end].to(device)
            recon, _acts = sae(x)
            resid = (x - recon).float()
            if per_dim_sum is None:
                per_dim_sum = torch.zeros(resid.shape[1], dtype=torch.float64, device="cpu")
            per_dim_sum += (resid ** 2).sum(dim=0).double().cpu()
            n_total += int(resid.shape[0])

        del shard, emb, x_flat

    if per_dim_sum is None or n_total == 0:
        raise RuntimeError("No tokens processed for per-dim MSE.")
    return (per_dim_sum / n_total).numpy(), n_total


@torch.no_grad()
def _dense_feature_correlations(
    sae,
    shard_paths: Sequence[str],
    batch_size: int,
    device: str,
    dense_feature_indices: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """
    For each dense feature, compute corr(feature_act, stat) for:
      - tok_pos
      - genomic_mid
      - emb_norm
      - recon_err (per-token squared error norm)
    """
    dense_feature_indices = [int(i) for i in dense_feature_indices]
    if not dense_feature_indices:
        return {}

    corrs: Dict[int, Dict[str, OnlineCorr]] = {
        i: {
            "tok_pos": OnlineCorr(),
            "genomic_mid": OnlineCorr(),
            "emb_norm": OnlineCorr(),
            "recon_err": OnlineCorr(),
        }
        for i in dense_feature_indices
    }

    for x_cpu, stats in _iter_shard_token_batches(shard_paths, batch_size):
        x = x_cpu.to(device)
        recon, acts = sae(x)  # acts [b, dict]
        acts_cpu = acts.detach().float().cpu().numpy()

        # per-token recon error magnitude (squared L2)
        recon_err = ((x - recon).float() ** 2).sum(dim=-1).detach().double().cpu().numpy()

        for fi in dense_feature_indices:
            a = acts_cpu[:, fi]
            corrs[fi]["tok_pos"].update(a, stats["tok_pos"])
            corrs[fi]["genomic_mid"].update(a, stats["genomic_mid"])
            corrs[fi]["emb_norm"].update(a, stats["emb_norm"])
            corrs[fi]["recon_err"].update(a, recon_err)

    out: Dict[int, Dict[str, float]] = {}
    for fi, d in corrs.items():
        out[fi] = {k: v.corr() for k, v in d.items()}
    return out


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sae_cfg", required=True, help="SAE cfg JSON used for checkpoint instantiation")
    p.add_argument("--checkpoints_glob", required=True, help="Glob for SAE checkpoints (e.g. runs/x/checkpoint_*.pt)")
    p.add_argument("--save_dir", required=True, help="Root dir with split/layer_{k}/shard_*.pt (must include coords)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--bed_dir", required=True, help="Directory containing *.bed concept annotation files")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval_batch_size", type=int, default=2048)

    p.add_argument("--dense_top_n", type=int, default=8, help="Always treat top-N freq features as dense")
    p.add_argument("--dense_freq_threshold", type=float, default=0.10, help="Also treat freq > threshold as dense")
    p.add_argument("--assoc_f1_threshold", type=float, default=0.10, help="Concept-feature assoc if F1 >= this")
    p.add_argument("--high_mse_top_k", type=int, default=20, help="How many worst per-dim MSE dims to analyze")
    return p.parse_args()


def _write_csv(path: str, header: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(header))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _checkpoint_name(path: str) -> str:
    p = Path(path)
    stem = p.stem
    # keep it filesystem-safe
    return stem.replace(":", "_")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.sae_cfg, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg = restore_cfg_types(cfg)
    cfg["device"] = args.device

    shard_paths = _collect_shards(args.save_dir, args.splits, args.layer)

    bed_paths = sorted(glob.glob(os.path.join(args.bed_dir, "*.bed")))
    if not bed_paths:
        raise FileNotFoundError(f"No .bed files found in {args.bed_dir}")
    bed_indices = [BEDIndex(p) for p in bed_paths]

    ckpt_paths = sorted(glob.glob(args.checkpoints_glob))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints matched glob: {args.checkpoints_glob}")

    for ckpt in ckpt_paths:
        run_dir = os.path.join(args.out_dir, _checkpoint_name(ckpt))
        os.makedirs(run_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Checkpoint: {ckpt}")
        print("=" * 80)

        sae = load_sae(cfg, ckpt, args.device)

        # 1) dense features by activation frequency
        freq, n_tokens = _compute_activation_frequencies(sae, shard_paths, args.eval_batch_size)
        order = np.argsort(freq)[::-1]
        top_n = int(min(args.dense_top_n, order.size))
        dense_top = order[:top_n].astype(int).tolist()
        dense_thr = np.where(freq > float(args.dense_freq_threshold))[0].astype(int).tolist()
        dense_set = sorted(set(dense_top) | set(dense_thr))
        sparse_mask = np.ones_like(freq, dtype=bool)
        sparse_mask[dense_set] = False

        print(f"Tokens scanned (freq): {n_tokens:,}")
        print(f"Dense features: top_n={len(dense_top)}, freq>{args.dense_freq_threshold} => {len(dense_thr)}; union={len(dense_set)}")
        if dense_set:
            print("  Top dense (idx:freq): " + ", ".join([f"{i}:{freq[i]:.3f}" for i in dense_top[: min(8, len(dense_top))]]))

        # 1+3) concept association full vs sparse (F1-based; BED concepts)
        counts = _compute_bed_counts(
            sae=sae,
            shard_paths=shard_paths,
            bed_indices=bed_indices,
            batch_size=args.eval_batch_size,
            act_size=int(cfg["act_size"]),
        )
        f1_full = _f1_matrix_from_counts(counts)  # [C,F]
        assoc_full = (f1_full >= float(args.assoc_f1_threshold))
        concepts_per_feature_full = assoc_full.sum(axis=0)  # [F]
        poly_full = float(concepts_per_feature_full.mean())

        f1_sparse = f1_full[:, sparse_mask]
        assoc_sparse = (f1_sparse >= float(args.assoc_f1_threshold))
        concepts_per_feature_sparse = assoc_sparse.sum(axis=0)
        poly_sparse = float(concepts_per_feature_sparse.mean()) if concepts_per_feature_sparse.size else 0.0

        print(f"Polysemanticity proxy (mean concepts/feature with F1≥{args.assoc_f1_threshold}): full={poly_full:.3f}, sparse_only={poly_sparse:.3f}")

        # 2) characterize dense features by correlations with token stats
        dense_corrs = _dense_feature_correlations(
            sae=sae,
            shard_paths=shard_paths,
            batch_size=args.eval_batch_size,
            device=args.device,
            dense_feature_indices=dense_set[: max(0, min(len(dense_set), 64))],  # hard cap for runtime safety
        )

        # 3) per-dim MSE and overlap with dense features (decoder energy on bad dims)
        per_dim_mse, n_tokens_mse = _per_dim_mse(sae, shard_paths, args.eval_batch_size, args.device)
        worst_dims = np.argsort(per_dim_mse)[::-1][: int(args.high_mse_top_k)].astype(int)
        per_dim_stats = {
            "per_dim_mse_mean": float(per_dim_mse.mean()),
            "per_dim_mse_max": float(per_dim_mse.max()),
            "per_dim_mse_top_k": int(args.high_mse_top_k),
            "worst_dims": worst_dims.tolist(),
            "worst_dims_mse": per_dim_mse[worst_dims].tolist(),
            "n_tokens_mse": int(n_tokens_mse),
        }

        W_dec = getattr(sae, "W_dec", None)
        dense_overlap_rows: List[Dict[str, object]] = []
        if W_dec is not None and dense_set:
            W = W_dec.detach().float().cpu().numpy()  # [F,D]
            for fi in dense_set:
                v = W[int(fi)]
                denom = float(np.linalg.norm(v) + 1e-12)
                frac = float(np.linalg.norm(v[worst_dims]) / denom)
                dense_overlap_rows.append(
                    {
                        "feature_idx": int(fi),
                        "activation_freq": float(freq[int(fi)]),
                        "decoder_energy_frac_in_worst_dims": frac,
                    }
                )

        # Write outputs
        summary = {
            "checkpoint": ckpt,
            "n_tokens_freq": int(n_tokens),
            "dense_top_n": int(args.dense_top_n),
            "dense_freq_threshold": float(args.dense_freq_threshold),
            "dense_features_union": dense_set,
            "dense_features_top_n": dense_top,
            "dense_features_over_threshold": dense_thr,
            "assoc_f1_threshold": float(args.assoc_f1_threshold),
            "polysemanticity_proxy_full": poly_full,
            "polysemanticity_proxy_sparse_only": poly_sparse,
            "per_dim_mse": per_dim_stats,
        }
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        # Dense features table
        dense_rows: List[Dict[str, object]] = []
        for fi in dense_set:
            row: Dict[str, object] = {"feature_idx": int(fi), "activation_freq": float(freq[int(fi)])}
            if fi in dense_corrs:
                for k, v in dense_corrs[fi].items():
                    row[f"corr_{k}"] = float(v) if np.isfinite(v) else ""
            dense_rows.append(row)
        _write_csv(
            os.path.join(run_dir, "dense_feature_stats.csv"),
            header=[
                "feature_idx",
                "activation_freq",
                "corr_tok_pos",
                "corr_genomic_mid",
                "corr_emb_norm",
                "corr_recon_err",
            ],
            rows=dense_rows,
        )

        if dense_overlap_rows:
            _write_csv(
                os.path.join(run_dir, "dense_feature_worst_dim_overlap.csv"),
                header=["feature_idx", "activation_freq", "decoder_energy_frac_in_worst_dims"],
                rows=dense_overlap_rows,
            )

        # epoch-level CSV accumulator (one line per checkpoint)
        epoch_row = {
            "checkpoint": ckpt,
            "dense_union_n": len(dense_set),
            "poly_full": poly_full,
            "poly_sparse": poly_sparse,
            "per_dim_mse_mean": per_dim_stats["per_dim_mse_mean"],
            "per_dim_mse_max": per_dim_stats["per_dim_mse_max"],
        }
        epoch_csv = os.path.join(args.out_dir, "epoch_summary.csv")
        write_header = not os.path.isfile(epoch_csv)
        with open(epoch_csv, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(epoch_row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(epoch_row)

        # free VRAM between checkpoints
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")
    print(f"Wrote per-checkpoint reports under: {args.out_dir}")
    print(f"Wrote aggregate CSV: {os.path.join(args.out_dir, 'epoch_summary.csv')}")


if __name__ == "__main__":
    main()

