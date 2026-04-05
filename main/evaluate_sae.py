#!/usr/bin/env python
"""
Evaluate a trained SAE on held-out DNA sequences using HyenaDNA embeddings.

Supports pre-saved embeddings (.pt or .npy files), skipping the need to re-run
the base model for reconstruction and sparsity metrics.

Fidelity (loss recovered) is measured using HyenaDNA's next-token prediction
task: activations at a given layer are patched mid-forward-pass via NNsight,
and the resulting cross-entropy is compared against the original and zero-ablated
baselines to compute the % loss recovered metric.

This script calculates three key metrics:
1. Reconstruction quality (variance explained, MSE)
2. Sparsity (L0, feature activation statistics)
3. Downstream task fidelity (loss recovered on next-token prediction)

Usage:
    # With pre-saved embeddings (val and test sets):
    python evaluate_sae_hyenadna.py \
        --sae_path models/my_sae/ae.pt \
        --val_embeddings_path data/val_embeddings/ \
        --test_embeddings_path data/test_embeddings/ \
        --output_file results/eval_metrics.yaml

    # With fidelity evaluation (requires sequences + HyenaDNA checkpoint):
    python evaluate_sae_hyenadna.py \
        --sae_path models/my_sae/ae.pt \
        --val_embeddings_path data/val_embeddings/ \
        --test_embeddings_path data/test_embeddings/ \
        --val_bed_path data/val_sequences.bed \
        --test_bed_path data/test_sequences.bed \
        --genome_path data/genome.fasta \
        --hyenadna_checkpoint_path pretrained/hyenadna-medium-160k \
        --layer_idx 8 \
        --output_file results/eval_metrics.yaml

Embeddings directory format:
    - Directory containing shard_*.pt files
    - Each shard .pt file is a dict with key "emb" containing a tensor of
      shape [B, L, D] (batch × sequence_length × d_model)
    - All shards are concatenated in sorted order

Sequences format (for fidelity only):
    - BED file (tab-separated: chrom, start, end, ...)
"""

import glob
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Set, Tuple

try:
    from pyfaidx import Fasta
except ImportError as e:
    raise SystemExit("Missing dependency pyfaidx. Install: pip install pyfaidx") from e


import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from SAE_training.sae import (
    BatchTopKSAE,
    TopKSAE,
    VanillaSAE,
    JumpReLUSAE,
    JumpReLUInferenceSAE,
    GatedSAE,
    GatedInferenceSAE,
)



def restore_cfg_types(cfg):
    if isinstance(cfg.get("dtype"), str):
        cfg["dtype"] = getattr(torch, cfg["dtype"].replace("torch.", ""))
    if isinstance(cfg.get("device"), str):
        cfg["device"] = torch.device(cfg["device"])
    return cfg

def load_cfg(cfg_path, device):
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg = restore_cfg_types(cfg)
    cfg["device"] = device
    return cfg

def load_sae(cfg: dict, checkpoint_path: str, device: str):
    state = torch.load(checkpoint_path, map_location=device)
    # print(cfg.keys())

    # Unwrap nested checkpoint formats
    sae_state = state.get("state_dict") or state.get("model_state_dict") or state
    saved_cfg  = state.get("cfg", cfg)       # checkpoint may carry its own cfg
    theta      = state.get("theta") or saved_cfg.get("theta")

    arch = cfg.get("sae_type", "batchtopk").lower()
    cls_map = {
        "batchtopk": BatchTopKSAE,
        "top_k":       TopKSAE,
        "vanilla":     VanillaSAE,
        "jumprelu":    JumpReLUSAE,
        "gated":       GatedSAE,
    }
    sae = cls_map[arch](cfg)
    sae.load_state_dict(sae_state, strict=False)

    # Wrap BatchTopK with JumpReLU inference gate using saved theta
    if arch == "batchtopk":
        if theta is None:
            raise ValueError(
                "BatchTopKSAE checkpoint has no 'theta'. "
                "Ensure save_checkpoint stores it (cfg['theta'] = theta)."
            )
        print(f"Wrapping BatchTopKSAE with JumpReLUInferenceSAE (theta={theta:.6f})")
        sae = JumpReLUInferenceSAE(sae, theta=theta)
    elif arch == "gated":
        print("Wrapping GatedSAE with GatedInferenceSAE")
        sae = GatedInferenceSAE(sae)

    sae.eval().to(device)
    return sae

# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def get_shard_paths(embeddings_dir: Path) -> List[Path]:
    """
    Return sorted list of shard_*.pt files in the given directory.

    Args:
        embeddings_dir: Directory containing shard files.

    Returns:
        Sorted list of Path objects for each shard.

    Raises:
        FileNotFoundError: If the directory doesn't exist or has no shards.
    """
    embeddings_dir = Path(embeddings_dir)
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    if not embeddings_dir.is_dir():
        raise ValueError(f"Expected a directory, got a file: {embeddings_dir}")

    shard_paths = sorted(embeddings_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard_*.pt files found in: {embeddings_dir}\n"
            f"Contents: {list(embeddings_dir.iterdir())[:10]}"
        )
    return shard_paths


def iter_shard_batches(
    shard_paths: List[Path],
    batch_size: int,
    device: str = "cpu",
):
    """
    Stream mini-batches from a list of shard .pt files without loading all
    shards into memory at once.

    Each shard is a dict {"emb": tensor [B, L, D] or [N, D]}.  Rows are
    yielded in contiguous mini-batches of `batch_size` tokens.  Shards are
    loaded one at a time and discarded after use, so peak host memory is
    bounded by one shard + one batch.

    Args:
        shard_paths: Ordered list of shard .pt file paths.
        batch_size:  Number of tokens per yielded mini-batch.
        device:      Device to move each mini-batch to before yielding.

    Yields:
        torch.Tensor: [<=batch_size, d_model] float32 on `device`.
    """
    if not shard_paths:
        raise FileNotFoundError("No shard files provided.")

    buffer: Optional[torch.Tensor] = None   # leftover rows from previous shard

    for shard_path in shard_paths:
        shard_path = Path(shard_path)
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        shard = torch.load(shard_path, map_location="cpu")
        emb: torch.Tensor = shard["emb"]

        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        emb = emb.float()

        if emb.ndim == 3:
            B, L, D = emb.shape
            emb = emb.reshape(B * L, D)
        elif emb.ndim != 2:
            raise ValueError(
                f"Shard {shard_path.name}: expected 2D [N, D] or 3D [B, L, D], "
                f"got shape {list(emb.shape)}"
            )

        if torch.isnan(emb).any():
            n_nan = torch.isnan(emb).sum().item()
            print(f"  ⚠️  {shard_path.name}: {n_nan} NaN values replaced with 0")
            emb = torch.nan_to_num(emb, nan=0.0)

        # Prepend any leftover rows from the previous shard
        if buffer is not None:
            emb = torch.cat([buffer, emb], dim=0)
            buffer = None

        # Yield full batches; keep the tail for the next shard
        start = 0
        while start + batch_size <= emb.shape[0]:
            yield emb[start : start + batch_size].to(device)
            start += batch_size

        if start < emb.shape[0]:
            buffer = emb[start:]   # keep on CPU until next shard

    # Yield the final partial batch (if any)
    if buffer is not None and buffer.shape[0] > 0:
        yield buffer.to(device)


def _eval_resume_dir(output_file: Path) -> Path:
    return output_file.parent / f".evaluate_sae_resume_{output_file.stem}"


def _eval_paths_sha(paths: List[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.resolve()).encode())
        h.update(b"\n")
    return h.hexdigest()


def _eval_split_fingerprint(
    sae_path: str,
    cfg_path: str,
    embed_dir: Path,
    eval_batch_size: int,
    paths_sha: str,
) -> dict:
    h = hashlib.sha256()
    h.update(os.path.normpath(os.path.abspath(sae_path)).encode())
    h.update(os.path.normpath(os.path.abspath(cfg_path)).encode())
    return {
        "sae_cfg_sha256": h.hexdigest(),
        "embed_dir":    str(Path(embed_dir).resolve()),
        "eval_batch_size": eval_batch_size,
        "shard_paths_sha256": paths_sha,
    }


def _eval_fp_match(a: dict, b: dict) -> bool:
    keys = ("sae_cfg_sha256", "embed_dir", "eval_batch_size", "shard_paths_sha256")
    return all(a.get(k) == b.get(k) for k in keys)


def _eval_save_resume_tmp(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _eval_fidelity_fingerprint(
    split_fp: dict,
    bed_path: Path,
    hyenadna_checkpoint_path: str,
    layer_idx: int,
    fidelity_max_seq_len: int,
) -> dict:
    out = dict(split_fp)
    out["bed_path"] = str(Path(bed_path).resolve())
    out["hyenadna_checkpoint_path"] = hyenadna_checkpoint_path
    out["layer_idx"] = int(layer_idx)
    out["fidelity_max_seq_len"] = int(fidelity_max_seq_len)
    return out


def _eval_fid_match(a: dict, b: dict) -> bool:
    keys = (
        "sae_cfg_sha256", "embed_dir", "eval_batch_size", "shard_paths_sha256",
        "split", "bed_path", "hyenadna_checkpoint_path", "layer_idx",
        "fidelity_max_seq_len",
    )
    return all(a.get(k) == b.get(k) for k in keys)

# --------------------------------------------------------------------------
# Read BED coordinates
# --------------------------------------------------------------------------
BedRow = Tuple[str, int, int]

def read_bed(path: str, expected_len: int) -> List[BedRow]:
    rows: List[BedRow] = []
    with open(path, "r", encoding="utf-8") as f:
        # print(f"Reading BED file: {path}")
        for line in f:
            # print(f"Reading line:  {line.strip()}")
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            if end <= start:
                continue
            if (end - start) != expected_len:
                # You said these are fixed windows already; skip anything else.
                continue
            rows.append((chrom, start, end))
    print(f"  Found {len(rows)} valid BED rows with length {expected_len}")
    return rows

def sanitize_dna(seq: str) -> str:
    # Uppercase and map anything outside A/C/G/T/N to N (hg38 can include other IUPAC codes)
    seq = seq.upper()
    allowed = {"A", "C", "G", "T", "N"}
    if all(c in allowed for c in seq):
        return seq
    return "".join(c if c in allowed else "N" for c in seq)

def fetch_batch(genome: Fasta, batch: List[BedRow]) -> List[str]:
    print(f"  Fetching {len(batch)} sequences from genome FASTA...")
    seqs: List[str] = []
    for chrom, start, end in batch:
        # pyfaidx uses 0-based slicing consistent with BED: [start:end)
        s = str(genome[chrom][start:end])
        seqs.append(sanitize_dna(s))
    return seqs


# ---------------------------------------------------------------------------
# Metric calculations — batched to avoid OOM on large embedding sets
# ---------------------------------------------------------------------------

def _update_welford(
    n: int,
    mean: torch.Tensor,
    M2: torch.Tensor,
    batch: torch.Tensor,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    One step of Chan's parallel variance algorithm (Welford generalisation).

    Accumulates sufficient statistics (count, per-dim mean, per-dim M2) so
    that variance can be computed exactly over an arbitrary stream of
    mini-batches without storing any activations.

    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Args:
        n:     Token count seen so far (before this batch).
        mean:  Running per-dim mean [d_model], accumulated so far.
        M2:    Running per-dim sum of squared deviations [d_model], accumulated so far.
        batch: Current mini-batch [b, d_model].

    Returns:
        (n_new, mean_new, M2_new) — updated sufficient statistics.
    """
    b = batch.shape[0]
    batch_mean = batch.mean(dim=0)
    batch_M2   = ((batch - batch_mean) ** 2).sum(dim=0)

    n_new    = n + b
    delta    = batch_mean - mean
    mean_new = mean + delta * (b / n_new)
    M2_new   = M2 + batch_M2 + delta ** 2 * (n * b / n_new)
    return n_new, mean_new, M2_new


def calculate_reconstruction_metrics(
    sae,
    shard_paths: List[Path],
    batch_size: int = 4096,
    device: str = "cpu",
    resume: bool = False,
    resume_path: Optional[Path] = None,
    split_fp: Optional[dict] = None,
) -> dict:
    """
    Calculate reconstruction quality metrics in mini-batches to avoid OOM.

    Streams shards one at a time; never materialises the full token matrix on
    the GPU.  Variance is accumulated exactly via Chan's parallel algorithm.

    Args:
        sae:         Trained SAE model.
        shard_paths: Ordered list of shard .pt file paths.
        batch_size:  Tokens processed per forward pass.
        device:      Torch device string.
        resume:      If True, load/save per-shard progress under resume_path.
        resume_path: File to store reconstruction accumulator state.
        split_fp:    Fingerprint dict validated against saved checkpoints.

    Returns:
        dict with mse, variance_explained, per-dimension MSE stats, n_tokens.
    """
    d_model = None

    # Sufficient statistics for Chan's algorithm — kept on CPU to save VRAM
    n_total  = 0
    x_mean   = None   # [d_model]
    x_M2     = None   # [d_model]
    res_mean = None   # [d_model]  (residual = x - recon)
    res_M2   = None

    sq_err_sum  = torch.tensor(0.0)   # scalar accumulators (CPU)
    per_dim_mse_sum = None            # [d_model]

    completed: Set[str] = set()
    if (
        resume
        and resume_path is not None
        and split_fp is not None
        and resume_path.is_file()
    ):
        blob = torch.load(resume_path, map_location="cpu")
        if _eval_fp_match(blob.get("fingerprint", {}), split_fp):
            completed = set(blob.get("completed_shards", []))
            st = blob.get("state", {})
            if st and int(st.get("n_total", 0)) > 0:
                d_model = int(st["d_model"])
                n_total = int(st["n_total"])
                x_mean  = st["x_mean"].cpu().float()
                x_M2    = st["x_M2"].cpu().float()
                res_mean = st["res_mean"].cpu().float()
                res_M2   = st["res_M2"].cpu().float()
                sq_err_sum = st["sq_err_sum"].cpu().float()
                per_dim_mse_sum = st["per_dim_mse_sum"].cpu().float()
                print(f"  [resume] Reconstruction: {len(completed)}/{len(shard_paths)} shards already done.")
        else:
            completed = set()

    with torch.no_grad():
        for sp in shard_paths:
            key = str(sp.resolve())
            if key in completed:
                continue
            for batch in tqdm(
                iter_shard_batches([sp], batch_size, device=device),
                desc="Reconstruction",
                leave=False,
            ):
                recon, acts    = sae(batch)
                residual = batch - recon         # [b, d_model]
                b        = batch.shape[0]

                if d_model is None:
                    d_model      = batch.shape[1]
                    x_mean       = torch.zeros(d_model)
                    x_M2         = torch.zeros(d_model)
                    res_mean     = torch.zeros(d_model)
                    res_M2       = torch.zeros(d_model)
                    per_dim_mse_sum = torch.zeros(d_model)

                # Accumulate MSE
                sq_err_sum      += (residual ** 2).sum().cpu()
                per_dim_mse_sum += (residual ** 2).sum(dim=0).cpu()

                # Accumulate variance statistics (on CPU)
                n_total, x_mean, x_M2     = _update_welford(n_total, x_mean, x_M2,     batch.cpu())
                _,       res_mean, res_M2 = _update_welford(n_total - b, res_mean, res_M2, residual.cpu())

            completed.add(key)
            if resume and resume_path is not None and split_fp is not None and d_model is not None:
                _eval_save_resume_tmp(
                    resume_path,
                    {
                        "fingerprint": split_fp,
                        "completed_shards": sorted(completed),
                        "state": {
                            "d_model": d_model,
                            "n_total": n_total,
                            "x_mean": x_mean.cpu(),
                            "x_M2": x_M2.cpu(),
                            "res_mean": res_mean.cpu(),
                            "res_M2": res_M2.cpu(),
                            "sq_err_sum": sq_err_sum.cpu(),
                            "per_dim_mse_sum": per_dim_mse_sum.cpu(),
                        },
                    },
                )

    if n_total == 0:
        raise RuntimeError("No tokens were processed — check shard_paths.")

    mse            = (sq_err_sum / (n_total * d_model)).item()
    per_dim_mse    = per_dim_mse_sum / n_total

    total_variance   = (x_M2   / (n_total - 1)).sum().item()
    residual_variance = (res_M2 / (n_total - 1)).sum().item()
    variance_explained = (1 - residual_variance / total_variance) if total_variance > 0 else 0.0

    return {
        "mse": mse,
        "variance_explained": float(variance_explained),
        "per_dim_mse_mean": per_dim_mse.mean().item(),
        "per_dim_mse_std":  per_dim_mse.std().item(),
        "per_dim_mse_max":  per_dim_mse.max().item(),
        "n_tokens": int(n_total),
    }


def calculate_sparsity_metrics(
    sae,
    shard_paths: List[Path],
    batch_size: int = 4096,
    device: str = "cpu",
    resume: bool = False,
    resume_path: Optional[Path] = None,
    split_fp: Optional[dict] = None,
) -> dict:
    """
    Calculate sparsity metrics in mini-batches to avoid OOM.

    Streams shards one at a time and accumulates per-feature activation
    counts, so peak GPU memory is O(batch_size × d_model + dict_size).

    Args:
        sae:         Trained SAE model.
        shard_paths: Ordered list of shard .pt file paths.
        batch_size:  Tokens processed per forward pass.
        device:      Torch device string.
        resume:      If True, load/save per-shard progress under resume_path.
        resume_path: File to store sparsity accumulator state.
        split_fp:    Fingerprint dict validated against saved checkpoints.

    Returns:
        dict with L0, dead features, highly active features, etc., and n_tokens.
    """
    n_total         = 0
    l0_sum          = 0.0
    feature_act_sum = None   # [dict_size] — accumulated on CPU

    completed: Set[str] = set()
    if (
        resume
        and resume_path is not None
        and split_fp is not None
        and resume_path.is_file()
    ):
        blob = torch.load(resume_path, map_location="cpu")
        if _eval_fp_match(blob.get("fingerprint", {}), split_fp):
            completed = set(blob.get("completed_shards", []))
            st = blob.get("state", {})
            if st and int(st.get("n_total", 0)) > 0:
                n_total = int(st["n_total"])
                l0_sum  = float(st["l0_sum"])
                feature_act_sum = st["feature_act_sum"].cpu().float()
                print(f"  [resume] Sparsity: {len(completed)}/{len(shard_paths)} shards already done.")
        else:
            completed = set()

    with torch.no_grad():
        for sp in shard_paths:
            key = str(sp.resolve())
            if key in completed:
                continue
            for batch in tqdm(
                iter_shard_batches([sp], batch_size, device=device),
                desc="Sparsity",
                leave=False,
            ):
                recon, acts = sae(batch)   # [b, dict_size]
                active   = (acts != 0)     # [b, dict_size] bool

                if feature_act_sum is None:
                    feature_act_sum = torch.zeros(acts.shape[1])

                l0_sum          += active.float().sum(dim=-1).sum().item()
                feature_act_sum += active.float().sum(dim=0).cpu()
                n_total         += batch.shape[0]

            completed.add(key)
            if (
                resume
                and resume_path is not None
                and split_fp is not None
                and feature_act_sum is not None
            ):
                _eval_save_resume_tmp(
                    resume_path,
                    {
                        "fingerprint": split_fp,
                        "completed_shards": sorted(completed),
                        "state": {
                            "n_total": n_total,
                            "l0_sum": l0_sum,
                            "feature_act_sum": feature_act_sum.cpu(),
                        },
                    },
                )

    if n_total == 0:
        raise RuntimeError("No tokens were processed — check shard_paths.")

    feature_active = feature_act_sum / n_total   # activation frequency per feature
    l0_sparsity    = l0_sum / n_total
    dead_features  = (feature_active == 0).sum().item()
    highly_active  = (feature_active > 0.5).sum().item()

    return {
        "l0_sparsity": l0_sparsity,
        "l0_sparsity_pct": (l0_sparsity / sae.dict_size) * 100,
        "dead_features": int(dead_features),
        "dead_features_pct": (dead_features / sae.dict_size) * 100,
        "highly_active_features": int(highly_active),
        "highly_active_pct": (highly_active / sae.dict_size) * 100,
        "mean_feature_activation_freq": feature_active.mean().item() * 100,
        "n_tokens": int(n_total),
    }


# ---------------------------------------------------------------------------
# HyenaDNA fidelity — next-token prediction with activation patching
# ---------------------------------------------------------------------------

def load_hyenadna_model(checkpoint_path: str, device: str):
    """
    Load a HyenaDNA model from a local checkpoint directory or HuggingFace hub.

    HyenaDNA checkpoints follow the standard HuggingFace AutoModel interface.
    Supported checkpoint names (from LongSafari/hyenadna-* on HF Hub):
        hyenadna-tiny-1k-seqlen
        hyenadna-small-32k-seqlen
        hyenadna-medium-160k-seqlen
        hyenadna-medium-450k-seqlen
        hyenadna-large-1m-seqlen

    Args:
        checkpoint_path: Local path or HuggingFace model ID
        device: torch device string

    Returns:
        (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
    except ImportError:
        raise ImportError("transformers is required: pip install transformers")

    cfg = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).eval()
    return model, tok


def get_hyenadna_layer_module(model, layer_idx: int):
    """
    Return the specific transformer/hyena layer submodule to patch.
 
    Tries several known attribute paths for the HyenaDNA backbone layers
    ModuleList, handling both the raw HyenaDNAModel and the HuggingFace
    AutoModelForCausalLM wrapper (which nests the backbone under model.model).
    """
    # for attr_path in [
    #     "backbone.layers",           # HyenaDNAModel used directly
    #     "model.backbone.layers",     # HuggingFace CausalLM wrapper
    #     "hyena.backbone.layers",     # some checkpoint variants
    #     "layers",                    # bare backbone
    #     "hyena.layers",
    # ]:
    #     obj = model
    #     try:
    #         for part in attr_path.split("."):
    #             obj = getattr(obj, part)
    #         return obj[layer_idx]
    #     except (AttributeError, IndexError):
    #         continue
    return model.hyena.backbone.layers[layer_idx]
 


def calculate_cross_entropy_causal(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    """
    Next-token prediction cross-entropy, ignoring the last position.

    Args:
        logits: [seq_len, vocab_size]
        tokens: [seq_len]  (input token IDs)

    Returns:
        Mean cross-entropy over predicted positions.
    """
    # Predict token t+1 from position t
    pred_logits = logits[:-1]        # [seq_len-1, vocab]
    targets = tokens[1:]             # [seq_len-1]
    return F.cross_entropy(pred_logits, targets).item()


def run_hyenadna_with_patch(
    model,
    tokens: torch.Tensor,
    layer_idx: int,
    patch_tensor: Optional[torch.Tensor] = None,
    device: str = "cpu",
):
    """
    Run HyenaDNA with an optional activation patch at layer_idx using hooks.

    Args:
        model: HyenaDNA model
        tokens: [1, seq_len] input token IDs
        layer_idx: Which layer's output to intercept/replace
        patch_tensor: If provided, replace the layer output with this tensor.
                      Shape must match the layer's output: [1, seq_len, d_model]
                      If None, run unmodified (identity).
        device: torch device

    Returns:
        logits: [seq_len, vocab_size]
        hidden: [1, seq_len, d_model] — captured hidden state at layer_idx
    """
    captured_hidden = {}

    def capture_hook(module, input, output):
        # output is usually a tensor or (tensor, ...) tuple
        hidden = output[0] if isinstance(output, tuple) else output
        captured_hidden["hidden"] = hidden.detach()
        if patch_tensor is not None:
            if isinstance(output, tuple):
                return (patch_tensor,) + output[1:]
            return patch_tensor
        return output

    layer_module = get_hyenadna_layer_module(model, layer_idx)
    hook = layer_module.register_forward_hook(capture_hook)

    tokens = tokens.to(device)
    try:
        with torch.no_grad():
            outputs = model(tokens)
            # HuggingFace CausalLM: outputs.logits shape [batch, seq_len, vocab]
            logits = outputs.logits[0]  # [seq_len, vocab]
    finally:
        hook.remove()

    return logits, captured_hidden.get("hidden")


class HyenaDNAFidelityEvaluator:
    """
    Evaluates SAE fidelity on HyenaDNA next-token prediction.

    Mirrors the ESMFidelityFunction logic but adapted for:
    - Causal (next-token) rather than masked language modelling
    - HyenaDNA's layer structure
    - bed files
    """

    def __init__(
        self,
        checkpoint_path: str,
        bed_path: str,
        genome: Fasta,
        layer_idx: int,
        batch_size: int = 4,
        max_seq_len: int = 1024,
        device: str = "cpu",
    ):
        self.device = device
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len

        print("  Loading HyenaDNA model for fidelity evaluation...")
        self.model, self.tokenizer = load_hyenadna_model(checkpoint_path, device)

        # read bed
        rows = read_bed(bed_path, max_seq_len)
        seqs = fetch_batch(genome, rows)

        # Load and tokenize sequences
        
        self.tokenized = []
        for seq in seqs:
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            enc = self.tokenizer(seq, return_tensors="pt")
            self.tokenized.append(enc["input_ids"])  # [1, seq_len]

        # Pre-compute original and zero-ablation baselines
        print(f"  Computing baselines on {len(self.tokenized)} sequences...")
        self.orig_loss, self.zero_loss = self._baseline_losses()
        print(f"  Baseline CE (identity):       {self.orig_loss:.4f}")
        print(f"  Baseline CE (zero-ablation):  {self.zero_loss:.4f}")

    def _baseline_losses(self):
        orig_losses, zero_losses = [], []

        for tokens in tqdm(self.tokenized, desc="Baselines"):
            # Original (no patch)
            logits, hidden = run_hyenadna_with_patch(
                self.model, tokens, self.layer_idx, patch_tensor=None, device=self.device
            )
            orig_losses.append(calculate_cross_entropy_causal(logits, tokens[0].to(self.device)))

            # Zero-ablation
            zero_patch = torch.zeros_like(hidden)
            logits_zero, _ = run_hyenadna_with_patch(
                self.model, tokens, self.layer_idx, patch_tensor=zero_patch, device=self.device
            )
            zero_losses.append(
                calculate_cross_entropy_causal(logits_zero, tokens[0].to(self.device))
            )

        return float(np.mean(orig_losses)), float(np.mean(zero_losses))

    def calculate_fidelity(self, sae) -> dict:
        """
        Compute % loss recovered when patching with SAE reconstructions.

        Args:
            sae: Trained SAE model

        Returns:
            dict with pct_loss_recovered and CE_w_sae_patching
        """
        sae_losses = []

        for tokens in tqdm(self.tokenized, desc="SAE fidelity"):
            # Get original hidden state
            _, hidden = run_hyenadna_with_patch(
                self.model, tokens, self.layer_idx, patch_tensor=None, device=self.device
            )
            # hidden: [1, seq_len, d_model]
            # SAE expects [n_tokens, d_model]; flatten, reconstruct, reshape
            h_flat = hidden.squeeze(0)                          # [seq_len, d_model]
            with torch.no_grad():
                recon, acts = sae(h_flat)      # [seq_len, d_model]
            recon_patch = recon.unsqueeze(0)               # [1, seq_len, d_model]

            logits_sae, _ = run_hyenadna_with_patch(
                self.model, tokens, self.layer_idx, patch_tensor=recon_patch, device=self.device
            )
            sae_losses.append(
                calculate_cross_entropy_causal(logits_sae, tokens[0].to(self.device))
            )

        ce_sae = float(np.mean(sae_losses))

        # Loss recovered formula (same as original ESM version)
        numerator = ce_sae - self.orig_loss
        denominator = self.zero_loss - self.orig_loss
        if np.isclose(denominator, 0):
            loss_recovered = 0.0
        else:
            loss_recovered = float(np.clip(1 - numerator / denominator, 0, 1)) * 100

        return {
            "pct_loss_recovered": loss_recovered,
            "CE_w_sae_patching": ce_sae,
            "CE_identity_baseline": self.orig_loss,
            "CE_zero_ablation_baseline": self.zero_loss,
        }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_sae(
    sae_path: str,
    cfg_path: str,
    val_embeddings_path: Path,
    test_embeddings_path: Path,
    output_file: Optional[Path] = None,
    device_str: str = "cuda",
    eval_batch_size: int = 4096,
    # Fidelity args (all optional — skip fidelity if not provided)
    val_bed_path: Optional[Path] = None,
    test_bed_path: Optional[Path] = None,
    genome_path: Optional[Path] = None,
    hyenadna_checkpoint_path: Optional[str] = None,
    layer_idx: Optional[int] = None,
    fidelity_batch_size: int = 4,
    fidelity_max_seq_len: int = 1024,
    resume: bool = False,
):
    """
    Evaluate SAE on pre-saved val and test embeddings stored as sharded .pt files.

    Each embeddings directory should contain shard_*.pt files, where each shard
    is a dict {"emb": tensor of shape [B, L, D]}.  Metric computation streams
    shards in mini-batches of `eval_batch_size` tokens so that the full token
    matrix is never materialised on the GPU.

    Args:
        sae_path: Path to SAE .pt checkpoint file
        cfg_path: Path to SAE config JSON file
        val_embeddings_path: Directory containing val shard_*.pt files
        test_embeddings_path: Directory containing test shard_*.pt files
        output_file: Path to save YAML results (default: print to stdout)
        device_str: Torch device string (e.g. "cuda", "cpu")
        eval_batch_size: Tokens per mini-batch for reconstruction/sparsity
                         metrics.  Reduce if you still hit OOM (default 4096).
        val_bed_path: BED file for val sequences — required for fidelity
        test_bed_path: BED file for test sequences — required for fidelity
        genome_path: Path to genome FASTA — required for fidelity
        hyenadna_checkpoint_path: HuggingFace model ID or local path to HyenaDNA checkpoint
        layer_idx: Layer to patch for fidelity evaluation
        fidelity_batch_size: Batch size for fidelity (unused, kept for future batching)
        fidelity_max_seq_len: Truncate sequences to this length for fidelity eval
        resume: Skip embedding shards (and cached fidelity) already processed;
                requires --output_file. State lives beside the YAML output.
    """

    run_fidelity = all([
        val_bed_path is not None,
        test_bed_path is not None,
        genome_path is not None,
        hyenadna_checkpoint_path is not None,
        layer_idx is not None,
    ])

    if not run_fidelity and any([
        val_bed_path, test_bed_path, genome_path, hyenadna_checkpoint_path, layer_idx
    ]):
        print(
            "⚠️  Partial fidelity args provided. To run fidelity, supply ALL of:\n"
            "   --val_bed_path, --test_bed_path, --genome_path,\n"
            "   --hyenadna_checkpoint_path, --layer_idx\n"
            "Skipping fidelity evaluation."
        )

    print("=" * 70)
    print("SAE Evaluation (HyenaDNA / pre-saved embeddings)")
    print("=" * 70)
    print(f"SAE:              {sae_path}")
    print(f"Val embeddings:   {val_embeddings_path}")
    print(f"Test embeddings:  {test_embeddings_path}")
    if run_fidelity:
        print(f"HyenaDNA:         {hyenadna_checkpoint_path}")
        print(f"Layer idx:        {layer_idx}")
    print()

    # ---- Load SAE ----
    print(f"Device: {device_str}")

    cfg = load_cfg(cfg_path=cfg_path, device=device_str)
    print(f"Loaded config keys: {list(cfg.keys())}  ")
    sae = load_sae(cfg, checkpoint_path=sae_path, device=device_str)
    print(f"SAE: {cfg['dict_size']} features, {cfg['act_size']}D\n")

    use_resume = bool(resume and output_file)
    if resume and not output_file:
        print("⚠️  --resume ignored without --output_file (no resume directory).")
    resume_dir = _eval_resume_dir(Path(output_file)) if use_resume else None

    results = {
        "sae_path": str(sae_path),
        "val_embeddings_path": str(val_embeddings_path),
        "test_embeddings_path": str(test_embeddings_path),
        "sae_dict_size": cfg['dict_size'],
        "sae_activation_dim": cfg['act_size'],
    }

    # ---- Evaluate on each split ----
    split_configs = [
        ("val",  val_embeddings_path,  val_bed_path),
        ("test", test_embeddings_path, test_bed_path),
    ]

    for split_name, emb_dir, bed_path in split_configs:
        if not emb_dir:
            print(f"⚠️  No embeddings path provided for {split_name} split. Skipping.")
            continue
        print("=" * 70)
        print(f"SPLIT: {split_name.upper()}")
        print("=" * 70)

        # Discover shards for this split's directory
        shard_paths = get_shard_paths(Path(emb_dir))
        print(f"\nFound {len(shard_paths)} shard(s) in {emb_dir}")

        paths_sha = _eval_paths_sha(shard_paths)
        split_fp = _eval_split_fingerprint(
            sae_path, cfg_path, Path(emb_dir), eval_batch_size, paths_sha
        )
        split_fp["split"] = split_name

        recon_pt = (resume_dir / f"{split_name}_recon.pt") if resume_dir else None
        sparse_pt = (resume_dir / f"{split_name}_sparsity.pt") if resume_dir else None
        fid_pt = (resume_dir / f"{split_name}_fidelity.pt") if resume_dir else None

        # 1. Reconstruction
        print(f"\n--- 1. Reconstruction Quality ({split_name}) ---")
        recon_metrics = calculate_reconstruction_metrics(
            sae,
            shard_paths,
            batch_size=eval_batch_size,
            device=device_str,
            resume=use_resume,
            resume_path=recon_pt,
            split_fp=split_fp,
        )
        n_tokens = int(recon_metrics.pop("n_tokens"))
        results[split_name] = {"n_tokens": n_tokens}
        print(f"  Total tokens: {n_tokens:,}")
        results[split_name]["reconstruction"] = recon_metrics
        for k, v in recon_metrics.items():
            print(f"  {k}: {v:.6f}")

        # 2. Sparsity
        print(f"\n--- 2. Sparsity ({split_name}) ---")
        sparsity_metrics = calculate_sparsity_metrics(
            sae,
            shard_paths,
            batch_size=eval_batch_size,
            device=device_str,
            resume=use_resume,
            resume_path=sparse_pt,
            split_fp=split_fp,
        )
        sparsity_metrics.pop("n_tokens", None)
        results[split_name]["sparsity"] = sparsity_metrics
        for k, v in sparsity_metrics.items():
            fmt = f"{v:.2f}%" if ("pct" in k or "freq" in k) else str(v)
            print(f"  {k}: {fmt}")

        # 3. Fidelity
        if run_fidelity:
            print(f"\n--- 3. Fidelity — Loss Recovered ({split_name}) ---")
            fid_fp = _eval_fidelity_fingerprint(
                split_fp,
                Path(bed_path),
                hyenadna_checkpoint_path,
                layer_idx,
                fidelity_max_seq_len,
            )
            loaded_fid = False
            if use_resume and fid_pt is not None and fid_pt.is_file():
                blob = torch.load(fid_pt, map_location="cpu")
                if _eval_fid_match(blob.get("fingerprint", {}), fid_fp):
                    results[split_name]["fidelity"] = blob["metrics"]
                    loaded_fid = True
                    print("  [resume] Fidelity metrics loaded from cache.")
                    for k, v in results[split_name]["fidelity"].items():
                        fmt = f"{v:.2f}%" if "pct" in k else f"{v:.6f}"
                        print(f"  {k}: {fmt}")

            if not loaded_fid:
                print("  Patching HyenaDNA activations with SAE reconstructions...")
                genome = Fasta(genome_path, as_raw=True, sequence_always_upper=True)

                fidelity_eval = HyenaDNAFidelityEvaluator(
                    checkpoint_path=hyenadna_checkpoint_path,
                    bed_path=str(bed_path),
                    genome=genome,
                    layer_idx=layer_idx,
                    batch_size=fidelity_batch_size,
                    max_seq_len=fidelity_max_seq_len,
                    device=device_str,
                )
                fidelity_metrics = fidelity_eval.calculate_fidelity(sae)
                results[split_name]["fidelity"] = fidelity_metrics

                for k, v in fidelity_metrics.items():
                    fmt = f"{v:.2f}%" if "pct" in k else f"{v:.6f}"
                    print(f"  {k}: {fmt}")

                if use_resume and fid_pt is not None:
                    _eval_save_resume_tmp(
                        fid_pt,
                        {"fingerprint": fid_fp, "metrics": fidelity_metrics},
                    )
        else:
            results[split_name]["fidelity"] = "skipped"
            print(f"\n--- 3. Fidelity ({split_name}): SKIPPED ---")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Save / print ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to: {output_file}")
        if use_resume and resume_dir is not None and resume_dir.is_dir():
            shutil.rmtree(resume_dir, ignore_errors=True)
    else:
        print(yaml.dump(results, default_flow_style=False))

    print("✅ Evaluation complete!")
    return results


import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAE model")

    # Required args
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--val_embeddings_path", type=Path, required=True,
                        help="Directory containing val shard_*.pt files")
    parser.add_argument("--test_embeddings_path", type=Path, required=True,
                        help="Directory containing test shard_*.pt files")

    # Optional args
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--device_str", type=str, default="cuda")
    parser.add_argument(
        "--eval_batch_size", type=int, default=4096,
        help="Tokens per mini-batch for reconstruction/sparsity metrics. "
             "Reduce (e.g. 512) if you hit OOM. (default: 4096)",
    )

    # Fidelity args
    parser.add_argument("--val_bed_path", type=Path, default=None)
    parser.add_argument("--test_bed_path", type=Path, default=None)
    parser.add_argument("--genome_path", type=Path, default=None)
    parser.add_argument("--hyenadna_checkpoint_path", type=str, default=None)
    parser.add_argument("--layer_idx", type=int, default=None)
    parser.add_argument("--fidelity_batch_size", type=int, default=4)
    parser.add_argument("--fidelity_max_seq_len", type=int, default=1024)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip embedding shards already merged (and cached fidelity). Requires --output_file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_sae(
        sae_path=args.sae_path,
        cfg_path=args.cfg_path,
        val_embeddings_path=args.val_embeddings_path,
        test_embeddings_path=args.test_embeddings_path,
        output_file=args.output_file,
        device_str=args.device_str,
        eval_batch_size=args.eval_batch_size,
        val_bed_path=args.val_bed_path,
        test_bed_path=args.test_bed_path,
        genome_path=args.genome_path,
        hyenadna_checkpoint_path=args.hyenadna_checkpoint_path,
        layer_idx=args.layer_idx,
        fidelity_batch_size=args.fidelity_batch_size,
        fidelity_max_seq_len=args.fidelity_max_seq_len,
        resume=args.resume,
    )