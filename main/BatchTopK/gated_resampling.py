"""
Dead-feature resampling for Gated SAE (Rajamanoharan et al., arXiv:2404.16014, Appendix D).

Neuron resampling follows Bricken et al. (2023) more closely than a uniform random baseline:
  - Input directions are sampled with probability proportional to per-example squared
    reconstruction error (harder-to-reconstruct activations seed new features).
  - Encoder columns are scaled by gated_resample_enc_norm_frac * mean L2 norm of alive
    encoder columns (decoder rows stay unit-norm for the tied dictionary direction).

Post-resample LR: 0.1× base, cosine to 1.0× over gated_resample_warmup_steps.

Contract (warm-up state):
  - Call apply_gated_lr(optimizer, cfg, gated_state, base_lr) at the start of each
    training step when using gated resampling.
  - Pass the same gated_state dict into resample_dead_gated_features(..., gated_state);
    if any feature is resampled, warm-up is reset (in_warmup=True, warmup_step=0).
  - Do not set warm-up fields manually unless you skip resample_dead_gated_features.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from sae import GatedSAE


@torch.no_grad()
def _per_sample_recon_sq_sum(sae: GatedSAE, batch: torch.Tensor) -> torch.Tensor:
    """
    Squared reconstruction error per batch row in preprocessed space (matches L_recon).

    preprocess_input must not subtract b_dec; we apply x_cent = x - b_dec here only.
    """
    x, _, _ = sae.preprocess_input(batch)
    x_cent = x - sae.b_dec
    gating_pre = x_cent @ sae.W_enc + sae.b_gate
    active = (gating_pre > 0).to(x.dtype)
    mag_pre = x_cent @ (sae.W_enc * sae.r_mag.exp()) + sae.b_mag
    mags = F.relu(mag_pre)
    feature_acts = active * mags
    x_reconstruct = feature_acts @ sae.W_dec + sae.b_dec
    return (x_reconstruct - x).pow(2).sum(dim=-1)


@torch.no_grad()
def _avg_alive_encoder_column_norm(sae: GatedSAE, dead_indices: torch.Tensor) -> torch.Tensor:
    """Mean L2 norm of W_enc columns for features not in dead_indices."""
    d = sae.cfg["dict_size"]
    device = sae.W_enc.device
    dtype = sae.W_enc.dtype
    alive = torch.ones(d, dtype=torch.bool, device=device)
    if dead_indices.numel() > 0:
        alive[dead_indices] = False
    if not alive.any():
        return torch.tensor(1.0, device=device, dtype=dtype)
    cols = sae.W_enc[:, alive]
    return cols.norm(dim=0).mean().clamp(min=1e-8)


@torch.no_grad()
def resample_dead_gated_features(
    sae: GatedSAE,
    batch: torch.Tensor,
    cfg: dict,
    gated_state: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Reinitialize dead features (Bricken-style: loss-weighted inputs + encoder scaling).

    If gated_state is not None and at least one feature is resampled, sets
    gated_state['in_warmup']=True and gated_state['warmup_step']=0 for LR warm-up.

    Returns:
        Number of features resampled.
    """
    thresh = int(cfg.get("gated_resample_dead_batches", 200))
    dead = (sae.num_batches_not_active >= thresh).nonzero(as_tuple=False).squeeze(-1)
    if dead.numel() == 0:
        return 0

    x, _, _ = sae.preprocess_input(batch)
    x_cent = x - sae.b_dec
    n = batch.shape[0]

    loss_weighted = bool(cfg.get("gated_resample_loss_weighted", True))
    if loss_weighted:
        sq = _per_sample_recon_sq_sum(sae, batch)
        w = sq + 1e-8
        if w.sum() <= 0:
            w = torch.ones_like(w)
        idx_choices = torch.multinomial(
            w, num_samples=dead.numel(), replacement=True
        )
    else:
        idx_choices = torch.randint(0, n, (dead.numel(),), device=batch.device)

    avg_enc_norm = _avg_alive_encoder_column_norm(sae, dead)
    frac = float(cfg.get("gated_resample_enc_norm_frac", 0.2))
    scale = frac * avg_enc_norm

    for k, j in enumerate(dead.tolist()):
        idx = int(idx_choices[k])
        v = x_cent[idx].squeeze(0)
        d_unit = F.normalize(v, dim=0, eps=1e-8)
        sae.W_dec.data[j] = d_unit
        sae.W_enc.data[:, j] = d_unit * scale
        sae.b_gate.data[j] = 0
        sae.b_mag.data[j] = 0
        sae.r_mag.data[j] = 0
        sae.num_batches_not_active[j] = 0

    n_res = int(dead.numel())
    if gated_state is not None and n_res > 0:
        gated_state["in_warmup"] = True
        gated_state["warmup_step"] = 0
    return n_res


def gated_lr_multiplier_after_resample(warmup_step: int, warmup_steps: int) -> float:
    """
    Cosine from exactly 0.1 at warmup_step=0 to exactly 1.0 at warmup_step=warmup_steps-1.

    For warmup_step >= warmup_steps, returns 1.0 (caller should exit warm-up).
    """
    if warmup_step >= warmup_steps:
        return 1.0
    if warmup_steps <= 1:
        return 1.0
    t = warmup_step / float(warmup_steps - 1)
    min_m, max_m = 0.1, 1.0
    return min_m + (max_m - min_m) * (1.0 - math.cos(math.pi * t)) / 2.0


def apply_gated_lr(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    state: Dict[str, Any],
    base_lr: float,
) -> None:
    """Set optimizer LR for gated training (post-resample cosine warm-up)."""
    warmup_steps = int(cfg.get("gated_resample_warmup_steps", 1000))
    if state.get("in_warmup", False):
        ws = int(state.get("warmup_step", 0))
        mult = gated_lr_multiplier_after_resample(ws, warmup_steps)
    else:
        mult = 1.0
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * mult
