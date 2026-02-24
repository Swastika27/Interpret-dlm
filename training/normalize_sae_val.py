#!/usr/bin/env python3
"""
interPLM-style feature normalization for your SAE trained on HyenaDNA embeddings.

What it does (per feature j):
  s_j = max activation over a calibration set (your val shards)
  W_e[:, j] <- W_e[:, j] / s_j
  b_e[j]    <- b_e[j] / s_j
  W_d[:, j] <- W_d[:, j] * s_j

This makes (approximately) a'_j = a_j / s_j so activations become comparable across features,
while preserving reconstructions x_hat (except for edge cases at ReLU threshold).

Inputs:
  - SAE checkpoint .pt produced by your training script (contains model_state, d_in, d_hidden)
  - Validation embedding shards (.pt) with key "emb": [B, L, D] saved by your extraction script

Output:
  - New checkpoint with normalized weights + saved "feature_scales" vector.

Example:
  python normalize_sae_interplm.py \
    --ckpt runs/sae/.../ckpt_step_00020000.pt \
    --val_layer_dir data/embeddings/val/layer_5 \
    --out_ckpt runs/sae/.../ckpt_step_00020000.interplm_norm.pt \
    --device cuda --batch_tokens 8192
"""

import argparse
import glob
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Your SAE definition (must match training) ---
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, use_relu: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.use_relu = use_relu

        self.W_e = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_e = nn.Parameter(torch.zeros(d_hidden))
        self.W_d = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_d = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_e + self.b_e
        return F.relu(pre) if self.use_relu else pre

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_d.T + self.b_d

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.encode(x)
        x_hat = self.decode(a)
        return x_hat, a


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to SAE checkpoint .pt")
    ap.add_argument("--val_layer_dir", required=True, help="Dir with val layer shards: shard_*.pt")
    ap.add_argument("--out_ckpt", required=True, help="Output normalized checkpoint path")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_tokens", type=int, default=8192, help="Tokens per forward pass")
    ap.add_argument("--max_shards", type=int, default=0, help="0 = use all shards, else limit")
    ap.add_argument("--eps", type=float, default=1e-8, help="Clamp min for scales (dead features)")
    ap.add_argument("--compute_dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    return ap.parse_args()


def to_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


@torch.inference_mode()
def compute_feature_maxima(
    model: SparseAutoencoder,
    shard_paths,
    device: torch.device,
    batch_tokens: int,
    eps: float,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Scans val shards and returns s: [d_hidden] where s_j = max activation of feature j.
    """
    d_hidden = model.d_hidden
    s = torch.zeros(d_hidden, device=device, dtype=torch.float32)

    # Put model in eval (no dropout/bn here, but correct practice)
    model.eval()

    for si, p in enumerate(shard_paths):
        obj = torch.load(p, map_location="cpu")
        if "emb" not in obj:
            raise KeyError(f"{p}: missing key 'emb'")
        emb = obj["emb"]  # [B,L,D] on CPU (float16/float32)
        if emb.ndim != 3:
            raise ValueError(f"{p}: expected emb rank 3 [B,L,D], got {tuple(emb.shape)}")
        B, L, D = emb.shape
        if D != model.d_in:
            raise ValueError(f"{p}: d_in mismatch. shard D={D}, model d_in={model.d_in}")

        x_all = emb.reshape(B * L, D)  # [N, D] CPU
        n = x_all.size(0)

        # Process in token batches to control GPU RAM
        for start in range(0, n, batch_tokens):
            end = min(n, start + batch_tokens)
            x = x_all[start:end].to(device=device, dtype=compute_dtype, non_blocking=True)
            a = model.encode(x)  # [b, d_hidden]

            # Update max per feature (do max in float32 for stability)
            a_max = a.max(dim=0).values.to(dtype=torch.float32)
            s = torch.maximum(s, a_max)

        if (si + 1) % 10 == 0:
            print(f"Scanned {si+1}/{len(shard_paths)} shards")

    s = s.clamp_min(eps)
    return s


@torch.inference_mode()
def apply_interplm_scaling_(model: SparseAutoencoder, s: torch.Tensor) -> None:
    """
    In-place reciprocal scaling:
      W_e[:,j] /= s_j
      b_e[j]   /= s_j
      W_d[:,j] *= s_j
    """
    if s.ndim != 1 or s.numel() != model.d_hidden:
        raise ValueError(f"Bad scale shape {tuple(s.shape)} (expected [{model.d_hidden}])")

    # Ensure s is on same device as params
    s = s.to(device=model.W_e.device, dtype=torch.float32)

    # Encoder: divide feature columns
    model.W_e.div_(s.view(1, -1))
    model.b_e.div_(s)

    # Decoder: multiply feature columns
    model.W_d.mul_(s.view(1, -1))
    # b_d unchanged


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    compute_dtype = to_dtype(args.compute_dtype)

    shard_paths = sorted(glob.glob(os.path.join(args.val_layer_dir, "shard_*.pt")))
    if not shard_paths:
        raise SystemExit(f"No shards found under: {args.val_layer_dir}/shard_*.pt")
    if args.max_shards and args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model_state" not in ckpt:
        raise KeyError("Checkpoint missing 'model_state'")

    d_in = int(ckpt.get("d_in", 0))
    d_hidden = int(ckpt.get("d_hidden", 0))
    if d_in <= 0 or d_hidden <= 0:
        raise ValueError("Checkpoint missing or invalid d_in/d_hidden")

    model = SparseAutoencoder(d_in=d_in, d_hidden=d_hidden, use_relu=True).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    print(f"Loaded SAE: d_in={d_in}, d_hidden={d_hidden}")
    print(f"Scanning {len(shard_paths)} val shards for per-feature maxima...")

    s = compute_feature_maxima(
        model=model,
        shard_paths=shard_paths,
        device=device,
        batch_tokens=args.batch_tokens,
        eps=args.eps,
        compute_dtype=compute_dtype,
    )
    print("Done. Example scales:", s[:10].detach().cpu().tolist())

    # Apply scaling in-place
    apply_interplm_scaling_(model, s)

    # Save new checkpoint
    out = dict(ckpt)
    out["model_state"] = model.to("cpu").state_dict()
    out["feature_scales"] = s.detach().cpu()
    out["feature_scale_type"] = "max_activation_val_split"
    out["feature_scale_source"] = {
        "val_layer_dir": os.path.abspath(args.val_layer_dir),
        "num_shards_scanned": len(shard_paths),
        "batch_tokens": args.batch_tokens,
        "eps": args.eps,
        "compute_dtype": args.compute_dtype,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_ckpt)), exist_ok=True)
    torch.save(out, args.out_ckpt)
    print("Wrote normalized checkpoint:", args.out_ckpt)


if __name__ == "__main__":
    main()