#!/usr/bin/env python3
"""
Train a Batch-TopK Sparse Autoencoder (SAE) on token embeddings saved in shard .pt files.

Your data (per split, per layer):
  root/{train,val,test}/layer_{L}/shard_00000.pt ...
Each shard contains:
  - "emb": FloatTensor [B_shard, 2000, D]
  - (optional) "coords": list aligned with emb rows

Token shuffling objective:
  Batch size = 256 tokens, sampled as:
    - pick 256 distinct sequences (windows)
    - pick 1 random token position per sequence
  => zero same-sequence collisions inside a batch.

Batch-TopK:
  Given activations A in [B, M] after ReLU:
  - Flatten to [B*M], keep top K_total activations (K_total = B*k_per_token)
  - Zero everything else
  - Reconstruction: x_hat = z @ W_dec + b_dec

Loss:
  recon_loss = MSE(x_hat, x)
  sparsity_loss = l1_coeff * mean(|z|)
  total = recon_loss + sparsity_loss

Logging:
  - step/epoch, lr
  - train recon, sparsity, total
  - fraction_nonzero, mean_active_value
  - val recon, sparsity, total, fraction_nonzero
Checkpointing:
  - periodic + best-val
Activation stats saved for future scaling:
  per-feature (M): min, max, mean, std over TRAIN activations z (post-topk)

Usage example:
  python train_batchtopk.py \
    --data_root /path/to/embeddings \
    --split_train train --split_val val \
    --layer_dir_name layer_5 \
    --d_in 256 --d_sae 8192 \
    --batch_tokens 256 --seq_len 2000 \
    --k_per_token 8 \
    --l1_coeff 1e-4 \
    --lr 2e-4 --weight_decay 0.0 \
    --max_steps 200000 \
    --log_every 50 --val_every 2000 --ckpt_every 5000 \
    --out_dir runs/sae_layer5_bt8
"""

import argparse
import csv
import glob
import math
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Simple LRU cache for shard tensors
# ----------------------------
class LRUCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self._cache: OrderedDict[str, Dict] = OrderedDict()

    def get(self, path: str) -> Optional[Dict]:
        obj = self._cache.get(path, None)
        if obj is None:
            return None
        self._cache.move_to_end(path)
        return obj

    def put(self, path: str, obj: Dict) -> None:
        self._cache[path] = obj
        self._cache.move_to_end(path)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


# ----------------------------
# Index sequences across shards
# ----------------------------
@dataclass
class SeqRef:
    shard_path: str
    row_idx: int


def list_shards(split_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt found in {split_dir}")
    return paths


def build_sequence_index(layer_dir: str) -> Tuple[List[SeqRef], int]:
    """Returns (seq_refs, total_sequences). Loads each shard header to get emb.shape[0]."""
    shard_paths = list_shards(layer_dir)
    print(f"Total shard count {len(shard_paths)}")
    seq_refs: List[SeqRef] = []
    total = 0
    for sp in shard_paths:
        print(f"Building index for {sp}")
        obj = torch.load(sp, map_location="cpu")
        emb = obj["emb"]
        if emb.ndim != 3:
            raise ValueError(f"{sp}: expected emb [B, L, D], got {tuple(emb.shape)}")
        b = int(emb.shape[0])
        for r in range(b):
            seq_refs.append(SeqRef(shard_path=sp, row_idx=r))
        total += b
    return seq_refs, total


# ----------------------------
# Running per-feature stats for activations (Welford + min/max)
# ----------------------------
class RunningFeatureStats:
    """
    Tracks per-feature mean/std/min/max over streaming batches of z in [B, M].
    """
    def __init__(self, n_features: int, device: torch.device):
        self.n_features = n_features
        self.device = device
        self.count = torch.zeros((), dtype=torch.long, device=device)  # total samples accumulated
        self.mean = torch.zeros((n_features,), dtype=torch.float32, device=device)
        self.M2 = torch.zeros((n_features,), dtype=torch.float32, device=device)
        self.minv = torch.full((n_features,), float("inf"), dtype=torch.float32, device=device)
        self.maxv = torch.full((n_features,), float("-inf"), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        """
        z: [B, M], float32/float16 ok. Updates in float32.
        """
        if z.ndim != 2 or z.shape[1] != self.n_features:
            raise ValueError(f"stats.update expected [B,{self.n_features}], got {tuple(z.shape)}")
        zf = z.detach().to(dtype=torch.float32)

        # min/max
        self.minv = torch.minimum(self.minv, zf.min(dim=0).values)
        self.maxv = torch.maximum(self.maxv, zf.max(dim=0).values)

        # Welford batch update
        b = zf.shape[0]
        if b == 0:
            return
        batch_mean = zf.mean(dim=0)
        batch_M2 = ((zf - batch_mean) ** 2).sum(dim=0)

        n = self.count.to(dtype=torch.float32)
        b_f = torch.tensor(float(b), device=self.device)

        delta = batch_mean - self.mean
        new_n = n + b_f
        self.mean = self.mean + delta * (b_f / new_n)
        self.M2 = self.M2 + batch_M2 + (delta ** 2) * (n * b_f / new_n)
        self.count += b

    @torch.no_grad()
    def finalize(self) -> Dict[str, torch.Tensor]:
        count = int(self.count.item())
        if count < 2:
            var = torch.zeros_like(self.mean)
        else:
            var = self.M2 / torch.clamp(self.count.to(torch.float32) - 1.0, min=1.0)
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "count": torch.tensor(count, dtype=torch.long),
            "mean": self.mean.detach().cpu(),
            "std": std.detach().cpu(),
            "min": self.minv.detach().cpu(),
            "max": self.maxv.detach().cpu(),
        }


# ----------------------------
# Batch-TopK SAE
# ----------------------------
class BatchTopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        # Encoder/decoder
        self.W_enc = nn.Linear(d_in, d_sae, bias=True)
        self.W_dec = nn.Linear(d_sae, d_in, bias=True)

        # Optional: initialize decoder as transpose-like can help, but keep simple/robust
        nn.init.kaiming_uniform_(self.W_enc.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_enc.bias)
        nn.init.zeros_(self.W_dec.bias)

    def forward(self, x: torch.Tensor, k_total: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [B, D]
        k_total: number of nonzeros to keep across the whole batch (flattened).
        Returns: x_hat, z (post-topk), metrics
        """
        # pre-activation and ReLU
        a = F.relu(self.W_enc(x))  # [B, M]

        B, M = a.shape
        k_total = int(k_total)

        if k_total <= 0:
            z = torch.zeros_like(a)
        elif k_total >= B * M:
            z = a
        else:
            flat = a.view(-1)  # [B*M]
            # topk values/indices
            vals, idx = torch.topk(flat, k_total, largest=True, sorted=False)
            z = torch.zeros_like(flat)
            z[idx] = vals
            z = z.view(B, M)

        x_hat = self.W_dec(z)

        # diagnostics
        nnz = (z > 0).sum()
        frac_nnz = nnz.to(torch.float32) / (B * M)
        mean_active = (z.sum() / torch.clamp(nnz.to(torch.float32), min=1.0))

        metrics = {
            "frac_nnz": frac_nnz.detach(),
            "mean_active": mean_active.detach(),
            "nnz": nnz.detach(),
        }
        return x_hat, z, metrics


# ----------------------------
# Batch builder: sample 1 token per distinct sequence
# ----------------------------
@torch.no_grad()
def make_token_batch(
    seq_refs: List[SeqRef],
    cache: LRUCache,
    batch_seq_indices: List[int],
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns x: [B, D] where each item is one random token from one sequence.
    """
    # Group requested rows by shard to reduce loads
    by_shard: Dict[str, List[int]] = {}
    for gi in batch_seq_indices:
        sr = seq_refs[gi]
        by_shard.setdefault(sr.shard_path, []).append(sr.row_idx)

    tokens: List[torch.Tensor] = []
    for sp, rows in by_shard.items():
        obj = cache.get(sp)
        if obj is None:
            obj = torch.load(sp, map_location="cpu")
            cache.put(sp, obj)
        emb = obj["emb"]  # [B_shard, L, D] on CPU
        if int(emb.shape[1]) != seq_len:
            raise ValueError(f"{sp}: expected seq_len={seq_len}, got {int(emb.shape[1])}")

        # sample positions for each requested row
        # (sampling on CPU is fine)
        for r in rows:
            pos = random.randrange(seq_len)
            tok = emb[r, pos, :]  # [D]
            tokens.append(tok)

    x = torch.stack(tokens, dim=0).to(device=device, dtype=torch.float32)  # compute in fp32
    return x


# ----------------------------
# Logging helpers
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_csv_row(path: str, header: List[str], row: Dict[str, float]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

# ---- ETA helpers (drop-in) -----------------------------------------------
def format_seconds(s: float) -> str:
    s = int(max(0, s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m > 0:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"

# ----------------------------
# Training
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root containing train/val/test directories")
    ap.add_argument("--split_train", default="train")
    ap.add_argument("--split_val", default="val")
    ap.add_argument("--layer_dir_name", required=True, help="e.g., layer_5")
    ap.add_argument("--d_in", type=int, required=True, help="Embedding dim D")
    ap.add_argument("--d_sae", type=int, required=True, help="SAE hidden size M")
    ap.add_argument("--seq_len", type=int, default=2000)
    ap.add_argument("--batch_tokens", type=int, default=256, help="Tokens per batch (also sequences per batch)")
    ap.add_argument("--k_per_token", type=int, default=8, help="Top-k per token on average. K_total = B*k_per_token")
    ap.add_argument("--l1_coeff", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda or cpu")
    ap.add_argument("--cache_shards", type=int, default=8, help="LRU cache size (number of shard files)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--val_batches", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--save_best", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "checkpoints"))

    train_layer_dir = os.path.join(args.data_root, args.split_train, args.layer_dir_name)
    val_layer_dir = os.path.join(args.data_root, args.split_val, args.layer_dir_name)

    print("Building indices...")
    train_seq_refs, n_train_seq = build_sequence_index(train_layer_dir)
    val_seq_refs, n_val_seq = build_sequence_index(val_layer_dir)

    print(f"Train sequences: {n_train_seq}")
    print(f"Val sequences:   {n_val_seq}")
    if n_train_seq < args.batch_tokens:
        raise ValueError(f"Need at least batch_tokens sequences. Have {n_train_seq}, batch_tokens={args.batch_tokens}")

    model = BatchTopKSAE(d_in=args.d_in, d_sae=args.d_sae).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Simple LR schedule: cosine decay to 10% of lr (optional but helpful)
    def lr_at(step: int) -> float:
        if args.max_steps <= 1:
            return args.lr
        t = step / float(args.max_steps - 1)
        min_lr = 0.1 * args.lr
        return min_lr + 0.5 * (args.lr - min_lr) * (1.0 + math.cos(math.pi * t))

    train_cache = LRUCache(max_items=args.cache_shards)
    val_cache = LRUCache(max_items=max(2, args.cache_shards // 2))

    # Activation stats over TRAIN z (post-topk)
    act_stats = RunningFeatureStats(n_features=args.d_sae, device=device)

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    header = [
        "time_s",
        "step",
        "seq_epochs",
        "lr",
        "split",
        "recon_mse",
        "sparsity_l1",
        "total_loss",
        "frac_nnz",
        "mean_active",
    ]

    # Epoch accounting over sequences (true “pass” approximation via reshuffle cycles)
    # We implement a rolling permutation of sequence indices and take contiguous chunks.
    train_perm = list(range(n_train_seq))
    random.shuffle(train_perm)
    train_ptr = 0
    seq_epochs = 0.0
    best_val = float("inf")

    start_time = time.time()
    last_log_time = start_time
    step_times = []  # moving average of seconds/step over recent log intervals
    model.train()

    for step in range(1, args.max_steps + 1):
        # Refresh permutation if needed
        if train_ptr + args.batch_tokens > n_train_seq:
            # completed one pass over sequences (sequence-epoch)
            random.shuffle(train_perm)
            train_ptr = 0

        batch_seq_indices = train_perm[train_ptr:train_ptr + args.batch_tokens]
        train_ptr += args.batch_tokens
        seq_epochs = (step * args.batch_tokens) / float(n_train_seq)

        x = make_token_batch(
            seq_refs=train_seq_refs,
            cache=train_cache,
            batch_seq_indices=batch_seq_indices,
            seq_len=args.seq_len,
            device=device,
        )  # [B, D]

        k_total = args.batch_tokens * args.k_per_token

        x_hat, z, z_metrics = model(x, k_total=k_total)

        recon = F.mse_loss(x_hat, x)
        sparsity = z.abs().mean()  # mean L1 per element (post-topk)
        loss = recon + args.l1_coeff * sparsity

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # LR schedule
        lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Update activation stats (TRAIN only)
        act_stats.update(z)

        # Logging
        if step % args.log_every == 0:
            row = {
                "time_s": time.time() - start_time,
                "step": step,
                "seq_epochs": seq_epochs,
                "lr": lr,
                "split": "train",
                "recon_mse": float(recon.detach().cpu().item()),
                "sparsity_l1": float(sparsity.detach().cpu().item()),
                "total_loss": float(loss.detach().cpu().item()),
                "frac_nnz": float(z_metrics["frac_nnz"].detach().cpu().item()),
                "mean_active": float(z_metrics["mean_active"].detach().cpu().item()),
            }
            write_csv_row(metrics_path, header, row)
            now = time.time()
            steps_left = args.max_steps - step

            # seconds/step estimated over last log interval, smoothed over last 50 intervals
            interval_sec_per_step = (now - last_log_time) / float(args.log_every)
            last_log_time = now
            step_times.append(interval_sec_per_step)
            if len(step_times) > 50:
                step_times.pop(0)

            avg_sec_per_step = sum(step_times) / len(step_times)
            eta_str = format_seconds(avg_sec_per_step * steps_left)

            print(
                f"[train] step={step}/{args.max_steps} seq_epochs={seq_epochs:.3f} "
                f"recon={row['recon_mse']:.6g} l1={row['sparsity_l1']:.6g} "
                f"loss={row['total_loss']:.6g} frac_nnz={row['frac_nnz']:.3g} "
                f"mean_active={row['mean_active']:.3g} ETA={eta_str}"
            )

        # Validation
        if step % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                val_recon_sum = 0.0
                val_l1_sum = 0.0
                val_loss_sum = 0.0
                val_frac_sum = 0.0
                val_meanact_sum = 0.0

                # sample batches by random distinct sequences (per batch)
                for _ in range(args.val_batches):
                    # choose distinct sequences without replacement inside the batch
                    batch_seq_indices = random.sample(range(n_val_seq), k=args.batch_tokens)
                    x = make_token_batch(
                        seq_refs=val_seq_refs,
                        cache=val_cache,
                        batch_seq_indices=batch_seq_indices,
                        seq_len=args.seq_len,
                        device=device,
                    )
                    x_hat, z, z_metrics = model(x, k_total=args.batch_tokens * args.k_per_token)
                    recon = F.mse_loss(x_hat, x)
                    sparsity = z.abs().mean()
                    loss = recon + args.l1_coeff * sparsity

                    val_recon_sum += float(recon.cpu().item())
                    val_l1_sum += float(sparsity.cpu().item())
                    val_loss_sum += float(loss.cpu().item())
                    val_frac_sum += float(z_metrics["frac_nnz"].cpu().item())
                    val_meanact_sum += float(z_metrics["mean_active"].cpu().item())

                denom = float(args.val_batches)
                row = {
                    "time_s": time.time() - start_time,
                    "step": step,
                    "seq_epochs": seq_epochs,
                    "lr": lr,
                    "split": "val",
                    "recon_mse": val_recon_sum / denom,
                    "sparsity_l1": val_l1_sum / denom,
                    "total_loss": val_loss_sum / denom,
                    "frac_nnz": val_frac_sum / denom,
                    "mean_active": val_meanact_sum / denom,
                }
                write_csv_row(metrics_path, header, row)
                print(
                    f"[val]   step={step} "
                    f"recon={row['recon_mse']:.6g} l1={row['sparsity_l1']:.6g} "
                    f"loss={row['total_loss']:.6g} frac_nnz={row['frac_nnz']:.3g} "
                    f"mean_active={row['mean_active']:.3g}"
                )

                if args.save_best and row["total_loss"] < best_val:
                    best_val = row["total_loss"]
                    ckpt_path = os.path.join(args.out_dir, "checkpoints", "best.pt")
                    stats = act_stats.finalize()
                    torch.save(
                        {
                            "step": step,
                            "seq_epochs": seq_epochs,
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "args": vars(args),
                            "best_val_loss": best_val,
                            "activation_stats": stats,
                        },
                        ckpt_path,
                    )
                    print(f"Saved best checkpoint to {ckpt_path}")

            model.train()

        # Periodic checkpoint
        if step % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.out_dir, "checkpoints", f"step_{step:08d}.pt")
            stats = act_stats.finalize()
            torch.save(
                {
                    "step": step,
                    "seq_epochs": seq_epochs,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                    "best_val_loss": best_val,
                    "activation_stats": stats,
                },
                ckpt_path,
            )
            # also write activation stats as a standalone file for easy reuse
            stats_path = os.path.join(args.out_dir, "activation_stats_latest.pt")
            torch.save(stats, stats_path)
            print(f"Saved checkpoint to {ckpt_path}")
            print(f"Saved activation stats to {stats_path}")

    # Final save
    final_ckpt = os.path.join(args.out_dir, "checkpoints", "final.pt")
    stats = act_stats.finalize()
    torch.save(
        {
            "step": args.max_steps,
            "seq_epochs": (args.max_steps * args.batch_tokens) / float(n_train_seq),
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val,
            "activation_stats": stats,
        },
        final_ckpt,
    )
    torch.save(stats, os.path.join(args.out_dir, "activation_stats_final.pt"))
    print(f"Done. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()