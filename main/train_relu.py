#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LRUCache:
    """Simple LRU cache for loaded shards to reduce disk I/O."""
    def __init__(self, max_items: int = 32):
        self.max_items = max_items
        self._cache: OrderedDict[str, Dict] = OrderedDict()

    def get(self, path: str) -> Optional[Dict]:
        if path not in self._cache:
            return None
        self._cache.move_to_end(path)
        return self._cache[path]

    def put(self, path: str, obj: Dict) -> None:
        self._cache[path] = obj
        self._cache.move_to_end(path)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


# ----------------------------
# Dataset: sample tokens from distinct windows
# ----------------------------
@dataclass(frozen=True)
class WindowRef:
    shard_path: str
    win_idx: int  # index into first dim of emb in that shard


class TokenSampler:
    """
    Builds batches by selecting distinct windows, then selecting a random token from each window.
    This guarantees no two tokens in a batch come from the same window (unless user requests >1 token per window).
    """
    def __init__(
        self,
        shard_paths: List[str],
        seq_len: int,
        cache_max: int = 32,
        map_location: str = "cpu",
        assume_one_window_per_shard: bool = False,
    ):
        self.shard_paths = shard_paths
        self.seq_len = seq_len
        self.cache = LRUCache(max_items=cache_max)
        self.map_location = map_location
        self.assume_one_window_per_shard = assume_one_window_per_shard

        # Build window index: list of WindowRef entries
        # If assume_one_window_per_shard=True, we avoid loading all shards at init.
        # Otherwise we load each shard once to find emb.shape[0].
        self.windows: List[WindowRef] = []
        if assume_one_window_per_shard:
            for p in shard_paths:
                self.windows.append(WindowRef(p, 0))
        else:
            for p in shard_paths:
                obj = torch.load(p, map_location=self.map_location)
                emb = obj["emb"]
                if emb.ndim != 3:
                    raise ValueError(f"{p}: expected emb rank 3 [Bwin,L,D], got {tuple(emb.shape)}")
                bwin, L, _D = emb.shape
                if L != self.seq_len:
                    raise ValueError(f"{p}: seq_len mismatch: expected {self.seq_len}, got {L}")
                for wi in range(bwin):
                    self.windows.append(WindowRef(p, wi))

        if len(self.windows) == 0:
            raise ValueError("No windows found from shard_paths")

    def _load_shard(self, shard_path: str) -> Dict:
        cached = self.cache.get(shard_path)
        if cached is not None:
            return cached
        obj = torch.load(shard_path, map_location=self.map_location)
        if "emb" not in obj:
            raise KeyError(f"{shard_path}: missing key 'emb'")
        self.cache.put(shard_path, obj)
        return obj

    def sample_batch_tokens(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        tokens_per_window: int = 1,
    ) -> torch.Tensor:
        """
        Returns X of shape [batch_size * tokens_per_window, d_model]
        with no duplicate windows within the batch (for tokens_per_window=1).
        """
        if tokens_per_window < 1:
            raise ValueError("tokens_per_window must be >= 1")
        if batch_size > len(self.windows):
            raise ValueError(f"batch_size {batch_size} > num_windows {len(self.windows)}")

        # Choose distinct windows
        chosen = random.sample(self.windows, k=batch_size)

        xs: List[torch.Tensor] = []
        for wref in chosen:
            obj = self._load_shard(wref.shard_path)
            emb = obj["emb"]  # [Bwin, L, D] on CPU
            # pick tokens_per_window distinct positions
            if tokens_per_window == 1:
                pos = random.randrange(self.seq_len)
                x = emb[wref.win_idx, pos, :]  # [D]
                xs.append(x)
            else:
                # If you ever want more than 1 token per window, keep them distinct per window
                positions = random.sample(range(self.seq_len), k=tokens_per_window)
                x = emb[wref.win_idx, positions, :]  # [k, D]
                xs.append(x)

        X = torch.cat([t.unsqueeze(0) if t.ndim == 1 else t for t in xs], dim=0)  # [B*k, D]
        X = X.to(device=device, dtype=dtype, non_blocking=True)
        return X


# ----------------------------
# SAE model
# ----------------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, use_relu: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.use_relu = use_relu

        # Encoder: a = f(x W_e + b_e)
        self.W_e = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_e = nn.Parameter(torch.zeros(d_hidden))

        # Decoder: x_hat = a W_d^T + b_d  (store W_d as [d_in, d_hidden] for convenience)
        self.W_d = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_d = nn.Parameter(torch.zeros(d_in))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_e, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W_d, a=5 ** 0.5)
        nn.init.zeros_(self.b_e)
        nn.init.zeros_(self.b_d)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_in]
        pre = x @ self.W_e + self.b_e
        if self.use_relu:
            return F.relu(pre)
        return pre

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        # a: [B, d_hidden]
        return a @ self.W_d.T + self.b_d  # [B, d_in]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.encode(x)
        x_hat = self.decode(a)
        return x_hat, a

    @torch.no_grad()
    def normalize_decoder_columns_(self, eps: float = 1e-8) -> None:
        # Normalize decoder columns to unit norm (common SAE stabilization trick)
        col_norms = torch.linalg.norm(self.W_d, dim=0, keepdim=True).clamp_min(eps)  # [1, d_hidden]
        self.W_d.div_(col_norms)


# ----------------------------
# Training
# ----------------------------
def run_eval(
    model: SparseAutoencoder,
    sampler: TokenSampler,
    device: torch.device,
    steps: int,
    batch_size: int,
    l1_coeff: float,
    tokens_per_window: int = 1,
) -> Dict[str, float]:
    model.eval()
    recon_losses = []
    l1_losses = []
    total_losses = []
    mean_active = []
    with torch.inference_mode():
        for _ in range(steps):
            x = sampler.sample_batch_tokens(
                batch_size=batch_size,
                device=device,
                dtype=torch.float32,
                tokens_per_window=tokens_per_window,
            )
            x_hat, a = model(x)
            recon = F.mse_loss(x_hat, x)
            l1 = a.abs().mean()
            loss = recon + l1_coeff * l1

            recon_losses.append(float(recon))
            l1_losses.append(float(l1))
            total_losses.append(float(loss))

            # “active” fraction: fraction of features with activation > 0 (ReLU) averaged over batch
            if model.use_relu:
                mean_active.append(float((a > 0).float().mean()))
            else:
                mean_active.append(float((a.abs() > 1e-6).float().mean()))

    return {
        "loss": sum(total_losses) / len(total_losses),
        "recon_mse": sum(recon_losses) / len(recon_losses),
        "l1_mean": sum(l1_losses) / len(l1_losses),
        "active_frac": sum(mean_active) / len(mean_active),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_root", type=str, default="data/embeddings", help="Root dir containing train/val/test folders")
    ap.add_argument("--layer", type=int, required=True, help="Layer index folder name: layer_{L}")
    ap.add_argument("--seq_len", type=int, required=True, help="Expected window length L in tokens")
    ap.add_argument("--d_hidden", type=int, default=4096, help="SAE hidden size (#features)")
    ap.add_argument("--batch_size", type=int, default=2048, help="Tokens per batch (one token per window by default)")
    ap.add_argument("--tokens_per_window", type=int, default=1, help="Tokens sampled per window (keep 1 to avoid duplicates)")
    ap.add_argument("--steps", type=int, default=20000, help="Training steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--l1_coeff", type=float, default=1e-3, help="L1 penalty coefficient on activations")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=50, help="How many batches to average for eval")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--out_dir", type=str, default="runs/sae", help="Output dir for logs and checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_max", type=int, default=32, help="How many shard files to keep cached in RAM")
    ap.add_argument("--assume_one_window_per_shard", action="store_true",
                    help="Set if each shard contains exactly one window (faster init)")
    ap.add_argument("--tensorboard", action="store_true", help="Also log to TensorBoard if installed")
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve shard paths
    train_dir = os.path.join(args.emb_root, "train", f"layer_{args.layer}")
    val_dir = os.path.join(args.emb_root, "val", f"layer_{args.layer}")
    test_dir = os.path.join(args.emb_root, "test", f"layer_{args.layer}")

    train_shards = sorted(glob.glob(os.path.join(train_dir, "shard_*.pt")))
    val_shards = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    test_shards = sorted(glob.glob(os.path.join(test_dir, "shard_*.pt")))

    if not train_shards:
        raise SystemExit(f"No training shards found in {train_dir}")
    if not val_shards:
        raise SystemExit(f"No validation shards found in {val_dir}")

    # Create samplers
    train_sampler = TokenSampler(
        shard_paths=train_shards,
        seq_len=args.seq_len,
        cache_max=args.cache_max,
        map_location="cpu",
        assume_one_window_per_shard=args.assume_one_window_per_shard,
    )
    val_sampler = TokenSampler(
        shard_paths=val_shards,
        seq_len=args.seq_len,
        cache_max=max(4, args.cache_max // 2),
        map_location="cpu",
        assume_one_window_per_shard=args.assume_one_window_per_shard,
    )
    test_sampler = None
    if test_shards:
        test_sampler = TokenSampler(
            shard_paths=test_shards,
            seq_len=args.seq_len,
            cache_max=max(4, args.cache_max // 2),
            map_location="cpu",
            assume_one_window_per_shard=args.assume_one_window_per_shard,
        )

    # Infer d_in from one shard
    sample_obj = torch.load(train_shards[0], map_location="cpu")
    emb = sample_obj["emb"]
    if emb.ndim != 3 or emb.shape[1] != args.seq_len:
        raise SystemExit(f"Unexpected emb shape in {train_shards[0]}: {tuple(emb.shape)}")
    d_in = emb.shape[2]

    run_name = f"layer{args.layer}_din{d_in}_dh{args.d_hidden}_L{args.seq_len}_l1{args.l1_coeff}_bs{args.batch_size}_seed{args.seed}"
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # CSV logger
    csv_path = os.path.join(out_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "step",
        "train_loss",
        "train_recon_mse",
        "train_l1_mean",
        "train_active_frac",
        "val_loss",
        "val_recon_mse",
        "val_l1_mean",
        "val_active_frac",
        "lr",
        "sec_per_step",
    ])
    writer.writeheader()
    csv_file.flush()

    # Optional TensorBoard
    tb = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=out_dir)
        except Exception:
            tb = None

    # Model + optimizer
    model = SparseAutoencoder(d_in=d_in, d_hidden=args.d_hidden, use_relu=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    model.train()
    t0 = time.time()
    last_t = t0

    for step in range(1, args.steps + 1):
        x = train_sampler.sample_batch_tokens(
            batch_size=args.batch_size,
            device=device,
            dtype=torch.float32,
            tokens_per_window=args.tokens_per_window,
        )

        x_hat, a = model(x)
        recon = F.mse_loss(x_hat, x)
        l1 = a.abs().mean()
        loss = recon + args.l1_coeff * l1

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        # Normalize decoder columns (common SAE stabilization)
        model.normalize_decoder_columns_()

        # Training stats
        with torch.no_grad():
            if model.use_relu:
                active_frac = float((a > 0).float().mean())
            else:
                active_frac = float((a.abs() > 1e-6).float().mean())

        # Logging
        if step % args.log_every == 0 or step == 1:
            print(f"Logging step {step}/{args.steps}")
            now = time.time()
            sec_per_step = (now - last_t) / max(1, args.log_every if step % args.log_every == 0 else 1)
            last_t = now

            train_metrics = {
                "train_loss": float(loss),
                "train_recon_mse": float(recon),
                "train_l1_mean": float(l1),
                "train_active_frac": active_frac,
                "lr": float(optim.param_groups[0]["lr"]),
                "sec_per_step": sec_per_step,
            }

            if tb is not None:
                tb.add_scalar("train/loss", train_metrics["train_loss"], step)
                tb.add_scalar("train/recon_mse", train_metrics["train_recon_mse"], step)
                tb.add_scalar("train/l1_mean", train_metrics["train_l1_mean"], step)
                tb.add_scalar("train/active_frac", train_metrics["train_active_frac"], step)
                tb.add_scalar("train/lr", train_metrics["lr"], step)

        # Eval
        if step % args.eval_every == 0 or step == args.steps:

            val_metrics = run_eval(
                model=model,
                sampler=val_sampler,
                device=device,
                steps=args.eval_steps,
                batch_size=args.batch_size,
                l1_coeff=args.l1_coeff,
                tokens_per_window=args.tokens_per_window,
            )

            if tb is not None:
                tb.add_scalar("val/loss", val_metrics["loss"], step)
                tb.add_scalar("val/recon_mse", val_metrics["recon_mse"], step)
                tb.add_scalar("val/l1_mean", val_metrics["l1_mean"], step)
                tb.add_scalar("val/active_frac", val_metrics["active_frac"], step)

            row = {
                "step": step,
                "train_loss": float(loss),
                "train_recon_mse": float(recon),
                "train_l1_mean": float(l1),
                "train_active_frac": active_frac,
                "val_loss": val_metrics["loss"],
                "val_recon_mse": val_metrics["recon_mse"],
                "val_l1_mean": val_metrics["l1_mean"],
                "val_active_frac": val_metrics["active_frac"],
                "lr": float(optim.param_groups[0]["lr"]),
                "sec_per_step": (time.time() - t0) / step,
            }
            writer.writerow(row)
            csv_file.flush()

        # Checkpoint
        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = os.path.join(out_dir, f"ckpt_step_{step:07d}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "d_in": d_in,
                    "d_hidden": args.d_hidden,
                    "layer": args.layer,
                    "seq_len": args.seq_len,
                    "l1_coeff": args.l1_coeff,
                    "model_class": "SparseAutoencoder",
                },
                ckpt_path,
            )

    # Final test eval (optional)
    if test_sampler is not None:
        test_metrics = run_eval(
            model=model,
            sampler=test_sampler,
            device=device,
            steps=args.eval_steps,
            batch_size=args.batch_size,
            l1_coeff=args.l1_coeff,
            tokens_per_window=args.tokens_per_window,
        )
        with open(os.path.join(out_dir, "test_metrics.txt"), "w", encoding="utf-8") as f:
            for k, v in test_metrics.items():
                f.write(f"{k}\t{v}\n")
        if tb is not None:
            tb.add_scalar("test/loss", test_metrics["loss"], args.steps)
            tb.add_scalar("test/recon_mse", test_metrics["recon_mse"], args.steps)
            tb.add_scalar("test/l1_mean", test_metrics["l1_mean"], args.steps)
            tb.add_scalar("test/active_frac", test_metrics["active_frac"], args.steps)

    if tb is not None:
        tb.flush()
        tb.close()

    csv_file.close()
    print(f"Done. Logs: {csv_path}")
    print(f"Run dir: {out_dir}")


if __name__ == "__main__":
    main()