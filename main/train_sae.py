# train_sae.py
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dna_stream import DNABERT2Streamer, GenomeWindowStream


class SAE(nn.Module):
    """
    Vanilla sparse autoencoder:
      z = ReLU((x - b_dec) @ W_enc^T + b_enc)
      x_hat = z @ W_dec^T + b_dec

    Untied encoder/decoder weights.
    """

    def __init__(self, d_in: int, d_dict: int):
        super().__init__()
        self.d_in = d_in
        self.d_dict = d_dict

        self.W_enc = nn.Parameter(torch.empty(d_dict, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_dict))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc.t() + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def layer_out_path(base_out: str, layer_id: int) -> str:
    root, ext = os.path.splitext(base_out)
    if ext == "":
        ext = ".pt"
    return f"{root}.layer{layer_id}{ext}"


def maybe_makedirs(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@dataclass
class ShuffledTokenPool:
    tokens: torch.Tensor  # [N, d_in] CPU float32
    cursor: int = 0

    def remaining(self) -> int:
        return int(self.tokens.size(0) - self.cursor)

    def next_batch(self, batch_tokens: int) -> torch.Tensor:
        if self.cursor >= self.tokens.size(0):
            raise StopIteration
        end = min(self.tokens.size(0), self.cursor + batch_tokens)
        out = self.tokens[self.cursor:end]
        self.cursor = end
        return out


def _subsample_tokens_per_window(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return x
    T = int(x.size(0))
    if T <= k:
        return x
    idx = torch.randperm(T)[:k]
    return x[idx]


def build_shuffled_pools_from_stream(
    gen: Iterator[Tuple[Dict[int, torch.Tensor], Tuple[str, int, int]]],
    layers: Sequence[int],
    pool_windows: int,
    tokens_per_window: int,
) -> Dict[int, ShuffledTokenPool]:
    per_layer_chunks: Dict[int, List[torch.Tensor]] = {lid: [] for lid in layers}

    for _ in range(pool_windows):
        layer_to_tokens, _meta = next(gen)
        for lid in layers:
            toks = layer_to_tokens[lid]  # [T, D] CPU
            toks = _subsample_tokens_per_window(toks, tokens_per_window)
            per_layer_chunks[lid].append(toks)

    pools: Dict[int, ShuffledTokenPool] = {}
    for lid in layers:
        all_tokens = (
            torch.cat(per_layer_chunks[lid], dim=0)
            if len(per_layer_chunks[lid]) > 1
            else per_layer_chunks[lid][0]
        )
        perm = torch.randperm(all_tokens.size(0))
        pools[lid] = ShuffledTokenPool(tokens=all_tokens[perm], cursor=0)
    return pools


@torch.no_grad()
def evaluate_sae_on_pools(
    saes: Dict[int, SAE],
    pools: Dict[int, ShuffledTokenPool],
    layers: Sequence[int],
    device: torch.device,
    batch_tokens: int,
    l1_lambda: float,
    amp_dtype: Optional[torch.dtype],
    val_batches: int,
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if (device.type == "cuda" and amp_dtype is not None)
        else None
    )

    for lid in layers:
        sae = saes[lid]
        sae.eval()

        losses: List[float] = []
        recons: List[float] = []
        l1s: List[float] = []

        local_cursor = pools[lid].cursor
        for _ in range(val_batches):
            if pools[lid].remaining() <= 0:
                break
            x_cpu = pools[lid].next_batch(batch_tokens)
            x = x_cpu.to(device, non_blocking=True)

            if autocast_ctx is not None:
                with autocast_ctx:
                    x_hat, z = sae(x)
                    recon = F.mse_loss(x_hat, x)
                    l1 = z.abs().mean()
                    loss = recon + l1_lambda * l1
            else:
                x_hat, z = sae(x)
                recon = F.mse_loss(x_hat, x)
                l1 = z.abs().mean()
                loss = recon + l1_lambda * l1

            losses.append(float(loss.detach().cpu()))
            recons.append(float(recon.detach().cpu()))
            l1s.append(float(l1.detach().cpu()))

        pools[lid].cursor = local_cursor
        sae.train()

        if losses:
            out[lid] = {
                "val_loss": sum(losses) / len(losses),
                "val_recon": sum(recons) / len(recons),
                "val_l1": sum(l1s) / len(l1s),
            }
        else:
            out[lid] = {"val_loss": float("nan"), "val_recon": float("nan"), "val_l1": float("nan")}

    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--fasta", required=True)
    ap.add_argument("--train_bed", required=True)
    ap.add_argument("--val_bed", default=None)

    ap.add_argument("--hf_model", default="zhihan1996/DNABERT-2-117M")
    ap.add_argument("--layers", default="2,6,10", help="Comma-separated layer ids.")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="fp16", choices=["off", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--max_length", type=int, default=2050)

    ap.add_argument("--dict_mult", type=int, default=8)
    ap.add_argument("--l1_lambda", type=float, default=1e-3)

    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch_tokens", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Pool/shuffle (probabilistic decorrelation)
    ap.add_argument("--pool_windows", type=int, default=512)
    ap.add_argument("--tokens_per_window", type=int, default=16)

    # Optional logging/validation/plotting
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=1000, help="0 disables validation.")
    ap.add_argument("--val_batches", type=int, default=10)

    ap.add_argument("--metrics_csv", default=None)
    ap.add_argument("--plot_png", default=None)

    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    if not layers:
        raise ValueError("No layers specified")

    if args.amp == "off":
        amp_dtype = None
    elif args.amp == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16

    print("log: reading arguments complete")
    train_genome = GenomeWindowStream(args.fasta, args.train_bed, seed=args.seed)
    val_genome = GenomeWindowStream(args.fasta, args.val_bed, seed=args.seed + 12345) if args.val_bed else None

    print("log: loaded training and validation sequences")

    streamer = DNABERT2Streamer(
        hf_model=args.hf_model,
        device=device,
        amp=args.amp,
        trust_remote_code=args.trust_remote_code,
        move_to_cpu=True,
        drop_special_tokens=True,
        max_length=args.max_length,
    )
    print("log: constructed streamer")

    train_gen = streamer.stream_token_embeddings(train_genome, layers)
    layer_to_tokens, _ = next(train_gen)
    d_in_by_layer: Dict[int, int] = {lid: int(layer_to_tokens[lid].size(1)) for lid in layers}

    saes: Dict[int, SAE] = {}
    opts: Dict[int, torch.optim.Optimizer] = {}
    for lid in layers:
        d_in = d_in_by_layer[lid]
        d_dict = args.dict_mult * d_in
        sae = SAE(d_in=d_in, d_dict=d_dict).to(device)
        saes[lid] = sae
        opts[lid] = torch.optim.AdamW(sae.parameters(), lr=args.lr)

    val_gen = streamer.stream_token_embeddings(val_genome, layers) if val_genome is not None else None

    train_pools: Optional[Dict[int, ShuffledTokenPool]] = None
    val_pools: Optional[Dict[int, ShuffledTokenPool]] = None

    rows: List[Dict[str, float]] = []
    write_metrics = args.metrics_csv is not None
    make_plots = args.plot_png is not None

    print("log: constructed streamers and files")

    def rebuild_train_pools() -> Dict[int, ShuffledTokenPool]:
        return build_shuffled_pools_from_stream(train_gen, layers, args.pool_windows, args.tokens_per_window)

    def rebuild_val_pools() -> Dict[int, ShuffledTokenPool]:
        if val_gen is None:
            raise RuntimeError("Validation not configured")
        return build_shuffled_pools_from_stream(val_gen, layers, args.pool_windows, args.tokens_per_window)

    print("log: before starting training")
    for step in range(1, args.steps + 1):
        if train_pools is None or any(train_pools[lid].remaining() < args.batch_tokens for lid in layers):
            train_pools = rebuild_train_pools()

        for lid in layers:
            sae = saes[lid]
            opt = opts[lid]

            x_cpu = train_pools[lid].next_batch(args.batch_tokens)
            x = x_cpu.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if device.type == "cuda" and amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    x_hat, z = sae(x)
                    recon = F.mse_loss(x_hat, x)
                    l1 = z.abs().mean()
                    loss = recon + args.l1_lambda * l1
                loss.backward()
            else:
                x_hat, z = sae(x)
                recon = F.mse_loss(x_hat, x)
                l1 = z.abs().mean()
                loss = recon + args.l1_lambda * l1
                loss.backward()

            opt.step()

            if (step % args.log_every) == 0:
                l0 = (z > 0).float().sum(dim=1).mean()
                rows.append(
                    {
                        "step": float(step),
                        "layer": float(lid),
                        "train_loss": float(loss.detach().cpu()),
                        "train_recon": float(recon.detach().cpu()),
                        "train_l1": float(l1.detach().cpu()),
                        "train_l0": float(l0.detach().cpu()),
                        "train_batch_tokens": float(x.size(0)),
                    }
                )
        print(f"log: step {step}/{args.steps} validating")
        do_val = (args.val_every > 0) and (val_gen is not None) and ((step % args.val_every) == 0)
        if do_val:
            if val_pools is None or any(val_pools[lid].remaining() < args.batch_tokens for lid in layers):
                val_pools = rebuild_val_pools()

            val_metrics = evaluate_sae_on_pools(
                saes=saes,
                pools=val_pools,
                layers=layers,
                device=device,
                batch_tokens=args.batch_tokens,
                l1_lambda=args.l1_lambda,
                amp_dtype=amp_dtype,
                val_batches=args.val_batches,
            )
            for lid in layers:
                v = val_metrics[lid]
                # attach to last train row for this step/layer if present; else append
                found = False
                for r in reversed(rows):
                    if int(r.get("step", -1)) == step and int(r.get("layer", -1)) == lid:
                        r.update(v)
                        found = True
                        break
                if not found:
                    rows.append({"step": float(step), "layer": float(lid), **v})

        if (step % max(1, args.val_every)) == 0 or step == args.steps:
            for lid in layers:
                out_path = layer_out_path(args.out, lid)
                maybe_makedirs(out_path)
                torch.save(
                    {
                        "step": step,
                        "layer": lid,
                        "d_in": saes[lid].d_in,
                        "d_dict": saes[lid].d_dict,
                        "state_dict": saes[lid].state_dict(),
                        "optimizer": opts[lid].state_dict(),
                        "args": vars(args),
                    },
                    out_path,
                )

    if write_metrics and rows:
        maybe_makedirs(args.metrics_csv)
        keys = sorted({k for r in rows for k in r.keys()})
        with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    if make_plots and rows:
        import matplotlib.pyplot as plt

        for lid in layers:
            tr_x = [r["step"] for r in rows if int(r.get("layer", -1)) == lid and "train_loss" in r]
            tr_y = [r["train_loss"] for r in rows if int(r.get("layer", -1)) == lid and "train_loss" in r]
            va_x = [r["step"] for r in rows if int(r.get("layer", -1)) == lid and "val_loss" in r]
            va_y = [r["val_loss"] for r in rows if int(r.get("layer", -1)) == lid and "val_loss" in r]

            plt.figure()
            if tr_x:
                plt.plot(tr_x, tr_y, label="train_loss")
            if va_x:
                plt.plot(va_x, va_y, label="val_loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title(f"SAE loss (layer {lid})")
            plt.legend()

            root, ext = os.path.splitext(args.plot_png)
            if ext.lower() != ".png":
                ext = ".png"
            out_png = f"{root}.layer{lid}{ext}"
            maybe_makedirs(out_png)
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()