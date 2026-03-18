"""
find_top_activations.py  (optimized)

For each SAE feature, find the top-N tokens (by activation value) across
train / val / test shards, and save them with l-token genomic context windows.

Key optimizations over the original:
  1. Vectorized per-chunk topk — eliminates the O(chunk * n_features) Python loop.
  2. TensorTopN accumulator — pure-tensor top-N bookkeeping; no Python heaps/dicts.
  3. Parallel shard loading via DataLoader (num_workers + pin_memory).
  4. Whole-shard SAE batching — sub-batches accumulate on CPU, single topk call per shard.
  5. Non-zero mask — only iterate the small fraction of nonzero (feat, rank) entries
     when harvesting metadata for the final save.

Usage:
    python find_top_activations.py \
        --sae_checkpoint  runs/my_run/checkpoint_step10000.pt \
        --sae_cfg         runs/my_run/cfg.json \
        --embed_dir       /data/embeddings \
        --layer           2 \
        --splits          train val test \
        --top_n           10 \
        --context_len     5 \
        --out_dir         results/top_activations \
        --device          cuda \
        --batch_size      4096 \
        --num_workers     4

Output layout:
    out_dir/
        top_activations.pt   – dict (see below)
        top_activations.csv  – human-readable flat table

top_activations.pt keys:
    act_values   FloatTensor [n_features, top_n]
    token_pos    LongTensor  [n_features, top_n]   position within sequence
    coords       list[list[tuple]]                 [n_features][top_n] → (chrom,start,end)
    context_seqs list[list[list]]                  [n_features][top_n] → context coord window
    split        list[list[str]]                   [n_features][top_n] → "train"/"val"/"test"
    shard_path   list[list[str]]                   [n_features][top_n] → source shard file
    cfg          dict
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from BatchTopK.sae import BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE, JumpReLUInferenceSAE


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def restore_cfg_types(cfg: dict) -> dict:
    if isinstance(cfg.get("dtype"), str):
        cfg["dtype"] = getattr(torch, cfg["dtype"].replace("torch.", ""))
    if isinstance(cfg.get("device"), str):
        cfg["device"] = torch.device(cfg["device"])
    return cfg


def load_sae(cfg: dict, checkpoint_path: str, device: str):
    state = torch.load(checkpoint_path, map_location=device)

    sae_state = state.get("sae_state_dict") or state.get("model_state_dict") or state
    saved_cfg  = state.get("cfg", cfg)
    theta      = state.get("theta") or saved_cfg.get("theta")

    arch = cfg.get("sae_type", "batchtopk").lower()
    cls_map = {
        "batchtopk": BatchTopKSAE,
        "top_k":     TopKSAE,
        "vanilla":   VanillaSAE,
        "jumprelu":  JumpReLUSAE,
    }
    sae = cls_map[arch](cfg)
    sae.load_state_dict(sae_state, strict=False)

    if arch == "batchtopk":
        if theta is None:
            raise ValueError(
                "BatchTopKSAE checkpoint has no 'theta'. "
                "Ensure save_checkpoint stores it (cfg['theta'] = theta)."
            )
        print(f"Wrapping BatchTopKSAE with JumpReLUInferenceSAE (theta={theta:.6f})")
        sae = JumpReLUInferenceSAE(sae, theta=theta)

    sae.eval().to(device)
    return sae


@torch.no_grad()
def get_activations(sae, x: torch.Tensor) -> torch.Tensor:
    """Returns (N, n_features) float32 CPU tensor."""
    x = x.to(next(sae.parameters()).device).to(next(sae.parameters()).dtype)
    if isinstance(sae, JumpReLUInferenceSAE):
        _, acts = sae(x)
        return acts.float().cpu()
    raise NotImplementedError("Add activation extraction for your SAE type here.")


# ---------------------------------------------------------------------------
# Context window helper
# ---------------------------------------------------------------------------

def build_context_window(
    per_tok_coords: list,
    token_pos: int,
    context_len: int,
) -> list:
    n = len(per_tok_coords)
    return [
        per_tok_coords[token_pos + offset]
        if 0 <= token_pos + offset < n else None
        for offset in range(-context_len, context_len + 1)
    ]


# ---------------------------------------------------------------------------
# Tensor-based top-N accumulator  (optimization #2)
# ---------------------------------------------------------------------------

class TensorTopN:
    """
    Maintains the top-N (value, global_token_index, shard_index) per feature
    entirely in tensors — no Python heaps or dicts.

    global_token_index encodes (shard_id * MAX_SEQ_LEN + flat_position) so we
    can recover (seq_idx, tok_pos) later without storing separate arrays.
    """

    def __init__(self, n_features: int, top_n: int, device: str = "cpu"):
        self.top_n      = top_n
        self.n_features = n_features
        self.device     = device
        # (n_features, top_n)
        self.vals       = torch.full((n_features, top_n), -float("inf"), device=device)
        self.tok_idxs   = torch.zeros(n_features, top_n, dtype=torch.long, device=device)
        self.shard_ids  = torch.zeros(n_features, top_n, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(
        self,
        new_vals: torch.Tensor,    # (k, n_features)  — already on self.device
        new_tok:  torch.Tensor,    # (k, n_features)  flat token index within shard
        shard_id: int,
    ):
        k = new_vals.shape[0]
        shard_col = torch.full((k, self.n_features), shard_id,
                               dtype=torch.long, device=self.device)

        # Concatenate existing best with new candidates along the rank axis
        cat_vals  = torch.cat([self.vals.T,      new_vals],  dim=0)  # (top_n+k, F)
        cat_tok   = torch.cat([self.tok_idxs.T,  new_tok],   dim=0)
        cat_shard = torch.cat([self.shard_ids.T, shard_col], dim=0)

        # Re-select top-N across the merged candidates
        k2 = min(self.top_n, cat_vals.shape[0])
        top_vals, top_pos = torch.topk(cat_vals, k2, dim=0)  # (k2, F)

        self.vals      = top_vals.T                                     # (F, k2)
        self.tok_idxs  = cat_tok.gather(0, top_pos).T
        self.shard_ids = cat_shard.gather(0, top_pos).T


# ---------------------------------------------------------------------------
# Shard dataset for parallel loading  (optimization #3)
# ---------------------------------------------------------------------------

class ShardDataset(Dataset):
    def __init__(self, shard_paths: List[str]):
        self.paths = shard_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        shard = torch.load(self.paths[i], map_location="cpu")
        return shard["emb"], shard["coords"], self.paths[i]


def _collate_shards(batch):
    """DataLoader collate: keep each shard as-is (batch size = 1 shard)."""
    assert len(batch) == 1, "ShardDataset must be loaded with batch_size=1"
    emb, coords, path = batch[0]
    return emb, coords, path


# ---------------------------------------------------------------------------
# Core processing loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_shards(
    sae,
    shard_paths: List[str],
    split_name: str,
    context_len: int,
    accumulator: TensorTopN,
    batch_size: int,
    act_size: int,
    device: str,
    num_workers: int,
    shard_registry: List[dict],   # appended in place; index = shard_id
):
    """
    Stream through all shards for one split, updating `accumulator` in place.
    `shard_registry` maps shard_id → {path, split, B, L, coords_list}.
    """
    dataset = ShardDataset(shard_paths)
    loader  = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=(device != "cpu"),
        collate_fn=_collate_shards,
        persistent_workers=(num_workers > 0),
    )

    for emb, coords_list, shard_path in tqdm(loader, desc=f"  {split_name}", leave=False):
        # emb: (B, L, D)   coords_list: list[B] of coord info
        B, L, D = emb.shape
        assert D == act_size, f"Embedding dim {D} != act_size {act_size}"

        shard_id = len(shard_registry)
        shard_registry.append({
            "path":        shard_path,
            "split":       split_name,
            "B":           B,
            "L":           L,
            "coords_list": coords_list,
        })

        emb_flat = emb.reshape(B * L, D)  # (B*L, D)

        # --- SAE forward pass in sub-batches with rolling topk merge ---
        # Peak VRAM: O(batch_size * n_features) instead of O(B*L * n_features).
        # After each sub-batch we immediately call topk and push into the
        # accumulator, so the full-shard activation matrix is never materialised.
        for start in range(0, B * L, batch_size):
            end      = min(start + batch_size, B * L)
            chunk    = emb_flat[start:end].to(device)
            acts     = get_activations(sae, chunk)          # (chunk_size, n_features) CPU

            k        = min(accumulator.top_n, acts.shape[0])
            topk_vals, topk_local_idx = torch.topk(acts, k, dim=0)  # (k, n_features)

            # Offset local sub-batch indices to shard-level flat indices
            topk_flat_idx = topk_local_idx + start          # (k, n_features)

            topk_vals     = topk_vals.to(accumulator.device)
            topk_flat_idx = topk_flat_idx.to(accumulator.device)
            accumulator.update(topk_vals, topk_flat_idx, shard_id)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    accumulator: TensorTopN,
    shard_registry: List[dict],
    top_n: int,
    context_len: int,
    layer: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    n_features = accumulator.n_features

    # Move tensors to CPU for indexing
    vals_np  = accumulator.vals.cpu()       # (n_features, top_n)
    tok_np   = accumulator.tok_idxs.cpu()   # (n_features, top_n)
    shard_np = accumulator.shard_ids.cpu()  # (n_features, top_n)

    act_values  = torch.zeros(n_features, top_n)
    token_pos_t = torch.zeros(n_features, top_n, dtype=torch.long)
    coords_out  = []
    context_out = []
    split_out   = []
    shard_out   = []

    for fi in range(n_features):
        coords_row  = []
        context_row = []
        split_row   = []
        shard_row   = []

        for rank in range(top_n):
            val      = vals_np[fi, rank].item()
            flat_idx = tok_np[fi, rank].item()
            sh_id    = shard_np[fi, rank].item()

            if val == -float("inf") or sh_id >= len(shard_registry):
                # Unfilled slot
                coords_row.append(None)
                context_row.append([None] * (2 * context_len + 1))
                split_row.append("")
                shard_row.append("")
                continue

            reg  = shard_registry[sh_id]
            L    = reg["L"]
            seq_idx = flat_idx // L
            tok_pos = flat_idx  % L

            act_values[fi, rank]  = val
            token_pos_t[fi, rank] = tok_pos

            coords_list = reg["coords_list"]
            seq_coord   = coords_list[seq_idx]

            # Resolve per-token vs per-sequence coords
            if isinstance(seq_coord[0], (list, tuple)) and len(seq_coord) == L:
                per_tok_coords = seq_coord
            else:
                per_tok_coords = [seq_coord] * L

            coord   = per_tok_coords[tok_pos]
            context = build_context_window(per_tok_coords, tok_pos, context_len)

            coords_row.append(coord)
            context_row.append(context)
            split_row.append(reg["split"])
            shard_row.append(reg["path"])

        coords_out.append(coords_row)
        context_out.append(context_row)
        split_out.append(split_row)
        shard_out.append(shard_row)

    save_dict = {
        "act_values":   act_values,
        "token_pos":    token_pos_t,
        "coords":       coords_out,
        "context_seqs": context_out,
        "split":        split_out,
        "shard_path":   shard_out,
        "cfg": {
            "top_n":       top_n,
            "context_len": context_len,
            "layer":       layer,
            "n_features":  n_features,
        },
    }

    pt_path = os.path.join(out_dir, "top_activations.pt")
    torch.save(save_dict, pt_path)
    print(f"Saved {pt_path}")

    # CSV — only iterate the (small) nonzero set  (optimization #5)
    csv_path = os.path.join(out_dir, "top_activations.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature_idx", "rank", "activation_value",
            "split", "shard_path", "seq_idx", "tok_pos",
            "coord_chrom", "coord_start", "coord_end",
            "context_window",
        ])
        # Only visit filled slots
        filled = (act_values > 0).nonzero(as_tuple=False)  # (M, 2)
        for fi, rank in filled.tolist():
            val   = act_values[fi, rank].item()
            tp    = token_pos_t[fi, rank].item()
            coord = coords_out[fi][rank]
            chrom, cs, ce = (coord if coord and len(coord) == 3 else ("", "", ""))
            ctx_str = ";".join(
                f"{c[0]}:{c[1]}-{c[2]}" if (c and len(c) == 3) else "None"
                for c in context_out[fi][rank]
            )
            sh_id   = shard_np[fi, rank].item()
            reg     = shard_registry[sh_id] if sh_id < len(shard_registry) else {}
            flat_idx = tok_np[fi, rank].item()
            L       = reg.get("L", 1)
            seq_idx = flat_idx // L

            writer.writerow([
                fi, rank, f"{val:.6f}",
                split_out[fi][rank], shard_out[fi][rank],
                seq_idx, tp,
                chrom, cs, ce,
                ctx_str,
            ])
    print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Find top-N activating tokens per SAE feature (optimized).")
    p.add_argument("--sae_checkpoint", required=True, help="Path to SAE .pt checkpoint")
    p.add_argument("--sae_cfg",        required=True, help="Path to SAE cfg JSON file")
    p.add_argument("--embed_dir",      required=True, help="Root dir containing train/val/test splits")
    p.add_argument("--layer",          type=int, required=True, help="Which layer's embeddings to use")
    p.add_argument("--splits",         nargs="+", default=["train", "val", "test"])
    p.add_argument("--top_n",          type=int, default=10,   help="Top-N tokens per feature")
    p.add_argument("--context_len",    type=int, default=5,    help="Tokens of context on each side")
    p.add_argument("--out_dir",        required=True,          help="Where to write results")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size",     type=int, default=4096, help="Sub-batch size for SAE forward pass")
    p.add_argument("--num_workers",    type=int, default=4,    help="DataLoader worker processes")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.sae_cfg) as f:
        cfg = json.load(f)
    cfg = restore_cfg_types(cfg)
    cfg["device"] = args.device

    print(f"Loading SAE from {args.sae_checkpoint} ...")
    sae        = load_sae(cfg, args.sae_checkpoint, args.device)
    n_features = cfg["dict_size"]
    act_size   = cfg["act_size"]
    print(f"SAE: dict_size={n_features}, act_size={act_size}")
    print(f"Scanning splits: {args.splits}  |  top_n={args.top_n}  |  context_len={args.context_len}")

    # Single accumulator across all splits  (optimization #2)
    accumulator    = TensorTopN(n_features, args.top_n, device=args.device)
    shard_registry: List[dict] = []

    for split in args.splits:
        layer_dir   = os.path.join(args.embed_dir, split, f"layer_{args.layer}")
        shard_paths = sorted(glob.glob(os.path.join(layer_dir, "shard_*.pt")))
        if not shard_paths:
            print(f"  [WARNING] No shards found in {layer_dir}, skipping.")
            continue
        print(f"\nProcessing {split}: {len(shard_paths)} shards in {layer_dir}")
        process_shards(
            sae=sae,
            shard_paths=shard_paths,
            split_name=split,
            context_len=args.context_len,
            accumulator=accumulator,
            batch_size=args.batch_size,
            act_size=act_size,
            device=args.device,
            num_workers=args.num_workers,
            shard_registry=shard_registry,
        )

    print("\nSaving results ...")
    save_results(
        accumulator    = accumulator,
        shard_registry = shard_registry,
        top_n          = args.top_n,
        context_len    = args.context_len,
        layer          = args.layer,
        out_dir        = args.out_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()