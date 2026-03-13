"""
find_top_activations.py

For each SAE feature, find the top-N tokens (by activation value) across
train / val / test shards, and save them with l-token genomic context windows.

Usage:
    python find_top_activations.py \
        --sae_checkpoint  runs/my_run/checkpoint_step10000.pt \
        --sae_cfg         runs/my_run/cfg.json \
        --save_dir        /data/embeddings \
        --layer           2 \
        --splits          train val test \
        --top_n           10 \
        --context_len     5 \
        --out_dir         results/top_activations \
        --device          cuda

Output layout:
    out_dir/
        top_activations.pt   – dict with keys below
        top_activations.csv  – human-readable flat table

top_activations.pt structure:
    {
      "feature_idx":   LongTensor  [n_features, top_n],
      "act_values":    FloatTensor [n_features, top_n],
      "token_pos":     LongTensor  [n_features, top_n],   # position within sequence
      "coords":        list[list[tuple]]                  # [n_features][top_n] → (chrom,start,end)
      "context_seqs":  list[list[list]]                   # [n_features][top_n] → context coord window
      "split":         list[list[str]]                    # [n_features][top_n] → "train"/"val"/"test"
      "shard_path":    list[list[str]]                    # [n_features][top_n] → source shard file
      "cfg": {
          "top_n": ...,
          "context_len": ...,
          "layer": ...,
      }
    }
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
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import your SAE classes
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(__file__))
from BatchTopK.sae import BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE, JumpReLUInferenceSAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def restore_cfg_types(cfg):
    if isinstance(cfg.get("dtype"), str):
        cfg["dtype"] = getattr(torch, cfg["dtype"].replace("torch.", ""))
    if isinstance(cfg.get("device"), str):
        cfg["device"] = torch.device(cfg["device"])
    return cfg

def load_sae(cfg: dict, checkpoint_path: str, device: str):
    state = torch.load(checkpoint_path, map_location=device)

    # Unwrap nested checkpoint formats
    sae_state = state.get("sae_state_dict") or state.get("model_state_dict") or state
    saved_cfg  = state.get("cfg", cfg)       # checkpoint may carry its own cfg
    theta      = state.get("theta") or saved_cfg.get("theta")

    arch = cfg.get("sae_type", "batchtopk").lower()
    cls_map = {
        "batchtopk": BatchTopKSAE,
        "top_k":       TopKSAE,
        "vanilla":     VanillaSAE,
        "jumprelu":    JumpReLUSAE,
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

    sae.eval().to(device)
    return sae

@torch.no_grad()
def get_activations(sae, x: torch.Tensor) -> torch.Tensor:
    x = x.to(next(sae.parameters()).device).to(next(sae.parameters()).dtype)

    if isinstance(sae, JumpReLUInferenceSAE):
        _, acts = sae(x)        # returns (reconstruction, acts)
        return acts.float().cpu()

    # Fallback for TopKSAE / VanillaSAE / JumpReLUSAE
    # x, x_mean, x_std = sae.preprocess_input(x)
    # x_cent = x - sae.b_dec
    # acts = F.relu(x_cent @ sae.W_enc)
    # if hasattr(sae, 'jumprelu'):
    #     acts = sae.jumprelu(acts)
    # elif 'top_k' in sae.cfg:
    #     k = sae.cfg['top_k']
    #     topk = torch.topk(acts, min(k, acts.shape[-1]), dim=-1)
    #     acts = torch.zeros_like(acts).scatter(-1, topk.indices, topk.values)
    # return acts.float().cpu()


def build_context_window(
    coords: List[Tuple],
    token_pos: int,
    context_len: int
) -> List[Tuple]:
    """
    Return up to (2*context_len + 1) coord tuples centred on token_pos,
    padding with None where the window falls outside the sequence.
    """
    n = len(coords)
    window = []
    for offset in range(-context_len, context_len + 1):
        idx = token_pos + offset
        window.append(coords[idx] if 0 <= idx < n else None)
    return window


# ---------------------------------------------------------------------------
# Per-feature heap: keep top-N (value, metadata) without storing everything
# ---------------------------------------------------------------------------

import heapq

class TopNHeap:
    """Min-heap of size N: cheaply track the N largest values seen so far."""
    def __init__(self, n: int):
        self.n = n
        self.heap: list = []   # (value, counter, metadata_dict)
        self._counter = 0      # tie-break to avoid comparing dicts

    def push(self, value: float, metadata: dict):
        self._counter += 1
        entry = (value, self._counter, metadata)
        if len(self.heap) < self.n:
            heapq.heappush(self.heap, entry)
        elif value > self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)

    def sorted_results(self):
        """Return list of (value, metadata) sorted descending."""
        return [(v, m) for v, _, m in sorted(self.heap, key=lambda x: -x[0])]


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_shards(
    sae,
    shard_paths: List[str],
    split_name: str,
    context_len: int,
    top_heaps: List[TopNHeap],
    batch_size: int,
    act_size: int,
    device: str,
):
    """
    Stream through all shards for one split, updating top_heaps in place.
    """
    n_features = len(top_heaps)

    for shard_path in tqdm(shard_paths, desc=f"  {split_name}", leave=False):
        shard = torch.load(shard_path, map_location="cpu")
        emb: torch.Tensor = shard["emb"]          # (B, L, D)
        coords_list = shard["coords"]              # List of (chrom, start, end), length B
        seq_len = shard.get("seq_len", emb.shape[1])

        B, L, D = emb.shape
        assert D == act_size, f"Embedding dim {D} != act_size {act_size}"
        assert len(coords_list) == B, "coords length must match batch size"

        # Flatten to (B*L, D) for batched SAE inference
        emb_flat = emb.reshape(B * L, D)

        # Process in sub-batches to control memory
        for start in range(0, B * L, batch_size):
            print(f"Processed {start}/{B*L} tokens in shard {shard_path}")
            end = min(start + batch_size, B * L)
            chunk = emb_flat[start:end]                        # (chunk, D)
            acts  = get_activations(sae, chunk)                # (chunk, n_features)

            # Iterate over tokens in this chunk
            for local_idx in range(acts.shape[0]):
                global_tok = start + local_idx
                seq_idx  = global_tok // L   # which sequence in the shard
                tok_pos  = global_tok  % L   # position within that sequence

                # coords for the full sequence (list of L tuples if available,
                # or we synthesise approximate coords from the sequence coord)
                seq_coord = coords_list[seq_idx]   # (chrom, start, end) for whole seq

                token_act = acts[local_idx]        # (n_features,)

                for feat_idx in range(n_features):
                    val = token_act[feat_idx].item()
                    if val == 0.0:
                        continue   # skip zero activations for efficiency

                    # Build context window using the sequence-level coord.
                    # If coords_list stores per-token coords (list of lists),
                    # adjust indexing below.
                    if isinstance(seq_coord[0], (list, tuple)) and len(seq_coord) == L:
                        # per-token coords stored
                        per_tok_coords = seq_coord
                    else:
                        # only sequence-level coord; store as single entry
                        per_tok_coords = [seq_coord] * L

                    context = build_context_window(per_tok_coords, tok_pos, context_len)

                    metadata = {
                        "split":      split_name,
                        "shard_path": shard_path,
                        "seq_idx":    seq_idx,
                        "tok_pos":    tok_pos,
                        "coord":      per_tok_coords[tok_pos],
                        "context":    context,
                    }
                    top_heaps[feat_idx].push(val, metadata)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(top_heaps: List[TopNHeap], top_n: int, context_len: int, layer: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    n_features = len(top_heaps)

    # Build tensors / lists
    act_values   = torch.zeros(n_features, top_n)
    token_pos    = torch.zeros(n_features, top_n, dtype=torch.long)
    coords_out   = []
    context_out  = []
    split_out    = []
    shard_out    = []

    for fi, heap in enumerate(top_heaps):
        results = heap.sorted_results()   # descending by activation
        coords_row   = []
        context_row  = []
        split_row    = []
        shard_row    = []
        for rank, (val, meta) in enumerate(results):
            if rank >= top_n:
                break
            act_values[fi, rank]  = val
            token_pos[fi, rank]   = meta["tok_pos"]
            coords_row.append(meta["coord"])
            context_row.append(meta["context"])
            split_row.append(meta["split"])
            shard_row.append(meta["shard_path"])
        # Pad if fewer than top_n hits found
        while len(coords_row) < top_n:
            coords_row.append(None)
            context_row.append([None] * (2 * context_len + 1))
            split_row.append("")
            shard_row.append("")
        coords_out.append(coords_row)
        context_out.append(context_row)
        split_out.append(split_row)
        shard_out.append(shard_row)

    save_dict = {
        "act_values":   act_values,
        "token_pos":    token_pos,
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

    # CSV (flat table — one row per (feature, rank))
    csv_path = os.path.join(out_dir, "top_activations.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature_idx", "rank", "activation_value",
            "split", "shard_path", "seq_idx", "tok_pos",
            "coord_chrom", "coord_start", "coord_end",
            "context_window",
        ])
        for fi in range(n_features):
            results = top_heaps[fi].sorted_results()
            for rank, (val, meta) in enumerate(results):
                coord = meta["coord"]
                if coord is not None and len(coord) == 3:
                    chrom, cs, ce = coord
                else:
                    chrom, cs, ce = "", "", ""
                ctx_str = ";".join(
                    f"{c[0]}:{c[1]}-{c[2]}" if (c is not None and len(c) == 3) else "None"
                    for c in meta["context"]
                )
                writer.writerow([
                    fi, rank, f"{val:.6f}",
                    meta["split"], meta["shard_path"],
                    meta["seq_idx"], meta["tok_pos"],
                    chrom, cs, ce,
                    ctx_str,
                ])
    print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Find top-N activating tokens per SAE feature.")
    p.add_argument("--sae_checkpoint", required=True, help="Path to SAE .pt checkpoint")
    p.add_argument("--sae_cfg",        required=True, help="Path to SAE cfg JSON file")
    p.add_argument("--save_dir",       required=True, help="Root dir containing train/val/test splits")
    p.add_argument("--layer",          type=int, required=True, help="Which layer's embeddings to use")
    p.add_argument("--splits",         nargs="+", default=["train", "val", "test"],
                   help="Which splits to scan (default: train val test)")
    p.add_argument("--top_n",          type=int, default=10, help="Top-N tokens per feature")
    p.add_argument("--context_len",    type=int, default=5,  help="Tokens of context on each side")
    p.add_argument("--out_dir",        required=True, help="Where to write results")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size",     type=int, default=2048,
                   help="Sub-batch size for SAE forward pass (tune to VRAM)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.sae_cfg) as f:
        cfg = json.load(f)
    cfg = restore_cfg_types(cfg)
    cfg["device"] = args.device

    print(f"Loading SAE from {args.sae_checkpoint} ...")
    sae = load_sae(cfg, args.sae_checkpoint, args.device)
    n_features = cfg["dict_size"]
    act_size   = cfg["act_size"]

    print(f"SAE: dict_size={n_features}, act_size={act_size}")
    print(f"Scanning splits: {args.splits}  |  top_n={args.top_n}  |  context_len={args.context_len}")

    # Initialise one heap per feature
    top_heaps = [TopNHeap(args.top_n) for _ in range(n_features)]

    for split in args.splits:
        layer_dir = os.path.join(args.save_dir, split, f"layer_{args.layer}")
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
            top_heaps=top_heaps,
            batch_size=args.batch_size,
            act_size=act_size,
            device=args.device,
        )

    print("\nSaving results ...")
    save_results(top_heaps, args.top_n, args.context_len, args.layer, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()