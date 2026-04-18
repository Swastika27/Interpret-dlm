#!/usr/bin/env python3
"""
Extract token-level HyenaDNA embeddings for BED windows (train/val/test subsets),
for selected hidden-state layers, and save them as shard .pt files.

Inputs:
  - genome FASTA (hg38 primary assembly)
  - BED files: train.sub.bed, val.sub.bed, test.sub.bed (3-column BED)
Output:
  save_dir/
    train/layer_2/shard_00000.pt
    train/layer_5/shard_00000.pt
    train/layer_9/shard_00000.pt
    train/coords/shard_00000.pt
    ... (same for val/test)

Each shard file contains:
  - "emb": FloatTensor [B, L, D]  (token-level, no pooling)
  - "coords": List[Tuple[str,int,int]]  (chrom,start,end) aligned to emb rows
  - "model_id", "layer_index", "dtype", "seq_len"

Notes:
  - Uses output_hidden_states=True, so layer indices are:
      0 = embedding output, 1..n_layer = block outputs.
  - Keeps compute in float32 (recommended for your setup).
"""

import argparse
import os
import sys
from typing import List, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.gpu_setup import (  # noqa: E402
    configure_cuda_performance,
    resolve_device_str,
    tensor_to_device_fast,
)

try:
    from pyfaidx import Fasta
except ImportError as e:
    raise SystemExit("Missing dependency pyfaidx. Install: pip install pyfaidx") from e


BedRow = Tuple[str, int, int]


def read_bed(path: str, expected_len: int) -> List[BedRow]:
    rows: List[BedRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
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
    return rows


def sanitize_dna(seq: str) -> str:
    # Uppercase and map anything outside A/C/G/T/N to N (hg38 can include other IUPAC codes)
    seq = seq.upper()
    allowed = {"A", "C", "G", "T", "N"}
    if all(c in allowed for c in seq):
        return seq
    return "".join(c if c in allowed else "N" for c in seq)


def fetch_batch(genome: Fasta, batch: List[BedRow]) -> List[str]:
    seqs: List[str] = []
    for chrom, start, end in batch:
        # pyfaidx uses 0-based slicing consistent with BED: [start:end)
        s = str(genome[chrom][start:end])
        seqs.append(sanitize_dna(s))
    return seqs


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="Path to hg38 primary assembly FASTA")
    ap.add_argument("--bed", required=True, help="Path to BED subset (e.g., train.sub.bed)")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"], help="Split name for output folders")
    ap.add_argument("--save_dir", required=True, help="Output directory root")
    ap.add_argument("--model_id", default="LongSafari/hyenadna-large-1m-seqlen-hf", help="HF model id")
    ap.add_argument("--layers", default="2,5,9", help="Comma-separated hidden_state indices to save (e.g., 2,5,9)")
    ap.add_argument("--seq_len", type=int, default=10000, help="Window length (must match BED window size)")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size (increase carefully for long seq)")
    ap.add_argument("--num_shards_per_layer", type=int, default=0,
                    help="Optional: stop after writing this many shards (0 = no limit)")
    ap.add_argument("--device", default=None, help="cuda, cpu, or leave unset for auto")
    ap.add_argument("--dtype_save", default="float32", choices=["float32", "float16"],
                    help="Storage dtype for embeddings (compute stays float32). float16 saves disk.")
    args = ap.parse_args()

    device = resolve_device_str(args.device)
    configure_cuda_performance()
    print("Using device:", device)

    selected_layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    if not selected_layers:
        raise SystemExit("No layers selected. Use --layers like 2,5,9")

    # Load genome
    genome = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)

    # Load model/tokenizer
    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Keep init + compute in float32 (your preference; avoids precision concerns)
    model = AutoModel.from_pretrained(
        args.model_id,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).eval()

    # Read BED
    rows = read_bed(args.bed, expected_len=args.seq_len)
    if not rows:
        raise SystemExit(f"No BED rows found with length {args.seq_len} in {args.bed}")
    print(f"Loaded {len(rows)} windows from {args.bed}")

    # Prepare output dirs
    split_root = os.path.join(args.save_dir, args.split)
    ensure_dir(split_root)
    coords_dir = os.path.join(split_root, "coords")
    ensure_dir(coords_dir)

    layer_dirs = {}
    for li in selected_layers:
        d = os.path.join(split_root, f"layer_{li}")
        ensure_dir(d)
        layer_dirs[li] = d

    # Run in shards
    shard_idx = 0
    i = 0
    save_dtype = torch.float16 if args.dtype_save == "float16" else torch.float32

    # Validate layer indices once (using model.config.n_layer if present)
    # hidden_states len should be n_layer + 1, but we validate after first forward too.
    while i < len(rows):
        batch_rows = rows[i:i + args.batch_size]
        seqs = fetch_batch(genome, batch_rows)

        # Tokenize
        # Important: add_special_tokens=False keeps 1 token per base (for this tokenizer implementation)
        inputs = tok(seqs, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)
        # Expect exact length == seq_len for all sequences
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] != args.seq_len:
            raise RuntimeError(
                f"Tokenized length {input_ids.shape[1]} != expected {args.seq_len}. "
                f"Check tokenizer behavior / special tokens."
            )

        inputs = {k: tensor_to_device_fast(v, device) for k, v in inputs.items()}

        out = model(**inputs, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states  # tuple of (B, L, D)

        if shard_idx == 0 and i == 0:
            print("num hidden_states:", len(hs))
            print("hidden_state[0] shape:", tuple(hs[0].shape))
            for li in selected_layers:
                if li < 0 or li >= len(hs):
                    raise SystemExit(f"Requested layer {li}, but hidden_states has len={len(hs)}")

        coords = batch_rows  # list of tuples (chrom,start,end)

        # Save coords shard once per shard (shared across layers)
        coords_path = os.path.join(coords_dir, f"shard_{shard_idx:05d}.pt")
        torch.save(
            {
                "coords": coords,
                "model_id": args.model_id,
                "seq_len": args.seq_len,
            },
            coords_path,
        )

        # Save each selected layer embeddings
        for li in selected_layers:
            emb = hs[li].detach().to("cpu")  # (B, L, D) float32
            if save_dtype != torch.float32:
                emb = emb.to(dtype=save_dtype)

            out_path = os.path.join(layer_dirs[li], f"shard_{shard_idx:05d}.pt")
            torch.save(
                {
                    "emb": emb,  # (B, L, D)
                    "coords": coords,  # duplicated for convenience
                    "model_id": args.model_id,
                    "layer_index": li,
                    "dtype": str(emb.dtype).replace("torch.", ""),
                    "seq_len": args.seq_len,
                },
                out_path,
            )

        shard_idx += 1
        i += args.batch_size

        if args.num_shards_per_layer and shard_idx >= args.num_shards_per_layer:
            print(f"Stopping early after {shard_idx} shards due to --num_shards_per_layer")
            break

        if shard_idx % 50 == 0:
            print(f"Wrote {shard_idx} shards...")

    print(f"Done. Wrote {shard_idx} shard(s) for split={args.split} to {split_root}")


if __name__ == "__main__":
    main()