"""
Shared genomic coordinate helpers: chromosome name normalization (chr prefix)
and token-level positions aligned with concept_feature_analysis labelling.
"""

from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np


def infer_use_chr_from_chroms(chroms: Iterable[str]) -> bool:
    """Majority vote: True if most non-empty chrom strings use the 'chr' prefix."""
    chroms = [str(c) for c in chroms if c is not None and str(c) != ""]
    if not chroms:
        return True
    n_chr = sum(1 for c in chroms if c.startswith("chr"))
    return n_chr >= len(chroms) - n_chr


def normalize_chrom_name(chrom: str, use_chr: bool) -> str:
    chrom = str(chrom)
    if use_chr and not chrom.startswith("chr"):
        return "chr" + chrom
    if not use_chr and chrom.startswith("chr"):
        return chrom[3:]
    return chrom


def token_midpoint_genomic_pos(seq_start: int, seq_end: int, L: int, tok_pos: int) -> int:
    """
    Same genomic point as concept_feature_analysis._build_token_positions uses
    for BED contains_batch (half-open interval membership for integer position).
    """
    bp_per_token = (float(seq_end) - float(seq_start)) / float(L)
    mid = float(seq_start) + (float(tok_pos) + 0.5) * bp_per_token
    return int(np.int64(mid))


def token_one_bp_bed(
    chrom: str,
    seq_start: int,
    seq_end: int,
    L: int,
    tok_pos: int,
    use_chr: bool,
) -> Tuple[str, int, int]:
    """
    Half-open BED interval [p, p+1) for the activating token, chrom-normalized.
    """
    cnorm = normalize_chrom_name(chrom, use_chr)
    p = token_midpoint_genomic_pos(int(seq_start), int(seq_end), int(L), int(tok_pos))
    return (cnorm, p, p + 1)


def infer_use_chr_from_save_dir(save_dir: str, splits: List[str], layer: int) -> bool:
    """
    Match concept BEDs to embedding shards: infer chr convention from coordinates
    in the first available shard under save_dir/<split>/layer_<layer>/.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required for infer_use_chr_from_save_dir") from e

    for split in splits:
        layer_dir = os.path.join(save_dir, split, f"layer_{layer}")
        shards = sorted(glob.glob(os.path.join(layer_dir, "shard_*.pt")))
        for sp in shards:
            shard = torch.load(sp, map_location="cpu")
            coords = shard.get("coords")
            if not coords:
                continue
            chroms: List[str] = []
            for row in coords:
                if isinstance(row, (list, tuple)) and len(row) >= 1 and row[0]:
                    chroms.append(str(row[0]))
            if chroms:
                return infer_use_chr_from_chroms(chroms)
    return True


def infer_use_chr_from_top_activation_coords(coords: list) -> bool:
    """Infer chr convention from nested coords lists in top_activations.pt."""
    chroms: List[str] = []
    for row in coords:
        if not isinstance(row, list):
            continue
        for c in row:
            if c is None:
                continue
            if isinstance(c, (list, tuple)) and len(c) >= 1 and c[0]:
                chroms.append(str(c[0]))
    return infer_use_chr_from_chroms(chroms)
