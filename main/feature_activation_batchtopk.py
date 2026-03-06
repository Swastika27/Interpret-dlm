#!/usr/bin/env python3
import os, random, heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch

from train_batchtopk import BatchTopKSAE, LRUCache, build_sequence_index

# -----------------------------
# Metadata
# -----------------------------
@dataclass(frozen=True)
class TokMeta:
    shard_path: str
    row_idx: int
    pos: int
    chrom: str
    start: int
    end: int

    @property
    def genome_pos0(self) -> int:
        return self.start + self.pos


# -----------------------------
# Top-N heap per feature
# -----------------------------
class TopNHeap:
    def __init__(self, N: int):
        self.N = N
        self.h: List[Tuple[float, int, Any]] = []  # (score, tie, payload)
        self._tie = 0

    def push(self, score: float, payload: Any) -> None:
        self._tie += 1
        item = (float(score), self._tie, payload)
        if len(self.h) < self.N:
            heapq.heappush(self.h, item)
        else:
            if score > self.h[0][0]:
                heapq.heapreplace(self.h, item)

    def items_desc(self):
        # return list of (score, payload) sorted descending
        items = sorted(self.h, key=lambda x: x[0], reverse=True)
        return [(s, p) for (s, _tie, p) in items]


# -----------------------------
# Batch sampler (token + meta)
# -----------------------------
@torch.no_grad()
def make_token_batch_with_meta(seq_refs, cache, batch_seq_indices, seq_len, device):
    # group requested rows by shard
    by_shard: Dict[str, List[int]] = {}
    for gi in batch_seq_indices:
        sr = seq_refs[gi]
        by_shard.setdefault(sr.shard_path, []).append(sr.row_idx)

    tokens = []
    metas: List[TokMeta] = []

    for sp, rows in by_shard.items():
        obj = cache.get(sp)
        if obj is None:
            obj = torch.load(sp, map_location="cpu")
            cache.put(sp, obj)

        emb = obj["emb"]  # [B_shard, L, D]
        coords = obj.get("coords", None)  # List[(chrom,start,end)] aligned to rows
        if coords is None:
            raise KeyError(f"{sp}: missing 'coords' in shard. Your embedding script should save it.")

        if int(emb.shape[1]) != seq_len:
            raise ValueError(f"{sp}: expected seq_len={seq_len}, got {int(emb.shape[1])}")

        for r in rows:
            pos = random.randrange(seq_len)
            tok = emb[r, pos, :]
            chrom, start, end = coords[r]

            tokens.append(tok)
            metas.append(
                TokMeta(
                    shard_path=sp,
                    row_idx=int(r),
                    pos=int(pos),
                    chrom=str(chrom),
                    start=int(start),
                    end=int(end),
                )
            )

    x = torch.stack(tokens, dim=0).to(device=device, dtype=torch.float32)
    return x, metas


# -----------------------------
# Main
# -----------------------------
def main():
    ckpt_path = "../trained_models/layer8_bt8/checkpoints/final.pt"
    data_root = "../data/embeddings"
    split = "train"
    layer_dir_name = "layer_8"
    seq_len = 2000
    batch_tokens = 256
    k_per_token = 8
    steps = 2000
    topN = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    d_in = int(args["d_in"])
    d_sae = int(args["d_sae"])

    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Decoder column norms for comparability
    Wdec = model.W_dec.weight.detach()         # [d_in, d_sae]
    dec_norm = torch.norm(Wdec, dim=0).to(device)
    dec_norm = torch.clamp(dec_norm, min=1e-8)

    # sequence index + cache
    layer_dir = os.path.join(data_root, split, layer_dir_name)
    seq_refs, n_seq = build_sequence_index(layer_dir)
    cache = LRUCache(max_items=8)

    # heaps and counts
    heaps = [TopNHeap(topN) for _ in range(d_sae)]
    fire_count = torch.zeros((d_sae,), dtype=torch.long)  # CPU is fine
    seen_tokens = 0

    for step in range(steps):
        batch_seq_indices = random.sample(range(n_seq), k=batch_tokens)
        x, metas = make_token_batch_with_meta(seq_refs, cache, batch_seq_indices, seq_len, device)

        k_total = batch_tokens * k_per_token
        _, z, z_metrics = model(x, k_total=k_total)  # z: [B, d_sae]

        z_scaled = z * dec_norm.unsqueeze(0)

        idx = (z > 0).nonzero(as_tuple=False)  # [nnz, 2] -> (i, j)
        if idx.numel() > 0:
            i = idx[:, 0]
            j = idx[:, 1]

            scores = z_scaled[i, j].detach().cpu()
            raws = z[i, j].detach().cpu()

            # update fire counts (on CPU)
            fire_count.index_add_(0, j.detach().cpu(), torch.ones_like(j.detach().cpu(), dtype=torch.long))

            # push top examples
            for t in range(idx.shape[0]):
                ii = int(i[t].item())
                jj = int(j[t].item())
                meta = metas[ii]
                payload = {
                    "score": float(scores[t].item()),
                    "raw": float(raws[t].item()),
                    "shard_path": meta.shard_path,
                    "row_idx": meta.row_idx,
                    "pos": meta.pos,
                    "chrom": meta.chrom,
                    "start": meta.start,
                    "end": meta.end,
                    "genome_pos0": meta.genome_pos0,
                }
                heaps[jj].push(float(scores[t].item()), payload)

        seen_tokens += batch_tokens
        if (step + 1) % 200 == 0:
            print(f"scanned batches={step+1}, tokens={seen_tokens}, nnz={int(z_metrics['nnz'].item())}")

    out = {
        "ckpt": ckpt_path,
        "split": split,
        "layer": layer_dir_name,
        "seen_tokens": seen_tokens,
        "k_per_token": k_per_token,
        "topN": topN,
        "dec_norm": dec_norm.detach().cpu(),
        "fire_count": fire_count,
        "features": [],
    }

    for fid in range(d_sae):
        examples = []
        for score, payload in heaps[fid].items_desc():
            examples.append(payload)
        out["features"].append(
            {
                "feature_id": fid,
                "fire_count": int(fire_count[fid].item()),
                "examples": examples,
            }
        )

    torch.save(out, "feature_top_examples.pt")
    print("Saved feature_top_examples.pt")


if __name__ == "__main__":
    main()