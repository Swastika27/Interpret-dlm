#!/usr/bin/env python3
import os, glob, random, heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- copy your SAE class exactly (from training script) ----
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


@dataclass(frozen=True)
class TokMeta:
    shard_path: str
    win_idx: int
    pos: int
    chrom: str
    start: int
    end: int

    @property
    def genome_pos0(self) -> int:
        return self.start + self.pos


class LRUCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self._cache: Dict[str, Any] = {}
        self._order: List[str] = []

    def get(self, path: str) -> Optional[Dict]:
        if path not in self._cache:
            return None
        # refresh
        self._order.remove(path)
        self._order.append(path)
        return self._cache[path]

    def put(self, path: str, obj: Dict) -> None:
        if path in self._cache:
            self._order.remove(path)
        self._cache[path] = obj
        self._order.append(path)
        while len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._cache.pop(old, None)


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


@torch.no_grad()
def sample_token_batch_with_meta(
    shard_paths: List[str],
    cache: LRUCache,
    batch_windows: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[TokMeta]]:
    # sample distinct windows by sampling distinct (shard_path, win_idx)
    # easiest: sample distinct global window ids by first expanding counts lazily
    # here: load each chosen shard and sample a random win_idx within it
    chosen_shards = random.sample(shard_paths, k=min(batch_windows, len(shard_paths)))

    xs = []
    metas: List[TokMeta] = []

    # if batch_windows > len(shard_paths), we allow reusing shards but not windows within a shard
    while len(xs) < batch_windows:
        sp = random.choice(chosen_shards)
        obj = cache.get(sp)
        if obj is None:
            obj = torch.load(sp, map_location="cpu")
            cache.put(sp, obj)
        emb = obj["emb"]          # [Bwin, L, D]
        coords = obj["coords"]    # list aligned to rows
        Bwin, L, _ = emb.shape
        if L != seq_len:
            raise ValueError(f"{sp}: seq_len mismatch: expected {seq_len}, got {L}")

        win_idx = random.randrange(Bwin)
        pos = random.randrange(seq_len)
        x = emb[win_idx, pos, :]
        chrom, start, end = coords[win_idx]

        xs.append(x)
        metas.append(TokMeta(
            shard_path=sp,
            win_idx=win_idx,
            pos=pos,
            chrom=str(chrom),
            start=int(start),
            end=int(end),
        ))

    X = torch.stack(xs, dim=0).to(device=device, dtype=torch.float32)
    return X, metas


def main():
    ckpt_path = "PATH/TO/ckpt_step_XXXXXXX.pt"
    emb_dir = "PATH/TO/data/embeddings/train/layer_5"  # must match ckpt["layer"]
    steps = 5000
    batch_tokens = 256
    topN = 200
    topk_per_token = 32   # only consider top 32 activations per token (speed)
    cache_items = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    d_in = int(ckpt["d_in"])
    d_hidden = int(ckpt["d_hidden"])
    seq_len = int(ckpt["seq_len"])
    sae_layer = int(ckpt["layer"])

    shard_paths = sorted(glob.glob(os.path.join(emb_dir, "shard_*.pt")))
    if not shard_paths:
        raise SystemExit(f"No shards found in {emb_dir}")

    # sanity: shard layer_index matches
    sh0 = torch.load(shard_paths[0], map_location="cpu")
    if int(sh0["layer_index"]) != sae_layer:
        raise SystemExit(f"Layer mismatch: ckpt layer={sae_layer} but shards layer_index={sh0['layer_index']}")

    model = SparseAutoencoder(d_in=d_in, d_hidden=d_hidden, use_relu=True).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    heaps = [TopNHeap(topN) for _ in range(d_hidden)]
    fire_count = torch.zeros((d_hidden,), dtype=torch.long)
    cache = LRUCache(max_items=cache_items)

    for s in range(steps):
        x, metas = sample_token_batch_with_meta(
            shard_paths=shard_paths,
            cache=cache,
            batch_windows=batch_tokens,
            seq_len=seq_len,
            device=device,
        )

        a = model.encode(x)  # [B, H]

        # consider only top-k features per token
        vals, idx = torch.topk(a, k=min(topk_per_token, d_hidden), dim=1)
        vals = vals.detach().cpu()
        idx = idx.detach().cpu()

        for i in range(batch_tokens):
            meta = metas[i]
            for r in range(vals.shape[1]):
                v = float(vals[i, r].item())
                if v <= 0:
                    break
                j = int(idx[i, r].item())
                fire_count[j] += 1
                heaps[j].push(v, {
                    "act": v,
                    "chrom": meta.chrom,
                    "start": meta.start,
                    "end": meta.end,
                    "pos": meta.pos,
                    "genome_pos0": meta.genome_pos0,
                    "shard_path": meta.shard_path,
                    "win_idx": meta.win_idx,
                })

        if (s + 1) % 200 == 0:
            print(f"scanned batches={s+1}, tokens={(s+1)*batch_tokens}")

    out = {
        "ckpt": ckpt_path,
        "layer": sae_layer,
        "seq_len": seq_len,
        "seen_tokens": steps * batch_tokens,
        "topN": topN,
        "topk_per_token": topk_per_token,
        "fire_count": fire_count,
        "features": [
            {"feature_id": j, "fire_count": int(fire_count[j].item()),
             "examples": [p for _, p in heaps[j].items_desc()]}
            for j in range(d_hidden)
        ],
    }
    torch.save(out, "relu_sae_feature_top_examples.pt")
    print("Saved relu_sae_feature_top_examples.pt")


if __name__ == "__main__":
    main()