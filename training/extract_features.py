#!/usr/bin/env python3
import os, glob, random, heapq, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

# ---- import your model definition exactly ----
# from train_batchtopk import BatchTopKSAE, LRUCache, SeqRef, build_sequence_index

@dataclass
class TokMeta:
    shard_path: str
    row_idx: int
    pos: int
    # optionally: coords info (chr, start) if present in shard

class TopN:
    """Min-heap keeping top-N (score, meta, raw) entries."""
    def __init__(self, N: int):
        self.N = N
        self.h: List[Tuple[float, TokMeta, float]] = []

    def push(self, score: float, meta: TokMeta, raw: float):
        item = (float(score), meta, float(raw))
        if len(self.h) < self.N:
            heapq.heappush(self.h, item)
        else:
            if score > self.h[0][0]:
                heapq.heapreplace(self.h, item)

    def sorted_desc(self):
        return sorted(self.h, key=lambda x: x[0], reverse=True)

@torch.no_grad()
def make_token_batch_with_meta(seq_refs, cache, batch_seq_indices, seq_len, device):
    # group by shard like your original
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

        for r in rows:
            pos = random.randrange(seq_len)
            tok = emb[r, pos, :]
            tokens.append(tok)
            metas.append(TokMeta(shard_path=sp, row_idx=r, pos=pos))

    x = torch.stack(tokens, dim=0).to(device=device, dtype=torch.float32)
    return x, metas

def main():
    ckpt_path = "runs/.../checkpoints/best.pt"
    data_root = "/path/to/embeddings"
    split = "train"
    layer_dir_name = "layer_5"
    seq_len = 2000
    batch_tokens = 256
    k_per_token = 8
    steps = 20000               # how many batches to scan
    topN = 200                  # top examples per feature
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    d_in = int(args["d_in"])
    d_sae = int(args["d_sae"])

    model = BatchTopKSAE(d_in=d_in, d_sae=d_sae).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # decoder column norms for comparability
    # W_dec.weight: [d_in, d_sae] in PyTorch Linear (out_features x in_features)
    Wdec = model.W_dec.weight.detach()  # [d_in, d_sae]
    dec_norm = torch.norm(Wdec, dim=0).to(device)  # [d_sae]
    dec_norm = torch.clamp(dec_norm, min=1e-8)

    # sequence index + cache
    layer_dir = os.path.join(data_root, split, layer_dir_name)
    seq_refs, n_seq = build_sequence_index(layer_dir)
    cache = LRUCache(max_items=8)

    # heaps for each feature
    heaps = [TopN(topN) for _ in range(d_sae)]
    fire_count = torch.zeros((d_sae,), dtype=torch.long)  # count of times feature selected (nonzero)
    seen_tokens = 0

    for step in range(steps):
        batch_seq_indices = random.sample(range(n_seq), k=batch_tokens)
        x, metas = make_token_batch_with_meta(seq_refs, cache, batch_seq_indices, seq_len, device)

        k_total = batch_tokens * k_per_token
        x_hat, z, z_metrics = model(x, k_total=k_total)  # z: [B, d_sae]

        # rank by scaled score for top examples
        z_scaled = z * dec_norm.unsqueeze(0)

        idx = (z > 0).nonzero(as_tuple=False)  # [nnz, 2] with (i, j)
        if idx.numel() > 0:
            i = idx[:, 0]
            j = idx[:, 1]
            scores = z_scaled[i, j].detach().cpu()
            raws = z[i, j].detach().cpu()

            # update counts + heaps
            fire_count.index_add_(0, j.detach().cpu(), torch.ones_like(j.detach().cpu(), dtype=torch.long))
            for t in range(idx.shape[0]):
                ii = int(i[t].item())
                jj = int(j[t].item())
                meta = metas[ii]
                heaps[jj].push(float(scores[t].item()), meta, float(raws[t].item()))

        seen_tokens += batch_tokens

        if (step + 1) % 200 == 0:
            print(f"scanned batches={step+1}, tokens={seen_tokens}, nnz~={int(z_metrics['nnz'].item())}")

    # save feature cards
    out = {
        "ckpt": ckpt_path,
        "split": split,
        "layer": layer_dir_name,
        "seen_tokens": seen_tokens,
        "k_per_token": k_per_token,
        "topN": topN,
        "dec_norm": dec_norm.detach().cpu(),
        "fire_count": fire_count,
        "features": []
    }

    for j in range(d_sae):
        examples = []
        for score, meta, raw in heaps[j].sorted_desc():
            examples.append({
                "score": score,
                "raw": raw,
                "shard_path": meta.shard_path,
                "row_idx": meta.row_idx,
                "pos": meta.pos,
            })
        out["features"].append({
            "feature_id": j,
            "fire_count": int(fire_count[j].item()),
            "examples": examples,
        })

    torch.save(out, "feature_top_examples.pt")
    print("Saved feature_top_examples.pt")

if __name__ == "__main__":
    main()