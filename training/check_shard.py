import torch

print(torch.load("../data/embeddings/train/layer_5/shard_00000.pt")["emb"].shape)