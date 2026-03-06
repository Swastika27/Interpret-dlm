import torch

print("Checkpoint keys: ", torch.load("../runs/sae/ckpt_step_0020000.pt").keys())

print("Shard keys: ", torch.load("../data/embeddings/train/layer_8/shard_00000.pt").keys())