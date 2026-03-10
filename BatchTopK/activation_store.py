import torch
from torch.utils.data import TensorDataset, DataLoader
import glob
import random

class StreamingActivationsStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]

        # Find all shard files
        self.shard_paths = sorted(glob.glob(cfg["embedding_glob"]))
        print(f"Found {len(self.shard_paths)} activation shards")

        self.batch_size = cfg["batch_size"]
        self.act_size = cfg["act_size"]

        self._start_new_epoch()

    def _start_new_epoch(self):
        # Shuffle shards at the start of each epoch
        self.shard_order = self.shard_paths.copy()
        random.shuffle(self.shard_order)
        self.current_shard_idx = 0
        self.current_shard_tensor = None
        self.current_pos = 0

    def _load_shard(self, shard_path):
        shard_dict = torch.load(shard_path, map_location="cpu")
        x = shard_dict['emb'].reshape(-1, self.act_size)  # flatten (B, L, D) -> (B*L, D)
        # optional: shuffle tokens **within shard** to increase randomness
        perm = torch.randperm(x.size(0))
        x = x[perm]
        return x

    def next_batch(self):
        while True:
            if self.current_shard_tensor is None or self.current_pos >= len(self.current_shard_tensor):
                # Move to next shard
                if self.current_shard_idx >= len(self.shard_order):
                    # Finished all shards → new epoch
                    self._start_new_epoch()
                shard_path = self.shard_order[self.current_shard_idx]
                self.current_shard_tensor = self._load_shard(shard_path)
                self.current_pos = 0
                self.current_shard_idx += 1

            # Take a batch from current shard
            end_pos = self.current_pos + self.batch_size
            batch = self.current_shard_tensor[self.current_pos:end_pos]
            self.current_pos = end_pos

            if len(batch) < self.batch_size:
                # If last batch in shard is smaller than batch_size, continue to next shard
                continue

            return batch.to(self.device)