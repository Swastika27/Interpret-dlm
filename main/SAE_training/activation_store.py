import glob
import json
import os
import random
import time
from typing import Any, Dict

import torch

_DBG_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "debug-2bbb4e.log")
)


def _agent_dbg_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    run_id: str = "pre-fix",
) -> None:
    # region agent log
    try:
        with open(_DBG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "2bbb4e",
                        "timestamp": int(time.time() * 1000),
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "runId": run_id,
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # endregion

class StreamingActivationsStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        # region agent log
        _agent_dbg_log(
            "H1",
            "activation_store.py:StreamingActivationsStore.__init__",
            "cfg device vs normalized self.device",
            {
                "cfg_device_repr": repr(cfg.get("device")),
                "cfg_device_type_name": type(cfg.get("device")).__name__,
                "self_device_type_name": type(self.device).__name__,
                "self_device_type_field": getattr(self.device, "type", None),
            },
            run_id="post-fix",
        )
        # endregion

        # Find all shard files
        self.shard_paths = sorted(glob.glob(cfg["embedding_glob"]))
        print(f"Found {len(self.shard_paths)} activation shards")

        self.batch_size = cfg["batch_size"]
        self.act_size = cfg["act_size"]

        self._start_new_epoch()

    def state_dict(self) -> Dict[str, Any]:
        """
        Snapshot iterator state for resume. Persists the in-memory shard tensor
        (CPU) so mid-shard resume stays consistent with the same row order.
        """
        state: Dict[str, Any] = {
            "shard_order": list(self.shard_order),
            "current_shard_idx": int(self.current_shard_idx),
            "current_pos": int(self.current_pos),
            "python_random": random.getstate(),
        }
        if self.current_shard_tensor is not None:
            state["current_shard_tensor"] = self.current_shard_tensor.detach().cpu().clone()
        else:
            state["current_shard_tensor"] = None
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.shard_order = list(state["shard_order"])
        self.current_shard_idx = int(state["current_shard_idx"])
        self.current_pos = int(state["current_pos"])
        random.setstate(state["python_random"])
        t = state.get("current_shard_tensor")
        if t is not None:
            # Checkpoints store CPU tensors; torch.load(map_location=cuda) may move nested
            # tensors to GPU — pin_memory requires CPU.
            self.current_shard_tensor = t.detach().contiguous().cpu()
            if self.device.type == "cuda":
                self.current_shard_tensor = self.current_shard_tensor.pin_memory()
        else:
            self.current_shard_tensor = None

    def _start_new_epoch(self):
        # Shuffle shards at the start of each epoch
        self.shard_order = self.shard_paths.copy()
        random.shuffle(self.shard_order)
        self.current_shard_idx = 0
        self.current_shard_tensor = None
        self.current_pos = 0

    def _load_shard(self, shard_path):
        # region agent log
        _agent_dbg_log(
            "H2",
            "activation_store.py:_load_shard",
            "self.device before .type access",
            {
                "self_device_type_name": type(self.device).__name__,
                "self_device_repr": repr(self.device),
                "device_type_eq_cuda": self.device.type == "cuda",
            },
            run_id="post-fix",
        )
        # endregion
        shard_dict = torch.load(shard_path, map_location="cpu")
        x = shard_dict['emb'].reshape(-1, self.act_size)  # flatten (B, L, D) -> (B*L, D)
        # optional: shuffle tokens **within shard** to increase randomness
        perm = torch.randperm(x.size(0))
        x = x[perm]
        if self.device.type == "cuda":
            x = x.pin_memory()
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

            if self.device.type == "cuda":
                return batch.to(self.device, non_blocking=True)
            return batch.to(self.device)