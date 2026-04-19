# import wandb
import json
import os
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# region agent log
_AGENT_DEBUG_LOG = Path(__file__).resolve().parents[2] / "debug-d522a1.log"


def _agent_debug_ndjson(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    try:
        payload = {
            "sessionId": "d522a1",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
        }
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def _approx_tensor_bytes(obj: Any) -> int:
    n = 0

    def walk(x: Any) -> None:
        nonlocal n
        if torch.is_tensor(x):
            n += int(x.numel()) * int(x.element_size())
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(obj)
    return n


# endregion

def init_wandb(cfg):
    return wandb.init(project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True)

def log_wandb(output, step, wandb_run, index=None):
    metrics_to_log = ["loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)

@torch.no_grad()
def log_model_performance(wandb_run, step, model, activations_store, sae, index=None, batch_tokens=None):
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[:sae.cfg["batch_size"] // sae.cfg["seq_len"]]
    batch = activations_store.get_activations(batch_tokens).reshape(-1, sae.cfg["act_size"])

    sae_output = sae(batch)["sae_out"].reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], partial(reconstr_hook, sae_out=sae_output))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

def save_checkpoint(wandb_run, sae, cfg, step):
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    # Create and log artifact
    artifact = wandb.Artifact(
        name=f"{cfg['name']}_{step}",
        type="model",
        description=f"Model checkpoint at step {step}",
    )
    artifact.add_file(sae_path)
    artifact.add_file(config_path)
    wandb_run.log_artifact(artifact)

    print(f"Model and config saved as artifact at step {step}")


def _cfg_json_safe(cfg: dict) -> dict:
    out: Dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        elif isinstance(value, torch.dtype):
            out[key] = str(value)
        elif isinstance(value, torch.device):
            out[key] = str(value)
        else:
            out[key] = str(value)
    return out


def _update_latest_pointer(save_dir: str, target_abs_path: str, link_name: str) -> None:
    """Point link_name in save_dir at target_abs_path (symlink, or copy on Windows)."""
    latest_path = os.path.join(save_dir, link_name)
    rel = os.path.relpath(os.path.abspath(target_abs_path), start=os.path.abspath(save_dir))
    try:
        if os.path.lexists(latest_path) or os.path.isfile(latest_path):
            try:
                os.unlink(latest_path)
            except OSError:
                pass
        os.symlink(rel, latest_path, target_is_directory=False)
    except OSError:
        shutil.copy2(target_abs_path, latest_path)


def save_checkpoint(sae, cfg, theta, step):
    save_dir = f"{cfg['run_dir']}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, f"step_{step}.pt")
    payload = {"state_dict": sae.state_dict(), "cfg": _cfg_json_safe(cfg)}
    if theta is not None:
        payload["theta"] = theta
    torch.save(payload, sae_path)
    # region agent log
    try:
        du = shutil.disk_usage(save_dir)
        st = os.stat(sae_path)
        _agent_debug_ndjson(
            "H4",
            "logs.py:save_checkpoint",
            "after sae-only torch.save",
            {
                "sae_path": sae_path,
                "step": int(step),
                "written_bytes": st.st_size,
                "disk_free_bytes": du.free,
            },
            run_id="pre-fix",
        )
    except Exception:
        pass
    # endregion

    _update_latest_pointer(save_dir, sae_path, "latest.pt")


def save_training_checkpoint(
    sae,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    theta: Optional[float],
    global_step: int,
    gated_state: Dict[str, Any],
    activation_store,
) -> str:
    """
    Full training state for resume: model, optimizer, RNG, gated warm-up, data iterator.

    global_step: number of completed optimizer steps (next loop index).
    Writes training_step_{global_step}.pt and updates latest_training.pt.
    """
    save_dir = f"{cfg['run_dir']}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"training_step_{global_step}.pt")

    rng: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    else:
        rng["cuda"] = None

    payload = {
        "format_version": 1,
        "global_step": int(global_step),
        "state_dict": sae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "theta": theta,
        "gated_state": dict(gated_state),
        "cfg_snapshot": _cfg_json_safe(cfg),
        "rng": rng,
        "activation_store": activation_store.state_dict(),
    }
    # region agent log
    try:
        du = shutil.disk_usage(save_dir)
        approx_b = _approx_tensor_bytes(payload)
        _agent_debug_ndjson(
            "H1",
            "logs.py:save_training_checkpoint",
            "pre torch.save disk + payload",
            {
                "path": path,
                "save_dir": save_dir,
                "global_step": int(global_step),
                "disk_free_bytes": du.free,
                "disk_total_bytes": du.total,
                "approx_payload_tensor_bytes": approx_b,
            },
            run_id="pre-fix",
        )
    except Exception as e:
        _agent_debug_ndjson(
            "H3",
            "logs.py:save_training_checkpoint",
            "pre-save probe failed",
            {"err": repr(e)},
            run_id="pre-fix",
        )
    # endregion
    try:
        torch.save(payload, path)
    except Exception as e:
        # region agent log
        _agent_debug_ndjson(
            "H1",
            "logs.py:save_training_checkpoint",
            "torch.save raised",
            {
                "path": path,
                "exc_type": type(e).__name__,
                "exc_repr": repr(e),
                "errno": getattr(e, "errno", None),
            },
            run_id="pre-fix",
        )
        # endregion
        raise
    # region agent log
    try:
        st = os.stat(path)
        _agent_debug_ndjson(
            "H2",
            "logs.py:save_training_checkpoint",
            "torch.save ok",
            {"path": path, "written_bytes": st.st_size},
            run_id="pre-fix",
        )
    except Exception as e:
        _agent_debug_ndjson(
            "H3",
            "logs.py:save_training_checkpoint",
            "post-save stat failed",
            {"path": path, "err": repr(e)},
            run_id="pre-fix",
        )
    # endregion
    _update_latest_pointer(save_dir, path, "latest_training.pt")
    print(f"  [checkpoint] Saved full training state: {path}")
    return path


def resolve_training_checkpoint_path(raw: str) -> str:
    """File path, or directory containing latest_training.pt."""
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    if p.is_dir():
        cand = p / "latest_training.pt"
        if cand.is_file():
            return str(cand.resolve())
    raise FileNotFoundError(
        f"Training checkpoint not found: {raw} "
        "(expected a .pt file or a directory with latest_training.pt)"
    )


def load_training_checkpoint(
    path: str,
    sae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    activation_store,
) -> Dict[str, Any]:
    map_dev = next(sae.parameters()).device
    # PyTorch 2.6+ defaults weights_only=True; full checkpoints contain numpy/python RNG state.
    try:
        payload = torch.load(path, map_location=map_dev, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_dev)

    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")

    sae.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    activation_store.load_state_dict(payload["activation_store"])

    rng = payload.get("rng") or {}
    # map_location=cuda moves every tensor in the file; RNG states must stay CPU for set_rng_state*.
    def _rng_to_cpu(x):
        return x.cpu().contiguous() if torch.is_tensor(x) else x

    if "torch" in rng:
        torch.set_rng_state(_rng_to_cpu(rng["torch"]))
    if rng.get("cuda") is not None and torch.cuda.is_available():
        cuda_st = rng["cuda"]
        if isinstance(cuda_st, (list, tuple)):
            cuda_st = [_rng_to_cpu(t) for t in cuda_st]
        else:
            cuda_st = _rng_to_cpu(cuda_st)
        torch.cuda.set_rng_state_all(cuda_st)
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])

    return {
        "global_step": int(payload["global_step"]),
        "gated_state": payload.get("gated_state") or {},
        "theta": payload.get("theta"),
        "cfg_snapshot": payload.get("cfg_snapshot"),
    }