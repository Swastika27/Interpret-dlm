# import wandb
import json
import os
import random
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

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
    torch.save(payload, path)
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
    payload = torch.load(path, map_location=map_dev)

    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")

    sae.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    activation_store.load_state_dict(payload["activation_store"])

    rng = payload.get("rng") or {}
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
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