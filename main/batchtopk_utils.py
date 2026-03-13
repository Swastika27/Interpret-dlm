import torch
import json

from BatchTopK.sae import BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE, JumpReLUInferenceSAE


def restore_cfg_types(cfg):
    if isinstance(cfg.get("dtype"), str):
        cfg["dtype"] = getattr(torch, cfg["dtype"].replace("torch.", ""))
    if isinstance(cfg.get("device"), str):
        cfg["device"] = torch.device(cfg["device"])
    return cfg

def load_cfg(cfg_path, device):
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg = restore_cfg_types(cfg)
    cfg["device"] = device
    return cfg

def load_sae(cfg: dict, checkpoint_path: str, device: str):
    state = torch.load(checkpoint_path, map_location=device)
    print(cfg.keys())

    # Unwrap nested checkpoint formats
    sae_state = state.get("sae_state_dict") or state.get("model_state_dict") or state
    saved_cfg  = state.get("cfg", cfg)       # checkpoint may carry its own cfg
    theta      = state.get("theta") or saved_cfg.get("theta")

    arch = cfg.get("sae_type", "batchtopk").lower()
    cls_map = {
        "batchtopk": BatchTopKSAE,
        "top_k":       TopKSAE,
        "vanilla":     VanillaSAE,
        "jumprelu":    JumpReLUSAE,
    }
    sae = cls_map[arch](cfg)
    sae.load_state_dict(sae_state, strict=False)

    # Wrap BatchTopK with JumpReLU inference gate using saved theta
    if arch == "batchtopk":
        if theta is None:
            raise ValueError(
                "BatchTopKSAE checkpoint has no 'theta'. "
                "Ensure save_checkpoint stores it (cfg['theta'] = theta)."
            )
        print(f"Wrapping BatchTopKSAE with JumpReLUInferenceSAE (theta={theta:.6f})")
        sae = JumpReLUInferenceSAE(sae, theta=theta)

    sae.eval().to(device)
    return sae