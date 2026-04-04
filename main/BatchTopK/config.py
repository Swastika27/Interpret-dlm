# import transformer_lens.utils as utils
import torch


def get_default_cfg():
    return {
        "seed": 49,
        "layer": 6,

        "batch_size": 512,
        "lr": 3e-4,

        "num_tokens": 2048000,

        "l1_coeff": 0,

        "beta1": 0.9,
        "beta2": 0.99,

        "max_grad_norm": 1.0,

        "dtype": torch.float32,

        "device": "cuda:0",

        "act_size": 256,  # HyenaDNA embedding dimension
        "dict_size": 8192,

        "embedding_glob": "embeddings/*.pt",
        "log_path": "logs.csv",

        "wandb_project": "Interpret-dlm",

        "input_unit_norm": True,

        "perf_log_freq": 10,
        "checkpoint_freq": 500,

        "n_batches_to_dead": 5,

        "sae_type": "batchtopk",

        "top_k": 16,
        "top_k_aux": 64,
        "aux_penalty": 1 / 32,

        "bandwidth": 0.001,

        # Gated SAE: auxiliary reconstruction loss scale (matches sae_lens default pairing)
        "gated_aux_coeff": 1.0,
    }


def post_init_cfg(cfg):
    cfg["embedding_glob"] = f"data/embeddings/train/layer_{cfg['layer']}/*.pt"
    st = cfg.get("sae_type", "batchtopk").lower()
    if st == "gated":
        cfg["name"] = (
            f"layer{cfg['layer']}_{cfg['dict_size']}_{cfg['sae_type']}_"
            f"l1{cfg['l1_coeff']}_aux{cfg.get('gated_aux_coeff', 1.0)}_{cfg['lr']}"
        )
    else:
        cfg["name"] = (
            f"layer{cfg['layer']}_{cfg['dict_size']}_{cfg['sae_type']}_"
            f"{cfg['top_k']}_{cfg['lr']}"
        )
    cfg["run_dir"] = f"trained_models/{cfg['name']}"
    return cfg