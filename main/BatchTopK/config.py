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

        # Gated SAE (arXiv:2404.16014 Appendix D / G): aux matches Eq. 8 (coefficient 1).
        "gated_aux_coeff": 1.0,
        # Paper: Adam beta1=0, beta2=0.999; lr=3e-4 for gated (vs 1e-3 baseline in paper).
        "gated_use_paper_optimizer": True,
        # Resampling (Appendix D): Bricken-style; interval not fixed in paper — default 25k steps.
        "gated_resample_steps": 25_000,
        "gated_resample_dead_batches": 200,
        "gated_resample_warmup_steps": 1_000,
        # Bricken neuron resampling: sample inputs ∝ per-example squared recon error.
        "gated_resample_loss_weighted": True,
        # Scale new encoder column to frac * mean(||W_enc[:, alive]||_2); decoder row stays unit.
        "gated_resample_enc_norm_frac": 0.2,
    }


def post_init_cfg(cfg):
    cfg["embedding_glob"] = f"data/embeddings/train/layer_{cfg['layer']}/*.pt"
    st = cfg.get("sae_type", "batchtopk").lower()
    if st == "gated" and cfg.get("gated_use_paper_optimizer", True):
        cfg["beta1"] = 0.0
        cfg["beta2"] = 0.999
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