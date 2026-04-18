#%%
import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.gpu_setup import configure_cuda_performance, resolve_device_str

from training import train_sae_wo_model
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE, GatedSAE
from activation_store import StreamingActivationsStore
from config import get_default_cfg, post_init_cfg
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder (SAE)")

    parser.add_argument("--sae_type", type=str, default="batchtopk",
                        choices=["vanilla", "topk", "batchtopk", "jumprelu", "gated"],
                        help="Type of SAE to train")
    parser.add_argument("--layer", type=int, default=6,
                        help="Layer index to train on")
    parser.add_argument("--num_tokens", type=int, default=409600000,
                        help="Total number of tokens to train on")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--embedding_glob", type=str, default="data/embeddings/train/layer_6/*.pt",
                        help="Glob pattern for embedding files")
    parser.add_argument("--aux_penalty", type=float, default=1/32,
                        help="Auxiliary penalty coefficient")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--input_unit_norm", type=lambda x: x.lower() != "false", default=True,
                        help="Whether to unit-normalize inputs (true/false)")
    parser.add_argument("--top_k", type=int, default=32,
                        help="Top-k value for TopK/BatchTopK SAE")
    parser.add_argument("--dict_size", type=int, default=256*64,
                        help="Dictionary (feature) size")
    parser.add_argument("--wandb_project", type=str, default="Interpret-dlm",
                        help="Weights & Biases project name")
    parser.add_argument("--l1_coeff", type=float, default=0.0,
                        help="L1 regularization coefficient (gated: gate-path L1 scale)")
    parser.add_argument("--gated_aux_coeff", type=float, default=1.0,
                        help="Auxiliary reconstruction loss coefficient (gated SAE; paper Eq. 8 uses 1)")
    parser.add_argument("--gated_resample_steps", type=int, default=25_000,
                        help="Run dead-feature resampling every N steps (0=off; paper Appendix D)")
    parser.add_argument("--gated_resample_dead_batches", type=int, default=200,
                        help="Resample features inactive for this many consecutive batches")
    parser.add_argument("--gated_resample_warmup_steps", type=int, default=1_000,
                        help="Cosine LR warm-up steps after resampling (paper: 1000)")
    parser.add_argument("--gated_resample_enc_norm_frac", type=float, default=0.2,
                        help="Bricken-style: new encoder column scale = frac * mean alive column L2 norm")
    parser.add_argument("--gated_resample_uniform", action="store_true",
                        help="Use uniform random batch indices instead of loss-weighted resampling")
    parser.add_argument("--no_gated_paper_optimizer", action="store_true",
                        help="Do not override Adam betas to beta1=0, beta2=0.999 for gated SAE")
    parser.add_argument("--act_size", type=int, default=256,
                        help="Activation size (input dimensionality)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda (default if available), cpu, … — falls back to CPU if CUDA unavailable.",
    )
    parser.add_argument("--bandwidth", type=float, default=0.001,
                        help="Bandwidth parameter for JumpReLU SAE")
    parser.add_argument("--perf_log_freq", type=int, default=100,
                        help="Performance logging frequency (steps)")
    parser.add_argument("--checkpoint_freq", type=int, default=10000,
                        help="Checkpoint saving frequency (steps)")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training: path to training_step_*.pt, or checkpoints/ dir (uses latest_training.pt). "
             "Use the same --name / run_dir as the original job.",
    )
    parser.add_argument("--name", type=str, default=None,
                        help="Model name. Checkpoints and logs will be saved in trained_models/<name> directory")

    return parser.parse_args()


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, "__str__") and "torch" in str(type(obj)):
        return str(obj)
    else:
        return obj


def main():
    args = parse_args()

    args.device = resolve_device_str(args.device)
    configure_cuda_performance()

    cfg = get_default_cfg()

    # Override defaults with CLI arguments
    cfg["sae_type"]        = args.sae_type
    cfg["layer"]           = args.layer
    cfg["num_tokens"]      = args.num_tokens
    cfg["batch_size"]      = args.batch_size
    cfg["embedding_glob"]  = args.embedding_glob
    cfg["aux_penalty"]     = args.aux_penalty
    cfg["lr"]              = args.lr
    cfg["input_unit_norm"] = args.input_unit_norm
    cfg["top_k"]           = args.top_k
    cfg["dict_size"]       = args.dict_size
    cfg["wandb_project"]   = args.wandb_project
    cfg["l1_coeff"]        = args.l1_coeff
    cfg["gated_aux_coeff"] = args.gated_aux_coeff
    cfg["gated_resample_steps"] = args.gated_resample_steps
    cfg["gated_resample_dead_batches"] = args.gated_resample_dead_batches
    cfg["gated_resample_warmup_steps"] = args.gated_resample_warmup_steps
    cfg["gated_resample_enc_norm_frac"] = args.gated_resample_enc_norm_frac
    cfg["gated_resample_loss_weighted"] = not args.gated_resample_uniform
    cfg["gated_use_paper_optimizer"] = not args.no_gated_paper_optimizer
    cfg["act_size"]        = args.act_size
    cfg["device"]          = args.device
    cfg["bandwidth"]       = args.bandwidth
    cfg["perf_log_freq"]   = args.perf_log_freq
    cfg["checkpoint_freq"] = args.checkpoint_freq
    cfg["resume_path"]     = args.resume
    cfg["name"]            = args.name

    SAE_CLASSES = {
        "vanilla":   VanillaSAE,
        "topk":      TopKSAE,
        "batchtopk": BatchTopKSAE,
        "jumprelu":  JumpReLUSAE,
        "gated":     GatedSAE,
    }
    sae = SAE_CLASSES[cfg["sae_type"]](cfg)
    print("Created SAE")

    cfg = post_init_cfg(cfg)

    activations_store = StreamingActivationsStore(cfg)
    print("Starting training")
    train_sae_wo_model(sae, activations_store, cfg)

    cfg_path = f"{cfg['run_dir']}/config.json"
    with open(cfg_path, "w") as f:
        json.dump(make_json_serializable(cfg), f, indent=4)
    print(f"Config saved to {cfg_path}")


if __name__ == "__main__":
    main()