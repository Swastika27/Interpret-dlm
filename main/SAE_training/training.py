import torch
import tqdm
from logs import init_wandb, log_wandb, log_model_performance, save_checkpoint
import os
import csv
from typing import Dict, List, Optional, Any

from gated_resampling import (
    apply_gated_lr,
    resample_dead_gated_features,
)

def write_csv_row(path: str, header: List[str], row: Dict[str, float]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def _theta_for_checkpoint(sae, cfg) -> Optional[float]:
    """BatchTopK tracks a JumpReLU threshold; other SAE types omit theta."""
    if cfg.get("sae_type", "").lower() != "batchtopk":
        cfg.pop("theta", None)
        return None
    if not hasattr(sae, "theta_count") or sae.theta_count == 0:
        return None
    theta = (sae.theta_sum / sae.theta_count).item()
    cfg["theta"] = theta
    return theta


def train_sae_wo_model(sae, activation_store, cfg):
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    base_lr = float(cfg["lr"])
    cfg["base_lr"] = base_lr
    optimizer = torch.optim.Adam(
        sae.parameters(), lr=base_lr, betas=(cfg["beta1"], cfg["beta2"])
    )
    pbar = tqdm.trange(num_batches)

    gated_state: Dict[str, Any] = {"in_warmup": False, "warmup_step": 0}
    is_gated = cfg.get("sae_type", "").lower() == "gated"

    # wandb_run = init_wandb(cfg)
    os.makedirs(cfg["run_dir"], exist_ok=True)
    os.chmod(cfg["run_dir"], 0o777)
    log_path = os.path.join(cfg["run_dir"], cfg["log_path"])
    print(f"Logging to {log_path}")
    header = [
        "step",
        "total_loss",
        "l0_norm",
        "l2_loss",
        "l1_loss",
        "l1_norm",
        "aux_loss",
        "num_dead_features",
    ]
    
    for i in pbar:
        batch = activation_store.next_batch()
        if is_gated:
            apply_gated_lr(optimizer, cfg, gated_state, base_lr)

        sae_output = sae(batch)
        loss = sae_output["loss"]
        # log_wandb(sae_output, i, wandb_run)
        if (i+1) % cfg["perf_log_freq"]  == 0:
            row = {"step": i+1, 
                   "total_loss": f"{loss.item()}", 
                   "l0_norm": f"{sae_output['l0_norm']}", 
                   "l2_loss": f"{sae_output['l2_loss']}", 
                   "l1_loss": f"{sae_output['l1_loss']}", 
                   "l1_norm": f"{sae_output['l1_norm']}",
                   "aux_loss": f"{sae_output['aux_loss']}",
                   "num_dead_features": f"{sae_output['num_dead_features']}",
                }
            write_csv_row(log_path, header, row)

        if (i+1) % cfg["checkpoint_freq"] == 0:
            theta = _theta_for_checkpoint(sae, cfg)
            save_checkpoint(sae, cfg, theta, i+1)

        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

        if is_gated:
            interval = int(cfg.get("gated_resample_steps", 25_000))
            just_reset_warmup = False
            if interval > 0 and i > 0 and i % interval == 0:
                n_res = resample_dead_gated_features(
                    sae, batch, cfg, gated_state=gated_state
                )
                just_reset_warmup = n_res > 0
                if n_res > 0:
                    print(
                        f"  [gated] Resampled {n_res} dead features; LR warm-up "
                        f"({cfg.get('gated_resample_warmup_steps', 1000)} steps)"
                    )
            if gated_state.get("in_warmup", False) and not just_reset_warmup:
                gated_state["warmup_step"] = int(gated_state.get("warmup_step", 0)) + 1
                ws_lim = int(cfg.get("gated_resample_warmup_steps", 1000))
                if gated_state["warmup_step"] >= ws_lim:
                    gated_state["in_warmup"] = False
                    gated_state["warmup_step"] = 0

    theta = _theta_for_checkpoint(sae, cfg)
    save_checkpoint(sae, cfg, theta, i)


def train_sae(sae, activation_store, model, cfg):
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)
    
    for i in pbar:
        batch = activation_store.next_batch()
        sae_output = sae(batch)
        log_wandb(sae_output, i, wandb_run)
        if i % cfg["perf_log_freq"]  == 0:
            log_model_performance(wandb_run, i, model, activation_store, sae)

        if i % cfg["checkpoint_freq"] == 0:
            save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"]
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(wandb_run, sae, cfg, i)
    

def train_sae_group(saes, activation_store, model, cfgs):
    num_batches = cfgs[0]["num_tokens"] // cfgs[0]["batch_size"]
    optimizers = [torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])) for sae, cfg in zip(saes, cfgs)]
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfgs[0])

    batch_tokens = activation_store.get_batch_tokens()

    for i in pbar:
        batch = activation_store.next_batch()
        counter = 0
        for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
            sae_output = sae(batch)
            loss = sae_output["loss"]
            log_wandb(sae_output, i, wandb_run, index=counter)
            if i % cfg["perf_log_freq"]  == 0:
                log_model_performance(wandb_run, i, model, activation_store, sae, index=counter, batch_tokens=batch_tokens)

            if i % cfg["checkpoint_freq"] == 0:
                save_checkpoint(wandb_run, sae, cfg, i)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
            counter += 1
   
    for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
        save_checkpoint(wandb_run, sae, cfg, i)
