#!/usr/bin/env python
"""
steer_feature.py — causal feature steering on HyenaDNA next-token prediction.

For a target SAE feature (e.g. the best CpG-island or SINE feature), we run HyenaDNA,
capture the residual stream at the SAE's layer, and SCALE that feature's activation by
a set of factors (0 = ablate, 1 = control, 10 = amplify) ALONG ITS DECODER DIRECTION,
then re-run the rest of the model and measure the change in next-token cross-entropy.

Clean steering (default, --steer_mode decoder_add)
--------------------------------------------------
We do NOT replace the residual with the lossy SAE reconstruction. Instead we add only
the *change* in the feature's decoded contribution, leaving the model's true computation
(and SAE reconstruction error) untouched, isolating the causal role of this one feature:

  a_f      = feature f activation (post-threshold), per token   (from the SAE encoder)
  dir_raw  = W_dec[f] * x_std    (decoder direction mapped to raw residual scale,
                                   x_std from the SAE's per-token input normalisation)
  delta[t] = (factor - 1) * a_f[t] * dir_raw[t]
  patched  = hidden + delta            # factor=0 removes f; factor=1 is identity; 10 amplifies

Because a_f = 0 where the feature is inactive, steering only perturbs the tokens where
the feature fires — i.e. (for a good feature) the concept positions.

We report mean CE at three position sets, vs the unpatched baseline:
  - all predicted positions
  - feature-active positions  (a_f > 0)
  - concept-annotated positions  (token inside the concept BED interval)

A causally-used concept feature should INCREASE CE at concept/active positions when
ablated (factor 0), with little change at all-positions (the effect is local).

Negative baseline (--control_feature)
-------------------------------------
To show the CE change is SPECIFIC to the concept feature and not just an artefact of
perturbing the residual at all, we also steer a matched control feature: one that is
concept-agnostic (|MCC| ~ 0 for this concept) but whose overall firing rate matches the
target's, so the steering delta has comparable magnitude. With --control_feature auto
(default) it is chosen automatically from the concept's all_features.csv; pass an int to
fix it, or "off" to skip. A good result: the control's dCE stays near zero while the
target's grows — the effect tracks the feature's meaning, not the size of the poke.

Example
-------
  python main/steer_feature.py \
    --targets cpg_cpg_islands.hg38 repeats_SINE \
    --results_root results --bed_dir all_annotations \
    --test_bed data/preprocessed/test.sub.bed \
    --genome_path data/raw/GRCh38.primary_assembly.genome.fa \
    --hyenadna_checkpoint_path LongSafari/hyenadna-large-1m-seqlen-hf \
    --factors 0 1 10 --n_seq 100 --seq_len 512 --device cuda \
    --out_dir results/analysis/steering
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(__file__))

from utils.gpu_setup import configure_cuda_performance, resolve_device_str  # noqa: E402
from evaluate_sae import (  # noqa: E402
    load_hyenadna_model, run_hyenadna_with_patch, read_bed, fetch_batch,
)
from concept_feature_analysis import (  # noqa: E402
    load_sae, BEDIndex, restore_cfg_types,
)
from pyfaidx import Fasta  # noqa: E402


def per_position_ce(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Next-token CE per position (predict t+1 from t). Returns [L-1] on CPU float."""
    pred = logits[:-1]
    tgt = tokens[1:].to(pred.device)
    return F.cross_entropy(pred, tgt, reduction="none").detach().float().cpu()


# ---------------------------------------------------------------------------
# Target resolution: concept -> (layer, tag, step, feature_idx, checkpoint, cfg)
# ---------------------------------------------------------------------------

def resolve_target(concept_spec: str, results_root: str, best_epochs: dict,
                   sae_subdir: str) -> dict:
    """concept_spec = 'concept' | 'concept:layer' | 'concept:layer:feature_idx'."""
    parts = concept_spec.split(":")
    concept = parts[0]
    layer = int(parts[1]) if len(parts) > 1 and parts[1] else None
    feat = int(parts[2]) if len(parts) > 2 and parts[2] else None

    if layer is None:
        # best layer for this concept from the SAE null comparison table, else error
        nc = os.path.join(results_root, "analysis", "null_comparison_SAE.csv")
        if os.path.isfile(nc):
            for r in csv.DictReader(open(nc, newline="")):
                if r["concept"] == concept:
                    layer = int(r["best_layer"]); break
        if layer is None:
            raise SystemExit(f"Could not resolve best layer for '{concept}'. "
                             f"Pass it explicitly as '{concept}:LAYER'.")
    info = best_epochs[f"L{layer}"]
    tag, step = info["tag"], info["best_step"]
    if feat is None:
        tf = os.path.join(results_root, tag, f"step{step}", sae_subdir, concept, "top_features.csv")
        if not os.path.isfile(tf):
            raise SystemExit(f"top_features.csv not found: {tf} (pass feature idx explicitly)")
        feat = int(float(next(csv.DictReader(open(tf, newline="")))["feature_idx"]))
    return {
        "concept": concept, "layer": layer, "tag": tag, "step": step, "feature": feat,
        "checkpoint": os.path.join(_REPO, "trained_models", tag, "checkpoints", f"step_{step}.pt"),
        "cfg": os.path.join(_REPO, "trained_models", tag, "config.json"),
        "layer_idx": layer - 1,   # model hook layer (hidden_states[L] == layers[L-1] output)
    }


def pick_control_feature(target: dict, results_root: str, sae_subdir: str,
                         mcc_max: float = 0.02) -> Optional[int]:
    """Negative-baseline control: a concept-agnostic feature (|MCC| <= mcc_max) whose
    overall firing rate best matches the target feature's, from the concept's
    all_features.csv. Returns None if no suitable feature exists.

    Overall firing rate = (tp + fp) / (n_positive_tokens + n_negative_tokens); both the
    rate and MCC are stored per feature, so this is pure CSV math (no GPU pass)."""
    af = os.path.join(results_root, target["tag"], f"step{target['step']}",
                      sae_subdir, target["concept"], "all_features.csv")
    if not os.path.isfile(af):
        return None
    rows = list(csv.DictReader(open(af, newline="")))

    def rate(r):
        denom = float(r["n_positive_tokens"]) + float(r["n_negative_tokens"])
        return (float(r["tp"]) + float(r["fp"])) / denom if denom else 0.0

    tgt = next((r for r in rows if int(float(r["feature_idx"])) == target["feature"]), None)
    if tgt is None:
        return None
    tgt_rate = rate(tgt)
    cands = []
    for r in rows:
        fi = int(float(r["feature_idx"]))
        if fi == target["feature"]:
            continue
        try:
            if abs(float(r["mcc"])) > mcc_max:
                continue
        except (ValueError, KeyError):
            continue
        fr = rate(r)
        if fr <= 0:
            continue
        cands.append((abs(fr - tgt_rate), fi))
    if not cands:
        return None
    cands.sort()
    return cands[0][1]


def select_windows(concept_bed: BEDIndex, rows, use_chr_positions, n_seq) -> List[int]:
    """Indices of BED windows whose span overlaps the concept (so the feature can fire)."""
    keep = []
    for i, (chrom, start, end) in enumerate(rows):
        # any concept base within [start,end)? cheap midpoint + endpoints probe then full
        probe = np.arange(start, end, 8, dtype=np.int64)  # subsample probe for speed
        if concept_bed.contains_batch(chrom, probe).any():
            keep.append(i)
        if len(keep) >= n_seq:
            break
    return keep


@torch.no_grad()
def steer_one(target: dict, model, tokenizer, sae, genome, rows, concept_bed,
              factors: List[float], device: str, seq_len: int, steer_mode: str,
              feature: Optional[int] = None, role: str = "target") -> List[dict]:
    f = target["feature"] if feature is None else feature
    li = target["layer_idx"]
    w_dec_f = sae.W_dec[f].detach().to(device).float()        # [D]
    # accumulators: factor -> sums
    acc = {k: {"ce_all": 0.0, "n_all": 0, "ce_act": 0.0, "n_act": 0,
               "ce_con": 0.0, "n_con": 0} for k in factors}
    n_seq_used = 0
    total_active = 0

    for (chrom, start, end) in rows:
        seq = fetch_batch(genome, [(chrom, start, end)])[0]
        if len(seq) != seq_len:
            continue
        enc = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
        tokens = enc["input_ids"].to(device)                  # [1, L]
        L = tokens.shape[1]
        if L != seq_len:
            continue

        # baseline pass: capture hidden + logits
        logits0, hidden = run_hyenadna_with_patch(model, tokens, li, patch_tensor=None, device=device)
        ce0 = per_position_ce(logits0, tokens[0])             # [L-1]

        # SAE acts + per-token std (input_unit_norm path)
        h_flat = hidden.squeeze(0).to(device).float()         # [L, D]
        _, acts = sae(h_flat)                                 # [L, dict]
        a_f = acts[:, f].float()                              # [L]
        x_std = h_flat.std(dim=-1, keepdim=True)              # [L, 1]
        active = (a_f > 0)                                    # [L]
        total_active += int(active.sum().item())

        # concept label per token (token i -> base start+i for 1bp/token tokenisation)
        pos = np.arange(start, start + L, dtype=np.int64)
        con = torch.from_numpy(concept_bed.contains_batch(chrom, pos)).to(device)  # [L] bool

        # masks aligned to CE positions (predict t+1 from t) -> use [:-1]
        m_act = active[:-1].cpu()
        m_con = con[:-1].cpu()
        n_seq_used += 1

        for k in factors:
            if k == 1.0:
                ce = ce0                                      # identity control
            else:
                if steer_mode == "decoder_add":
                    delta = ((k - 1.0) * a_f).unsqueeze(-1) * x_std * w_dec_f.unsqueeze(0)  # [L,D]
                    patched = (h_flat + delta).unsqueeze(0)
                elif steer_mode == "full_recon":
                    recon, acts2 = sae(h_flat)
                    acts2 = acts2.clone(); acts2[:, f] = k * a_f
                    patched = (acts2 @ sae.W_dec + sae.b_dec)
                    patched = (patched * x_std + h_flat.mean(dim=-1, keepdim=True)).unsqueeze(0)
                else:
                    raise ValueError(steer_mode)
                logits, _ = run_hyenadna_with_patch(model, tokens, li,
                                                    patch_tensor=patched.to(device), device=device)
                ce = per_position_ce(logits, tokens[0])
            acc[k]["ce_all"] += float(ce.sum()); acc[k]["n_all"] += ce.numel()
            if m_act.any():
                acc[k]["ce_act"] += float(ce[m_act].sum()); acc[k]["n_act"] += int(m_act.sum())
            if m_con.any():
                acc[k]["ce_con"] += float(ce[m_con].sum()); acc[k]["n_con"] += int(m_con.sum())

    # build rows + deltas vs factor 1 (baseline)
    def mean(d, s, n):
        return acc[d][s] / acc[d][n] if acc[d][n] else float("nan")
    base = {s: mean(1.0, f"ce_{s}", f"n_{s}") for s in ("all", "act", "con")}
    out = []
    for k in factors:
        row = {
            "concept": target["concept"], "layer": target["layer"], "feature_idx": f,
            "role": role, "steer_mode": steer_mode, "factor": k, "n_seq": n_seq_used,
            "n_active_tokens": total_active,
            "ce_all": mean(k, "ce_all", "n_all"),
            "ce_active": mean(k, "ce_act", "n_act"),
            "ce_concept": mean(k, "ce_con", "n_con"),
        }
        row["dCE_all"] = row["ce_all"] - base["all"]
        row["dCE_active"] = row["ce_active"] - base["act"]
        row["dCE_concept"] = row["ce_concept"] - base["con"]
        out.append(row)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="+",
                   default=["cpg_cpg_islands.hg38", "repeats_SINE"],
                   help="concept | concept:layer | concept:layer:feature_idx")
    p.add_argument("--results_root", default=os.path.join(_REPO, "results"))
    p.add_argument("--best_epochs", default=os.path.join(_REPO, "results", "best_epochs.json"))
    p.add_argument("--sae_subdir", default="feature_concept_analysis_mcc")
    p.add_argument("--bed_dir", default=os.path.join(_REPO, "all_annotations"))
    p.add_argument("--test_bed", required=True, help="BED of fixed-length test windows")
    p.add_argument("--genome_path", required=True)
    p.add_argument("--hyenadna_checkpoint_path", default="LongSafari/hyenadna-large-1m-seqlen-hf")
    p.add_argument("--factors", nargs="+", type=float, default=[0.0, 1.0, 10.0])
    p.add_argument("--n_seq", type=int, default=100)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--steer_mode", choices=["decoder_add", "full_recon"], default="decoder_add")
    p.add_argument("--control_feature", default="auto",
                   help="Negative baseline: 'auto' (matched-magnitude, concept-agnostic), "
                        "an int feature idx, or 'off'")
    p.add_argument("--control_mcc_max", type=float, default=0.02,
                   help="Max |MCC| for an auto-selected control feature (concept-agnostic)")
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default=os.path.join(_REPO, "results", "analysis", "steering"))
    return p.parse_args()


def main():
    a = parse_args()
    a.device = resolve_device_str(a.device)
    configure_cuda_performance()
    os.makedirs(a.out_dir, exist_ok=True)
    best = json.load(open(a.best_epochs))

    print("Loading HyenaDNA ...")
    model, tok = load_hyenadna_model(a.hyenadna_checkpoint_path, a.device)
    genome = Fasta(a.genome_path, as_raw=True, sequence_always_upper=True)
    rows_all = read_bed(a.test_bed, a.seq_len)

    all_rows: List[dict] = []
    saes = {}  # cache SAE per (tag,step)
    for spec in a.targets:
        t = resolve_target(spec, a.results_root, best, a.sae_subdir)
        print(f"\n=== TARGET {t['concept']}  L{t['layer']} feat {t['feature']} "
              f"(hook layers[{t['layer_idx']}], ckpt step {t['step']}) ===")
        key = (t["tag"], t["step"])
        if key not in saes:
            with open(t["cfg"]) as fh:
                cfg = restore_cfg_types(json.load(fh)); cfg["device"] = a.device
            saes[key] = load_sae(cfg, t["checkpoint"], a.device)
        sae = saes[key]

        bed = BEDIndex(os.path.join(a.bed_dir, f"{t['concept']}.bed"))
        keep = select_windows(bed, rows_all, True, a.n_seq)
        if not keep:
            print(f"  [skip] no test windows overlap {t['concept']}")
            continue
        rows = [rows_all[i] for i in keep]
        print(f"  {len(rows)} windows overlap the concept; steering factors {a.factors}")
        res = steer_one(t, model, tok, sae, genome, rows, bed,
                        a.factors, a.device, a.seq_len, a.steer_mode)
        for r in res:
            print(f"   factor {r['factor']:>5}: CE_all={r['ce_all']:.4f} (d{r['dCE_all']:+.4f})  "
                  f"CE_active={r['ce_active']:.4f} (d{r['dCE_active']:+.4f})  "
                  f"CE_concept={r['ce_concept']:.4f} (d{r['dCE_concept']:+.4f})")
        all_rows.extend(res)

        # negative baseline: matched-magnitude, concept-agnostic control feature
        cf = None
        if str(a.control_feature).lower() not in ("off", "none", ""):
            if str(a.control_feature).lower() == "auto":
                cf = pick_control_feature(t, a.results_root, a.sae_subdir, a.control_mcc_max)
            else:
                cf = int(a.control_feature)
        if cf is None:
            if str(a.control_feature).lower() not in ("off", "none", ""):
                print("  [control] no matched concept-agnostic feature found; skipping baseline")
        elif cf == t["feature"]:
            print("  [control] matched control == target feature; skipping baseline")
        else:
            print(f"  [control] negative baseline: control feat {cf} "
                  f"(concept-agnostic, |MCC|<={a.control_mcc_max}, rate-matched)")
            cres = steer_one(t, model, tok, sae, genome, rows, bed,
                             a.factors, a.device, a.seq_len, a.steer_mode,
                             feature=cf, role="control")
            for r in cres:
                print(f"   factor {r['factor']:>5}: CE_active={r['ce_active']:.4f} "
                      f"(d{r['dCE_active']:+.4f})  CE_concept={r['ce_concept']:.4f} "
                      f"(d{r['dCE_concept']:+.4f})")
            all_rows.extend(cres)

    # write CSV
    hdr = ["concept", "layer", "feature_idx", "role", "steer_mode", "factor", "n_seq",
           "n_active_tokens", "ce_all", "ce_active", "ce_concept",
           "dCE_all", "dCE_active", "dCE_concept"]
    out_csv = os.path.join(a.out_dir, "steering_results.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in hdr})
    print(f"\nWrote {out_csv}")

    # figure: CE vs factor per target, at active/concept/all positions
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = sorted({(r["concept"], r["layer"]) for r in all_rows})
        series = [("ce_active", "active", "tab:red"),
                  ("ce_concept", "concept", "tab:purple"),
                  ("ce_all", "all", "tab:blue")]
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4),
                                 squeeze=False)
        for j, (c, L) in enumerate(panels):
            ax = axes[0][j]
            fx = None
            feats = {}
            for role, ls, mk in [("target", "-", "o"), ("control", "--", "x")]:
                rr = sorted([r for r in all_rows if r["concept"] == c and r["layer"] == L
                             and r.get("role", "target") == role], key=lambda x: x["factor"])
                if not rr:
                    continue
                feats[role] = rr[0]["feature_idx"]
                fx = [r["factor"] for r in rr]
                for col, lab, color in series:
                    ax.plot(range(len(fx)), [r[col] for r in rr], marker=mk, ls=ls,
                            color=color, alpha=0.6 if role == "control" else 1.0,
                            label=f"{lab} ({role})")
            if fx is None:
                ax.axis("off"); continue
            ax.set_xticks(range(len(fx))); ax.set_xticklabels([f"{v:g}x" for v in fx])
            ttl = f"{c}\nL{L} feat {feats.get('target', '?')}"
            if "control" in feats:
                ttl += f"  (ctrl {feats['control']})"
            ax.set_title(ttl, fontsize=9)
            ax.set_xlabel("steering factor"); ax.set_ylabel("mean next-token CE")
            ax.grid(ls=":", alpha=0.5)
            if j == 0:
                ax.legend(fontsize=7, ncol=2)
        fig.suptitle("Causal feature steering — next-token CE vs activation scaling "
                     "(solid = target, dashed = control)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_png = os.path.join(a.out_dir, "steering_ce.png")
        fig.savefig(out_png, dpi=150)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
