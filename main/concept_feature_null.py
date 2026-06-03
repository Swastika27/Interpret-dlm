"""
concept_feature_null.py — significance testing for concept↔feature associations.

WHY THIS EXISTS
---------------
`concept_feature_analysis.py` reports, for each concept, the single best feature's
F1 (selected over `dict_size` features). Two problems make a bare F1 uninterpretable:

  1. Spatial autocorrelation. Genomic annotations are contiguous runs of positive
     bases, and HyenaDNA features fire in contiguous runs too (long convolutions ⇒
     neighbouring tokens share context). Two autocorrelated binary tracks overlap
     far more than two independent Bernoulli tracks, so the *chance* F1 is well
     above the naïve 0.5/0.667 intuition. A per-token label shuffle DESTROYS this
     structure and therefore badly understates the null — manufacturing "significance".

  2. Selection over the dictionary. Taking the max F1 over 16,384 features inflates
     the reported value even with no true association.

NULL MODELS (both preserve each track's run-length structure)
-------------------------------------------------------------
  --null_mode circular  (default): independently circular-shift each window's concept
      label vector by a uniform random offset in [0, L). Preserves, exactly and per
      window: the feature firing rate, n_pos, n_neg, and the run-length distribution
      of BOTH tracks. Breaks only the *within-window positional alignment* between a
      feature and a concept. This is the standard circular-randomization null used in
      genomics (cf. GAT / regioneR), adapted to the per-window data layout.
      Note: it tests SUB-WINDOW-scale positional specificity. For a concept that is
      constant within 512 bp windows (covers whole windows), there is no within-window
      structure to permute and the test is intentionally conservative there.

  --null_mode blockswap: within each shard, randomly permute which window's label
      block is scored against which window's activations. Preserves each window's
      label run-structure and each feature's activations; breaks the locus
      correspondence. Tests WHETHER A FEATURE FIRES IN THE RIGHT WINDOWS (locus-level),
      complementary to `circular`.

The F1 is computed with the IDENTICAL class-imbalance correction as
`concept_feature_analysis.compute_metrics_from_counts` (negatives scaled to
min(n_neg, n_pos)), so observed and null F1 live on the same scale.

OUTPUTS (in --out_dir)
----------------------
  null_feature_pvalues.csv   one row per (concept, candidate feature):
      observed_f1, observed_precision, observed_recall, null_mean_f1, null_std_f1,
      null_q95_f1, z_score, p_value, p_value_bh, significant_bh, n_positive_tokens
  null_concept_summary.csv   one row per concept:
      best candidate by observed F1, its p / p_bh, and the null distribution of the
      *max over candidates* (null_max_mean_f1, null_max_q95_f1, exceeds_null_max_q95)
      — the selection-aware floor among the features actually tested.
  null_run_meta.json         run configuration for reproducibility.

EXAMPLE
-------
  # SAE features (test the top-10 reported per concept)
  python main/concept_feature_null.py \
      --sae_checkpoint trained_models/layer4_16384_batchtopk_64_0.0003/checkpoints/step_3200000.pt \
      --sae_cfg        trained_models/layer4_16384_batchtopk_64_0.0003/config.json \
      --save_dir       /mnt/disk2/2005027/data/embeddings \
      --layer          4 --splits test \
      --bed_dir        all_annotations/ \
      --candidates_from results/layer4_16384_batchtopk_64_0.0003/step3200000/feature_concept_analysis \
      --candidate_top_k 10 \
      --n_permutations 200 --null_mode circular --seed 42 \
      --out_dir        results/layer4_16384_batchtopk_64_0.0003/step3200000/concept_feature_null

  # Raw-neuron baseline (all 256 neurons are candidates)
  python main/concept_feature_null.py --raw_neurons \
      --sae_cfg ... --save_dir ... --layer 4 --splits test --bed_dir all_annotations/ \
      --n_permutations 200 --out_dir results/.../neuron_concept_null
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from utils.gpu_setup import configure_cuda_performance, resolve_device_str  # noqa: E402
from concept_feature_analysis import (  # noqa: E402
    BEDIndex,
    RawNeuronSAE,
    load_sae,
    get_activations,
    restore_cfg_types,
    _build_token_positions,
    _cofa_collect_shard_plan,
    _cofa_bootstrap_use_chr_from_shard_plan,
)


# ---------------------------------------------------------------------------
# Balanced F1 (vectorised; matches concept_feature_analysis exactly)
# ---------------------------------------------------------------------------

def balanced_f1(
    tp: np.ndarray,        # (..., F)  positive-token firings
    firing: np.ndarray,    # (F,)      total firings (over all tokens)
    n_pos: float,
    n_neg: float,
) -> np.ndarray:
    """
    Returns F1 of shape tp.shape using the same negative-rescaling as
    compute_metrics_from_counts: FP = (firing - tp) * min(n_neg, n_pos)/n_neg.
    """
    n_neg_eff = min(n_neg, n_pos)
    scale = (n_neg_eff / n_neg) if n_neg > 0 else 0.0
    tp = tp.astype(np.float64)
    fp = (firing.astype(np.float64) - tp) * scale
    fn = n_pos - tp
    denom_p = tp + fp
    denom_r = tp + fn
    precision = np.divide(tp, denom_p, out=np.zeros_like(tp), where=denom_p > 0)
    recall = np.divide(tp, denom_r, out=np.zeros_like(tp), where=denom_r > 0)
    denom_f = precision + recall
    f1 = np.divide(2 * precision * recall, denom_f,
                   out=np.zeros_like(precision), where=denom_f > 0)
    return f1, precision, recall


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR adjusted p-values (monotone, clipped to 1)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Candidate-feature selection
# ---------------------------------------------------------------------------

def candidates_from_results(results_dir: str, concept_names: List[str],
                            top_k: int) -> Tuple[List[int], Dict[str, List[int]]]:
    """Union of the top-`top_k` feature_idx per concept from all_features.csv."""
    per_concept: Dict[str, List[int]] = {}
    union: List[int] = []
    seen = set()
    for c in concept_names:
        path = os.path.join(results_dir, c, "all_features.csv")
        if not os.path.isfile(path):
            print(f"  [candidates] no all_features.csv for '{c}' — skipping")
            per_concept[c] = []
            continue
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        idxs = [int(float(r["feature_idx"])) for r in rows[:top_k]]
        per_concept[c] = idxs
        for fi in idxs:
            if fi not in seen:
                seen.add(fi)
                union.append(fi)
    return union, per_concept


# ---------------------------------------------------------------------------
# Streaming accumulation of observed + null TP counts
# ---------------------------------------------------------------------------

def accumulate(
    sae,
    shard_plan: List[Tuple[str, str]],
    bed_indices: List[BEDIndex],
    cand_idx: torch.Tensor,        # (nc,) long, on device
    act_size: int,
    batch_size: int,
    device: str,
    use_chr: bool,
    n_perm: int,
    null_mode: str,
    gen: torch.Generator,
    max_windows: Optional[int],
    preloaded_first: Optional[dict],
    first_path: Optional[str],
) -> dict:
    dev = torch.device(device)
    C = len(bed_indices)
    nc = int(cand_idx.numel())

    firing = torch.zeros(nc, dtype=torch.float64)            # total firings per cand
    n_pos = torch.zeros(C, dtype=torch.float64)
    total_tokens = 0
    tp_obs = torch.zeros(C, nc, dtype=torch.float64)
    tp_null = torch.zeros(n_perm, C, nc, dtype=torch.float64)

    windows_used = 0
    for split, shard_path in tqdm(shard_plan, desc="shards"):
        if not shard_path or not os.path.isfile(shard_path):
            print(f"  [WARN] missing shard {shard_path}")
            continue
        if preloaded_first is not None and first_path and \
                os.path.normpath(shard_path) == os.path.normpath(first_path):
            shard = preloaded_first
            preloaded_first = None
        else:
            shard = torch.load(shard_path, map_location="cpu")
        emb = shard["emb"]
        coords = shard["coords"]
        B, L, D = emb.shape
        assert D == act_size, f"emb dim {D} != act_size {act_size}"

        if max_windows is not None and windows_used + B > max_windows:
            B = max(0, max_windows - windows_used)
            if B == 0:
                break
            emb = emb[:B]
            coords = coords[:B]
        windows_used += B

        # ---- labels [B, L, C] (bool) ----
        chroms, positions = _build_token_positions(coords, L, use_chr)
        labels_np = np.zeros((B * L, C), dtype=bool)
        chroms_arr = np.array(chroms)
        for ch in np.unique(chroms_arr):
            flat = np.where(chroms_arr == ch)[0]
            pos_arr = positions[flat]
            for ci, bed in enumerate(bed_indices):
                hits = bed.contains_batch(ch, pos_arr)
                labels_np[flat[hits], ci] = True
        labels = torch.from_numpy(labels_np).to(dev).view(B, L, C).float()

        # ---- candidate activations [B, L, nc] (0/1 float) ----
        emb_flat = emb.reshape(B * L, D)
        act_cand = torch.empty(B * L, nc, dtype=torch.float32, device=dev)
        for s in range(0, B * L, batch_size):
            e = min(s + batch_size, B * L)
            acts = get_activations(sae, emb_flat[s:e], return_cpu=False)  # [b, F] dev
            act_cand[s:e] = (acts[:, cand_idx] > 0).float()
        act_cand = act_cand.view(B, L, nc)

        # ---- marginals (shift-invariant) ----
        firing += act_cand.sum(dim=(0, 1)).double().cpu()
        n_pos += labels.sum(dim=(0, 1)).double().cpu()
        total_tokens += B * L

        # ---- observed TP: einsum over windows & tokens ----
        tp_obs += torch.einsum("blc,blf->cf", labels, act_cand).double().cpu()

        # ---- null TP ----
        arangeL = torch.arange(L, device=dev)
        for p in range(n_perm):
            if null_mode == "circular":
                shifts = torch.randint(0, L, (B,), generator=gen, device=dev)
                idx = (arangeL.view(1, L) - shifts.view(B, 1)) % L      # [B, L]
                lab_p = torch.gather(labels, 1, idx.unsqueeze(-1).expand(B, L, C))
            elif null_mode == "blockswap":
                perm = torch.randperm(B, generator=gen, device=dev)
                lab_p = labels[perm]
            else:
                raise ValueError(f"unknown null_mode {null_mode}")
            tp_null[p] += torch.einsum("blc,blf->cf", lab_p, act_cand).double().cpu()

        del shard, emb, emb_flat, labels, act_cand
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "firing": firing.numpy(),
        "n_pos": n_pos.numpy(),
        "n_neg": (total_tokens - n_pos.numpy()),
        "tp_obs": tp_obs.numpy(),
        "tp_null": tp_null.numpy(),
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae_checkpoint", default=None,
                   help="SAE .pt (required unless --raw_neurons)")
    p.add_argument("--sae_cfg", required=True)
    p.add_argument("--save_dir", required=True, help="root with <split>/layer_<L>/shard_*.pt")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--bed_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--raw_neurons", action="store_true",
                   help="Test raw model neurons (all act_size dims) instead of SAE features")
    # candidate selection (mutually informative; precedence: indices > from > all)
    p.add_argument("--feature_indices", default=None,
                   help="Comma-separated explicit candidate feature indices")
    p.add_argument("--candidates_from", default=None,
                   help="A feature_concept_analysis dir; use top-K features per concept")
    p.add_argument("--candidate_top_k", type=int, default=10)
    p.add_argument("--all_features", action="store_true",
                   help="Test ALL dict features (slow; use --max_windows and small --n_permutations)")
    # null configuration
    p.add_argument("--n_permutations", type=int, default=200)
    p.add_argument("--null_mode", choices=["circular", "blockswap"], default="circular")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_windows", type=int, default=None,
                   help="Cap number of windows processed (for speed / all-features mode)")
    p.add_argument("--alpha", type=float, default=0.05, help="BH-FDR significance level")
    args = p.parse_args()
    if not args.raw_neurons and not args.sae_checkpoint:
        p.error("--sae_checkpoint required unless --raw_neurons")
    return args


def main():
    args = parse_args()
    args.device = resolve_device_str(args.device)
    configure_cuda_performance()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.sae_cfg) as fh:
        cfg = json.load(fh)
    cfg = restore_cfg_types(cfg)
    cfg["device"] = args.device
    act_size = cfg["act_size"]
    dict_size = act_size if args.raw_neurons else cfg["dict_size"]

    # ---- concepts ----
    bed_paths = sorted(glob.glob(os.path.join(args.bed_dir, "*.bed")))
    if not bed_paths:
        raise FileNotFoundError(f"No .bed files in {args.bed_dir}")
    shard_plan = _cofa_collect_shard_plan(args.save_dir, args.splits, args.layer)
    if not shard_plan:
        raise FileNotFoundError(
            f"No shards under {args.save_dir}/<split>/layer_{args.layer}")
    use_chr, first_path, first_shard = _cofa_bootstrap_use_chr_from_shard_plan(shard_plan)
    bed_indices = [BEDIndex(p, use_chr=use_chr) for p in bed_paths]
    concept_names = [b.name for b in bed_indices]
    C = len(bed_indices)
    print(f"\n{C} concepts; use_chr={use_chr}; {len(shard_plan)} shards; "
          f"null_mode={args.null_mode}; n_perm={args.n_permutations}")

    # ---- candidates ----
    per_concept_cands: Dict[str, List[int]] = {}
    if args.feature_indices:
        cand = sorted({int(x) for x in args.feature_indices.split(",")})
    elif args.raw_neurons:
        cand = list(range(act_size))
    elif args.candidates_from:
        cand, per_concept_cands = candidates_from_results(
            args.candidates_from, concept_names, args.candidate_top_k)
    elif args.all_features:
        print("  [WARN] --all_features: testing every dict feature. "
              "This is slow; prefer --max_windows and small --n_permutations.")
        cand = list(range(dict_size))
    else:
        raise SystemExit(
            "Specify candidates: --candidates_from DIR (recommended), "
            "--feature_indices, --raw_neurons, or --all_features.")
    if not cand:
        raise SystemExit("Empty candidate set.")
    cand = [c for c in cand if 0 <= c < dict_size]
    print(f"  {len(cand)} candidate features")
    cand_t = torch.tensor(cand, dtype=torch.long, device=args.device)

    # ---- SAE ----
    if args.raw_neurons:
        print("Raw-neuron mode (ReLU passthrough).")
        sae = RawNeuronSAE(act_size).eval().to(args.device)
    else:
        sae = load_sae(cfg, args.sae_checkpoint, args.device)

    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)

    boot_path = first_path
    if first_shard is not None and (args.max_windows is None or args.max_windows > 0):
        preloaded = first_shard
    else:
        preloaded = None

    acc = accumulate(
        sae=sae, shard_plan=shard_plan, bed_indices=bed_indices, cand_idx=cand_t,
        act_size=act_size, batch_size=args.batch_size, device=args.device,
        use_chr=use_chr, n_perm=args.n_permutations, null_mode=args.null_mode,
        gen=gen, max_windows=args.max_windows,
        preloaded_first=preloaded, first_path=boot_path,
    )

    firing = acc["firing"]                 # (nc,)
    n_pos_all = acc["n_pos"]               # (C,)
    n_neg_all = acc["n_neg"]               # (C,)
    tp_obs = acc["tp_obs"]                 # (C, nc)
    tp_null = acc["tp_null"]               # (P, C, nc)
    P = tp_null.shape[0]
    print(f"\nProcessed {acc['total_tokens']:,} tokens.")

    # ---- per (concept, feature) significance ----
    cand_arr = np.array(cand)
    pair_rows: List[dict] = []
    summary_rows: List[dict] = []
    for ci, cname in enumerate(concept_names):
        n_pos = float(n_pos_all[ci])
        n_neg = float(n_neg_all[ci])
        if n_pos == 0:
            print(f"  [skip] {cname}: no positive tokens")
            continue
        obs_f1, obs_prec, obs_rec = balanced_f1(tp_obs[ci], firing, n_pos, n_neg)  # (nc,)
        null_f1 = np.empty((P, len(cand)), dtype=np.float64)
        for p in range(P):
            null_f1[p], _, _ = balanced_f1(tp_null[p, ci], firing, n_pos, n_neg)
        null_mean = null_f1.mean(axis=0)
        null_std = null_f1.std(axis=0)
        null_q95 = np.quantile(null_f1, 0.95, axis=0)
        # one-sided empirical p (with +1 smoothing) that null F1 >= observed
        ge = (null_f1 >= obs_f1[None, :]).sum(axis=0)
        pval = (1.0 + ge) / (P + 1.0)
        z = np.divide(obs_f1 - null_mean, null_std,
                      out=np.zeros_like(obs_f1), where=null_std > 1e-12)

        # restrict reported rows to this concept's own candidates if available
        report_idx = list(range(len(cand)))
        if per_concept_cands.get(cname):
            keep = set(per_concept_cands[cname])
            report_idx = [j for j in range(len(cand)) if cand[j] in keep]

        for j in report_idx:
            pair_rows.append({
                "concept": cname,
                "feature_idx": int(cand_arr[j]),
                "observed_f1": float(obs_f1[j]),
                "observed_precision": float(obs_prec[j]),
                "observed_recall": float(obs_rec[j]),
                "null_mean_f1": float(null_mean[j]),
                "null_std_f1": float(null_std[j]),
                "null_q95_f1": float(null_q95[j]),
                "z_score": float(z[j]),
                "p_value": float(pval[j]),
                "n_positive_tokens": int(n_pos),
                "prevalence": float(n_pos / (n_pos + n_neg)),
            })

        # selection-aware: null distribution of the MAX over candidates
        null_max = null_f1.max(axis=1)                       # (P,)
        best_j = int(np.argmax(obs_f1))
        best_p = (1.0 + (null_f1[:, best_j] >= obs_f1[best_j]).sum()) / (P + 1.0)
        summary_rows.append({
            "concept": cname,
            "best_feature_idx": int(cand_arr[best_j]),
            "observed_best_f1": float(obs_f1[best_j]),
            "p_value": float(best_p),
            "null_max_mean_f1": float(null_max.mean()),
            "null_max_q95_f1": float(np.quantile(null_max, 0.95)),
            "exceeds_null_max_q95": bool(obs_f1[best_j] > np.quantile(null_max, 0.95)),
            "n_candidates": len(cand),
            "n_positive_tokens": int(n_pos),
            "prevalence": float(n_pos / (n_pos + n_neg)),
        })

    # ---- BH-FDR across all reported pairs ----
    if pair_rows:
        pvals = np.array([r["p_value"] for r in pair_rows])
        p_bh = benjamini_hochberg(pvals)
        for r, pb in zip(pair_rows, p_bh):
            r["p_value_bh"] = float(pb)
            r["significant_bh"] = bool(pb < args.alpha)
    # also BH the per-concept best p
    if summary_rows:
        sp = benjamini_hochberg(np.array([r["p_value"] for r in summary_rows]))
        for r, pb in zip(summary_rows, sp):
            r["p_value_bh"] = float(pb)
            r["significant_bh"] = bool(pb < args.alpha)

    # ---- write ----
    pair_hdr = ["concept", "feature_idx", "observed_f1", "observed_precision",
                "observed_recall", "null_mean_f1", "null_std_f1", "null_q95_f1",
                "z_score", "p_value", "p_value_bh", "significant_bh",
                "n_positive_tokens", "prevalence"]
    with open(os.path.join(args.out_dir, "null_feature_pvalues.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pair_hdr)
        w.writeheader()
        for r in sorted(pair_rows, key=lambda x: (x["concept"], -x["observed_f1"])):
            w.writerow({k: r.get(k, "") for k in pair_hdr})

    sum_hdr = ["concept", "best_feature_idx", "observed_best_f1", "p_value",
               "p_value_bh", "significant_bh", "null_max_mean_f1", "null_max_q95_f1",
               "exceeds_null_max_q95", "n_candidates", "n_positive_tokens", "prevalence"]
    with open(os.path.join(args.out_dir, "null_concept_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sum_hdr)
        w.writeheader()
        for r in sorted(summary_rows, key=lambda x: -x["observed_best_f1"]):
            w.writerow({k: r.get(k, "") for k in sum_hdr})

    with open(os.path.join(args.out_dir, "null_run_meta.json"), "w") as fh:
        json.dump({
            "sae_checkpoint": args.sae_checkpoint,
            "raw_neurons": args.raw_neurons,
            "layer": args.layer, "splits": args.splits,
            "null_mode": args.null_mode, "n_permutations": P, "seed": args.seed,
            "n_candidates": len(cand), "candidate_top_k": args.candidate_top_k,
            "candidates_from": args.candidates_from,
            "max_windows": args.max_windows, "total_tokens": acc["total_tokens"],
            "alpha": args.alpha, "concepts": concept_names,
        }, fh, indent=2)

    # ---- console report ----
    print(f"\n{'concept':28s} {'best_f':>4s} {'obs_F1':>7s} {'nullμ':>7s} "
          f"{'null_q95':>8s} {'p_bh':>7s}  sig")
    print("-" * 78)
    for r in sorted(summary_rows, key=lambda x: -x["observed_best_f1"]):
        print(f"{r['concept'][:28]:28s} {r['best_feature_idx']:>4d} "
              f"{r['observed_best_f1']:>7.3f} {r['null_max_mean_f1']:>7.3f} "
              f"{r['null_max_q95_f1']:>8.3f} {r['p_value_bh']:>7.4f}  "
              f"{'YES' if r['significant_bh'] else 'no'}")
    n_sig = sum(r["significant_bh"] for r in summary_rows)
    print(f"\n{n_sig}/{len(summary_rows)} concepts have a best feature significant "
          f"above the autocorrelation-preserving null (BH-FDR < {args.alpha}).")
    print(f"Wrote: {args.out_dir}/null_feature_pvalues.csv, null_concept_summary.csv")


if __name__ == "__main__":
    main()
