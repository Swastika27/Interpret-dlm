"""
run_null_sweep.py — drive concept_feature_null.py across all layers at each layer's
BEST checkpoint (the epoch with the most ALIVE SAE features, i.e. fewest dead),
for both null modes (circular + blockswap), for SAE features AND raw neurons.

Best-epoch selection
--------------------
For each results/<layer_tag>/step<N>/eval_metrics.yaml we read
test.sparsity.dead_features and pick the step minimising it (ties → earliest step).
Rationale: dead features measure dictionary utilisation; later epochs often kill
features at the extreme layers, so the final checkpoint is frequently NOT the best.

What it runs (per layer, per mode)
----------------------------------
  SAE   : main/concept_feature_null.py --candidates_from <best step feature_concept_analysis>
          → results/<tag>/step<best>/concept_feature_null_<mode>/
  neuron: main/concept_feature_null.py --raw_neurons        (epoch-independent)
          → results/<tag>/neuron_concept_null_<mode>/

Use --dry_run to print the plan + exact commands without executing (works anywhere,
no torch/GPU needed). Drop --dry_run on the GPU box (with the embeddings mounted at
--save_dir) to actually run.

Example (GPU box):
  python utils/run_null_sweep.py \
      --save_dir /mnt/disk2/2005027/data/embeddings \
      --bed_dir  all_annotations/ \
      --n_permutations 200 --device cuda --also_plot

  # plan only, anywhere:
  python utils/run_null_sweep.py --dry_run
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))


def read_dead(eval_yaml):
    """Return test.sparsity.dead_features (int) from an eval_metrics.yaml, or None."""
    try:
        import yaml
        with open(eval_yaml) as f:
            d = yaml.safe_load(f)
        return int(d["test"]["sparsity"]["dead_features"])
    except Exception:
        pass
    # fallback: flat scan (the file is simple 2-space-indented yaml)
    try:
        for line in open(eval_yaml):
            s = line.strip()
            if s.startswith("dead_features:"):
                return int(float(s.split(":", 1)[1]))
    except Exception:
        return None
    return None


def find_best_step(layer_dir):
    """(best_step:int, dead:int, {step:dead}) minimising dead_features; ties→earliest."""
    by_step = {}
    for sd in os.listdir(layer_dir):
        m = re.fullmatch(r"step(\d+)", sd)
        if not m:
            continue
        y = os.path.join(layer_dir, sd, "eval_metrics.yaml")
        if os.path.isfile(y):
            d = read_dead(y)
            if d is not None:
                by_step[int(m.group(1))] = d
    if not by_step:
        return None, None, {}
    best = min(by_step, key=lambda s: (by_step[s], s))
    return best, by_step[best], by_step


def layer_of(tag):
    m = re.search(r"layer(\d+)", tag)
    return int(m.group(1)) if m else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", default=os.path.join(REPO, "results"))
    p.add_argument("--trained_models_root", default=os.path.join(REPO, "trained_models"))
    p.add_argument("--save_dir", default="/mnt/disk2/2005027/data/embeddings",
                   help="Embedding root with <split>/layer_<L>/shard_*.pt")
    p.add_argument("--bed_dir", default=os.path.join(REPO, "all_annotations"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--n_permutations", type=int, default=200)
    p.add_argument("--candidate_top_k", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", nargs="+", default=["circular", "blockswap"],
                   choices=["circular", "blockswap"])
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Restrict to these layer numbers (default: all found)")
    p.add_argument("--skip_neurons", action="store_true")
    p.add_argument("--skip_sae", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print plan + commands, do not run")
    p.add_argument("--also_plot", action="store_true",
                   help="After running, call utils/plot_null_concept.py --sweep per mode")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    tags = []
    for name in sorted(os.listdir(a.results_root)):
        if re.match(r"layer\d+_.*batchtopk", name) and \
                os.path.isdir(os.path.join(a.results_root, name)):
            L = layer_of(name)
            if a.layers and L not in a.layers:
                continue
            tags.append((L, name))
    tags.sort()
    if not tags:
        sys.exit(f"No layer*_batchtopk* dirs under {a.results_root}")

    # ---- select best epoch per layer ----
    plan = []
    print(f"{'layer':5s} {'best_step':>10s} {'epoch':>5s} {'dead':>6s} {'alive':>7s}")
    print("-" * 40)
    best_epochs = {}
    for L, tag in tags:
        ldir = os.path.join(a.results_root, tag)
        best_step, dead, _ = find_best_step(ldir)
        if best_step is None:
            print(f"L{L}: no eval_metrics.yaml — skipping")
            continue
        # dict_size from tag (layer{L}_{dict}_batchtopk_...)
        m = re.search(r"layer\d+_(\d+)_", tag)
        dict_size = int(m.group(1)) if m else 16384
        epoch = best_step // 800000
        alive = dict_size - dead
        best_epochs[f"L{L}"] = {"tag": tag, "best_step": best_step, "epoch": epoch,
                                "dead": dead, "alive": alive}
        print(f"L{L:<4d} {best_step:>10d} {epoch:>5d} {dead:>6d} {alive:>7d}")
        plan.append((L, tag, best_step))

    os.makedirs(a.results_root, exist_ok=True)
    with open(os.path.join(a.results_root, "best_epochs.json"), "w") as f:
        json.dump(best_epochs, f, indent=2)
    print(f"\nWrote {os.path.join(a.results_root, 'best_epochs.json')}")

    # ---- build commands ----
    null_script = os.path.join(REPO, "main", "concept_feature_null.py")
    jobs = []  # (label, cmd-list)
    for L, tag, best_step in plan:
        ckpt = os.path.join(a.trained_models_root, tag, "checkpoints", f"step_{best_step}.pt")
        cfg = os.path.join(a.trained_models_root, tag, "config.json")
        cand = os.path.join(a.results_root, tag, f"step{best_step}", "feature_concept_analysis")
        for mode in a.modes:
            if not a.skip_sae:
                out = os.path.join(a.results_root, tag, f"step{best_step}",
                                   f"concept_feature_null_{mode}")
                jobs.append((f"L{L} SAE {mode}", [
                    a.python, null_script,
                    "--sae_checkpoint", ckpt, "--sae_cfg", cfg,
                    "--save_dir", a.save_dir, "--layer", str(L),
                    "--splits", *a.splits, "--bed_dir", a.bed_dir,
                    "--candidates_from", cand, "--candidate_top_k", str(a.candidate_top_k),
                    "--n_permutations", str(a.n_permutations), "--null_mode", mode,
                    "--batch_size", str(a.batch_size), "--seed", str(a.seed),
                    "--device", a.device, "--out_dir", out,
                ]))
            if not a.skip_neurons:
                out = os.path.join(a.results_root, tag, f"neuron_concept_null_{mode}")
                jobs.append((f"L{L} neuron {mode}", [
                    a.python, null_script, "--raw_neurons", "--sae_cfg", cfg,
                    "--save_dir", a.save_dir, "--layer", str(L),
                    "--splits", *a.splits, "--bed_dir", a.bed_dir,
                    "--n_permutations", str(a.n_permutations), "--null_mode", mode,
                    "--batch_size", str(a.batch_size), "--seed", str(a.seed),
                    "--device", a.device, "--out_dir", out,
                ]))

    print(f"\n{len(jobs)} jobs ({len(plan)} layers × {len(a.modes)} modes × "
          f"{(0 if a.skip_sae else 1)+(0 if a.skip_neurons else 1)} kinds)\n")

    failures = []
    for i, (label, cmd) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {label}")
        if a.dry_run:
            print("    " + " ".join(cmd))
            continue
        if "--sae_checkpoint" in cmd:
            ck = cmd[cmd.index("--sae_checkpoint") + 1]
            if not os.path.isfile(ck):
                print(f"    [WARN] checkpoint missing: {ck} — skipping")
                failures.append((label, "missing checkpoint"))
                continue
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc != 0:
            print(f"    [FAIL] rc={rc}")
            failures.append((label, f"rc={rc}"))
            if not a.continue_on_error:
                sys.exit(f"Aborting after failure in: {label} (use --continue_on_error)")

    if a.also_plot and not a.dry_run:
        plot = os.path.join(HERE, "plot_null_concept.py")
        for mode in a.modes:
            cmd = [a.python, plot, "--sweep", "--results_root", a.results_root,
                   "--mode", mode, "--out",
                   os.path.join(a.results_root, f"null_concept_sweep_{mode}.png")]
            print("PLOT:", " ".join(cmd))
            subprocess.run(cmd, cwd=REPO)

    if failures:
        print(f"\n{len(failures)} job(s) failed/skipped:")
        for lab, why in failures:
            print(f"  - {lab}: {why}")
    elif not a.dry_run:
        print("\nAll jobs completed.")


if __name__ == "__main__":
    main()
