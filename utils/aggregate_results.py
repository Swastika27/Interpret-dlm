"""Ad-hoc aggregation of results/ across layers and epochs. Standalone, read-only."""
import csv
import glob
import json
import os
import re
from statistics import mean, median

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS = os.path.abspath(RESULTS)

# Concepts to drop as degenerate (prevalence ~1 -> trivial F1); keep a clean set too.
# DEGENERATE = {"clinvar_GRCh38_benign", "clinvar_GRCh38_pathogenic", "encode_CTCF-only"}
DEGENATE = {}

def parse_simple_yaml(path):
    """Minimal parser for the flat 2-space-indented eval_metrics.yaml produced here."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        pass
    root = {}
    stack = [(-1, root)]
    with open(path) as f:
        for raw in f:
            if not raw.strip() or raw.strip().startswith("#"):
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            key, _, val = raw.strip().partition(":")
            val = val.strip()
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1]
            if val == "":
                d = {}
                parent[key] = d
                stack.append((indent, d))
            elif val == "[]":
                parent[key] = []
            else:
                try:
                    parent[key] = float(val) if ("." in val or "e" in val.lower()) else int(val)
                except ValueError:
                    parent[key] = val
    return root


def step_num(s):
    m = re.search(r"step(\d+)", s)
    return int(m.group(1)) if m else 0


def layer_num(d):
    m = re.search(r"layer(\d+)", d)
    return int(m.group(1)) if m else 0


def read_concept_summary(path):
    """Return dict concept->f1 (and precision, recall)."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["concept"]] = {
                    "f1": float(row["f1"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall_tpr"]),
                }
            except (KeyError, ValueError):
                continue
    return out


def mean_over_concepts(cdict, key, exclude_degenerate=True):
    vals = [v[key] for c, v in cdict.items()
            if not (exclude_degenerate and c in DEGENERATE)]
    return mean(vals) if vals else float("nan")


rows = []
neuron_by_layer = {}

for ldir in sorted(glob.glob(os.path.join(RESULTS, "layer*_batchtopk_*")), key=layer_num):
    L = layer_num(ldir)
    neuron_csv = os.path.join(ldir, "neuron_concept_analysis", "summary.csv")
    neuron_by_layer[L] = read_concept_summary(neuron_csv)
    for sdir in sorted(glob.glob(os.path.join(ldir, "step*")), key=step_num):
        step = step_num(sdir)
        epoch = step // 800000  # checkpoint_freq == batches_per_epoch
        ev = parse_simple_yaml(os.path.join(sdir, "eval_metrics.yaml"))
        t = ev.get("test", {})
        recon = t.get("reconstruction", {})
        spars = t.get("sparsity", {})
        fid = t.get("fidelity", {})
        jpath = glob.glob(os.path.join(sdir, "epoch_diagnostics", "step_*", "summary.json"))
        diag = {}
        if jpath:
            with open(jpath[0]) as f:
                diag = json.load(f)
        cdict = read_concept_summary(
            os.path.join(sdir, "feature_concept_analysis", "summary.csv"))
        rows.append({
            "layer": L,
            "epoch": epoch,
            "step": step,
            "var_exp": recon.get("variance_explained", float("nan")),
            "mse": recon.get("mse", float("nan")),
            "L0": spars.get("l0_sparsity", float("nan")),
            "dead": spars.get("dead_features", float("nan")),
            "mean_freq": spars.get("mean_feature_activation_freq", float("nan")),
            "fidelity": fid.get("pct_loss_recovered", float("nan")) if isinstance(fid, dict) else float("nan"),
            "n_dense": len(diag.get("dense_features_over_threshold", [])),
            "poly_full": diag.get("polysemanticity_proxy_full", float("nan")),
            "poly_sparse": diag.get("polysemanticity_proxy_sparse_only", float("nan")),
            "feat_f1": mean_over_concepts(cdict, "f1"),
            "feat_prec": mean_over_concepts(cdict, "precision"),
            "feat_recall": mean_over_concepts(cdict, "recall"),
            "_cdict": cdict,
        })

def fmt(v, p=4):
    try:
        return f"{v:.{p}f}"
    except (TypeError, ValueError):
        return str(v)

# ---- TABLE 1: final epoch (step 8000000) cross-layer ----
print("\n=== TABLE 1: FINAL EPOCH (epoch 10) — cross-layer comparison ===")
hdr = ["layer", "var_exp", "mse", "L0", "dead", "fidelity%", "n_dense", "poly_full", "poly_sparse", "feat_F1", "feat_prec"]
print("\t".join(hdr))
finals = [r for r in rows if r["step"] == 8000000]
for r in sorted(finals, key=lambda x: x["layer"]):
    print("\t".join([str(r["layer"]), fmt(r["var_exp"]), fmt(r["mse"],5), fmt(r["L0"],2),
                     str(r["dead"]), fmt(r["fidelity"],2), str(r["n_dense"]),
                     fmt(r["poly_full"],3), fmt(r["poly_sparse"],4),
                     fmt(r["feat_f1"],3), fmt(r["feat_prec"],3)]))

# ---- TABLE 2: SAE feature F1 vs raw-neuron F1 (final epoch), per layer ----
print("\n=== TABLE 2: SAE feature vs raw-neuron mean F1 (clean concept set, final epoch) ===")
print("layer\tfeat_F1\tneuron_F1\tdelta\tfeat_prec\tneuron_prec")
for r in sorted(finals, key=lambda x: x["layer"]):
    nd = neuron_by_layer.get(r["layer"], {})
    nf1 = mean_over_concepts(nd, "f1")
    npr = mean_over_concepts(nd, "precision")
    print("\t".join([str(r["layer"]), fmt(r["feat_f1"],3), fmt(nf1,3),
                     fmt(r["feat_f1"]-nf1,3), fmt(r["feat_prec"],3), fmt(npr,3)]))

# ---- TABLE 3: per-epoch trajectory for each layer (key metrics) ----
for metric, lbl, p in [("var_exp","Variance explained",4),
                         ("fidelity","Fidelity % loss recovered",2),
                         ("poly_sparse","Polysemanticity (sparse-only)",4),
                         ("feat_f1","Mean feature F1 (clean)",3),
                         ("dead","Dead features",0),
                         ("L0","L0",2)]:
    print(f"\n=== TABLE 3.{metric}: {lbl} — rows=layer, cols=epoch 1..10 ===")
    print("layer\t" + "\t".join(f"e{e}" for e in range(1,11)))
    for L in range(1,9):
        cells = []
        for e in range(1,11):
            rr = [x for x in rows if x["layer"]==L and x["epoch"]==e]
            cells.append(fmt(rr[0][metric], p) if rr else "-")
        print(f"{L}\t" + "\t".join(cells))

# ---- TABLE 4: per-concept feature F1 at final epoch, best layer per concept ----
print("\n=== TABLE 4: per-concept feature F1 (final epoch) across layers; best layer marked ===")
concepts = sorted({c for r in finals for c in r["_cdict"].keys()})
print("concept\t" + "\t".join(f"L{r['layer']}" for r in sorted(finals,key=lambda x:x['layer'])) + "\tbestL\tneuron(L_best)")
for c in concepts:
    vals = {}
    for r in sorted(finals, key=lambda x: x["layer"]):
        vals[r["layer"]] = r["_cdict"].get(c, {}).get("f1", float("nan"))
    best_layer = max(vals, key=lambda k: (vals[k] if vals[k]==vals[k] else -1))
    nf = neuron_by_layer.get(best_layer, {}).get(c, {}).get("f1", float("nan"))
    cells = [fmt(vals[L],3) for L in sorted(vals)]
    deg = " *deg" if c in DEGENERATE else ""
    print(f"{c}{deg}\t" + "\t".join(cells) + f"\t{best_layer}\t{fmt(nf,3)}")
