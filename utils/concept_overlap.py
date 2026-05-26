"""Pairwise Jaccard / containment between concept BED files at base resolution
(whole-genome, intersected only with autosomes + chrX so test/train chroms are
both reflected). Outputs a Jaccard CSV and a recommended drop list.

Run on the box that has all_annotations/. No genome FASTA or test BED needed.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_bed(path: Path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["chrom", "start", "end"], usecols=[0, 1, 2])
    return df


def base_coverage(bed: pd.DataFrame, chrom_sizes: dict) -> dict[str, np.ndarray]:
    """Per-chrom boolean mask at base resolution. Skips unknown chroms."""
    out = {c: np.zeros(L, dtype=bool) for c, L in chrom_sizes.items()}
    for chrom, sub in bed.groupby("chrom"):
        if chrom not in out:
            continue
        L = chrom_sizes[chrom]
        starts = sub["start"].clip(0, L).to_numpy()
        ends = sub["end"].clip(0, L).to_numpy()
        for s, e in zip(starts, ends):
            if e > s:
                out[chrom][s:e] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed_dir", type=Path, default=Path("all_annotations"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("results/_aggregate/concept_overlap"))
    ap.add_argument("--jaccard_drop", type=float, default=0.95)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # GRCh38 chrom sizes (autosomes + sex). Hardcoded to avoid FASTA dep.
    CHROM_SIZES = {
        "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
        "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
        "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
        "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
        "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
        "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
    }
    # Bin the genome into 100bp blocks to fit in memory comfortably (3.1B / 100 = 31M bools per concept)
    BIN = 100
    binned = {c: max(1, L // BIN + 1) for c, L in CHROM_SIZES.items()}

    def binned_mask(bed_df):
        out = {c: np.zeros(n, dtype=bool) for c, n in binned.items()}
        for chrom, sub in bed_df.groupby("chrom"):
            if chrom not in out: continue
            n = binned[chrom]
            for s, e in zip(sub["start"].to_numpy(), sub["end"].to_numpy()):
                lo = max(0, s // BIN)
                hi = min(n, (e + BIN - 1) // BIN)
                if hi > lo:
                    out[chrom][lo:hi] = True
        return out

    masks = {}
    for bed in sorted(args.bed_dir.glob("*.bed")):
        name = bed.stem
        m = binned_mask(load_bed(bed))
        total = sum(int(v.sum()) for v in m.values())
        denom = sum(binned.values())
        masks[name] = m
        print(f"  {name:38s}: {total*BIN:>13,} bp  ({100*total/denom:.3f}%)")

    names = sorted(masks)
    n = len(names)
    jac = np.zeros((n, n))
    cont = np.zeros((n, n))   # P(B | A) = |A ∩ B| / |A|
    sizes = np.array([sum(int(v.sum()) for v in masks[k].values()) for k in names])
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            inter = 0
            union = 0
            for chrom in masks[a]:
                ma = masks[a][chrom]; mb = masks[b][chrom]
                inter += int(np.logical_and(ma, mb).sum())
                union += int(np.logical_or(ma, mb).sum())
            jac[i, j] = inter / union if union else 0.0
            cont[i, j] = inter / sizes[i] if sizes[i] else 0.0

    jac_df = pd.DataFrame(jac, index=names, columns=names)
    cont_df = pd.DataFrame(cont, index=names, columns=names)
    jac_df.to_csv(args.out_dir / "jaccard.csv")
    cont_df.to_csv(args.out_dir / "containment.csv")
    print(f"Wrote {args.out_dir/'jaccard.csv'} and containment.csv")

    # Suggest drops: any pair with Jaccard >= threshold -> drop the alphabetically later one
    drop = set()
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i: continue
            if jac[i, j] >= args.jaccard_drop:
                # keep the shorter name (proxy for "less qualified")
                drop.add(b if len(b) >= len(a) else a)
    print(f"\nRecommended drops (Jaccard >= {args.jaccard_drop}):")
    for d in sorted(drop):
        print(f"  drop  {d}")
    (args.out_dir / "drop_list.txt").write_text("\n".join(sorted(drop)))

    print("\nNear-duplicates (Jaccard >= 0.80):")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i: continue
            if jac[i, j] >= 0.80:
                print(f"  {a:35s} <-> {b:35s}  J={jac[i,j]:.3f}")

    print("\nStrong containment (P(B|A) >= 0.90) — candidate hierarchies:")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j: continue
            if cont[i, j] >= 0.90 and sizes[i] < sizes[j]:
                print(f"  {a:35s}  ->  {b:35s}  |A|={sizes[i]:>8}  |B|={sizes[j]:>8}  P(B|A)={cont[i,j]:.3f}")


if __name__ == "__main__":
    main()