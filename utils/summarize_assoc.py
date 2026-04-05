#!/usr/bin/env python3
"""
For each concept-feature association CSV, write a separate summary text file:
- Top K features by best_f1 with metrics (f1, enrichment, precision, recall, threshold, n_active)
- For each of those features: motif/pattern from top activating examples (feature_top_examples.pt)
  using hg38 FASTA around genome_pos0:
    * consensus
    * PFM counts + frequencies
    * top centered k-mers

Outputs:
  --out_dir/<csv_basename>.summary.txt

Example:
python3 summarize_assoc.py \
  --csv_glob "../results/*_feature_assoc_final.csv" \
  --top_examples_pt "../results/feature_top_examples.pt" \
  --fasta "../data/hg38.fa" \
  --out_dir "../results/summaries" \
  --topk 10 \
  --examples_per_feature 100 \
  --radius 10 \
  --center_k 9
"""

import argparse
import csv
from email.mime import base
import glob
import os
from collections import Counter
from typing import Dict, List, Tuple, Any

import torch

try:
    from pyfaidx import Fasta
except ImportError as e:
    raise SystemExit("Missing dependency pyfaidx. Install: pip install pyfaidx") from e


BASES = ["A", "C", "G", "T", "N"]
BASE_SET = set(BASES)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)
    os.chmod(p, 0o777)


def sanitize_seq(s: str) -> str:
    s = s.upper()
    return "".join(ch if ch in BASE_SET else "N" for ch in s)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return int(float(s))
    except Exception:
        return default


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def load_top_examples(top_examples_pt: str) -> Dict[int, List[Dict[str, Any]]]:
    obj = torch.load(top_examples_pt, map_location="cpu")
    feats = obj.get("features", [])
    out: Dict[int, List[Dict[str, Any]]] = {}
    for f in feats:
        fid = int(f["feature_id"])
        out[fid] = f.get("examples", [])
    return out


def fetch_centered_sequence(genome: Fasta, chrom: str, center0: int, radius: int) -> str:
    L = 2 * radius + 1
    left = center0 - radius
    right = center0 + radius + 1  # exclusive

    pad_left = 0
    pad_right = 0

    if left < 0:
        pad_left = -left
        left = 0

    try:
        chrom_len = len(genome[chrom])
    except Exception:
        return "N" * L

    if right > chrom_len:
        pad_right = right - chrom_len
        right = chrom_len

    try:
        seq = str(genome[chrom][left:right])
    except Exception:
        seq = ""

    seq = sanitize_seq(seq)
    if pad_left:
        seq = ("N" * pad_left) + seq
    if pad_right:
        seq = seq + ("N" * pad_right)

    if len(seq) != L:
        if len(seq) < L:
            seq = seq + ("N" * (L - len(seq)))
        else:
            seq = seq[:L]
    return seq


def compute_pfm(seqs: List[str]) -> Tuple[List[Dict[str, int]], List[Dict[str, float]], str]:
    if not seqs:
        return [], [], ""
    L = len(seqs[0])
    counts = [dict((b, 0) for b in BASES) for _ in range(L)]

    for s in seqs:
        if len(s) != L:
            continue
        for i, ch in enumerate(s):
            if ch not in BASE_SET:
                ch = "N"
            counts[i][ch] += 1

    n = float(len(seqs))
    freqs: List[Dict[str, float]] = []
    consensus_chars: List[str] = []
    for i in range(L):
        freqs.append({b: counts[i][b] / n for b in BASES})
        consensus_chars.append(max(BASES, key=lambda b: counts[i][b]))
    return counts, freqs, "".join(consensus_chars)


def format_pfm(counts: List[Dict[str, int]], freqs: List[Dict[str, float]]) -> str:
    if not counts:
        return "(no sequences)\n"
    L = len(counts)
    lines = []
    lines.append("pos\t" + "\t".join(str(i - (L // 2)) for i in range(L)))
    for b in BASES:
        lines.append(f"{b}_count\t" + "\t".join(str(counts[i][b]) for i in range(L)))
    for b in BASES:
        lines.append(f"{b}_freq\t" + "\t".join(f"{freqs[i][b]:.3f}" for i in range(L)))
    return "\n".join(lines) + "\n"

def write_meme_file(motifs, out_path):
    """
    motifs: list of (feature_id, freqs) where freqs is list[pos][base]
    """
    with open(out_path, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies:\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        for fid, freqs in motifs:
            w = len(freqs)

            f.write(f"MOTIF feature_{fid}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {w}\n")

            for pos in freqs:
                f.write(
                    f"{pos['A']:.6f} "
                    f"{pos['C']:.6f} "
                    f"{pos['G']:.6f} "
                    f"{pos['T']:.6f}\n"
                )

            f.write("\n")


def centered_kmer_counts(seqs: List[str], center_k: int) -> Counter:
    c = Counter()
    if not seqs:
        return c
    L = len(seqs[0])
    if center_k <= 0 or center_k > L:
        return c
    mid = L // 2
    half = center_k // 2
    a = mid - half
    b = mid + half + 1
    for s in seqs:
        if len(s) != L:
            continue
        c[s[a:b]] += 1
    return c


def summarize_one_csv(
    csv_path: str,
    out_path: str,
    meme_out_path: str,
    top_examples: Dict[int, List[Dict[str, Any]]],
    genome: Fasta,
    topk: int,
    examples_per_feature: int,
    radius: int,
    center_k: int,
) -> None:
    rows = read_csv_rows(csv_path)
    with open(out_path, "w", encoding="utf-8") as out:
        motifs_for_meme = []
        out.write(f"CSV: {csv_path}\n")
        if not rows:
            out.write("(empty)\n")
            return

        parsed = []
        for r in rows:
            fid = safe_int(r.get("feature_id", ""), -1)
            f1 = safe_float(r.get("best_f1", 0.0), 0.0)
            enr = safe_float(r.get("enrichment_at_best_f1", 0.0), 0.0)
            prec = safe_float(r.get("best_precision", 0.0), 0.0)
            rec = safe_float(r.get("best_recall", 0.0), 0.0)
            thr = safe_float(r.get("best_t", ""), float("nan"))
            n_active = safe_int(r.get("n_active", 0), 0)
            parsed.append((fid, f1, enr, prec, rec, thr, n_active))

        parsed.sort(key=lambda x: x[1], reverse=True)
        top = parsed[: max(0, topk)]

        out.write("\nTop features by best_f1:\n")
        out.write("rank\tfid\tbest_f1\tenrichment\tprecision\trecall\tthreshold\tn_active\n")
        for i, (fid, f1, enr, prec, rec, thr, n_active) in enumerate(top, start=1):
            thr_s = f"{thr:.6g}" if thr == thr else ""
            out.write(f"{i}\t{fid}\t{f1:.6g}\t{enr:.6g}\t{prec:.6g}\t{rec:.6g}\t{thr_s}\t{n_active}\n")

        for (fid, f1, enr, prec, rec, thr, n_active) in top:
            out.write("\n" + "-" * 100 + "\n")
            out.write(f"Feature {fid} | best_f1={f1:.6g} enrichment={enr:.6g} "
                      f"precision={prec:.6g} recall={rec:.6g}\n")

            ex_list = top_examples.get(fid, [])
            if not ex_list:
                out.write("No examples found in feature_top_examples.pt for this feature.\n")
                continue

            ex_sorted = sorted(ex_list, key=lambda d: float(d.get("score", 0.0)), reverse=True)
            ex_sorted = ex_sorted[: max(1, examples_per_feature)]

            seqs = []
            used = 0
            for ex in ex_sorted:
                chrom = str(ex.get("chrom", ""))
                center0 = ex.get("genome_pos0", None)
                if center0 is None:
                    if "start" in ex and "pos" in ex:
                        center0 = int(ex["start"]) + int(ex["pos"])
                    else:
                        continue
                center0 = int(center0)
                seqs.append(fetch_centered_sequence(genome, chrom, center0, radius))
                used += 1

            if not seqs:
                out.write("Could not extract any sequences for motif.\n")
                continue

            counts, freqs, consensus = compute_pfm(seqs)

            # Save motif for MEME output
            motifs_for_meme.append((fid, freqs))
            out.write(f"Motif window: radius={radius} (len={2*radius+1}), examples_used={used}\n")
            out.write(f"Consensus: {consensus}\n")
            out.write("PFM (positions relative to center):\n")
            out.write(format_pfm(counts, freqs))

            if center_k and center_k > 0:
                kc = centered_kmer_counts(seqs, center_k)
                out.write(f"Top centered {center_k}-mers:\n")
                for kmer, cnt in kc.most_common(10):
                    out.write(f"  {kmer}\t{cnt}\t({cnt/len(seqs):.3f})\n")

        if meme_out_path:
            write_meme_file(motifs_for_meme, meme_out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", required=True)
    ap.add_argument("--top_examples_pt", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--examples_per_feature", type=int, default=100)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--center_k", type=int, default=9)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    csv_paths = sorted(glob.glob(args.csv_glob))
    if not csv_paths:
        raise SystemExit(f"No CSV files matched: {args.csv_glob}")

    top_examples = load_top_examples(args.top_examples_pt)
    genome = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)

    for csv_path in csv_paths:
        base = os.path.basename(csv_path)
        out_path = os.path.join(args.out_dir, base + ".summary.txt")
        meme_out_path = os.path.join(args.out_dir, base + ".motifs.meme")
        summarize_one_csv(
            csv_path=csv_path,
            out_path=out_path,
            meme_out_path=meme_out_path,
            top_examples=top_examples,
            genome=genome,
            topk=args.topk,
            examples_per_feature=args.examples_per_feature,
            radius=args.radius,
            center_k=args.center_k,
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()