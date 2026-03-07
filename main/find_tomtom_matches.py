#!/usr/bin/env python3
"""
Run TomTom motif comparison for SAE features and extract top matches.

Inputs
------
--meme_glob       glob for SAE motif files (*.meme)
--jaspar_db       JASPAR MEME motif database
--repeat_db       Repeat family MEME motif database
--out_dir         output directory

Outputs
-------
For each motif file:

1) tomtom results directories
2) summary table with top motif matches

Example
-------

python3 find_tomtom_matches.py \
  --meme_glob "../results/summaries/*.motifs.meme" \
  --jaspar_db "../motif_dbs/JASPAR2024_CORE_vertebrates.meme" \
  --repeat_db "../motif_dbs/repeat_motifs.meme" \
  --out_dir "../results/tomtom"
"""

import argparse
import glob
import os
import subprocess
import csv
from collections import defaultdict


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def run_tomtom(query, db, out_dir):

    cmd = [
        "tomtom",
        "-no-ssc",
        "-oc", out_dir,
        "-verbosity", "1",
        "-min-overlap", "5",
        "-dist", "pearson",
        "-evalue",
        "-thresh", "0.05",
        query,
        db,
    ]

    subprocess.run(cmd, check=True)


def parse_tomtom(tsv_file, topk=3):

    best = defaultdict(list)

    with open(tsv_file) as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            q = row["Query_ID"]

            if len(best[q]) < topk:
                best[q].append(
                    (
                        row["Target_ID"],
                        float(row["p-value"]),
                        float(row["E-value"]),
                        float(row["q-value"]),
                    )
                )

    return best


def write_summary(out_path, hits):

    features = set(hits.keys())

    with open(out_path, "w") as f:

        f.write(
            "feature\t"
            "match\tpvalue\tevalue\tqvalue\t"
        )

        for feat in sorted(features):

            j = hits.get(feat, [("", "", "", "")])[0]

            f.write(
                f"{feat}\t"
                f"{j[0]}\t{j[1]}\t{j[2]}\t{j[3]}\t"
            )


def process_motif_file(meme_file, db, out_root):

    base = os.path.basename(meme_file)

    tomtom_dir = os.path.join(out_root, f"{base}_{db}")

    ensure_dir(tomtom_dir)

    print(f"Running TomTom vs {db}:", base)

    run_tomtom(meme_file, db, tomtom_dir)

    hits = parse_tomtom(
        os.path.join(tomtom_dir, "tomtom.tsv")
    )

    summary_file = os.path.join(out_root, base + ".top_matches.tsv")

    write_summary(summary_file,hits)

    print("Wrote", summary_file)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--meme_glob", required=True)
    ap.add_argument("--db_path", required=True)
    ap.add_argument("--out_dir", required=True)

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    meme_files = sorted(glob.glob(args.meme_glob))

    if not meme_files:
        raise SystemExit("No MEME files found")

    for meme_file in meme_files:

        process_motif_file(
            meme_file,
            args.db_path,
            args.out_dir,
        )


if __name__ == "__main__":
    main()