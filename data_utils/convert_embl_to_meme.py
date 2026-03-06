#!/usr/bin/env python3
"""
Convert Dfam curated EMBL repeat sequences to MEME motif format.

Each repeat consensus sequence is converted to a PWM motif
(one-hot counts with pseudocount smoothing).

Input:
    Dfam_curatedonly.embl

Output:
    Dfam_curated.meme

Usage:
python embl_to_meme.py \
    --embl Dfam_curatedonly.embl \
    --out Dfam_curated.meme
"""

import argparse
import re

BASES = ["A","C","G","T"]


def parse_embl(path):
    """
    Parse EMBL entries.
    Returns list of (name, sequence).
    """
    motifs = []

    name = None
    seq_lines = []
    reading_seq = False

    with open(path) as f:
        for line in f:

            if line.startswith("ID"):
                name = line.split()[1]

            elif line.startswith("DE"):
                desc = line.strip()[2:].strip()
                name = f"{name}_{desc}".replace(" ", "_")

            elif line.startswith("SQ"):
                reading_seq = True
                seq_lines = []

            elif line.startswith("//"):
                seq = "".join(seq_lines).upper()
                seq = re.sub("[^ACGT]", "N", seq)

                if name and seq:
                    motifs.append((name, seq))

                name = None
                seq_lines = []
                reading_seq = False

            elif reading_seq:
                seq_lines.append("".join(line.strip().split()[0]))

    return motifs


def seq_to_pwm(seq, pseudocount=0.1):
    """
    Convert sequence to PWM rows.
    """
    L = len(seq)
    pwm = []

    for i in range(L):
        counts = {b:pseudocount for b in BASES}

        base = seq[i]
        if base in BASES:
            counts[base] += 1

        total = sum(counts.values())

        row = [counts[b]/total for b in BASES]
        pwm.append(row)

    return pwm


def write_meme(motifs, out_path):

    with open(out_path,"w") as out:

        out.write("MEME version 4\n\n")
        out.write("ALPHABET= ACGT\n\n")
        out.write("strands: + -\n\n")
        out.write("Background letter frequencies:\n")
        out.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        for name,seq in motifs:

            pwm = seq_to_pwm(seq)

            out.write(f"MOTIF {name}\n")
            out.write(f"letter-probability matrix: alength= 4 w= {len(seq)}\n")

            for row in pwm:
                out.write(" ".join(f"{x:.6f}" for x in row) + "\n")

            out.write("\n")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--embl", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    motifs = parse_embl(args.embl)

    print(f"Parsed {len(motifs)} repeat families")

    write_meme(motifs, args.out)

    print(f"Wrote MEME file: {args.out}")


if __name__ == "__main__":
    main()