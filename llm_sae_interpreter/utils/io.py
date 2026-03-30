"""
utils/io.py — CSV / JSONL loading and saving helpers used across all steps.
"""

import json
import pandas as pd
from pathlib import Path


# ── CSV helpers ────────────────────────────────────────────────────────────────

def load_top_activations(path) -> pd.DataFrame:
    """Load top_activations.csv with correct dtypes."""
    df = pd.read_csv(
        path,
        dtype={
            "feature_idx":      int,
            "rank":             int,
            "activation_value": float,
            "split":            str,
            "shard_path":       str,
            "seq_idx":          int,
            "tok_pos":          int,
            "coord_chrom":      str,
            "coord_start":      int,
            "coord_end":        int,
            "context_window":   str,
        },
    )
    return df


def load_enrichment(path) -> pd.DataFrame:
    """Load enrichment.csv indexed by feature_idx."""
    df = pd.read_csv(path)
    df = df.set_index("feature_idx")
    return df


def load_sequences(path) -> pd.DataFrame:
    """Load output/sequences.csv produced by step 1."""
    return pd.read_csv(path, dtype={"coord_chrom": str})


def load_normalized(path) -> pd.DataFrame:
    """Load output/normalized.csv produced by step 2."""
    return pd.read_csv(path, dtype={"coord_chrom": str})


# ── JSONL helpers ──────────────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path) -> None:
    """Write a list of dicts to a JSONL file (one JSON object per line)."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def load_jsonl(path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(record: dict, path) -> None:
    """Append a single record to a JSONL file (creates file if missing)."""
    path = Path(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_feature_ids(path) -> set[int]:
    """Return the set of feature_idx values already written to a JSONL file.
    Used to resume interrupted pipeline runs."""
    path = Path(path)
    if not path.exists():
        return set()
    ids = set()
    for record in load_jsonl(path):
        if "feature_idx" in record:
            ids.add(int(record["feature_idx"]))
    return ids
