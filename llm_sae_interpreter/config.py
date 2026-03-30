"""
config.py — Central configuration for the SAE interpreter pipeline.
Edit paths and thresholds here; all steps import from this file.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/mnt/disk1/swastika/Interpret-dlm")
MODEL_TO_ANALYZE = "layer6_16384_batchtopk_32_0.0003"

# Input files
TOP_ACTIVATIONS_CSV = BASE_DIR / "results" / MODEL_TO_ANALYZE / "top_activations" / "top_activations.csv"
ENRICHMENT_CSV      = BASE_DIR / "results" / MODEL_TO_ANALYZE / "feature_annotation_assoc" / "enrichment.csv"

# Reference genome (hg38 FASTA, must be indexed by pyfaidx)
GENOME_FASTA = BASE_DIR / "data" / "raw" / "GRCh38.primary_assembly.genome.fa"

# Output directory
OUTPUT_DIR = BASE_DIR / "results" / MODEL_TO_ANALYZE / "claude_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Per-step outputs
SEQUENCES_CSV      = OUTPUT_DIR / "sequences.csv"
NORMALIZED_CSV     = OUTPUT_DIR / "normalized.csv"
PROMPTS_JSONL      = OUTPUT_DIR / "prompts.jsonl"
EXPLANATIONS_JSONL = OUTPUT_DIR / "explanations.jsonl"
SCORES_CSV         = OUTPUT_DIR / "scores.csv"
RECLUSTERED_JSONL  = OUTPUT_DIR / "reclustered.jsonl"
FEATURE_ATLAS_CSV  = OUTPUT_DIR / "feature_atlas.csv"

# ── Anthropic API ──────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
EXPLAINER_MODEL     = "claude-opus-4-5"
SIMULATOR_MODEL     = "claude-opus-4-5"
CLUSTER_LABEL_MODEL = "claude-opus-4-5"
MAX_TOKENS_EXPLAIN  = 300
MAX_TOKENS_SIMULATE = 400
MAX_TOKENS_CLUSTER  = 200

# Delay between API calls (seconds) to avoid rate limits
API_CALL_DELAY = 0.5

# ── Pipeline parameters ────────────────────────────────────────────────────────

# Number of top examples shown to the explainer
N_EXPLAINER_EXAMPLES = 20

# Examples held out for simulation scoring (index range within the top 200)
SIMULATION_HOLDOUT_START = 100
SIMULATION_HOLDOUT_END   = 130   # 30 examples for simulation

# Minimum Pearson r to accept an explanation without revision
SCORE_ACCEPT_THRESHOLD = 0.5

# Minimum Pearson r after reclustering; below this, mark as uninterpretable
SCORE_MIN_THRESHOLD = 0.2

# Number of base pairs of flanking context to show per side (in addition to
# the activating window)  — used for reclustering with expanded context
FLANK_BP = 256

# Annotation columns to include in the enrichment summary (pct_ prefix)
ANNOTATION_PCT_COLS = [
    "pct_gencode_promoter",
    "pct_gencode_exon",
    "pct_gencode_intron",
    "pct_gencode_CDS",
    "pct_gencode_splice_acceptor",
    "pct_gencode_splice_donor",
    "pct_encode_CTCF-bound",
    "pct_encode_PLS",
    "pct_encode_dELS",
    "pct_encode_pELS",
    "pct_encode_DNase-H3K4me3",
    "pct_cpg_cpg_islands.hg38",
    "pct_repeats_SINE",
    "pct_repeats_LINE",
    "pct_repeats_LTR",
    "pct_repeats_Satellite",
    "pct_repeats_Simple_repeat",
    "pct_repeats_Low_complexity",
    "pct_clinvar_pathogenic",
    "pct_clinvar_benign",
]

# Human-readable labels for annotation columns (for prompt formatting)
ANNOTATION_LABELS = {
    "pct_gencode_promoter":       "promoter",
    "pct_gencode_exon":           "exon",
    "pct_gencode_intron":         "intron",
    "pct_gencode_CDS":            "CDS",
    "pct_gencode_splice_acceptor":"splice acceptor",
    "pct_gencode_splice_donor":   "splice donor",
    "pct_encode_CTCF-bound":      "CTCF-bound",
    "pct_encode_PLS":             "ENCODE PLS (promoter-like)",
    "pct_encode_dELS":            "distal enhancer-like (dELS)",
    "pct_encode_pELS":            "proximal enhancer-like (pELS)",
    "pct_encode_DNase-H3K4me3":   "DNase+H3K4me3 (open chromatin)",
    "pct_cpg_cpg_islands.hg38":   "CpG island",
    "pct_repeats_SINE":           "SINE repeat",
    "pct_repeats_LINE":           "LINE repeat",
    "pct_repeats_LTR":            "LTR repeat",
    "pct_repeats_Satellite":      "satellite repeat",
    "pct_repeats_Simple_repeat":  "simple repeat",
    "pct_repeats_Low_complexity": "low complexity repeat",
    "pct_clinvar_pathogenic":     "ClinVar pathogenic variant",
    "pct_clinvar_benign":         "ClinVar benign variant",
}

# Threshold (%) above which an annotation is considered "enriched" in the prompt
ENRICHMENT_DISPLAY_THRESHOLD = 5.0

# Number of sub-clusters for low-scoring features
N_SUBCLUSTERS = 3
