# SAE Feature Interpreter

Automated LLM-based interpretation of Sparse Autoencoder (SAE) features from a DNA language model.

## Project structure

```
sae_interpreter/
├── config.py                  # All paths, model names, thresholds
├── utils/
│   ├── genome.py              # Sequence fetching via pyfaidx
│   ├── enrichment.py          # Annotation enrichment formatting
│   └── io.py                  # CSV loading, result saving
├── steps/
│   ├── step1_fetch_sequences.py
│   ├── step2_normalize.py
│   ├── step3_build_prompts.py
│   ├── step4_explain.py
│   ├── step5_score.py
│   ├── step6_recluster.py
│   └── step7_aggregate.py
├── prompts/
│   └── templates.py           # All LLM prompt templates
└── run_pipeline.py            # End-to-end runner
```

## Setup

```bash
pip install anthropic pyfaidx pandas numpy scipy scikit-learn tqdm
```

Download hg38 reference genome:
```bash
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
# Index it
python -c "from pyfaidx import Fasta; Fasta('hg38.fa')"
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Running

```bash
# Full pipeline
python run_pipeline.py

# Individual steps
python steps/step1_fetch_sequences.py
python steps/step2_normalize.py
# ... etc
```

## Input files

- `top_activations.csv` — top 200 activating tokens per feature (genomic coordinates)
- `enrichment.csv` — annotation enrichment counts and percentages per feature

## Output files

- `output/sequences.csv` — top_activations with fetched DNA sequences
- `output/normalized.csv` — with activation_norm column (0–10)
- `output/prompts.jsonl` — formatted prompts per feature
- `output/explanations.jsonl` — LLM-generated explanations
- `output/scores.csv` — Pearson r scores per feature
- `output/reclustered.jsonl` — revised explanations for low-scoring features
- `output/feature_atlas.csv` — final atlas with clusters and labels
