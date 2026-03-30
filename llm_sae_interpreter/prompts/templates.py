"""
prompts/templates.py — All LLM prompt templates used in the pipeline.

Every API call in this project uses a prompt defined here.
Templates are plain strings with {placeholder} substitution.
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════

EXPLAINER_SYSTEM = """\
You are an expert computational genomicist analysing features learned by a \
Sparse Autoencoder (SAE) trained on a DNA language model.

The model was trained on the human genome using CHARACTER-LEVEL tokenisation: \
each token is a single nucleotide — A, C, G, T, or N. The SAE learns \
sparse linear features over the model's residual stream. Each feature \
corresponds to a learned pattern that the DNA model implicitly represents.

Your task is to propose a concise biological hypothesis explaining what \
genomic or sequence pattern a given SAE feature detects. You will be shown:
  1. Annotation enrichment statistics over the feature's top 200 activating loci.
  2. The top activating examples: 512 bp genomic windows with the most-\
activating nucleotide marked in square brackets [X] and a normalised \
activation score (0–10).

When formulating your hypothesis, consider:
  • Specific sequence motifs (e.g. TATA-box TATAWAW, Kozak GCCRCCAUGG, \
poly-A AATAAA, CTCF CCGCGNGGNGGCAG)
  • Repeat element families and subfamilies (Alu, L1, L2, LTR, satellite, \
simple repeats, low-complexity)
  • Functional regulatory elements (promoters, enhancers, insulators, \
silencers, splice sites)
  • Epigenetic or structural features (CpG islands, open chromatin, \
CTCF-bound insulator sites)
  • Variant-associated contexts (high ClinVar pathogenic enrichment may \
indicate the feature is sensitive to mutation-prone regulatory regions)
  • Strand or positional biases visible across examples

Consult online genome databases like ENCODE, GENCODE, JASPAR etc. as necessary.

Respond with EXACTLY this format — nothing else:
HYPOTHESIS: <1–2 sentences describing the specific genomic feature detected>
EVIDENCE: <1 sentence citing the strongest supporting signals from annotations and sequences>
CONFIDENCE: <HIGH | MEDIUM | LOW>
"""

EXPLAINER_USER = """\
=== FEATURE {feature_idx} ===

--- Annotation enrichment (top 200 activating loci) ---
{enrichment_block}

--- Top {n_examples} highest-activating examples ---
(Activating nucleotide marked as [X]; normalised activation score 0–10)

{examples_block}

Based on the above, what genomic or sequence feature does this SAE feature detect?
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

SIMULATOR_SYSTEM = """\
You are an expert computational genomicist. You will be given:
  1. A hypothesis about what pattern a DNA language model SAE feature detects.
  2. A list of genomic sequence windows (512 bp each), with one nucleotide \
marked as [X].

Your task is to PREDICT the normalised activation score (integer 0–10) that \
this SAE feature would produce at position [X] in each window, given the \
hypothesis. A score of 10 means the hypothesis pattern is strongly present at \
that exact position; 0 means it is absent.

Rules:
  • Respond ONLY with a JSON array of integers, one per example, in order.
  • Do NOT include any text, explanation, markdown, or code fences.
  • The array must have exactly {n_examples} elements.
  • Each element must be an integer from 0 to 10.

Example valid response for 3 examples: [8, 0, 5]
"""

SIMULATOR_USER = """\
=== HYPOTHESIS ===
{hypothesis}

=== SEQUENCES TO SCORE ({n_examples} examples) ===
{sequences_block}

Predict the activation at position [X] for each example.
Respond with ONLY a JSON array of {n_examples} integers (0–10).
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — RECLUSTERING (low-scoring features)
# ══════════════════════════════════════════════════════════════════════════════

RECLUSTER_SYSTEM = """\
You are an expert computational genomicist. A previous attempt to interpret \
an SAE feature from a DNA language model produced a low-quality explanation \
(low simulation score), suggesting the feature may be POLYSEMANTIC — \
activating on multiple distinct biological patterns.

You will be shown the top activating examples grouped into {n_clusters} \
sub-clusters based on their annotation profiles. For EACH sub-cluster, \
propose a separate biological hypothesis.

Respond with EXACTLY this format — one block per sub-cluster, nothing else:
SUBCLUSTER 1:
HYPOTHESIS: <1–2 sentences>
EVIDENCE: <1 sentence>
CONFIDENCE: <HIGH | MEDIUM | LOW>

SUBCLUSTER 2:
HYPOTHESIS: <1–2 sentences>
EVIDENCE: <1 sentence>
CONFIDENCE: <HIGH | MEDIUM | LOW>

(repeat for each sub-cluster)
"""

RECLUSTER_USER = """\
=== FEATURE {feature_idx} (low simulation score: {score:.3f}) ===

Original hypothesis (insufficient):
{original_hypothesis}

--- Sub-cluster annotation profiles ---
{subcluster_enrichment_block}

--- Sub-cluster example sequences ---
{subcluster_examples_block}

Propose a separate hypothesis for each sub-cluster.
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — CLUSTER LABELLING
# ══════════════════════════════════════════════════════════════════════════════

CLUSTER_LABEL_SYSTEM = """\
You are an expert computational genomicist. You will be given a group of SAE \
feature interpretations that have been clustered together by semantic \
similarity. Your task is to produce a single short label (3–7 words) that \
best describes the shared biological theme of the cluster.

Respond with EXACTLY this format — nothing else:
LABEL: <3–7 word label>
DESCRIPTION: <1 sentence summarising the shared theme>
"""

CLUSTER_LABEL_USER = """\
=== CLUSTER {cluster_id} ({n_features} features) ===

Feature hypotheses in this cluster:
{hypotheses_block}

What is the shared biological theme? Provide a 3–7 word label and a \
1-sentence description.
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — CONSISTENCY CHECK
# ══════════════════════════════════════════════════════════════════════════════

CONSISTENCY_CHECK_SYSTEM = """\
You are an expert computational genomicist performing quality control on \
automated SAE feature interpretations. You will be given:
  1. An LLM-generated hypothesis about what a DNA SAE feature detects.
  2. The annotation enrichment statistics for that feature's top 200 \
activating loci.

Assess whether the hypothesis is CONSISTENT with the annotation statistics. \
Flag any contradictions (e.g. hypothesis claims "promoter" but promoter \
enrichment is <1%).

Respond with EXACTLY this format:
CONSISTENT: <YES | NO | PARTIAL>
ISSUES: <brief description of any contradictions, or "none">
REVISED_HYPOTHESIS: <corrected 1–2 sentence hypothesis, or repeat original if consistent>
"""

CONSISTENCY_CHECK_USER = """\
=== FEATURE {feature_idx} ===

--- Generated hypothesis ---
{hypothesis}

--- Annotation enrichment ---
{enrichment_block}

Is this hypothesis consistent with the annotation statistics?
"""
