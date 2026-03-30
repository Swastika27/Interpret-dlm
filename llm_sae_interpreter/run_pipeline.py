"""
run_pipeline.py — End-to-end SAE feature interpretation pipeline.

Runs all 7 steps in sequence. Each step is independently resumable:
if the output file already exists and contains partial results, the step
will skip completed features and continue from where it left off.

Usage:
    python run_pipeline.py               # run all steps
    python run_pipeline.py --steps 1 2  # run only steps 1 and 2
    python run_pipeline.py --from 4     # run steps 4 through 7
    python run_pipeline.py --step 5     # run only step 5
"""

import sys
import time
import argparse
import traceback
from pathlib import Path

# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAE Feature Interpretation Pipeline"
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, metavar="N",
        help="Run specific steps (e.g. --steps 1 2 3)"
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, metavar="N",
        help="Run from step N to step 7"
    )
    parser.add_argument(
        "--step", type=int, metavar="N",
        help="Run only step N"
    )
    return parser.parse_args()


def get_steps_to_run(args) -> list[int]:
    all_steps = [1, 2, 3, 4, 5, 6, 7]
    if args.step:
        return [args.step]
    if args.steps:
        return sorted(args.steps)
    if args.from_step:
        return [s for s in all_steps if s >= args.from_step]
    return all_steps


# ── Pipeline runner ────────────────────────────────────────────────────────────

STEP_DESCRIPTIONS = {
    1: "Fetch DNA sequences from reference genome",
    2: "Normalise activations (0–10) and highlight positions",
    3: "Build explainer prompts",
    4: "Call explainer LLM — generate hypotheses",
    5: "Score explanations via LLM simulation",
    6: "Recluster low-scoring features",
    7: "Aggregate, consistency-check, cluster, and build atlas",
}


def run_step(step_num: int) -> bool:
    """Run a single pipeline step. Returns True on success."""
    print(f"\n{'#' * 70}")
    print(f"# STEP {step_num}: {STEP_DESCRIPTIONS[step_num]}")
    print(f"{'#' * 70}\n")

    t0 = time.time()
    try:
        if step_num == 1:
            from steps.step1_fetch_sequences import run
        elif step_num == 2:
            from steps.step2_normalize import run
        elif step_num == 3:
            from steps.step3_build_prompts import run
        elif step_num == 4:
            from steps.step4_explain import run
        elif step_num == 5:
            from steps.step5_score import run
        elif step_num == 6:
            from steps.step6_recluster import run
        elif step_num == 7:
            from steps.step7_aggregate import run
        else:
            print(f"Unknown step: {step_num}")
            return False

        run()
        elapsed = time.time() - t0
        print(f"\n✓ Step {step_num} completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n✗ Step {step_num} FAILED after {elapsed:.1f}s")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def preflight_check() -> bool:
    """Check that required config values are set before starting."""
    from config import GENOME_FASTA, ANTHROPIC_API_KEY, TOP_ACTIVATIONS_CSV, ENRICHMENT_CSV

    ok = True

    if not TOP_ACTIVATIONS_CSV.exists():
        print(f"ERROR: top_activations.csv not found at {TOP_ACTIVATIONS_CSV}")
        ok = False

    if not ENRICHMENT_CSV.exists():
        print(f"ERROR: enrichment.csv not found at {ENRICHMENT_CSV}")
        ok = False

    if str(GENOME_FASTA) == "/path/to/hg38.fa" or not GENOME_FASTA.exists():
        print(f"WARNING: GENOME_FASTA not set or not found: {GENOME_FASTA}")
        print("  Steps 1 and 6 (sequence fetching) will fail.")
        print("  Update GENOME_FASTA in config.py before running steps 1 or 6.")

    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  Export it with: export ANTHROPIC_API_KEY='sk-ant-...'")
        ok = False

    return ok


def main():
    args = parse_args()
    steps_to_run = get_steps_to_run(args)

    print("=" * 70)
    print("  SAE Feature Interpretation Pipeline")
    print("=" * 70)
    print(f"\nSteps to run: {steps_to_run}")

    # Preflight
    if not preflight_check():
        print("\nPreflight check failed. Fix the issues above and retry.")
        sys.exit(1)

    # Run steps
    t_total = time.time()
    results = {}

    for step in steps_to_run:
        success = run_step(step)
        results[step] = success
        if not success:
            print(f"\nPipeline halted at step {step}. Fix the error and re-run with --from {step}")
            break

    # Final report
    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {elapsed_total:.1f}s")
    print(f"{'=' * 70}")
    print("\nStep results:")
    for step, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  Step {step}: {status} — {STEP_DESCRIPTIONS[step]}")

    from config import FEATURE_ATLAS_CSV
    if FEATURE_ATLAS_CSV.exists():
        print(f"\nFinal atlas: {FEATURE_ATLAS_CSV}")


if __name__ == "__main__":
    main()
