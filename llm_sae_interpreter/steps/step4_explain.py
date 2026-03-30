"""
steps/step4_explain.py — Call the explainer LLM for each feature.

Reads prompts from output/prompts.jsonl, sends each to the Anthropic API,
parses the structured response (HYPOTHESIS / EVIDENCE / CONFIDENCE), and
saves results to output/explanations.jsonl.

Supports resuming: skips features already in the output file.

Input:  output/prompts.jsonl
Output: output/explanations.jsonl

Run:
    python steps/step4_explain.py
"""

import sys
import re
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROMPTS_JSONL,
    EXPLANATIONS_JSONL,
    EXPLAINER_MODEL,
    MAX_TOKENS_EXPLAIN,
    API_CALL_DELAY,
    ANTHROPIC_API_KEY,
)
from utils.io import load_jsonl, append_jsonl, load_existing_feature_ids

import anthropic
from tqdm import tqdm


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_explanation(raw_text: str) -> dict:
    """
    Parse the structured LLM response into hypothesis, evidence, confidence.

    Expected format:
        HYPOTHESIS: <text>
        EVIDENCE: <text>
        CONFIDENCE: <HIGH | MEDIUM | LOW>

    Returns a dict with keys: hypothesis, evidence, confidence, parse_ok.
    If parsing fails, stores the raw text and sets parse_ok=False.
    """
    result = {
        "hypothesis":  "",
        "evidence":    "",
        "confidence":  "",
        "raw_response": raw_text,
        "parse_ok":    False,
    }

    # Extract HYPOTHESIS
    m = re.search(r"HYPOTHESIS:\s*(.+?)(?=\nEVIDENCE:|\nCONFIDENCE:|$)", raw_text, re.S)
    if m:
        result["hypothesis"] = m.group(1).strip()

    # Extract EVIDENCE
    m = re.search(r"EVIDENCE:\s*(.+?)(?=\nCONFIDENCE:|$)", raw_text, re.S)
    if m:
        result["evidence"] = m.group(1).strip()

    # Extract CONFIDENCE
    m = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", raw_text, re.I)
    if m:
        result["confidence"] = m.group(1).upper()

    result["parse_ok"] = bool(
        result["hypothesis"] and result["confidence"]
    )

    return result


# ── API call ───────────────────────────────────────────────────────────────────

def call_explainer(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_prompt: str,
    feature_idx: int,
    max_retries: int = 3,
) -> dict:
    """
    Call the explainer LLM with retry logic.

    Returns a dict with the parsed explanation plus metadata.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=EXPLAINER_MODEL,
                max_tokens=MAX_TOKENS_EXPLAIN,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )

            raw_text = response.content[0].text
            parsed   = parse_explanation(raw_text)

            return {
                "feature_idx":    feature_idx,
                "model":          EXPLAINER_MODEL,
                "attempt":        attempt,
                "input_tokens":   response.usage.input_tokens,
                "output_tokens":  response.usage.output_tokens,
                "api_error":      None,
                **parsed,
            }

        except anthropic.RateLimitError as e:
            wait = 60 * attempt
            print(f"\n  [Feature {feature_idx}] Rate limit hit (attempt {attempt}). "
                  f"Waiting {wait}s ...")
            time.sleep(wait)
            last_error = str(e)

        except anthropic.APIError as e:
            print(f"\n  [Feature {feature_idx}] API error attempt {attempt}: {e}")
            time.sleep(5 * attempt)
            last_error = str(e)

        except Exception as e:
            print(f"\n  [Feature {feature_idx}] Unexpected error: {e}")
            traceback.print_exc()
            last_error = str(e)
            break  # Don't retry on unexpected errors

    # All retries exhausted
    return {
        "feature_idx":    feature_idx,
        "model":          EXPLAINER_MODEL,
        "attempt":        max_retries,
        "input_tokens":   0,
        "output_tokens":  0,
        "api_error":      last_error,
        "hypothesis":     "",
        "evidence":       "",
        "confidence":     "",
        "raw_response":   "",
        "parse_ok":       False,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> list[dict]:
    print("=" * 60)
    print("STEP 4: Calling explainer LLM")
    print("=" * 60)

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Export it before running:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    # ── Load prompts ─────────────────────────────────────────────────────────
    print(f"\nLoading {PROMPTS_JSONL} ...")
    prompts = load_jsonl(PROMPTS_JSONL)
    print(f"  {len(prompts)} features to explain")

    # ── Resume support ───────────────────────────────────────────────────────
    done_ids = load_existing_feature_ids(EXPLANATIONS_JSONL)
    todo     = [p for p in prompts if p["feature_idx"] not in done_ids]
    print(f"  Already done: {len(done_ids)} | Remaining: {len(todo)}")

    if not todo:
        print("  All features already explained.")
        return load_jsonl(EXPLANATIONS_JSONL)

    # ── API client ────────────────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Call LLM ─────────────────────────────────────────────────────────────
    results  = []
    n_ok     = 0
    n_fail   = 0
    n_parse_fail = 0
    total_in = 0
    total_out= 0

    for prompt_record in tqdm(todo, desc="Explaining features"):
        feat_idx = prompt_record["feature_idx"]

        result = call_explainer(
            client        = client,
            system_prompt = prompt_record["system_prompt"],
            user_prompt   = prompt_record["user_prompt"],
            feature_idx   = feat_idx,
        )

        # Carry over holdout data from prompt record for use in step 5
        result["holdout_sequences"]   = prompt_record.get("holdout_sequences", [])
        result["holdout_activations"] = prompt_record.get("holdout_activations", [])
        result["enrichment_block"]    = prompt_record.get("enrichment_block", "")

        append_jsonl(result, EXPLANATIONS_JSONL)
        results.append(result)

        if result["api_error"]:
            n_fail += 1
        elif not result["parse_ok"]:
            n_parse_fail += 1
        else:
            n_ok += 1

        total_in  += result["input_tokens"]
        total_out += result["output_tokens"]

        time.sleep(API_CALL_DELAY)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nResults:")
    print(f"  Successfully explained: {n_ok}")
    print(f"  Parse failures:         {n_parse_fail}")
    print(f"  API errors:             {n_fail}")
    print(f"  Total tokens used:      {total_in} in / {total_out} out")

    # Preview a sample explanation
    good = [r for r in results if r["parse_ok"]]
    if good:
        ex = good[0]
        print(f"\n--- SAMPLE EXPLANATION (feature {ex['feature_idx']}) ---")
        print(f"HYPOTHESIS:  {ex['hypothesis']}")
        print(f"EVIDENCE:    {ex['evidence']}")
        print(f"CONFIDENCE:  {ex['confidence']}")

    return load_jsonl(EXPLANATIONS_JSONL)


if __name__ == "__main__":
    run()
