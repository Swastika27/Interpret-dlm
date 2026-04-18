#!/usr/bin/env python3
"""
Compare HyenaDNA hidden representations for sanity checks:

1) AutoModel vs AutoModelForCausalLM: ``output_hidden_states`` tuples should match
   (same checkpoint, same inputs) within floating-point tolerance.

2) Causal LM only: for each ``li`` with ``1 <= li < len(hidden_states)``, the tensor
   ``hidden_states[li]`` should match the forward output of ``hyena.backbone.layers[li-1]``
   (hook capture), matching the indexing used for embedding extraction vs fidelity patching.

Usage (from repo root):

    python utils/compare_hyena_hidden_states.py \\
        --model_id LongSafari/hyenadna-large-1m-seqlen-hf \\
        --seq_len 512

Requires: transformers, torch
"""

from __future__ import annotations

import argparse
import os
import random
import sys

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from gpu_setup import (  # noqa: E402
    configure_cuda_performance,
    resolve_device_str,
    tensor_to_device_fast,
)


def _dna_seq(length: int, seed: int) -> str:
    rng = random.Random(seed)
    alphabet = "ACGT"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _get_backbone_layers(model):
    if hasattr(model, "hyena") and hasattr(model.hyena, "backbone"):
        layers = model.hyena.backbone.layers
        return layers, "model.hyena.backbone.layers"
    raise AttributeError(
        "Expected model.hyena.backbone.layers (HyenaDNA). Got: "
        + str(type(model))
    )


def _max_abs_diff(a, b) -> float:
    import torch

    return float((a.float() - b.float()).abs().max().item())


def _hook_layer_output(model, tokens, layer_idx: int, device: str):
    """Run forward once; return captured tensor from backbone.layers[layer_idx] output."""
    import torch

    captured: dict = {}

    def capture_hook(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    layers, _ = _get_backbone_layers(model)
    layer_mod = layers[layer_idx]
    hook = layer_mod.register_forward_hook(capture_hook)
    tokens = tensor_to_device_fast(tokens, device)
    with torch.no_grad():
        try:
            model(tokens)
        finally:
            hook.remove()
    h = captured.get("hidden")
    if h is None:
        raise RuntimeError("Hook did not capture hidden state.")
    return h


def main() -> None:
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model_id",
        default="LongSafari/hyenadna-large-1m-seqlen-hf",
        help="HuggingFace model id or local path",
    )
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    device = resolve_device_str(args.device)
    configure_cuda_performance()

    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    seq = _dna_seq(args.seq_len, args.seed)
    inputs = tok(seq, return_tensors="pt", add_special_tokens=False)
    if inputs["input_ids"].shape[1] != args.seq_len:
        print(
            f"ERROR: tokenized length {inputs['input_ids'].shape[1]} != --seq_len {args.seq_len}",
            file=sys.stderr,
        )
        sys.exit(1)
    inputs = {k: tensor_to_device_fast(v, device) for k, v in inputs.items()}

    model_enc = AutoModel.from_pretrained(
        args.model_id,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).eval()

    model_lm = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).eval()

    with torch.no_grad():
        out_enc = model_enc(**inputs, output_hidden_states=True, return_dict=True)
        out_lm = model_lm(**inputs, output_hidden_states=True, return_dict=True)

    hs_e = out_enc.hidden_states
    hs_l = out_lm.hidden_states

    print(f"model_id={args.model_id!r}  seq_len={args.seq_len}  device={device}")
    print(f"len(hidden_states) AutoModel={len(hs_e)}  CausalLM={len(hs_l)}")

    if len(hs_e) != len(hs_l):
        print("ERROR: hidden_states tuple lengths differ.", file=sys.stderr)
        sys.exit(2)

    n_layers = len(hs_e)
    worst_cross = 0.0
    for i in range(n_layers):
        d = _max_abs_diff(hs_e[i], hs_l[i])
        worst_cross = max(worst_cross, d)
        ok = np.allclose(
            hs_e[i].float().cpu().numpy(),
            hs_l[i].float().cpu().numpy(),
            rtol=args.rtol,
            atol=args.atol,
        )
        print(f"  [AutoModel vs CausalLM] hidden_states[{i}] max_abs_diff={d:.6e}  allclose={ok}")

    if worst_cross > args.atol * 10:
        print(
            f"WARNING: large cross-model diff (max {worst_cross:.6e}). "
            "Check dtype / identical weights.",
            file=sys.stderr,
        )

    # Hook test: hidden_states[li] vs backbone.layers[li-1] output (Causal LM)
    layers, path = _get_backbone_layers(model_lm)
    n_backbone = len(layers)
    print(f"\nHook path: {path}  (n_layers={n_backbone})")
    print("CausalLM: compare hidden_states[li] to hook output on backbone.layers[li-1]")

    worst_hook = 0.0
    for li in range(1, n_layers):
        if li - 1 >= n_backbone:
            print(f"  skip li={li}: no backbone.layers[{li - 1}]")
            continue
        hooked = _hook_layer_output(model_lm, inputs["input_ids"], li - 1, device)
        ref = hs_l[li]
        d = _max_abs_diff(hooked, ref)
        worst_hook = max(worst_hook, d)
        ok = np.allclose(
            hooked.float().cpu().numpy(),
            ref.float().cpu().numpy(),
            rtol=args.rtol,
            atol=args.atol,
        )
        print(
            f"  li={li}: max_abs_diff(hidden_states[{li}], hook@layers[{li-1}])={d:.6e}  allclose={ok}"
        )

    print("\nDone.")
    if worst_hook > args.atol * 10:
        print(
            f"WARNING: large hook vs hidden_states diff (max {worst_hook:.6e}).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
