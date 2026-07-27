#!/usr/bin/env python
"""Held-out perplexity before and after quantizing a merged adapter.

For the requested adapter, merges it into the frozen base weights, then
measures perplexity both in full precision and after a simulated
quantize-dequantize round-trip at the requested bit-width, on the same
held-out set and streaming forward pass used throughout this project.

Usage:
    python scripts/quantize_and_measure_ppl.py --variant svmo --bits 4
    python scripts/quantize_and_measure_ppl.py --variant lora --bits 4
(run one variant at a time; running both in the same process doubles peak
memory for no benefit, since each needs its own checkpoint resident)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prism.adapters import svmo_delta, lora_delta
from prism.checkpoints import (
    load_svd_factors, load_svmo_checkpoint, load_lora_checkpoint,
    svmo_mlp_state, lora_matrices,
)
from prism.config import MODEL_NAME, N_LAYERS, RESULTS_DIR
from prism.dataset import held_out_batches
from prism.metrics import fake_quantize
from prism.streaming import run_layerwise_perplexity


def make_weight_fn(variant: str, bits: int, ckpt_state: dict):
    """Build the per-(config, layer, matrix) weight function for one adapter.

    ``variant`` selects which checkpoint's delta to compute; the returned
    function itself decides whether to quantize based on which of the two
    configs (fp32 or q<bits>) is being requested.
    """
    def weight_fn(cfg, hf_sub, dir_name, layer_idx, base_W):
        if variant == "svmo":
            U_k, S_k, Vt_k = load_svd_factors(layer_idx, dir_name)
            delta = svmo_delta(S_k, U_k, Vt_k, svmo_mlp_state(ckpt_state, layer_idx, hf_sub))
        else:
            A, B = lora_matrices(ckpt_state, layer_idx, hf_sub)
            delta = lora_delta(A, B)
        merged = (base_W + delta).to(base_W.dtype)
        if cfg.endswith(f"q{bits}"):
            merged = fake_quantize(merged, bits=bits)
        return merged
    return weight_fn


def run_variant(variant: str, bits: int, n_layers: int = N_LAYERS) -> dict:
    from transformers import AutoTokenizer

    print(f"[prism] variant={variant} bits={bits}: loading tokenizer + checkpoint", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ckpt_state = load_svmo_checkpoint() if variant == "svmo" else load_lora_checkpoint()

    sequences = held_out_batches(tokenizer)
    print(f"[prism] held-out set: {len(sequences)} sequences, "
          f"lengths {[s.shape[1] for s in sequences]}", flush=True)

    configs = [f"{variant}_fp32", f"{variant}_q{bits}"]
    weight_fn = make_weight_fn(variant, bits, ckpt_state)

    def on_layer_done(layer_idx, elapsed):
        print(f"[prism] {variant}: layer {layer_idx + 1}/{n_layers} done ({elapsed:.0f}s)", flush=True)

    results = run_layerwise_perplexity(
        sequences, configs=configs, weight_fn=weight_fn,
        n_layers=n_layers, on_layer_done=on_layer_done,
    )
    elapsed = results.pop("_elapsed_seconds")
    for cfg in configs:
        r = results[cfg]
        print(f"[prism] {cfg}: ppl={r['perplexity']:.4f} tokens={r['total_tokens']}", flush=True)

    results["_meta"] = {
        "variant": variant, "model": MODEL_NAME, "n_layers": n_layers, "quant_bits": bits,
        "n_heldout": len(sequences), "elapsed_seconds": elapsed,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"quant_ppl_{variant}_{bits}bit.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[prism] wrote {out_path}", flush=True)

    delta = results[f"{variant}_q{bits}"]["perplexity"] - results[f"{variant}_fp32"]["perplexity"]
    print(f"[prism] delta-ppl {variant} (fp32 -> int{bits}): {delta:+.4f}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["svmo", "lora"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    args = parser.parse_args()
    run_variant(args.variant, args.bits, args.n_layers)
