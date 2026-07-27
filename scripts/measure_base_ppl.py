#!/usr/bin/env python
"""Held-out perplexity of the frozen base model, no adapter merged.

Baseline for the adapter deltas measured by ``quantize_and_measure_ppl.py``,
on the same held-out set and the same streaming forward pass.

Usage:
    python scripts/measure_base_ppl.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prism.config import MODEL_NAME, N_LAYERS, RESULTS_DIR
from prism.dataset import held_out_batches
from prism.streaming import run_layerwise_perplexity


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sequences = held_out_batches(tokenizer)
    print(f"[prism] held-out set: {len(sequences)} sequences, "
          f"lengths {[s.shape[1] for s in sequences]}", flush=True)

    def identity_weight(cfg, hf_sub, dir_name, layer_idx, base_W):
        return base_W

    def on_layer_done(layer_idx, elapsed):
        print(f"[prism] base: layer {layer_idx + 1}/{N_LAYERS} done ({elapsed:.0f}s)", flush=True)

    results = run_layerwise_perplexity(
        sequences, configs=["base"], weight_fn=identity_weight,
        n_layers=N_LAYERS, on_layer_done=on_layer_done,
    )
    ppl = results["base"]["perplexity"]
    tokens = results["base"]["total_tokens"]
    print(f"[prism] base_ppl={ppl:.4f} tokens={tokens}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "base_ppl.json")
    with open(out_path, "w") as f:
        json.dump({"perplexity": ppl, "total_tokens": tokens}, f, indent=2)
    print(f"[prism] wrote {out_path}")


if __name__ == "__main__":
    main()
