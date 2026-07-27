#!/usr/bin/env python
"""Per-channel weight-distribution statistics for the base, SVMO-merged, and
LoRA-merged weight matrices.

Writes ``results/outlier_stats.csv``: one row per (layer, matrix, variant),
with mean kurtosis, mean max/std ratio, and the merge's Frobenius norm
relative to the base weight's.

Usage:
    python scripts/analyze_merge_outliers.py
"""

import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prism.adapters import svmo_delta, lora_delta
from prism.checkpoints import (
    find_hf_snapshot_dir, build_weight_map, load_base_weight, load_svd_factors,
    load_svmo_checkpoint, load_lora_checkpoint, svmo_mlp_state, lora_matrices,
)
from prism.config import N_LAYERS, MATRICES, RESULTS_DIR
from prism.metrics import per_channel_stats

OUT_CSV = os.path.join(RESULTS_DIR, "outlier_stats.csv")

FIELDNAMES = [
    "layer", "matrix", "variant",
    "mean_kurtosis", "mean_max_std_ratio", "frob_norm",
    "delta_frob_over_W_frob",
]


def main():
    snapshot_dir = find_hf_snapshot_dir()
    weight_map = build_weight_map(snapshot_dir)
    svmo_state = load_svmo_checkpoint()
    lora_state = load_lora_checkpoint()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
        writer.writeheader()

        t0 = time.time()
        n_total = N_LAYERS * len(MATRICES)
        for n_done, (layer_idx, (hf_sub, dir_name)) in enumerate(
            ((l, m) for l in range(N_LAYERS) for m in MATRICES), start=1
        ):
            hf_name = f"model.layers.{layer_idx}.{hf_sub}.weight"
            W = load_base_weight(snapshot_dir, weight_map, hf_name)
            writer.writerow({
                "layer": layer_idx, "matrix": dir_name, "variant": "base",
                "delta_frob_over_W_frob": 0.0, **per_channel_stats(W),
            })

            U_k, S_k, Vt_k = load_svd_factors(layer_idx, dir_name)
            d_svmo = svmo_delta(S_k, U_k, Vt_k, svmo_mlp_state(svmo_state, layer_idx, hf_sub))
            svmo_stats = per_channel_stats(W + d_svmo)
            svmo_stats["delta_frob_over_W_frob"] = float(
                torch.linalg.norm(d_svmo) / (torch.linalg.norm(W) + 1e-12)
            )
            writer.writerow({"layer": layer_idx, "matrix": dir_name, "variant": "svmo", **svmo_stats})

            A, B = lora_matrices(lora_state, layer_idx, hf_sub)
            d_lora = lora_delta(A, B)
            lora_stats = per_channel_stats(W + d_lora)
            lora_stats["delta_frob_over_W_frob"] = float(
                torch.linalg.norm(d_lora) / (torch.linalg.norm(W) + 1e-12)
            )
            writer.writerow({"layer": layer_idx, "matrix": dir_name, "variant": "lora", **lora_stats})

            f_out.flush()
            if n_done % 20 == 0 or n_done == n_total:
                print(f"[prism] {n_done}/{n_total} matrices ({time.time()-t0:.1f}s)", flush=True)

    print(f"[prism] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
