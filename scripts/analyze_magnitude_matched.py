#!/usr/bin/env python
"""Magnitude-matched control: rescale LoRA's merge delta to SVMO's Frobenius
norm, per matrix, and re-measure weight-distribution statistics.

Separates the effect of update *geometry* from update *size*: the raw
comparison in ``outlier_stats.csv`` confounds the two, since LoRA's trained
update is much larger than SVMO's in every matrix. Writes
``results/magnitude_matched_stats.csv``.

Usage:
    python scripts/analyze_magnitude_matched.py
"""

import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prism.adapters import svmo_delta, lora_delta, magnitude_matched_lora_delta
from prism.checkpoints import (
    find_hf_snapshot_dir, build_weight_map, load_base_weight, load_svd_factors,
    load_svmo_checkpoint, load_lora_checkpoint, svmo_mlp_state, lora_matrices,
)
from prism.config import N_LAYERS, MATRICES, RESULTS_DIR
from prism.metrics import per_channel_stats

OUT_CSV = os.path.join(RESULTS_DIR, "magnitude_matched_stats.csv")

FIELDNAMES = [
    "layer", "matrix",
    "frob_svmo_over_W", "frob_lora_raw_over_W", "frob_lora_scaled_over_W",
    "kurt_base", "kurt_svmo", "kurt_lora_raw", "kurt_lora_scaled",
    "maxstd_base", "maxstd_svmo", "maxstd_lora_raw", "maxstd_lora_scaled",
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
            W_norm = torch.linalg.norm(W) + 1e-12
            base_stats = per_channel_stats(W)

            U_k, S_k, Vt_k = load_svd_factors(layer_idx, dir_name)
            d_svmo = svmo_delta(S_k, U_k, Vt_k, svmo_mlp_state(svmo_state, layer_idx, hf_sub))
            svmo_stats = per_channel_stats(W + d_svmo)

            A, B = lora_matrices(lora_state, layer_idx, hf_sub)
            d_lora_raw = lora_delta(A, B)
            lora_raw_stats = per_channel_stats(W + d_lora_raw)

            d_lora_scaled = magnitude_matched_lora_delta(d_lora_raw, d_svmo)
            lora_scaled_stats = per_channel_stats(W + d_lora_scaled)

            writer.writerow({
                "layer": layer_idx, "matrix": dir_name,
                "frob_svmo_over_W": float(torch.linalg.norm(d_svmo) / W_norm),
                "frob_lora_raw_over_W": float(torch.linalg.norm(d_lora_raw) / W_norm),
                "frob_lora_scaled_over_W": float(torch.linalg.norm(d_lora_scaled) / W_norm),
                "kurt_base": base_stats["mean_kurtosis"],
                "kurt_svmo": svmo_stats["mean_kurtosis"],
                "kurt_lora_raw": lora_raw_stats["mean_kurtosis"],
                "kurt_lora_scaled": lora_scaled_stats["mean_kurtosis"],
                "maxstd_base": base_stats["mean_max_std_ratio"],
                "maxstd_svmo": svmo_stats["mean_max_std_ratio"],
                "maxstd_lora_raw": lora_raw_stats["mean_max_std_ratio"],
                "maxstd_lora_scaled": lora_scaled_stats["mean_max_std_ratio"],
            })
            f_out.flush()

            if n_done % 20 == 0 or n_done == n_total:
                print(f"[prism] {n_done}/{n_total} matrices ({time.time()-t0:.1f}s)", flush=True)

    print(f"[prism] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
