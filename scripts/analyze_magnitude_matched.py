"""
PRISM — cierre del matiz de magnitud.

El paso anterior (analyze_merge_outliers.py) encontro que LoRA mueve mucho
mas las estadisticas de peso que SVMO, pero el delta de LoRA tambien es en
promedio 47x mas grande en norma de Frobenius. Este script aisla el efecto
del TIPO de actualizacion escalando el delta de LoRA, capa por capa, para
que tenga la MISMA norma de Frobenius que el delta de SVMO en esa matriz.
Si el efecto (mayor disrupcion de kurtosis/max-std) sobrevive tras igualar
la magnitud, es evidencia de que es el tipo de update lo que importa, no
solo su tamano. Si desaparece, el resultado original era solo un artefacto
de escala.

Cero GPU, una matriz a la vez, misma huella de memoria que el script previo.
"""

import csv
import os
import time

import numpy as np
import torch

from analyze_merge_outliers import (
    N_LAYERS, MATRICES, S3_CKPT_PATH, LORA_CKPT_PATH,
    find_hf_snapshot_dir, build_weight_map, load_base_weight,
    load_svd_factors, svmo_delta, lora_delta, per_channel_stats,
)

OUT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "magnitude_matched_stats.csv",
)


def main():
    snapshot_dir = find_hf_snapshot_dir()
    weight_map = build_weight_map(snapshot_dir)
    print(f"[PRISM] snapshot: {snapshot_dir}")

    s3_state = torch.load(S3_CKPT_PATH, map_location="cpu", weights_only=False)
    lora_state = torch.load(LORA_CKPT_PATH, map_location="cpu", weights_only=False)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "layer", "matrix",
        "frob_svmo_over_W", "frob_lora_raw_over_W", "frob_lora_scaled_over_W",
        "kurt_base", "kurt_svmo", "kurt_lora_raw", "kurt_lora_scaled",
        "maxstd_base", "maxstd_svmo", "maxstd_lora_raw", "maxstd_lora_scaled",
    ]
    f_out = open(OUT_CSV, "w", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    t0 = time.time()
    n_done = 0
    n_total = N_LAYERS * len(MATRICES)

    for layer_idx in range(N_LAYERS):
        for hf_sub, dir_name in MATRICES:
            hf_name = f"model.layers.{layer_idx}.{hf_sub}.weight"
            W = load_base_weight(snapshot_dir, weight_map, hf_name)
            W_norm = torch.linalg.norm(W) + 1e-12
            base_stats = per_channel_stats(W)

            U_k, S_k, Vt_k = load_svd_factors(layer_idx, dir_name)
            mlp_prefix = f"model.layers.{layer_idx}.{hf_sub}.modulation."
            mlp_state = {
                k[len(mlp_prefix):]: v
                for k, v in s3_state.items() if k.startswith(mlp_prefix)
            }
            d_svmo = svmo_delta(S_k, U_k, Vt_k, mlp_state)
            svmo_norm = torch.linalg.norm(d_svmo)
            svmo_stats = per_channel_stats(W + d_svmo)

            A = lora_state[f"model.layers.{layer_idx}.{hf_sub}.lora_A"]
            B = lora_state[f"model.layers.{layer_idx}.{hf_sub}.lora_B"]
            d_lora_raw = lora_delta(A, B)
            lora_raw_norm = torch.linalg.norm(d_lora_raw)
            lora_raw_stats = per_channel_stats(W + d_lora_raw)

            # Escalar el delta de LoRA para que tenga la MISMA norma que el de SVMO,
            # preservando su direccion (misma "forma" de update, magnitud igualada).
            scale = (svmo_norm / (lora_raw_norm + 1e-12)).item()
            d_lora_scaled = d_lora_raw * scale
            lora_scaled_stats = per_channel_stats(W + d_lora_scaled)

            writer.writerow({
                "layer": layer_idx, "matrix": dir_name,
                "frob_svmo_over_W": float(svmo_norm / W_norm),
                "frob_lora_raw_over_W": float(lora_raw_norm / W_norm),
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
            del W, d_svmo, d_lora_raw, d_lora_scaled, U_k, Vt_k, A, B

            n_done += 1
            if n_done % 20 == 0 or n_done == n_total:
                print(f"[PRISM] {n_done}/{n_total} ({time.time()-t0:.1f}s)")

    f_out.close()
    print(f"[PRISM] listo -> {OUT_CSV}")


if __name__ == "__main__":
    main()
