"""
PRISM — Nivel 1, paso 3: análisis de kurtosis / outlier-ratio por canal.

Compara, para cada matriz de peso adaptada de Qwen2.5-7B-Instruct:
  - W               (peso base congelado, sin adaptar)
  - W + Delta_SVMO  (fusión espectral acotada, de S3, checkpoint s3_full)
  - W + Delta_LoRA  (fusión aditiva sin cota, checkpoint lora_r8)

Metrica por canal de salida (fila de W, eje que usan los cuantizadores
per-channel como GPTQ/AWQ/Q4_K_M): kurtosis exceso (Fisher) y ratio max/std.
Kurtosis alta / ratio alto = distribucion mas pesada en colas = mas dificil
de cuantizar sin perder precision.

Cero GPU. Una matriz en memoria a la vez (lazy safetensors + descarte
inmediato) para correr dentro de los ~5GB de RAM disponibles.

Uso:
    ./.venv/bin/python analyze_merge_outliers.py
(ejecutar con el venv de lowrank-field-adapters, que ya tiene torch/safetensors)
"""

import csv
import glob
import json
import os
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

# ---------------------------------------------------------------------------
# Rutas (ajustar solo si el repo se mueve)
# ---------------------------------------------------------------------------
LFA_REPO = "/home/gula/Documentos/Proyectos Github/lowrank-field-adapters"
SVD_DIR = os.path.join(LFA_REPO, "svd_factors")
S3_CKPT_PATH = os.path.join(
    LFA_REPO, "results/ablations_baselines_20260719_193209/s3_full/adapters.pt"
)
LORA_CKPT_PATH = os.path.join(
    LFA_REPO, "results/ablations_baselines_20260719_193209/lora_r8/adapters.pt"
)
HF_HUB_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
)
OUT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "outlier_stats.csv",
)

N_LAYERS = 28
MATRICES = [
    ("self_attn.q_proj", "q_proj"),
    ("self_attn.k_proj", "k_proj"),
    ("self_attn.v_proj", "v_proj"),
    ("self_attn.o_proj", "o_proj"),
    ("mlp.gate_proj", "gate_proj"),
    ("mlp.up_proj", "up_proj"),
    ("mlp.down_proj", "down_proj"),
]

SVMO_ALPHA = 0.3
SVMO_EPS = 1e-8
LORA_R = 8
LORA_ALPHA = 16.0
LORA_SCALING = LORA_ALPHA / LORA_R


def find_hf_snapshot_dir() -> str:
    snaps = glob.glob(os.path.join(HF_HUB_DIR, "snapshots", "*"))
    if not snaps:
        raise FileNotFoundError(f"No hay snapshot de Qwen2.5-7B-Instruct en {HF_HUB_DIR}")
    return snaps[0]


def build_weight_map(snapshot_dir: str) -> dict:
    with open(os.path.join(snapshot_dir, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    return idx["weight_map"]


def load_base_weight(snapshot_dir: str, weight_map: dict, hf_name: str) -> torch.Tensor:
    """Carga UNA sola matriz del shard correspondiente, sin tocar el resto."""
    shard = weight_map[hf_name]
    shard_path = os.path.join(snapshot_dir, shard)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(hf_name).float()  # -> fp32 para las estadisticas


def load_svd_factors(layer_idx: int, matrix_dir_name: str):
    d = os.path.join(SVD_DIR, f"layer_{layer_idx}.{matrix_dir_name}")
    U_k = load_safetensors(os.path.join(d, "U_k.safetensors"))["U_k"].float()
    S_k = load_safetensors(os.path.join(d, "S_k.safetensors"))["S_k"].float()
    Vt_k = load_safetensors(os.path.join(d, "Vt_k.safetensors"))["Vt_k"].float()
    return U_k, S_k, Vt_k


class ModulationMLP(torch.nn.Module):
    """Debe coincidir exactamente con src/adapters/svmo_linear.py::ModulationMLP."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.fc1(x))
        x = torch.nn.functional.gelu(self.fc2(x))
        return self.fc3(x)


def svmo_delta(S_k: torch.Tensor, U_k: torch.Tensor, Vt_k: torch.Tensor,
               mlp_state: dict) -> torch.Tensor:
    mlp = ModulationMLP(hidden_dim=mlp_state["fc1.weight"].shape[0])
    mlp.load_state_dict(mlp_state)
    mlp.eval()
    with torch.no_grad():
        s_log = torch.log(S_k + SVMO_EPS).unsqueeze(-1).float()
        g = mlp(s_log).squeeze(-1)
        dsig = S_k * SVMO_ALPHA * torch.tanh(g)          # [k]
        delta = U_k @ torch.diag(dsig) @ Vt_k             # [d_out, d_in]
    return delta


def lora_delta(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return LORA_SCALING * (B.float() @ A.float())


def per_channel_stats(W: torch.Tensor) -> dict:
    """Kurtosis (exceso, Fisher) y ratio max/std por fila (canal de salida)."""
    Wn = W.numpy().astype(np.float64)
    mean = Wn.mean(axis=1, keepdims=True)
    std = Wn.std(axis=1, keepdims=True) + 1e-12
    z = (Wn - mean) / std
    kurt = (z ** 4).mean(axis=1) - 3.0                    # exceso de kurtosis
    max_abs = np.abs(Wn).max(axis=1)
    max_std_ratio = max_abs / std.squeeze(-1)
    return {
        "mean_kurtosis": float(np.mean(kurt)),
        "mean_max_std_ratio": float(np.mean(max_std_ratio)),
        "frob_norm": float(np.linalg.norm(Wn)),
    }


def main():
    snapshot_dir = find_hf_snapshot_dir()
    weight_map = build_weight_map(snapshot_dir)
    print(f"[PRISM] snapshot: {snapshot_dir}")

    print(f"[PRISM] cargando checkpoint S3 (s3_full): {S3_CKPT_PATH}")
    s3_state = torch.load(S3_CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"[PRISM] cargando checkpoint LoRA (lora_r8): {LORA_CKPT_PATH}")
    lora_state = torch.load(LORA_CKPT_PATH, map_location="cpu", weights_only=False)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "layer", "matrix", "variant",
        "mean_kurtosis", "mean_max_std_ratio", "frob_norm",
        "delta_frob_over_W_frob",
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
            base_stats = per_channel_stats(W)
            writer.writerow({
                "layer": layer_idx, "matrix": dir_name, "variant": "base",
                "delta_frob_over_W_frob": 0.0, **base_stats,
            })

            # ---- SVMO merge ----
            U_k, S_k, Vt_k = load_svd_factors(layer_idx, dir_name)
            mlp_prefix = f"model.layers.{layer_idx}.{hf_sub}.modulation."
            mlp_state = {
                k[len(mlp_prefix):]: v
                for k, v in s3_state.items() if k.startswith(mlp_prefix)
            }
            d_svmo = svmo_delta(S_k, U_k, Vt_k, mlp_state)
            W_svmo = W + d_svmo
            svmo_stats = per_channel_stats(W_svmo)
            svmo_stats["delta_frob_over_W_frob"] = float(
                torch.linalg.norm(d_svmo) / (torch.linalg.norm(W) + 1e-12)
            )
            writer.writerow({"layer": layer_idx, "matrix": dir_name, "variant": "svmo", **svmo_stats})
            del U_k, Vt_k, d_svmo, W_svmo

            # ---- LoRA merge ----
            A = lora_state[f"model.layers.{layer_idx}.{hf_sub}.lora_A"]
            B = lora_state[f"model.layers.{layer_idx}.{hf_sub}.lora_B"]
            d_lora = lora_delta(A, B)
            W_lora = W + d_lora
            lora_stats = per_channel_stats(W_lora)
            lora_stats["delta_frob_over_W_frob"] = float(
                torch.linalg.norm(d_lora) / (torch.linalg.norm(W) + 1e-12)
            )
            writer.writerow({"layer": layer_idx, "matrix": dir_name, "variant": "lora", **lora_stats})
            del A, B, d_lora, W_lora, W

            n_done += 1
            f_out.flush()
            if n_done % 20 == 0 or n_done == n_total:
                elapsed = time.time() - t0
                print(f"[PRISM] {n_done}/{n_total} matrices ({elapsed:.1f}s)")

    f_out.close()
    print(f"[PRISM] listo -> {OUT_CSV}")


if __name__ == "__main__":
    main()
