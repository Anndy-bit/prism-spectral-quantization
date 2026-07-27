"""Reading base weights, SVD factors, and adapter checkpoints from disk.

Every function here does I/O and returns tensors; no merge math or metrics
live in this module.
"""

import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

from prism.config import HF_HUB_DIR, SVD_DIR, S3_CKPT_PATH, LORA_CKPT_PATH


def find_hf_snapshot_dir() -> str:
    snapshots = glob.glob(os.path.join(HF_HUB_DIR, "snapshots", "*"))
    if not snapshots:
        raise FileNotFoundError(f"no snapshot found under {HF_HUB_DIR}")
    return snapshots[0]


def build_weight_map(snapshot_dir: str) -> dict:
    with open(os.path.join(snapshot_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    return index["weight_map"]


def load_base_weight(snapshot_dir: str, weight_map: dict, hf_name: str) -> torch.Tensor:
    """Load a single weight matrix from its shard, without touching the rest."""
    shard_path = os.path.join(snapshot_dir, weight_map[hf_name])
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(hf_name).float()


def load_svd_factors(layer_idx: int, matrix_dir_name: str):
    """Load the rank-k SVD factors precomputed for one weight matrix."""
    d = os.path.join(SVD_DIR, f"layer_{layer_idx}.{matrix_dir_name}")
    U_k = load_safetensors(os.path.join(d, "U_k.safetensors"))["U_k"].float()
    S_k = load_safetensors(os.path.join(d, "S_k.safetensors"))["S_k"].float()
    Vt_k = load_safetensors(os.path.join(d, "Vt_k.safetensors"))["Vt_k"].float()
    return U_k, S_k, Vt_k


def load_svmo_checkpoint(path: str = S3_CKPT_PATH) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_lora_checkpoint(path: str = LORA_CKPT_PATH) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def svmo_mlp_state(ckpt_state: dict, layer_idx: int, hf_sub: str) -> dict:
    """Extract one layer/matrix's modulation-MLP weights from the full checkpoint."""
    prefix = f"model.layers.{layer_idx}.{hf_sub}.modulation."
    return {k[len(prefix):]: v for k, v in ckpt_state.items() if k.startswith(prefix)}


def lora_matrices(ckpt_state: dict, layer_idx: int, hf_sub: str):
    """Extract one layer/matrix's LoRA A and B factors from the full checkpoint."""
    prefix = f"model.layers.{layer_idx}.{hf_sub}"
    return ckpt_state[f"{prefix}.lora_A"], ckpt_state[f"{prefix}.lora_B"]
