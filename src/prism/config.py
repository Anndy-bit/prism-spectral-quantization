"""Paths and experiment constants, in one place.

Every path can be overridden with an environment variable so the scripts do
not assume a specific machine or username. Defaults match the layout used to
produce the results reported in ``paper/main.tex``.
"""

import os

# --- Repository locations ---------------------------------------------------

# PRISM reuses checkpoints and streaming infrastructure from the sibling
# lowrank-field-adapters repository rather than duplicating them.
LFA_REPO = os.environ.get(
    "PRISM_LFA_REPO",
    os.path.expanduser("~/Documentos/Proyectos Github/lowrank-field-adapters"),
)

PRISM_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SVD_DIR = os.environ.get("PRISM_SVD_DIR", os.path.join(LFA_REPO, "svd_factors"))

_CKPT_RUN = os.environ.get(
    "PRISM_CKPT_RUN", "results/ablations_baselines_20260719_193209"
)
S3_CKPT_PATH = os.environ.get(
    "PRISM_S3_CKPT", os.path.join(LFA_REPO, _CKPT_RUN, "s3_full", "adapters.pt")
)
LORA_CKPT_PATH = os.environ.get(
    "PRISM_LORA_CKPT", os.path.join(LFA_REPO, _CKPT_RUN, "lora_r8", "adapters.pt")
)

HF_HUB_DIR = os.path.expanduser(
    os.environ.get(
        "PRISM_HF_HUB_DIR",
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct",
    )
)

RESULTS_DIR = os.environ.get("PRISM_RESULTS_DIR", os.path.join(PRISM_REPO, "results"))
LOG_DIR = os.environ.get("PRISM_LOG_DIR", os.path.join(PRISM_REPO, "results", "logs"))

# --- Model and experiment constants -----------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
N_LAYERS = 28
N_HELDOUT = 16
MAX_SEQ = 384

# (HF submodule path, short name used in filenames and CSV columns)
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
LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_SCALING = LORA_ALPHA / LORA_RANK
