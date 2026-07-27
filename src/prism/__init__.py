"""PRISM: merge-then-quantize comparison of LoRA and SVMO on Qwen2.5-7B-Instruct.

Package layout:
    config       -- paths and experiment constants, overridable via environment variables
    adapters     -- pure merge math for SVMO and LoRA (no I/O)
    metrics      -- weight-distribution statistics and fake quantization (no I/O)
    checkpoints  -- loading base weights, SVD factors, and adapter checkpoints from disk
    dataset      -- held-out evaluation batches
    streaming    -- layer-major forward pass shared by every measurement script

Setting ``PRISM_DISABLE_MKLDNN=1`` disables PyTorch's oneDNN backend, needed
on CPUs without AVX2 (it otherwise raises ``SIGILL``, not a normal exception).
"""

import os

if os.environ.get("PRISM_DISABLE_MKLDNN"):
    import torch
    torch.backends.mkldnn.enabled = False
