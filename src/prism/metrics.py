"""Weight-distribution statistics and simulated post-training quantization.

Pure numeric functions: given a weight matrix, they return numbers or a
quantized matrix. No file I/O.
"""

import numpy as np
import torch


def per_channel_stats(W: torch.Tensor) -> dict:
    """Excess (Fisher) kurtosis and max-to-standard-deviation ratio, per output
    channel (row), averaged over all rows of ``W``.

    Both statistics are near zero/moderate for a Gaussian row and grow when a
    few entries in the row sit far from the rest -- the property that
    per-channel quantizers such as GPTQ and AWQ calibrate against.
    """
    W_np = W.numpy().astype(np.float64)
    mean = W_np.mean(axis=1, keepdims=True)
    std = W_np.std(axis=1, keepdims=True) + 1e-12
    z = (W_np - mean) / std
    kurtosis = (z ** 4).mean(axis=1) - 3.0
    max_std_ratio = np.abs(W_np).max(axis=1) / std.squeeze(-1)
    return {
        "mean_kurtosis": float(np.mean(kurtosis)),
        "mean_max_std_ratio": float(np.mean(max_std_ratio)),
        "frob_norm": float(np.linalg.norm(W_np)),
    }


def fake_quantize(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel quantize-dequantize round-trip.

    Rounds and immediately dequantizes so all subsequent computation stays in
    full precision; isolates quantization error from quantization speed and
    requires no low-bit compute kernel.
    """
    q_max = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / q_max
    q = torch.clamp(torch.round(W / scale), -q_max - 1, q_max)
    return q * scale
