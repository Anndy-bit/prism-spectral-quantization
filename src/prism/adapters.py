"""Merge math for the two adapters compared in this paper.

Every function here is pure: given tensors, it returns tensors. Loading those
tensors from disk is the responsibility of :mod:`prism.checkpoints`.
"""

import torch
import torch.nn.functional as F

from prism.config import SVMO_ALPHA, SVMO_EPS, LORA_SCALING


class ModulationMLP(torch.nn.Module):
    """SVMO's pointwise modulation network.

    Must match ``src/adapters/svmo_linear.py::ModulationMLP`` in the
    lowrank-field-adapters repository exactly, since it loads that
    checkpoint's trained weights directly.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


def svmo_delta(
    S_k: torch.Tensor, U_k: torch.Tensor, Vt_k: torch.Tensor, mlp_state: dict
) -> torch.Tensor:
    """SVMO's merge delta: ``U_k @ diag(alpha * sigma * tanh(g(log sigma))) @ Vt_k``."""
    mlp = ModulationMLP(hidden_dim=mlp_state["fc1.weight"].shape[0])
    mlp.load_state_dict(mlp_state)
    mlp.eval()
    with torch.no_grad():
        s_log = torch.log(S_k + SVMO_EPS).unsqueeze(-1).float()
        g = mlp(s_log).squeeze(-1)
        d_sigma = S_k * SVMO_ALPHA * torch.tanh(g)
        return U_k @ torch.diag(d_sigma) @ Vt_k


def lora_delta(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """LoRA's merge delta: ``(alpha / r) * B @ A``."""
    return LORA_SCALING * (B.float() @ A.float())


def magnitude_matched_lora_delta(d_lora_raw: torch.Tensor, d_svmo: torch.Tensor) -> torch.Tensor:
    """Rescale a trained LoRA delta to exactly match SVMO's Frobenius norm.

    Isotropic rescaling: direction is preserved, only magnitude changes. Used
    by the magnitude-matched control to separate the effect of update size
    from the effect of update geometry.
    """
    svmo_norm = torch.linalg.norm(d_svmo)
    lora_norm = torch.linalg.norm(d_lora_raw) + 1e-12
    return d_lora_raw * (svmo_norm / lora_norm)
