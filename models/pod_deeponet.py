

import torch
import torch.nn as nn
from torch import Tensor

from .pod import PODBasis
from .neural_pod_mode import _mlp


class BranchNet(nn.Module):
    """Maps (u0_sensors, kappa) to P mode coefficients."""

    def __init__(self, m: int, P: int, hidden_dim: int = 128, n_layers: int = 4,
                 d_kappa: int = 0) -> None:
        super().__init__()
        self.d_kappa = d_kappa
        self.net = _mlp(m + d_kappa, P, hidden_dim, n_layers, act=nn.Tanh)

    def forward(self, u0: Tensor, kappa: Tensor = None) -> Tensor:
        """Returns (N, P) mode coefficients."""
        inp = torch.cat([u0, kappa], dim=-1) if self.d_kappa > 0 else u0
        return self.net(inp)


class PODDeepONet(nn.Module):
    """POD-DeepONet: mean + branch(u0) @ Phi.T (Lu et al., 2022)."""

    def __init__(self, basis: PODBasis, branch: BranchNet) -> None:
        super().__init__()
        self.basis = basis
        self.branch = branch
        for p in basis.parameters():
            p.requires_grad_(False)

    @property
    def P(self) -> int:
        return self.basis.num_modes

    def forward(self, u0: Tensor, kappa: Tensor = None) -> Tensor:
        """Returns (N, Nt*Nx) predicted spatiotemporal field."""
        beta = self.branch(u0, kappa)
        return self.basis.mean + beta @ self.basis.modes.T


