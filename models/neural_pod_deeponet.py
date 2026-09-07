
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .regime_basis import FourierRegimeBasis
from .pod_deeponet import BranchNet


@dataclass


class NeuralPODDeepONet(nn.Module):
    """NeuralPOD-DeepONet: mean_net(x,t) + branch(u0) @ Phi(x,t).T with K frozen Fourier modes."""

    def __init__(self, basis: FourierRegimeBasis, branch: BranchNet) -> None:
        super().__init__()
        self.basis = basis
        self.branch = branch
        for p in basis.parameters():
            p.requires_grad_(False)

    @property
    def K(self) -> int:
        return len(self.basis.modes)

    def forward(self, u0: Tensor, x_flat: Tensor) -> Tensor:
        """Returns (N, Nt*Nx) predicted field. u0: (N, m), x_flat: (Nt*Nx, 2)."""
        beta = self.branch(u0)
        phi = torch.stack([m.phi(x_flat) for m in self.basis.modes], dim=1)
        mean = self.basis.mean_net(x_flat)
        return mean.unsqueeze(0) + beta @ phi.T


