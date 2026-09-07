
import torch
import torch.nn as nn

from .neural_pod_mode import SpatialFourierNN, FourierPODMode




class FourierRegimeBasis(nn.Module):
    """Sequential Fourier basis: SpatialFourierNN mean + FourierPODMode list with per-trajectory lambdas."""

    def __init__(
        self,
        d_x: int,
        M: int,
        quad_weights: torch.Tensor,
        hidden_dim: int = 128,
        num_frequencies: int = 16,
        scale: float = 10.0,
        scales: list[float] | None = None,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.M = M
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        self.scale = scale
        self.scales = scales
        self.n_layers = n_layers

        self.mean_net = SpatialFourierNN(d_x, hidden_dim, num_frequencies, scale, scales, n_layers)
        self.modes = nn.ModuleList()
        self.register_buffer("quad_weights", quad_weights)

    @property
    def num_modes(self) -> int:
        return len(self.modes)

    def forward(self, x: torch.Tensor, t=None, kappa=None) -> torch.Tensor:
        """Returns (N, Ny) reconstruction: mean_net(x) + sum of mode outer products."""
        N = self.modes[0].lambda_ten.shape[0] if self.modes else 1
        out = self.mean_net(x).unsqueeze(0).expand(N, -1)
        for mode in self.modes:
            phi = mode.phi(x)
            out = out + torch.outer(phi, mode.lambda_ten).T
        return out

    def add_mode(self) -> FourierPODMode:
        mode = FourierPODMode(
            self.d_x, self.M, self.hidden_dim, self.num_frequencies,
            self.scale, self.scales, self.n_layers,
        )
        mode = mode.to(self.quad_weights.device)
        self.modes.append(mode)
        return mode
