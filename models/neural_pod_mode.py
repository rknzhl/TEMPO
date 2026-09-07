
import math
import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int,
          act: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    """Fully connected network with uniform hidden width."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)






class SpatialFourierNN(nn.Module):
    """Spatial basis via Random Fourier Features; multi-scale if `scales` is given."""

    def __init__(self, d_x: int = 1, hidden_dim: int = 128,
                 num_frequencies: int = 16, scale: float = 10.0,
                 scales: list[float] | None = None,
                 n_layers: int = 2) -> None:
        super().__init__()
        if scales is not None:
            k = num_frequencies // len(scales)
            Bs = [torch.randn(k, d_x) * s for s in scales[:-1]]
            # last scale takes the remainder so the total equals num_frequencies
            Bs.append(torch.randn(num_frequencies - k * (len(scales) - 1), d_x) * scales[-1])
            B = torch.cat(Bs, dim=0)
        else:
            B = torch.randn(num_frequencies, d_x) * scale
        self.register_buffer("B", B, persistent=False)

        layers: list[nn.Module] = [nn.Linear(2 * num_frequencies, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (Ny,) basis values."""
        x_proj = 2 * math.pi * x @ self.B.to(x.device).T
        feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(feats).squeeze(-1)








class FourierPODMode(nn.Module):
    """Fourier spatial basis with per-trajectory learnable amplitude coefficients."""

    def __init__(self, d_x: int, M: int, hidden_dim: int = 128,
                 num_frequencies: int = 16, scale: float = 10.0,
                 scales: list[float] | None = None,
                 n_layers: int = 2) -> None:
        super().__init__()
        self.phi = SpatialFourierNN(d_x, hidden_dim, num_frequencies, scale, scales, n_layers)
        self.lambda_ten = nn.Parameter(torch.ones(M))

    def forward(self, x: torch.Tensor, t=None, kappa=None) -> torch.Tensor:
        return torch.outer(self.phi(x), self.lambda_ten)
