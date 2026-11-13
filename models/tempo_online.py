
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .pod import PODTrainer
from .pod_deeponet import BranchNet
from .fourier_neural_pod import FourierNeuralPODTrainer


@torch.no_grad()
def _eval_basis(trainer, x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns (mean (Ny,), modes (Ny, P)) for a PODTrainer or FourierNeuralPODTrainer."""
    if isinstance(trainer, PODTrainer):
        return (
            trainer.basis.mean.to(x.device),
            trainer.basis.modes.to(x.device),
        )
    if isinstance(trainer, FourierNeuralPODTrainer):
        mean = trainer.basis.mean_net(x)
        modes = torch.stack([mode.phi(x) for mode in trainer.basis.modes], dim=1)
        return mean, modes
    raise TypeError(f"Unsupported trainer type: {type(trainer)}")


def _num_modes(trainer) -> int:
    if isinstance(trainer, PODTrainer):
        return trainer.basis.num_modes
    if isinstance(trainer, FourierNeuralPODTrainer):
        return trainer.num_modes
    raise TypeError(f"Unsupported trainer type: {type(trainer)}")


class GatingNet(nn.Module):
    """(u0_sensors, kappa) -> M simplex weights."""

    def __init__(self, m_sensors: int, d_kappa: int, M: int,
                 hidden_dim: int = 128, n_layers: int = 4) -> None:
        super().__init__()
        self.branch = BranchNet(m_sensors, M, hidden_dim, n_layers, d_kappa=d_kappa)

    def forward(self, u0: Tensor, kappa: Tensor) -> Tensor:
        """Returns (N, M) simplex weights. u0: (N, m_sensors), kappa: (N, d_kappa)."""
        return F.softmax(self.branch(u0, kappa), dim=-1)


class TEMPOOnline(nn.Module):
    """Gated operator with M frozen regime bases (Phase 2)."""

    def __init__(self, gating: GatingNet, branches: nn.ModuleList) -> None:
        super().__init__()
        self.gating = gating
        self.branches = branches

    @property
    def M(self) -> int:
        return len(self.branches)

    def forward(
        self,
        u0: Tensor,
        kappa: Tensor,
        means: list[Tensor],
        modes: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Returns (s_hat (N, Ny), w (N, M) gating weights)."""
        w = self.gating(u0, kappa)
        s_hat = torch.zeros(u0.shape[0], means[0].shape[0],
                            device=u0.device, dtype=u0.dtype)
        for m, branch in enumerate(self.branches):
            b_m = branch(u0, kappa)
            s_hat_m = means[m].unsqueeze(0) + b_m @ modes[m].T
            s_hat = s_hat + w[:, m : m + 1] * s_hat_m
        return s_hat, w


@dataclass
class TEMPOOnlineConfig:
    lr: float = 3e-4
    n_epochs: int = 1000
    batch_size: int = 64
    hidden_dim: int = 128
    n_layers: int = 4
    sensor_stride: int = 1
    lambda_kl: float = 0.1
    lambda_ent: float = 0.1
    log_every: int = 50
    val_every: int = 50
    data_loss: str = "mse"
    # Floor for per-trajectory mean target energy in the trainer's target units.
    relative_loss_eps: float = 1e-12
    gate_hidden_dim: int | None = None

    def __post_init__(self) -> None:
        if self.gate_hidden_dim is not None and self.gate_hidden_dim < 1:
            raise ValueError('gate_hidden_dim must be positive.')
        if self.data_loss not in {"mse", "relative"}:
            raise ValueError("data_loss must be 'mse' or 'relative'.")
        if not math.isfinite(self.relative_loss_eps) or self.relative_loss_eps <= 0:
            raise ValueError("relative_loss_eps must be finite and positive.")


class TEMPOOnlineTrainer:
    """Phase 2 trainer: GatingNet + M BranchNets with frozen bases."""

    def __init__(self, model: TEMPOOnline, cfg: TEMPOOnlineConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.val_history: list[tuple[int, float]] = []

    def _data_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.cfg.data_loss == "relative":
            squared_error = (prediction - target).square().mean(dim=1)
            target_energy = target.square().mean(dim=1).clamp_min(self.cfg.relative_loss_eps)
            return (squared_error / target_energy).mean()
        return F.mse_loss(prediction, target)

    def train(
        self,
        s: Tensor,
        u0: Tensor,
        kappa: Tensor,
        x_flat: Tensor,
        gamma_star: Tensor,
        trainers: list,
        val_s: Tensor = None,
        val_u0: Tensor = None,
        val_kappa: Tensor = None,
    ) -> dict[str, list[float]]:
        device = next(self.model.parameters()).device
        stride = self.cfg.sensor_stride

        x_dev = x_flat.to(device)
        means_dev, modes_dev = [], []
        for t in trainers:
            mean, modes = _eval_basis(t, x_dev)
            means_dev.append(mean.detach())
            modes_dev.append(modes.detach())

        u0_sensors = u0[:, ::stride]

        self.val_history = []
        do_val = val_s is not None and val_u0 is not None and val_kappa is not None
        if do_val:
            val_u0_sensors = val_u0[:, ::stride].to(device)
            val_kappa_d = val_kappa.to(device)
            val_s_d = val_s.to(device)

        dl = DataLoader(
            TensorDataset(u0_sensors, kappa, s, gamma_star),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )
        opt = torch.optim.AdamW(self.model.parameters(),
                                lr=self.cfg.lr, weight_decay=1e-4)

        history: dict[str, list[float]] = {'total': [], 'data': [], 'kl': [], 'ent': []}

        for epoch in range(self.cfg.n_epochs):
            self.model.train()
            epoch_loss = l_data_ep = l_kl_ep = l_ent_ep = 0.0

            for u0_b, kappa_b, s_b, gamma_b in dl:
                u0_b = u0_b.to(device, non_blocking=True)
                kappa_b = kappa_b.to(device, non_blocking=True)
                s_b = s_b.to(device, non_blocking=True)
                gamma_b = gamma_b.to(device, non_blocking=True)

                s_hat, w = self.model(u0_b, kappa_b, means_dev, modes_dev)

                l_data = self._data_loss(s_hat, s_b)

                w_c = w.clamp(min=1e-8)
                gamma_c = gamma_b.clamp(min=1e-8)
                l_kl = (w_c * (w_c.log() - gamma_c.log())).sum(dim=1).mean()
                l_ent = -(w_c * w_c.log()).sum(dim=1).mean()

                loss = l_data + self.cfg.lambda_kl * l_kl - self.cfg.lambda_ent * l_ent

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()

                n_b = u0_b.shape[0]
                epoch_loss += loss.item() * n_b
                l_data_ep += l_data.item() * n_b
                l_kl_ep += l_kl.item() * n_b
                l_ent_ep += l_ent.item() * n_b

            N = len(dl.dataset)
            history['total'].append(epoch_loss / N)
            history['data'].append(l_data_ep / N)
            history['kl'].append(l_kl_ep / N)
            history['ent'].append(l_ent_ep / N)

            if do_val and epoch % self.cfg.val_every == 0:
                self.model.eval()
                with torch.no_grad():
                    parts = []
                    for i in range(0, len(val_u0_sensors), 256):
                        sv, _ = self.model(val_u0_sensors[i:i+256], val_kappa_d[i:i+256],
                                           means_dev, modes_dev)
                        parts.append(sv)
                    vl = self._data_loss(torch.cat(parts, dim=0), val_s_d).item()
                self.model.train()
                self.val_history.append((epoch, vl))

            if epoch % self.cfg.log_every == 0:
                val_str = f"  val_data={self.val_history[-1][1]:.4e}" if self.val_history else ""
                print(f"  epoch {epoch:4d} | loss={epoch_loss/N:.4e} | "
                      f"data={l_data_ep/N:.4e}  kl={l_kl_ep/N:.4e}  ent={l_ent_ep/N:.4e}{val_str}")

        print(f"  done: final loss={history['total'][-1]:.4e}")
        return history

    @torch.no_grad()
    def predict(
        self,
        u0_new: Tensor,
        kappa_new: Tensor,
        x_flat: Tensor,
        trainers: list,
    ) -> tuple[Tensor, Tensor]:
        """Returns (s_hat (N, Ny), w (N, M)) for new inputs."""
        device = next(self.model.parameters()).device
        x_dev = x_flat.to(device)

        means_dev, modes_dev = [], []
        for t in trainers:
            mean, modes = _eval_basis(t, x_dev)
            means_dev.append(mean.detach())
            modes_dev.append(modes.detach())

        u0_sensors = u0_new[:, ::self.cfg.sensor_stride].to(device)
        kappa_dev = kappa_new.to(device)

        self.model.eval()
        return self.model(u0_sensors, kappa_dev, means_dev, modes_dev)


def build_tempo_online(
    trainers: list,
    d_kappa: int,
    Nx: int,
    cfg: TEMPOOnlineConfig,
) -> tuple[TEMPOOnline, TEMPOOnlineTrainer]:
    """Construct TEMPOOnline and trainer from frozen regime trainers."""
    M = len(trainers)
    m_sensors = math.ceil(Nx / cfg.sensor_stride)
    P_list = [_num_modes(t) for t in trainers]

    gate_width = cfg.gate_hidden_dim if cfg.gate_hidden_dim is not None else cfg.hidden_dim
    gating = GatingNet(m_sensors, d_kappa, M, gate_width, cfg.n_layers)
    branches = nn.ModuleList([
        BranchNet(m_sensors, P_list[m], cfg.hidden_dim, cfg.n_layers, d_kappa=d_kappa)
        for m in range(M)
    ])

    model = TEMPOOnline(gating, branches)
    trainer = TEMPOOnlineTrainer(model, cfg)
    return model, trainer
