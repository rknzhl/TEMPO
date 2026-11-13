"""Fixed-rank joint refinement of Fourier NeuralPOD functions."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RefinementConfig:
    steps: int = 200
    lr: float = 1e-4
    coefficient_every: int = 25
    trajectory_batch: int = 64
    coordinate_batch: int = 4096
    relative: bool = False
    normalize_modes: bool = True
    patience: int = 75
    rtol: float = 1e-10
    grad_clip: float = 1.0
    coefficient_solver: str = 'gram'
    min_improvement: float = 1e-9


def target_batches(targets, device, batch=64, scale=1.0):
    """Stream target batches to the model device."""
    if scale <= 0:
        raise ValueError('Target scale must be positive.')
    for lo in range(0, len(targets), batch):
        hi = min(lo + batch, len(targets))
        block = targets[lo:hi]
        if not isinstance(block, torch.Tensor):
            block = torch.from_numpy(np.array(block, copy=True))
        block = block.detach().reshape(hi-lo, -1).to(device, torch.float64) / scale
        if not torch.isfinite(block).all():
            raise ValueError('Nonfinite training targets.')
        yield slice(lo, hi), block


def functions(basis, coords):
    return torch.stack([basis.mean_net(coords)] + [m.phi(coords) for m in basis.modes], 1)


@torch.no_grad()
def basis_values(basis, coords, batch=4096):
    parameter = next(basis.mean_net.parameters())
    coords = coords.detach().to(device=parameter.device, dtype=parameter.dtype)
    return torch.cat([functions(basis, c).double() for c in coords.split(batch)])


def quadratic_loss(values, gram, cross, quadrature):
    return (quadrature * ((values @ gram * values).sum(1) - 2*(values*cross).sum(1))).sum()


@torch.no_grad()
def _moments(basis, targets, coords, q, weight, cfg, scale):
    values = basis_values(basis, coords, cfg.coordinate_batch)
    mean, modes = values[:, 0], values[:, 1:]
    if cfg.coefficient_solver == 'svd':
        inverse = torch.linalg.pinv(q.sqrt()[:, None]*modes, rtol=cfg.rtol).T
    else:
        inverse = (q[:, None]*modes) @ torch.linalg.pinv(modes.T @ (q[:, None]*modes), rtol=cfg.rtol, hermitian=True)
    coefficients = torch.empty(len(targets), len(basis.modes), device=q.device, dtype=torch.float64)
    for ix, target in target_batches(targets, q.device, cfg.trajectory_batch, scale):
        coefficients[ix] = ((target-mean)*q.sqrt()) @ inverse if cfg.coefficient_solver == 'svd' else (target-mean) @ inverse
    for j, mode in enumerate(basis.modes):
        mode.lambda_ten.copy_(coefficients[:, j].to(mode.lambda_ten.dtype))
    # Recompute moments at the saved coefficient precision.
    coefficients = torch.stack([m.lambda_ten for m in basis.modes], 1).double()
    design = torch.cat([torch.ones_like(coefficients[:, :1]), coefficients], 1)
    weighted = weight[:, None]*design
    gram = design.T @ weighted
    cross = torch.zeros_like(values)
    for ix, target in target_batches(targets, q.device, cfg.trajectory_batch, scale):
        cross.add_(target.T @ weighted[ix])
    return gram, cross


def refine_basis(basis, targets, coords, gamma=None, config=None, *, target_scale=1.0, callback=None):
    """Refine mean and modes at fixed rank and responsibilities.

    Targets follow saved coefficient order. Minimize weighted quadrature error,
    optionally relative to target energy; keep Fourier features fixed."""
    cfg = config or RefinementConfig()
    if (cfg.steps < 0 or cfg.lr <= 0 or min(cfg.coefficient_every, cfg.trajectory_batch,
            cfg.coordinate_batch, cfg.patience) < 1 or not 0 < cfg.rtol < 1 or cfg.grad_clip <= 0
            or cfg.coefficient_solver not in ('gram', 'svd') or cfg.min_improvement < 0):
        raise ValueError('Invalid refinement configuration.')
    rank, n = len(basis.modes), len(targets)
    if not rank or not n or any(len(m.lambda_ten) != n for m in basis.modes):
        raise ValueError('An existing basis and aligned training amplitudes are required.')
    parameter = next(basis.mean_net.parameters())
    coords = coords.detach().to(device=parameter.device, dtype=parameter.dtype)
    q = basis.quad_weights.detach().to(parameter.device).double()
    if (coords.ndim != 2 or len(coords) != q.numel() or np.prod(targets.shape[1:]) != len(coords)
            or not torch.isfinite(coords).all() or not torch.isfinite(q).all()
            or (q < 0).any() or q.sum() <= 0):
        raise ValueError('Coordinates, targets or quadrature weights are invalid.')
    gamma = torch.ones(n, device=q.device) if gamma is None else torch.as_tensor(gamma, device=q.device)
    gamma = gamma.detach().double()
    if gamma.shape != (n,) or not torch.isfinite(gamma).all() or (gamma < 0).any() or gamma.sum() <= 0:
        raise ValueError('Responsibilities must be finite, nonnegative and have positive mass.')
    energy = torch.empty(n, device=q.device, dtype=torch.float64)
    for ix, target in target_batches(targets, q.device, cfg.trajectory_batch, target_scale):
        energy[ix] = (target.square()*q).sum(1).clamp_min(1e-12)
    weight = gamma / gamma.sum()
    if cfg.relative:
        weight = weight / energy
    constant = weight @ energy
    flags = [(p, p.requires_grad) for p in basis.parameters()]
    parameters = list(basis.mean_net.parameters()) + [p for m in basis.modes for p in m.phi.parameters()]
    history = []
    try:
        if cfg.normalize_modes:
            with torch.no_grad():
                values = basis_values(basis, coords, cfg.coordinate_batch)
                for j, mode in enumerate(basis.modes):
                    rms = (values[:, j+1].square()*q).sum().sqrt().clamp_min(1e-6)
                    mode.phi.net[-1].weight.div_(rms)
                    mode.phi.net[-1].bias.div_(rms)
                    mode.lambda_ten.mul_(rms)
        for mode in basis.modes:
            mode.lambda_ten.requires_grad_(False)
        for p in parameters:
            p.requires_grad_(True)
        gram, cross = _moments(basis, targets, coords, q, weight, cfg, target_scale)
        with torch.no_grad():
            best = float(quadratic_loss(basis_values(basis, coords, cfg.coordinate_batch), gram, cross, q)+constant)
        initial, best_state, bad, last_check, best_step = best, deepcopy(basis.state_dict()), 0, 0, 0
        if cfg.steps:
            optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps)
        for step in range(1, cfg.steps+1):
            optimizer.zero_grad(set_to_none=True)
            for lo in range(0, len(coords), cfg.coordinate_batch):
                hi = min(lo+cfg.coordinate_batch, len(coords))
                loss = quadratic_loss(functions(basis, coords[lo:hi]).double(), gram, cross[lo:hi], q[lo:hi])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite neural refinement loss.')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            if step == 1 or step % cfg.coefficient_every == 0 or step == cfg.steps:
                gram, cross = _moments(basis, targets, coords, q, weight, cfg, target_scale)
                with torch.no_grad():
                    value = float(quadratic_loss(basis_values(basis, coords, cfg.coordinate_batch), gram, cross, q)+constant)
                if not np.isfinite(value):
                    raise FloatingPointError('Nonfinite reconstruction objective.')
                if value < best-cfg.min_improvement:
                    best, best_state, bad, best_step = value, deepcopy(basis.state_dict()), 0, step
                else:
                    bad += step-last_check
                last_check = step
                row = dict(step=step, weighted_mse=value, best=best)
                history.append(row)
                if callback:
                    callback(row)
                if bad >= cfg.patience:
                    break
        basis.load_state_dict(best_state)
    finally:
        for p, enabled in flags:
            p.requires_grad_(enabled)
    return dict(initial=initial, final=best, selected_step=best_step,
                steps_run=last_check, history=history)


def refine_neural_bases(trainer, targets, coords, config=None, *, kappa=None, em_steps=0,
                        subsequent_steps=None, callback=None):
    """Refine regime bases, optionally alternating the existing E-step.

    Rank, Fourier features, quadrature and calibrated noise stay fixed."""
    from dataclasses import replace
    cfg = config or RefinementConfig(relative=trainer.cfg.heteroscedastic)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets)
    targets = targets.reshape(len(targets), -1)
    if em_steps < 0 or trainer.gamma is None or len(trainer.trainers) != trainer.cfg.M:
        raise ValueError('An initialized TEMPO trainer and nonnegative EM budget are required.')
    if cfg.relative != trainer.cfg.heteroscedastic:
        raise ValueError('The E-step and basis reconstruction must use the same noise convention.')
    if em_steps and any(not torch.allclose(local.basis.quad_weights.to(trainer._w), trainer._w)
                        for local in trainer.trainers):
        raise ValueError('The E-step and basis reconstruction must use the same quadrature.')
    results = []
    for cycle in range(em_steps+1):
        current = cfg if not cycle or subsequent_steps is None else replace(cfg, steps=subsequent_steps)
        for regime, local in enumerate(trainer.trainers):
            if float(trainer.gamma[:, regime].sum()) == 0:
                continue
            log = refine_basis(local.basis, targets, coords, trainer.gamma[:, regime], current,
                callback=(lambda row, m=regime: callback(dict(cycle=cycle, regime=m, **row))) if callback else None)
            results.append(dict(cycle=cycle, regime=regime, **log))
        if cycle < em_steps:
            gamma, _ = trainer._e_step(targets, coords, kappa)
            trainer.gamma = gamma
            trainer.pi = gamma.mean(0)
            if trainer.alpha is not None:
                trainer.mu, trainer.Sigma = trainer._compute_stats(gamma)
    return results
