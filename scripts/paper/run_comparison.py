#!/usr/bin/env python3
"""Compare frozen-basis operators on one preassigned, group-disjoint split."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random
import shlex
import sys
import time

import numpy as np
import sklearn
import torch
from torch import nn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.fourier_neural_pod import FourierNeuralPODConfig
from models.pod import PODConfig
from models.pod_deeponet import BranchNet
from models.tempo import TEMPOConfig, TEMPOTrainer, fourier_pod_factory, pod_factory
from models.tempo_online import (TEMPOOnlineConfig, TEMPOOnlineTrainer, _eval_basis,
                                 _num_modes, build_tempo_online)
from scripts.paper.comparison_metrics import (ProjectionAudit, macro_mean, paired_interval,
                                                relative_errors, summarize)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='New directory. Existing paths are refused.')
    p.add_argument('--candidates', nargs='+', default=['2x2', '2x4'], metavar='MxK')
    p.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    p.add_argument('--parameter-input', action='store_true', help='Supply the training-standardized physical parameter to every gate and branch.')
    p.add_argument('--relative', action='store_true', help='Use relative squared residuals in offline fitting, including global basis fits.')
    p.add_argument('--online-loss', choices=['mse', 'relative'], default='mse',
                   help='Common online field loss for every operator. Relative divides each trajectory MSE by its mean target energy, floored at 1e-12 in normalized target units.')
    p.add_argument('--selection-only', action='store_true',
                   help='Fit on train and select on validation, then stop before test and held-out predictions or metrics.')
    p.add_argument('--kappa-init', action='store_true', help='Initialize adaptive TEMPO from the training log-parameter grid. Also fit a frozen-initialization control.')
    p.add_argument('--kappa-prior', type=float, default=0., metavar='RHO', help='Add rho times the log-parameter prior in adaptive E-steps independently of initialization.')
    p.add_argument('--init-noise', type=float, default=.05)
    p.add_argument('--fourier', action='store_true', help='Also fit joint global Fourier and adaptive Fourier TEMPO at matched per-basis K.')
    p.add_argument('--fourier-mean-epochs', type=int, default=165)
    p.add_argument('--fourier-mode-epochs', type=int, default=85)
    p.add_argument('--em-iters', type=int, default=10)
    p.add_argument('--p-global', type=int, default=10)
    p.add_argument('--epochs', type=int, default=170)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lambda-kl', type=float, default=.1)
    p.add_argument('--lambda-ent', type=float, default=.1)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--timing-batch', type=int, default=32)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--bootstrap-seed', type=int, default=117)
    p.add_argument('--burgers-ic', action='store_true', help='Record RMS/centroid of 1D periodic sensors; enable only for actual 1D initial-condition grids.')
    p.add_argument('--oracle-gate', action='store_true', help='Additionally optimize a target-assisted simplex gate for frozen experts on evaluation sets.')
    a = p.parse_args()
    for name in ('em_iters', 'p_global', 'epochs', 'hidden', 'layers', 'batch_size', 'threads',
                 'timing_batch', 'warmup', 'repeats', 'bootstrap', 'fourier_mean_epochs', 'fourier_mode_epochs'):
        if getattr(a, name) < 1:
            p.error(f'{name} must be positive')
    if len(set(a.seeds)) != len(a.seeds) or any(v < 0 or v >= 2**32 for v in a.seeds):
        p.error('Supply distinct nonnegative 32-bit seeds.')
    if a.lr <= 0 or min(a.kappa_prior, a.init_noise, a.lambda_kl, a.lambda_ent) < 0:
        p.error('Invalid optimizer or regularization settings.')
    if (a.kappa_prior or a.kappa_init) and not a.parameter_input:
        p.error('Explicitly enable --parameter-input when using physical-parameter supervision.')
    try:
        grid = [tuple(map(int, v.lower().split('x'))) for v in a.candidates]
        if any(len(c) != 2 or min(c) <= 0 for c in grid) or len(set(grid)) != len(grid):
            raise ValueError
    except ValueError:
        p.error('Use distinct MxK candidates with positive integers.')
    return a, grid


class Dataset:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as f:
            self.split, self.group = f['split'].astype(str), f['group'].astype(str)
            self.p = f['parameter'].astype(np.float64)
            n = len(self.p)
            self.source = f['source'].astype(str) if 'source' in f else np.full(n, 'unspecified')
            self.ids = f['sample_id'].astype(str) if 'sample_id' in f else np.arange(n).astype(str)
            self.coords = torch.as_tensor(f['coords'], dtype=torch.float32)
            self.x, self.y = (torch.as_tensor(f[k], dtype=torch.float32) for k in ('x', 'y'))
        if any(a.shape != (n,) for a in (self.split, self.group, self.p, self.source, self.ids)):
            raise ValueError('Each metadata column must have shape (N,).')
        if self.x.ndim != 2 or self.y.ndim != 2 or self.x.shape[0] != n or self.y.shape[0] != n:
            raise ValueError('Expected x[N,S] and y[N,D].')
        if self.coords.ndim != 2 or self.coords.shape[0] != self.y.shape[1] or not self.coords.isfinite().all():
            raise ValueError('Expected finite coords[D,d].')
        if len(np.unique(self.ids)) != n or not np.isfinite(self.p).all():
            raise ValueError('Invalid parameter or duplicate sample ID.')
        if set(self.split) - {'train', 'val', 'test', 'heldout_test', 'excluded'}:
            raise ValueError('Unrecognized split.')
        self.idx = {s: np.flatnonzero(self.split == s) for s in ('train', 'val', 'test', 'heldout_test')}
        if any(len(self.idx[s]) == 0 for s in ('train', 'val', 'test')):
            raise ValueError('A nonempty train, val and test set is required.')
        groups = [set(self.group[self.idx[s]]) for s in ('train', 'val')]
        evaluation = set(self.group[np.isin(self.split, ['test', 'heldout_test'])])
        if groups[0] & groups[1] or (groups[0] | groups[1]) & evaluation:
            raise ValueError('Source group crosses development/evaluation partitions.')
        known = set(self.p[self.idx['train']]); unseen = set(self.p[self.idx['heldout_test']])
        if set(self.p[self.idx['val']]) != known or set(self.p[self.idx['test']]) != known or unseen & known:
            raise ValueError('Train, validation and seen test must cover the same parameters, and heldout_test parameters must be unseen.')
        train = self.idx['train']
        # Global scalar normalization preserves relative errors, spatial weighting and amplitudes.
        x, y = self.x[train], self.y[train]
        self.scale = {'x_mean': float(x.mean()), 'x_scale': float(x.std(unbiased=False)) or 1.,
                      'y_scale': float(y.square().mean().sqrt()) or 1., 'p_mean': float(self.p[train].mean()),
                      'p_scale': float(self.p[train].std()) or 1.}

    def part(self, split, with_parameter):
        ix = self.idx[split]; c = self.scale
        x, y = (self.x[ix] - c['x_mean']) / c['x_scale'], self.y[ix] / c['y_scale']
        param = torch.as_tensor(self.p[ix, None], dtype=torch.float32)
        if not x.isfinite().all() or not y.isfinite().all():
            raise ValueError(f'Nonfinite input or target in {split}')
        k = ((param - c['p_mean']) / c['p_scale']) if with_parameter else param[:, :0]
        return x, k, y, param


class GlobalOperator(nn.Module):
    """One branch with a fixed POD or NeuralPOD basis."""
    def __init__(self, inputs, kappa_dim, rank, cfg):
        super().__init__()
        self.branch = BranchNet(inputs, rank, cfg.hidden_dim, cfg.n_layers, d_kappa=kappa_dim)

    def forward(self, x, k, means, modes):
        field = means[0] + self.branch(x, k) @ modes[0].T
        return field, field.new_ones((len(field), 1))


@dataclass
class Fitted:
    model: nn.Module
    bases: list
    gamma: torch.Tensor
    residual_stats: dict
    fit_log: dict

    def evaluate_bases(self, coords):
        basis = [_eval_basis(t, coords) for t in self.bases]
        return tuple(map(list, zip(*basis)))


def fit(method, m, k, seed, train, coords, a):
    seed_all(seed)
    x, param, y, raw = train
    global_model = method.startswith('global_')
    fourier = 'fourier' in method
    rank = m * k if method == 'global_pod_total' else k
    factory = fourier_pod_factory if fourier else pod_factory
    basis_cfg = (FourierNeuralPODConfig(max_modes=rank, tol=0., n_epochs_mean=a.fourier_mean_epochs,
                 n_epochs_mode=a.fourier_mode_epochs, traj_batch_size=a.batch_size,
                 pct_start=.15) if fourier else PODConfig(max_modes=rank, fixed_rank=True))
    synchronize(coords.device); start = time.perf_counter()
    residual_stats, em_history = {}, {}
    if global_model:
        basis = factory(y, coords, basis_cfg)
        weights = y.square().mean(1).clamp_min(1e-10).reciprocal() if a.relative else None
        basis.train(y, coords, None, gamma=weights)
        bases, gamma = [basis], torch.ones(len(x), 1)
    else:
        adaptive = method.startswith('tempo_')
        init = (adaptive and a.kappa_init) or method == 'fixed_kappa'
        rho = a.kappa_prior if adaptive else 0.
        cfg = TEMPOConfig(M=m, P_global=a.p_global, max_em_iters=a.em_iters if adaptive else 0,
                          kappa_init=init, kappa_init_noise=a.init_noise, kappa_prior_weight=rho,
                          kappa_prior_with_gmm=bool(rho and not init), heteroscedastic=a.relative,
                          eps_prune=0., basis_config=basis_cfg, basis_factory=factory)
        offline = TEMPOTrainer(cfg)
        offline.train(y, coords, None, raw if (init or rho) else None)
        bases, gamma, em_history = offline.trainers, offline.gamma.cpu(), offline.history_phase1
        if adaptive:
            residual_stats = {'pi': offline.pi.tolist(), 'sigma2': cfg.sigma2, 'relative': a.relative,
                              'rho': rho, 'centers': (offline._kappa_centers.tolist() if rho else []),
                              'prior_scale': offline._kappa_T}
    if any(_num_modes(t) != rank for t in bases):
        raise ValueError('Budget mismatch. Reduce requested K or increase training data size.')
    synchronize(coords.device); offline_s = time.perf_counter() - start
    # Match initialization and minibatch order between operators.
    seed_all(seed)
    online_cfg = TEMPOOnlineConfig(hidden_dim=a.hidden, n_layers=a.layers, n_epochs=a.epochs,
                 batch_size=a.batch_size, lr=a.lr, lambda_kl=a.lambda_kl, lambda_ent=a.lambda_ent,
                 log_every=max(1, a.epochs // 5), data_loss=a.online_loss)
    if global_model:
        model = GlobalOperator(x.shape[1], param.shape[1], rank, online_cfg)
        trainer = TEMPOOnlineTrainer(model, online_cfg)
    else:
        model, trainer = build_tempo_online(bases, param.shape[1], x.shape[1], online_cfg)
    model.to(coords.device)
    begin = time.perf_counter()
    history = trainer.train(y, x, param, coords, gamma, bases)
    synchronize(coords.device)
    fit_log = {'method': method, 'm': m, 'k': k, 'seed': seed, 'ranks': [_num_modes(t) for t in bases],
               'offline_s': offline_s, 'online_s': time.perf_counter() - begin,
               'online_parameters': sum(p.numel() for p in model.parameters()),
               'basis_parameters': sum(p.numel() for t in bases for p in t.basis.parameters()),
               'online_config': asdict(online_cfg), 'basis_config': asdict(basis_cfg),
               'offline_history': em_history, 'online_history': history}
    return Fitted(model.eval(), bases, gamma, residual_stats, fit_log)


def validation(model, data, coords, batch):
    x, k, y, _ = data
    basis = model.evaluate_bases(coords)
    errors = []
    for i in range(0, len(x), batch):
        sl = slice(i, i + batch)
        pred, _ = model.model(x[sl].to(coords.device), k[sl].to(coords.device), *basis)
        errors.append(relative_errors(pred.cpu(), y[sl]))
    return torch.cat(errors).numpy()


def save_fit(path, fitted, coords, scale, xdim, pdim):
    states = []
    for t in fitted.bases:
        states.append({'weights': t.basis.state_dict(),
                       # Retain the frequency field for compatibility with earlier checkpoints.
                       'frequencies': {n: b for n, b in t.basis.named_buffers() if n.endswith('.B')}})
    torch.save({'bases': states, 'online': fitted.model.state_dict(), 'gamma': fitted.gamma,
                'residual_stats': fitted.residual_stats, 'fit_log': fitted.fit_log,
                'coords': coords, 'scale': scale, 'xdim': xdim, 'pdim': pdim}, path)


def load_fit(path, device):
    from models.fourier_neural_pod import FourierNeuralPODTrainer
    from models.pod import PODTrainer
    from models.regime_basis import FourierRegimeBasis
    state = torch.load(path, weights_only=True, map_location='cpu')
    log = state['fit_log']; cfg = TEMPOOnlineConfig(**log['online_config'])
    bases = []
    for rank, item in zip(log['ranks'], state['bases']):
        sd = item['weights']
        if 'fourier' in log['method']:
            basis = FourierRegimeBasis(state['coords'].shape[1], len(state['gamma']), sd['quad_weights'])
            for _ in range(rank):
                basis.add_mode()
            basis.load_state_dict({**sd, **item.get('frequencies', {})}, strict=True)
            trainer = FourierNeuralPODTrainer(basis.to(device).eval(), FourierNeuralPODConfig(**log['basis_config']))
        else:
            trainer = PODTrainer(PODConfig(**log['basis_config']))
            trainer.basis.initialize(sd['mean'].to(device), sd['modes'].to(device), sd['coeffs'])
        trainer.num_modes = rank
        bases.append(trainer)
    if log['method'].startswith('global_'):
        model = GlobalOperator(state['xdim'], state['pdim'], log['ranks'][0], cfg)
    else:
        model, _ = build_tempo_online(bases, state['pdim'], state['xdim'], cfg)
    model.load_state_dict(state['online'])
    return Fitted(model.to(device).eval(), bases, state['gamma'], state['residual_stats'], log)


def optimal_gate(model, x, k, means, modes, y, gate):
    from scipy.optimize import minimize
    experts = ([model.branch] if isinstance(model, GlobalOperator) else model.branches)
    fields = torch.stack([mu + branch(x, k) @ phi.T for branch, mu, phi in zip(experts, means, modes)], 1)
    r = (fields.cpu().double() - y.double()[:, None, :]) / y.double().norm(dim=1)[:, None, None]
    gram = (r @ r.transpose(1, 2)).numpy()
    result = []
    for a, w in zip(gram, gate.cpu().double().numpy()):
        if len(w) == 1:
            result.append(np.sqrt(max(a[0, 0], 0.)))
            continue
        # Rescale the objective to keep tolerance comparable across physical parameter scales.
        mat = a / max(abs(a).max(), 1e-30)
        opt = minimize(lambda v: v@mat@v, w / w.sum(), jac=lambda v: 2*mat@v,
                       method='SLSQP', bounds=[(0., 1.)]*len(w),
                       constraints=[{'type': 'eq', 'fun': lambda v: v.sum()-1., 'jac': lambda v: np.ones_like(v)}],
                       options={'ftol': 1e-12, 'maxiter': 200})
        if not opt.success:
            raise RuntimeError(f'Oracle gate optimization failed: {opt.message}')
        result.append(np.sqrt(max(opt.x@a@opt.x, 0.)))
    return np.asarray(result)


@torch.inference_mode()
def diagnostics(fitted, dataset, split, coords, basis, audit, a):
    x, k, y, raw = dataset.part(split, a.parameter_input)
    idx = dataset.idx[split]; collected = {}
    for start in range(0, len(x), a.batch_size):
        s = slice(start, start + a.batch_size); xb, kb = x[s].to(coords.device), k[s].to(coords.device)
        pred, gate = fitted.model(xb, kb, *basis)
        block = audit.measure(y[s], pred, gate)
        if split == 'train':
            block['gamma'] = fitted.gamma[s].numpy()
        if a.oracle_gate and split in ('test', 'heldout_test'):
            block['oracle_gate_rel_l2'] = optimal_gate(fitted.model, xb, kb, *basis, y[s], gate)
        st = fitted.residual_stats
        if st:
            rel2 = torch.from_numpy(block['expert_projection_rel_l2']).square()
            norm2 = y[s].double().square().mean(1, keepdim=True)
            residual = rel2 * norm2
            if st['relative']:
                residual /= norm2.clamp_min(1e-10)
            lp = torch.tensor(st['pi'], dtype=torch.double).clamp_min(1e-30).log() - residual / (2*st['sigma2'])
            if st['rho']:
                d = (raw[s].double().log() - torch.tensor(st['centers'])) / st['prior_scale']
                lp += st['rho'] * (-.5*d.square()).log_softmax(1)
            block['target_residual_weights'] = lp.softmax(1).numpy()
        for name, value in block.items():
            collected.setdefault(name, []).append(value)
    rows = {key: np.concatenate(value) for key, value in collected.items()}
    if a.burgers_ic:
        ic = dataset.x[idx].numpy()
        power = abs(np.fft.rfft(ic, axis=1))**2
        power[:, 0] = 0.
        rows['ic_rms'] = np.sqrt((ic**2).mean(1))
        rows['ic_spectral_centroid_mode'] = (power @ np.arange(power.shape[1])) / np.maximum(power.sum(1), 1e-30)
    result = summarize(rows, dataset.p[idx], dataset.source[idx])
    if 'oracle_gate_rel_l2' in rows:
        for p in np.unique(dataset.p[idx]):
            result['parameter'][str(p)]['oracle_gate_mean_rel_l2'] = float(rows['oracle_gate_rel_l2'][dataset.p[idx] == p].mean())
    return rows, result


@torch.inference_mode()
def benchmark(fitted, test, coords, a):
    x, k = test[:2]
    x, k = x.to(coords.device), k.to(coords.device)
    synchronize(coords.device); start = time.perf_counter()
    basis = fitted.evaluate_bases(coords)
    synchronize(coords.device)
    result = {'basis_setup_ms': (time.perf_counter()-start)*1e3, 'timings': {}}
    for b in sorted({1, min(a.timing_batch, len(x))}):
        xb, kb = x[:b], k[:b]
        cached = lambda: fitted.model(xb, kb, *basis)
        uncached = lambda: fitted.model(xb, kb, *fitted.evaluate_bases(coords))
        pred, gate = cached(); ref, ref_gate = uncached()
        torch.testing.assert_close(pred, ref, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(gate, ref_gate, rtol=2e-5, atol=2e-6)
        timings = {'parity_max_abs': float((pred - ref).abs().max()), 'b': b}
        calls = [('cached', cached), ('uncached', uncached)]
        samples = {name: [] for name, _ in calls}
        for _ in range(a.warmup):
            for _, f in calls:
                f()
        # Alternate order to reduce systematic effects of warm caches and clock drift.
        resume_count = getattr(a, 'resume_count', lambda: 0)
        for rep in range(a.repeats):
            for name, f in (calls if rep % 2 == 0 else reversed(calls)):
                # A voluntary cooling pause must not count as inference latency.
                for attempt in range(20):
                    resume_id = resume_count()
                    synchronize(coords.device); start = time.perf_counter_ns()
                    f(); synchronize(coords.device)
                    elapsed_ms = (time.perf_counter_ns()-start)*1e-6
                    if resume_count() == resume_id:
                        samples[name].append(elapsed_ms)
                        break
                else:
                    raise RuntimeError("A timed call repeatedly crossed a cooling pause. "
                                       "Use a smaller timing grid or measure this model without pacing.")
        for name, times in samples.items():
            med = np.median(times)
            timings[name] = {'raw_batch_ms': times, 'median_batch_ms': float(med),
                             'p90_batch_ms': float(np.percentile(times, 90)),
                             'median_ms_per_sample': float(med/b), 'samples_per_second': float(1000*b/med)}
        result['timings'][str(b)] = timings
    return result


def run(a, grid):
    dataset = Dataset(a.data)
    train = dataset.part('train', a.parameter_input); val = dataset.part('val', a.parameter_input)
    if (a.kappa_init or a.kappa_prior) and (dataset.p <= 0).any():
        raise ValueError('A log-parameter prior requires positive parameter values.')
    max_rank = min(len(train[0])-1, train[2].shape[1])
    if any(m > len(train[0]) or m*k > max_rank for m, k in grid):
        raise ValueError(f'Global matched rank M*K exceeds the available centered rank bound {max_rank}.')
    a.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device); coords = dataset.coords.to(device)
    core = ['global_pod_k', 'global_pod_total', 'fixed_gmm', 'tempo_pod']
    methods = core + (['fixed_kappa'] if a.kappa_init else []) + (['global_fourier_k', 'tempo_fourier'] if a.fourier else [])
    arguments = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    hasher = hashlib.sha256()
    with a.data.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(block)
    write_json(a.out/'run.json', {'arguments': arguments, 'command': shlex.join([sys.executable] + sys.argv),
        'data_sha256': hasher.hexdigest(), 'normalization_train_only': dataset.scale,
        'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
        'sklearn': sklearn.__version__, 'platform': platform.platform(), 'machine': platform.machine(),
        'accelerator': torch.cuda.get_device_name(device) if device.type == 'cuda' else device.type,
        'input_dim': train[0].shape[1], 'output_dim': train[2].shape[1], 'coord_dim': coords.shape[1],
        'selection': 'minimum mean of the four core methods, seeds and per-parameter validation means; ties use candidate order',
        'timing_boundary': 'device-resident normalized inputs and coordinates through full output field and gate; cached excludes basis evaluation; POD modes are already tabulated',
        'bootstrap': 'paired differences of per-object errors averaged across fitted seeds; whole source groups resampled within parameter-presence profiles'})
    np.savez_compressed(a.out/'splits.npz', row=np.arange(len(dataset.p)), sample_id=dataset.ids,
                        group=dataset.group, parameter=dataset.p, split=dataset.split, source=dataset.source)
    records = []; candidate_scores = []
    for m,k in grid:
        scores = []
        for seed in a.seeds:
            for method in methods:
                dest = a.out/f'M{m}_K{k}'/f'seed{seed}'/method
                dest.mkdir(parents=True)
                model = fit(method, m, k, seed, train, coords, a)
                with torch.inference_mode():
                    errors = validation(model, val, coords, a.batch_size)
                p = dataset.p[dataset.idx['val']]; score = macro_mean(errors, p)
                if not np.isfinite(score):
                    raise ValueError(f'Nonfinite validation score for {dest}')
                if method in core:
                    scores.append(score)
                save_fit(dest/'weights.pt', model, coords.cpu(), dataset.scale, train[0].shape[1], train[1].shape[1])
                write_json(dest/'fit.json', model.fit_log)
                np.savez_compressed(dest/'validation.npz', row=dataset.idx['val'], relative_l2=errors)
                records.append({'m': m, 'k': k, 'seed': seed, 'method': method, 'score': score,
                                'parameter': {str(q): float(errors[p == q].mean()) for q in np.unique(p)}})
                del model
        candidate_scores.append(float(np.mean(scores)))
    best = int(np.argmin(candidate_scores)); m, k = grid[best]
    write_json(a.out/'selection.json', {'m': m, 'k': k, 'grid': grid, 'candidate_scores': candidate_scores, 'validation': records})
    if a.selection_only:
        print(f'Selected M={m}, K={k} on validation; evaluation was not performed. Results in {a.out}')
        return
    test = dataset.part('test', a.parameter_input)
    summaries = []; errors_by_split = {}
    for seed in a.seeds:
        for method in methods:
            dest = a.out/f'M{m}_K{k}'/f'seed{seed}'/method
            model = load_fit(dest/'weights.pt', device)
            with torch.inference_mode():
                reloaded = validation(model, val, coords, a.batch_size)
                with np.load(dest/'validation.npz') as f:
                    np.testing.assert_allclose(reloaded, f['relative_l2'], rtol=2e-5, atol=2e-6)
                basis = model.evaluate_bases(coords)
                audit = ProjectionAudit(*basis)
                result = {'method': method, 'seed': seed, 'ranks': model.fit_log['ranks'],
                          'number_of_means': len(model.bases), 'weights_file_bytes': (dest/'weights.pt').stat().st_size,
                          'cached_basis_bytes': sum(t.numel()*t.element_size() for family in basis for t in family),
                          'online_parameters': model.fit_log['online_parameters'], 'basis_parameters': model.fit_log['basis_parameters'],
                          'singular_values': audit.spectra, 'pairwise_principal_angles_deg': audit.angles,
                          'numerical_span_rank': audit.span.shape[1], 'offline_s': model.fit_log['offline_s'],
                          'online_s': model.fit_log['online_s'], 'splits': {}}
                for split in ['train', 'test', 'heldout_test']:
                    idx = dataset.idx[split]
                    if len(idx) == 0:
                        continue
                    rows, result['splits'][split] = diagnostics(model, dataset, split, coords, basis, audit, a)
                    np.savez_compressed(dest/f'{split}_rows.npz', row=idx, sample_id=dataset.ids[idx],
                                        group=dataset.group[idx], parameter=dataset.p[idx], source=dataset.source[idx], **rows)
                    if split != 'train':
                        errors_by_split.setdefault((split,method), []).append(rows['relative_l2'])
                result['timing'] = benchmark(model, test, coords, a)
                write_json(dest/'evaluation.json', result)
                summaries.append(result)
    comparisons = []
    for split in ['test', 'heldout_test']:
        idx = dataset.idx[split]
        if not len(idx):
            continue
        for target in ['tempo_pod'] + (['tempo_fourier'] if a.fourier else []):
            for control in methods:
                if control == target:
                    continue
                delta = np.array(errors_by_split[split,target]) - np.array(errors_by_split[split,control])
                by_seed = [macro_mean(d, dataset.p[idx]) for d in delta]
                paired = paired_interval(delta.mean(0), dataset.p[idx], dataset.group[idx], a.bootstrap, a.bootstrap_seed)
                comparisons.append({'split': split, 'target': target, 'control': control,
                                    'by_seed': by_seed, 'seed_sample_sd': float(np.std(by_seed, ddof=1)) if len(by_seed)>1 else None, **paired})
    write_json(a.out/'summary.json', {'m': m, 'k': k, 'seeds': a.seeds, 'evaluations': summaries, 'paired': comparisons})
    print(f'Finished M={m}, K={k}; results in {a.out}')


if __name__ == '__main__':
    args, candidates = parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args, candidates)
