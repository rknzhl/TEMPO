"""Target-assisted field diagnostics, kept separate from the deployed predictor."""
from itertools import combinations

import numpy as np
import torch


def relative_errors(pred, target):
    norm = torch.linalg.vector_norm(target, dim=1)
    if (norm == 0).any():
        raise ValueError('Relative L2 is undefined for a zero target.')
    return torch.linalg.vector_norm(pred - target, dim=1) / norm


def macro_mean(errors, parameter):
    return float(np.mean([errors[parameter == v].mean() for v in np.unique(parameter)]))


def orthobasis(modes):
    q, s, _ = torch.linalg.svd(modes.double().cpu(), full_matrices=False)
    # Use numerical precision of the tabulated spatial functions.
    keep = s > s.max() * 1e-7
    return q[:, keep], s.tolist()


class ProjectionAudit:
    """Least-squares projections onto frozen, potentially nonorthogonal modes."""

    def __init__(self, means, modes):
        self.means = torch.stack(means).double().cpu()
        factors = [orthobasis(b) for b in modes]
        self.local = [q for q, _ in factors]
        self.span, _ = orthobasis(torch.cat(modes, 1))
        self.spectra = [s for _, s in factors]
        self.angles = {}
        for i, j in combinations(range(len(modes)), 2):
            cosines = torch.linalg.svdvals(self.local[i].T @ self.local[j])
            self.angles[f'{i}-{j}'] = torch.rad2deg(cosines.clamp(0, 1).acos()).tolist()

    def measure(self, y, pred, gate):
        y, pred, gate = y.double().cpu(), pred.double().cpu(), gate.double().cpu()
        norm2 = y.square().sum(1)
        if (norm2 == 0).any():
            raise ValueError('Relative L2 is undefined for a zero target.')
        local = []
        for mean, q in zip(self.means, self.local):
            z = y - mean
            local.append((z - (z @ q) @ q.T).square().sum(1) / norm2)
        local = torch.stack(local, 1)
        mean = gate @ self.means
        z = y - mean
        projection = mean + (z @ self.span) @ self.span.T
        outside = (y - projection).square().sum(1) / norm2
        inside = (projection - pred).square().sum(1) / norm2
        total = (pred - y).square().sum(1) / norm2
        return {k: v.numpy() for k, v in {
            'relative_l2': total.sqrt(), 'expert_projection_rel_l2': local.sqrt(),
            'best_single_projection_rel_l2': local.min(1).values.sqrt(),
            'outside_span_relative_sq': outside, 'inside_span_relative_sq': inside,
            'pythagorean_abs_residual': (total - outside - inside).abs(),
            'gate': gate}.items()}


def summarize(rows, parameter, source):
    def subset(mask):
        gate = rows['gate'][mask]
        summ = {'n': int(mask.sum()), 'online_mean': float(rows['relative_l2'][mask].mean()),
                'projection_mean': float(rows['best_single_projection_rel_l2'][mask].mean()),
                'outside_span_mean_sq': float(rows['outside_span_relative_sq'][mask].mean()),
                'inside_span_mean_sq': float(rows['inside_span_relative_sq'][mask].mean()),
                'max_pythagorean_abs_residual': float(rows['pythagorean_abs_residual'][mask].max())}
        for name in ['gate', 'gamma', 'target_residual_weights']:
            if name not in rows:
                continue
            w = rows[name][mask]
            h = -(w * np.log(np.maximum(w, 1e-30))).sum(1)
            summ[name] = {'mass': w.mean(0).tolist(), 'mean_entropy_nats': float(h.mean()),
                          'entropy_q10_q50_q90': np.quantile(h, [.1, .5, .9]).tolist(),
                          'max_weight_q10_q50_q90': np.quantile(w.max(1), [.1, .5, .9]).tolist()}
        if 'gamma' in rows:
            summ['gate_gamma_mean_l1'] = float(abs(gate - rows['gamma'][mask]).sum(1).mean())
        return summ
    return {'macro_online_relative_l2': macro_mean(rows['relative_l2'], parameter),
            'overall': subset(np.ones(len(parameter), dtype=bool)),
            'parameter': {str(v): subset(parameter == v) for v in np.unique(parameter)},
            'source': {str(v): subset(source == v) for v in np.unique(source)}}


def paired_interval(delta, parameter, groups, bootstrap, seed):
    """CI conditional on fitted seeds; resample whole groups within parameter profiles."""
    pvals, p = np.unique(parameter, return_inverse=True)
    gvals, g = np.unique(groups, return_inverse=True)
    cells = g * len(pvals) + p
    shape = len(gvals), len(pvals)
    sums = np.bincount(cells, weights=delta, minlength=np.prod(shape)).reshape(shape)
    counts = np.bincount(cells, minlength=np.prod(shape)).reshape(shape)
    _, profile = np.unique(counts > 0, axis=0, return_inverse=True)
    strata = [np.flatnonzero(profile == v) for v in np.unique(profile)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        ids = np.concatenate([rng.choice(s, len(s)) for s in strata])
        draws.append((sums[ids].sum(0) / counts[ids].sum(0)).mean())
    return {'mean_difference': macro_mean(delta, parameter),
            'conditional_group_bootstrap_95pct': np.quantile(draws, [.025, .975]).tolist(),
            'groups': len(gvals), 'profile_stratum_sizes': [len(s) for s in strata]}
