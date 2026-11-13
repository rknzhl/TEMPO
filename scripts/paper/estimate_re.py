"""Estimate Reynolds number from the first 64×64 velocity field."""
import numpy as np
import torch
from torch.nn import functional as F


def spectral_features(initial, config, batch=64):
    """Six log band powers, summed over both velocity components."""
    initial = np.asarray(initial)
    if initial.shape[1:] not in ((8192,), (64, 64, 2)) or not len(initial) or batch < 1:
        raise ValueError('Expected initial velocities [N,64,64,2] or [N,8192].')
    kx = np.fft.fftfreq(64)*64
    ky = np.fft.rfftfreq(64)*64
    radius = np.sqrt(kx[:, None]**2+ky[None, :]**2)
    masks = [(radius >= lo) & (radius < hi) for lo, hi in config['radial_bands']]
    symmetry = np.ones(33)
    symmetry[1:-1] = 2
    rows = []
    for lo in range(0, len(initial), batch):
        velocity = initial[lo:lo+batch].reshape(-1, 64, 64, 2).astype('f8')
        if not np.isfinite(velocity).all():
            raise ValueError('Initial velocities must be finite.')
        spectrum = np.fft.rfft2(velocity, axes=(1, 2), norm='ortho')
        power = (abs(spectrum)**2).sum(3)*symmetry[None, None, :]
        rows.append(np.stack([np.log(np.maximum(power[:, mask].sum(1),
            config['power_floor'])) for mask in masks], 1))
    return np.concatenate(rows)


def fit(features, reynolds, config):
    """Fit the paper's fixed linear classifier on training data only."""
    classes = np.array(config['Re_classes'])
    labels = np.searchsorted(classes, reynolds)
    np.testing.assert_array_equal(classes[labels], reynolds)
    mean = features.mean(0)
    std = np.maximum(features.std(0), 1e-6)
    x = torch.from_numpy(((features-mean)/std).astype('f4'))
    y = torch.from_numpy(labels)
    torch.manual_seed(config['seed'])
    model = torch.nn.Linear(features.shape[1], len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    for _ in range(config['epochs']):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
    return dict(state_dict=model.state_dict(), feature_mean=torch.from_numpy(mean),
                feature_std=torch.from_numpy(std), classes=torch.from_numpy(classes),
                config=config)


@torch.inference_mode()
def predict(features, saved):
    x = torch.from_numpy(((features-saved['feature_mean'].numpy()) /
                         saved['feature_std'].numpy()).astype('f4'))
    state = saved['state_dict']
    probability = F.linear(x, state['weight'], state['bias']).softmax(1).numpy()
    reynolds = np.exp(probability.astype('f8') @ np.log(saved['classes'].numpy()))
    return reynolds, probability
