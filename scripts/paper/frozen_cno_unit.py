"""Darcy CNO with unit-source prediction and decoder u = beta*v(a)."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import torch
from torch import nn

CHECKPOINT_SHA256 = 'a69e944953e54b651d0f577f29dec1177eaa86087efba6fb6b9dddea774f7b51'
CNO_SOURCE_SHA256 = '30c1141b050eadd6f74ba34e33039a1488e498a8c1774bfa4ab09f75f323f47c'
DATA_MANIFEST_SHA256 = 'e67b8206d6bb6ead86a1171f1a9e06ab160d6ecdc97c344f4ffbc0005329d51f'
UNIT_PREDICTIONS_SHA256 = '09b6d0b70d3942fa0f5625660044c2e164374269f422348ae4b72cc9a5f295c3'
UNIT_ERRORS_SHA256 = 'efefd52a9188945279340a09404ad4879d0965a94c006b8658839430f92a7c0d'
PARAMETERS = 2_007_209


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class FrozenUnitCNO(nn.Module):
    """Map raw native128 permeability to pressure at beta=1."""
    def __init__(self, cno_class, coords):
        super().__init__()
        self.operator = cno_class(in_dim=4, out_dim=1, size=128, N_layers=4,
                                  N_res=4, N_res_neck=4, channel_multiplier=16, use_bn=False)
        xy = 2*(coords-coords.min(0).values)/(coords.max(0).values-coords.min(0).values)-1
        self.register_buffer('grid', xy.reshape(128, 128, 2).permute(2, 0, 1)[None].contiguous())

    @torch.inference_mode()
    def predict_unit(self, raw_a):
        a = torch.as_tensor(raw_a, dtype=self.grid.dtype, device=self.grid.device)
        if a.ndim == 2 and tuple(a.shape) == (128, 128):
            a = a.unsqueeze(0)
        if not ((a.ndim == 2 and a.shape[1] == 16384) or
                (a.ndim == 3 and tuple(a.shape[1:]) == (128, 128))):
            raise ValueError('Expected [N,128,128] or [N,16384] raw permeability.')
        a = a.reshape(len(a), 1, 128, 128)
        # The trained beta feature log10(beta)/2 is zero at beta=1.
        beta_feature = torch.zeros_like(a)
        features = torch.cat((a, beta_feature, self.grid.expand(len(a), -1, -1, -1)), 1)
        return self.operator(features).flatten(1)

    @torch.inference_mode()
    def forward(self, raw_a, beta=1.0):
        unit = self.predict_unit(raw_a)
        amplitude = torch.as_tensor(beta, dtype=unit.dtype, device=unit.device).reshape(-1, 1)
        if len(amplitude) not in (1, len(unit)):
            raise ValueError('beta must be scalar or have one value per field.')
        return unit * amplitude


def load_model(checkpoint, cno_source, device='cpu'):
    checkpoint, cno_source = Path(checkpoint), Path(cno_source)
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('The frozen beta1 checkpoint differs.')
    if sha256(cno_source) != CNO_SOURCE_SHA256:
        raise ValueError('The pinned author CNO source differs.')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['format'] == 'darcy_direct_cno_historical_v1'
    assert state['train_beta'] == [1.0] and state['native_shape'] == [128, 128]
    assert state['data_manifest_sha256'] == DATA_MANIFEST_SHA256
    assert state['selected_epoch'] == 170 and state['trained_epochs'] == 200
    norm = state['normalization']
    assert norm['x_mean'] == 0 and norm['x_std_population'] == 1 and norm['y_rms'] == 1
    assert norm['beta_log10_mean'] == 0 and norm['beta_log10_scale'] == 2
    assert norm['beta_input'] and not norm['output_beta_multiplier']
    spec = importlib.util.spec_from_file_location('pinned_darcy_cno', cno_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = FrozenUnitCNO(module.CNO2d, state['coords'])
    model.load_state_dict(state['model_state'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    assert all(v.isfinite().all() for v in model.state_dict().values())
    if torch.device(device).type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model.eval().to(device)
    model.checkpoint_metadata = dict(checkpoint_sha256=CHECKPOINT_SHA256, parameters=PARAMETERS,
                                     selected_epoch=170, trained_epochs=200, seed=42,
                                     unit_training_beta=1.0, output_beta_multiplier=True,
                                     data_manifest_sha256=DATA_MANIFEST_SHA256)
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cno-source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--betas', type=float, nargs='+', default=[1., .01])
    parser.add_argument('--cached-unit-predictions', type=Path)
    args = parser.parse_args()
    if args.batch < 1 or args.threads < 1 or any(b <= 0 for b in args.betas):
        parser.error('batch,threads,betas must be positive')
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(args.threads)
    if sha256(args.data/'manifest.json') != DATA_MANIFEST_SHA256:
        raise ValueError('Wrong stationary Darcy data.')
    manifest = json.loads((args.data/'manifest.json').read_text())
    for name in ('split.npy', 'source_row.npy', 'input_hash.npy', 'a.npy', 'unit_pressure.npy'):
        if sha256(args.data/name) != manifest['files'][name]['sha256']:
            raise ValueError('Data bytes differ: '+name)
    split = np.load(args.data/'split.npy', allow_pickle=False)
    rows = np.flatnonzero(split == 2)
    assert np.array_equal(np.flatnonzero(split == 0), np.arange(7000))
    assert np.array_equal(np.flatnonzero(split == 1), np.arange(7000, 8000))
    assert np.array_equal(rows, np.arange(8000, 9000))
    source = np.load(args.data/'source_row.npy', allow_pickle=False)
    inputs = np.load(args.data/'input_hash.npy', allow_pickle=False)
    a = np.load(args.data/'a.npy', mmap_mode='r')
    v = np.load(args.data/'unit_pressure.npy', mmap_mode='r')
    model = load_model(args.checkpoint, args.cno_source, args.device)
    cached = None
    if args.cached_unit_predictions:
        if sha256(args.cached_unit_predictions) != UNIT_PREDICTIONS_SHA256:
            raise ValueError('Cached original unit predictions differ.')
        error_path = args.cached_unit_predictions.with_name('test_beta_1_errors.npz')
        if sha256(error_path) != UNIT_ERRORS_SHA256:
            raise ValueError('Cached original test identities differ.')
        ids = np.load(error_path, allow_pickle=False)
        assert np.array_equal(ids['row'], rows)
        assert np.array_equal(ids['source_row'], source[rows])
        assert np.array_equal(ids['input_hash'], inputs[rows])
        cached = np.load(args.cached_unit_predictions, mmap_mode='r')
        assert cached.shape == (1000, 128, 128)
    errors = {beta: [] for beta in args.betas}
    for lo in range(0, len(rows), args.batch):
        hi = min(lo+args.batch, len(rows))
        unit = (np.asarray(cached[lo:hi]).reshape(hi-lo, -1) if cached is not None else
                model.predict_unit(np.array(a[rows[lo:hi]], copy=True)).cpu().numpy())
        truth = np.asarray(v[rows[lo:hi]], dtype='f8').reshape(hi-lo, -1)
        for beta in args.betas:
            amplitude = np.float32(beta)
            prediction = (unit*amplitude).astype('f8')
            reference = truth*float(amplitude)
            errors[beta].extend(np.linalg.norm(prediction-reference, axis=1)/np.linalg.norm(reference, axis=1))
    args.out.mkdir(parents=True)
    results = []
    for beta, values in errors.items():
        array = np.asarray(values)
        name = f'test_beta_{beta:g}_unit_decoder'
        np.savez_compressed(args.out/(name+'_errors.npz'), row=rows, source_row=source[rows],
                            input_hash=inputs[rows], relative_l2=array, beta=beta)
        results.append(dict(beta=beta, n=len(rows), mean_relative_l2=float(array.mean()),
                            mean_percent=float(array.mean()*100), errors_file=name+'_errors.npz'))
    report = dict(**model.checkpoint_metadata, status='complete', results=results,
                  inference='fixed original beta1 network query; external beta scaling',
                  evaluation_mode='bound_saved_GPU_unit_predictions' if cached is not None else 'frozen_model_replay',
                  device='cached_original_cuda' if cached is not None else args.device,
                  native_shape=[128, 128], input='raw full128 permeability',
                  model_weights_changed=False, training_performed=False)
    (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
