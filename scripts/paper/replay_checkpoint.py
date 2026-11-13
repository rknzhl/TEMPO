"""Predict from saved weights.

Input NPZ: x[N,S], kappa[N,P], coords[D,d], optional sample_id[N].
Normalized inputs are the default; raw units are described in README.md."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper.paper_models import load_paper_model, sha256


@torch.inference_mode()
def replay(checkpoint, inputs, out, expected_sha256=None, device='cpu', threads=2,
           input_units='normalized', cno_source=None, coordinate_batch=4096,
           parameter_estimator=None):
    if threads < 1 or coordinate_batch < 1:
        raise ValueError('CPU thread and coordinate batch counts must be positive.')
    torch.set_num_threads(threads)
    checkpoint, inputs, out = map(Path, (checkpoint, inputs, out))
    if out.exists():
        raise FileExistsError(out)
    checkpoint_sha = sha256(checkpoint)
    if expected_sha256 is not None and checkpoint_sha != expected_sha256:
        raise ValueError('Checkpoint SHA256 mismatch.')
    frozen = load_paper_model(checkpoint, device, cno_source)
    with np.load(inputs, allow_pickle=False) as arrays:
        x = torch.from_numpy(arrays['x'].copy()).float()
        k = torch.from_numpy(arrays['kappa'].copy())
        if input_units != 'raw' or frozen.format != 'author_operator':
            k = k.float()
        coords = torch.from_numpy(arrays['coords'].copy()).float()
        ids = arrays['sample_id'].astype(str) if 'sample_id' in arrays else np.arange(len(x)).astype(str)
    if (x.ndim != 2 or k.ndim != 2 or len(x) != len(k) or not len(x)
            or x.shape[1] != frozen.xdim or k.shape[1] != frozen.pdim
            or len(ids) != len(x) or coords.ndim != 2
            or any(not t.isfinite().all() for t in (x, k, coords))):
        raise ValueError('Replay fixture dimensions or finite values differ from the model contract.')
    if frozen.coords is not None and not torch.equal(coords, frozen.coords.cpu()):
        raise ValueError('Replay coordinates differ from the frozen checkpoint.')
    if frozen.coords is None:
        from scripts.paper.paper_models import validate_original_burgers_coordinates
        validate_original_burgers_coordinates(coords)
        frozen.coords = coords.to(device)
    estimate = None
    if parameter_estimator is not None:
        if input_units != 'raw' or frozen.format not in ('ns_original_tempo_pod_complete_v1', 'ns_tempo_npod_complete_v1'):
            raise ValueError('The Re estimator requires raw initial NS velocity inputs.')
        from scripts.paper.estimate_re import spectral_features, predict
        estimator = torch.load(parameter_estimator, map_location='cpu', weights_only=True)
        estimate, _ = predict(spectral_features(x.numpy(), estimator['config']), estimator)
        k = torch.from_numpy(estimate[:, None])
    x, k = x.to(device), k.to(device)
    if input_units == 'raw':
        x, k = frozen.normalize_inputs(x, k)
        if estimate is not None:
            k = k.float()
    elif input_units != 'normalized':
        raise ValueError('Input units must be normalized or raw.')
    prediction, gate = frozen.predict(x, k, coordinate_batch)
    components = 2 if frozen.format == 'ns_original_deeponet_complete_v1' else (3 if frozen.state.get('config', {}).get('dataset') == 'sod' else 1)
    if (prediction.shape != (len(x), components*len(coords)) or gate.ndim != 2
            or not prediction.isfinite().all() or not gate.isfinite().all()):
        raise ValueError('Frozen prediction dimensions or finite values failed.')
    values = dict(prediction_target=prediction.cpu().numpy(), gate=gate.cpu().numpy(),
                  sample_id=ids, checkpoint_sha256=np.array(checkpoint_sha),
                  input_units=np.array(input_units), checkpoint_format=np.array(frozen.format))
    if not frozen.format.startswith('darcy_') and frozen.format not in ('ns_global_pod',
                             'ns_original_tempo_pod_complete_v1', 'ns_tempo_npod_complete_v1',
                             'ns_original_deeponet_complete_v1', 'burgers_deeponet_complete_v1',
                             'fourier_deeponet_complete_v1', 'burgers_aux_tempo_npod_complete_v1',
                             'burgers_global_pod', 'burgers_global_neuralpod'):
        values['prediction_normalized'] = prediction.cpu().numpy()
    if estimate is not None:
        values['estimated_parameter'] = estimate
        values['parameter_estimator_sha256'] = np.array(sha256(Path(parameter_estimator)))
    values['prediction_prepared'] = frozen.prepared_output(prediction).cpu().numpy()
    values['prediction_physical'] = frozen.physical_output(prediction).cpu().numpy()
    if sha256(checkpoint) != checkpoint_sha:
        raise ValueError('Frozen checkpoint changed during replay.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        np.savez_compressed(stream, **values)
    method = frozen.state.get('fit_log', {}).get('method', frozen.state.get('config', {}).get('model', frozen.format))
    return dict(rows=len(x), method=method, format=frozen.format, checkpoint_sha256=checkpoint_sha,
                input_sha256=sha256(inputs), output_sha256=sha256(out),
                input_units=input_units, reload_verified=True, training=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'inputs', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--input-units', choices=('normalized', 'raw'), default='normalized')
    parser.add_argument('--cno-source', type=Path)
    parser.add_argument('--coordinate-batch', type=int, default=4096)
    parser.add_argument('--parameter-estimator', type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.checkpoint, args.inputs, args.out,
        args.expected_sha256, args.device, args.threads, args.input_units,
        args.cno_source, args.coordinate_batch, args.parameter_estimator), indent=2))


if __name__ == '__main__':
    main()
