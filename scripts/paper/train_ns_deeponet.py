"""Train NS DeepONet and save its weights and test errors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'paper/configs/ns_deeponet.json'
FORMAT = 'ns_original_deeponet_complete_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_model(recipe=None, seed=None):
    recipe = recipe or json.loads(CONFIG.read_text())
    if sha256(ROOT / recipe['source']) != recipe['source_sha256']:
        raise ValueError('The configured DeepONet source has changed.')
    from scripts.navier_stokes.train_deeponet_navier_stokes import BranchNet, TrunkNet, DeepONet
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    width, layers = recipe['hidden_dim'], recipe['n_layers']
    return DeepONet(BranchNet(8192, width, width, layers),
                   TrunkNet(width, width, layers), width)


def coordinates():
    # Keep the trained coordinate and target order.
    x, y, t = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64),
                         np.linspace(0, 1, 21), indexing='ij')
    return torch.from_numpy(np.stack([x.ravel(), y.ravel(), t.ravel()], 1).astype('f4'))


def check_gpu():
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid:
        raise ValueError('Select one free GPU by UUID in CUDA_VISIBLE_DEVICES.')
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    pids = [int(line.split(',')[1]) for line in lines.splitlines()
            if line.split(',')[0].strip() == uuid]
    if set(pids) - {os.getpid()}:
        raise RuntimeError('The selected GPU has another compute process.')
    if not pids:
        values = subprocess.check_output(['nvidia-smi', '--id='+uuid,
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = map(float, values.strip().split(','))
        if memory > 256 or utilization > 5:
            raise RuntimeError('The selected GPU is not free.')


def load_data(directory, recipe):
    arrays = {}
    for name, expected in recipe['data_files'].items():
        path = directory / name
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Original dataset identity failed: '+name)
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(value.shape) != expected['shape'] or value.dtype.str != expected['dtype']:
            raise ValueError('Original dataset shape or dtype failed: '+name)
        arrays[path.stem] = value
    for split, count in [('train', 4000), ('test', 1000)]:
        np.testing.assert_array_equal(arrays[split+'_Re'], np.repeat(recipe['Re'], count))
    return arrays


def batch(fields, reynolds, indices, device):
    target = np.array(fields[indices], dtype='f4', copy=True)
    x = torch.from_numpy(target[:, 0].reshape(len(indices), -1).copy()).to(device)
    k = torch.from_numpy(np.array(reynolds[indices], dtype='f4').reshape(-1, 1)).to(device)
    return x, k, target.reshape(len(indices), -1)


@torch.inference_mode()
def errors(model, fields, reynolds, indices, coords, device, batch_size):
    model.eval()
    result = []
    for lo in range(0, len(indices), batch_size):
        selected = indices[lo:lo+batch_size]
        x, k, target = batch(fields, reynolds, selected, device)
        prediction = model(x, k, coords).cpu().numpy()
        error = np.linalg.norm(target-prediction, axis=1) / np.linalg.norm(target, axis=1)
        if not np.isfinite(error).all():
            raise ValueError('Nonfinite test errors.')
        result.append(error)
    return np.concatenate(result)


def statistics(values):
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                std=float(values.std()), p95=float(np.percentile(values, 95)))


def train_case(job, data, recipe, output, device):
    run = output / ('deeponet_ns_'+('joint' if job == 'joint' else 're'+job)+'_seed39')
    if (run / 'complete.json').exists():
        print('Already completed:', run.name, flush=True)
        return
    run.mkdir(parents=True, exist_ok=False)
    train_re = recipe['Re'] if job == 'joint' else [int(job)]
    selected = np.flatnonzero(np.isin(data['train_Re'], train_re))
    own_test = np.isin(data['test_Re'], train_re)
    model = build_model(recipe, seed=recipe['seed']).to(device)
    coords = coordinates().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'],
                                  weight_decay=recipe['weight_decay'])
    history = []
    started = time.monotonic()
    for step in range(recipe['optimizer_steps']):
        model.train()
        indices = np.random.choice(len(selected), min(recipe['batch_size'], len(selected)), replace=False)
        x, k, target = batch(data['train'], data['train_Re'], selected[indices], device)
        loss = torch.nn.functional.mse_loss(model(x, k, coords), torch.from_numpy(target).to(device))
        if not loss.isfinite():
            raise ValueError('Nonfinite training loss.')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        if (step+1) % 50 == 0:
            if device.type == 'cuda':
                check_gpu()
            print(f'{run.name}: {step+1}/500 loss={history[-1]:.6g}', flush=True)
    checkpoint = dict(format=FORMAT, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                      coords=coords.cpu(), input_dim=8192, parameter_dim=1, output_components=2,
                      output_shape=recipe['grid'], recipe=recipe, training_Re=train_re,
                      training_source_sha256=sha256(__file__))
    torch.save(checkpoint, run / 'weights.pt.partial')
    os.replace(run / 'weights.pt.partial', run / 'weights.pt')
    (run / 'config.json').write_text(json.dumps(dict(recipe=recipe, training_Re=train_re), indent=2)+'\n')
    (run / 'history.json').write_text(json.dumps(history)+'\n')
    test_error = errors(model, data['test'], data['test_Re'], np.arange(4000), coords,
                        device, recipe['batch_size'])
    train_rows = selected[np.random.choice(len(selected), min(2000, len(selected)), replace=False)]
    train_error = errors(model, data['train'], data['train_Re'], train_rows, coords,
                         device, recipe['batch_size'])
    cross = {str(re): statistics(test_error[data['test_Re'] == re]) for re in recipe['Re']}
    metrics = dict(run_name=run.name, n_params=sum(p.numel() for p in model.parameters()),
                   n_train=len(selected), n_test=int(own_test.sum()), optimizer_steps=len(history),
                   train=statistics(train_error), test=statistics(test_error[own_test]), cross_re=cross,
                   checkpoint_sha256=sha256(run/'weights.pt'), seconds=time.monotonic()-started,
                   versions=dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda))
    np.savez_compressed(run/'test_errors.npz', relative_l2=test_error,
                        Re=data['test_Re'], source_index=data['test_source_index'])
    rows = np.arange(4)*1000
    x, k, target = batch(data['test'], data['test_Re'], rows, device)
    with torch.inference_mode():
        prediction = model(x, k, coords).cpu().numpy()
    np.savez_compressed(run/'example_inputs.npz', x=x.cpu().numpy(), kappa=k.cpu().numpy(),
                        coords=coords.cpu().numpy(), sample_id=np.array([f"Re{int(data['test_Re'][i])}:{int(data['test_source_index'][i])}" for i in rows]))
    np.savez_compressed(run/'example_output.npz', prediction=prediction, target=target)
    (run/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (run/'complete.json').write_text(json.dumps(dict(checkpoint_sha256=metrics['checkpoint_sha256'],
                                                    cross_re=cross), indent=2)+'\n')
    print(run.name, json.dumps(cross), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', nargs='+', choices=['100', '1000', '3600', '10000', 'joint'],
                        default=['100', '1000', '3600', '10000', 'joint'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        check_gpu()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.33, (8 << 30)/total))
    recipe = json.loads(CONFIG.read_text())
    data = load_data(args.data_dir, recipe)
    for job in args.jobs:
        if device.type == 'cuda':
            check_gpu()
        train_case(job, data, recipe, args.out, device)


if __name__ == '__main__':
    main()
