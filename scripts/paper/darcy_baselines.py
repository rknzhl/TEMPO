"""Train Darcy FNO, CNO or DeepONet for unit-source pressure v(a).

All models use the common pressure decoder u = beta*v(a)."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

CNO_SHA256 = '30c1141b050eadd6f74ba34e33039a1488e498a8c1774bfa4ab09f75f323f47c'
FNO_SHA256 = '32fbecfbd11ca733212c75ce8c728c1a4ba74c17eebb91d1e701b50c1788b4d8'
PAPER_DATA_MANIFEST_SHA256 = 'e67b8206d6bb6ead86a1171f1a9e06ab160d6ecdc97c344f4ffbc0005329d51f'
BETA_SEEN = [0.1, 1., 10., 100.]
BETA_UNSEEN = [0.01, 0.3]
FILES = ('a.npy', 'unit_pressure.npy', 'split.npy', 'source_row.npy', 'input_hash.npy', 'coords.npy')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def recipe(name):
    return dict(model=name, seed=17, epochs=500, optimizer='AdamW',
                lr=3e-4 if name == 'deeponet' else 1e-3, weight_decay=1e-4,
                effective_batch=64 if name == 'deeponet' else 32,
                microbatch=16 if name == 'deeponet' else 8,
                coordinate_batch=16384, loss='mean_per_trajectory_relative_squared_L2',
                relative_loss_eps_mean_energy=1e-12,
                checkpoint_selection='final_fixed_epoch', scheduler=None,
                beta_seen=BETA_SEEN, beta_unseen=BETA_UNSEEN, beta_input=False,
                physical_decoder='u_beta = beta * unit_pressure_prediction')


class DarcyData:
    def __init__(self, folder, smoke=False):
        self.folder = Path(folder)
        manifest_path = self.folder/'manifest.json'
        self.manifest_sha256 = sha256(manifest_path)
        if not smoke and self.manifest_sha256 != PAPER_DATA_MANIFEST_SHA256:
            raise ValueError('Require the fixed paper Darcy data manifest.')
        complete = json.loads((self.folder/'COMPLETE.json').read_text())
        if complete['manifest_sha256'] != self.manifest_sha256:
            raise ValueError('Data completion receipt does not bind this manifest.')
        self.manifest = json.loads(manifest_path.read_text())
        for name in FILES:
            path = self.folder/name
            if name == 'unit_pressure.npy':
                if path.stat().st_size != self.manifest['files'][name]['bytes']:
                    raise ValueError('Unit pressure payload size differs.')
                continue
            if sha256(path) != self.manifest['files'][name]['sha256']:
                raise ValueError('Data payload hash differs: '+name)
        self.target_fingerprint = (self.folder/'unit_pressure.npy').stat()
        self.a, self.v = [np.load(self.folder/name, mmap_mode='r', allow_pickle=False)
                          for name in FILES[:2]]
        self.split, self.source_row, self.input_hash, axis = [np.load(self.folder/name, allow_pickle=False)
                                                             for name in FILES[2:]]
        n = len(self.a)
        if (self.a.shape != (n, 128, 128) or self.a.dtype != np.dtype('<f4') or
                self.v.shape != self.a.shape or self.v.dtype != np.dtype('<f8') or
                self.split.shape != (n,) or self.split.dtype != np.dtype('uint8') or
                self.source_row.shape != (n,) or self.source_row.dtype != np.dtype('<i4') or
                self.input_hash.shape != (n,) or self.input_hash.dtype != np.dtype('S64') or
                len(set(self.input_hash.tolist())) != n):
            raise ValueError('Require distinct native128 conductivities and float64 unit solutions.')
        self.rows = {name: np.flatnonzero(self.split == code) for code, name in enumerate(('train', 'val', 'test'))}
        if set(np.unique(self.split)) != {0, 1, 2} or any(not len(rows) for rows in self.rows.values()):
            raise ValueError('Require nonempty disjoint train/validation/test partitions.')
        if not smoke and (n != 9000 or [len(self.rows[k]) for k in self.rows] != [7000, 1000, 1000]):
            raise ValueError('The paper recipe requires 7000/1000/1000 distinct conductivities.')
        expected = (np.arange(128, dtype='f8')+.5)/128
        if axis.shape != (128,) or not np.array_equal(axis, expected):
            raise ValueError('Coordinates must be native cell centers, axis0=x and axis1=y.')
        self.coords = torch.from_numpy(np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2).astype('f4'))

    def normalization(self):
        count, sx, sx2, sy2 = 0, 0., 0., 0.
        for rows in blocks(self.rows['train'], 32):
            x, y = np.asarray(self.a[rows], dtype='f8'), np.asarray(self.v[rows], dtype='f8')
            if not np.isfinite(x).all() or (x <= 0).any() or not np.isfinite(y).all():
                raise ValueError('Invalid training conductivity or unit pressure.')
            count += x.size; sx += x.sum(dtype='f8'); sx2 += np.square(x).sum(dtype='f8')
            sy2 += np.square(y).sum(dtype='f8')
        mean = sx/count; std = np.sqrt(max(sx2/count-mean*mean, 0.)); yrms = np.sqrt(sy2/count)
        if not std > 0 or not yrms > 0:
            raise ValueError('Degenerate training normalization.')
        return dict(x_mean=float(mean), x_std_population=float(std), y_rms=float(yrms),
                    train_rows=len(self.rows['train']), target_centered=False, beta_input=False,
                    x_arithmetic='raw_float32_then_torch_float32_affine',
                    target_arithmetic='raw_float64_divide_train_RMS_then_cast_float32')

    def x(self, rows, norm, device):
        x = torch.from_numpy(np.array(self.a[rows], dtype='f4', copy=True)).flatten(1)
        return ((x-norm['x_mean'])/norm['x_std_population']).to(device)

    def y(self, rows, norm):
        return torch.from_numpy((np.asarray(self.v[rows], dtype='f8').reshape(len(rows), -1)/norm['y_rms']).astype('f4'))


def blocks(rows, size):
    for first in range(0, len(rows), size):
        yield rows[first:first+size]


def author_class(name, cno_source):
    if name == 'fno':
        if importlib.metadata.version('neuraloperator') != '2.0.0':
            raise ValueError('Use the pinned original neuraloperator==2.0.0 implementation.')
        from neuralop.models import FNO
        source = Path(inspect.getfile(FNO))
        if sha256(source) != FNO_SHA256:
            raise ValueError('FNO author source differs from the pinned implementation.')
        return FNO, source
    source = Path(cno_source) if cno_source else None
    if source is None or sha256(source) != CNO_SHA256:
        raise ValueError('Supply the pinned original CNO2d.py with --cno-source.')
    spec = importlib.util.spec_from_file_location('darcy_author_cno', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.CNO2d, source


class NativeOperator(nn.Module):
    def __init__(self, name, coords, cno_source):
        super().__init__()
        cls, self.author_source = author_class(name, cno_source)
        self.operator = (cls(n_modes=(16, 16), in_channels=3, out_channels=1,
                            hidden_channels=32, n_layers=4, use_channel_mlp=True,
                            positional_embedding=None) if name == 'fno' else
                         cls(in_dim=3, out_dim=1, size=128, N_layers=4,
                             N_res=4, N_res_neck=4, channel_multiplier=16, use_bn=False))
        xy = 2*(coords-coords.min(0).values)/(coords.max(0).values-coords.min(0).values)-1
        self.register_buffer('grid', xy.reshape(128, 128, 2).permute(2, 0, 1)[None].contiguous())

    def forward(self, x):
        features = torch.cat((x.reshape(len(x), 1, 128, 128), self.grid.expand(len(x), -1, -1, -1)), 1)
        output = self.operator(features)
        if output.shape != (len(x), 1, 128, 128):
            raise ValueError('The author operator changed the native output grid.')
        return output.flatten(1)


class OrdinaryDeepONet(nn.Module):
    def __init__(self, coords):
        super().__init__()
        from models.pod_deeponet import BranchNet
        from models.neural_pod_mode import _mlp
        self.branch = BranchNet(16384, 128, hidden_dim=128, n_layers=3, d_kappa=0)
        self.trunk = _mlp(2, 128, 128, 3, act=nn.Tanh)
        self.bias = nn.Parameter(torch.zeros(1))
        # Same min/range coordinate normalization as the source native trainer.
        self.register_buffer('coords', (coords-coords.min(0).values)/(coords.max(0).values-coords.min(0).values))

    def forward(self, x, coordinate_batch=16384):
        branch = self.branch(x)
        return torch.cat([branch @ self.trunk(c).T+self.bias for c in self.coords.split(coordinate_batch)], 1)


def relative_squared_loss(prediction, target):
    energy=target.square().mean(1).clamp_min(1e-12)
    return ((prediction-target).square().mean(1)/energy).mean()


def backward_deeponet(model, x, y, coordinate_batch, weight=1.):
    """Accumulate full-grid relative-loss gradients by coordinate chunks.

    Backpropagate through the branch once and step the optimizer after all chunks."""
    branch = model.branch(x); leaf = branch.detach().requires_grad_(True)
    value = x.new_zeros(())
    energy=y.square().sum(1).clamp_min(y.shape[1]*1e-12).to(x.device)
    for first in range(0, len(model.coords), coordinate_batch):
        end = min(first+coordinate_batch, len(model.coords))
        prediction = leaf @ model.trunk(model.coords[first:end]).T+model.bias
        loss = (((prediction-y[:,first:end].to(x.device)).square().sum(1)/energy).mean())*weight
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite full-grid DeepONet loss.')
        loss.backward(); value += loss.detach()
    if leaf.grad is None or not leaf.grad.isfinite().all():
        raise FloatingPointError('Invalid accumulated branch gradient.')
    branch.backward(leaf.grad)
    return value


def tensor_state(model):
    state = dict(model.state_dict()); state.pop('_metadata', None)
    if not all(torch.is_tensor(v) and v.isfinite().all() for v in state.values()):
        raise ValueError('Checkpoint contains a non-tensor or nonfinite model state.')
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def build(name, coords, cno_source=None):
    return OrdinaryDeepONet(coords) if name == 'deeponet' else NativeOperator(name, coords, cno_source)


def load_checkpoint(path, source, cno_source=None, device='cpu'):
    """Restore saved weights and verify external operator implementations."""
    sys.path.insert(0, str(source))
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != 'darcy_unit_baseline_v1':
        raise ValueError('Unexpected Darcy checkpoint format.')
    model = build(state['recipe']['model'], state['coords'], cno_source)
    for name,expected in state['source_sha256'].items():
        if name=='darcy_baselines.py' or name.startswith('models/'):
            continue
        if name.startswith('neuralop/'):
            import neuralop
            actual=Path(neuralop.__file__).parent/name.removeprefix('neuralop/')
        else:actual=model.author_source
        if sha256(actual)!=expected:raise ValueError('Frozen model source changed: '+name)
    model.load_state_dict(state['model_state'], strict=True)
    if any(not torch.equal(v, state['model_state'][k]) for k, v in tensor_state(model).items()):
        raise ValueError('Restored model tensors differ from the saved checkpoint.')
    return model.to(device).eval(), state


class FrozenPhysicalPredictor:
    """Map raw permeability and beta to pressure using a cached fixed-grid trunk."""
    def __init__(self, model, normalization):
        self.model=model.eval(); self.normalization=normalization; self.trunk=None
        with torch.inference_mode():
            if isinstance(model, OrdinaryDeepONet):
                self.trunk=model.trunk(model.coords)

    @torch.inference_mode()
    def __call__(self, raw_a, raw_beta):
        norm=self.normalization
        x=(raw_a.reshape(len(raw_a),16384)-norm['x_mean'])/norm['x_std_population']
        prediction=(self.model.branch(x)@self.trunk.T+self.model.bias
                    if self.trunk is not None else self.model(x))
        beta=raw_beta.reshape(len(raw_a),1)
        return prediction*norm['y_rms']*beta


@torch.inference_mode()
def evaluate(model, data, rows, norm, device, batch_size, out, split):
    errors, unseen_errors, unit_errors = [], [], []
    path = out/(split+'_predictions_unit.npy')
    saved = np.lib.format.open_memmap(path, mode='w+', dtype='f4', shape=(len(rows), 128, 128))
    for start in range(0, len(rows), batch_size):
        chosen = rows[start:start+batch_size]
        prediction = model(data.x(chosen, norm, device))
        unit = (prediction.cpu().numpy().astype('f8')*norm['y_rms']).astype('f4').reshape(-1, 128, 128)
        truth = np.asarray(data.v[chosen], dtype='f8')
        if not np.isfinite(unit).all() or not np.isfinite(truth).all():
            raise ValueError('Nonfinite validation/test field.')
        saved[start:start+len(chosen)] = unit
        energy = np.linalg.norm(truth.reshape(len(chosen), -1), axis=1)
        if (energy <= 0).any():
            raise ValueError('Zero pressure norm makes relative error undefined.')
        error = np.linalg.norm((unit.astype('f8')-truth).reshape(len(chosen), -1), axis=1)/energy
        physical = []
        for beta in BETA_SEEN+BETA_UNSEEN:
            difference = (beta*unit.astype('f8')-beta*truth).reshape(len(chosen), -1)
            physical.append(np.linalg.norm(difference, axis=1)/(beta*energy))
        physical = np.stack(physical, 1)
        unit_errors.append(error); errors.append(physical[:, :4]); unseen_errors.append(physical[:, 4:])
    saved.flush(); del saved
    unit, physical, unseen = map(np.concatenate, (unit_errors, errors, unseen_errors))
    np.savez_compressed(out/(split+'_errors.npz'), row=rows, source_row=data.source_row[rows],
        input_hash=data.input_hash[rows], unit_relative_l2=unit, physical_relative_l2=physical,
        beta=np.array(BETA_SEEN), unseen_physical_relative_l2=unseen, beta_unseen=np.array(BETA_UNSEEN))
    maximum = float(max(abs(physical-unit[:, None]).max(), abs(unseen-unit[:, None]).max()))
    if maximum > 1e-12:
        raise ValueError('Shared positive-beta physical decoder failed relative-error invariance.')
    result = dict(status='complete', n=len(rows), mean_relative_l2=float(unit.mean()),
        primary_metric='physical_u_beta1_relative_L2', beta_values=BETA_SEEN, beta_unseen=BETA_UNSEEN,
        decoder_invariance_max_abs=maximum, prediction_unit_sha256=sha256(path),
        errors_sha256=sha256(out/(split+'_errors.npz')), beta_input=False,
        prediction_units='physical_unit_source_pressure_v', decoder='u_beta = beta*v')
    write_json(out/(split+'_summary.json'), result)
    return result


def train(args):
    torch.set_num_threads(2)
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0, str(args.source))
    cfg = recipe(args.model); cfg['seed'] = args.seed
    if args.smoke:
        cfg.update(epochs=1, effective_batch=2, microbatch=2)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is not available.')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., (12*(1<<30))/total))
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    data = DarcyData(args.data, args.smoke); norm = data.normalization()
    if args.smoke:
        data.rows={name:rows[:4 if name=='train' else 2] for name,rows in data.rows.items()}
    model = build(args.model, data.coords, args.cno_source).to(device)
    cfg['trainable_parameters']=sum(p.numel() for p in model.parameters())
    cfg['unique_optimizer_inputs_per_epoch']=len(data.rows['train'])
    cfg['architecture'] = (dict(branch_input=16384, hidden_dim=128, layers=3, latent_dim=128,
                               activation='Tanh', trainable_trunk=True) if args.model=='deeponet' else
                           dict(input_channels=3, output_channels=1, native_shape=[128,128],
                                resampling=False, n_modes=[16,16], width=32, layers=4) if args.model=='fno' else
                           dict(input_channels=3, output_channels=1, native_shape=[128,128],
                                resampling=False, N_layers=4, N_res=4, N_res_neck=4,
                                channel_multiplier=16, use_bn=False))
    source_hashes = {'darcy_baselines.py': sha256(__file__)}
    if args.model=='deeponet':
        for name in ('pod_deeponet.py','neural_pod_mode.py'):
            source_hashes['models/'+name]=sha256(args.source/'models'/name)
    else:
        source_hashes['CNO2d.py' if args.model=='cno' else 'neuralop/models/fno.py']=sha256(model.author_source)
        if args.model=='fno':
            package=model.author_source.resolve().parents[1]
            for name in ('fno_block.py','spectral_convolution.py','channel_mlp.py','skip_connections.py'):
                source_hashes['neuralop/layers/'+name]=sha256(package/'layers'/name)
    args.out.mkdir(parents=True)
    write_json(args.out/'normalization.json', norm)
    report = dict(status='training', recipe=cfg, data_manifest_sha256=data.manifest_sha256,
                  source_sha256=source_hashes, normalization=norm, smoke=args.smoke,
                  native_shape=[128,128], train_rows=len(data.rows['train']),
                  test_targets_read=False, target_payload_hash_verified=False, started_epoch=1, versions=dict(torch=str(torch.__version__),numpy=np.__version__))
    write_json(args.out/'run.json',report)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    start=time.monotonic(); history=[]
    for epoch in range(1,cfg['epochs']+1):
        model.train(); order=data.rows['train'][np.random.permutation(len(data.rows['train']))]
        total_loss=0.; count=0
        for rows in blocks(order,cfg['effective_batch']):
            optimizer.zero_grad(set_to_none=True); batch_loss=0.
            for micro in blocks(rows,cfg['microbatch']):
                x=data.x(micro,norm,device); y=data.y(micro,norm)
                if args.model=='deeponet' and cfg['coordinate_batch']<16384:
                    loss=backward_deeponet(model,x,y,cfg['coordinate_batch'],len(micro)/len(rows))
                else:
                    loss=relative_squared_loss(model(x),y.to(device))*(len(micro)/len(rows))
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite relative field loss.')
                    loss.backward()
                batch_loss+=float(loss.detach())
            checks=[p.grad.isfinite().all() for p in model.parameters() if p.grad is not None]
            if checks and not torch.stack(checks).all():
                raise FloatingPointError('Nonfinite training gradient.')
            optimizer.step(); total_loss+=batch_loss*len(rows); count+=len(rows)
        record=dict(epoch=epoch,training_relative_squared_field_loss=total_loss/count,optimized_rows=count,
                    elapsed_seconds=time.monotonic()-start)
        history.append(record)
        if epoch==1 or epoch%10==0 or epoch==cfg['epochs']:
            write_json(args.out/'progress.json',record); print(json.dumps(record),flush=True)
    model.eval()
    training_seconds=time.monotonic()-start
    state=dict(format='darcy_unit_baseline_v1', model_state=tensor_state(model), recipe=cfg,
               coords=data.coords, normalization=norm, data_manifest_sha256=data.manifest_sha256,
               source_sha256=source_hashes, training_history=history, training_seconds=training_seconds, test_used_for_training_or_selection=False)
    checkpoint=args.out/'checkpoint.pt'; torch.save(state,checkpoint); checkpoint_sha=sha256(checkpoint)
    restored,loaded=load_checkpoint(checkpoint,args.source,args.cno_source,args.device)
    probe=data.x(data.rows['val'][:2],norm,device)
    with torch.inference_mode(): first=model(probe); second=restored(probe)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    cpu,_=load_checkpoint(checkpoint,args.source,args.cno_source,'cpu')
    with torch.inference_mode(): cpu_prediction=cpu(probe.cpu())
    torch.testing.assert_close(first.cpu(),cpu_prediction,rtol=5e-4,atol=5e-5)
    parity=dict(saved_model_state_exact=True,same_device_prediction_max_abs=float(abs(first-second).max()),
                cpu_prediction_max_abs=float(abs(first.cpu()-cpu_prediction).max()),rows=len(probe),
                checkpoint_sha256=checkpoint_sha,native_output_points=16384)
    raw=torch.from_numpy(np.array(data.a[data.rows['val'][:2]],dtype='f4',copy=True)).to(device)
    beta=raw.new_tensor(BETA_SEEN[:len(raw)]).reshape(-1,1)
    physical=FrozenPhysicalPredictor(restored,norm)
    timed_boundary=physical(raw,beta)
    reference=second*norm['y_rms']*beta
    torch.testing.assert_close(timed_boundary,reference,rtol=5e-4,atol=5e-8)
    retained=physical(raw[:1],beta[:1]); clone=retained.clone(); another=physical(raw[1:2],beta[1:2])
    if retained.data_ptr()==another.data_ptr() or not torch.equal(retained,clone):
        raise ValueError('Physical inference reused a mutable output buffer.')
    parity.update(raw_to_physical_max_abs=float(abs(timed_boundary-reference).max()),
                  independent_output=True,tf32=False)
    write_json(args.out/'reload_parity.json',parity)
    report['val']=evaluate(restored,data,data.rows['val'],norm,device,8,args.out,'val')
    report['val']['seed']=args.seed
    write_json(args.out/'val_summary.json',report['val'])
    report.update(status='complete',checkpoint_sha256=checkpoint_sha,reload_parity=parity,
                  elapsed_seconds=time.monotonic()-start,training_seconds=training_seconds,test_targets_read=False)
    if args.evaluate_test:
        if sha256(args.data/'unit_pressure.npy') != data.manifest['files']['unit_pressure.npy']['sha256']:
            raise ValueError('Locked test target payload hash differs.')
        report['test']=evaluate(restored,data,data.rows['test'],norm,device,8,args.out,'test')
        report['test']['seed']=args.seed
        write_json(args.out/'test_summary.json',report['test'])
        report['test_targets_read']=True; report['target_payload_hash_verified']=True
    current_target=(args.data/'unit_pressure.npy').stat()
    if (current_target.st_size,current_target.st_mtime_ns,current_target.st_ino) != (data.target_fingerprint.st_size,data.target_fingerprint.st_mtime_ns,data.target_fingerprint.st_ino):
        raise ValueError('Unit-pressure payload changed during the run.')
    if sha256(checkpoint)!=checkpoint_sha or sha256(args.data/'manifest.json')!=data.manifest_sha256:
        raise ValueError('Frozen checkpoint or data manifest changed during evaluation.')
    if device.type=='cuda':
        report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    write_json(args.out/'run.json',report)
    write_json(args.out/'COMPLETE.json',dict(status='smoke_complete' if args.smoke else 'complete',
        checkpoint_sha256=checkpoint_sha,data_manifest_sha256=data.manifest_sha256,
        test_evaluated=args.evaluate_test,epochs=cfg['epochs']))
    print(json.dumps(dict(status=report['status'],model=args.model,checkpoint_sha256=checkpoint_sha)),flush=True)


def evaluate_existing(args):
    torch.set_num_threads(2)
    if args.device=='cuda':
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,(12*(1<<30))/total))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=args.checkpoint or args.out/'checkpoint.pt'
    checkpoint_sha=sha256(checkpoint)
    model,state=load_checkpoint(checkpoint,args.source,args.cno_source,args.device)
    data=DarcyData(args.data,args.smoke)
    if args.smoke:
        data.rows={name:rows[:4 if name=='train' else 2] for name,rows in data.rows.items()}
    if state['data_manifest_sha256']!=data.manifest_sha256 or state['recipe']['model']!=args.model:
        raise ValueError('Frozen checkpoint is bound to another model or dataset.')
    split=args.evaluate_only
    if (args.out/(split+'_summary.json')).exists():
        raise FileExistsError(args.out/(split+'_summary.json'))
    if split=='test' and sha256(args.data/'unit_pressure.npy')!=data.manifest['files']['unit_pressure.npy']['sha256']:
        raise ValueError('Locked test target payload hash differs.')
    summary=evaluate(model,data,data.rows[split],state['normalization'],torch.device(args.device),8,args.out,split)
    summary.update(seed=state['recipe']['seed'],checkpoint_sha256=checkpoint_sha)
    write_json(args.out/(split+'_summary.json'),summary)
    if sha256(checkpoint)!=checkpoint_sha:
        raise ValueError('Frozen checkpoint changed during evaluation.')
    run_path=args.out/'run.json'
    if run_path.exists():
        run=json.loads(run_path.read_text());run[split]=summary
        if split=='test':
            run.update(test_targets_read=True,target_payload_hash_verified=True,
                       test_evaluated_after_checkpoint_sha256=checkpoint_sha)
        write_json(run_path,run)
    complete_path=args.out/'COMPLETE.json'
    if complete_path.exists():
        complete=json.loads(complete_path.read_text())
        complete['test_evaluated']=complete.get('test_evaluated',False) or split=='test'
        write_json(complete_path,complete)
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','source','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=('fno','cno','deeponet'),required=True)
    p.add_argument('--cno-source',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--evaluate-test',action='store_true')
    p.add_argument('--evaluate-only',choices=('val','test'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not 0<=args.seed<2**32:p.error('Seed must be a valid unsigned32 integer.')
    if args.evaluate_only:
        evaluate_existing(args)
    else:
        train(args)


if __name__=='__main__':main()
