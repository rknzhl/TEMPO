"""Refine a saved TEMPO NeuralPOD model.

NPZ inputs use saved units: coords and train/val x, kappa, group and y.
Training rows must follow saved responsibilities; selection uses validation."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.neural_refinement import RefinementConfig, basis_values, refine_basis, target_batches
from scripts.paper.paper_models import load_paper_model, sha256


def write(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


@torch.no_grad()
def statistics(parts, B, q, scale, batch):
    G = B.T @ (q[:, None]*B)
    for p in parts.values():
        p['b'] = torch.empty(len(p['x']), B.shape[1], device=B.device, dtype=torch.float64)
        p['energy'] = torch.empty(len(p['x']), device=B.device, dtype=torch.float64)
        for ix, y in target_batches(p['y'], B.device, batch, scale):
            p['b'][ix] = y @ (q[:, None]*B)
            p['energy'][ix] = (y.square()*q).sum(1).clamp_min(1e-12)
    return G


def coefficients(model, p, ix):
    gate = model.gating(p['x'][ix], p['k'][ix])
    return torch.cat([gate[:, m:m+1]*torch.cat([torch.ones_like(gate[:, m:m+1]), b(p['x'][ix], p['k'][ix])], 1)
                      for m, b in enumerate(model.branches)], 1).double()


def residual(z, p, G, ix):
    return ((z@G*z).sum(1)-2*(z*p['b'][ix]).sum(1)+p['energy'][ix])/p['energy'][ix]


@torch.no_grad()
def calibrate_noise(part, G, gamma, ranks, relative):
    """Calibrate EM noise from train-only, conditional least-squares residuals."""
    errors, at = [], 0
    for rank in ranks:
        ix = slice(at+1, at+rank+1)
        cross = part['b'][:, ix]-G[at, ix]
        coeff = cross @ torch.linalg.pinv(G[ix, ix], rtol=1e-10, hermitian=True)
        error = part['energy']-2*part['b'][:, at]+G[at, at]-(coeff*cross).sum(1)
        if float(error.min()) < -1e-7:
            raise FloatingPointError('Negative projection residual during noise calibration.')
        error = error.clamp_min(0)
        errors.append(error/part['energy'] if relative else error)
        at += rank+1
    gamma = gamma.to(device=G.device, dtype=G.dtype)
    return max(float(.5*(gamma*torch.stack(errors, 1)).sum()/len(gamma)), 1e-12)


@torch.no_grad()
def evaluate(model, p, G):
    model.eval()
    e = []
    for lo in range(0, len(p['x']), 256):
        ix = slice(lo, lo+256)
        squared = residual(coefficients(model, p, ix), p, G, ix)
        if float(squared.min()) < -1e-7:
            raise FloatingPointError('Negative squared error: check target units and numerical precision.')
        e.append(squared.clamp_min(0).sqrt())
    return float(torch.cat(e).mean())


def train_online(model, parts, G, ranks, cfg):
    """Condition coefficients for training, then fold the transform into heads."""
    model = deepcopy(model)
    T = torch.eye(len(G), device=G.device, dtype=G.dtype)
    offsets, scales, start = [], [], 0
    with torch.no_grad():
        for branch, rank in zip(model.branches, ranks):
            ix = slice(start+1, start+rank+1)
            C = (parts['train']['b'][:, ix]-G[start, ix]) @ torch.linalg.pinv(G[ix, ix], rtol=1e-10, hermitian=True)
            offset, scale = C.mean(0).float(), C.std(0, unbiased=False).float().clamp_min(1e-4)
            branch.net[-1].weight.div_(scale[:, None]); branch.net[-1].bias.sub_(offset).div_(scale)
            T[ix, start] = offset.double(); T[ix, ix] = torch.diag(scale.double())
            offsets.append(offset); scales.append(scale); start += rank+1
    conditioned = {name:dict(p, b=p['b']@T) for name,p in parts.items()}
    gram = T.T@G@T
    initial = evaluate(model, conditioned['val'], gram)
    best, weights, epoch_best, bad, history = initial, deepcopy(model.state_dict()), 0, 0, []
    heads = [p for b in model.branches for p in b.net[-1].parameters()]
    hidden = [p for b in model.branches for layer in list(b.net.children())[:-1] for p in layer.parameters()]
    opt = torch.optim.AdamW([dict(params=model.gating.parameters(), lr=cfg['gate_lr']),
        dict(params=hidden, lr=cfg['hidden_lr']), dict(params=heads, lr=cfg['head_lr'])], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    for epoch in range(1, cfg['epochs']+1):
        model.train(); p = conditioned['train']; order = torch.randperm(len(p['x']), device=G.device)
        for lo in range(0, len(order), cfg['batch_size']):
            ix = order[lo:lo+cfg['batch_size']]
            opt.zero_grad(set_to_none=True)
            loss = residual(coefficients(model, p, ix), p, gram, ix).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite online loss.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg['epochs']:
            score = evaluate(model, conditioned['val'], gram)
            if score < best-1e-7:
                best, weights, epoch_best, bad = score, deepcopy(model.state_dict()), epoch, 0
            else:
                bad += 1 if epoch == 1 else 10
            history.append(dict(epoch=epoch, validation=score, best=best))
            if bad >= cfg['patience']: break
    model.load_state_dict(weights)
    with torch.no_grad():
        for branch, offset, scale in zip(model.branches, offsets, scales):
            branch.net[-1].weight.mul_(scale[:, None]); branch.net[-1].bias.mul_(scale).add_(offset)
    score = evaluate(model, parts['val'], G)
    if abs(score-best) > 2e-6:
        raise AssertionError('Exported coefficients differ from conditioned predictions.')
    return model, dict(initial=initial, final=score, selected_epoch=epoch_best, history=history)


def save_checkpoint(state, fmt, bases, model, gamma, record, path):
    saved = deepcopy(state)
    weights = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    if fmt == 'revision_basis_fit':
        saved['online'] = weights; saved['gamma'] = gamma.cpu()
        for item, b in zip(saved['bases'], bases):
            item['weights'] = {k:v.detach().cpu() for k,v in b.state_dict().items()}
        saved['residual_stats']['pi'] = gamma.mean(0).tolist()
        saved['fit_log']['joint_refinement'] = record
    else:
        saved['online']['online_state'] = weights
        saved['online']['conditioning_folded'] = True
        for item, b in zip(saved['offline']['bases'], bases):
            item['state'] = {k:v.detach().cpu() for k,v in b.state_dict().items()}
        saved['offline']['gamma'] = gamma.cpu(); saved['offline']['pi'] = gamma.mean(0).cpu()
        saved['joint_refinement'] = record
    temp = path.with_suffix('.partial'); torch.save(saved, temp); temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data', 'config', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--train-targets', type=Path); p.add_argument('--val-targets', type=Path)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--threads', type=int, choices=[1,2], default=2)
    a = p.parse_args(); torch.set_num_threads(a.threads)
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = json.loads(a.config.read_text())
    if 'phases' in config:
        raise ValueError('This file records a multi-stage fit. The continuation command accepts one basis/online schedule; use the trainer API for the recorded EM stages.')
    cfg = RefinementConfig(**config['basis'])
    torch.manual_seed(config.get('seed',17))
    if a.out.exists(): raise FileExistsError(a.out)
    frozen = load_paper_model(a.checkpoint, a.device)
    if frozen.format not in ('revision_basis_fit','ns_tempo_npod_complete_v1') or not hasattr(frozen.model,'branches'):
        raise ValueError('A TEMPO NeuralPOD checkpoint is required.')
    state = frozen.state
    if frozen.format == 'revision_basis_fit':
        bases = [t.basis for t in frozen.basis.bases]; gamma = state['gamma'].clone()
        scale = state['scale']['y_scale']; noise = state['residual_stats']
    else:
        bases = list(frozen.basis); gamma = state['offline']['gamma'].clone()
        scale = state['normalization']['s_scale']; noise = state['offline']
        if config.get('em_steps',0):
            raise ValueError('The stored NS continuation keeps responsibilities fixed; use the live TEMPO trainer API to change its EM schedule.')
    if any(not hasattr(b,'mean_net') for b in bases): raise ValueError('POD bases are not neural functions.')
    source_sha = sha256(a.checkpoint)
    if config.get('source_sha256') and source_sha != config['source_sha256']:
        raise ValueError('Initialization checkpoint differs from the configured source.')
    device = next(frozen.model.parameters()).device; coords = frozen.coords
    parts = {}
    with np.load(a.data, allow_pickle=False) as z:
        if not np.array_equal(z['coords'],coords.cpu().numpy()): raise ValueError('Output grid differs from checkpoint.')
        for name, targets in [('train',a.train_targets),('val',a.val_targets)]:
            y = np.load(targets,mmap_mode='r') if targets else z[name+'_y'].copy()
            x = torch.from_numpy(z[name+'_x'].copy()).float().to(device)
            k = torch.from_numpy(z[name+'_kappa'].copy()).float().to(device)
            if (x.shape != (len(y),frozen.xdim) or k.shape != (len(y),frozen.pdim)
                    or not torch.isfinite(x).all() or not torch.isfinite(k).all() or not len(y)):
                raise ValueError('Inputs and targets are not aligned.')
            parts[name] = dict(x=x,k=k,y=y,group=z[name+'_group'].astype(str))
        raw_parameter = torch.from_numpy(z['train_parameter'].copy()).float() if 'train_parameter' in z else None
    if len(gamma) != len(parts['train']['x']) or set(parts['train']['group']) & set(parts['val']['group']):
        raise ValueError('Training rows differ from gamma, or train/validation groups overlap.')
    if config.get('target_units','prepared') not in ('normalized','prepared'):
        raise ValueError('Target units must be prepared or normalized.')
    scale = 1. if config.get('target_units','prepared') == 'normalized' else scale
    if config.get('em_steps',0) and cfg.relative != noise.get('relative',False) and not config.get('recalibrate_noise',False):
        raise ValueError('EM and basis refinement must use the same noise convention.')
    q = bases[0].quad_weights.to(device).double()
    if any(not torch.equal(b.quad_weights.to(device).double(),q) for b in bases):
        raise ValueError('All experts must use the same quadrature.')
    ranks = [len(b.modes) for b in bases]
    old = [basis_values(b,coords,cfg.coordinate_batch) for b in bases]
    G = statistics(parts,torch.cat(old,1),q,scale,cfg.trajectory_batch)
    if config.get('recalibrate_noise',False):
        noise['sigma2'] = calibrate_noise(parts['train'], G, gamma, ranks, cfg.relative)
        noise['relative'] = cfg.relative
    baseline = evaluate(frozen.model,parts['val'],G)
    a.out.mkdir(parents=True)
    write(a.out/'config.json',config)
    history = []; best = baseline; best_path = a.checkpoint
    model = frozen.model
    for cycle in range(config.get('em_steps',0)+1):
        current = cfg if cycle == 0 else replace(cfg,steps=config.get('subsequent_steps',cfg.steps))
        basis_log = [refine_basis(b,parts['train']['y'],coords,gamma[:,m],current,target_scale=scale)
                     for m,b in enumerate(bases)]
        new = [basis_values(b,coords,cfg.coordinate_batch) for b in bases]
        with torch.no_grad():
            for branch,v0,v1 in zip(model.branches,old,new):
                phi = v1[:,1:]
                inverse = torch.linalg.pinv(phi.T@(q[:,None]*phi),rtol=cfg.rtol,hermitian=True)@(phi.T*q)
                T,d = inverse@v0[:,1:],inverse@(v0[:,0]-v1[:,0])
                head = branch.net[-1]
                head.weight.copy_((T@head.weight.double()).float()); head.bias.copy_((T@head.bias.double()+d).float())
        G = statistics(parts,torch.cat(new,1),q,scale,cfg.trajectory_batch)
        model,online_log = train_online(model,parts,G,ranks,config['online'])
        record = dict(cycle=cycle,basis=basis_log,online=online_log,config=config,source_sha256=source_sha,
                      selection='Validation only',test_used=False)
        path = a.out/('cycle_'+str(cycle)+'.pt')
        save_checkpoint(state,frozen.format,bases,model,gamma,record,path)
        history.append(dict(cycle=cycle,validation=online_log['final'],checkpoint=path.name))
        if online_log['final'] < best:
            best,best_path = online_log['final'],path
        write(a.out/'result.json',dict(baseline_validation=baseline,selected_validation=best,
            selected_checkpoint=str(best_path),history=history,test_used=False))
        if cycle < config.get('em_steps',0):
            sigma = float(noise['sigma2']); errors = []; at = 0
            with torch.no_grad():
                for b,rank in zip(bases,ranks):
                    C = torch.stack([m.lambda_ten for m in b.modes],1).double()
                    A = torch.cat([torch.ones_like(C[:,:1]),C],1);ix = slice(at,at+rank+1)
                    error = parts['train']['energy']-2*(A*parts['train']['b'][:,ix]).sum(1)+(A@G[ix,ix]*A).sum(1)
                    if cfg.relative: error = error/parts['train']['energy']
                    errors.append(error.clamp_min(0)); at += rank+1
                logp = gamma.mean(0).to(device).clamp_min(1e-30).log()[None]-torch.stack(errors,1)/(2*sigma)
                rho = float(noise.get('rho',0.))
                if rho:
                    if raw_parameter is None: raise ValueError('E-step prior requires train_parameter in physical units.')
                    center = torch.as_tensor(noise['centers'],device=device)
                    logprior = -.5*(raw_parameter.to(device).reshape(-1,1).log()-center[None]).square()/float(noise['prior_scale'])**2
                    logp += rho*(logprior-logprior.logsumexp(1,keepdim=True))
                gamma = logp.softmax(1).cpu()
        old = new
    if sha256(a.checkpoint) != source_sha: raise AssertionError('Initialization checkpoint changed.')
    print(json.dumps(dict(validation_before=baseline,validation_after=best,selected_checkpoint=str(best_path)),indent=2))


if __name__ == '__main__':
    main()
