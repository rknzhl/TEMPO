"""Darcy operators: raw permeability and beta -> physical pressure beta*v(a)."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from models.pod import PODConfig,PODTrainer
from models.fourier_neural_pod import FourierNeuralPODConfig,FourierNeuralPODTrainer
from models.regime_basis import FourierRegimeBasis
from models.tempo_online import TEMPOOnlineConfig,build_tempo_online
from models.pod_deeponet import BranchNet

SIZE=128
DIM=SIZE*SIZE


def restore(module,saved):
    module.load_state_dict(saved,strict=True)
    actual=module.state_dict()
    if set(actual)!=set(saved) or any(not torch.equal(actual[key].cpu(),value.cpu()) for key,value in saved.items()):
        raise ValueError('Restored tensors differ from the frozen checkpoint.')
    return module.eval()


class GlobalOperator(torch.nn.Module):
    """Reconstruct a field from one branch and its fixed basis."""
    def __init__(self,inputs,kappa_dim,rank,cfg):
        super().__init__()
        self.branch=BranchNet(inputs,rank,cfg.hidden_dim,cfg.n_layers,d_kappa=kappa_dim)

    def forward(self,x,k,means,modes):
        field=means[0]+self.branch(x,k)@modes[0].T
        return field,field.new_ones((len(field),1))


def _features(raw,state):
    raw=raw.reshape(-1,SIZE,SIZE)
    name=state.get('input_transform',{}).get('name','full')
    if name=='full':small=raw.reshape(len(raw),DIM)
    elif name=='flat_stride4':small=raw.reshape(len(raw),DIM)[:,::4].contiguous()
    elif name=='block_mean4':small=raw.reshape(-1,32,4,32,4).mean(dim=(2,4)).reshape(len(raw),-1)
    else:raise ValueError('Unknown saved permeability transform.')
    norm=state['normalization']
    return (small-norm['x_mean'])/norm['x_std_population']


def _bases(saved,device):
    trainers=[]
    for item in saved:
        weights=item['state'];rank=item['rank']
        if item['kind']=='pod':
            trainer=PODTrainer(PODConfig(**item['config']))
            trainer.basis.initialize(weights['mean'],weights['modes'],weights['coeffs'])
        elif item['kind']=='fourier':
            basis=FourierRegimeBasis(**item['architecture'],quad_weights=weights['quad_weights'])
            for _ in range(rank):basis.add_mode()
            for module in basis.modules():module._non_persistent_buffers_set.discard('B')
            trainer=FourierNeuralPODTrainer(basis,FourierNeuralPODConfig(**item['config']))
        else:raise ValueError('Unknown frozen Darcy basis kind.')
        restore(trainer.basis,weights).to(device)
        trainer.num_modes=rank;trainers.append(trainer)
    return trainers


@dataclass
class DarcyModel:
    model:torch.nn.Module
    trainers:list
    state:dict
    coords:torch.Tensor
    basis:tuple=None

    @torch.inference_mode()
    def _physical(self,raw,beta):
        x=_features(raw,self.state)
        means,modes=self.basis if self.basis is not None else self.cache_basis()
        unit,gate=self.model(x,x.new_empty((len(x),0)),means,modes)
        return unit*self.state['normalization']['y_rms']*beta.reshape(-1,1),gate

    @torch.inference_mode()
    def cache_basis(self,coordinate_batch=8192):
        means,modes=[],[]
        for trainer in self.trainers:
            basis=trainer.basis
            if isinstance(trainer,PODTrainer):
                means.append(basis.mean);modes.append(basis.modes)
            else:
                def values(net):return torch.cat([net(c) for c in self.coords.split(coordinate_batch)])
                means.append(values(basis.mean_net))
                modes.append(torch.stack([values(mode.phi) for mode in basis.modes],1))
        self.basis=(means,modes)
        return self.basis

    @torch.inference_mode()
    def predict(self,permeability,beta=1.):
        """Return physical pressure(B,16384) and all soft gate weights(B,M)."""
        device=next(self.model.parameters()).device
        raw=torch.as_tensor(permeability,dtype=torch.float32,device=device)
        if raw.ndim==1 and raw.shape==(DIM,):raw=raw[None]
        elif raw.ndim==2 and raw.shape==(SIZE,SIZE):raw=raw.reshape(1,DIM)
        elif raw.ndim==3 and raw.shape[1:]==(SIZE,SIZE):raw=raw.flatten(1)
        elif raw.ndim!=2 or raw.shape[1]!=DIM:raise ValueError('Expected raw128x128 fields or flattened16384 rows.')
        beta=torch.as_tensor(beta,dtype=torch.float32,device=device)
        if beta.numel() not in (1,len(raw)) or not torch.isfinite(beta).all() or not (beta>0).all():
            raise ValueError('Beta must be finite, positive and scalar or one value per field.')
        if not torch.isfinite(raw).all() or not (raw>0).all():raise ValueError('Permeability must be finite and positive.')
        return self._physical(raw,beta)


def load_model(checkpoint,device='cpu'):
    """Load saved weights and normalization."""
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('format') not in ('darcy_shape_complete_v1','darcy_shape_sensor_v1'):
        raise ValueError('Unsupported frozen Darcy reduced checkpoint format.')
    inputs=DIM
    if state['format']=='darcy_shape_sensor_v1':
        transform=state['input_transform'];inputs={'flat_stride4':4096,'block_mean4':1024}.get(transform['name'])
        if inputs is None or (state['input_dim'],transform['dimension'])!=(inputs,inputs) or (state['output_dim'],transform['native_output_dimension'])!=(DIM,DIM):
            raise ValueError('Saved sensor construction differs from the native-output contract.')
    if state.get('beta_is_network_input') is not False:raise ValueError('Unexpected beta-input contract.')
    conditioning=state.get('coefficient_conditioning',{})
    if conditioning and conditioning.get('folded') is not True:raise ValueError('Final coefficient affine must already be folded into the head.')
    norm=state['normalization']
    if norm['x_std_population']<=0 or norm['y_rms']<=0 or norm.get('target_centered') is not False:
        raise ValueError('Invalid saved Darcy normalization.')
    coords=state['coords'];axis=(torch.arange(SIZE,dtype=torch.float32)+.5)/SIZE
    xx,yy=torch.meshgrid(axis,axis,indexing='ij')
    if not torch.equal(coords,torch.stack((xx.flatten(),yy.flatten()),1)):
        raise ValueError('Checkpoint must retain all native128 cell centers in ij order.')
    device=torch.device(device);trainers=_bases(state['offline']['bases'],device)
    cfg=TEMPOOnlineConfig(**state['online_config'])
    if cfg.sensor_stride!=1:raise ValueError('The saved head must use already constructed features.')
    if state['global_model']:
        if len(trainers)!=1:raise ValueError('Global checkpoint must contain one basis.')
        model=GlobalOperator(inputs,0,trainers[0].num_modes,cfg)
    else:model,_=build_tempo_online(trainers,0,inputs,cfg)
    restore(model,state['online_state']).to(device)
    result=DarcyModel(model,trainers,state,coords.to(device));result.cache_basis()
    return result


def make_raw_predictor(checkpoint,device='cpu'):
    """Cache fixed-grid basis values for physical-pressure prediction."""
    fitted=load_model(checkpoint,device)
    @torch.inference_mode()
    def predict(raw,beta=1.):
        raw=torch.as_tensor(raw,dtype=torch.float32,device=fitted.coords.device)
        beta=torch.as_tensor(beta,dtype=torch.float32,device=fitted.coords.device)
        return fitted._physical(raw,beta)[0]
    predict.fitted=fitted
    return predict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input-npz',type=Path,required=True,help='NPZ containing raw a and optional beta(default1).')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Output already exists.')
    with np.load(args.input_npz,allow_pickle=False) as data:
        permeability=data['a'].copy();beta=data['beta'].copy() if 'beta' in data else 1.
    fitted=load_model(args.checkpoint,args.device);pressure,gate=fitted.predict(permeability,beta)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,u=pressure.cpu().numpy().reshape(-1,SIZE,SIZE),gate=gate.cpu().numpy(),beta=np.asarray(beta))


if __name__=='__main__':main()
