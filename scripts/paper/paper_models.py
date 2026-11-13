"""Load saved paper models and their input and output transforms."""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from models.pod_deeponet import BranchNet
from models.neural_pod_mode import _mlp
from models.tempo_online import GatingNet, TEMPOOnline
from scripts.paper.run_comparison import load_fit

CNO_SHA256 = '30c1141b050eadd6f74ba34e33039a1488e498a8c1774bfa4ab09f75f323f47c'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def restore(module, saved):
    module.load_state_dict(saved, strict=True)
    actual = dict(module.state_dict())
    actual.pop('_metadata', None)  # neuraloperator constructor metadata is not a tensor weight.
    if set(actual) != set(saved) or any(not torch.equal(actual[k].cpu(), v.cpu()) for k, v in saved.items()):
        raise ValueError('Reloaded tensor state differs from the frozen checkpoint.')
    return module.eval()


class OrdinaryDeepONet(nn.Module):
    """DeepONet contraction with its trained scalar bias."""
    def __init__(self, architecture):
        super().__init__()
        a = architecture
        if a['activation'] != 'Tanh' or a['parameter_dim'] != 1:
            raise ValueError('Unsupported ordinary DeepONet architecture.')
        self.branch = BranchNet(a['input_dim'], a['latent_dim'], a['hidden_dim'], a['layers'], 1)
        self.trunk = _mlp(a['coordinate_dim'], a['latent_dim'], a['hidden_dim'], a['layers'])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, k, trunk):
        return self.branch(x, k) @ trunk.T + self.bias


def author_operator(name, cno_source=None):
    if name == 'fno':
        if importlib.metadata.version('neuraloperator') != '2.0.0':
            raise ValueError('The saved FNO recipe requires neuraloperator==2.0.0.')
        from neuralop.models import FNO
        return FNO
    if name != 'cno' or cno_source is None:
        raise ValueError('CNO replay requires --cno-source pointing to the original CNO2d.py.')
    source = Path(cno_source)
    if sha256(source) != CNO_SHA256:
        raise ValueError('CNO2d.py differs from the frozen author source.')
    spec = importlib.util.spec_from_file_location('paper_author_cno', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CNO2d


class AuthorOperator(nn.Module):
    """FNO/CNO input and output transforms for Sod, Coanda and Helmholtz."""
    def __init__(self, state, cno_source=None):
        super().__init__()
        cfg = self.config = state['config']
        dataset, name = cfg['dataset'], cfg['model']
        if dataset not in ('sod', 'coanda_reference_grid', 'helmholtz_four_omega'):
            raise ValueError('This author adapter only covers the current paper datasets.')
        cls = author_operator(name, cno_source)
        if name == 'fno':
            package = Path(inspect.getfile(cls)).resolve().parents[1]
            expected = {Path(k).name: v for k, v in state['source_hashes'].items()}
            for relative in ('models/fno.py', 'layers/fno_block.py',
                             'layers/spectral_convolution.py', 'layers/channel_mlp.py',
                             'layers/skip_connections.py'):
                source = package / relative
                if source.name in expected and sha256(source) != expected[source.name]:
                    raise ValueError('Frozen FNO source differs: '+source.name)
            self.operator = cls(n_modes=(16, 16), in_channels=cfg['in_channels'],
                out_channels=cfg['out_channels'], hidden_channels=32, n_layers=4,
                use_channel_mlp=True, positional_embedding=None)
        else:
            size = {'sod': 128, 'coanda_reference_grid': 256, 'helmholtz_four_omega': 101}[dataset]
            self.operator = cls(in_dim=cfg['in_channels'], out_dim=cfg['out_channels'], size=size,
                N_layers=4, N_res=4, N_res_neck=4, channel_multiplier=16, use_bn=False)
        self.register_buffer('grid', state['grid'][None].clone())
        if dataset == 'coanda_reference_grid':
            self.register_buffer('readout_grid', state['model_state']['readout_grid'].clone())

    def forward(self, x, k):
        n, cfg = len(x), self.config
        kind = cfg['dataset']
        if kind == 'sod':
            nx, nt = cfg['nx'], cfg['nt']
            initial = x[:, :, None, None].expand(-1, -1, nx, nt)
            features = torch.cat((initial, k[:, :, None, None].expand(-1, -1, nx, nt),
                                  self.grid.expand(n, -1, -1, -1)), 1)
            if cfg['model'] == 'cno':
                features = F.interpolate(features, (128, 128), mode='bilinear', align_corners=False)
            output = self.operator(features)
            if cfg['model'] == 'cno':
                output = F.interpolate(output, (nx, nt), mode='bilinear', align_corners=False)
            output = output.permute(0, 1, 3, 2).reshape(n, -1)
        elif kind == 'coanda_reference_grid':
            features = torch.cat((x[:, :, None, None].expand(-1, -1, 256, 256),
                k[:, :, None, None].expand(-1, -1, 256, 256), self.grid.expand(n, -1, -1, -1)), 1)
            raster = self.operator(features)
            output = F.grid_sample(raster, self.readout_grid.expand(n, -1, -1, -1),
                mode='bilinear', padding_mode='zeros', align_corners=True).reshape(n, -1)
        else:
            features = torch.cat((x.reshape(n, 1, 101, 101),
                k[:, :, None, None].expand(-1, -1, 101, 101), self.grid.expand(n, -1, -1, -1)), 1)
            output = self.operator(features)
            if output.shape != (n, 1, 101, 101):
                raise ValueError('Author operator changed the native output shape.')
            output = output.reshape(n, -1)
        return output, output.new_ones((n, 1))


@dataclass
class PaperModel:
    model: nn.Module
    state: dict
    coords: torch.Tensor
    xdim: int
    pdim: int
    format: str
    basis: object = None

    @torch.inference_mode()
    def predict(self, x, k, coordinate_batch=4096):
        if self.format.startswith('darcy_'):
            if self.format in ('darcy_shape_complete_v1', 'darcy_shape_sensor_v1'):
                return self.basis.predict(x, k)
            output = self.basis(x, k) if self.basis is not None else self.model(x, k)
            return output, output.new_ones((len(output), 1))
        if self.format == 'ns_global_pod':
            mean, modes = self.basis
            output = mean + (self.model(x, k)*self.state['coefficient_std'].to(x.device)) @ modes.T
            return output, output.new_ones((len(output), 1))
        if self.format == 'revision_basis_fit':
            means, modes = [], []
            from models.pod import PODTrainer
            for trainer in self.basis.bases:
                if isinstance(trainer, PODTrainer):
                    means.append(trainer.basis.mean)
                    modes.append(trainer.basis.modes)
                else:
                    basis = trainer.basis
                    means.append(torch.cat([basis.mean_net(c) for c in self.coords.split(coordinate_batch)]))
                    modes.append(torch.stack([torch.cat([m.phi(c) for c in self.coords.split(coordinate_batch)])
                                              for m in basis.modes], 1))
            return self.model(x, k, means, modes)
        if self.format == 'ordinary_deeponet_native_v1':
            coords = (self.coords - self.state['coordinate_min'].to(x.device)) / self.state['coordinate_range'].to(x.device)
            trunk = torch.cat([self.model.trunk(c) for c in coords.split(coordinate_batch)])
            output = self.model(x, k, trunk)
        elif self.format in ('ns_original_deeponet_complete_v1', 'burgers_deeponet_complete_v1'):
            output = self.model(x, k, self.coords)
        elif self.format == 'ns_original_tempo_pod_complete_v1':
            means, modes = self.basis
            return self.model(x, k, means, modes)
        elif self.format in ('ns_tempo_npod_complete_v1', 'burgers_aux_tempo_npod_complete_v1'):
            means = [torch.cat([basis.mean_net(c) for c in self.coords.split(coordinate_batch)])
                     for basis in self.basis]
            modes = [torch.stack([torch.cat([mode.phi(c) for c in self.coords.split(coordinate_batch)])
                                  for mode in basis.modes], 1) for basis in self.basis]
            prediction, gate = self.model(x, k, means, modes)
            if self.format == 'ns_tempo_npod_complete_v1':
                prediction = prediction * self.state['normalization']['s_scale']
            return prediction, gate
        elif self.format == 'burgers_global_pod':
            mean, modes = self.basis
            coefficients = self.model(x, k)
            coefficients = coefficients * self.state['coefficient_scale'].to(x.device) + self.state['coefficient_offset'].to(x.device)
            output = mean + coefficients @ modes.T
        elif self.format in ('burgers_global_neuralpod', 'fourier_deeponet_complete_v1'):
            basis = self.basis
            mean = torch.cat([basis.mean_net(c) for c in self.coords.split(coordinate_batch)])
            modes = torch.stack([torch.cat([m.phi(c) for c in self.coords.split(coordinate_batch)])
                                 for m in basis.modes], 1)
            output = mean + self.model(x, k) @ modes.T
        else:
            return self.model(x, k)
        return output, output.new_ones((len(x), 1))

    def normalize_inputs(self, x, parameter):
        """Apply saved input normalization; dense Helmholtz inputs are already prepared."""
        if self.format.startswith('darcy_') or self.format == 'ns_global_pod':
            return x, parameter
        if self.format in ('ns_original_tempo_pod_complete_v1', 'ns_tempo_npod_complete_v1'):
            if (parameter <= 0).any():
                raise ValueError('Reynolds numbers must be positive.')
            scale = (self.state['input_scaling'] if self.format == 'ns_original_tempo_pod_complete_v1'
                     else self.state['normalization']['input_scaling'])
            k = (parameter.log() - scale['logRe_train_mean']) / scale['logRe_train_std_population']
            return x, k
        if self.format in ('burgers_global_pod', 'burgers_global_neuralpod',
                           'ns_original_deeponet_complete_v1', 'burgers_deeponet_complete_v1',
                           'fourier_deeponet_complete_v1', 'burgers_aux_tempo_npod_complete_v1'):
            return x, parameter
        scale = self.state['scale']
        if not {'x_scale', 'p_mean', 'p_scale'} <= scale.keys():
            raise ValueError('Use normalized mode with the saved prepared-field input contract.')
        transform = scale.get('p_transform', self.state.get('fit_log', {}).get('parameter_transform', 'raw'))
        if transform == 'log':
            if (parameter <= 0).any():
                raise ValueError('Log-encoded physical parameters must be positive.')
            parameter = parameter.log()
        elif transform != 'raw':
            raise ValueError('Unknown saved parameter transform.')
        if self.format == 'author_operator':
            parameter = parameter.double()
        return (x-scale['x_mean'])/scale['x_scale'], ((parameter-scale['p_mean'])/scale['p_scale']).float()[:, :self.pdim]

    def prepared_output(self, prediction):
        """Restore the prepared-data target units."""
        if self.format.startswith('darcy_') or self.format == 'ns_global_pod':
            return prediction
        if self.format in ('ns_original_tempo_pod_complete_v1', 'ns_tempo_npod_complete_v1',
                           'ns_original_deeponet_complete_v1', 'burgers_deeponet_complete_v1',
                           'fourier_deeponet_complete_v1', 'burgers_aux_tempo_npod_complete_v1',
                           'burgers_global_pod', 'burgers_global_neuralpod'):
            return prediction
        scale = self.state['scale']
        if 'y_scale' in scale:
            return prediction.double()*scale['y_scale'] + scale.get('y_mean', 0.)
        if 'y_std' in scale:
            return prediction.double()*float(scale['y_std'][0]) + float(scale['y_mean'][0])
        raise ValueError('Checkpoint lacks an explicit target inverse transform.')

    def physical_output(self, prediction):
        """Restore physical output units."""
        prepared = self.prepared_output(prediction)
        if self.format == 'ordinary_deeponet_native_v1':
            return (prepared*self.state['physical_gain'].to(prediction.device).double()
                    + self.state['physical_offset'].to(prediction.device).double())
        is_sod = (self.state.get('config', {}).get('dataset') == 'sod' or
                  (self.xdim == 6 and self.coords.shape == (6144, 3)))
        if is_sod:
            from scripts.paper.sod import GAMMA
            gain = prepared.new_tensor([1., GAMMA**.5, 1.])
            return (prepared.reshape(len(prepared), 3, -1)*gain[None, :, None]).reshape(len(prepared), -1)
        return prepared


def load_paper_model(checkpoint, device='cpu', cno_source=None):
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    device = torch.device(device)
    if 'fit_log' in state:
        fitted = load_fit(checkpoint, device)
        restore(fitted.model, state['online'])
        for trainer, saved in zip(fitted.bases, state['bases']):
            restore(trainer.basis, {**saved['weights'], **saved.get('frequencies', {})})
            buffers = dict(trainer.basis.named_buffers())
            for key, value in saved.get('frequencies', {}).items():
                if not torch.equal(buffers[key].cpu(), value.cpu()):
                    raise ValueError('Fourier frequencies were not restored exactly.')
        return PaperModel(fitted.model, state, state['coords'].to(device), state['xdim'], state['pdim'], 'revision_basis_fit', fitted)
    fmt = state.get('format')
    if fmt in ('darcy_shape_complete_v1', 'darcy_shape_sensor_v1'):
        from scripts.paper.darcy import load_model
        fitted = load_model(checkpoint, device)
        return PaperModel(fitted.model, state, fitted.coords, 16384, 1, fmt, fitted)
    if fmt == 'darcy_unit_baseline_v1':
        from scripts.paper.darcy_baselines import build, FrozenPhysicalPredictor
        model = restore(build(state['recipe']['model'], state['coords'], cno_source), state['model_state']).to(device)
        predictor = FrozenPhysicalPredictor(model, state['normalization'])
        return PaperModel(model, state, state['coords'].to(device), 16384, 1, fmt, predictor)
    if fmt == 'darcy_direct_cno_historical_v1':
        from scripts.paper.frozen_cno_unit import load_model
        model = load_model(checkpoint, cno_source, device)
        return PaperModel(model, state, state['coords'].to(device), 16384, 1, fmt)
    if {'branch', 'mean', 'modes', 'coefficient_std'} <= state.keys():
        cfg = state['recipe']
        model = restore(BranchNet(8192, cfg['rank'], cfg['hidden'], cfg['layers'], 1), state['branch']).to(device)
        tt, xx, yy, cc = torch.meshgrid(torch.arange(21)/20, torch.arange(64)/64,
            torch.arange(64)/64, torch.tensor([0., 1.]), indexing='ij')
        coords = torch.stack((xx.flatten(), yy.flatten(), tt.flatten(), cc.flatten()), 1)
        if state['mean'].shape != (172032,) or state['modes'].shape != (172032, cfg['rank']):
            raise ValueError('The NS POD control requires the original 21x64x64x2 output grid.')
        return PaperModel(model, state, coords.to(device), 8192, 1, 'ns_global_pod',
                          (state['mean'].to(device), state['modes'].to(device)))
    if fmt == 'ordinary_deeponet_native_v1':
        model = restore(OrdinaryDeepONet(state['architecture']), state['state_dict']).to(device)
        a = state['architecture']
        return PaperModel(model, state, state['coords'].to(device), a['input_dim'], a['parameter_dim'], fmt)
    if fmt == 'ns_original_deeponet_complete_v1':
        from scripts.paper.train_ns_deeponet import build_model, coordinates
        if not torch.equal(state['coords'], coordinates()) or state['output_components'] != 2:
            raise ValueError('NS DeepONet requires the original native coordinate grid.')
        model = restore(build_model(state['recipe']), state['state_dict']).to(device)
        return PaperModel(model, state, state['coords'].to(device), 8192, 1, fmt)
    if fmt == 'burgers_deeponet_complete_v1':
        from scripts.burgers.train_deeponet_burgers import BranchNet as DeepBranch, TrunkNet, DeepONet
        saved = state['state_dict']
        width = saved['branch.net.0.weight'].shape[0]
        layers = sum(k.startswith('branch.') and k.endswith('.weight') for k in saved)-1
        model = DeepONet(DeepBranch(state['input_dim'], width, width, layers),
                        TrunkNet(width, width, layers), width)
        if state['coords'].shape != (int(torch.tensor(state['output_shape']).prod()), 2):
            raise ValueError('Burgers DeepONet coordinate grid differs from its saved output.')
        restore(model, saved)
        return PaperModel(model.to(device), state, state['coords'].to(device), state['input_dim'], 1, fmt)
    if fmt == 'burgers_aux_tempo_npod_complete_v1':
        from models.regime_basis import FourierRegimeBasis
        bases = []
        for saved in state['bases']:
            basis = FourierRegimeBasis(**saved['config'], quad_weights=saved['state']['quad_weights'])
            for _ in range(saved['rank']):
                basis.add_mode()
            bases.append(restore(basis, saved['state']).to(device))
        cfg = state['online_config']
        model = TEMPOOnline(GatingNet(101, 1, 3, cfg['hidden_dim'], cfg['n_layers']),
            nn.ModuleList([BranchNet(101, b.num_modes, cfg['hidden_dim'], cfg['n_layers'], 1) for b in bases]))
        restore(model, state['online'])
        if len(bases) != 3 or state['coords'].shape != (25856, 2):
            raise ValueError('The supplementary Burgers model requires three bases on the 256x101 grid.')
        return PaperModel(model.to(device), state, state['coords'].to(device), 101, 1, fmt, nn.ModuleList(bases))
    if fmt == 'ns_original_tempo_pod_complete_v1':
        cfg = state['online_config']
        bases = state['offline']['bases_raw']
        ranks = [v['modes'].shape[1] for v in bases]
        if ranks != [64]*4:
            raise ValueError('The selected NS checkpoint requires all four rank-64 experts.')
        model = TEMPOOnline(GatingNet(8192, 1, 4, cfg['hidden_dim'], cfg['n_layers']),
            nn.ModuleList([BranchNet(8192, rank, cfg['hidden_dim'], cfg['n_layers'], 1) for rank in ranks]))
        restore(model, state['online_state'])
        basis = ([v['mean'].to(device) for v in bases], [v['modes'].to(device) for v in bases])
        return PaperModel(model.to(device), state, state['coords'].to(device), 8192, 1, fmt, basis)
    if fmt == 'ns_tempo_npod_complete_v1':
        from models.regime_basis import FourierRegimeBasis
        saved_bases, online = state['offline']['bases'], state['online']
        if len(saved_bases) != 4 or any(saved['rank'] != 64 for saved in saved_bases):
            raise ValueError('The selected NS NeuralPOD checkpoint requires four rank-64 experts.')
        if not online.get('conditioning_folded', False):
            raise ValueError('NS NeuralPOD replay requires the exported coefficient inverse.')
        bases = []
        for saved in saved_bases:
            basis = FourierRegimeBasis(**saved['basis_config'], quad_weights=saved['state']['quad_weights'])
            for _ in range(saved['rank']):
                basis.add_mode()
            for module in basis.modules():
                module._non_persistent_buffers_set.discard('B')
            bases.append(restore(basis, saved['state']).to(device))
        cfg = online['online_config']
        model = TEMPOOnline(GatingNet(8192, 1, 4, cfg['hidden_dim'], cfg['n_layers']),
            nn.ModuleList([BranchNet(8192, 64, cfg['hidden_dim'], cfg['n_layers'], 1) for _ in bases]))
        restore(model, online['online_state'])
        tt, xx, yy, cc = torch.meshgrid(torch.arange(21, dtype=torch.float32) / 20,
            torch.arange(64, dtype=torch.float32) / 64, torch.arange(64, dtype=torch.float32) / 64,
            torch.tensor([0., 1.]), indexing='ij')
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1), cc.reshape(-1)], 1)
        return PaperModel(model.to(device), state, coords.to(device), 8192, 1, fmt, nn.ModuleList(bases))
    if 'basis_config' in state and 'branch_config' in state:
        from models.regime_basis import FourierRegimeBasis
        basis = FourierRegimeBasis(**state['basis_config'], quad_weights=state['basis']['quad_weights'])
        for _ in range(state['actual_rank']):
            basis.add_mode()
        for module in basis.modules():
            if 'B' in module._buffers:
                module._non_persistent_buffers_set.discard('B')
        restore(basis, state['basis'])
        branch = restore(BranchNet(**state['branch_config']), state['branch']).to(device)
        cfg = state['branch_config']
        return PaperModel(branch, state, state['coords'].to(device), cfg['m'], cfg['d_kappa'], fmt if fmt == 'fourier_deeponet_complete_v1' else 'burgers_global_neuralpod', basis.to(device))
    if 'rank' in state and 'coefficient_offset' in state:
        saved, cfg = state['model_state'], state['recipe']
        rank = state['rank']
        branch_state = {k.removeprefix('branch.'): v for k, v in saved.items() if k.startswith('branch.')}
        xdim = branch_state['net.0.weight'].shape[1]-1
        branch = restore(BranchNet(xdim, rank, cfg['hidden_dim'], cfg['n_layers'], 1), branch_state).to(device)
        # Coordinates come from the pinned prediction input.
        return PaperModel(branch, state, None, xdim, 1, 'burgers_global_pod',
                          (saved['basis.mean'].to(device), saved['basis.modes'].to(device)))
    if 'model_state' in state and 'config' in state:
        model = restore(AuthorOperator(state, cno_source), state['model_state']).to(device)
        cfg = state['config']
        xdim = {'sod': cfg['in_channels']-3, 'coanda_reference_grid': 1,
                'helmholtz_four_omega': 10201}[cfg['dataset']]
        return PaperModel(model, state, state['coords'].to(device), xdim, 1, 'author_operator')
    raise ValueError('Checkpoint is outside the supported current-paper formats.')


def validate_original_burgers_coordinates(coords):
    """Validate the native Burgers 201x1024 coordinate grid."""
    expected = 'e0e96fc51fae253d4639656d15fd86749c9b85efa159cdb539cd15fbe3b57a04'
    if coords.shape != (205824, 2) or hashlib.sha256(coords.cpu().numpy().tobytes()).hexdigest() != expected:
        raise ValueError('Original Burgers POD replay requires its pinned native t,x grid.')
