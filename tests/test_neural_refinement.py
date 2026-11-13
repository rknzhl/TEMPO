"""Check the optimizer against direct weighted reconstruction on CPU."""
from copy import deepcopy
import unittest
import torch

from models.regime_basis import FourierRegimeBasis
from models.neural_refinement import RefinementConfig, refine_basis, functions, quadratic_loss, _moments


class NeuralRefinementTest(unittest.TestCase):
    def test_noise_calibration_matches_direct_projection(self):
        from scripts.paper.refine_neuralpod import calibrate_noise
        torch.set_num_threads(2)
        torch.manual_seed(19)
        y = torch.randn(9,17,dtype=torch.float64)
        B = torch.randn(17,8,dtype=torch.float64)
        q = torch.rand(17,dtype=torch.float64); q /= q.sum()
        G = B.T @ (q[:,None]*B)
        part = dict(b=y@(q[:,None]*B), energy=(y.square()*q).sum(1))
        gamma = torch.rand(9,2,dtype=torch.float64); gamma /= gamma.sum(1,keepdim=True)
        residuals = []
        for start in [0,4]:
            phi = B[:,start+1:start+4]
            c = torch.linalg.lstsq(q.sqrt()[:,None]*phi,
                (q.sqrt()[None,:]*(y-B[:,start])).T).solution.T
            residuals.append(((y-B[:,start]-c@phi.T).square()*q).sum(1))
        r = torch.stack(residuals,1)
        for relative in [False,True]:
            expected = .5*(gamma*(r/part['energy'][:,None] if relative else r)).sum()/len(y)
            actual = calibrate_noise(part,G,gamma,[3,3],relative)
            self.assertAlmostEqual(actual,float(expected),places=12)

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.coords = torch.rand(37,2)
        self.q = torch.rand(37); self.q /= self.q.sum()
        self.basis = FourierRegimeBasis(2,11,self.q,hidden_dim=8,num_frequencies=4,n_layers=2)
        for _ in range(3): self.basis.add_mode()
        self.y = torch.randn(11,37)
        self.gamma = torch.rand(11); self.gamma[0] = 0
        self.cfg = RefinementConfig(steps=5,lr=1e-3,coefficient_every=2,trajectory_batch=4,
                                   coordinate_batch=7,normalize_modes=False,relative=True)

    def test_weighted_loss_and_gradients(self):
        q = self.q.double()
        energy = (self.y.double().square()*q).sum(1)
        weight = self.gamma.double()/self.gamma.double().sum()/energy
        G,R = _moments(self.basis,self.y,self.coords,q,weight,self.cfg,1.)
        C = torch.stack([m.lambda_ten for m in self.basis.modes],1).detach().double()
        V = functions(self.basis,self.coords).double()
        direct = (((V[:,0]+C@V[:,1:].T-self.y.double()).square()*q).sum(1)*weight).sum()
        reduced = quadratic_loss(V,G,R,q)+weight@energy
        parameters = list(self.basis.mean_net.parameters())+[p for m in self.basis.modes for p in m.phi.parameters()]
        a = torch.autograd.grad(direct,parameters,retain_graph=True)
        b = torch.autograd.grad(reduced,parameters)
        torch.testing.assert_close(direct,reduced,atol=1e-10,rtol=1e-10)
        for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=3e-7,rtol=3e-5)
        svd_basis = deepcopy(self.basis)
        cfg = RefinementConfig(**dict(self.cfg.__dict__,coefficient_solver='svd',rtol=1e-7))
        _moments(svd_basis,self.y,self.coords,q,weight,cfg,1.)
        for old,new in zip(self.basis.modes,svd_basis.modes):
            torch.testing.assert_close(old.lambda_ten,new.lambda_ten,atol=2e-6,rtol=2e-5)

    def test_batches_preserve_objective_and_architecture(self):
        other = deepcopy(self.basis)
        buffers = {k:v.clone() for k,v in self.basis.named_buffers()}
        flags = [p.requires_grad for p in self.basis.parameters()]
        count = sum(p.numel() for p in self.basis.parameters())
        a = refine_basis(self.basis,self.y,self.coords,self.gamma,self.cfg)
        cfg = RefinementConfig(**dict(self.cfg.__dict__,coordinate_batch=37,trajectory_batch=11))
        b = refine_basis(other,self.y,self.coords,self.gamma,cfg)
        self.assertLessEqual(a['final'],a['initial']+1e-10)
        self.assertAlmostEqual(a['final'],b['final'],places=6)
        self.assertEqual(count,sum(p.numel() for p in self.basis.parameters()))
        self.assertEqual(flags,[p.requires_grad for p in self.basis.parameters()])
        for k,v in self.basis.named_buffers(): self.assertTrue(torch.equal(v,buffers[k]),k)

    def test_live_tempo_continuation(self):
        from models.tempo import TEMPOConfig, TEMPOTrainer
        from models.fourier_neural_pod import FourierNeuralPODConfig, FourierNeuralPODTrainer
        from models.neural_refinement import refine_neural_bases
        trainer = TEMPOTrainer(TEMPOConfig(M=1,sigma2=.2,heteroscedastic=False,kappa_prior_weight=0.))
        local = FourierNeuralPODTrainer(self.basis,FourierNeuralPODConfig(max_modes=3))
        local.num_modes = 3
        trainer.trainers = [local]
        trainer.gamma = torch.ones(11,1); trainer.pi = torch.ones(1); trainer._w = self.q
        cfg = RefinementConfig(steps=1,coefficient_every=1,coordinate_batch=7,trajectory_batch=4)
        logs = refine_neural_bases(trainer,self.y,self.coords,cfg,em_steps=1)
        self.assertEqual(len(logs),2)
        self.assertEqual(trainer.cfg.sigma2,.2)
        torch.testing.assert_close(trainer.gamma.sum(1),torch.ones(11))


if __name__ == '__main__':
    unittest.main()
