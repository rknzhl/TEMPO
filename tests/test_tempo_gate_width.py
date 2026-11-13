"""Check gate width defaults and checkpoint replay."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import torch

from models.pod import PODConfig,PODTrainer
from models.tempo_online import TEMPOOnlineConfig,build_tempo_online
from scripts.paper.run_comparison import Fitted,save_fit,load_fit


class GateWidthTest(unittest.TestCase):
    def test_checkpoint_roundtrip_and_legacy_default(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        bases=[]
        for _ in range(2):
            b=PODTrainer(PODConfig(max_modes=2,fixed_rank=True))
            b.basis.initialize(torch.randn(11),torch.randn(11,2),torch.randn(5,2));b.num_modes=2;bases.append(b)
        x,k=torch.randn(3,6),torch.randn(3,1);coords=torch.randn(11,2)
        for gate_width in [None,4]:
            cfg=TEMPOOnlineConfig(hidden_dim=16,gate_hidden_dim=gate_width,n_layers=2)
            model,_=build_tempo_online(bases,1,6,cfg)
            self.assertEqual(model.gating.branch.net[0].out_features,16 if gate_width is None else 4)
            log=dict(method='tempo_pod',ranks=[2,2],online_config=asdict(cfg),basis_config=asdict(bases[0].cfg))
            if gate_width is None:log['online_config'].pop('gate_hidden_dim')
            fit=Fitted(model,bases,torch.full((5,2),.5),{},log)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'weights.pt';save_fit(path,fit,coords,{},6,1);restored=load_fit(path,torch.device('cpu'))
                with torch.no_grad():
                    expected,p=model(x,k,*fit.evaluate_bases(coords));actual,q=restored.model(x,k,*restored.evaluate_bases(coords))
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                torch.testing.assert_close(p,q,atol=0,rtol=0)
                torch.testing.assert_close(p.sum(1),torch.ones(3))

    def test_invalid_width(self):
        with self.assertRaises(ValueError):TEMPOOnlineConfig(gate_hidden_dim=0)


if __name__=='__main__':unittest.main()
