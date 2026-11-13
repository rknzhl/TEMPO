"""Check saved Fourier frequencies across checkpoint formats."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from models.fourier_neural_pod import FourierNeuralPODConfig, FourierNeuralPODTrainer
from models.regime_basis import FourierRegimeBasis
from models.tempo_online import TEMPOOnlineConfig, build_tempo_online
from scripts.paper.run_comparison import Fitted, load_fit, save_fit


class FourierCheckpointTest(unittest.TestCase):
    def test_new_and_legacy_frequency_storage(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)
        coords, x, k = torch.randn(17, 2), torch.randn(3, 5), torch.randn(3, 1)
        basis = FourierRegimeBasis(2, 7, torch.ones(17)/17)
        for _ in range(2):
            basis.add_mode()
        self.assertEqual(sum(name.endswith('.B') for name in basis.state_dict()), 3)
        local = FourierNeuralPODTrainer(basis, FourierNeuralPODConfig(max_modes=2))
        local.num_modes = 2
        cfg = TEMPOOnlineConfig(hidden_dim=8, n_layers=2)
        model, _ = build_tempo_online([local], 1, 5, cfg)
        fit = Fitted(model, [local], torch.ones(7, 1), {},
            dict(method='tempo_fourier', ranks=[2], online_config=asdict(cfg),
                 basis_config=asdict(local.cfg)))
        with torch.no_grad():
            expected, _ = model(x, k, *fit.evaluate_bases(coords))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.pt'
            save_fit(path, fit, coords, {}, 5, 1)
            state = torch.load(path, map_location='cpu', weights_only=True)
            for legacy in [False, True]:
                if legacy:
                    for item in state['bases']:
                        item['weights'] = {key: value for key, value in item['weights'].items()
                                           if not key.endswith('.B')}
                    torch.save(state, path)
                torch.manual_seed(97)
                restored = load_fit(path, torch.device('cpu'))
                with torch.no_grad():
                    actual, _ = restored.model(x, k, *restored.evaluate_bases(coords))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                from scripts.paper.paper_models import load_paper_model
                public = load_paper_model(path)
                actual, _ = public.predict(x, k)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
