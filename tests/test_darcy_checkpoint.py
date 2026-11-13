"""Darcy deployment preserves weights and checks external operator sources."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from scripts.paper import darcy_baselines


class DarcyCheckpointTest(unittest.TestCase):
    def checkpoint(self, path, source_hashes):
        state = {
            'format': 'darcy_unit_baseline_v1',
            'recipe': {'model': 'deeponet'},
            'coords': torch.tensor([[0., 0.], [1., 1.]]),
            'model_state': {'weight': torch.tensor([[.25, -.5]]),
                            'bias': torch.tensor([.75])},
            'source_sha256': source_hashes,
        }
        torch.save(state, path)
        return state

    def test_internal_provenance_does_not_block_saved_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.pt'
            hashes = {'darcy_baselines.py': '0'*64,
                      'models/pod_deeponet.py': '1'*64}
            state = self.checkpoint(path, hashes)
            with patch.object(darcy_baselines, 'build', side_effect=lambda *args: nn.Linear(2, 1)):
                model, restored = darcy_baselines.load_checkpoint(path, tmp)
            x = torch.tensor([[1., 2.], [-2., .5]])
            expected = torch.nn.functional.linear(x, state['model_state']['weight'],
                                                   state['model_state']['bias'])
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
            self.assertEqual(restored['source_sha256'], hashes)
            self.assertFalse(model.training)

    def test_external_author_source_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.pt'
            author = Path(tmp)/'CNO2d.py'
            author.write_text('author implementation\n')
            self.checkpoint(path, {'CNO2d.py': darcy_baselines.sha256(author)})

            def build(*args):
                model = nn.Linear(2, 1)
                model.author_source = author
                return model

            with patch.object(darcy_baselines, 'build', side_effect=build):
                darcy_baselines.load_checkpoint(path, tmp)
                author.write_text('different implementation\n')
                with self.assertRaisesRegex(ValueError, 'Frozen model source changed: CNO2d.py'):
                    darcy_baselines.load_checkpoint(path, tmp)


    def test_neuraloperator_source_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.pt'
            self.checkpoint(path, {'neuralop/models/fno.py': '0'*64})
            with patch.object(darcy_baselines, 'build', side_effect=lambda *args: nn.Linear(2, 1)):
                with self.assertRaisesRegex(ValueError, 'Frozen model source changed: neuralop/models/fno.py'):
                    darcy_baselines.load_checkpoint(path, tmp)


if __name__ == '__main__':
    unittest.main()
