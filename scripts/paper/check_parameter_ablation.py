"""Check parameter-removal errors against the saved paired arrays."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    summary = json.loads((ROOT / 'paper/results/parameter_ablation/summary.json').read_text())
    writer = csv.writer(sys.stdout)
    writer.writerow(['task', 'model', 'n', 'known_percent', 'mean_parameter_percent', 'increase_pp'])
    for row in summary['models']:
        path = ROOT / row['arrays']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['arrays_sha256']
        checkpoint = ROOT / row['checkpoint']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
        with np.load(path, allow_pickle=False) as data:
            known = data['real_error'].astype('f8')
            changed = data['mean_parameter_error'].astype('f8')
            assert known.shape == changed.shape == (row['n'],)
            assert len(np.unique(data['sample_id'])) == row['n']
        values = [100 * known.mean(), 100 * changed.mean(), 100 * (changed - known).mean()]
        expected = [row['real_error_percent'], row['mean_parameter_error_percent'],
                    row['increase_percentage_points']]
        np.testing.assert_allclose(values, expected, rtol=1e-8, atol=1e-8)
        writer.writerow([row['task'], row['method'], len(known), *values])


if __name__ == '__main__':
    main()
