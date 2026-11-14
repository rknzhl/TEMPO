"""Check Burgers adaptive-update counts against the saved EM deltas."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = json.loads((ROOT / 'paper/results/supplementary/adaptive_update_counts.json').read_text())
    writer = csv.writer(sys.stdout)
    writer.writerow(['task', 'basis', 'M', 'EM_iterations', 'skip_percent',
                     'incremental_percent', 'full_percent'])
    for row in rows:
        if row['benchmark'] != 'Burgers':
            continue
        path = ROOT / row['source']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['source_sha256']
        metrics = json.loads(path.read_text())
        delta = np.asarray(metrics['phase1_log']['delta'])
        assert delta.shape == (row['EM_iterations'], row['M'])
        assert np.isfinite(delta).all()
        small, large = row['thresholds']['theta_s'], row['thresholds']['theta_l']
        counts = [int((delta < small).sum()),
                  int(((delta >= small) & (delta < large)).sum()),
                  int((delta >= large).sum())]
        assert counts == [row['counts'][key] for key in ['skip', 'incremental', 'full']]
        assert delta.size == row['denominator']
        percent = [100 * n / delta.size for n in counts]
        np.testing.assert_allclose(percent, [row['percent'][key]
                                            for key in ['skip', 'incremental', 'full']])
        writer.writerow(['burgers', row['basis'], row['M'], len(delta), *percent])


if __name__ == '__main__':
    main()
