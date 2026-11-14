"""Check known versus estimated parameters using saved errors and classifier logs."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    summary = json.loads((ROOT / 'paper/results/parameter_estimation/summary.json').read_text())
    writer = csv.writer(sys.stdout)
    writer.writerow(['task', 'model', 'known_relative_l2', 'estimated_relative_l2'])
    row = summary['burgers']
    source = json.loads((ROOT / row['source']).read_text())['statistics']['test']
    known = np.mean(list(source['field_errors']['known']['seed_mean_relative_l2'].values()))
    estimated = np.mean(list(source['field_errors']['hard']['seed_mean_relative_l2'].values()))
    np.testing.assert_allclose([known, estimated],
                              [row['known_mean_relative_l2'], row['estimated_mean_relative_l2']])
    classifiers = json.loads((ROOT / 'paper/configs/paper_supplementary_models.json').read_text())
    accuracy = []
    for seed in row['seeds']:
        model = next(m for m in classifiers['models'] if m['id'] == f'viscosity_classifier_seed{seed}')
        metrics = json.loads((ROOT / model['metrics']).read_text())['classifier_test']
        confusion = np.asarray(metrics['confusion_matrix'])
        assert confusion.sum() == row['n']
        accuracy.append(100 * confusion.trace() / confusion.sum())
    np.testing.assert_allclose(np.mean(accuracy), row['accuracy_percent'])
    writer.writerow(['burgers', 'TEMPO(POD), M=4', known, estimated])
    for row in summary['navier_stokes']:
        path = ROOT / row['per_sample']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        checkpoint = ROOT / row['checkpoint']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
        with np.load(path, allow_pickle=False) as data:
            known = data['known_error'].astype('f8')
            estimated = data['estimated_error'].astype('f8')
            assert known.shape == estimated.shape == (row['n'],)
            assert len(np.unique(data['sample_id'])) == row['n']
        values = [known.mean(), estimated.mean()]
        np.testing.assert_allclose(values, [row['known_mean_relative_l2'],
                                            row['estimated_mean_relative_l2']], rtol=1e-8)
        writer.writerow(['navier_stokes', row['method'], *values])


if __name__ == '__main__':
    main()
