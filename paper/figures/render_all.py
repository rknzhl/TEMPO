"""Build the 14 article figures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE.parents[1] / 'output' / 'figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for item in manifest['numeric_inputs']:
        if sha(HERE / item['path']) != item['sha256']:
            raise ValueError('Changed frozen input: ' + item['path'])
    environment = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[key] = '1'
    environment['CUDA_VISIBLE_DEVICES'] = ''
    environment.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tempo-paper-figures'))
    data = HERE / 'data'
    commands = [
        ['plot_burgers_basis.py', '--data', data / 'burgers_joint_bases.npz', '--style-dir', HERE],
        ['plot_combined_regimes.py', '--burgers', data / 'regimes_burgers.npz',
         '--other-tasks', data / 'regimes_sod_coanda.npz',
         '--helmholtz', data / 'regimes_helmholtz_four_omega.npz',
         '--darcy', data / 'regimes_darcy.npz', '--style-dir', HERE],
        ['render_neuralpod_fields.py', '--data', data / 'neuralpod_fields'],
        ['plot_joint_errors.py', '--data', data],
        ['render_darcy_fields.py', '--data', data],
        ['plot_coanda_training.py', '--data', data / 'regimes_sod_coanda.npz'],
    ]
    for script, *arguments in commands:
        subprocess.run([sys.executable, str(HERE / script), *map(str, arguments),
                        '--out', str(args.out)], env=environment, check=True)
    for asset in manifest['assets']:
        if asset['reproduction'] == 'preserved_static_asset':
            source = HERE / asset['archive']
            if sha(source) != asset['sha256']:
                raise ValueError('Changed preserved article asset: ' + source.name)
            target = args.out / asset['file']
            if target.resolve() != source.resolve():
                shutil.copy2(source, target)
    outputs = []
    for asset in manifest['assets']:
        path = args.out / asset['file']
        outputs.append(dict(file=path.name, bytes=path.stat().st_size,
                            sha256=sha(path), reproduction=asset['reproduction']))
    report = dict(status='complete', numerical_figures=11, preserved_assets=3,
                  inputs_verified=len(manifest['numeric_inputs']), outputs=outputs,
                  model_training=False, embedding_refit=False, smoothing=False,
                  scientific_values_modified=False, cpu_threads=1,
                  limitations=manifest['limitations'])
    (args.out / 'render_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Rendered 11 numerical figures; copied 3 preserved article assets.')


if __name__ == '__main__':
    main()
