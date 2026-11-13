"""Prepare prediction inputs using saved normalization and coordinates.

Accept prepared field archives or raw NS query arrays. No statistics are refitted."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper.paper_models import load_paper_model, validate_original_burgers_coordinates


def prepare(checkpoint, data, out, split='val', rows=(0,), cno_source=None):
    torch.set_num_threads(2)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    rows = np.asarray(rows, dtype=int)
    if rows.ndim != 1 or not len(rows) or (rows < 0).any():
        raise ValueError('Select a nonempty list of nonnegative row positions.')
    frozen = load_paper_model(checkpoint, 'cpu', cno_source)
    x = None
    with np.load(data, allow_pickle=False) as arrays:
        prefix = split+'_'
        already_normalized = prefix+'kappa' in arrays and frozen.format == 'revision_basis_fit'
        if 'u0_raw' in arrays and 'Re_raw' in arrays:
            raw_x, raw_p = arrays['u0_raw'][rows], arrays['Re_raw'][rows]
            ids = arrays['sample_id'][rows].astype(str)
        elif prefix+'x' in arrays:
            raw_x = arrays[prefix+'x'][rows]
            id_key = prefix+('id' if prefix+'id' in arrays else 'sample_id')
            ids = arrays[id_key][rows].astype(str)
            if already_normalized:
                k = arrays[prefix+'kappa'][rows]
            else:
                pkey = next((prefix+name for name in ('parameter', 'raw_tcool') if prefix+name in arrays), None)
                if pkey is None:
                    raise ValueError('This format requires explicit raw physical parameters.')
                raw_p = arrays[pkey][rows]
                if (prefix+'kappa' in arrays and frozen.format == 'author_operator' and
                        frozen.state['config']['dataset'] == 'helmholtz_four_omega'):
                    x = np.asarray(raw_x*101, dtype='f4')
                    k = arrays[prefix+'kappa'][rows]
                    already_normalized = True
        else:
            selected = (np.flatnonzero(arrays['split'].astype(str) == split)[rows]
                        if 'split' in arrays else rows)
            raw_x, raw_p = arrays['x'][selected], arrays['parameter'][selected]
            ids = arrays['sample_id'][selected].astype(str)
        if already_normalized:
            if x is None:
                x = np.asarray(raw_x, dtype='f4')
            k = np.asarray(k, dtype='f4')[:, :frozen.pdim]
        else:
            raw_p = np.asarray(raw_p).reshape(-1, 1)
            if frozen.format == 'author_operator':
                # Match the author adapter's float64 parameter normalization.
                scale = frozen.state['scale']
                p = np.log(raw_p.astype('f8')) if scale.get('p_transform') == 'log' else raw_p.astype('f8')
                x = (np.asarray(raw_x, dtype='f4')-scale['x_mean'])/scale['x_scale']
                k = ((p-scale['p_mean'])/scale['p_scale']).astype('f4')
            else:
                x, k = frozen.normalize_inputs(torch.tensor(raw_x).float(), torch.tensor(raw_p).float())
                x, k = x.numpy(), k.numpy()
        if frozen.coords is None:
            coords = arrays['coords'].copy()
            validate_original_burgers_coordinates(torch.tensor(coords).float())
        else:
            coords = frozen.coords.cpu().numpy()
            if 'coords' in arrays:
                supplied = arrays['coords']
                if (frozen.state.get('config', {}).get('dataset') == 'sod' and supplied.shape == (6144, 3)):
                    supplied = supplied[:2048, 1:]
                if not np.array_equal(supplied, coords):
                    raise ValueError('Prepared data and checkpoint coordinates differ.')
    if (np.shape(x) != (len(rows), frozen.xdim) or np.shape(k) != (len(rows), frozen.pdim)
            or not all(np.isfinite(v).all() for v in (x, k, coords))):
        raise ValueError('Selected inputs differ from the saved model dimensions.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        np.savez_compressed(stream, x=np.asarray(x, dtype='f4'), kappa=np.asarray(k, dtype='f4'),
                            coords=coords, sample_id=ids, input_units=np.array('normalized'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'data', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--rows', nargs='+', type=int, default=[0])
    parser.add_argument('--cno-source', type=Path)
    args = parser.parse_args()
    prepare(args.checkpoint, args.data, args.out, args.split, args.rows, args.cno_source)
    print(args.out.name)


if __name__ == '__main__':
    main()
