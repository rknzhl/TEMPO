#!/usr/bin/env python3
"""Adapt the authors' Coanda velocity fields to a fixed geometry-disjoint split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SOURCE_SHA256 = 'c337c3cfcda6a11a32be28169e5c80f6e251b6258bed72c484b1b32801bc5dff'
PRIMARY_URLS = {
    'data': 'https://raw.githubusercontent.com/fpichi/gca-rom/main/dataset/navier_stokes_unstructured.mat',
    'github_metadata': 'https://api.github.com/repos/fpichi/gca-rom/contents/dataset/navier_stokes_unstructured.mat?ref=main',
    'loader': 'https://github.com/fpichi/gca-rom/blob/main/gca_rom/loader.py',
    'pde': 'https://github.com/fpichi/gca-rom/blob/main/gca_rom/pde.py',
    'paper': 'https://doi.org/10.1016/j.jcp.2024.112762',
}
GEOMETRY_SPLIT = {'train': [0, 1, 3, 5, 7, 9, 10], 'val': [2, 8], 'test': [4, 6]}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New NPZ; existing files are refused.')
    a = parser.parse_args()
    if a.out.exists():
        parser.error('Output already exists; choose a new NPZ path.')
    source_hash = file_sha256(a.source)
    if source_hash != SOURCE_SHA256:
        raise ValueError('MAT SHA256 differs from the verified author dataset.')
    data = loadmat(a.source, variable_names=['VX', 'VY', 'xx', 'yy', 'T', 'E'])
    for key in ['VX', 'VY', 'xx', 'yy']:
        if key not in data or data[key].shape != (2719, 341) or not np.isfinite(data[key]).all():
            raise ValueError(f'Expected finite {key}[2719,341].')
    for key, shape in [('T', (5068, 3)), ('E', (7786, 2))]:
        mesh = data.get(key)
        if mesh is None or mesh.shape != shape or not np.isfinite(mesh).all():
            raise ValueError(f'Invalid {key} connectivity shape or values.')
        if not np.equal(mesh, np.rint(mesh)).all() or mesh.min() < 1 or mesh.max() > 2719:
            raise ValueError(f'{key} must contain 1-based integer node IDs in [1,2719].')
        if not (np.diff(np.sort(mesh, axis=1), axis=1) > 0).all():
            raise ValueError(f'{key} contains a repeated node within an element.')
    geometry_hashes = []
    for i in range(11):
        columns = slice(31*i, 31*(i+1))
        for key in ['xx', 'yy']:
            block = data[key][:, columns]
            if not np.array_equal(block, np.broadcast_to(block[:, :1], block.shape)):
                raise ValueError(f'Geometry {i} coordinates vary within its 31-viscosity block.')
        xy = np.stack([data['xx'][:, 31*i], data['yy'][:, 31*i]]).astype('<f8')
        geometry_hashes.append(hashlib.sha256(xy.tobytes()).hexdigest())
    if len(set(geometry_hashes)) != 11:
        raise ValueError('Expected exactly 11 distinct geometry coordinate blocks.')
    widths = (50 + 15*np.arange(11)) / 100.
    viscosities = (200 - 5*np.arange(31)) / 100.
    geometry = np.repeat(np.arange(11), 31)
    x = widths[geometry, None].astype(np.float32)
    parameter = np.tile(viscosities, 11)
    if len(set(zip(x[:, 0].tolist(), parameter.tolist()))) != 341:
        raise ValueError('Duplicate/conflicting geometry-width and viscosity input pairs.')
    group = np.array([f'coanda_width_{widths[i]:.2f}' for i in geometry])
    split = np.full(341, 'excluded', dtype='U16')
    for name, indices in GEOMETRY_SPLIT.items():
        split[np.isin(geometry, indices)] = name
    counts = {name: int((split == name).sum()) for name in GEOMETRY_SPLIT}
    if counts != {'train': 217, 'val': 62, 'test': 62} or (split == 'excluded').any():
        raise ValueError('The fixed geometry split is incomplete or has incorrect counts.')
    y = np.concatenate([data['VX'].T, data['VY'].T], axis=1).astype(np.float32)
    reference_column = 31 * 5
    xy = np.column_stack([data['xx'][:, reference_column], data['yy'][:, reference_column]])
    coords = np.concatenate([np.column_stack([xy, np.zeros(2719)]),
                             np.column_stack([xy, np.ones(2719)])]).astype(np.float32)
    if not all(np.isfinite(v).all() for v in [x, y, coords, parameter]):
        raise ValueError('Conversion produced nonfinite dataset values.')
    metadata = {'source': 'Coanda-author-dataset', 'source_file': a.source.name,
        'source_sha256': source_hash, 'github_blob_sha1': '46a4c8a35ece6ea93666c2462858ff6b7c50e6ef',
        'primary_urls': PRIMARY_URLS, 'column_order': 'j=31*geometry_index+viscosity_index; zero-based',
        'geometry_widths': widths.tolist(), 'viscosities': viscosities.tolist(),
        'input': 'geometry width only; parameter column supplies viscosity nu',
        'target': 'component-major [VX at node IDs 1..2719, VY at node IDs 1..2719]',
        'normalization': 'none; VX and VY retain their shared units; no component-specific scaling',
        'pressure': 'not loaded or included in targets', 'group_key': 'geometry width',
        'split_geometry_indices_zero_based': GEOMETRY_SPLIT, 'split_policy': 'fixed upfront geometry interpolation',
        'counts': counts, 'heldout': [], 'reference_geometry_index': 5,
        'reference_source_column_zero_based': reference_column, 'reference_width': float(widths[5]),
        'coords': '[reference x, reference y, component ID 0=VX or 1=VY]',
        'metric': 'relative nodal L2 on corresponding source node IDs, after reference-geometry '
                  'pullback; unweighted Euclidean velocity-vector norm, not physical-integral L2',
        'physical_coordinates': 'The original MAT retains xx/yy for every geometry; common source '
                                'node IDs and the single T/E connectivity define correspondence. '
                                'Only fixed reference coordinates are model output coordinates.',
        'geometry_coordinate_sha256': geometry_hashes,
        'connectivity': {'triangles': [5068, 3], 'edges': [7786, 2], 'node_index_base': 1}}
    sample_id = np.array([f'width{w:.2f}_nu{nu:.2f}' for w, nu in zip(widths[geometry], parameter)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('xb') as stream:
        np.savez(stream, x=x, y=y, coords=coords, parameter=parameter, group=group, split=split,
                 sample_id=sample_id, source=np.repeat('Coanda-author-dataset', 341),
                 metadata=json.dumps(metadata))
    print(a.out, x.shape, y.shape, counts, flush=True)


if __name__ == '__main__':
    main()
