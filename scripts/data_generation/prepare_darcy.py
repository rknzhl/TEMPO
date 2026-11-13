"""Solve the stationary Darcy problem on distinct PDEBench permeability fields."""
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

SOURCE_MD5 = "81694ed31306ff2e5f6b76349b0b4389"
PERMEABILITY_SHA256 = "347f3bcd0630bb55891ae785df14542293cb9f80b370bba7f7c9f0d17c9d27f5"
TOL = 1e-8
NATIVE = 128


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def harmonic(left, right):
    return 2 * left * right / (left + right)


def operator(a):
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    if a.shape != (n, n) or not np.isfinite(a).all() or a.min() <= 0:
        raise ValueError("Permeability must be finite, positive and square.")
    faces_x = harmonic(a[:-1], a[1:])
    faces_y = harmonic(a[:, :-1], a[:, 1:])
    diag = np.zeros_like(a)
    diag[:-1] += faces_x
    diag[1:] += faces_x
    diag[:, :-1] += faces_y
    diag[:, 1:] += faces_y
    # Zero Dirichlet pressure at physical faces, half a cell from the centers.
    diag[0] += 2*a[0]
    diag[-1] += 2*a[-1]
    diag[:, 0] += 2*a[:, 0]
    diag[:, -1] += 2*a[:, -1]
    off_x = -faces_x.ravel()
    off_y = np.zeros_like(a)
    off_y[:, :-1] = -faces_y
    off_y = off_y.ravel()[:-1]
    return sparse.diags((off_x, off_y, diag.ravel(), off_y, off_x),
                        (-n, -1, 0, 1, n), shape=(n*n, n*n), format="csc") * n**2


def solve(a, rhs):
    mat = operator(a)
    f = np.asarray(rhs, dtype=np.float64).ravel()
    u = splu(mat, permc_spec="MMD_AT_PLUS_A").solve(f).reshape(a.shape)
    residual = np.linalg.norm(mat @ u.ravel() - f) / np.linalg.norm(f)
    return u, float(residual)


def flux_balance(a, u):
    # Integral of outward Darcy flux around the boundary; unit-square source integral is 1.
    return 2*(np.sum(a[0]*u[0]) + np.sum(a[-1]*u[-1]) +
              np.sum(a[:, 0]*u[:, 0]) + np.sum(a[:, -1]*u[:, -1]))


def field_hash(a):
    return hashlib.sha256(np.asarray(a, dtype="<f4", order="C").tobytes()).hexdigest()


def select_rows(hdf, sizes, seed):
    perm = hdf["nu"]
    if perm.shape != (10000, NATIVE, NATIVE) or perm.dtype != np.dtype("<f4"):
        raise ValueError(f"Unexpected source shape: {perm.shape}")
    x = (np.arange(NATIVE) + .5) / NATIVE
    for axis in ("x-coordinate", "y-coordinate"):
        if axis in hdf:
            np.testing.assert_allclose(hdf[axis][:], x, atol=1e-8, rtol=0)
    first, duplicates = {}, []
    logical_digest = hashlib.sha256()
    for start in range(0, len(perm), 32):
        block = np.ascontiguousarray(perm[start:start+32], dtype="<f4")
        logical_digest.update(block.tobytes())
        for i, sample in enumerate(block, start):
            key = field_hash(sample)
            if key in first:
                duplicates.append((i, first[key]))
            else:
                first[key] = i
    if logical_digest.hexdigest() != PERMEABILITY_SHA256:
        raise ValueError("Permeability payload does not match the original 10,000 PDEBench fields.")
    total = sum(sizes)
    if len(first) < total:
        raise ValueError(f"Only {len(first)} distinct inputs for {total} requested.")
    keys = list(first)
    chosen = np.random.Generator(np.random.PCG64(seed)).permutation(len(keys))[:total]
    keys = [keys[i] for i in chosen]
    return np.array([first[k] for k in keys], dtype="<i4"), np.array(keys, dtype="S64"), duplicates, len(first)


def store_inputs(hdf, root, sizes, seed):
    selected, hashes, duplicates, unique_count = select_rows(hdf, sizes, seed)
    np.save(root/"source_row.npy", selected)
    np.save(root/"input_hash.npy", hashes)
    np.save(root/"split.npy", np.repeat(np.arange(3, dtype=np.uint8), sizes))
    np.save(root/"coords.npy", (np.arange(NATIVE) + .5) / NATIVE)
    write_json(root/"duplicates.json", {"pairs_duplicate_row_first_row":duplicates})
    shape = (len(selected), NATIVE, NATIVE)
    a = np.lib.format.open_memmap(root/"a.npy", mode="w+", dtype="<f4", shape=shape)
    destination = np.full(len(hdf["nu"]), -1, dtype=int)
    destination[selected] = np.arange(len(selected))
    for start in range(0, len(destination), 32):
        dest = destination[start:start+32]
        keep = dest >= 0
        if np.any(keep):
            block = hdf["nu"][start:start+32][keep]
            if any(field_hash(v).encode() != hashes[j] for v, j in zip(block, dest[keep])):
                raise RuntimeError("Permeability changed between selection and export.")
            a[dest[keep]] = block
    a.flush()
    del a
    p = np.lib.format.open_memmap(root/"unit_pressure.npy", mode="w+", dtype="<f8", shape=shape)
    p.flush()
    del p
    return {"source_rows":len(destination), "unique_inputs":unique_count, "duplicate_rows":len(duplicates),
            "selected_distinct":len(selected), "seed":seed, "rng":"PCG64",
            "splits":dict(zip(("train","validation","test"), sizes))}


def open_maps(root):
    global PERMEABILITY, PRESSURE
    PERMEABILITY = np.load(root/"a.npy", mmap_mode="r")
    PRESSURE = np.load(root/"unit_pressure.npy", mmap_mode="r+")


def solve_row(i):
    a = np.array(PERMEABILITY[i], dtype=np.float64)
    u, residual = solve(a, np.ones(a.shape))
    balance = abs(flux_balance(a, u)-1.)
    if not np.isfinite(u).all() or u.min() <= 0 or not (residual <= TOL and balance <= TOL):
        raise ValueError(f"Invalid solution at {i}: residual={residual}, flux imbalance={balance}")
    PRESSURE[i] = u
    return residual, balance, float(u.min()), float(u.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sourceHDF", type=Path, required=True)
    parser.add_argument("--outNEW", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1,2), default=2)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--sizes", nargs=3, type=int, metavar=("TRAIN","VALIDATION","TEST"), default=(7000,1000,1000))
    args = parser.parse_args()
    if min(args.sizes) < 1:
        parser.error("Each split must have at least one distinct permeability.")
    if args.outNEW.exists():
        parser.error("Output directory already exists; supply a new dataset directory.")
    started = time.monotonic()
    source = args.sourceHDF.resolve()
    stat = source.stat()
    root = args.outNEW
    with h5py.File(source, "r") as hdf:
        kind = "original_hdf" if "tensor" in hdf else "permeability_only_export"
        if kind == "original_hdf" and digest(source, "md5") != SOURCE_MD5:
            raise ValueError("Source file does not match the published beta1 MD5.")
        read_datasets = [k for k in ("nu", "x-coordinate", "y-coordinate") if k in hdf]
        root.mkdir(parents=True)
        selection = store_inputs(hdf, root, args.sizes, args.seed)
    recipe = {"operator":"-div(a grad v)=1 on unit square, zero pressure at physical boundary faces",
              "grid":"128x128 uniform cell centers; axis0=x axis1=y; spacing 1/128",
              "interior_face":"harmonic(a_left,a_right)/h^2","boundary_face":"2*a_cell/h^2",
              "matrix":"symmetric five-point conservative finite volume","solver":"scipy.sparse.linalg.splu/MMD_AT_PLUS_A float64",
              "residual_tolerance":TOL,"u_beta":"beta * unit_pressure",
              "training_beta":[.1,1.,10.,100.],"heldout_beta":[.01,.3],
              "split":"First occurrence of each SHA256 of C-order little-endian float32 permeability; permute unique IDs and take disjoint consecutive splits.",
              "selection":selection}
    write_json(root/"recipe.json",recipe)
    total = sum(args.sizes)
    diagnostics = np.empty((total, 4), dtype=np.float64)
    pool = None
    try:
        if args.workers == 1:
            open_maps(root)
            results = map(solve_row, range(total))
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers, initializer=open_maps, initargs=(root,))
            results = pool.map(solve_row, range(total), chunksize=4)
        for i, row in enumerate(results, 1):
            diagnostics[i-1] = row
            if i == 1 or i % 200 == 0 or i == total:
                print(f"{i}/{total} unit-pressure solves complete, elapsed {time.monotonic()-started:.1f} s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        elif args.workers == 1:
            PRESSURE.flush()
    np.savez(root/"diagnostics.npz", relative_residual=diagnostics[:,0], absolute_flux_imbalance=diagnostics[:,1],
             pressure_min=diagnostics[:,2], pressure_max=diagnostics[:,3])
    if (source.stat().st_size, source.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError("Source file changed during generation.")
    files = {p.name:{"sha256":digest(p),"bytes":p.stat().st_size} for p in sorted(root.iterdir()) if p.is_file()}
    import scipy
    manifest = {"schema_version":1,"selection":selection,"files":files,"source_name":source.name,"source_kind":kind,"original_hdf_md5":SOURCE_MD5,
                "all_source_permeabilities_sha256":PERMEABILITY_SHA256,"source_datasets_read":read_datasets,"solver_sha256":digest(Path(__file__)),
                "recipe_sha256":files["recipe.json"]["sha256"],"relative_residual_max":float(diagnostics[:,0].max()),
                "absolute_flux_imbalance_max":float(diagnostics[:,1].max()),"pressure_range":[float(diagnostics[:,2].min()),float(diagnostics[:,3].max())],
                "elapsed_seconds":time.monotonic()-started,"workers":args.workers,"numpy":np.__version__,"scipy":scipy.__version__}
    write_json(root/"manifest.json", manifest)
    write_json(root/"COMPLETE.json", {"rows":total,"manifest_sha256":digest(root/"manifest.json")})


if __name__ == "__main__":
    main()
