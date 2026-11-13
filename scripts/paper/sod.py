"""Solve the Euler shock tube and prepare density or full-state trajectories.

The exact Riemann solver follows
https://www.clawpack.org/riemann_book/html/Euler.html.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.optimize import brentq


def group_id(initial):
    field = np.ascontiguousarray(initial, dtype="<f4")
    return hashlib.sha256(field.tobytes()).hexdigest()


def split_groups(group, parameter, heldout, seed):
    """Assign by realization alone. Held-out parameters are visible only on test groups."""
    keys = np.unique(group)
    if len(keys) < 10:
        raise ValueError("At least ten independent groups are required.")
    keys = np.random.default_rng(seed).permutation(keys)
    n_test = max(2, len(keys) // 5)
    n_val = max(2, len(keys) // 5)
    partitions = dict.fromkeys(keys, "train")
    partitions.update(dict.fromkeys(keys[:n_test], "test"))
    partitions.update(dict.fromkeys(keys[n_test : n_test + n_val], "val"))
    seen = np.array([partitions[k] for k in group])
    outside = np.isin(parameter, heldout)
    split = np.where(outside, np.where(seen == "test", "heldout_test", "excluded"), seen)
    for label in ["train", "val", "test"]:
        if set(np.unique(parameter[split == label])) != set(np.unique(parameter[~outside])):
            raise ValueError(f"Missing training-range parameter in {label}; increase group count.")
    if len(heldout) and set(np.unique(parameter[split == "heldout_test"])) != set(heldout):
        raise ValueError("Held-out parameters lack independent test groups; increase group count.")
    return split


def write_data(output, x, y, coords, parameter, group, heldout, split_seed, source, settings):
    x, y, coords = [np.asarray(a, dtype=np.float32) for a in (x, y, coords)]
    parameter, group = np.asarray(parameter, dtype=np.float64), np.asarray(group, dtype=str)
    if not (len(x) == len(y) == len(parameter) == len(group)) or y.shape[1] != len(coords):
        raise ValueError("Inputs, targets and output coordinates are not aligned.")
    if not all(np.isfinite(a).all() for a in (x, y, coords, parameter)):
        raise ValueError("Nonfinite data.")
    if len(set(zip(group.tolist(), parameter.tolist()))) != len(x):
        raise ValueError(
            "Duplicate (independent group, parameter); remove duplicate source records."
        )
    split = split_groups(group, parameter, heldout, split_seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        np.savez(
            f,
            x=x,
            y=y,
            coords=coords,
            parameter=parameter,
            group=group,
            split=split,
            source=np.repeat(source, len(x)),
            metadata=json.dumps(
                dict(source=source, split_seed=split_seed, heldout=list(heldout), **settings)
            ),
        )
    print(output, x.shape, y.shape, dict(zip(*np.unique(split, return_counts=True))))


GAMMA = 1.4
PRESSURE_RATIOS = (0.05, 0.1, 0.15, 0.2, 0.4)
HELDOUT_RATIO = 0.15


def sound_speed(state):
    density, _, pressure = state
    return np.sqrt(GAMMA * pressure / density)


def pressure_wave(pressure, state):
    density, _, p0 = state
    if pressure > p0:  # Rankine-Hugoniot shock relation
        beta = (GAMMA - 1) / (GAMMA + 1)
        return (pressure - p0) * np.sqrt(2 / ((GAMMA + 1) * density * (pressure + beta * p0)))
    exponent = (GAMMA - 1) / (2 * GAMMA)
    return 2 * sound_speed(state) / (GAMMA - 1) * ((pressure / p0) ** exponent - 1)


def star_state(left, right):
    """Return contact pressure and velocity for left/right (density, velocity, pressure)."""
    states = np.asarray([left, right], dtype=float)
    if states.shape != (2, 3) or not np.isfinite(states).all() or np.any(states[:, [0, 2]] <= 0):
        raise ValueError(
            "Riemann states require finite velocities and positive density and pressure."
        )
    p_l, p_r = states[:, 2]
    u_l, u_r = states[:, 1]
    a_l, a_r = sound_speed(left), sound_speed(right)
    numerator = a_l + a_r - (GAMMA - 1) * (u_r - u_l) / 2
    if numerator <= 0:
        raise ValueError(
            "The separating velocities produce vacuum, outside this positive-state benchmark."
        )

    # Solve two isentropic rarefactions analytically, including the near-vacuum case.
    z = (GAMMA - 1) / (2 * GAMMA)
    p_two_fans = (numerator / (a_l / p_l**z + a_r / p_r**z)) ** (1 / z)
    if p_two_fans <= min(p_l, p_r):
        pressure = p_two_fans
    else:

        def residual(p):
            return pressure_wave(p, left) + pressure_wave(p, right) + u_r - u_l

        upper = max(p_l, p_r)
        while residual(upper) < 0:
            upper *= 2
        pressure = brentq(residual, 0.0, upper, xtol=np.finfo(float).tiny)
    velocity = (u_l + u_r + pressure_wave(pressure, right) - pressure_wave(pressure, left)) / 2
    return pressure, velocity


def outer_speed(state, pressure, direction):
    """Speed of the outward shock or leading edge of the rarefaction fan."""
    _, velocity, p0 = state
    multiplier = (
        np.sqrt(1 + (GAMMA + 1) / (2 * GAMMA) * (pressure / p0 - 1)) if pressure > p0 else 1.0
    )
    return velocity + direction * sound_speed(state) * multiplier


def _sample_side(xi, state, pressure, velocity, direction):
    """Evaluate one half of the solution in an outward-pointing similarity coordinate."""
    rho0, u0, p0 = state
    speed = direction * u0
    xi_out = direction * xi
    ratio = pressure / p0
    a0 = sound_speed(state)
    if pressure > p0:
        beta = (GAMMA - 1) / (GAMMA + 1)
        rho_star = rho0 * (ratio + beta) / (beta * ratio + 1)
    else:
        rho_star = rho0 * ratio ** (1 / GAMMA)
    values = np.tile([rho_star, velocity, pressure], (len(xi), 1))
    head = direction * outer_speed(state, pressure, direction)
    values[xi_out >= head] = state
    if pressure < p0:
        a_star = a0 * ratio ** ((GAMMA - 1) / (2 * GAMMA))
        tail = direction * velocity + a_star
        fan = (xi_out > tail) & (xi_out < head)
        a = (2 * a0 + (GAMMA - 1) * (xi_out[fan] - speed)) / (GAMMA + 1)
        u = (2 * (xi_out[fan] - a0) + (GAMMA - 1) * speed) / (GAMMA + 1)
        values[fan] = np.column_stack(
            (
                rho0 * (a / a0) ** (2 / (GAMMA - 1)),
                direction * u,
                p0 * (a / a0) ** (2 * GAMMA / (GAMMA - 1)),
            )
        )
    return values


def sample(left, right, x, time, x0=0.5):
    """Sample (density, velocity, pressure) at broadcastable arrays x and time."""
    pressure, velocity = star_state(left, right)
    x, time = np.broadcast_arrays(np.asarray(x, float), np.asarray(time, float))
    if (
        not np.isfinite(x0)
        or not np.isfinite(x).all()
        or not np.isfinite(time).all()
        or (time < 0).any()
    ):
        raise ValueError("Positions must be finite and sample times nonnegative and finite.")
    values = np.where((x < x0)[..., None], left, right).astype(float)
    positive = time > 0
    xi = (x[positive] - x0) / time[positive]
    solution = np.empty((len(xi), 3))
    on_left = xi < velocity
    solution[on_left] = _sample_side(xi[on_left], left, pressure, velocity, -1)
    solution[~on_left] = _sample_side(xi[~on_left], right, pressure, velocity, 1)
    values[positive] = solution
    return values


def generate(output, groups=128, nx=64, nt=8, final_time=0.12, data_seed=2026, split_seed=143):
    """Sample paired pressure ratios and retain complete independent groups for testing."""
    if groups < 10 or nx < 2 or nt < 1 or not np.isfinite(final_time) or final_time <= 0:
        raise ValueError(
            "Use at least ten groups, two spatial points, one time and a positive final time."
        )
    rng = np.random.default_rng(data_seed)
    # x features: rho_L, u_L, p_L, rho_R, u_R, diaphragm position. p_R = parameter * p_L.
    low = [0.85, -0.08, 0.85, 0.11, -0.08, 0.45]
    high = [1.15, 0.08, 1.15, 0.17, 0.08, 0.55]
    realizations = rng.uniform(low, high, (groups, 6)).astype(np.float32)
    nodes = (np.arange(nx) + 0.5) / nx
    times = final_time * np.arange(1, nt + 1) / nt
    x_grid, t_grid = np.broadcast_arrays(nodes[None, :], times[:, None])
    coords = np.stack((x_grid.ravel(), t_grid.ravel()), axis=1)
    inputs = np.repeat(realizations, len(PRESSURE_RATIOS), axis=0)
    group_hashes = np.repeat([group_id(row) for row in realizations], len(PRESSURE_RATIOS))
    ratios = np.tile(PRESSURE_RATIOS, groups)
    targets = np.empty((len(inputs), nx * nt), dtype=np.float32)
    min_clearance = np.inf
    for i, (row, ratio) in enumerate(zip(inputs, ratios)):
        rho_l, u_l, p_l, rho_r, u_r, x0 = map(float, row)
        left, right = (rho_l, u_l, p_l), (rho_r, u_r, ratio * p_l)
        p_star, _ = star_state(left, right)
        left_edge = x0 + final_time * outer_speed(left, p_star, -1)
        right_edge = x0 + final_time * outer_speed(right, p_star, 1)
        clearance = min(x0, 1 - x0, left_edge, 1 - right_edge)
        if clearance <= 0:
            raise ValueError(
                f"A wave reaches a boundary for realization {i // len(PRESSURE_RATIOS)} "
                f"at pressure ratio {ratio}; reduce --final-time."
            )
        min_clearance = min(min_clearance, clearance)
        targets[i] = sample(left, right, x_grid, t_grid, x0)[..., 0].ravel()
    settings = dict(
        gamma=GAMMA,
        shape_time_space=[nt, nx],
        coords_columns=["x", "t"],
        output="density",
        units="nondimensional throughout",
        metric="per-trajectory relative L2 density",
        domain=[0, 1],
        final_time=final_time,
        minimum_boundary_clearance=min_clearance,
        target_grid="Uniform cell-center point samples at nt positive times; x varies fastest.",
        input_columns=["rho_L", "u_L", "p_L", "rho_R", "u_R", "x_diaphragm"],
        parameter_definition="p_R / p_L; append parameter to the six input features",
        ratios=list(PRESSURE_RATIOS),
        sampling="Independent uniform components per group.",
        input_lower=low,
        input_upper=high,
        data_seed=data_seed,
        group_key="SHA256 of the six base input features as little-endian float32",
    )
    write_data(
        output,
        inputs,
        targets,
        coords,
        ratios,
        group_hashes,
        [HELDOUT_RATIO],
        split_seed,
        "Euler-Sod-exact",
        settings,
    )


SOURCE_SHA256 = "b2e6be36139bd9fbc1bcc85031b9028f40669758d57d4306b45ceec41b2d7c38"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def density_rows(path, rows, n, points):
    """Read selected density rows without loading the full target array."""
    with zipfile.ZipFile(path) as archive, archive.open("y.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version not in ((1, 0), (2, 0)):
            raise ValueError("Unsupported source density NPY version.")
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran, dtype = reader(stream)
        if shape != (n, points) or fortran or dtype.kind != "f":
            raise ValueError("Expected source density y[N,nt*nx] in C order.")
        start, row_bytes = stream.tell(), points * dtype.itemsize
        for row in rows:
            stream.seek(start + int(row) * row_bytes)
            raw = stream.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError("Truncated selected source density row.")
            yield int(row), np.frombuffer(raw, dtype=dtype)


def prepare(source, out, expected_sha256=SOURCE_SHA256):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(out)
    source_hash = sha256(source)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError("Source SHA256 differs from the approved Sod512 dataset.")
    with np.load(source, allow_pickle=False) as archive:
        x, parameter, group, split, field_coords = [
            archive[key].copy() for key in ("x", "parameter", "group", "split", "coords")
        ]
        original = json.loads(str(archive["metadata"].item()))
        ids = (
            archive["sample_id"].astype(str)
            if "sample_id" in archive
            else np.arange(len(x)).astype(str)
        )
    n = len(x)
    if (
        x.shape != (n, 6)
        or any(value.shape != (n,) for value in (parameter, group, split, ids))
        or len(np.unique(ids)) != n
        or len(set(zip(group.tolist(), parameter.tolist()))) != n
    ):
        raise ValueError("Invalid IC/metadata shapes or duplicate sample/group-parameter IDs.")
    nt, nx = map(int, original["shape_time_space"])
    final_time = float(original["final_time"])
    if original["gamma"] != GAMMA or nt < 1 or nx < 2 or final_time <= 0:
        raise ValueError("Unsupported analytic Sod metadata.")
    # Reconstruct the original float64 sampling grid before its float32 storage.
    xx, tt = np.broadcast_arrays(
        ((np.arange(nx) + 0.5) / nx)[None, :], (final_time * np.arange(1, nt + 1) / nt)[:, None]
    )
    exact_coords = np.column_stack((xx.ravel(), tt.ravel()))
    if not np.array_equal(exact_coords.astype(np.float32), field_coords):
        raise ValueError("Source coordinates differ from the metadata-defined cell-center grid.")
    rows = np.flatnonzero(np.isin(split, ("train", "val")))
    if not len(rows) or not np.isfinite(x[rows]).all() or not np.isfinite(parameter[rows]).all():
        raise ValueError("Missing development rows or nonfinite development inputs.")
    groups = {label: set(group[split == label]) for label in ("train", "val")}
    evaluation = set(group[np.isin(split, ("test", "heldout_test"))])
    if groups["train"] & groups["val"] or (groups["train"] | groups["val"]) & evaluation:
        raise ValueError("An initial-condition group crosses development/evaluation partitions.")
    seen = set(parameter[split == "train"])
    if not seen or set(parameter[split == "val"]) != seen or seen & set(original["heldout"]):
        raise ValueError("Train/validation parameter coverage or held-out exclusion differs.")
    physical = np.empty((len(rows), 3, nx * nt), dtype=np.float32)
    clearance, maximum_parity_error = float("inf"), 0.0
    for position, (row, density) in enumerate(density_rows(source, rows, n, nx * nt)):
        rho_l, u_l, p_l, rho_r, u_r, x0 = map(float, x[row])
        left, right = (rho_l, u_l, p_l), (rho_r, u_r, float(parameter[row]) * p_l)
        pressure, _ = star_state(left, right)
        margin = min(
            x0,
            1 - x0,
            x0 + final_time * outer_speed(left, pressure, -1),
            1 - x0 - final_time * outer_speed(right, pressure, 1),
        )
        values = sample(left, right, exact_coords[:, 0], exact_coords[:, 1], x0)
        if margin <= 0 or not np.isfinite(values).all() or not (values[:, [0, 2]] > 0).all():
            raise ValueError(
                f"Invalid boundary clearance, positivity or finite state at source row {row}."
            )
        physical[position] = values.T
        error = float(np.max(np.abs(physical[position, 0].astype(np.float64) - density)))
        if not np.allclose(physical[position, 0], density, rtol=1e-6, atol=5e-7):
            raise ValueError(f"Density parity failed at development source row {row}: {error}.")
        maximum_parity_error, clearance = max(maximum_parity_error, error), min(clearance, margin)
    scales = np.array([1.0, np.sqrt(GAMMA), 1.0], dtype=np.float64)
    scaled = (physical / scales[None, :, None]).astype(np.float32).reshape(len(rows), -1)
    output = {}
    for label in ("train", "val"):
        selected = split[rows] == label
        output.update(
            {
                f"{label}_x": x[rows[selected]],
                f"{label}_parameter": parameter[rows[selected]],
                f"{label}_group": group[rows[selected]],
                f"{label}_sample_id": ids[rows[selected]],
                f"{label}_row": rows[selected],
                f"{label}_split": split[rows[selected]],
                f"{label}_y": scaled[selected],
                f"{label}_physical": physical[selected],
            }
        )
    code = {str(Path(__file__).resolve()): sha256(__file__)}
    metadata = {
        "source_metadata": original,
        "source_sha256": source_hash,
        "source_code_sha256": code,
        "data_seed": original["data_seed"],
        "split_seed": original["split_seed"],
        "shape_time_space": [nt, nx],
        "channel_order": ["rho", "u", "p"],
        "flatten_order": "component,time,space; space varies fastest",
        "scaling": "q=(rho,u/sqrt(gamma),p); fixed reference scales; equal channel/grid weights",
        "source_density_access": "Only train/val y.npy rows decoded for parity; no test targets exported or scored.",
        "density_parity": {
            "passed": True,
            "rows": len(rows),
            "max_abs_error": maximum_parity_error,
            "rtol": 1e-6,
            "atol": 5e-7,
        },
        "minimum_boundary_clearance": clearance,
        "finite_positive_rho_p": True,
        "sample_id_policy": "Preserved source IDs; source row-number strings when IDs are absent.",
    }
    if sha256(source) != source_hash or any(sha256(path) != value for path, value in code.items()):
        raise RuntimeError("Source data or analytic adapter code changed during preparation.")
    coords = np.column_stack(
        (np.repeat(np.arange(3), nx * nt), np.tile(field_coords, (3, 1)))
    ).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez_compressed(
            stream,
            **output,
            coords=coords,
            field_coords=field_coords,
            channel_scales=scales,
            channels=np.array(["rho", "u", "p"]),
            metadata=json.dumps(metadata),
        )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--source", type=Path, help="Export full states from a saved density dataset."
    )
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--nt", type=int, default=8)
    parser.add_argument("--final-time", type=float, default=0.12)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=143)
    args = parser.parse_args()
    if args.source is None:
        generate(
            args.out,
            args.groups,
            args.nx,
            args.nt,
            args.final_time,
            args.data_seed,
            args.split_seed,
        )
    else:
        result = prepare(args.source, args.out)
        print(
            f"Prepared full states; density parity error {result['density_parity']['max_abs_error']:.3g}"
        )


if __name__ == "__main__":
    main()
