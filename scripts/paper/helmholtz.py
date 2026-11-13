"""Prepare the four-frequency Helmholtz benchmark on the native FEM grid.

Source inputs_train.bin, inputs_val.bin and inputs_test.bin are the author's
little-endian float64 fields in Julia/Fortran order. Normalization uses train.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, eigsh


AUTHOR_COMMIT = "982f5ebed5aade6b70230c7e5c9193312be371ed"
OMEGAS = (600.0, 800.0, 1000.0, 1200.0)


def source_pairs(count=100):
    limit = int(np.sqrt(2 * count)) + 1
    pairs = [(i, j) for i in range(limit + 1) for j in range(limit + 1) if i + j]
    return np.array(sorted(pairs, key=lambda p: p[0] ** 2 + p[1] ** 2)[:count], dtype=int)


PAIRS = source_pairs()


def recover_field(c):
    c = np.asarray(c, dtype=float)
    assert c.shape == (101, 101) and np.all(np.abs(c - 20) < 1)
    a = np.arctanh(c - 20)
    all_coeff = dctn(a, type=1) / 10000.0
    all_coeff[[0, -1], :] *= 0.5
    all_coeff[:, [0, -1]] *= 0.5
    coeff = all_coeff[PAIRS[:, 0], PAIRS[:, 1]]
    represented = np.zeros_like(all_coeff)
    represented[PAIRS[:, 0], PAIRS[:, 1]] = coeff
    x = np.linspace(0, 1, 101)
    reconstructed = evaluate_a(coeff, x, x)
    audit = dict(
        max_abs_a_error=float(np.max(np.abs(reconstructed - a))),
        max_abs_c_error=float(np.max(np.abs(20 + np.tanh(reconstructed) - c))),
        discarded_coeff_l2=float(np.linalg.norm(all_coeff - represented)),
        constant_coefficient=float(all_coeff[0, 0]),
    )
    return coeff, audit


def evaluate_a(coeff, x, y):
    # Preserve the author's KL-mode ordering, including ties.
    cx = np.cos(np.pi * np.asarray(x)[:, None] * PAIRS[None, :, 0])
    cy = np.cos(np.pi * np.asarray(y)[:, None] * PAIRS[None, :, 1])
    return (cx * coeff) @ cy.T


class AuthorMesh:
    def __init__(self, ne):
        self.ne = ne
        self.n = ne + 1
        self.h = 1.0 / ne
        x, y = np.meshgrid(np.arange(ne), np.arange(ne), indexing="ij")
        a = x * self.n + y
        self.nodes = np.stack((a, a + self.n, a + self.n + 1, a + 1), axis=-1).reshape(-1, 4)
        self.rows = np.repeat(self.nodes, 4, axis=1).ravel()
        self.cols = np.tile(self.nodes, (1, 4)).ravel()
        self.gauss, self.weights = np.polynomial.legendre.leggauss(3)
        self.quad = []
        stiff = np.zeros((4, 4))
        for ga, wa in zip(self.gauss, self.weights):
            for gb, wb in zip(self.gauss, self.weights):
                shape = (
                    np.array(
                        [
                            (1 - ga) * (1 - gb),
                            (1 + ga) * (1 - gb),
                            (1 + ga) * (1 + gb),
                            (1 - ga) * (1 + gb),
                        ]
                    )
                    / 4
                )
                grad = np.array(
                    [
                        [-(1 - gb), -(1 - ga)],
                        [(1 - gb), -(1 + ga)],
                        [(1 + gb), (1 + ga)],
                        [-(1 + gb), (1 - ga)],
                    ]
                ) / (2 * self.h)
                weight = self.h**2 * wa * wb / 4
                stiff += (grad @ grad.T) * weight
                self.quad.append((ga, gb, weight, np.outer(shape, shape)))
        self.stiff = self.assemble(np.broadcast_to(stiff, (ne * ne, 4, 4)))
        # The top-edge Neumann load excludes the endpoints x = 0.35 and x = 0.65.
        top = np.zeros(self.n)
        top[int(round(0.35 * ne)) + 1 : int(round(0.65 * ne))] = 1.0
        load = np.zeros((self.n, self.n))
        load[:-1, -1] += self.h * (2 * top[:-1] + top[1:]) / 6
        load[1:, -1] += self.h * (top[:-1] + 2 * top[1:]) / 6
        self.load = load.ravel()

    def assemble(self, local):
        return coo_matrix(
            (np.asarray(local).ravel(), (self.rows, self.cols)), shape=(self.n**2, self.n**2)
        ).tocsc()

    def mass(self, coeff):
        local = np.zeros((self.ne * self.ne, 4, 4))
        base = np.arange(self.ne) * self.h
        for ga, gb, weight, outer in self.quad:
            c = 20 + np.tanh(
                evaluate_a(coeff, base + self.h * (1 + ga) / 2, base + self.h * (1 + gb) / 2)
            )
            local += weight * (1 / c**2).ravel()[:, None, None] * outer
        return self.assemble(local)

    def solve(self, mass, omega, eigen=False):
        matrix = self.stiff - omega**2 * mass
        lu = splu(matrix, permc_spec="COLAMD")
        u = lu.solve(self.load)
        relative_residual = float(
            np.linalg.norm(matrix @ u - self.load) / np.linalg.norm(self.load)
        )
        info = dict(
            relative_weak_residual=relative_residual,
            nodal_rms=float(np.sqrt(np.mean(u * u))),
            max_abs=float(np.max(np.abs(u))),
        )
        if eigen:
            nearest = float(
                eigsh(
                    self.stiff,
                    k=1,
                    M=mass,
                    sigma=omega**2,
                    which="LM",
                    return_eigenvectors=False,
                    tol=1e-9,
                )[0]
            )
            info.update(
                nearest_generalized_eigenvalue=nearest,
                relative_spectral_gap=abs(nearest - omega**2) / omega**2,
            )
        return u.reshape(self.n, self.n), info


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--training-scales", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory.")
    if args.test_only and args.training_scales is None:
        parser.error("--test-only requires saved --training-scales.")
    bounds = {"test": (8000, 8200)} if args.test_only else {"train": (0, 1000), "val": (8200, 8400)}
    mesh = AuthorMesh(100)
    inputs, targets, source_hashes, diagnostics = {}, {}, {}, {}
    args.out.mkdir(parents=True)
    for split, (lo, hi) in bounds.items():
        path = args.source / ("inputs_" + split + ".bin")
        flat = np.memmap(path, dtype="<f8", mode="r")
        count = len(flat) // 10201
        if len(flat) != count * 10201 or count < hi - lo:
            raise ValueError("Invalid released input file shape: " + path.name)
        fields = flat.reshape(101, 101, count, order="F").transpose(2, 0, 1)[: hi - lo]
        inputs[split] = fields
        source_hashes[path.name] = sha(path)
        outputs = np.lib.format.open_memmap(
            args.out / ("solutions_" + split + ".npy"),
            mode="w+",
            dtype="<f8",
            shape=(hi - lo, 4, 101, 101),
        )
        max_recovery, max_residual = 0.0, 0.0
        for index, field in enumerate(fields):
            coefficients, audit = recover_field(field)
            max_recovery = max(max_recovery, audit["max_abs_c_error"])
            if audit["max_abs_c_error"] >= 1e-10 or audit["discarded_coeff_l2"] >= 1e-10:
                raise ValueError("Released wavespeed is outside the author KL field family.")
            mass = mesh.mass(coefficients)
            for j, omega in enumerate(OMEGAS):
                solution, audit = mesh.solve(mass, omega)
                max_residual = max(max_residual, audit["relative_weak_residual"])
                if not np.isfinite(solution).all() or audit["relative_weak_residual"] >= 1e-7:
                    raise ValueError("Nonfinite solution or FEM residual failed.")
                outputs[index, j] = solution
            if (index + 1) % 25 == 0:
                outputs.flush()
                print(split, index + 1, "/", hi - lo, flush=True)
        outputs.flush()
        targets[split] = outputs
        diagnostics[split] = dict(
            max_c_recovery_abs=max_recovery, max_relative_weak_residual=max_residual
        )
    if args.test_only:
        scales = json.loads(args.training_scales.read_text())
    else:
        scales = dict(
            x_mean=[float(inputs["train"].mean())],
            x_std=[float(inputs["train"].std())],
            y_mean=[0.0],
            y_std=[float(np.sqrt(np.square(targets["train"]).mean()))],
            kappa_mean=[float(np.mean(OMEGAS))],
            kappa_std=[float(np.std(OMEGAS))],
        )
        (args.out / "training_scales.json").write_text(json.dumps(scales, indent=2) + "\n")
    arrays = {}
    for split, (lo, hi) in bounds.items():
        n = (hi - lo) * 4
        physical = np.asarray(targets[split]).reshape(n, 101, 101)
        omega = np.tile(OMEGAS, hi - lo).astype("f4")
        group = np.repeat(np.arange(lo, hi), 4)
        arrays.update(
            {
                split + "_x": np.repeat(
                    (
                        (inputs[split].reshape(hi - lo, -1) - scales["x_mean"][0])
                        / (scales["x_std"][0] * 101)
                    ).astype("f4"),
                    4,
                    axis=0,
                ),
                split + "_y": (physical.reshape(n, -1) / scales["y_std"][0]).astype("f4"),
                split + "_physical": physical[:, None, None].astype("f4"),
                split + "_kappa": (
                    (omega[:, None] - scales["kappa_mean"][0]) / scales["kappa_std"][0]
                ).astype("f4"),
                split + "_raw_tcool": omega,
                split + "_group": group,
                split + "_indices": group * 4 + np.tile(np.arange(4), hi - lo),
                split + "_id": np.array(
                    [f"helmholtz4w_{g:05}_omega{int(w)}" for g, w in zip(group, omega)]
                ),
            }
        )
    xy = np.linspace(0, 1, 101)
    xx, yy = np.meshgrid(xy, xy, indexing="ij")
    arrays["coords"] = np.column_stack(
        (np.zeros(10201), np.zeros(10201), xx.ravel(), yy.ravel())
    ).astype("f4")
    metadata = dict(
        channels=["excitation"],
        field_shape=[1, 1, 101, 101],
        input_divisor=101,
        scales=scales,
        frequencies=list(OMEGAS),
        author_commit=AUTHOR_COMMIT,
        source_file_sha256=source_hashes,
        generation_diagnostics=diagnostics,
        counts={
            split: dict(rows=(hi - lo) * 4, groups=hi - lo) for split, (lo, hi) in bounds.items()
        },
        split_base_field_rows=bounds,
        test_accessed=args.test_only,
        normalization_scope="4000 training trajectories only; zero offset and training physical RMS.",
        numerical_limitation="Native101 author discrete benchmark; no continuum-convergence claim.",
    )
    arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    path = args.out / ("prepared_test.npz" if args.test_only else "prepared_train_val_rms.npz")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps(dict(path=path.name, sha256=sha(path), **metadata), indent=2))


if __name__ == "__main__":
    main()
