"""Reproduce the refinement and regime-weight tables from saved arrays."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def refinement_table(manifest):
    writer = csv.writer(sys.stdout)
    writer.writerow(["task", "seed", "n_test", "relative_l2"])
    for item in json.loads(manifest.read_text())["models"]:
        with np.load(ROOT / item["error_data"], allow_pickle=False) as z:
            error = z[item["error_key"]].astype("f8")
        if (
            error.shape != (item["split"]["test"],)
            or not np.isfinite(error).all()
            or (error < 0).any()
        ):
            raise ValueError("Invalid saved test errors for " + item["task"])
        value = float(error.mean())
        if not np.isclose(value, item["test_relative_l2"], rtol=1e-10, atol=1e-12):
            raise ValueError("Saved test errors differ from the model manifest: " + item["task"])
        writer.writerow([item["task"], item["seed"], len(error), value])


def regime_weights():
    with np.load(
        ROOT / "paper/results/regime_weights/burgers_m2_validation.npz", allow_pickle=False
    ) as data:
        weights = data["gate"]
        viscosity = data["viscosity"]
        ids = data["ic_source_row"]
    assert weights.shape == (len(ids), len(viscosity), 2)
    assert len(ids) == len(set(ids)) == 400
    assert np.isfinite(weights).all() and np.allclose(weights.sum(-1), 1.0)
    labels = weights.argmax(-1)
    changed = np.any(labels != labels[:, :1], axis=1)
    print(
        json.dumps(
            dict(
                mean_weights=weights.mean(0).tolist(),
                viscosity=viscosity.tolist(),
                changed=int(changed.sum()),
                total=len(ids),
                changed_percent=100 * float(changed.mean()),
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", choices=("refinement", "regimes"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper/configs/refinement.json")
    args = parser.parse_args()
    if args.table == "refinement":
        refinement_table(args.manifest)
    else:
        regime_weights()


if __name__ == "__main__":
    main()
