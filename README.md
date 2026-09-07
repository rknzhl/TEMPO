# TEMPO: Regime-Aware Operator Learning with Locally Adapted POD Bases

TEMPO recovers latent solution regimes from snapshot data by expectation-maximization, fits
a dedicated linear (POD) or nonlinear (NeuralPOD) basis to each, and routes new inputs with
a gating network trained online.

![TEMPO overview](docs/overview.png)

## Install

```bash
pip install -r requirements.txt
git clone https://github.com/camlab-ethz/ConvolutionalNeuralOperator ~/CNO
```

The scripts also look for CNO next to this repository or at `$CNO_PATH`.

## Data

1D Burgers' and 2D Darcy flow come from [PDEBench](https://github.com/pdebench/PDEBench);
`utils/datasets.py` downloads them on first use. The 2D Navier-Stokes data is generated
here:

```bash
python scripts/data_generation/generate_navier_stokes_spectral.py
```

## Layout

```
models/   TEMPO (offline EM, online gating), POD and NeuralPOD bases, DeepONet branches
utils/    dataset loading, plotting
scripts/  one training script per method and benchmark, plus ablations and benchmarks
runs/     metrics, EM logs and hyperparameters of the runs reported in the paper
```

## Running

Run from the repository root; every script supports `--help`.

```bash
# TEMPO
python scripts/burgers/train_tempo.py --nu_values 0.001 0.1 1.0 --M 3 --basis_type pod
python scripts/darcy/train_tempo_darcy.py --M 5 --basis_type pod
python scripts/navier_stokes/train_tempo_navier_stokes.py --re_values 100 1000 3600 10000 --M 4 --no-kappa-init

# baselines
python scripts/burgers/train_pod.py --nu_values 0.001 0.1 1.0
python scripts/darcy/train_fno_darcy.py
python scripts/navier_stokes/train_cno_navier_stokes.py

# analysis
python scripts/benchmark_inference.py
python scripts/verify_theorem.py
python scripts/ablation_kappa.py --benchmark darcy --run_dir <dir with trainer.pt>
python scripts/burgers/generate_basis_figures.py
```

On Navier-Stokes the regimes are initialised from `log Re` by default; `--no-kappa-init`
falls back to the GMM initialisation.

`runs/` holds the metrics and configuration of the eight TEMPO runs; see
`runs/HPARAMS.md`. Checkpoints are not tracked here because of their size.
