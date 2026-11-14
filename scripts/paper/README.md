# Saved-model prediction

`replay_checkpoint.py` loads the model formats indexed in
`paper/configs/checkpoints.json` and `paper/configs/refinement.json`. It restores saved
weights and Fourier frequencies without training or recomputing data statistics.

```sh
python scripts/paper/replay_checkpoint.py \
  --checkpoint artifacts/checkpoints/navier_stokes/ns_original_improved_seed42_online_refined_20261111/checkpoint.pt \
  --inputs paper/examples/ns/input.npz --input-units raw --out output/ns.npz
```

Inputs are an NPZ with `x[N,S]`, `kappa[N,P]`, `coords[D,d]` and optional
`sample_id[N]`. Coordinates must match the saved model.
`paper/examples/manifest.json` records example paths and input units.

## Units

Normalized model inputs are the default. With `--input-units raw`:

| Model family | Input |
| --- | --- |
| Burgers POD/NeuralPOD | Initial field and viscosity, without additional normalization |
| NS DeepONet / global POD | Raw initial velocities and Re, without normalization |
| Darcy | Raw 128×128 permeability and positive beta; all normalization and pressure recovery are applied |
| Navier–Stokes TEMPO | Both initial velocity components and positive Re; the saved log-Re scaling is applied |
| POD/NeuralPOD basis fits | Raw prepared-dataset inputs, using saved scalar scales |
| Ordinary DeepONet | Prepared-dataset inputs, using its additional saved scales |
| FNO/CNO | Inputs in the saved author adapter's physical units |

Prepared Helmholtz dense inputs already include their recorded normalization;
use normalized mode for those files. For Helmholtz FNO/CNO, raw inputs are
physical wavespeeds. `prepare_replay_inputs.py` handles task-specific conversions.
No scales are estimated from query examples.

Outputs include the model target, prepared and physical fields, gate weights,
sample IDs and checkpoint hash. Sod restores the component scales from its
data contract; NS targets are already physical velocities.

## Baselines

FNO uses `neuraloperator==2.0.0`. Use the recorded CUDA environment to reproduce
the paper's FNO errors. CNO uses `scripts/paper/CNO2d.py`; pass it with
`--cno-source scripts/paper/CNO2d.py`. Source versions are recorded in
`paper/configs/dependencies.json` and checked by the loaders.

Coanda predictions are read out at the native mesh nodes using the saved map.
Darcy models predict unit-source pressure, followed by the common decoder
`u = beta * v`; the native output grid is 128×128.

## Refinement and supporting results

`refine_neuralpod.py` takes a saved fit, a task config and prepared train/validation
arrays. It keeps the rank and Fourier features fixed. Large targets can be passed
as memory-mapped NPY files with `--train-targets` and `--val-targets`.
The five task configurations are indexed in `paper/configs/refinement.json`.

- `summarize.py refinement`: paired test errors.
- `summarize.py regimes`: the main Burgers model's regime weights.
- `python scripts/paper/check_parameter_ablation.py`: parameter replacement errors.
- `python scripts/paper/check_parameter_estimation.py`: known versus estimated parameters.
- `python scripts/paper/check_adaptive_updates.py`: Burgers EM update branches.
- `paper/configs/sod_capacity.json`: the compact Sod models and training settings.

The NS Re estimator uses the initial velocity field. Its weights and training
settings are in `paper/configs/parameter_estimator.json`. To use it during replay, add
`--parameter-estimator artifacts/checkpoints/parameter_estimator/ns_linear/weights.pt`
to the NS command above. `estimate_re.py` contains feature extraction, fitting
and prediction; no future velocity fields are used.

NS DeepONet training settings are in `paper/configs/ns_deeponet.json`.
Cross-Re errors are in `paper/results/ns_deeponet/metrics.csv`.
Training uses 500 minibatch updates, as in the archived script.
