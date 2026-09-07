# Hyperparameters

Each directory here holds `metrics.json` for one row of the paper's tables: test errors, the
EM log (`phase1_log`), the online-phase log (`phase2_log`) and, where it was recorded, the
run configuration (`hparams`).

| run | $\eta_{\mathrm{KL}}$ | $\eta_H$ | $\theta_s$ | $\theta_l$ | $\theta_c$ | $K_{\max}$ | $d_0$ | stride | $M$ |
|---|---|---|---|---|---|---|---|---|---|
| `burgers_tempo_npod_M3` | 0.1 | 0.1 | 1e-2 | 1e-1 | 5e-3 | 32 | 25 | 1 | 3 |
| `burgers_tempo_npod_M2` | 0.1 | 0.1 | 1e-2 | 1e-1 | 5e-3 | 32 | 25 | 2 | 2 |
| `darcy_tempo_pod_M5` | 0.1 | 0.1 | 1e-10 | 1e-1 | 0 | 32 | 25 | 4 | 5 |
| `darcy_tempo_npod_M5` | 0.1 | 0.1 | 1e-10 | 1e-1 | 0 | 32 | 25 | 4 | 5 |

Columns map to `hparams` keys in order: `lambda_kl`, `lambda_ent`, `eps_skip`, `eps_large`,
`eps_conv`, `basis_max_modes`, `P_global`, `sensor_stride`, `M`. All four runs used
`max_em_iters=30` and `seed=42`. $m_s$ is the sensor count after subsampling by `stride`.

`burgers_tempo_pod_M3`, `burgers_tempo_pod_M2`, `ns_tempo_pod_M4` and `ns_tempo_npod_M4`
predate configuration logging; take their settings from the `argparse` defaults of the
corresponding script in `scripts/`.

Two symbols were never exposed as command-line arguments, so the code defaults hold for
every reported run: $\theta_p$ is `TEMPOConfig.eps_prune` = 0.01 (`models/tempo.py`) and
$\tau$ is `FourierNeuralPODConfig.tol` = 1e-3 (`models/fourier_neural_pod.py`). Note that
`PODConfig.tol` = 0.9999 shares the name but is a cumulative-energy threshold, not $\tau$.
