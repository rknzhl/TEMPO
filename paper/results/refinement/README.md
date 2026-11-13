The five checkpoints in `paper/configs/refinement.json` use fixed-rank joint
NeuralPOD refinement. `paper/results/refinement.csv` reports paired test errors as
unitless relative L2, one seed per task. The original task summaries remain
separate; these rows are not averages over the earlier three-seed series.

Each `*_test.npz` retains individual test errors and sample identities.
Run `python scripts/paper/summarize.py refinement` to reproduce the five means.
Burgers, NS and Sod field examples are in `paper/figures/data/refinement/`; they select
one low-error illustration per physical parameter, while aggregate metrics
include every test trajectory.

On NS, the online-only control reached 9.007456% validation error, compared
with 9.156109% initially and 8.989394% after joint basis/online refinement.
The full test was evaluated for the validation-selected joint model.

The optimizer implementation is `models/neural_refinement.py`.
The task configs record the saved training settings. Helmholtz and Sod use successive refinement stages; their configs record
these stages rather than a single continuation command.

Render the Sod fields with `python paper/figures/plot_sod_refinement.py`.
