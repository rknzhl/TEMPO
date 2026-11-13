Sod: 48 modes, three primitive fields, the original 16×128 grid and 1024 test cases.
One seed (17); TEMPO settings were selected on validation.

The small TEMPO has 11,866 neural parameters and 3.270% test error;
global POD has 11,952 parameters and 3.369% after 2000 training epochs.
Both use the same data and total mode count. Fixed POD arrays are separate
from the neural parameter count.

The larger global POD with 427,440 neural parameters remains more accurate
(3.139%). These candidates do not replace the original task summaries.

Weights and training settings are listed in `paper/configs/sod_capacity.json`.
Render the saved examples with `python paper/figures/plot_sod_capacity.py`.
