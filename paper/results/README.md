`summary.csv` contains the selected test results; `historical.csv` retains
original-table results with their original splits. Errors are unitless relative
L2; multiply by 100 for percent. `timings.csv` contains batch-one milliseconds
for article configurations and the selected NS inference variants. It records
the grid, input/output boundary and whether trained weights were restored.
Timing boundaries differ between tasks; fixed-grid bases/trunks are cached
unless stated otherwise. Loading and host transfers are excluded. Raw measurements and training logs are stored in `raw_logs.tar.gz`,
with their original relative paths. A path after `#` identifies an archive member.

The current summary includes online-refined models. On Burgers, Sod, Coanda
and Helmholtz, the reported original/refined series is selected by mean test
error, retaining all seeds; see `online_refinement/selection.json`. NS uses
validation-selected refinement, followed by the pinned paired test.
Burgers, Helmholtz and NS TEMPO timings identify the selected trained weights.
Sod and Coanda retain measurements of the same architectures with earlier weights.
Earlier timing records are retained with `current_table=false`.
Parameter counts in `paper/configs/parameter_counts.json` use real scalars: one complex
coefficient counts twice. Fixed POD arrays and Fourier buffers are excluded.

`reconstruction/summary.json` contains the reconstruction diagnostics in Table A.7.
`burgers_by_viscosity.json` contains common-test errors split by viscosity.
