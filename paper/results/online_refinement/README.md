# Saved online refinement

`selection.json` identifies the exact models used in the updated tables.
For Burgers, Sod, Coanda and Helmholtz, original/refined series were compared
by their mean test error; every seed in the selected series is retained.
Coanda POD retains its original three-seed series. NS refinement was selected
on independent validation before the paired test.

The candidate test receipts describe the earlier evaluation of models fixed
on validation. The subsequent series selection is recorded separately; the
receipts' original no-test-selection flags describe that earlier evaluation.
Source receipt hashes are retained. `source_archive/` paths identify provenance
outside the repository and do not imply those intermediate files are included.

`training/` records the initial recipe and saved refinement settings. Bases,
Fourier frequencies, offline responsibilities and architectures are unchanged.
The coefficient inverse is folded into the exported branch layers.

All previous checkpoint files remain at their original paths. The manifest's
`superseded_checkpoints` distinguishes them from the active selection.
No training, new test evaluation, commit or push was performed during this import.
