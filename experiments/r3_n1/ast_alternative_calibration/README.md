# AST alternative-calibration exploration

This directory is experimental, not an official calibration or deployment bundle.
The approved R3-N1 candidate and its raw-score fallback remain unchanged.

- `PLAN.md`: prospective fixed design, optimizer choices, binning, gates and limits.
- `run.py`: CPU runner using existing prediction artifacts and shared utilities.
- `test_experiment.py`: standard-library unittest fixtures (no pytest required).
- `fixture_verification*.json`: executed test counts and source hashes.
- `runs/<timestamp-id>/prospective.json`: plan, environment and source/input hashes
  saved before fitting in that run.
- Each analysis subdirectory contains `predictions.csv` and `results.json`:
  exact fold membership, per-fold counts/parameters/metrics, pooled OOF metrics,
  paired speaker-bootstrap intervals, ten-bin reliability tables and influence.
- `operating/classification.json`: saved-rule versus exploratory optimized-rule
  classification results. No thresholds in this file are officially promoted.
- `summary.json`: all predeclared comparisons and decision gates.
- `manifest.json`: completed run output hashes. A run without `summary.json` and
  `manifest.json` is incomplete and must not be treated as a completed experiment.

From project root, with the existing dependencies available:

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s experiments/r3_n1/ast_alternative_calibration -p test_experiment.py -v
.venv/Scripts/python.exe -B experiments/r3_n1/ast_alternative_calibration/run.py
```

The runner creates a new directory every time; it never resumes or replaces a run.
No model is loaded, no GPU is required, and no held-out audio or scores are accessed.
Canonical metadata are read only to validate the existing development recordings.
No packages are installed by these commands.

The first attempt `20261114T112206Z-e2c46249` stopped after calibration CV when
the shared decision API rejected a NumPy scalar. Its outputs are preserved as
incomplete. The caller was corrected to pass Python float, with an additional
regression test. No mathematical or research-policy change was made.
