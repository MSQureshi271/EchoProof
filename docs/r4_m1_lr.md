# R4-M1 LR-A fit

`scripts/r4_m1_lr.py` runs the verified A preflight and fits one fresh LR model using the recipe in `configs/r4_m1_recipes.json`. It reuses the selected B hyperparameters, recording-equal weights and train-only median imputation and weighted scaling. It evaluates mean chunk probabilities at recording level on the selection role at threshold 0.5. Calibration and operating features are validated by preflight but are not scored in this step. No test waveform or test feature table is read.

On Drive, after the A preflight inputs and canonical shards are available:

```bash
PYTHONPATH=src python scripts/r4_m1_lr.py --root /path/to/project --shard-dir /path/to/shards --output /path/to/project/runs/r4/UNIQUE_LR_RUN
```

The output must be a new direct child of `runs/r4`. The fit saves `model/weights.npz`, `model/lr_model.json`, selection chunk and recording Parquet predictions, coefficients, and evidence with fit diagnostics, provenance, selection metrics, and save/reload parity. A nonconvergent fit or other failure writes `failure.json` in the new run directory; inspect it and use a fresh run directory instead of overwriting. Ledger updates and the remaining validation exports are later R4-M1 steps.

Local fixture tests validate training-only preprocessing, isolation from calibration and operating feature values, selection filtering, and model reload. Real A inputs and training must still be run on Drive. The preflight validates shard availability, but actual waveform PCM checks occur only if those waveforms are decoded by the datasets; this LR fit uses prepared features and does not decode them.
