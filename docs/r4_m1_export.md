# R4-M1 validation export

After LR-A and BiGRU-A training complete, run the exporter once for each model in new direct children of `runs/r4`:

```bash
PYTHONPATH=src python scripts/r4_m1_export.py --root /path/to/project --shard-dir /path/to/shards --run-dir /path/to/project/runs/r4/LR_RUN --output /path/to/project/runs/r4/LR_EXPORT --model lr
PYTHONPATH=src python scripts/r4_m1_export.py --root /path/to/project --shard-dir /path/to/shards --run-dir /path/to/project/runs/r4/BIGRU_RUN --output /path/to/project/runs/r4/BIGRU_EXPORT --model bigru
```

The exporter rechecks the A preflight, completed training evidence, selected model hashes and saved selection predictions before scoring calibration or operating data. LR selection scores are compared to the saved selection predictions within `1e-12`; BiGRU uses CUDA and a `1e-6` probability tolerance. A mismatch fails and must be investigated rather than relaxing the tolerance.

Each export writes chunk and recording Parquet files for selection, calibration and operating roles, with identity, group, component, label, role, raw score and recording chunk counts. It validates coverage and persisted readback, recomputes recording means, and records hashes in `evidence.json`. These are raw validation predictions for R4-N1, not calibrated bundles or independent test results. LR parity here is on prepared features; full waveform-to-bundle parity is a later packaging task.

The output folder is never overwritten. If an export fails after creating it, preserve the partial folder for inspection and use a new output directory after fixing the issue. The Drive inputs and both trained models are needed to run the real exports; local synthetic tests do not establish GPU or real-data parity.
