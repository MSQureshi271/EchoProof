# R2-M2: BiGRU diagnostics

Sync `src/`, `scripts/r2_m2_bigru.py`, `tests/test_r2_m2_diagnostic.py` and `notebooks/R2_M2_Colab_BiGRU_Diagnostics.ipynb` to the Drive project. Select a T4 runtime and run the notebook in order. If shards have been copied to local scratch, set `SHARDS` to that directory before running.

The diagnostic selects 32 retained Protocol B training recordings, balanced by label and reproducible with seed 42. It checks the frozen split and R2-S1 mel statistics, verifies each selected recording's canonical PCM hash, and extracts its first canonical four-second window. The per-bin statistics are the frozen training-only R2-S1 statistics. It checks valid frame lengths, mask sums, and whether the BiGRU's output for a short sequence changes when batched beside a longer one.

The real two-layer BiGRU attempts to memorize this tiny fixed training subset for at most 100 epochs, with an early success target of 95% training accuracy. This is a diagnostic, not an estimate of model performance. No validation, test or Chatterbox audio is scored. The model and optimizer remain in memory and are discarded at process exit; no research checkpoint is produced. The evidence JSON contains identifiers, data and source hashes, frame lengths, training history, padding difference and environment. A `verified: false` result or an exception requires investigation; do not report a pass from the notebook's existence.

Share `reports/r2/r2_m2_<timestamp>.json` for Usman's review. R3 BiGRU training and selection are separate work.
