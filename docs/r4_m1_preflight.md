# R4-M1 Protocol A preflight

Step 2 adds `echoproof.training.r4_m1_data.build_data` and the read-only `scripts/r4_m1_preflight.py`. No training or artifact promotion occurs in preflight. The frozen recipe is `configs/r4_m1_recipes.json`.

On the Drive project copy, after syncing the repository and confirming the R4-S1 outputs and canonical shard directory are available:

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_r4_m1_data.py
PYTHONPATH=src python scripts/r4_m1_preflight.py --root /path/to/project --shard-dir /path/to/canonical/shards
```

The preflight checks the recorded B lock and LR/BiGRU locked bundle component hashes, selected BiGRU configuration, A split/cohort/chunk inputs and R4-S1 output hashes, then rebuilds the A plan and checks all crop and validation feature identities, metadata, counts, schema, missing values and group separation. It checks A train-only mel statistics and that the named shard files exist. Dataset waveform decoding validates PCM identity and length lazily as audio is used; preflight does not scan all shard contents.

The GR3 lock is checked by a pinned canonical JSON hash (sorted keys, compact separators, UTF-8, `ensure_ascii=False`), so line-ending and JSON formatting differences do not change its identity. The preflight reports both the actual file hash and the canonical hash. Its locked LR and BiGRU bundle component files continue to require their exact recorded hashes.

The cohort and chunk manifest metadata may include test rows. Sealed split assignments are filtered and no test audio or test feature rows are loaded. The loader exposes separate validation-role datasets; the later training orchestration must enforce that calibration and operating outcomes do not affect model selection. LR uses the fixed feature crop bank; BiGRU uses the B recipe's seeded per-epoch random crop per training recording. This preflight cannot establish that Drive matches until run there.
