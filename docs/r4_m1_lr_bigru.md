# R4-M1: LR-A and BiGRU-A (Muneeb)

Code: `src/echoproof/training/r4_m1_{data,lr,bigru,export}.py`, `scripts/r4_m1_{preflight,lr,bigru,export}.py`, `configs/r4_m1_recipes.json`, `notebooks/R4_M1_Colab.ipynb`. Tests: `tests/test_r4_m1_*.py`. Full training report (gitignored folder): `reports/r4/r4_m1_training_report.md`. Reviewer: Usman (review not recorded).

## What was run

The B-selected configurations were refitted on Protocol A (split `a9075367...`, recipe `d5ff4f13...`): LR with 67 features, C 0.01, no class weighting (36 iterations); BiGRU with two bidirectional layers, hidden 128, dropout 0.3, AdamW 3e-4, batch 32, 20 epochs, selected epoch 19 (resumed once, so not bit-exact). Both use A training-only preprocessing and checkpoint selection on the selection role at fixed 0.5. Calibration and operating predictions were exported only after selection; no test data was touched.

## Results (raw scores, recording macro F1 at 0.5)

| Model | Selection | Calibration | Operating | Genuine flagged (operating) |
|---|---|---|---|---|
| WavLM-A | 0.9994 | 0.9961 | 0.9976 | 0.8% |
| AST-A | 0.9969 | 0.9935 | 0.9940 | 1.5% |
| BiGRU-A | 0.9750 | 0.9908 | 0.9716 | 8.1% |
| LR-A | 0.8533 | 0.8476 | 0.8607 | 29.6% |

All four exports share the same chunk and recording ids and labels in each role (2,500 selection, 1,200 calibration, 1,300 operating recordings). LR-A also misses about 10% of VITS recordings at 0.5.

## Checked locally (15 November 2026)

Every model, training-evidence and export hash in the report matches its file. The LR run's source hashes equal the repository files. The BiGRU contract's hashes of `acoustic.py` and `bigru_waveform.py` differ from the repository; the report says the Drive copies differed (moved librosa imports; optional frontend buffer arguments) with unchanged calculations on the default path, a source inspection and not a GPU reproduction.

## Limits and open items

Single seed; in-distribution A only (no unseen generator); the BiGRU checkpoint is the best of 20 selection epochs; no LR or BiGRU bundle and no waveform-to-bundle parity yet; thresholds and calibration belong to R4-N1.
