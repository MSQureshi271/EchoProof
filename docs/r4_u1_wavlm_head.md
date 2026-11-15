# R4-U1a: WavLM head, Protocol A

Code: `src/echoproof/training/r4_u1_wavlm_data.py` (A data, alignment, parity sample), `src/echoproof/training/r4_u1_wavlm_train.py` (recipe guard, training, selection, export), `scripts/r4_u1_wavlm_head.py` (`check-data`, `train`, `parity`, `package`).
Tests: `tests/test_r4_u1_wavlm.py`. Colab: `notebooks/R4_U1_Colab_WavLM_Head_A.ipynb`. Recipe: `configs/r4_u1_wavlm_head_a.json` (status `team_agreed_user_reported_2026-11-15`).
Owner: Usman (ran it on Colab). Reviewer: Muneeb (plan table); review not yet reported.

## What it is

The B-selected configuration run unchanged on Protocol A: `Linear(1536,128) -> GELU -> Dropout(0.2) -> Linear(128,2)` on mean+std WavLM embeddings, AdamW lr 1e-3, weight decay 0.01, batch 128, constant lr, seed 42, always 20 epochs, best selection epoch kept, crop weight 1 / (crops of its recording). The loader refuses any difference from the B recipe's model and train blocks. Inputs are the 60,194 A embeddings read through `r4_u0.read_embeddings_a` (B cache plus the R4-U0 cache); the A test partition is never read.

## Result (Colab run 15 November 2026, reviewed locally)

| Item | Value |
|---|---|
| Alignment | passed: 35,782 crops (7,550 recordings, 1 to 5 crops each; chatterbox 7,222, genuine 7,234, melo 7,130, vits 7,106, xtts 7,090), 24,412 chunks (selection 12,331, calibration 5,777, operating 6,304); A split hash `a9075367…` |
| Selected epoch | 13 of 20; selection-role recording macro F1 at 0.5 = **0.999375** (2,500 recordings) |
| Selection panels (informational) | genuine versus chatterbox, vits, xtts: F1 1.000, ROC-AUC 1.0; versus melo F1 0.999 (one miss, FNR 0.002) |
| Training | 0.46 minutes on CPU; macro F1 0.987 at epoch 1, flat at 0.999 from epoch 9 |
| Parity (direct WavLM waveforms against cached vectors) | 50 selection chunks, all five components, 25 short; max embedding difference 8.3e-7, max probability difference 1.2e-7, 0 decision flips (tolerance 1e-4); GPU run, TF32 off |
| Bundle | `bundles/r4_u1_wavlm_head_A` (`bundle:r4-u1-wavlm-head-A-mean_std`, protocol A, raw uncalibrated, 248 encoder tensors equal to the pinned source); all five component hashes verified on this machine |
| Selection frozen | `selection.json` sha256 `0af09a9b…`, written before the calibration and operating roles were scored; checkpoint `d38dded7…` |
| Exports | `predictions_{chunks,recordings}_{selection,calibration,operating}.parquet`, B column layout; all hashes and the selection, checkpoint, evidence and parity hashes match the Colab report |

Colab: Tesla T4, torch 2.14.1+cu130, transformers 5.22.0. The extraction identity of the A cache equals B's. Source hashes recorded in `environment.json` equal this repository's files.

## How to read the result

- The score is very high because A is in-distribution: every generator is in training, validation and test. It is not comparable with B (selection F1 0.914), where XTTS was unseen. A high A score says nothing yet about new generators; that is a limit of the protocol, not of the head.
- The selection role mixes five generators, so the per-component panels above should be read beside the headline number.
- Single seed. From epoch 9 the selection F1 is within a few recordings of the best, so the exact epoch (13) is not meaningful beyond "the plateau".
- Raw scores are uncalibrated; thresholds come from R4-N1.

## Open items

- Ledger: the row `r4_u1_wavlm_head_A_20261114T235631Z` (protocol A, seed 42, 20 epochs, best epoch 13, F1 0.99938, EER 0.0, FPR 0.0, split hash `a9075367…`, config hash `b7514d21…`, completed) is in the synced `reports/ledger.csv` and matches `status.json`.
- The review by Muneeb is not recorded.
