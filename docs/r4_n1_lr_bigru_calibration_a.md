# R4-N1: LR-A and BiGRU-A calibration (Nouman)

Code: `src/echoproof/evaluation/r4_n1.py`, `scripts/r4_n1_calibrate.py` (`--model {lr,bigru}`), recipe `configs/r4_n1_calibration_a.json` (status `user_predeclared_2026-11-15`), `tests/test_r4_n1_core.py`, notebooks `notebooks/R4_N1_LR_A_Calibration.ipynb` and `notebooks/R4_N1_BiGRU_A_Calibration.ipynb`.
Runs (Colab, 15 November 2026, one run per model): `runs/r4/r4_n1_lr`, `runs/r4/r4_n1_bigru` (`input_audit.json`, `fit/`, `operating/`, `execution_environment.json`). Owner: Nouman. Review not recorded.

## Procedure

The team-agreed B WavLM procedure (R3-N1) on Protocol A: identity, temperature and affine candidates on log-odds, leave-one-speaker-out per-speaker log loss on the calibration role (1,200 recordings, 12 speakers), the simplest candidate within one standard error of the best, strict acceptance on the operating role (strictly lower log loss, Brier not higher), macro-F1 threshold with the contract tie-breaks. The operating role (1,300 recordings, 13 speakers) was opened after the mapping was frozen; no test partition was read (`test_data_read` false in every output).

## Result

| Item | LR-A | BiGRU-A |
|---|---|---|
| Leave-one-speaker-out mean log loss (identity / temperature / affine) | 0.2273 / 0.2261 / 0.2213 | 0.0241 / 0.0347 / 0.0247 |
| Best candidate / standard error | affine, 0.0538 | identity, 0.0109 |
| Selected | **identity** (all candidates within one standard error, simplest wins) | **identity** (best outright) |
| Calibration accepted | no ("identity mapping; retain raw scores") | no |
| Score type | `model_score` (raw) | `model_score` (raw) |
| Threshold (`score >= t`, raw scores) | 0.5911 | 0.6109 |
| Operating macro F1 at the threshold (at 0.5) | 0.8912 (0.8607) | 0.9741 (0.9716) |
| Operating genuine FPR at the threshold | 0.1808 (47 of 260) | 0.0731 (19 of 260) |
| Operating log loss / Brier / ECE / AUC / EER | 0.2130 / 0.0600 / 0.0538 / 0.9571 / 0.0933 | 0.1243 / 0.0159 / 0.0169 / 0.9807 / 0.0500 |
| Speaker-bootstrap macro F1 (95%) | 0.8356 to 0.9328 | 0.9227 to 1.0000 |
| Speaker-bootstrap genuine FPR (95%) | 0.0692 to 0.3154 | 0.0000 to 0.2192 |
| Threshold plateau | 1 partition of 1,301 at the maximum | 1 partition of 1,139 at the maximum |

Neither model was calibrated: identity is rejected by definition, so both keep their raw scores and the operating role only sets the threshold.

### Per-component panels (operating, at the threshold)

| Generator | LR-A macro F1 / TTS recall | BiGRU-A macro F1 / TTS recall |
|---|---|---|
| Melo | 0.9070 / 0.9962 | 0.9634 / 1.0000 |
| VITS | 0.8403 / 0.8615 | 0.9596 / 0.9923 |
| XTTS | 0.9051 / 0.9923 | 0.9634 / 1.0000 |
| Chatterbox | 0.9012 / 0.9846 | 0.9634 / 1.0000 |

VITS is the weakest generator for both models. Each panel pairs the same 260 genuine recordings with 260 TTS recordings, so the genuine false-alarm rate is identical across panels.

### Speaker concentration of genuine false alarms

- **BiGRU-A:** all 19 false alarms come from one speaker, `speaker:6454` (19 of 20 genuine recordings flagged); the other 12 speakers have none. The 7.3% genuine FPR is therefore a one-speaker effect, and the wide bootstrap interval reflects that.
- **LR-A:** six speakers have false alarms (`speaker:3699` 7/20, `4406` 1/20, `5561` 11/20, `6454` 7/20, `6476` 11/20, `7517` 10/20). `speaker:6454` is the common speaker.

## Checked locally after sync (16 November 2026)

- The operating results' hash matches `operating/stage_complete.json`; the fit files match the hashes in `fit/stage_complete.json`; `input_audit.json` passed with no problems (2,500 selection and 1,200 calibration recordings, 25 and 12 speakers); chunk means equal the recording scores to 2e-16 (LR) and 0 (BiGRU).
- Every prediction file, export evidence and model file named in the recipe matches its hash; the recipe file hash equals the hash recorded in `execution_environment.json` (`cfa3c590...`).
- `tests/test_r4_n1_core.py` passes (43 tests with the AST and WavLM suites); no tracked B source or lock changed.

## Bundles and waveform parity (built 16 November 2026)

Both models keep raw scores, so each development bundle is the model plus a serialized threshold rule (`decision/score_rule.json`, also in `bundle.json`). Code: `src/echoproof/models/r4_n1_raw_bundles.py` (builder and loaders `load_lr_bundle_a`, `load_bigru_bundle_a`, which refuse a threshold override), `scripts/r4_n1_raw_bundle.py`, `scripts/r4_n1_raw_parity.py`, `notebooks/R4_N1_Colab_LR_BiGRU_Parity.ipynb`, `tests/test_r4_n1_raw_bundles.py`. No B file was edited. The bundles are not promoted and not locked.

| | LR-A | BiGRU-A |
|---|---|---|
| Bundle | `bundles/r4_n1_lr_A_raw_threshold` (16 KB), id `bundle:r4-n1-lr-A-raw-threshold`, `logistic_regression:A` | `bundles/r4_n1_bigru_A_raw_threshold` (1.8 MB), id `bundle:r4-n1-bigru-A-raw-threshold`, `bigru:A` |
| `bundle.json` sha256 | `308640fd...` | `3cf8b5c7...` |
| Rule, score type | `score >= 0.5910800081638822`, `model_score` | `score >= 0.610876834796857`, `model_score` |
| Saved-score replay (1,300 operating recordings, no audio) | 47 FP, 43 FN, macro F1 0.8912, as recorded | 19 FP, 2 FN, macro F1 0.9741, as recorded |
| Waveform parity (Colab; CPU / Tesla T4) | passed: 1,500 recordings (1,300 operating, 100 selection, 100 calibration) | passed: 1,500 recordings, same roles |
| Largest recording-score difference (tolerance) | 3.9e-7 (1e-6) | 1.2e-7 (1e-5) |
| Chunk-count mismatches against the export | none | none |
| Tier 2 (re-encoded WAV through `analyze`) against tier 1 | identical (0.0) | identical (0.0) |
| Decision flips | 1, explained (see below) | 1, explained (see below) |

Reports: `runs/r4/r4_n1_lr/handoff/waveform_parity.json`, `runs/r4/r4_n1_bigru/handoff/waveform_parity.json` (bundle hashes in both equal the packaged bundles). Environment: torch 2.14.1+cu130, numpy 2.4.6, librosa 0.11.0 on Colab (the local environment has librosa 1.0.0, so a local recomputation of LR features is not the verified path). Tolerances were the exports' own (1e-6 and 1e-5) and were not relaxed.

### The one flip per model

The threshold is an observed operating score (the contract picks the lowest score that reaches the best partition), so the recording that sits exactly on it is decided by float rounding. LR-A: `phonemedf:vits:1040-133433-0040`, saved 0.5910800081638822, replayed 0.591079883470258 (difference 1.2e-7), decided genuine instead of TTS. BiGRU-A: `phonemedf:xtts:1040-133433-0029`, saved 0.610876834796857, replayed 0.6108768222423221 (difference 1.3e-8), same effect. Both flips are within the tolerance of the threshold, so the parity rule counts them as explained, and the scoring is otherwise reproduced to 1e-7. In practice one recording per model can change side across hardware or library versions; the reported operating figures count those recordings as TTS. LR-A also has many recordings near its threshold (11 within 0.01), consistent with its weak separation.

### Reading the result

This verifies decode, chunking, scoring and the serialized decision from real audio against the saved scores. It does not test resampling of the original files (the shards hold canonical audio), another library version, or performance on the sealed A test partition. Operating figures remain development results.

## Limits and open items

- **Bundles exist now** (see above); the app and CLI routing for A bundles is an R4-N2 decision, and `locks/A_lock.json` is not written.
- Thresholds were chosen on the same 260 genuine recordings used for the figures above: development results, not test performance. A has no unseen generator. Single seed.
- The recipe's `scope` note says the WavLM-A and AST-A exports were not in that checkout; it applies to these two runs only, which use LR and BiGRU inputs.
- `fit/stage_complete.json` and `operating/stage_complete.json` of the BiGRU run carry the task label "R4-N1 LR-A ..." (a copy of the LR label); the `model` field and the results themselves are BiGRU. A labelling slip, not a data problem.
- The six new source, test and notebook files of the calibration (and the bundle and parity files) are not yet committed. Independent review is not recorded.
