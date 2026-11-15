# R4-N1 summary: A calibration for all four models

Date of the check: 16 November 2026. Per-model documents: `docs/r4_n1_wavlm_calibration.md` (Subhan), `docs/r4_n1_ast_calibration_a.md` (Usman), `docs/r4_n1_lr_bigru_calibration_a.md` (Nouman). Decision log: the R4-N1 entries of 15 and 16 November 2026.

## Outcome per model (operating role, 1,300 recordings from 13 speakers, 260 genuine)

| Model | Mapping | Accepted | Score type | Threshold | Macro F1 (at 0.5) | Genuine FPR | Errors at the rule |
|---|---|---|---|---|---|---|---|
| WavLM-A | affine (a 1.4385, b 0.1976) | yes | calibrated probability | 0.6254 | 1.0000 (0.9976) | 0 / 260 | 0 of 1,300 |
| AST-A | temperature (T 0.2921) | yes | calibrated probability | 0.9703 | 0.9964 (0.9940) | 1 / 260 | 3 of 1,300 (1 FP, 2 FN) |
| BiGRU-A | identity | no | raw model score | 0.6109 | 0.9741 (0.9716) | 19 / 260 | 21 of 1,300 (19 FP, 2 FN) |
| LR-A | identity | no | raw model score | 0.5911 | 0.8912 (0.8607) | 47 / 260 | 90 of 1,300 (47 FP, 43 FN) |

Thresholds and figures come from the same operating role, so they are development results; the A test partition was never read by any of the four runs. A has no unseen generator.

## What the evidence shows

- WavLM-A and AST-A are near ceiling. Calibration was accepted for both but changed little: operating log loss 0.0164 to 0.0090 (WavLM) and 0.0133 to 0.0120 (AST). For AST-A the Brier-difference bootstrap interval includes zero, so passing the point-estimate rule is not statistical confirmation. Ranking is unchanged by either mapping (AUC and EER identical to raw).
- LR-A and BiGRU-A kept their raw scores. Their calibration fits were not better than identity by the one-standard-error rule, so the work reduced to choosing a threshold. LR-A remains the weak model (genuine FPR 18%, VITS recall 86%).
- BiGRU-A false alarms are one speaker: `speaker:6454` accounts for all 19 (19 of 20 of that speaker's genuine recordings); the same speaker is also among LR-A's false-alarm speakers.
- Separation on the operating role: WavLM-A separates it perfectly, so one of 1,179 partitions reaches macro F1 1.0 and the threshold equals the lowest TTS score; the real margin is the interval (0.5689, 0.6254] on calibrated scores. In AST-A, LR-A and BiGRU-A one partition also reaches the maximum, which is a count of observed partitions, not the width of a threshold interval.

## What was synced and checked locally (16 November 2026)

| Item | State |
|---|---|
| `runs/r4/r4_n1_wavlm/20261115T180831Z-2ecb50405784` | complete; hashes match `handoff.json` and `evidence.json` |
| `bundles/r4_n1_wavlm_head_A_calibrated` | present; model components byte-identical to the source bundle; replay passed (0 flips of 1,500) |
| `runs/r4/r4_n1_ast/candidate_20261115_approved_recipe` | complete (fit, operating, decisions, local verification, reliability plot); recipe input hashes match the files |
| `reports/r4/r4_n1_ast_parity/1b6e0a55....json` | present; sha256 `3e0fd2ef...` equals the hash in the AST document; parity passed (163 chunks of 50 selection recordings, 0 flips) |
| `bundles/r4_n1_ast_A_candidate_20261115` | **not present locally** (the AST document names it); the receipt's bundle hash `88d26ab4...` cannot be rechecked until it is synced |
| `runs/r4/r4_n1_lr`, `runs/r4/r4_n1_bigru` | complete (with `handoff/bundle_packaging.json` and `waveform_parity.json`); results and fit hashes match their stage records; export, prediction and model hashes match the recipe |
| LR-A and BiGRU-A bundles | built 16 November 2026 (`bundles/r4_n1_lr_A_raw_threshold`, `bundles/r4_n1_bigru_A_raw_threshold`); saved-score replay and Colab waveform parity passed on 1,500 recordings each (one explained boundary flip per model); see `docs/r4_n1_lr_bigru_calibration_a.md` |
| Git | WavLM and AST files are committed; the LR/BiGRU files (`configs/r4_n1_calibration_a.json`, `src/echoproof/evaluation/r4_n1.py`, `scripts/r4_n1_calibrate.py`, `tests/test_r4_n1_core.py`, two notebooks) are untracked |
| Tests | `test_r4_n1_core.py`, `test_r4_n1_ast.py`, `test_r4_n1_ast_parity_runner.py`, `test_r4_n1_wavlm.py`: 43 passed; no B source or lock changed |

## Open before `locks/A_lock.json`

1. Sync `bundles/r4_n1_ast_A_candidate_20261115` and recheck its hashes against the parity receipt.
2. LR-A and BiGRU-A bundles are built and verified; the lock script still has to record them, with the threshold sitting on an observed score (one boundary recording per model is decided by float rounding).
3. Independent reviews of R4-N1 (all four) and of R4-M1, R4-U1 are not recorded; the AST candidate and the recipes use different file layouts for the same procedure.
4. Commit the untracked LR/BiGRU files; R4-N2/N3, `locks/A_lock.json`, and R5 on the sealed test partition remain.
