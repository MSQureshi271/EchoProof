# R4-N1: WavLM-A calibration, threshold and bundle

Code: `src/echoproof/evaluation/r4_n1_wavlm.py` (audit, fit, operating), `src/echoproof/models/r4_n1_wavlm_bundle.py` (packaging and the A loader), `scripts/r4_n1_wavlm_calibrate.py` (`audit`, `fit`, `operating`, `package`), `scripts/r4_n1_wavlm_replay.py`.
Recipe: `configs/r4_n1_wavlm_calibration_a.json` (status `team_agreed_user_reported_2026-11-15`). Tests: `tests/test_r4_n1_wavlm.py` (28). Colab: `notebooks/R4_N1_Colab_WavLM_Calibration_A.ipynb`.
Owner: Subhan (WavLM-A). AST-A is calibrated by Usman, LR-A and BiGRU-A by Nouman, with the same procedure.

**Status: run on Colab on 15 November 2026 (run `20261115T180831Z-2ecb50405784`); reviewed locally by Claude, independent review not recorded.** Audit, fit, operating, package and replay all completed; the calibration was accepted and the calibrated bundle exists but is not promoted or locked.

## Result

| Item | Value |
|---|---|
| Mapping selected | affine, `a` 1.4385, `b` 0.1976 (`u` 0.3636), fitted on 1,200 calibration recordings from 12 speakers (3 errors at 0.5) |
| Leave-one-speaker-out log loss | affine 0.0107, identity 0.0165, temperature 0.0240 (temperature is worst because of one speaker, `speaker:587`, loss 0.253) |
| Acceptance | accepted: operating log loss 0.0164 to 0.0090, Brier 0.0026 to 0.0016, ECE 0.0141 to 0.0079 |
| Score type and rule | `calibrated_probability`, `score >= 0.6254` (raw-equivalent 0.5545) |
| Operating role at the rule | macro F1 1.0000, genuine FPR 0 of 260, TTS recall 1040 of 1040 (at 0.5: 2 genuine false alarms, macro F1 0.9976) |
| Replay (T4, transformers 5.22.0) | 1,500 recordings (1,300 operating, 100 selection, 100 calibration), 0 decision flips, largest score difference 2.6e-6 (tolerance 1e-3), the two tiers agree exactly |
| Bundle | `bundles/r4_n1_wavlm_head_A_calibrated`, `bundle.json` sha256 `c7d8fefb...`, `calibration.json` `9c9bc1e6...`; the five model components are byte-identical to `bundles/r4_u1_wavlm_head_A` |

Checked locally after syncing: every hash recorded in `handoff.json`, the 12 operating outputs against `evidence.json`, the bundle files and the model components against the source bundle.

### How to read the threshold

The operating role is perfectly separated: every genuine score is below every TTS score, so exactly one score-induced partition (of 1,179) reaches macro F1 1.0. That is expected under perfect separation and does not show the threshold is fragile. The reported threshold, 0.6254, equals the lowest TTS score (a VITS recording) because candidate thresholds are observed scores and the contract tie-break picks that one; it is not learned from that recording.

The real margin is the interval between the highest genuine score and the lowest TTS score: calibrated scores (0.5689, 0.6254], raw scores (0.5139, 0.5545]. Any threshold in that interval gives zero errors on this role; 0.5 gives two genuine false alarms. The two highest genuine scores (0.569 and 0.561) are the closest to the threshold. The contract rule puts the threshold at the top edge of the interval; a value near its middle would give the same operating result, but the rule was not changed.

## Limits of this result

- The threshold and the operating figures come from the same role, so macro F1 1.0 is a development result, not test performance; the A test partition was never read.
- With 260 genuine recordings from 13 speakers, 0 false alarms still allows a true rate of about 1.9% (bootstrap interval 0 to 0.019).
- The mapping was fitted on a calibration role with only 3 errors, so it is weakly determined; the leave-one-speaker-out comparison nevertheless clearly favoured affine.
- Protocol A has no unseen generator, so nothing here measures generalisation. Single seed. The replay's second tier re-encodes canonical PCM and does not test resampling of the original files.

## What it does

The B WavLM procedure (R3-N1) applied unchanged to Protocol A, as accepted by the team:

1. **Audit.** Hash-checks the six WavLM-A exports, `evidence.json`, `selection.json` and the source bundle; checks the roles (2,500 selection, 1,200 calibration, 1,300 operating recordings; 25, 12, 13 speakers; 500, 240, 260 per component, five components), that recording scores equal the mean of their chunk probabilities, and joins every recording to the A cohort and split manifests with the test partition withheld. It recomputes the raw summaries and saturation counts recorded in the recipe.
2. **Fit (calibration role only).** Candidates identity, temperature (bounds 0.01 to 50) and affine (`a = exp(u)`, bounds, regularisation 0.01, L-BFGS-B) on log-odds with epsilon 1e-7; leave-one-speaker-out per-speaker log loss; the simplest candidate within one standard error of the best; failed or bound-hitting fits are excluded and reported. The result is frozen in `stage_complete.json` before the operating file is read.
3. **Operating (opened once).** Acceptance: a valid fit, strictly lower log loss and not higher Brier score than raw; otherwise the raw score stays a `model_score`. The threshold maximises macro F1 over all score-induced partitions (`select_threshold`, contract tie-breaks, `score >= threshold`, explicit all-negative rule). Rows A to D, probability quality, reliability, speaker diagnostics, a speaker-group bootstrap, per-component panels (genuine against each generator) and the selection-role description are written; a second pass in the same run is refused.
4. **Package.** `bundles/r4_n1_wavlm_head_A_calibrated` holds the source model components byte for byte plus `calibration/calibration.json` and the serialized decision rule; `wavlm_head:A`, stage `r4_n1_wavlm_development`, not promoted, not locked.
5. **Replay (Colab, transformers 5.22.0).** All operating recordings and 100 each from selection and calibration through the bundle in two tiers (scorer on canonical windows; WAV bytes through `analyze`), tolerance 1e-3 on recording scores; every flip is listed.

## What is new compared with B, and why

- **No B file is edited.** The B modules are hashed in `locks/B_lock_gr3.json` and hard-code Protocol B, so the A modules import their protocol-free helpers and restate the A-bound parts: the recipe guard (the A recipe's procedure blocks must equal the B recipe's, so A cannot drift; only the reporting key `plateau_report` is additive), five-component role checks, the A manifest join, per-component panels (A has no unseen generator and no duration-matched subset) and A labels in every record. A test checks the imported B sources still match the lock.
- **Plateau report.** WavLM-A makes 3 errors in 1,200 calibration recordings and 2 in 1,300 operating recordings at 0.5, so thresholds could tie on macro F1. The number of tied partitions and the score interval they span are written beside the threshold (`threshold.json`, `metrics.json`, `comparison.md`) so the tie-break is not read as a finding. In the real run the operating role was perfectly separated, so one partition reached the maximum; the plateau report counts partitions and does not give the width of the separating gap (see Result).
- **Loader.** `echoproof.inference.load_inference_bundle` routes the family `wavlm_head` to the B loader, which refuses `wavlm_head:A`. That file is locked, so A bundles load with `load_wavlm_bundle_a`. Routing A bundles in the app and the CLI is a decision for R4-N2.
- **CLI check dropped.** The B notebook compared `echoproof.cli predict` with the library; for the same reason it is not part of the A notebook.

## Expectations before the run (kept for the record)

- WavLM-A is near ceiling (raw operating macro F1 0.9976, genuine FPR 0.8%), so the calibration fit is weakly determined and may well select identity or be rejected by the acceptance rule; whichever outcome the rule gives is reported as it is, and it is not tuned.
- Operating metrics are development results: the threshold is chosen on 260 genuine and 1,040 TTS operating recordings from 13 speakers. They are not test performance, and A has no unseen generator.
- No FPR constraint, prevalence correction or abstention (these need a recorded pre-result amendment). Single seed. Review of this task is not recorded.

## To run (as it was run)

Sync the new files to Drive (list in the decision log), then run the notebook on Colab; or run the four script stages locally with `.venv/Scripts/python.exe scripts/r4_n1_wavlm_calibrate.py <stage> --owner Subhan [--run-dir ...]` (the replay needs the model and the shards, so it runs on Colab). B practice was an official Colab run; a local run is possible because the fit needs only numpy, scipy and pyarrow.
