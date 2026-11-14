# R3-N1: WavLM calibration and operating policy, Protocol B

Status: **finalised as is by the project owner on 14 November 2026** (user-reported; there is no individual or team approval record in the repository). The development bundle exists and is verified. It is not promoted, not locked, and the application's default model is unchanged. Production selection and the lock remain GR3 steps.

Recipe: `configs/r3_n1_wavlm_calibration_b.json` (frozen before any fit). Official run: `runs/r3/r3_n1_wavlm/20261114T083533Z-e6ff5a288068/`. Bundle: `bundles/r3_n1_wavlm_head_B_calibrated/` (bundle id `bundle:r3-n1-wavlm-head-B-calibrated`). Full plan and execution record: `WAVLM_CALIBRATION.md` (kept out of git by the owner's choice). Side-by-side comparison with the other models: `REPORT.md`.

## What was done

Calibration of the selected WavLM head (R3-U1: `mean_std` pooling, epoch 14) was done independently of how the LR and BiGRU models were calibrated. It neither reuses nor matches their methods, thresholds or loaders.

1. **Inputs.** The raw recording scores exported by R3-U1 (`reports/r3/r3_u1_B/`), checked by hash against the recipe.
2. **Candidates.** Identity, recording-level temperature scaling, and an affine mapping of log-odds, `q = sigmoid(a * logit(p) + b)` with a = exp(u), u in [-4, 4], b in [-5, 5], regularisation 0.01.
3. **Mapping selection (calibration role only).** Leave-one-speaker-out held-out log loss over the 12 calibration speakers, with a one-standard-error, simplest-first rule.
4. **Acceptance rule (declared before the operating role was opened).** The mapping is accepted only if it gives a valid fit, strictly lower operating log loss than raw, and no higher Brier score. Otherwise raw scores are kept and labelled "model score".
5. **Threshold.** The contract rule: maximise macro F1 on the operating role, then lower genuine FPR, then closeness to 0.5, then the smaller threshold; `score >= threshold`. No FPR constraint was added, because the plan forbids one.
6. **Operating role opened once, after the fit was frozen.**

## Result

| Item | Value |
|---|---|
| Selected mapping | Affine on log-odds, a = 1.1079811437013474, b = 0.9165647745687742 |
| Temperature reference | T = 1.044 (raw scores are already close to well scaled in slope; the gain is the offset) |
| LOSO mean held-out log loss | identity 0.1553, temperature 0.1568, affine 0.1236 (SE 0.0153, limit 0.1390) |
| Acceptance | Accepted |
| Operating log loss | 0.1749 to 0.1565 |
| Operating Brier | 0.0581 to 0.0497 |
| Operating ECE (10 bins) | 0.0442 to 0.0227 |
| ROC-AUC, EER | 0.9791 and 0.0923, unchanged (the mapping is monotone) |
| **Deployed rule** | `score >= 0.4040371283027592` on the calibrated score (raw-equivalent 0.2354); score type `calibrated_probability` |

| Row (operating role) | Rule | Macro F1 | Genuine FPR | TTS recall |
|---|---|---:|---:|---:|
| A raw | 0.5 | 0.8837 | 0.0923 | 0.9077 |
| B affine | 0.5 | 0.9015 | 0.1615 | 0.9564 |
| C raw | optimum 0.2354 | 0.9129 | 0.1885 | 0.9795 |
| D affine | optimum 0.4040 | 0.9129 | 0.1885 | 0.9795 |

Rows C and D are the same partition. The macro-F1 gain over row A comes from the threshold chosen on the operating role, not from the mapping. The mapping's effect is on probability quality, and on where 0.5 falls (row B). The probability-quality gain is not distinguishable from zero at 13 speakers (paired speaker bootstrap, 1,000 resamples: log-loss gain 0.018, interval [-0.007, 0.043]).

## Verification

- **Colab reproduction.** The official run reproduced the first (local, since deleted) execution exactly: same mapping, same a and b to the last digit, same threshold, same operating metrics, despite different Python, numpy and scipy versions.
- **Tests.** 128 passed on Colab before the run; one skip, by design (the "matplotlib is installed" test).
- **Audit.** 38 checks passed (hashes, identities, role counts, speaker separation, score-equals-mean-of-chunks, saturation, join to the cohort and split manifests).
- **Replay.** 1,240 recordings (all 1,040 operating, 100 selection, 100 calibration) through the bundle, on transformers 5.22.0. Largest score difference from the saved scores 1.2e-5 against the 1e-3 tolerance, no decision flips, tier 2 (WAV bytes through `analyze`) equal to tier 1; eight recordings within 0.01 of the threshold, none flipped. The CLI with no `--threshold` equals the library result.
- **Independent recomputation** (numpy and scipy, locally): the LOSO losses, the selection, the final fit, the acceptance rule, the threshold and every deployed score and decision were recomputed from the R3-U1 exports and agree. The model components are byte-identical to `bundles/r3_u1_wavlm_head_B/`.
- **Freeze order.** The fit read only the calibration role. The operating role was opened afterwards.

Hashes (sha256) recorded in `handoff/handoff.json` of the run:

| Item | sha256 |
|---|---|
| `bundle.json` | `491310e8e563c4cca12063ea5edb3af2a2f1f7be788aa77432eef4ac274e7f82` |
| calibration component | `4baef8ca6cb64261effee0170358e50d92c3dfd3730d424c18a8a72498b93de3` |
| fit `stage_complete.json` | `58c0d512a7ce8d87543e1d30af2fb4e88ae99f30603eb9ceb47ef5beaaddf946` |
| `input_audit.json` | `25dc64c29ba66779f410ef0dbd62d1597de895682472c13ab36404a309b21b94` |
| operating `evidence.json` | `2bc677fac167fd6a123650b8170b6aacee90ba676b41a373a7e788ec6923c5b1` |
| replay report | `975c38ae3a1e62098eefc41c5e90009c8f70720d86488b26927fe80b40bfc80b` |

## Known limits and findings carried forward

These are findings of the finalised policy. They do not change it.

1. **The policy buys recall with false alarms.** Operating recall goes from 0.908 to 0.980, genuine false alarms from 9.2% to 18.8% (49 of 260). The macro-F1 rule rewards recall because the validation roles are 75% TTS. Calibrated probabilities and the threshold both assume that prevalence, so they must not be quoted as real-world rates.
2. **False alarms are speaker-specific.** Four of 13 operating speakers (3699, 6476, 4406, 5561) hold 47 of the 49. Eight have none. Duration is not the driver (rank correlation of genuine score with duration about -0.09), and no quality flag is set on any genuine recording. Speaker 6476 is also hard for AST and LR; the other three are flagged mostly by WavLM.
3. **Confident, not borderline.** Median false-alarm score 0.705; only 14 of the 49 fall between the threshold and 0.6.
4. **Listening audit, user-reported.** The owner reports having listened to genuine recordings that the model flags and that they sound like TTS. This is not a recorded audit (no list of recordings, count or protocol). The cross-model check supports it only partly: the other models score these recordings higher than other genuine speech (ROC-AUC 0.77 to 0.83) but rarely flag them (0 to 22% at raw 0.5), and only 13 of the 49 are flagged by any other model. If the labels prove wrong, any exclusion must be a documented, pre-declared rule applied to every model and role before the GR3 lock, not a change for WavLM alone.
5. **Known-generator panel.** The deployed rule is worse than raw at 0.5 there (macro F1 0.9255 against 0.9645); recall is 1.0 in both and false alarms double. The policy's whole benefit is on unseen XTTS.
6. **Duration-matched selection subset (810 recordings).** Raw at 0.5 scores macro F1 0.9444 with genuine FPR 1.5%, the deployed rule 0.9270 with 12.4%. The deployed rule's advantage on the full selection role (0.9418 against 0.9136) does not survive duration matching. At raw 0.5 the model itself is first or tied on every measure on this subset.
7. **Not independent.** The threshold is chosen on the operating role, so its figures are development results, not independent test performance. Thirteen speakers are few; differences among WavLM, AST and BiGRU on the operating role are within noise (`REPORT.md`, Section 9).

## Where things are

| What | Path | In git |
|---|---|---|
| Recipe | `configs/r3_n1_wavlm_calibration_b.json` | yes |
| Pipeline code | `src/echoproof/evaluation/{reliability,affine_calibration,r3_n1_wavlm,r3_n1_wavlm_operating,r3_n1_wavlm_replay}.py` | yes |
| Bundle writer and loader | `src/echoproof/models/r3_n1_wavlm_bundle.py` | yes |
| Scripts | `scripts/r3_n1_wavlm_calibrate.py` (audit, fit, operating, package), `scripts/r3_n1_wavlm_replay.py` | yes |
| Notebook (Colab) | `notebooks/R3_N1_WavLM_Colab.ipynb` | yes |
| Tests | `tests/test_r3_n1_wavlm_{numeric,pipeline,bundle,replay}.py` | yes |
| Shared files edited (additive) | `src/echoproof/inference.py` (`load_inference_bundle`), `src/echoproof/cli.py` (`--threshold` defaults to the bundle's rule), `app/main.py` (`ECHOPROOF_BUNDLE`, score details), `src/echoproof/app_support.py` | yes |
| Model comparison and its scripts | `REPORT.md`, `scripts/r3_model_comparison/` | yes |
| Official run (audit, fit, operating, handoff, replay) | `runs/r3/r3_n1_wavlm/20261114T083533Z-e6ff5a288068/` | no (`/runs/` is ignored; travels by Drive) |
| Calibrated bundle | `bundles/r3_n1_wavlm_head_B_calibrated/` | no (`/bundles/` is ignored; travels by Drive) |
| Comparison output | `reports/r3/model_comparison/compare.json` | no (`/reports/` is ignored) |
| Plan and execution record | `WAVLM_CALIBRATION.md` | no (listed in `.gitignore`) |

## Reproducing the run

Everything heavy runs on Colab with `notebooks/R3_N1_WavLM_Colab.ipynb`, with transformers pinned to 5.22.0. It runs the tests first, then audit, fit, operating, package, replay and a CLI check, and refuses to start if the bundle folder exists. The recipe is hash-bound, so keep its line endings unchanged when syncing (`core.autocrlf=true` changes the bytes on Windows; the contract hashes match when LF-normalised).

## Still open

Independent review of this work; the GR3 comparison with a common reference calibration; calibration of AST (no owner or work found) and a team decision on BiGRU's calibration; promotion, locking and any change of the application's default model (still the LR bundle); latency qualification (R3-N2) for LR and BiGRU; whether the false-alarm profile warrants an amendment (prevalence correction, FPR cap or abstention) before Protocol A; the outcome of the listening audit.
