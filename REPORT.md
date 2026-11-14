# EchoProof Protocol B: model comparison (WavLM against LR, BiGRU and AST)

Generated 14 November 2026 from saved recording-level predictions. No model was run, no test partition was read, and no audio was used.

## 1. What this report is

It compares the four Protocol B candidates side by side on every metric the saved predictions support:

| Model | Selected configuration | Prediction source |
|---|---|---|
| WavLM-head-B | `mean_std` pooling, epoch 14 | `reports/r3/r3_u1_B/` |
| AST partial fine-tune | baseline recipe, epoch 1 | `runs/r3/r3_u2_20261112T134049Z/` |
| BiGRU | `b01_lower_lr_full20`, epoch 16 | `experiments/r3_m1/runs/batch01/selected/` |
| Logistic regression | `selected_lr_v1` | `runs/r3/r3_n1_lr/selected_lr_v1/export/` |

Validation roles (Protocol B, split hash `0854de6b…`, 75% TTS in every role):

| Role | Recordings | Speakers | Used for |
|---|---:|---:|---|
| selection | 2,000 | 25 | model and epoch selection, contract ranking |
| calibration | 960 | 12 | fitting calibration mappings |
| operating | 1,040 | 13 | evaluating calibration and choosing thresholds |

Every role contains unseen XTTS recordings beside MeloTTS, VITS and genuine speech. XTTS is the hardest generator for every model (LR also misses about 9% of VITS).

**How the comparison is kept fair.** The four models were calibrated by different people with different methods (LR by temperature scaling, BiGRU by Platt and isotonic scaling, AST not at all, WavLM by an affine mapping). Comparing those results directly would compare the calibration choices as much as the models. So the headline comparisons use **raw scores** and **one common reference calibration fitted the same way for every model on the calibration role only** (Section 7). The owners' own calibrations are listed separately.

**What it is not.** It is not the GR3 lock and it does not select a production model. It uses no test data and no Protocol A model.

## 2. Summary

1. **Under the contract's ranking rule WavLM ranks first** on the selection role (macro F1 0.9136, EER 0.0467, FPR 1.2%), ahead of AST, BiGRU and LR. The margin over AST is visible in the selection-role point estimates (no interval was computed there). It does not carry over to the operating role.
2. **On the operating role WavLM, AST and BiGRU cannot be separated on ROC-AUC, EER, macro F1 or false-alarm rate.** Every paired speaker-bootstrap interval for WavLM minus AST contains zero, and so does every one for BiGRU except recall, where WavLM is higher (Section 9). With 13 speakers the data cannot do more. Only LR is clearly worse.
3. **LR is clearly the weakest.** Its EER is 20.4% and it detects 17.7% of unseen XTTS at 0.5. It is not competitive.
4. **WavLM has the best probability quality** under the common calibration: lowest log loss, Brier score and ECE (Section 7).
5. **WavLM is the most sensitive and least specific model once thresholds are chosen on the operating role.** At the contract threshold it detects 93.8% of XTTS but flags 18.8% of genuine recordings, against AST's 83.5% and 7.3%. AST's operating macro F1 (0.9228) is slightly higher than WavLM's (0.9129) and it does so with far fewer false alarms.
6. **WavLM does not depend on the duration shortcut.** On the duration-matched selection subset it is first or tied on every measure at raw 0.5 (Section 11).
7. **Deployment qualification differs.** On the development CPU, AST needs an estimated 13.7 s for a ten-second recording and WavLM 4.1 s, against the contract's five-second target. LR and BiGRU have not been timed (Section 4).
8. **The false alarms are speaker-specific for every model**, and the problem speakers differ by model. Speaker 6476 is hard for WavLM, AST and LR. Speaker 6454 is a BiGRU-only problem (Section 10).

## 3. Calibration status of each model

| Model | Calibration so far | Status |
|---|---|---|
| WavLM | Affine on log-odds (a = 1.108, b = 0.917), accepted by the pre-declared rule; deployed threshold 0.4040 on the calibrated score | Complete, verified, reproduced on Colab. Sits outside the GR3 lock. |
| LR | Temperature scaling, reported by its owner as the team decision | Complete. It leaves LR badly miscalibrated (Section 7). |
| BiGRU | Platt and smoothed isotonic, investigated by its owner | The owner's report says this is not a final team decision. |
| AST | None | No owner or work found. |

The mappings the common reference fitted on each model's own calibration role:

| Model | Temperature T | Affine a | Affine b |
|---|---:|---:|---:|
| WavLM | 1.044 | 1.108 | 0.917 |
| AST | 0.838 | 1.237 | 0.727 |
| BiGRU | 1.691 | 0.656 | 0.916 |
| LR | 1.997 | 0.559 | 1.042 |

A temperature above 1 means a model is overconfident, and below 1 underconfident. LR and BiGRU are overconfident, AST is slightly underconfident, and WavLM is almost exactly right (T = 1.044). The affine offset b is positive for all four (0.73 to 1.04), the same order as ln 3 = 1.10, the log-odds of a 75% TTS prior. This is consistent with the validation prevalence driving the offset, although I did not isolate that. **Either way, the fitted offset describes the validation roles, not real-world prevalence.**

## 4. Ranking under the contract rule and deployment qualification

The contract ranks deployable candidates by unseen-generator selection-role macro F1 at 0.5, then lower EER, then lower genuine FPR at 0.5, then lower warm end-to-end p95 latency.

| Rank | Model | Selection macro F1 @0.5 | EER (tie-break 2) | Genuine FPR @0.5 (tie-break 3) |
|---|---:|---:|---:|---:|
| 1 | WavLM | 0.9136 | 0.0467 | 1.2% |
| 2 | AST | 0.8887 | 0.0860 | 7.4% |
| 3 | BiGRU | 0.8691 | 0.0920 | 8.4% |
| 4 | LR | 0.6923 | 0.2533 | 16.4% |

The first criterion separates all four, so the tie-breaks do not come into play.

**Latency.** Only AST and WavLM have been measured, on one development machine (Windows, Intel CPU, fp32, 4 threads, synthetic noise, batch 8, excluding audio decoding). The contract requires a declared benchmark, warm-up and repetitions. These numbers are a preliminary estimate, not the R3-N2 qualification.

| Model | Per 4 s chunk, batch 1 | Estimated 10 s recording | Against the 5 s target |
|---|---:|---:|---|
| WavLM | 1.03 s | 4.1 s | Under (estimate) |
| AST | 1.92 s | 13.7 s | Over on this CPU |
| BiGRU | not measured | not measured | open |
| LR | not measured | not measured | open |

Source: `reports/r3/r3_u2_latency_cpu_fp32_20261112T195158Z.json`. AST would need a GPU or a different deployment target to meet the target.

## 5. Discrimination at raw score 0.5, all three roles

Each table uses raw model scores with the plain 0.5 threshold, so no calibration or threshold choice is involved.

### Selection role (25 speakers, 2,000 recordings)
| Model | Macro F1 | ROC-AUC | EER | Avg precision | Genuine FPR | TTS recall | TTS precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WavLM | 0.9136 | 0.9901 | 0.0467 | 0.9969 | 0.0120 | 0.9107 | 0.9956 | 0.9300 |
| AST | 0.8887 | 0.9681 | 0.0860 | 0.9890 | 0.0740 | 0.9060 | 0.9735 | 0.9110 |
| BiGRU | 0.8691 | 0.9520 | 0.0920 | 0.9716 | 0.0840 | 0.8867 | 0.9694 | 0.8940 |
| LR | 0.6923 | 0.8203 | 0.2533 | 0.9404 | 0.1640 | 0.6820 | 0.9258 | 0.7205 |

### Calibration role (12 speakers, 960 recordings)
| Model | Macro F1 | ROC-AUC | EER | Avg precision | Genuine FPR | TTS recall | TTS precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WavLM | 0.9224 | 0.9908 | 0.0444 | 0.9972 | 0.0083 | 0.9194 | 0.9970 | 0.9375 |
| AST | 0.9076 | 0.9823 | 0.0653 | 0.9945 | 0.0417 | 0.9153 | 0.9851 | 0.9260 |
| BiGRU | 0.8777 | 0.9745 | 0.0750 | 0.9914 | 0.0542 | 0.8847 | 0.9800 | 0.9000 |
| LR | 0.6910 | 0.8167 | 0.2639 | 0.9405 | 0.1708 | 0.6833 | 0.9231 | 0.7198 |

### Operating role (13 speakers, 1,040 recordings)
| Model | Macro F1 | ROC-AUC | EER | Avg precision | Genuine FPR | TTS recall | TTS precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WavLM | 0.8837 | 0.9791 | 0.0923 | 0.9930 | 0.0923 | 0.9077 | 0.9672 | 0.9077 |
| AST | 0.8937 | 0.9756 | 0.0705 | 0.9920 | 0.0538 | 0.9038 | 0.9805 | 0.9144 |
| BiGRU | 0.8597 | 0.9670 | 0.0936 | 0.9878 | 0.0731 | 0.8705 | 0.9728 | 0.8846 |
| LR | 0.7390 | 0.8690 | 0.2038 | 0.9599 | 0.0423 | 0.6936 | 0.9801 | 0.7596 |

**Reading it.**
- WavLM has the highest ROC-AUC in all three roles (0.9901, 0.9908, 0.9791), with AST close behind.
- On the operating role WavLM's EER (0.0923) is worse than AST's (0.0705) and about the same as BiGRU's (0.0936). On the selection role, with 25 speakers, it is clearly the best (0.0467). The operating ordering rests on 13 speakers and is noisier.
- WavLM's genuine FPR at 0.5 rises from 1.2% (selection) to 9.2% (operating). That change comes from the operating speakers, and Section 10 shows which ones.

## 6. Recall by generator (raw 0.5)

| Model | Role | MeloTTS recall | VITS recall | XTTS recall | Genuine FPR |
|---|---:|---:|---:|---:|---:|
| WavLM | selection | 100.0% | 100.0% | 73.2% | 1.2% |
| AST | selection | 100.0% | 100.0% | 71.8% | 7.4% |
| BiGRU | selection | 100.0% | 99.6% | 66.4% | 8.4% |
| LR | selection | 99.2% | 91.2% | 14.2% | 16.4% |
| WavLM | calibration | 100.0% | 100.0% | 75.8% | 0.8% |
| AST | calibration | 100.0% | 99.6% | 75.0% | 4.2% |
| BiGRU | calibration | 100.0% | 99.6% | 65.8% | 5.4% |
| LR | calibration | 99.2% | 90.8% | 15.0% | 17.1% |
| WavLM | operating | 100.0% | 100.0% | 72.3% | 9.2% |
| AST | operating | 100.0% | 100.0% | 71.2% | 5.4% |
| BiGRU | operating | 100.0% | 99.2% | 61.9% | 7.3% |
| LR | operating | 100.0% | 90.4% | 17.7% | 4.2% |

- **Every model catches essentially all MeloTTS and VITS** (LR is the exception on VITS, about 90 to 91%).
- **Unseen XTTS is the whole difficulty.** At 0.5 WavLM, AST and BiGRU detect between 62% and 72% of it on the operating role. LR detects 18%.
- WavLM and AST are within one percentage point of each other on XTTS at 0.5. Whatever separates them lies elsewhere.

## 7. Probability quality on the operating role

### 7.1 One common reference calibration, fitted on each model's own calibration role

Temperature scaling and the affine log-odds mapping, both fitted on the calibration role only and applied unchanged to the operating role. Lower is better for all three measures.

| Model | Log loss raw | Log loss temp. | Log loss affine | Brier raw | Brier temp. | Brier affine | ECE raw | ECE temp. | ECE affine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WavLM | 0.1749 | 0.1737 | 0.1565 | 0.0581 | 0.0578 | 0.0497 | 0.0442 | 0.0443 | 0.0227 |
| AST | 0.2059 | 0.2104 | 0.1816 | 0.0633 | 0.0633 | 0.0537 | 0.0623 | 0.0605 | 0.0314 |
| BiGRU | 0.3083 | 0.2559 | 0.2067 | 0.0840 | 0.0804 | 0.0599 | 0.0988 | 0.0912 | 0.0474 |
| LR | 0.5925 | 0.5319 | 0.4048 | 0.1771 | 0.1737 | 0.1328 | 0.2035 | 0.2115 | 0.0890 |

- **WavLM is best on every measure, both raw and after calibration.** Its raw scores were already the best calibrated, which matches its temperature of 1.044.
- **The affine mapping helps all four models** by more than temperature scaling does. It helps LR and BiGRU most (log loss drops 0.188 and 0.102).
- **Temperature scaling does not fix LR.** Its ECE stays near 0.21, so the mapping family LR was given cannot correct its shape.
- **After affine calibration the models rank WavLM, then AST, BiGRU and LR** on all three measures.

### 7.2 The owners' own calibrations (operating role)

| Calibration as run by the owner | Log loss | Brier | ECE |
|---|---:|---:|---:|
| LR, temperature scaling | 0.5319 | 0.1737 | 0.2115 |
| BiGRU, Platt | 0.1996 | 0.0571 | 0.0320 |
| BiGRU, smoothed isotonic | 0.2091 | 0.0572 | 0.0340 |

These use different methods and are shown for context only. BiGRU's Platt mapping reaches log loss 0.1996, better than the common affine mapping on BiGRU (0.2067) but still behind AST (0.1816) and WavLM (0.1565) under the common mapping. LR's temperature scaling leaves it far behind.

## 8. Operating thresholds (contract rule: maximise macro F1)

The contract's threshold rule is monotone-invariant, so the decisions below are the same whether the threshold is applied to raw or calibrated scores. All three tables score the same operating-role recordings. They differ in where the threshold was chosen.

### 8.1 Threshold chosen on the operating role (the contract's rule, optimistic)
| Model | Raw threshold | Macro F1 | Genuine FPR | False alarms | TTS recall | Missed TTS | XTTS recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| WavLM | 0.2354 | 0.9129 | 18.8% | 49 | 97.9% | 16 | 93.8% |
| AST | 0.4010 | 0.9228 | 7.3% | 19 | 94.5% | 43 | 83.5% |
| BiGRU | 0.1998 | 0.9077 | 13.8% | 36 | 95.4% | 36 | 86.5% |
| LR | 0.2586 | 0.7625 | 22.7% | 59 | 81.4% | 145 | 46.2% |

This is the number each model would report under the contract. Because the threshold was tuned on the same recordings it is scored on, it slightly flatters every model.

### 8.2 Threshold chosen on the calibration role (not seen by the operating role)
| Model | Raw threshold | Macro F1 | Genuine FPR | False alarms | TTS recall | Missed TTS | XTTS recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| WavLM | 0.2827 | 0.9028 | 17.7% | 46 | 96.4% | 28 | 89.2% |
| AST | 0.4176 | 0.9148 | 7.3% | 19 | 93.6% | 50 | 80.8% |
| BiGRU | 0.2749 | 0.8949 | 12.3% | 32 | 93.3% | 52 | 80.4% |
| LR | 0.4024 | 0.7508 | 10.4% | 27 | 73.8% | 204 | 27.7% |

### 8.3 Threshold chosen on the selection role (25 speakers)
| Model | Raw threshold | Macro F1 | Genuine FPR | False alarms | TTS recall | Missed TTS | XTTS recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| WavLM | 0.2331 | 0.9114 | 19.2% | 50 | 97.9% | 16 | 93.8% |
| AST | 0.4185 | 0.9137 | 7.3% | 19 | 93.5% | 51 | 80.4% |
| BiGRU | 0.1609 | 0.9015 | 16.2% | 42 | 95.6% | 34 | 87.3% |
| LR | 0.3938 | 0.7492 | 11.2% | 29 | 74.0% | 203 | 28.1% |

**Reading it.**
- **AST has the best balance in all three versions.** Its false-alarm rate stays at 7.3% and its macro F1 stays between 0.914 and 0.923.
- **WavLM trades false alarms for recall.** It gives the highest recall (96 to 98%) and the highest XTTS recall (89 to 94%), at 17.7 to 19.2% false alarms. Its macro F1 is comparable to AST's (0.903 to 0.913 against 0.914 to 0.923).
- **BiGRU sits between them** (12 to 16% false alarms, F1 0.895 to 0.908).
- **LR is far behind** whichever version is used.
- **The macro-F1 rule rewards recall because validation is 75% TTS.** If real TTS prevalence is lower, a high-recall, high-false-alarm profile would cost more than these numbers show, which would favour AST's profile.

## 9. Uncertainty: speaker bootstrap on the operating role

Speakers are resampled with replacement (13 operating speakers, 1,000 resamples, seed 42). All figures use raw scores at 0.5.

| Model | ROC-AUC [95% CI] | EER [95% CI] | Macro F1 @0.5 [95% CI] | Genuine FPR @0.5 [95% CI] | TTS recall @0.5 [95% CI] |
|---|---:|---:|---:|---:|---:|
| WavLM | 0.979 [0.961, 0.993] | 0.092 [0.042, 0.131] | 0.884 [0.847, 0.913] | 0.092 [0.023, 0.165] | 0.908 [0.886, 0.928] |
| AST | 0.976 [0.952, 0.991] | 0.071 [0.038, 0.123] | 0.894 [0.859, 0.919] | 0.054 [0.000, 0.146] | 0.904 [0.891, 0.919] |
| BiGRU | 0.967 [0.921, 0.994] | 0.094 [0.038, 0.196] | 0.860 [0.803, 0.894] | 0.073 [0.000, 0.219] | 0.871 [0.855, 0.887] |
| LR | 0.869 [0.831, 0.901] | 0.204 [0.162, 0.242] | 0.739 [0.717, 0.759] | 0.042 [0.008, 0.089] | 0.694 [0.677, 0.714] |

Paired differences, WavLM minus each model, on the same resamples (an asterisk marks an interval that excludes zero):

| WavLM minus | ROC-AUC | EER | Macro F1 @0.5 | Genuine FPR @0.5 | TTS recall @0.5 |
|---|---:|---:|---:|---:|---:|
| AST | +0.003 [-0.012, +0.021] | +0.022 [-0.022, +0.054] | -0.010 [-0.032, +0.010] | +0.038 [-0.031, +0.108] | +0.004 [-0.014, +0.022] |
| BiGRU | +0.012 [-0.023, +0.063] | -0.001 [-0.133, +0.065] | +0.024 [-0.031, +0.100] | +0.019 [-0.177, +0.154] | +0.037 [+0.019, +0.054] * |
| LR | +0.110 [+0.084, +0.136] * | -0.112 [-0.141, -0.087] * | +0.145 [+0.120, +0.169] * | +0.050 [+0.008, +0.100] * | +0.214 [+0.194, +0.237] * |

- **Only LR is separated from WavLM.** For AST and BiGRU every interval for the discrimination measures contains zero.
- **WavLM has higher recall than BiGRU at 0.5** (the interval excludes zero), but not significantly higher than AST's.
- **LR's low false-alarm rate at 0.5 is not a strength.** It reflects a model that rarely flags anything (recall 69%).
- **The intervals on false-alarm rate are very wide for every model** (up to 0 to 22%). Thirteen speakers cannot support a firm claim about the false-alarm rate of any of them.

## 10. False alarms by speaker (threshold chosen on the operating role)

| Model | False alarms | Speakers with any | Speakers with at least 25% of their recordings flagged | Share held by the top 3 speakers | Worst speakers (flagged of 20) |
|---|---:|---:|---:|---:|---:|
| WavLM | 49 | 5 of 13 | 4 of 13 | 78% | 3699 (14), 6476 (14), 4406 (10) |
| AST | 19 | 4 of 13 | 1 of 13 | 95% | 6476 (15), 3699 (2), 4406 (1) |
| BiGRU | 36 | 5 of 13 | 2 of 13 | 83% | 6454 (20), 4406 (6), 6476 (4) |
| LR | 59 | 10 of 13 | 5 of 13 | 56% | 6476 (14), 5561 (10), 4406 (9) |

- **WavLM:** four of 13 speakers (3699, 6476, 4406, 5561) hold 47 of 49 false alarms. Eight speakers have none.
- **AST:** one speaker, 6476, holds 15 of its 19.
- **BiGRU:** it flags all 20 genuine recordings of speaker 6454, which is 20 of its 36 false alarms.
- **LR:** false alarms are spread over 10 speakers.
- **Speaker 6476 is hard for WavLM, AST and LR.** Speaker 6454 is a BiGRU-only problem. Speakers 3699, 4406 and 5561 are flagged mostly by WavLM (BiGRU flags a few of 4406 and 3699).
- **A listening check of the WavLM false alarms is under way** (see Section 13). It matters for how this table is read.

## 11. Robustness to the duration shortcut (selection role, duration-matched subset)

The R2-S3 probe found a duration shortcut in the development data. This subset (810 recordings) is matched on duration, so duration cannot help. Raw scores at 0.5:

| Model | Recordings | Macro F1 | ROC-AUC | EER | Genuine FPR | TTS recall | XTTS recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| WavLM | 810 | 0.9444 | 0.9876 | 0.0568 | 1.5% | 90.4% | 72.3% |
| AST | 810 | 0.9024 | 0.9588 | 0.1012 | 8.1% | 88.6% | 67.4% |
| BiGRU | 810 | 0.9086 | 0.9512 | 0.0914 | 8.6% | 90.4% | 72.3% |
| LR | 810 | 0.7438 | 0.8139 | 0.2617 | 16.8% | 65.9% | 10.6% |

- **WavLM is first or tied on every measure** (recall and XTTS recall tie with BiGRU). Its ROC-AUC is almost unchanged from the full role (0.9876 against 0.9901).
- AST and BiGRU lose a little ranking quality (AST AUC falls from 0.9681 to 0.9588), although macro F1 at 0.5 rises for both on this subset.
- A model leaning on duration would be expected to lose ranking quality when duration is matched. WavLM's AUC barely moves, so at raw 0.5 there is no sign that it leans on duration. The calibrated WavLM policy is a different matter and was shown separately: the deployed rule does worse on this subset than raw 0.5.

## 12. Agreement between models and a descriptive ensemble

### 12.1 Agreement on the operating role (raw 0.5)

| Pair | Same decision at raw 0.5 | Rank correlation of scores |
|---|---:|---:|
| LR and BiGRU | 80.6% | 0.802 |
| LR and AST | 83.0% | 0.848 |
| LR and WavLM | 81.5% | 0.802 |
| BiGRU and AST | 86.6% | 0.838 |
| BiGRU and WavLM | 85.8% | 0.802 |
| AST and WavLM | 94.1% | 0.831 |

- **AST and WavLM agree most** (94% of decisions). They are the two models whose decisions are most alike.
- Rank correlations of 0.80 to 0.85 mean the models share a lot of information but are not copies of each other.

### 12.2 Unweighted mean of raw scores (descriptive only)

| Combination (unweighted mean of raw scores) | ROC-AUC | EER | Macro F1 @0.5 | Genuine FPR | TTS recall |
|---|---:|---:|---:|---:|---:|
| mean of all four | 0.9810 | 0.0628 | 0.8923 | 2.3% | 89.0% |
| mean of WavLM and AST | 0.9821 | 0.0692 | 0.9044 | 3.8% | 91.0% |
| mean of LR, BiGRU, AST | 0.9745 | 0.0795 | 0.8667 | 3.1% | 86.2% |

The plan defers ensembles (§9 and §15). These rows are not a candidate, only an indication of what combining scores might offer.

- **Averaging gives small gains in ranking, not a step change.** WavLM and AST together reach ROC-AUC 0.9821, against 0.9791 for WavLM alone, and false alarms fall to 3.8% at 0.5.
- Averaging uncalibrated scores from models with different scales is crude. A proper combination would calibrate each score first.

## 13. Interpretation and limits

**Where WavLM stands.**
- **Strengths:** best ranking on the selection role, best probability quality, robust to the duration shortcut, and the lowest estimated CPU latency of the two measured transformer models.
- **Weakness:** at the contract threshold it flags 18.8% of genuine operating recordings, mostly from four speakers. AST achieves a similar macro F1 with 7.3%.
- **What would change the picture:** if the audited genuine recordings that WavLM flags are in fact synthetic or processed (Subhan reports hearing TTS-like quality in them; this is user-reported and not yet a recorded audit), part of that false-alarm rate is a labelling problem. The cross-model check supports this only partly. The other models score those recordings higher than other genuine speech (ROC-AUC 0.77 to 0.83) but rarely flag them (0 to 22%), and only speaker 6476 is flagged by several models. This is unresolved.

**What this report cannot show.**
- **It is not an independent confirmation of the GR3 ranking.** The ranking under the contract is shown, but the lock, the deployment qualification and the production decision are separate GR3 steps that have not happened.
- **Thirteen operating speakers are few.** Differences of a few points in macro F1 or EER between WavLM, AST and BiGRU are within noise. Speaker-level effects (Section 10) are large relative to the model differences.
- **All figures are development results.** Thresholds chosen on the operating role are optimistic. No test panel, corruption condition or Protocol A model was touched.
- **The known-generator panel is not compared.** The earlier WavLM result there (the deployed rule scoring macro F1 0.9255 against 0.9645 for raw at 0.5) was computed for WavLM only.
- **Latency is partial** (AST and WavLM only, one machine, synthetic input).
- **AST has no calibration owner,** so its probabilities here come from the common reference mapping, not a team-approved one.
- **Validation prevalence is 75% TTS.** Calibrated probabilities and the macro-F1 thresholds both assume it. They should not be quoted as real-world rates.

**Decisions this report points to, for the team.**
1. Whether the contract's macro-F1 threshold is acceptable given WavLM's false-alarm profile, or whether an amendment (prevalence correction, FPR cap or abstention) is wanted before Protocol A.
2. Who calibrates AST, and with which method, so GR3 compares four calibrated candidates.
3. The outcome of the listening audit of the flagged genuine recordings, recorded as user-reported evidence.
4. Latency measurement for LR and BiGRU, and a GPU or alternative deployment target for AST if it is to qualify.

## 14. Reproduction

Inputs (first 16 hex digits of each file's sha256):

- `LR/selection`: sha256 `7b3e2a587ccca9b6…`
- `LR/calibration`: sha256 `770eb72938178dfb…`
- `LR/operating`: sha256 `a41ad477fca4121a…`
- `BiGRU/selection`: sha256 `70e8a8063066a787…`
- `BiGRU/calibration`: sha256 `4de93f8d1f78d4de…`
- `BiGRU/operating`: sha256 `78fdc20ded6ee0f3…`
- `AST/selection`: sha256 `4a5d0690f393becd…`
- `AST/calibration`: sha256 `24cfe378ed01ae29…`
- `AST/operating`: sha256 `651d6265c4b1dfb9…`
- `WavLM/selection`: sha256 `def069b46eced013…`
- `WavLM/calibration`: sha256 `bcd449b4628cdd76…`
- `WavLM/operating`: sha256 `d957a8f2a39e1f3c…`

The calibration run `runs/r3/r3_n1_wavlm/20261114T083533Z-e6ff5a288068` provides the WavLM raw scores for the cross-check; they equal the R3-U1 export exactly (checked: True).

Method notes:
- Metrics use the project's `compute_metrics` and `select_threshold` (contract tie-breaks) and the affine fit and ECE code from the WavLM calibration work. ECE uses 10 equal-width bins.
- The common temperature is fitted by bounded 1-D minimisation of log loss. The affine mapping uses the same settings as the WavLM calibration (regularisation 0.01).
- Bootstrap: 1,000 speaker resamples, seed 42, the same resample for every model so paired differences are paired.
- Run from the project root (numpy, scipy, pyarrow only). The scripts are in `scripts/r3_model_comparison/`: `compare_models.py <out.json>` computes every figure, `make_report.py <out.json> REPORT.md` writes this report from that file, and `wavlm_false_alarm_analysis.py` and `wavlm_false_alarm_cross_model.py` produce the speaker and cross-model false-alarm figures quoted in Sections 10 and 13. The saved computation output is `reports/r3/model_comparison/compare.json` (git-ignored; it travels by Drive).
