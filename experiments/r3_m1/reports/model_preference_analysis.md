# R3-M1 model preference and operational tradeoffs

Date: 2026-11-08  
Status: batch01 and batch02 results reviewed; hidden-size-128 selection retained; independent review pending. Batch02 remains an approved experimental-only post-lock diagnostic.

This note records the comparison requested by Muneeb. It does not change the selection rule, checkpoint, threshold, code, or experiment outputs. R3-N1 belongs to Nouman; this note does not perform that task or record Usman's approval.

## Evidence and scope

Sources: [batch comparison](http://batch01_comparison.json), [selection decision](http://final_selection.json), [handoff](http://final_handoff.json), and the saved recording predictions in the original and experimental run folders. The frozen experiment contract is [batch01\_resolved.json](../configs/batch01_resolved.json).

The hidden-size comparison additionally uses [batch02 comparison](http://batch02/comparison.json), [experimental preference](http://batch02/experimental_preference.json), [Colab preflight](http://batch02/preflight.json), and the verified [batch02 resolved contract](../configs/batch02_resolved.json). Its scope is defined in the [batch02 plan](http://batch02/plan.md) and [Muneeb's specific exception approval](../configs/batch02_authorization.json). This diagnostic does not reopen or replace locked research results.

All classification results below use uncalibrated recording scores at threshold 0.5 on the same Protocol B selection role: 2,000 recordings from 25 speakers, comprising 500 genuine, 500 Melo, 500 VITS, and 500 XTTS recordings. The unseen-XTTS panel contains the 500 genuine and 500 XTTS recordings. These are selection results, not final-test results. Calibration and operating outcomes were not used for this comparison.

| Candidate | Best epoch | Full selection macro F1 | XTTS detected / 500 | Genuine false alarms / 500 | XTTS-panel macro F1 | XTTS-panel ROC-AUC |
| :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| Original baseline | 3 | 85.98% | 463 | 155 | 80.53% | 0.884704 |
| Previous selected lower LR | 10 | 86.11% | 338 | 59 | 77.66% | 0.883184 |
| Batch01 baseline, hidden 128 | 7 | 86.76% | 351 | 60 | 78.93% | 0.882236 |
| Batch01 selected lower LR, hidden 128 | 16 | 86.91% | 332 | 42 | 78.66% | 0.904064 |
| Batch02 lower LR, hidden 64 | 10 | 83.84% | 302 | 63 | 73.42% | 0.841564 |

Previous versus experimental selected checkpoint

The experimental lower-LR checkpoint improves full selection macro F1 by 0.80 percentage points. Genuine false-positive rate falls from 11.8% to 8.4%, while unseen-XTTS recall falls from 67.6% to 66.4%. It therefore prevents 17 genuine false alarms but adds six XTTS misses and one known-family miss on this selection set.

Paired comparison of the same recordings gives a more precise account:

| Component | Both correct | Only previous correct | Only experimental correct | Both wrong |
| :---- | ----: | ----: | ----: | ----: |
| Genuine | 440 | 1 | 18 | 41 |
| Melo | 500 | 0 | 0 | 0 |
| VITS | 498 | 1 | 0 | 1 |
| XTTS | 311 | 27 | 21 | 141 |

The new checkpoint corrects 18 previous genuine false alarms but introduces one. It detects 21 XTTS recordings previously missed but loses 27 previous detections. Its changed decisions are not simply a uniform shift toward predicting genuine speech.

Its XTTS-panel ROC-AUC improves from 0.883184 to 0.904064, and XTTS-panel EER falls from 16.4% to 14.6%. This supports better aggregate separation across thresholds, but does not guarantee better recall at every fixed false-positive rate. XTTS-panel log loss worsens from 0.598164 to 0.660917; raw confidence estimates have not improved on every measure.

## Hidden-size-64 experiment: completed result

Batch02 tested a fresh two-layer BiGRU with hidden size 64 per direction against the preserved hidden-size-128 candidates. The initial LR remained 0.0003, seed 42, dropout 0.3, and the same training configuration, frozen input definitions and selection rule were retained. Both matched runs allowed 20 epochs, with patience 21 disabling early stopping within that limit. This changed width, not layer count. The scheduler rule was unchanged; its actual learning-rate trajectory could differ with validation scores.

The hidden64 run completed all 20 epochs, selected epoch 10, and recorded zero skipped training steps. The single completed training attempt took 1,511.47 seconds (25.19 minutes), within its 60-minute allowance. Training/validation epoch time in the ledger was 25.062 minutes; the attempt total includes additional runner overhead.

| Metric | Selected hidden 128 | New hidden 64 |
| :---- | ----: | ----: |
| Full selection macro F1 | 86.91% | 83.84% |
| Full selection accuracy | 89.40% | 86.85% |
| Genuine false-positive rate | 8.4% | 12.6% |
| XTTS recall | 66.4% | 60.4% |
| XTTS-panel EER | 14.6% | 20.2% |
| XTTS-panel ROC-AUC | 0.904064 | 0.841564 |
| Known-family synthetic detections / 1,000 | 998 | 998 |
| Trainable parameters | 445,954 | 124,674 |
| Recorded peak training VRAM (MB) | 369.2 | 239.4 |

Hidden64 lost 3.07 percentage points of full selection macro F1, detected 30 fewer XTTS recordings, and produced 21 additional genuine false alarms. Unlike the earlier comparison between the original and batch01 selected checkpoints, this observed change worsened both XTTS recall and genuine false-positive rate at 0.5. Known-family detection remained 998/1,000, so reducing width did not close the known-versus-unseen detection gap in this run.

The smaller model has approximately 72% fewer parameters and used less recorded peak VRAM. These resource savings do not establish faster inference; deployment latency was not measured by this experiment. They also did not compensate for worse detection under the declared selection objective.

Replaying patience 3 on the hidden64 history would stop at epoch 7 and retain epoch 4 (macro F1 0.824930). Continuing to epoch 20 found the better epoch-10 checkpoint (0.838414), a gain of 1.35 percentage points within that trajectory, but it still fell below hidden128's 0.869120. Thus, the completed stopping-rule and capacity investigations answer different questions: longer training found later improvements, while this particular smaller-capacity configuration did not improve selection generalization. This does not establish that every smaller model is inferior or that more layers would help.

## Batch02 verification and evidence limits

- Colab preflight recorded 39 passing tests, zero failures/errors/skips, including hidden64 CPU and CUDA gradient, padding-invariance and reload fixtures. Its XML hash matches the preflight report.  
- All 158 artifact hashes in the comparison matched the copied files. Full-selection, known-family and XTTS metrics were independently recomputed for all five candidates from the saved Parquet predictions.  
- All 20 hidden64 epoch prediction pairs matched their histories and the same reference chunk identities. Best-epoch selection, the hypothetical patience-3 result and the five-candidate advisory ranking were verified.  
- The copied `batch02_resolved.json` SHA-256 is `18b3f97501f62d670794d8fe15e06be79e41f34531c5ffe265f370adcd90d9ba`. It matches the comparison, preflight and run provenance. All 17 frozen dependency-file hashes match locally. Architecture, training configuration, budget, selection rules and authorization agree; recorded input/environment definitions match batch01.  
- The current local B lock matches the approved content; its difference from the Colab raw hash is the verified CRLF/LF line-ending conversion. The existing batch01 selection remains unchanged.  
- Dataset input files and the protected LR-bundle files are absent locally. Their hashes are recorded in the contract, but those bytes cannot be independently rechecked in this local review. Checkpoint file hashes were checked; GPU inference was not rerun locally. The copied contract resolves the earlier missing-contract issue without retraining.

Batch02 remains an isolated post-lock diagnostic under Muneeb's explicit exception, with no automatic promotion. Its code uses only B train/selection audio and does not export calibration/operating predictions. The review did not use calibration, operating or final-test outcomes to choose a model. Team review is still required; this report does not certify absent artifacts or unrecorded external data access.

## Operational interpretation

If a flag can lead to an accusation, rejection, or blocked access for a genuine speaker, reducing false alarms matters substantially. The experimental selected checkpoint is attractive under that priority, although its observed 8.4% false-positive rate still limits the case for automatic adverse decisions.

If a flag only adds an item to a human review queue, missed synthetic speech may be more costly than additional review. A more sensitive operating point can then be preferable, subject to reviewer capacity. The original baseline detects 92.6% of XTTS at 0.5, but flags 31% of genuine recordings. That workload may be impractical, especially when synthetic speech is uncommon.

For the equally sized genuine/XTTS selection panel only, let C\_FP be the cost of one genuine false alarm and C\_FN the cost of one missed XTTS recording. The experimental selected checkpoint changes error cost relative to the previous selected checkpoint by:

`experimental_cost - previous_cost = -17 * C_FP + 6 * C_FN`

Thus, on this panel at threshold 0.5, the previous checkpoint has lower error cost when C\_FN / C\_FP exceeds 17/6, approximately 2.83. This illustration excludes known-family speech and assumes constant per-error costs. Different deployment prevalence, generators, or consequences change the comparison. Selection-set precision should not be transferred directly to a population with a different synthetic-speech prevalence.

The experimental baseline also merits explicit mention: compared with the previous selected lower-LR model, it detects 13 more XTTS recordings for one additional genuine false alarm. This is an attractive observed tradeoff for detection-focused use at 0.5, but it does not win the predefined full-selection macro-F1 rule. Favoring it after inspecting these results would require a disclosed change of objective, not an undocumented replacement of the selected model.

## Model scores, thresholds, and calibration

Improving a model can improve both detection and false alarms; deterioration of one is not an unavoidable requirement when comparing different models. For a fixed model, lowering its threshold generally increases both detections and false alarms. A fair operational comparison can therefore examine recall at a common false-positive rate, or false-positive rate at a common recall requirement, rather than assuming the same numerical threshold represents the same operating behavior for different models.

Threshold changes affect classifications without changing the underlying ranking. See the [scikit-learn threshold documentation](https://scikit-learn.org/stable/modules/classification_threshold.html). A higher overall AUC does not establish superiority in every region of the ROC curve. Temperature calibration alone does not guarantee recovery of missed detections or better ranking; choosing an operating threshold is a separate decision.

No new operating threshold is proposed or fitted here. Under the existing plan, Nouman performs R3-N1 using the designated calibration and operating roles, including the prescribed macro-F1 threshold objective. A deployment-specific cost objective would require an explicitly documented protocol decision rather than silently changing that objective.

## Recommendation and limits

Retain `b01_lower_lr_full20`, hidden size 128, best epoch 16, as the existing R3-M1 selection under the agreed rule: full selection recording-level macro F1 at threshold 0.5. Batch02's advisory ranking independently retains this candidate among the five compared checkpoints; `b02_hidden64_full20` should not replace it on the observed performance evidence. This recommendation makes no change to any lock, bundle or official result. Relative to the original selected lower-LR checkpoint, its defensible description remains:

> It improves overall selection macro F1, reduces genuine false alarms, and improves unseen-XTTS ranking, while slightly reducing unseen-XTTS recall at threshold 0.5.

Preserve the alternative checkpoints as comparison evidence. This selection is not a claim that the experimental checkpoint is preferable for every application. R3-M1 checkpoint selection and the later GR3 cross-model production selection have different prescribed criteria; GR3 retains its unseen-XTTS selection rule from REVISED\_PLAN.md.

The longer runs reproduced the original validation metrics through the former stopping points and then reached better checkpoints. This supports the conclusion that patience-3 stopping missed later improvements in these runs. The preceding audit found no confirmed implementation defect. Neither result establishes a global optimum.

These are result-informed, single-seed comparisons on one unseen synthesis family and a limited speaker cohort. Small metric differences do not establish statistical superiority, and XTTS performance does not establish generalization to every unseen generator or real-world recording condition. Usman's actual review disposition remains pending; this analysis does not substitute for it.