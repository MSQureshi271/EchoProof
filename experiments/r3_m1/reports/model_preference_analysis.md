# R3-M1 model preference and operational tradeoffs

Date: 2026-11-08  
Status: analysis documented; existing selection retained; independent review pending.

This note records the comparison requested by Muneeb. It does not change the selection rule, checkpoint, threshold, code, or experiment outputs. R3-N1 belongs to Nouman; this note does not perform that task or record Usman's approval.

## Evidence and scope

Sources: [batch comparison](batch01_comparison.json), [selection decision](final_selection.json), [handoff](final_handoff.json), and the saved recording predictions in the original and experimental run folders. The frozen experiment contract is [batch01_resolved.json](../configs/batch01_resolved.json).

All classification results below use uncalibrated recording scores at threshold 0.5 on the same Protocol B selection role: 2,000 recordings from 25 speakers, comprising 500 genuine, 500 Melo, 500 VITS, and 500 XTTS recordings. The unseen-XTTS panel contains the 500 genuine and 500 XTTS recordings. These are selection results, not final-test results. Calibration and operating outcomes were not used for this comparison.

| Candidate | Best epoch | Full selection macro F1 | XTTS detected / 500 | Genuine false alarms / 500 | XTTS-panel macro F1 | XTTS-panel ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Original baseline | 3 | 85.98% | 463 | 155 | 80.53% | 0.884704 |
| Previous selected lower LR | 10 | 86.11% | 338 | 59 | 77.66% | 0.883184 |
| Experimental baseline | 7 | 86.76% | 351 | 60 | 78.93% | 0.882236 |
| Experimental selected lower LR | 16 | 86.91% | 332 | 42 | 78.66% | 0.904064 |

## Previous versus experimental selected checkpoint

The experimental lower-LR checkpoint improves full selection macro F1 by 0.80 percentage points. Genuine false-positive rate falls from 11.8% to 8.4%, while unseen-XTTS recall falls from 67.6% to 66.4%. It therefore prevents 17 genuine false alarms but adds six XTTS misses and one known-family miss on this selection set.

Paired comparison of the same recordings gives a more precise account:

| Component | Both correct | Only previous correct | Only experimental correct | Both wrong |
|---|---:|---:|---:|---:|
| Genuine | 440 | 1 | 18 | 41 |
| Melo | 500 | 0 | 0 | 0 |
| VITS | 498 | 1 | 0 | 1 |
| XTTS | 311 | 27 | 21 | 141 |

The new checkpoint corrects 18 previous genuine false alarms but introduces one. It detects 21 XTTS recordings previously missed but loses 27 previous detections. Its changed decisions are not simply a uniform shift toward predicting genuine speech.

Its XTTS-panel ROC-AUC improves from 0.883184 to 0.904064, and XTTS-panel EER falls from 16.4% to 14.6%. This supports better aggregate separation across thresholds, but does not guarantee better recall at every fixed false-positive rate. XTTS-panel log loss worsens from 0.598164 to 0.660917; raw confidence estimates have not improved on every measure.

## Operational interpretation

If a flag can lead to an accusation, rejection, or blocked access for a genuine speaker, reducing false alarms matters substantially. The experimental selected checkpoint is attractive under that priority, although its observed 8.4% false-positive rate still limits the case for automatic adverse decisions.

If a flag only adds an item to a human review queue, missed synthetic speech may be more costly than additional review. A more sensitive operating point can then be preferable, subject to reviewer capacity. The original baseline detects 92.6% of XTTS at 0.5, but flags 31% of genuine recordings. That workload may be impractical, especially when synthetic speech is uncommon.

For the equally sized genuine/XTTS selection panel only, let C_FP be the cost of one genuine false alarm and C_FN the cost of one missed XTTS recording. The experimental selected checkpoint changes error cost relative to the previous selected checkpoint by:

`experimental_cost - previous_cost = -17 * C_FP + 6 * C_FN`

Thus, on this panel at threshold 0.5, the previous checkpoint has lower error cost when C_FN / C_FP exceeds 17/6, approximately 2.83. This illustration excludes known-family speech and assumes constant per-error costs. Different deployment prevalence, generators, or consequences change the comparison. Selection-set precision should not be transferred directly to a population with a different synthetic-speech prevalence.

The experimental baseline also merits explicit mention: compared with the previous selected lower-LR model, it detects 13 more XTTS recordings for one additional genuine false alarm. This is an attractive observed tradeoff for detection-focused use at 0.5, but it does not win the predefined full-selection macro-F1 rule. Favoring it after inspecting these results would require a disclosed change of objective, not an undocumented replacement of the selected model.

## Model scores, thresholds, and calibration

Improving a model can improve both detection and false alarms; deterioration of one is not an unavoidable requirement when comparing different models. For a fixed model, lowering its threshold generally increases both detections and false alarms. A fair operational comparison can therefore examine recall at a common false-positive rate, or false-positive rate at a common recall requirement, rather than assuming the same numerical threshold represents the same operating behavior for different models.

Threshold changes affect classifications without changing the underlying ranking. See the [scikit-learn threshold documentation](https://scikit-learn.org/stable/modules/classification_threshold.html). A higher overall AUC does not establish superiority in every region of the ROC curve. Temperature calibration alone does not guarantee recovery of missed detections or better ranking; choosing an operating threshold is a separate decision.

No new operating threshold is proposed or fitted here. Under the existing plan, Nouman performs R3-N1 using the designated calibration and operating roles, including the prescribed macro-F1 threshold objective. A deployment-specific cost objective would require an explicitly documented protocol decision rather than silently changing that objective.

## Recommendation and limits

Retain `b01_lower_lr_full20`, best epoch 16, as the selected R3-M1 checkpoint under the agreed rule: full selection recording-level macro F1 at threshold 0.5. Its defensible description is:

> It improves overall selection macro F1, reduces genuine false alarms, and improves unseen-XTTS ranking, while slightly reducing unseen-XTTS recall at threshold 0.5.

Preserve the alternative checkpoints as comparison evidence. This selection is not a claim that the experimental checkpoint is preferable for every application. R3-M1 checkpoint selection and the later GR3 cross-model production selection have different prescribed criteria; GR3 retains its unseen-XTTS selection rule from REVISED_PLAN.md.

The longer runs reproduced the original validation metrics through the former stopping points and then reached better checkpoints. This supports the conclusion that patience-3 stopping missed later improvements in these runs. The preceding audit found no confirmed implementation defect. Neither result establishes a global optimum.

These are result-informed, single-seed comparisons on one unseen synthesis family and a limited speaker cohort. Small metric differences do not establish statistical superiority, and XTTS performance does not establish generalization to every unseen generator or real-world recording condition. Usman's actual review disposition remains pending; this analysis does not substitute for it.
