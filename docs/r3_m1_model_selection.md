# R3-M1 model selection record: `b01_lower_lr_full20` (epoch 16)

Status: **provisional**. Independent review (Usman) is pending, and R3-N1 calibration and thresholding have not been done. This is a development selection on Protocol B's selection role, not a final-test result and not a lock. Prepared 9 November 2026 from the experiment artifacts in `experiments/r3_m1/` and an independent recomputation of the saved predictions.

## 1. Decision

Select the BiGRU checkpoint **`b01_lower_lr_full20`, best epoch 16** (hidden size 128, learning rate 0.0003, seed 42, Protocol B) as the R3-M1 candidate to hand to R3-N1.

| Item | Value |
|---|---|
| Checkpoint | `experiments/r3_m1/runs/batch01/b01_lower_lr_full20/best.pt` |
| Checkpoint SHA-256 | `435d47b36431e7f0f6b6f1bdd2459c8a6d06074b79fcc7ff69631c0c93cc687e` |
| Config hash (ledger) | `6004aafdfe81494e71b41761474bfb2e0ae046c6218504c821f192ea36dc61a9` |
| Split hash | `0854de6b00ae059a858d60f18f95df3f1501f579ea3e2fdefbd8de076e2034d7` |
| Batch 01 contract SHA-256 | `57785aaa39437dbca1e755ad2ccb6acac41d3835de5208c5fe2ce4cbf65816db` |
| Selection macro F1 at 0.5 | 0.8691 (2,000 recordings, 25 speakers) |
| Hand-off predictions | `experiments/r3_m1/runs/batch01/selected/` (hashes in `experiments/r3_m1/reports/final_handoff.json`) |
| Fallback | `b01_baseline_full20`, epoch 7 (F1 0.8676) |

## 2. Lineage: what was selected before, and where this model came from

| Date | Event | Selected |
|---|---|---|
| 7 Nov 2026 | Original R3-M1 run `r3_m1_20261107T144037Z`: two BiGRUs (baseline LR 0.001, lower LR 0.0003), patience-3 early stopping. | **`lower_lr`, epoch 10**, F1 0.8611 (checkpoint `3154da86d9faf603c4f8bc5bf7abefdf5aa5c8fd6fc2e3aebe12324df194e640`). It beat the baseline's epoch 3 (F1 0.8598) by 0.0014. |
| 8 Nov 2026 | Muneeb's batch 01 follow-up: the same two configurations retrained from fresh initialization (seed 42) with early stopping disabled and a 20-epoch budget. Audit `audit_20261107T165239Z.json` passed 11 of 11 checks. | **`b01_lower_lr_full20`, epoch 16**, F1 0.8691. |
| 8 Nov 2026 | Muneeb's batch 02: a hidden-size-64 diagnostic (approved experimental-only exception). | Not adopted. F1 0.8384, clearly worse. |

**`b01_lower_lr_full20` is a product of Muneeb's experiment. It did not exist before.** It is a new checkpoint (different SHA from the original) trained under the original lower-LR settings with early stopping removed. Its trajectory peaked at epoch 16, three epochs after the original run's patience-3 rule would have stopped at epoch 13. The experiment's question was whether patience 3 discarded a better later checkpoint, and on this trajectory the answer was yes (+0.0080 F1). The unsuccessful experiment was batch 02 (hidden size 64), not batch 01.

Two things about the lineage limit how much to read into this:
- **The fresh baseline did not reproduce the original.** The original baseline peaked at epoch 3, the fresh one at epoch 7, and both fresh runs scored higher than both original runs. Part of the improvement is therefore plausibly run-to-run variation, not only the removed stopping rule. Seeds are fixed but GPU training is not bit-reproducible.
- **The two selection records are different decisions.** `runs/r3/r3_m1_20261107T144037Z/selection.json` still names the older `lower_lr` epoch 10 and its predictions. The current selection and predictions live under `experiments/r3_m1/`. R3-N1 must use the latter, and the team should record that the earlier record is superseded.

## 3. Evidence

Measured on the Protocol B selection role, 2,000 recordings (500 each genuine, Melo, VITS, XTTS), raw uncalibrated scores. Calibration and operating data were not used.

**Candidates (Muneeb's table, matches his saved files)**

| Candidate | Best epoch | Macro F1 at 0.5 | Genuine false alarms / 500 | XTTS detected / 500 | XTTS-panel ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Original baseline | 3 | 0.8598 | 155 | 463 | 0.8847 |
| Original lower LR (previous pick) | 10 | 0.8611 | 59 | 338 | 0.8832 |
| Batch 01 baseline | 7 | 0.8676 | 60 | 351 | 0.8822 |
| **Batch 01 lower LR (selected)** | **16** | **0.8691** | **42** | 332 | **0.9041** |
| Batch 02 hidden 64 | 10 | 0.8384 | 63 | 302 | 0.8416 |

**Independent recomputation with speaker-clustered 95% bootstrap intervals** (25 speakers; paired differences are D minus the other model)

| Comparison | Overall AUC difference | Unseen-XTTS AUC difference |
|---|---|---|
| D vs original lower LR | +0.001 [-0.014, +0.011] | **+0.021 [+0.006, +0.035]** |
| D vs batch 01 baseline | -0.001 [-0.024, +0.014] | +0.022 [-0.008, +0.045] |
| D vs original baseline | -0.002 [-0.028, +0.017] | +0.019 [-0.021, +0.056] |
| D vs hidden 64 | **+0.029 [+0.018, +0.041]** | **+0.063 [+0.038, +0.084]** |
| D vs R2-M1 handcrafted LR | **+0.132 [+0.076, +0.186]** | **+0.378 [+0.278, +0.486]** |

Macro F1 at 0.5, D minus original lower LR: +0.0080, interval [-0.0007, +0.0168].

**What the evidence supports**
- D is clearly better than hidden 64 and than the handcrafted LR, and has the best unseen-XTTS ranking (ROC-AUC 0.904, EER 14.6%).
- D has the lowest false-alarm rate at 0.5 (8.4%) among the finalists and the highest threshold-robust F1 (0.909 with a cross-validated threshold).
- D versus the previous pick: 17 fewer genuine false alarms, 6 fewer XTTS detections, 1 more known-family miss.

**What it does not support**
- D is **not significantly better** than the original baseline or the batch 01 baseline on overall quality, and the F1 gain over the previous pick has an interval that includes zero. Muneeb's own report says no statistical superiority is claimed, and that is correct.
- D is the best epoch of a noisy trajectory (winner's curse). Across epochs 8 to 20 of the same run, F1 at 0.5 averages 0.821 ± 0.034 and unseen-XTTS AUC averages 0.885 ± 0.014, against 0.869 and 0.904 at epoch 16. Expect a fresh estimate to be nearer the averages.

## 4. Why D, and the trade-offs

**Reasons**
1. It is the checkpoint ranked first by the rule R3-M1 declared before running batch 01 (full selection macro F1 at 0.5, ties to the older lower-LR model).
2. It leads on several independent measures at once: F1, false-alarm rate, XTTS ranking and EER.
3. No candidate beats it significantly on any measure we tested. The choice against the batch 01 baseline is close to a coin flip, and D is preferred because it leads on more measures, not because the gap is proven.

**Trade-offs**
- **Calibration is worse.** ECE 0.093 and log loss 0.335, against about 0.055 and 0.27 for the original baseline and batch 01 baseline. R3-N1's temperature scaling should address this.
- **It misses a third of unseen XTTS at 0.5** (recall 66.4%). The original baseline catches 92.6% but falsely flags 31% of genuine speech, which is unusable. No candidate delivers both at a fixed 0.5.
- **Low-false-alarm region.** At a 5% false-alarm budget the baselines detect more XTTS than D (about 41 to 43% against 13%). Three speakers (1578, 302, 7190) account for 33 of D's 42 false alarms, a false-alarm rate of about 55% against 2% for the other speakers. Measured audio properties (noise floor, loudness, silence, high-band share) do not explain it. Intervals at this operating point are very wide, so treat it as unresolved.
- **Known-family AUC is lower** (0.976 against 0.985 to 0.994), caused by the same speakers. D still detects 998 of 1,000 known-family synthetic recordings at 0.5.

**Rejected:** hidden 64 (about 0.03 lower F1 and AUC with intervals excluding zero), the epoch-20 checkpoints of both full runs (F1 at 0.5 collapses to about 0.77), and the R2-M1 handcrafted LR (near chance on unseen XTTS).

## 5. Conflict with the GR3 selection rule

REVISED_PLAN.md (R3-ALL) states the GR3 rule as unseen-XTTS selection macro F1, then lower EER, lower FPR, lower p95 latency. R3-M1 ranked on **full** selection macro F1 instead. They disagree at a fixed 0.5:

| Candidate | Unseen-XTTS macro F1 at 0.5 |
|---|---:|
| Original baseline | 0.8053 |
| Batch 01 baseline | 0.7893 |
| **Batch 01 lower LR (D)** | 0.7866 |
| Original lower LR | 0.7766 |

Read literally at 0.5, GR3 would favor the original baseline. That ranking rewards its 31% genuine false-alarm rate (it flags almost everything as synthetic), so it is a property of the fixed threshold, not evidence of a better detector. Before GR3 the team should decide, and log, that the GR3 rule is evaluated at the R3-N1 operating threshold on the final candidates, not at 0.5. Until then D is the R3-M1 candidate, not the GR3 production choice.

## 6. Actions and open items

1. **R3-N1 (Nouman):** use the batch 01 hand-off predictions in `experiments/r3_m1/runs/batch01/selected/`. Fit temperature on the calibration role and choose the threshold on the operating role. Do not use 0.5. Report recall at fixed false-alarm rates beside macro F1.
2. **Review (Usman):** independent review of `final_selection.json` and `final_handoff.json` is pending and must not be assumed.
3. **Decision log:** record (a) that D supersedes the earlier `lower_lr` epoch 10 selection, and (b) the resolution of the GR3 evaluation threshold above. `docs/decision_log.md` currently contains the "R3-M1 experiment settings" section three times (merge duplication) and no entry for this selection.
4. **Main ledger:** `reports/ledger.csv` has the original two R3-M1 rows but not the batch 01 rows, which live in `experiments/r3_m1/ledger.csv`. Sync them or state which ledger is authoritative.
5. **Speaker follow-up:** check what is special about speakers 1578, 302 and 7190 before trusting any false-alarm estimate.
6. **Fallback:** keep `b01_baseline_full20` epoch 7 as the documented alternative if review or R3-N1 findings count against D.

## 7. Limits of this record

Single seed, result-informed search, one selection role of 25 speakers, uncalibrated scores. The recomputation used the saved prediction files and the project's `paired_group_bootstrap`. It did not retrain anything. No final-test or Chatterbox data was touched.
