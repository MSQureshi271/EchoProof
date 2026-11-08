# Batch 02 proposal: hidden-size-64 diagnostic

Date: 2026-11-08. Owner: Muneeb. Reviewer: Usman. Status: implemented; Muneeb explicitly approved the experimental-only exception; GPU execution pending.

## Purpose and interpretation

Test whether reducing capacity from 128 to 64 hidden units per direction improves selection performance, particularly transfer to XTTS. The current selected model detects 998/1,000 known-family synthetic recordings and 332/500 XTTS recordings at 0.5. This gap motivates a capacity hypothesis; it does not establish overfitting or prove a smaller model will help.

This is a result-informed, single-seed diagnostic. Existing `locks/B_lock.json` describes a provisional GR2 LR bundle, but occupies the final B-lock path. The team has not clarified its final-lock status or Protocol A exposure history. Results must therefore be labeled isolated post-lock exploratory evidence, not independent pre-lock research evidence.

Muneeb authorized preparing separate experiments and required asking before violating any lock. After reviewing the concrete exception question, Muneeb replied: "Approve this experimental-only exception." The approval covers one additional B train/selection fit despite the existing lock file, with immutable locked artifacts and no promotion. The [approval record](../../configs/batch02_authorization.json) quotes that reply and binds the machine-readable plan hash and existing lock hashes. Other lock exceptions require asking Muneeb again. Do not modify that record to bypass a refusal.

## Frozen design

| Setting | Value |
|---|---|
| Candidate | `b02_hidden64_full20` |
| Architecture | 64-mel input, two bidirectional GRU layers, hidden size 64 per direction, masked mean pooling, dropout 0.3, two logits |
| Trainable parameters | 124,674, versus 445,954 for hidden size 128 |
| Matched reference | `b01_lower_lr_full20`, best epoch 16 |
| Initialization | Fresh weights and separate process, seed 42; no weight transfer |
| Optimizer | AdamW, LR 0.0003, weight decay 0.01, gradient clip 1.0 |
| Batches and precision | Train 32, evaluation 64, fp32 |
| Schedule | ReduceLROnPlateau, factor 0.5, patience 1 |
| Stop | At most 20 epochs; patience 21 disables early stopping within that budget |
| Data | Same frozen B cohort/statistics; 4,530 train recordings/151 speakers; 2,000 selection recordings/25 speakers/10,029 chunks |
| Training crop policy | One seeded four-second crop per recording per epoch, unchanged |
| Primary comparison | Full selection recording macro F1 at 0.5, mean chunk probabilities |
| Supporting diagnostics | Known/XTTS panels, XTTS recall, genuine FPR, AUC, EER, runtime, per-epoch scores, hypothetical patience-3 stopping |
| Budget | One fresh fit; 60 cumulative minutes training/selection validation; 15 cumulative minutes fixture checks |

Only hidden size changes from the matched reference. No new seeds, extra layers, loss weighting, augmentations, cohort changes or threshold tuning are included. The scheduler retains the same rule; realized learning-rate changes can differ because validation scores differ. A different architecture consumes random numbers differently, so identical seed does not imply identical weights or dropout masks. Neither improved nor worse performance from this one fit proves a general capacity effect.

Compare all five candidates: both original checkpoints, both batch01 checkpoints, and hidden64. Preserve the existing earliest-epoch tie rule and between-candidate tie preference for `original_lower_lr`, then lexicographic candidate ID. XTTS diagnostics do not replace the declared primary metric. Report the advisory winner even if hidden64 loses; do not launch replacement fits until an improvement appears.

## Isolation and lock handling

- All new files stay inside `experiments/r3_m1/`. Original production modules and frozen batch01 scripts/contracts are read-only dependencies.
- `configs/batch02_authorization.json` records actual approval and binds `approved_by: Muneeb`, `approved: true`, the machine-readable plan hash, current `locks/*_lock.json` hashes, and exact restrictions defined in `batch02_contract.py`.
- The resolved contract additionally pins locked bundle files, all prior comparison artifacts, source/dependency hashes, data hashes, architecture, budget and authorization. A change to lock records requires asking Muneeb again. A changed locked bundle after freeze stops execution.
- The LR bundle referenced by the local lock (`bundles/protocol_B_lr`) is absent locally. The workflow records a missing bundle as absent, not as verified. If present on Colab its files are hashed; if absent its absence is frozen. A later creation or change invalidates that snapshot. This diagnostic neither repairs nor creates locked bundles.
- Only B train and selection audio are used. No calibration, operating, test or Chatterbox audio is scored or used for tuning. There is no post-selection export stage in this batch.
- Reports are advisory: `reports/batch02/comparison.json`, `comparison.md`, and `experimental_preference.json`. Existing `reports/final_selection.json`, `final_handoff.json`, batch01 files, project ledger, bundles and locks remain unchanged.
- Separate outputs live in `runs/batch02/`, including `ledger.csv`, attempt timing, fixtures, checkpoints and epoch predictions. No automatic integration or official selection change is permitted. The team must separately decide whether any result can be used, subject to project protocol and lock constraints.

## Execution after explicit approval

1. Sync the new experiment files and the approved authorization record to the Drive project. Preserve exact batch01 dependencies and source files. Existing lock records on Drive must match the approved lock hashes; do not delete or edit locks to get past a refusal.
2. Open `notebooks/R3_M1_Hidden64_Colab.ipynb` using the same T4/runtime versions as batch01. Set the project root and shard location.
3. Freeze `configs/batch02_resolved.json` once. The contract checks authorization, protected artifacts, prior results and runtime compatibility. Reconnection reuses it.
4. Run recorded fixture preflight. No failed or skipped tests are accepted in Colab. Hidden64 CPU/CUDA checks exercise gradients, padded-sample invariance and state reload. Test attempts count against the separate 15-minute allowance.
5. Train one candidate. Fresh starts refuse an existing run directory; resume requires consistent best/latest/history/epoch predictions and the same architecture/configuration provenance. CUDA GRU dropout means a resumed trajectory need not be bit-identical to uninterrupted training.
6. Recompute selection metrics for all five candidates and every hidden64 epoch. Write the advisory comparison. Do not promote the winner or score other roles.
7. Copy back `configs/batch02_resolved.json`, `reports/batch02/`, and `runs/batch02/` for review. Keep checkpoints and every epoch's predictions.

Training time is counted across recorded attempts and checked at epoch boundaries; the last running epoch can slightly exceed the limit. A hard disconnect can leave an active marker whose elapsed time is unresolved. Preserve that marker and all files for investigation; never delete it, raise the budget, or modify hashes to bypass a refusal. A failed/incomplete run is not a valid completed comparison.

## Pending review

Local validation: 23 tests passed and one PyTorch-dependent fixture module was skipped because PyTorch is absent locally; see [local validation](local_validation.json). Script and notebook syntax checks passed. All six frozen batch01 experiment script hashes remain unchanged. The approval record passes the plan/lock checks. No GPU run or new scientific result is claimed. Colab must pass the CPU/CUDA fixtures with zero skipped tests before training.

Nouman's R3-N1 task and Usman's review remain outside this diagnostic implementation. Any future change to locked research artifacts requires a new explicit discussion with Muneeb and the team.
