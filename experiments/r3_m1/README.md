# R3-M1 follow-up experiment plan

Status: batch 01 completed; selected lower-LR checkpoint at epoch 16 (selection macro F1 0.869120). Independent review remains pending. Batch 02 is implemented as a separate hidden-size-64 diagnostic; Muneeb approved the specific experimental-only post-lock exception. GPU execution is pending. Owner: Muneeb. Reviewer: Usman.
Authority: the 7 November 2026 extension in [REVISED_PLAN.md](../../REVISED_PLAN.md).

## Batch 02: hidden-size-64 diagnostic (8 November 2026)

See the [bounded proposal and execution instructions](reports/batch02/plan.md) and [machine-readable plan](configs/batch02_plan.json). Muneeb requested a smaller-capacity comparison, then clarified that experiments must remain separate and the team decides any adoption, with explicit approval before a lock exception.

`locks/B_lock.json` now exists and describes a provisional GR2 LR bundle. Its relationship to the final GR3 lock is unresolved; Protocol A exposure history is also unconfirmed. The batch01 guard remains unchanged. Muneeb explicitly approved this experimental-only exception; [the authorization record](configs/batch02_authorization.json) binds it to the plan hash and existing lock hashes. Other lock exceptions still require asking Muneeb first. This permission does not authorize changing locked research artifacts or treating the diagnostic as pre-lock evidence.

The new [Hidden64 notebook](notebooks/R3_M1_Hidden64_Colab.ipynb) runs one two-layer, hidden-size-64 model from scratch with the same LR 0.0003, seed 42, training data and 20-epoch procedure as batch01's selected candidate. Its advisory comparison includes all four prior checkpoints. It does not export calibration/operating predictions or replace the previous selection. Its separate reports, ledger and runs cannot promote a model into a locked result. Batch01 instructions below are retained as historical documentation; do not rerun batch01 to bypass the current lock.

## Question and existing evidence

Does patience 3 discard a better later checkpoint within the existing 20-epoch budget? Separately, is there an implementation defect that explains the fluctuations in selection macro F1?

The original run `r3_m1_20261107T144037Z` is the read-only reference. Baseline stopped at epoch 6 with its best selection macro F1 of 0.8597542210891138 at epoch 3. Lower LR stopped at epoch 13 with its best of 0.8611277924573625 at epoch 10. These are completed results, not proof that either stopping point was optimal. The lower-LR run already showed recovery after two non-improving epochs. No defect has yet been established by this planning review.

All new files belong here: `configs/`, `scripts/`, `notebooks/`, `tests/`, `runs/`, `reports/`, and `ledger.csv`. Existing project code and data are read-only dependencies. Do not invoke the original end-to-end runner: it writes to the original locations and automatically scores calibration and operating data after selection.

## Stage 1: audit before new model fitting

Implement `scripts/audit.py` and `tests/test_audit.py`, then produce `reports/audit.json` and `reports/audit.md`. Every check must distinguish passed, failed and not measured. Record actual source, dependency and input hashes.

| Check | Evidence and acceptance |
|---|---|
| Data and roles | Match the frozen split and R2-S1 hashes; exactly 4,530 training recordings from genuine/Melo/VITS, 151 speakers; 2,000 selection recordings, 25 disjoint speakers, 10,029 fixed chunks. Check labels, chunk identities, speaker separation, valid sample counts and PCM hashes. Load audio only from train and selection. |
| Frontend | CPU reference vs GPU log-mel parity, frozen 64-bin normalization, short clips, silence, final tails and padded neighbours. Preserve the original declared tolerances; do not loosen them simply to pass. |
| Model update | On disposable fixtures, verify finite loss, gradients in the GRU and head, changed trainable parameters after an update, and unchanged normalization buffers. Run diagnostics in a separate process from training. |
| Evaluation | Eval mode, finite complete probabilities, mean-of-chunks recording aggregation, fixed 0.5 threshold, authoritative macro F1. Recompute original selection results and check best checkpoint identities. |
| Schedule and stop | Replay histories to reproduce best epoch and patience-3 stop. Distinguish the LR used during an epoch from the LR stored after the scheduler step. Verify the plateau scheduler on fixtures. |
| Recovery | Test interruption at epoch boundaries, rejection of changed configuration/provenance, and consistency of history, best/latest checkpoints and ledger. Inspect the crash window where best/history files can be written before latest; do not assume multi-file writes are one atomic transaction. Record CUDA GRU dropout continuation limitations. |

A material defect blocks the training comparison until documented and resolved in experiment-local code. Record which original results it affects and whether an affected candidate must be excluded. Infrastructure fixes must not silently change numerical training settings. If GPU or data checks are unavailable locally, mark them pending and execute them in Colab before fitting.

The experiment-local `scripts/checkpoint_guard.py` checks that latest/best checkpoint metadata, the complete epoch history and both prediction files agree. The follow-up runner must call `check_run` before every resume and stop without modifying artifacts if it raises `GuardError`. It is a fail-closed guard, not automatic recovery: an inconsistent run requires inspection and a separately documented resolution. The guard cannot restore cuDNN's private GRU dropout state or promise bit-identical CUDA continuation. `notebooks/R3_M1_Audit_Colab.ipynb` exercises the guard's fixture tests and checks original checkpoint consistency; repeat the notebook after syncing the updated experiment folder to obtain the new audit report.

## Stage 2: batch 01 — stopping-rule comparison

Use [configs/batch01_plan.json](configs/batch01_plan.json). Before running, generate an immutable resolved contract under `configs/` with the source commit, dependency versions, architecture, full TrainConfig, original artifact hashes, data hashes, budget and selection rules. Planning JSON alone is not evidence of a frozen or completed run.

| Experiment | Initial LR | Training start | Epoch limit | Early stopping |
|---|---:|---|---:|---|
| `b01_baseline_full20` | 0.001 | Fresh initialization, seed 42 | 20 | Disabled |
| `b01_lower_lr_full20` | 0.0003 | Fresh initialization, seed 42 | 20 | Disabled |

Retain the original architecture, batch size 32, eval batch 64, AdamW weight decay 0.01, fp32, no class weighting, gradient clip 1.0, one seeded four-second crop per recording per epoch, and ReduceLROnPlateau with factor 0.5 and patience 1. The only training-policy change from the corresponding original configuration is removal of early stopping. With the existing trainer this can be represented by `max_epochs=20, patience=21`; document that exact representation. No thresholds, seeds, cohort sizes, hidden sizes or augmentation variants are added in this batch.

Use a fresh process per run, setting determinism and seeds before model construction. Do not reopen finished original checkpoints or bypass their configuration hashes. Fresh initialization avoids editing their stopping state; it does not promise bit-identical reproduction of earlier GPU runs. Compare the new prefix with the corresponding original history and report discrepancies.

The original observations suggest roughly 50 minutes for 40 epochs, plus setup, audits and exports. This is an estimate. The planned execution budget is 120 GPU-runtime minutes for the two training runs including per-epoch validation, with 30 additional GPU-runtime minutes for audit and final scoring. Record actual time and interrupted attempts; resume does not reset the budget. If the budget is exhausted, checkpoint at a safe boundary and report an incomplete comparison. Do not silently extend it or launch replacement candidates.

## Stage 3: answer the early-stopping question

Implement `scripts/train.py` and `scripts/compare.py` with tests under `tests/`. For each full run:

1. Save epoch-level training loss, selection loss, macro F1, EER, FPR, LR before/after the scheduler, gradient summaries, elapsed time and peak VRAM. Save selection chunk/recording probabilities each epoch, plus best/latest checkpoints and the checkpoint at the hypothetical patience-3 stop when it occurs.
2. Replay the recorded selection trajectory from epoch 1, updating best F1 only on a strict increase and stopping after three consecutive non-improving epochs. This is the paired, within-run early-stopping reference; no extra fit is required.
3. Compare its retained best checkpoint with the best over all 20 epochs. Report the hypothetical stop epoch, any later recovery epoch, F1 difference and runtime cost. If patience 3 would not stop before epoch 20, report that fact. This comparison isolates the stopping decision within the observed trajectory more directly than comparing different runs alone.
4. Explain score changes using selection-only confusion matrices, known and XTTS panels, per-speaker errors, and score distributions near 0.5. These diagnostics do not override the declared model-selection metric. No threshold tuning is included.

Write `reports/batch01_comparison.json`, `reports/batch01_comparison.md` and learning-curve figures. A later improvement supports the claim that patience 3 missed a better checkpoint on that trajectory; it does not prove a global optimum or a general benefit across seeds. No improvement is a valid result. A resumed CUDA run must disclose the dropout-state limitation; it cannot establish exact equivalence to an uninterrupted run.

## Stage 4: select and hand off

After the audit and batch comparison, compare the two original best checkpoints and the two full-run best checkpoints, excluding any invalidated by a confirmed defect. Use full B selection recording macro F1 at 0.5. Within a run, exact ties retain the earliest epoch. Across candidates, an exact tie prefers the original selected lower-LR model, then stable lexicographic candidate ID. Publish small differences as descriptive results; do not claim statistical significance from this single-seed, result-informed search.

Save `reports/final_selection.json` with candidate identities, source/config/checkpoint hashes, metric values, exclusions, decision rule and rationale. If a new hypothesis or defect requires another batch, record its bounded plan before running it and defer final selection until that declared batch completes. No unspecified grid search is authorized by this document.

Only after the candidate choice is fixed, export its raw predictions on all three B validation roles under this folder and prepare the R3-N1 handoff. If the original winner is retained, verify and reuse its existing predictions as read-only inputs where appropriate. Store the final evidence and Usman's actual review under `reports/`; do not invent a review or lock. Calibration and thresholds must belong to the selected checkpoint.

## Colab packaging and completion

### Run batch 01

1. Sync this entire experiment directory to the Drive project, retaining the original source and run artifacts unchanged. Open `notebooks/R3_M1_Experiments_Colab.ipynb` on a T4. Configure `ROOT`, `SHARDS` and the passed audit path (`reports/audit_20261107T165239Z.json` by default).
2. Confirm with the team that B is not locked and A training has not started. The script checks visible lock files and the main ledger, but cannot discover unrecorded external work. Use the same original runtime versions; no packages are automatically upgraded.
3. Run the notebook in order: tests, immutable contract creation, two fits in separate processes, comparison, and post-selection exports. The full inference/training workflow must still be validated by this Colab execution. Keep the runtime connected and avoid simultaneous notebooks using this experiment folder.
4. Share `reports/batch01_comparison.json`, `reports/batch01_comparison.md`, `reports/final_selection.json`, `reports/final_handoff.json` and `ledger.csv`. Keep `runs/batch01/` on Drive, including both histories, checkpoints, every epoch's selection predictions and selected-role predictions. The comparison SVG uses saved selection histories. These are development results, not final-test results or a reviewed lock.

### Resume and failure handling

On a normal caught interruption, repeat the notebook with the existing contract. The runner uses resume for existing run directories, checks the checkpoint guard and exact epoch-output coverage, and records elapsed time in `runs/batch01/attempts.jsonl`. The shared two-hour training budget is checked at epoch boundaries, so a last in-progress epoch can exceed the boundary slightly. Do not interpret ledger timing alone as the budget counter. The extra audit/export allowance is a planning limit; the historical audit report did not record GPU duration, so its prior consumption cannot be reconstructed. Export wall time is recorded separately and checked between roles.

A hard disconnection can leave `active_attempt.json` because elapsed time could not be finalized. That deliberately blocks further execution: preserve the marker and artifacts for investigation. Likewise, a missing latest checkpoint or partially written epoch predictions is not automatically repaired. Do not delete markers, edit hashes, or raise the budget to bypass a refusal. Original completed files remain read-only. New contract/source changes after training starts require a documented disposition, not a silent restart.

The separate `final_selection.json` is frozen before calibration/operating inference. If the same completed batch is rechecked, the comparison may reproduce that decision; it refuses to overwrite a different decision. No calibration or operating metrics are used for the comparison. Temperature scaling and operating thresholds remain R3-N1 work.

`notebooks/R3_M1_Experiments_Colab.ipynb` is the batch entry point for frozen-contract generation, training and reporting; the audit has its own notebook. Keep source, run directories and ledger below this experiment folder on Drive. If shards are copied to scratch, use an experiment-specific scratch directory and treat it as disposable; persist authoritative artifacts under this folder. Do not extract or score test/Chatterbox audio. Configure caches and test temporary outputs within the experiment workspace as well.

Completion requires a passing audit, two completed full-budget runs or a clearly disclosed incomplete result, reproducible comparison tables, a justified final choice, complete validated handoff predictions, and reviewer disposition. Calibration/operating outcomes and final-test outcomes must never guide tuning. Confirm GR3 is not locked and Protocol A has not begun before execution. The existing final-test firewall and cross-model GR3 production-selection rule remain in force.
