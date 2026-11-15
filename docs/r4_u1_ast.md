# R4-U1b: AST-A, Protocol A warm-up and partial fine-tune

Code: `src/echoproof/training/r4_u1_ast_data.py` (guarded A data, parity sample), `src/echoproof/training/r4_u1_ast_train.py` (recipe guards, warm-up, fine-tune proofs, selection, export), `scripts/r4_u1_ast.py` (`warmup`, `finetune`, `package`, `parity`).
Tests: `tests/test_r4_u1_ast.py`. Colab: `notebooks/R4_U1_Colab_AST_Warmup_A.ipynb` (1/2) and `notebooks/R4_U1_Colab_AST_FineTune_A.ipynb` (2/2). Recipes: `configs/r4_u1_ast_warmup_a.json`, `configs/r4_u1_ast_ft_a.json`.
Owner: Usman. Reviewer: Muneeb (plan table).

**Status: implemented and tested locally with a tiny random AST. The warm-up and the fine-tune have run on Colab and are verified (results below); the bundle `bundles/r4_u1_ast_partial_A` is packaged and passed parity. Both recipes carry `team_agreed_user_reported_2026-11-15`** (the project owner reported the team's agreement on 15 November 2026; no individual approval record exists in the repository). Both commands refuse a recipe whose status does not start with `team_agreed`.

## What it does

Protocol A reuses the B-selected configuration. The user chose a fresh A warm-up (not the B warm-up checkpoint) and only the B-selected baseline fine-tune.

1. **Warm-up (head only).** The B R2-U2 recipe unchanged (AdamW 1e-4, 2 epochs, microbatch 4 x accumulation 4, fp16, constant lr, seed 42) on one seeded 4-second crop per A training recording per epoch (7,550 recordings, all five components; 6,040 TTS and 1,510 genuine, natural composition, no rebalancing). The backbone must equal the 14 pinned pretrained groups at the start, at the end and in `best.pt`; the classifier must change. Output `warmup_result.json`: best epoch, selection macro F1, checkpoint SHA-256, group checksums.
2. **Fine-tune (partial).** The B baseline unchanged (last two blocks, final norm and classifier; backbone 1e-5, head 1e-4, weight decay 0.01, microbatch 8 x accumulation 2, 8 epochs always, cosine, fp16 with GradScaler, seed 42). The only recipe difference is `warmup_steps = 189` instead of 114, keeping the B proportion of 5 percent of the optimiser steps for the larger A training set (3,776 steps against 2,272).
   - Initializer: only the warm-up result's checkpoint, after its SHA-256, strict weights-only loading and the pretrained group check.
   - Before training: partial-stage check; gradient proof on a copy (non-zero finite gradients and changed checksums exactly in the four unfrozen groups, frozen groups bit-identical); epoch-0 evaluation of the loaded warm-up, whose macro F1 must equal the warm-up result's within 0.003 (in B the reference was the R2-U2 value).
   - After training: the checksums of the final model and of `best.pt` must show the frozen groups unchanged and the unfrozen groups changed, or the run fails. `adaptation_evidence.json` records this, the skipped GradScaler steps and the per-epoch history.
   - Selection on selection-role recording macro F1 at fixed 0.5, earliest epoch on ties, frozen in `selection.json` before the calibration and operating roles are scored. Exports `predictions_{chunks,recordings}_{selection,calibration,operating}.parquet` in the BiGRU column layout, raw and uncalibrated, after the reloaded checkpoint reproduces its saved predictions (tolerance 1e-3).
   - Paired comparison of the A warm-up and the adapted model on the same selection chunks and recordings, overall and per component (genuine versus each generator), with the number of recordings fixed and broken at 0.5. A has no unseen generator, so the B "unseen XTTS" panel is replaced by these per-component panels, which are informational only.
3. **Bundle and parity.** `package` converts the selected checkpoint to safetensors with the pinned config (tensor-by-tensor check; frozen groups equal the pinned weights, unfrozen groups differ) as protocol A, raw uncalibrated. `parity` scores 50 selection chunks covering all five components from raw audio with the bundle adapter; the fp16 CUDA adapter must reproduce the exported probabilities within 0.01, fp32 on CUDA and CPU are measured and reported, not gated.

## Why new modules

The B sources (`r3_u2_train`, `r3_u2_data`, `r2_u2_scope`) are frozen by the GR3 lock and hard-code Protocol B (recipe protocol check, two experiments, XTTS-not-in-training rule, B panels, protocol "B" in ledger rows). `r4_u1_ast_*` imports what is generic (`load_warmup`, `check_partial_stage`, `gradient_proof`, `proof_batch`, `state_group_digests`, `score_amp`, `adaptation_evidence`, the shared Trainer, `ASTDetector`, `package_ast_detector`, `ASTBundleAdapter`) and restates only what is bound to B. No B file is edited; a test checks the B source hashes still match the lock.

## Verified locally

- 30 AST-A tests pass (with the R3-U2 and WavLM suites, 88 tests): recipe equality with B and each departure refused (including a wrong `warmup_steps`), the team-agreement guard, guarded A data against a miniature A project (no test member readable, counts must equal the R4-S1 manifest), the warm-up (head only, backbone unchanged, result recorded, checkpoint re-hashed on loading), the fine-tune evidence, refusal of a reused folder, a wrong warm-up score or a forged checkpoint hash, the frozen selection, exports with all five components and the per-component paired comparison.
- On the real manifests (no audio): the A plan is 7,550 training recordings and 24,412 validation chunks (selection 12,331, calibration 5,777, operating 6,304), equal to the verified R4-S1 manifest; plan sha256 `b92611b2…`; split hash `a9075367…`.

## Expected cost on one Colab T4 (estimates from the B rates)

Warm-up about 15 minutes; fine-tune about 1 to 1.5 hours including the epoch-0 evaluation and the export of the three roles; bundle and parity a few minutes. Both are resumable by `RUN_ID`.

## Limits

- Single seed. The difference between warm-up and adapted, and between A and B models, may be within seed noise.
- The A selection role mixes five generators; read the per-component panels beside the headline F1. A's in-distribution scores are not comparable with B's.
- The scheduler clock follows the shared Trainer (one step per optimiser attempt, including GradScaler-skipped steps), as in B; the skipped count is reported.
- Raw scores are uncalibrated; thresholds come from R4-N1. fp16 and fp32 numerics differ slightly (as in B).

## Warm-up result (Colab, 15 November 2026, reviewed locally)

Run `r4_u1_ast_warmup_A_20261115T065205Z`, Tesla T4, transformers 5.22.1: 2 epochs in 19.0 minutes, 0 skipped GradScaler steps. Selection-role recording macro F1 at 0.5 was 0.7722 after epoch 1 and **0.8541 after epoch 2 (selected)**; ROC-AUC 0.9815, EER 0.0785, TTS recall 0.9965, genuine recall 0.616. Checkpoint `48d23cef...`. The backbone equals the 14 pinned pretrained groups at the start, at the end and in `best.pt`; the classifier changed; the report, checkpoint, result, source and ledger hashes were checked. As in B the head-only warm-up leans towards TTS; the fine-tune is what adapts the backbone. The fine-tune's epoch-0 check must reproduce 0.8541 within 0.003.

## Fine-tune result (Colab, 15 November 2026, reviewed locally)

Run `r4_u1_ast_ft_A_20261115T102637Z` from the warm-up above, Tesla T4, transformers 5.22.1: 8 epochs in 86.8 minutes, 4 skipped GradScaler steps, peak 1,288 MB. Selection-role recording macro F1 at 0.5 rose from 0.9767 (epoch 1) to **0.9969 (epoch 8, selected; epoch 7 is 5e-6 lower)**; EER 0.006, genuine recall 0.992. Checkpoint `b0d81b5b...`. Epoch 0 reproduced the warm-up F1 exactly; the gradient proof and the final and best checksums show only `block.10`, `block.11`, `final_norm` and the classifier changed. Against the warm-up on the same recordings: 195 fixed, 1 broken, macro F1 +0.1427. The fp16 bundle adapter reproduces the exported probabilities exactly (0 flips); fp32 differs by at most 0.0026 (0 flips). On the later roles at 0.5: calibration F1 0.9935, operating F1 0.9940 (genuine flagged 1.5%). Selection was frozen before those roles were scored. Single seed; raw scores; in-distribution only; thresholds come from R4-N1.

## Run-time fixes (15 November 2026)

- The frozen-source test accepts the raw, LF and CRLF forms of a file, because Drive copies may have LF endings while the GR3 lock records the Windows CRLF bytes.
- Resuming compares the saved contract field by field and names every difference. Only the Colab platform string and this script's own hash may differ, and each tolerated difference is appended to `<run>_resume_log.jsonl`. The first fine-tune attempt (`r4_u1_ast_ft_A_20261115T074952Z`, one epoch, no evidence) was not continued; the completed run was started again from the same warm-up. The first folder is kept unchanged.
