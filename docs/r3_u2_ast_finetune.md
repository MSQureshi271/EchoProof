# R3-U2 — AST partial fine-tuning, Protocol B

Status: the training run `r3_u2_20261112T134049Z` is complete and reviewed (see the decision log). The inference adapter and bundle described at the end are implemented and syntax-checked but **not yet run**: their tests, packaging and parity check run on Colab (`notebooks/R3_U2_Colab_AST_Bundle.ipynb`).
Recipe: `configs/r3_u2_ast_ft_b.json` (status `team_agreed_user_reported_2026-11-12`: the team's agreement was reported by the project owner on 12 November 2026; there is no individual approval record in the repository).

## What it does

Starts from the accepted R2-U2 warm-up (`best.pt`, sha256 `6a3883e5…0bd57`, epoch 2) and fine-tunes the last two AST encoder blocks, the final LayerNorm and the classifier head. Blocks 0 to 9 and the embeddings stay frozen, parameters and buffers.

| Item | Value |
|---|---|
| Optimiser | AdamW, backbone lr 1e-5, head lr 1e-4, weight decay 0.01, rebuilt from scratch |
| Batch | microbatch 8 × accumulation 2 = effective 16; eval microbatch 16 |
| Schedule | 114 linear warm-up steps, then cosine to zero over 2,272 optimiser steps (8 epochs × 284) |
| Precision | fp16 autocast with GradScaler (T4) |
| Epochs | 8, always all of them; the best epoch is kept (early stopping cannot fire) |
| Data | one seeded 4-second crop per Protocol B training recording per epoch (1,510 genuine, 3,020 TTS); selection-role chunks for evaluation |
| Experiments | `baseline` (backbone 1e-5) and one alternative `lower_backbone_lr` (backbone 3e-6) |

## Rules it enforces

1. **Fresh start.** Only the checkpoint's `model` weights are loaded, strictly, after its SHA-256 and the untouched pretrained groups (14 pinned group checksums) are verified. No optimiser, scaler, scheduler or RNG state is restored.
2. **Partial stage check.** The trainable set must be exactly `classifier.*`, the final norm and the last two blocks; no group may be partially trainable.
3. **Gradient proof** (before training, on a copy of the model): non-zero finite gradients and changed checksums exactly in the four unfrozen groups; frozen groups bit-identical.
4. **Epoch-0 reproduction.** The loaded warm-up is scored on the selection role; its macro F1 must equal the R2-U2 value 0.7913689 within 0.003. This catches wrong weights or wrong preprocessing. Its predictions are the paired comparison.
5. **After training**, checksums of the final model and of `best.pt` must show the frozen groups unchanged and the four unfrozen groups changed, or the run fails.
6. **Selection** is on selection-role recording macro F1 at fixed 0.5 (mean of chunk probabilities). Ties: earliest epoch, then baseline. It is frozen in `selection.json` before calibration and operating audio is scored, and an existing selection is never overwritten.
7. **Export**: raw uncalibrated predictions for the three validation roles in the BiGRU column layout (`predictions_{chunks,recordings}_{role}.parquet`), after the reloaded winner reproduces its saved best-epoch predictions (tolerance 1e-3).
8. **Paired comparison** in `evidence.json`: warm-up and adapted on exactly the same selection chunks and recordings, overall and for the known-generator and unseen-XTTS panels, with counts of recordings fixed and broken at 0.5.

## Files

`configs/r3_u2_ast_ft_b.json`, `src/echoproof/training/r3_u2_data.py`, `src/echoproof/training/r3_u2_train.py`, `scripts/r3_u2_ast_ft.py`, `tests/test_r3_u2_ast.py`, `notebooks/R3_U2_Colab_AST_FineTune.ipynb`. The shared `Trainer`, `ASTDetector`, `CropDataset`/`ChunkDataset` and the R3-M1 selection helpers are used unchanged.

## Running it (Colab)

Sync the new files and the warm-up `best.pt` to Drive, open the notebook on a T4 runtime and run the cells in order. A disconnected run resumes by entering its `RUN_ID`; the frozen experiment contract must be unchanged. Expect roughly 45 minutes per experiment (R1-U2 measured 36 clips/s partial training and 49 clips/s evaluation), plus a few minutes for the export.

## Known limits and open points

- **Scheduler clock.** The shared Trainer advances the scheduler once per optimiser attempt, including fp16 GradScaler-skipped steps (the convention of the accepted R2-U2 run). The skipped count is reported in each `adaptation_evidence.json`. Changing it to an applied-update clock would need team approval.
- **One seed.** Differences between the experiments, and between warm-up and adapted, may be within seed noise.
- **Selection F1 mixes generators.** Report the unseen-XTTS panel beside it; the frozen warm-up misses about half of XTTS at 0.5.
- **Warm-up predictions** are paired on the selection role only. Calibration and operating comparisons are later work.
- **Raw scores are uncalibrated**; R3-N1 is still required.

## Inference adapter and bundle

`src/echoproof/models/ast_adapter.py` (`ASTBundleAdapter`) satisfies `InferenceModelAdapter`: mono 16 kHz float waveforms of 1 to 64,000 valid samples in, P(TTS) out, using the same `ASTDetector` code path that produced the exports. `src/echoproof/models/r3_u2_bundle.py` packages the checkpoint named by the frozen `selection.json`:

1. The selected `best.pt` is loaded strictly into the pinned AST. Its frozen groups (embeddings, blocks 0 to 9) must equal the 14 pinned pretrained checksums, and its unfrozen groups (blocks 10 and 11, the final norm, the classifier) must differ.
2. The weights are re-saved as safetensors with a two-class config (labels genuine/tts) and the pinned feature-extractor config. The reloaded tensors must equal the checkpoint's tensor by tensor. Inference reads no pickle.
3. The bundle records the base model identity, the warm-up and selected checkpoint hashes, the recipe hash, the split hash, the group checksums and per-component SHA-256 values. It is marked `development_uncalibrated`; R3-N1 is still required.

**Precision.** The exports were scored with fp16 autocast on a CUDA device. `precision="fp16"` (CUDA only) reproduces that numerics; `precision="fp32"` is the default and the only option on CPU, and gives slightly different probabilities. Thresholds fitted by R3-N1 on the exports therefore carry that difference when the adapter runs in fp32. The parity step measures it (maximum and mean probability difference and the number of decisions that would flip at 0.5) for fp32 on CUDA and on CPU, and reports it without gating on it.

**Parity gate.** Fifty deterministic selection-role chunks (stratified over component and short or full chunk, Chatterbox excluded) are read from the cohort shards, whose loader checks every recording's PCM hash. The fp16 CUDA adapter must reproduce the exported probabilities within 0.01. The tolerance is an implementer's choice: fp16 results depend on the batch shape, and the parity sample is a subset scored in batches of 8, so bit equality is not expected.

Files: `src/echoproof/models/ast_adapter.py`, `src/echoproof/models/r3_u2_bundle.py`, `src/echoproof/training/r3_u2_parity.py`, `scripts/r3_u2_ast_bundle.py` (`package`, `parity`), `tests/test_r3_u2_adapter.py`, `notebooks/R3_U2_Colab_AST_Bundle.ipynb`.
