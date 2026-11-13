# R3-U1 — WavLM head, Protocol B

Status: implemented; real-data results are recorded in `reports/r3/r3_u1_B/` and the decision log once the runs finish.
Recipe: `configs/r3_u1_wavlm_head_b.json` (team agreement reported by the user on 11 November 2026; no individual approval record in the repository).

## What it is

A small head on the frozen WavLM embeddings cached by R2-U1 (`data/features/r2_u1_B`):
`Linear(in, 128) → GELU → Dropout(0.2) → Linear(128, 2)`, with `in = 768` (mean pooling) or `1536` (mean+std pooling).
AdamW, lr 1e-3, weight decay 0.01, batch 128, constant learning rate, seed 42, always 20 epochs, best epoch kept (no early stopping).
Each training crop is weighted by 1 / (crops of its recording).

## Contract with the rest of the pipeline

| Item | Rule |
|---|---|
| Inputs | Training crops and validation chunks of the R2-U1 development index. The test partition is never read; Chatterbox is absent. |
| Selection | Per variant: the epoch with the best selection-role recording macro F1 at fixed 0.5 (mean of chunk probabilities; earliest epoch on ties, strict improvement). Between variants: higher F1 wins, an exact tie goes to mean pooling. |
| Freeze | `selection.json` is written before calibration and operating roles are scored. An existing selection is never overwritten. |
| Outputs | `predictions_{chunks,recordings}_{selection,calibration,operating}.parquet`, same columns as the BiGRU export, raw uncalibrated scores (R3-N1 still required). |
| Adapter | `WavLMBundleAdapter` (`models/wavlm_adapter.py`) satisfies `InferenceModelAdapter`: waveforms in, P(TTS) out, encoder inside the bundle. |
| Safety | Inference reads tensors only (head weights-only, encoder safetensors). The pinned `pytorch_model.bin` is a pickle, so it is converted once at packaging time after its SHA-256 matches the pin, and the conversion is checked tensor by tensor. |

## Checks built in

1. **Alignment** (`check-data`): cache rows equal the Protocol B validation chunks and training recordings defined by the split, chunk manifest and cohort (ids, labels, components, groups, roles, counts). Identity only, no model scores.
2. **Data builder**: rejects the test partition, Chatterbox, duplicate ids, bad labels, non-finite vectors, missing classes and speaker overlap.
3. **Parity**: for the selected head, scoring raw waveforms (selection-role chunks read from the cohort shards, hash-verified against the manifest, every component except Chatterbox) must match scoring the cached vectors (atol/rtol 1e-4), and the adapter must match too.
4. **Bundle**: per-component SHA-256, tampering detected before use.

## Running it (Colab)

Everything runs on Colab with [R3_U1_Colab_WavLM_Head.ipynb](../notebooks/R3_U1_Colab_WavLM_Head.ipynb); nothing is run locally. Sync the new code, the recipe, the R2-U1 cache and the cohort shards to Drive first. The notebook runs the fixture tests, then `check-data`, `train` (CPU is enough), `parity` (GPU, TF32 off, pinned WavLM downloaded and hash-verified), `package`, and verifies the handoff. Every run writes to a new location and refuses to reuse a non-empty one.

Subcommands of `scripts/r3_u1_wavlm_head.py`, all requiring `--owner`: `check-data`, `train`, `parity`, `package`.

## Known limits

- One seed. A difference between the two poolings may be within seed noise; reports must say so.
- Parity covers a deterministic sample (default 50 selection-role chunks), not every chunk.
- Raw scores are uncalibrated; thresholds come from R3-N1.
