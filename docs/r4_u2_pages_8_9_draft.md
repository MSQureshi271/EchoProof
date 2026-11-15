# R4-U2 — pretrained models and compute (draft for peer review)

Owner: Muhammad Usman Murtaza. Scope: page 8 and the compute portion of page 9 in REVISED_PLAN.md, R4-M2 / R4-U2. Pagination is provisional until the team assembles the report. The training-procedure portion of page 9 belongs to Muneeb. This draft does not establish formal run acceptance or a protocol lock.

## Page 8 — pretrained speech representations

EchoProof uses microsoft/wavlm-base-plus and MIT/ast-finetuned-audioset-10-10-0.4593 for recording-level genuine-versus-TTS detection. Canonical audio is mono 16 kHz. Evaluation uses four-second chunks and their arithmetic mean P(TTS) per recording. Checkpoint selection uses the selection role at threshold 0.5; calibration and operating roles have separate downstream purposes. Protocol B tests transfer to generators excluded from B training; Protocol A includes all five components in development. Their validation scores measure different conditions and should not be interpreted as a controlled improvement from B to A. [REVISED_PLAN.md, R1-U1, R3-U1/U2, Phase R4; docs/r3_u1_wavlm_head.md; docs/r3_u2_ast_finetune.md]

### Frozen WavLM encoder and trained classifier

The encoder stays frozen and produces cached embeddings. A 128-unit GELU classifier with dropout 0.2 and two output logits is trained with AdamW (learning rate 0.001, weight decay 0.01), batch 128, constant schedule and seed 42. Recording crop weights prevent recordings with more crops from dominating. B compares mean pooling (768 features) with mean plus standard deviation (1536 features); mean+std was selected. The head contains 98,690 parameters for mean and 196,994 for mean+std, calculated from the two linear layers including biases; these exclude the frozen encoder. [configs/r3_u1_wavlm_head_b.json; src/echoproof/models/wavlm_head.py]

B mean selection macro F1 was 0.888522; mean+std was 0.913570, selected at epoch 14. A reused mean+std and selected epoch 13 with macro F1 0.999375. These are raw-score selection results. Direct waveform/cache parity and bundle checks support the exported representation path; sample parity does not prove equality on every possible recording. [reports/r3/r3_u1_B/selection.json; reports/r4/r4_u1_wavlm_A/selection.json; docs/r4_u1_wavlm_head.md]

### AST warm-up and partial adaptation

AST first trains its classifier while the backbone is frozen. Partial adaptation then trains blocks 10 and 11, the final normalization and classifier; embeddings and blocks 0–9 remain frozen. A fresh optimizer uses backbone learning rate 1e-5 and head learning rate 1e-4, weight decay 0.01, microbatch 8 with accumulation 2, FP16 with GradScaler, and warm-up followed by cosine decay. A uses 189 warm-up steps over eight epochs. Training takes one seeded four-second crop per training recording per epoch. [configs/r4_u1_ast_ft_a.json; completed run resolved_config.json and adaptation_evidence.json]

The A run records 86,190,338 total and 14,180,354 trainable parameters (16.45%). Its saved checksums verify unchanged frozen groups in the final and selected states and changed unfrozen groups. A separate gradient proof is retained. This is evidence of actual backbone adaptation, beyond training a new classifier. [runs/r4/r4_u1_ast_ft_A_20261115T102637Z/{resolved_config,gradient_proof,adaptation_evidence}.json]

| Selection-validation result | Warm-up macro F1 | Adapted macro F1 | Selected epoch |
|---|---:|---:|---:|
| AST-B baseline | 0.791369 | 0.888652 | 1 |
| AST-A | 0.854139 | 0.996868 | 8 |

The paired A comparison covers 2,500 identical selection recordings: 195 decisions were corrected and 1 became incorrect at 0.5. Fifty selection chunks, including 25 short chunks and all five components, passed CUDA FP16 adapter parity with maximum score difference 0.0. FP32 CPU maximum difference was 0.002621, with zero sampled decision flips at 0.5; that does not establish parity at a future calibrated threshold. [reports/r4/r4_u1_ast_colab_20261115T122234Z.json; linked parity report]

Four cumulative GradScaler skips are recorded in A fine-tuning. The shared Trainer can report Infinity in epoch gradient norms, and its scheduler uses optimizer attempts rather than confirmed updates. These limitations remain visible; the isolated R2-U2 AMP-aware reporting correction did not change the shared Trainer or automatically apply to AST fine-tuning. No claim of eliminated overflow is made. [docs/r2_u2_acceptance.md; docs/r3_u2_ast_finetune.md; A history and adaptation evidence]

## Page 9 contribution — measured compute and limitations

AST-A ran on a Tesla T4 with torch 2.14.1+cu130 and transformers 5.22.1. The saved A fine-tuning history records peak allocated VRAM of 1288.0 MiB; this is a framework-reported allocation, not total GPU occupancy. WavLM heads train on CPU from cached representations, while encoder extraction and parity require separate accounting. [A Colab report; docs/r4_u1_wavlm_head.md]

| Saved history | Recorded epochs | Epoch-loop minutes | Device | Scope |
|---|---:|---:|---|---|
| `reports/r3/r3_u1_B/run_mean/history.jsonl` | 20 | 0.21 | CPU | saved epoch history |
| `reports/r3/r3_u1_B/run_mean_std/history.jsonl` | 20 | 0.28 | CPU | saved epoch history |
| `reports/r4/r4_u1_wavlm_A/run_mean_std/history.jsonl` | 20 | 0.45 | CPU | saved epoch history |
| `runs/r3/r3_u2_20261112T134049Z/r3_u2_20261112T134049Z-baseline/history.jsonl` | 8 | 60.39 | CUDA | saved epoch history |
| `runs/r3/r3_u2_20261112T134049Z/r3_u2_20261112T134049Z-lower_backbone_lr/history.jsonl` | 8 | 60.67 | CUDA | saved epoch history |
| `runs/r4/r4_u1_ast_warmup_A_20261115T065205Z/history.jsonl` | 2 | 19.00 | CUDA | saved epoch history |
| `runs/r4/r4_u1_ast_ft_A_20261115T074952Z/history.jsonl` | 2 | 22.12 | CUDA | partial historical attempt |
| `runs/r4/r4_u1_ast_ft_A_20261115T102637Z/history.jsonl` | 8 | 86.80 | CUDA | saved epoch history |

Epoch-loop timings include validation executed inside the training loop. They exclude unrecorded setup, model downloads, data staging, gradient-proof passes, packaging and later parity; they are neither billed GPU time nor a complete wall-clock total. The interrupted A attempt is shown separately and is not the selected run. R2 extraction, original failed warm-up work, R2-U2 Option B, feasibility runs and other retries also consumed resources; this table is a bounded subtotal and must not be labelled total project compute. A consolidated account of those stages and actual paid costs remains unavailable to this draft. Do not infer zero cost from missing billing evidence.

Single-seed selection estimates can be optimistic. The model/epoch choices use development data; these figures do not replace sealed evaluation, confidence intervals or R5 error analysis. No additional fitting, prediction generation or held-out test access was performed to prepare this draft.

## Review and integration checklist

- Peer reviewer must verify the descriptions against source and the linked evidence; no reviewer approval is implied here.
- Reconcile the compute contribution with Muneeb's page 9 training section and the team's account/time records. Preserve missing measurements as limitations if unavailable.
- Add the draft to the final report's numbering and bibliography; retain the source-bound facts file for audit.
- Final test, calibration, robustness and deployment claims belong to the relevant approved later-stage evidence and must not be inferred from these selection scores.
