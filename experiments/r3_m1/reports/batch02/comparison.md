# Batch 02: hidden-size-64 comparison

| Candidate | Hidden size | Best epoch | Full macro F1 | XTTS recall | Genuine FPR | XTTS AUC |
|---|---:|---:|---:|---:|---:|---:|
| original_baseline | 128 | 3 | 0.859754 | 0.926000 | 0.310000 | 0.884704 |
| original_lower_lr | 128 | 10 | 0.861128 | 0.676000 | 0.118000 | 0.883184 |
| b01_baseline_full20 | 128 | 7 | 0.867587 | 0.702000 | 0.120000 | 0.882236 |
| b01_lower_lr_full20 | 128 | 16 | 0.869120 | 0.664000 | 0.084000 | 0.904064 |
| b02_hidden64_full20 | 64 | 10 | 0.838414 | 0.604000 | 0.126000 | 0.841564 |

Experimental preference: b01_lower_lr_full20. Advisory only; team review pending.

Isolated post-lock, result-informed single-seed diagnostic. Advisory only; no automatic promotion, calibration/operating evaluation, statistical superiority, or unseen-generator guarantee.

Choice follows full selection macro F1 at 0.5. XTTS diagnostics do not override that rule.

Hidden64 F1 difference from the batch01 selected checkpoint: -0.030706.
