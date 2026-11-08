# Batch 01 comparison

| Candidate | Best epoch | Macro F1 | XTTS F1 | FPR |
|---|---:|---:|---:|---:|
| original_baseline | 3 | 0.859754 | 0.805289 | 0.310000 |
| original_lower_lr | 10 | 0.861128 | 0.776630 | 0.118000 |
| b01_baseline_full20 | 7 | 0.867587 | 0.789331 | 0.120000 |
| b01_lower_lr_full20 | 16 | 0.869120 | 0.786612 | 0.084000 |

Selected: b01_lower_lr_full20. Independent review pending.

Result-informed single-seed selection; no statistical superiority or global optimum claimed.

b01_baseline_full20: hypothetical patience-3 stop 6; gain from full 20 epochs 0.007833.

b01_lower_lr_full20: hypothetical patience-3 stop 13; gain from full 20 epochs 0.007992.
