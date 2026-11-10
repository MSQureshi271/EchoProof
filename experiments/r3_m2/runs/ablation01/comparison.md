# R3-M2 LR feature-group ablation

| Model | Macro F1 | FPR | XTTS recall | XTTS AUC |
|---|---:|---:|---:|---:|
| saved_R2_M1 | 0.692348 | 0.164000 | 0.142000 | 0.526104 |
| full67_control | 0.692348 | 0.164000 | 0.142000 | 0.526104 |
| without_energy_activity60 | 0.710667 | 0.266000 | 0.436000 | 0.672560 |

Advisory only. No official model changed. Group removal does not eliminate all energy/silence cues.
Fixed C=0.01; no feature-set-specific tuning, calibration/operating scoring or test access. Subhan review pending.
