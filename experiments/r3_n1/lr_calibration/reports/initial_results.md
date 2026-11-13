# LR calibration: initial experiment

Exploratory only; no official promotion. The operating role was previously inspected.

| Method | Speaker-held-out log loss | Brier |
|---|---:|---:|
| raw | 0.621644 | 0.194129 |
| temperature | 0.555549 | 0.185641 |
| platt | 0.448425 | 0.151946 |

Platt had lower log loss for 12 of 12 held-out speakers. The 1,000-resample speaker bootstrap interval for the pooled log-loss difference (Platt minus temperature) was [-0.128189, -0.085655]. This interval is descriptive, based on only 12 speakers and fixed out-of-fold predictions; it does not include uncertainty from repeating the full fitting process.

Selected method: **platt**. Operating log loss: 0.592537 raw versus 0.401856 selected; Brier: 0.177082 versus 0.133135.

Platt is monotonic here; the gain concerns scores, not ranking or the attainable threshold trade-off. Confidence may shift with deployment prevalence and unseen generators. Review before any adoption.

## Operating reliability and decision thresholds

Equal-width ECE is 0.20351 raw, 0.21148 temperature, and 0.09357 Platt. See `runs/initial/analysis/operating_reliability.svg` and `reliability_bins.csv` for bin counts. ECE depends on binning and does not establish calibration under deployment distribution shift.

| Method and rule | Threshold | Macro F1 | Genuine FPR | TTS recall |
|---|---:|---:|---:|---:|
| Raw, 0.5 | 0.5000 | 0.7390 | 4.23% | 69.36% |
| Temperature, 0.5 | 0.5000 | 0.7390 | 4.23% | 69.36% |
| Platt, 0.5 | 0.5000 | 0.6584 | 62.69% | 91.28% |
| Raw, operating-selected | 0.2586 | 0.7625 | 22.69% | 81.41% |
| Temperature, operating-selected | 0.3711 | 0.7625 | 22.69% | 81.41% |
| Platt, operating-selected | 0.6461 | 0.7625 | 22.69% | 81.41% |

The operating-selected rules produce identical confusion matrices (TP 635, FP 59, TN 201, FN 145). These thresholds were chosen and measured on the same operating data, so their metrics are optimistic and do not predict independent-test performance. Platt's 0.5 result illustrates why a threshold must not be transferred blindly across calibration methods. TTS recall aggregates known and unseen families; it is not an unseen-generator-specific result.
