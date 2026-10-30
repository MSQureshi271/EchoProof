# P1-13 baseline Colab reproduction review

Review date: 30 October 2026 (Asia/Karachi). Runs use UTC identifiers from 29 October. Guidance: `.agents/data-ai/mlops-engineer.md`.

**Disposition: baseline technical reproduction is evidenced; proceed to P1-14 review and handoff.** Independent M/U team sign-off is still pending. Full G1 and pretrained reproduction remain deferred.

## Evidence reviewed

| Evidence | Finding |
|---|---|
| `reports/phase1/p1_13/export-20261029T191526Z-25ab99cf/` | Failed export: missing `jedi`; report checksum matches |
| `reports/phase1/p1_13/export-20261029T192056Z-073bd927/` | Same missing-dependency failure; report checksum matches |
| `reports/phase1/p1_13/export-20261029T192113Z-93db044e/` | Missing `jedi` repaired, torch version unchanged, dependency check and export passed; report checksum matches |
| `locks/colab-baseline-20261029T192113Z-93db044e/` | All seven manifest-listed export files verify; 70 exact baseline dependency pins |
| `reports/phase1/p1_13/reproduce-20261029T192537Z-816edf2c/` | `ready_for_review`; report checksum matches |
| All ten directories under `runs/phase1/` | All 45 manifest-listed artifacts verify locally |

The selected lock SHA-256 is `d60c8db107fe2a94bc6ec93c493db516ab0401a3465fce38115f0738c6179c8b`. It agrees with the export inventory, reproduction report and all four new probe environment records. The local copies also match all 48 source-file hashes and all 31 historical recovery hashes, including the six historical manifests. The two earlier failed reports remain valid failure history; the successful export supersedes them for reproduction.

## Fresh-runtime results

The report records a different VM boot identity and matching Python/platform. Exact locked packages, FFmpeg and CUDA build match before and after the suite. The only differing package installed during recreation was `jedi==0.20.1`; EchoProof was installed editable without dependency resolution/build isolation. PyTorch's existing matching build was retained and its METADATA/WHEEL fingerprints matched. `pip check`, installed CLI and configuration validation all passed.

| Test module | Passed |
|---|---:|
| P1-04 configuration | 5 |
| P1-05 contracts | 9 |
| P1-06 reproducibility | 5 |
| P1-07 audio | 4 |
| P1-08 baseline models | 4 |
| P1-12 budget/profiling | 6 |
| P1-13 environment | 8 |
| **Total** | **41, without skips** |

Fresh probe runs:

| Run | Evidence |
|---|---|
| `20261029T192718Z-59b38541fdf2` | Storage/readback and equal seeded subprocess outputs; four artifacts verify |
| `20261029T192729Z-84bc3647c15b` | Reproducible audio fixtures and successful WAV/MP3 checks; four artifacts verify |
| `20261029T192738Z-7586c548e409` | LR feature/prediction/reload checks; zero reload difference; six artifacts verify |
| `20261029T192744Z-0198dce4227a` | BiGRU finite gradients/head update, zero padding/reload differences and single/batch difference 5.96e-8; six artifacts verify |

The CUDA generated-tensor check produced `[4, 2]` logits, finite gradients and a successful AdamW head update on Tesla T4. The standalone environment inventory reports deterministic algorithms disabled before seeding in that process; all four seeded probe environment records report them enabled. This is consistent with per-process configuration and is not a failed reproducibility assertion.

## Environment and limits

Observed runtime: Python 3.13.16, torch 2.11.0+cu130, CUDA build 13.0, Tesla T4, driver 580.82.07 and FFmpeg 6.1.1-3ubuntu5. The GPU and driver match the export, with `budget_reprofile_required: false`. The earlier fixture budget remains provisional; audited-data profiling and current storage/work availability still need checking before training.

This run reproduced compatibility with the matching Colab-installed torch build. It did not establish downloading/installing that wheel from an empty environment, its original archive URL/hash, offline installation, interrupted training continuation, research accuracy or pretrained-model readiness. The saved PyTorch source record explicitly identifies its reinstall channel as inferred and the original wheel origin as unavailable.

## Next steps

1. Complete baseline P1-14: prepare the intermediate readiness checklist, central evidence index, setup/recovery handoff and actual S/M/U/N review dispositions. Retain pending pretrained requirements and full G1 separately.
2. Preserve the selected lock, reports and historical runs. Record intentional source/dependency changes as new provenance rather than editing this export.
3. After the baseline handoff, audit the full genuine/transcript/XTTSv2 copy, pairing, provenance, corruption and duplicate groups. Establish leakage-safe grouped partitions before segmentation/augmentation.
4. Reconfirm pilot eligibility, actual cohort sizes, capacity and representative-data throughput before research baseline training.
