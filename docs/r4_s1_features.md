# R4-S1: Protocol A mel statistics, crop bank and LR feature tables

Code: `src/echoproof/features/r4_s1.py` (plan, shard jobs, verification, outputs, cross-check, WavLM gap list), `scripts/r4_s1_features.py` (command line).
Tests: `tests/test_r4_s1_features.py`. Colab: `notebooks/R4_S1_Colab_features.ipynb`. Evidence: `reports/r4/r4_s1_colab_20261114T212857Z.json`.
Outputs: `data/features/r4_s1_A/` (Drive only, not in git).

Owner: Subhan. Reviewer: Muneeb (review is user-reported, see the decision log). Run on 14 November 2026.

## Why a separate module

`src/echoproof/features/r2_s1.py` is Protocol B only and its source hash is recorded in the GR3 lock (`locks/B_lock_gr3.json`), so it was not edited.
`r4_s1.py` reuses the protocol-independent parts (part reading, merging, the plan object, the spot check, `stats.py`) and restates the shard processing and job runner
with `crop:A:` ids. B outputs were not written to or changed.

## Scope

| Item | Decision |
|---|---|
| Crops per training recording | K = 5, seed 42; starts depend only on (seed, recording id), so crops of recordings shared with B are the same as B's |
| Mel statistics | Every valid frame of every retained A-train recording, frame-equal weights, population std |
| LR features | The crops of every A-train recording and every manifest chunk of every A validation recording |
| Not computed | Any A test row (sealed) |

All five components (genuine, Melo, VITS, XTTS, Chatterbox) are in A train and validation. The A split hash is `a9075367…`, checked against `reports/r1/split_report.json`.

## Results (Colab run, 2 workers, 3,910 s)

| Quantity | Value |
|---|---|
| Train recordings / crops | 7,550 (1,510 per component) / **35,782** |
| Validation recordings / chunks | 5,000 (1,000 per component) / 24,412 (selection 12,331, calibration 5,777, operating 6,304) |
| Test rows, unretained recordings | 0, 0 |
| Mel frames | 8,152,374 over 7,550 recordings (B: 5,059,633) |
| Plan sha256 | `b92611b216ebf20a5b59e9fb3ca50fc60bd395eeb1027c3c6498e5c32ef53fb8` |
| Spot check | 50 rows recomputed, 0 mismatches; streaming vs two-pass mel max difference 1.8e-15 |
| NaN | only `voiced_f0_mean` and `voiced_f0_std`, 9 train crops each (no voiced frames); same imputation rule as B |
| Tests on Colab | 52 passed (A and B suites) |

The plan estimated 37,750 crops. That was wrong: recordings of about 4 s or less give fewer than five crops, as in B. 35,782 is the correct count (the local preflight on the manifests gave the same number before the run).

### Cross-check against B

For the recordings both protocols share, every B row (21,470 train crops, 19,876 validation chunks) was found in A with **maximum absolute difference 0.0** and identical NaN patterns
(tolerance 1e-9). The Colab environment equalled the B build (librosa 0.11.0, numpy 2.4.6, Python 3.13.18), which is why the match is exact.

### Mel statistics differ from B by design

Adding XTTS and Chatterbox to training moves the statistics: maximum absolute shift of the per-bin mean 1.316, of the std 0.507 (14.1% relative). LR-A, BiGRU-A and any A model must use `mel_stats_A_v1.npz`, never B's.

## Outputs (sha256 verified on this machine against the Colab report)

| File | sha256 |
|---|---|
| `crop_bank_A_v1.parquet` | `a52c6cee86fa8d7ba3be4061179a9211909ac2399f96173ee01221b9a27f3d1d` |
| `mel_stats_A_v1.npz` | `e3d4f2f069212a4cd7e3dc1234538a410d64bdea398cd6f8ab5de3f3982cc56b` |
| `lr_train_crops_A_v1.npz` | `277c378b500eeff5441009f059d5533c83d64382159a569ef82d83b0afefb912` |
| `lr_validation_A_v1.npz` | `8b38d45e0ca1849db060167dc2d40bd2093cd2ee79d8833561a59c04827ca87c` |
| `wavlm_cache_gap_A_v1.json` | `0a48919355fb3b4ca9265638eb5868717046f303e7c91fd5c523b361aad33f17` |
| `r4_s1_vs_b_crosscheck.json` | `d51641dc529904fb73656ba309cf90bae4488a79c9d5539c29a350412cb9953e` |
| `r4_s1_manifest.json` | `c9eb82f61e3a98e177758856706480688cbfdca0e8cbe300eb1b97837775c70f` |

Inputs: cohort file `51af27a2…`, chunks `d3f8b8d6…`, A split file `6b3a5649…`, preprocessing `879a76da…`, feature schema `fce2751a…`.
The manifest is not listed in its own `outputs`; its hash is in `r4_s1_side_outputs.json`.

## Hand-off

- **Usman, R4-U0:** `wavlm_cache_gap_A_v1.json` lists the 18,848 rows the B WavLM cache lacks: 7,222 Chatterbox train crops, 7,090 XTTS train crops and 4,536 Chatterbox validation chunks. No test rows. XTTS validation chunks are already cached by B.
- **Muneeb, R4-M1:** LR-A and BiGRU-A use `lr_train_crops_A_v1.npz`, `lr_validation_A_v1.npz`, `crop_bank_A_v1.parquet` and `mel_stats_A_v1.npz`.

## Limits

- The Colab build ran with `--resume` (the manifest says `resumed: true`); every shard is recorded and verified.
- The `parts/` directory in the output folder holds intermediate per-shard files; the final outputs above are what downstream code should read.
- Review by Muneeb is user-reported, not independently evidenced here.
- The R4-S1 schedule slip (R4 was planned for D9 to D11) is recorded in the decision log.
