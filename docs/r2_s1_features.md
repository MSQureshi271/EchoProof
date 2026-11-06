# R2-S1: Protocol B mel statistics, crop bank and LR feature tables

Code: `src/echoproof/features/stats.py` (crop bank, streaming statistics), `src/echoproof/features/r2_s1.py` (plan, shard jobs, merge,
verification, outputs), `scripts/r2_s1_features.py` (command line). Tests: `tests/test_r2_s1_features.py`. Colab: `notebooks/R2_S1_Colab_features.ipynb`.

## Scope (sign-off of 5 November 2026)

| Item | Decision |
|---|---|
| Crops per training recording | **K = 5**, seed 42, drawn once and stored as `crop_bank_B_v1.parquet` |
| Mel statistics | **Every valid frame of every retained B-train recording**, full length (not the crops). Frame-equal weights, population std |
| LR features | The 5 crops of every B-train recording, and every manifest chunk of every B validation recording (selection, calibration, operating) |
| Not computed | B test chunks (kept for the sealed evaluation, R5), anything from Protocol A, any ChatterboxTTS recording |

Expected counts, derived locally from the locked manifests (the Colab run must reproduce them): 4,530 train recordings
(1,510 each of genuine, Melo, VITS), 21,470 crops (4,235 recordings give 5 crops; 295 recordings of 4 s or less give one), 4,000 validation
recordings, 19,876 validation chunks (selection 10,029, calibration 4,712, operating 5,135).

## Rules the code enforces (and the tests break on purpose)

- Test-partition assignments are dropped before planning, and `plan_b` rejects any Chatterbox recording in B train or validation, and XTTS in train.
- Validation chunks must equal the canonical windows of the recording's audio length; group ids must match the stem's speaker.
- Each recording is hash-checked against the manifest (`pcm_sha256`) when it is read from its shard.
- The protocol B split hash must equal the value frozen in `reports/r1/split_report.json`.
- After merging, `verify` re-checks components, speaker disjointness, classes, finite values, NaN only in the two F0 columns, the mel frame total
  implied by the manifest, and chunk counts per role. A random spot check recomputes rows from the audio and compares exactly.
- Finished outputs are never overwritten; an interrupted run resumes by skipping part files whose plan hash still matches.

## Using the outputs

- `load_mel_stats(path)` returns float32 `mean` and `std` (64 values). `normalize_log_mel(mel, mean, std)` applies them; call it on valid frames only.
- `lr_train_crops_B_v1.npz`: `X` [n, 67], `y`, `recording_id`, `group_id`, `component`, `crop_id`, `start`, `valid_samples`, `feature_names`.
  Pass `X`, `y`, `recording_id` to `run_lr_grid` as the train set (recording-equal weights are applied there).
- `lr_validation_B_v1.npz`: the same plus `chunk_id`, `role`, `end`. Filter `role == "selection"` for model selection.
- NaN appears only in `voiced_f0_mean` and `voiced_f0_std` (clips with no voiced frames); the LR imputer fills them with the training median.
- R2-U1 (WavLM cache) should read `crop_bank_B_v1.parquet` so both models see the same crops.

## Not covered

- Protocol A statistics and features (R4-S1, after GR3).
- Time per feature extraction on Colab is measured by the run, not assumed (R1-M2 measured 45.2 ms per 4 s chunk on 2 CPUs).
