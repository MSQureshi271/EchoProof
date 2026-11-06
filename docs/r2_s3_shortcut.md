# R2-S3: shortcut probe (metadata-only, Protocol B)

Code: `src/echoproof/eda/selection.py` (step 1, needs audio), `src/echoproof/eda/shortcut.py` (step 2, no audio), `scripts/r2_s3_shortcut.py`.
Tests: `tests/test_r2_s3_shortcut.py`. Colab: `notebooks/R2_S3_Colab_selection_measure.ipynb`. Plan row: "A metadata-only LR (duration, RMS, silence fraction,
original sample rate) trained on B train and scored on B selection validation. Also compare duration-matched subsets. Done when a memo states whether any
nuisance variable separates the classes."

## Why a second Colab step

R2-S2 measured B train. The probe scores B *selection* validation, so the 2,000 selection-role recordings (genuine, MeloTTS, VITS, XTTS; 25 speakers) need
the same measurements, with the unchanged `eda/measure.py`. Calibration, operating, test and ChatterboxTTS audio is never read.

## Steps

1. **Colab:** run the notebook. It writes `data/eda/r2_s3_B_selection/` and `reports/r2/r2_s3_colab_<timestamp>.json`.
2. **Local:** `python scripts/r2_s3_shortcut.py probe` writes
   - `reports/r2/r2_s3_shortcut_memo.md` (the memo the plan asks for),
   - `reports/r2/r2_s3_shortcut_probe.json` (every number, input hashes, settings),
   - `reports/r2/r2_s3_shortcut_auc.png` / `.svg` (with a two-line interpretation),
   - `data/eda/r2_s3_shortcut/duration_matched_B_v1.json` (the matched recording ids for train and selection).

## Design (fixed in code before the selection results were seen)

| Choice | Value |
|---|---|
| Train / score | B train / B selection validation, recording level |
| Primary model | Logistic regression on duration, RMS, silence fraction, sample rate; standardised with train statistics; class-balanced |
| Constant columns | Dropped and reported (the sample rate is 16 kHz everywhere); a model with nothing left scores a constant 0.5 |
| Wider sets | plus leading/trailing silence; plus level measures (noise floor, active level, peak, clipping, DC); plus high-band shares |
| Single variables | Each measurement alone, direction learned on train |
| Supplementary | Shallow gradient boosting (depth 2, 150 trees) on all measurements, for non-linear shortcuts |
| LR strength | Speaker-grouped 5-fold cross-validation on train only, grid 0.01 to 100 |
| Duration matching | 1 s bins; both classes keep the same count per bin, chosen by a seeded hash |
| Configurations | A all selection; B matched selection with the model trained on all train; C matched selection with the model trained on matched train |
| Uncertainty | Speaker-clustered bootstrap (R1-N1 `paired_group_bootstrap`), 1,000 resamples, seed 42 |
| Verdict | *separates*: lower end of the 95% AUC interval above 0.60; *marginal*: estimate at least 0.60 but lower end not above; otherwise *does not separate* |

## Reading the result

- AUC 0.5 means the variables carry no class information. The 0.60 floor means a metadata-only rule beats a coin by ten points with confidence; it is a
  project choice, not a statistical law.
- "Separates" does not mean the detectors are broken; it means a model could score well on these data without listening to the speech. The memo therefore
  recommends reporting every B model on both the full and the duration-matched selection sets (ids in `duration_matched_B_v1.json`).
- A pass is not proof of absence: speaker, text and other shortcuts are outside this probe.
