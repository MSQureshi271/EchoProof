# R2-S2: training-partition EDA (Protocol B train only)

Code: `src/echoproof/eda/measure.py` (step 1, needs audio), `src/echoproof/eda/figures.py` (step 2, no audio), `scripts/r2_s2_eda.py` (command line).
Tests: `tests/test_r2_s2_eda.py`. Colab: `notebooks/R2_S2_Colab_eda.ipynb`. Plan row: `REVISED_PLAN.md`, R2-S2 ("at least 6 figure families in `reports/eda/`").

## Scope

B train only: the 4,530 retained genuine, MeloTTS and VITS recordings of the 151 training speakers (the R2-S1 plan, plan hash `adfca8c9…`).
No validation, test or ChatterboxTTS audio is read. Partition totals in figure 01 come from `split_report.json` (counts only).

## Two steps

1. **Measure (Colab; locally 0.06 s per recording, so a few minutes of CPU plus shard reading):** `notebooks/R2_S2_Colab_eda.ipynb` reads the 14 shards that hold training audio, hash-checks every recording against the
   manifest, and writes to `data/eda/r2_s2_B_train/`: `recording_stats_B_train_v1.parquet`, `mean_logmel_B_train_v1.npz`, `examples_B_train_v1.npz`,
   `r2_s2_manifest.json`, plus the evidence report `reports/r2/r2_s2_colab_<timestamp>.json`.
2. **Figures (local):** `python scripts/r2_s2_eda.py figures` verifies the measurement files against the manifest hashes and writes `reports/eda/`.

## Measurement definitions (frozen in `measure.py`)

| Column | Meaning |
|---|---|
| `duration_s`, `n_samples` | From the audio; must equal the cohort manifest |
| `rms_dbfs`, `peak`, `clip_fraction`, `dc_offset` | Whole-recording level, largest sample, share of samples at 0.999 or more, mean |
| active frame | 20 ms frames every 10 ms; a frame is active above a per-recording two-class (Otsu) energy threshold, if the classes are at least 10 dB apart. Otherwise all frames are active; if the loudest frame is at or below -70 dBFS none is |
| `leading_silence_s`, `trailing_silence_s` | Time before the first / after the last active frame |
| `silence_fraction` | Share of non-active (low-energy) frames, pauses inside the recording included |
| `noise_floor_dbfs`, `active_level_dbfs` | 10th percentile of frame levels; mean level of active frames |
| `hf_ratio_4k`, `hf_ratio_7k` | Share of STFT power (512-point Hann, hop 160) at or above 4 kHz / 7 kHz |
| mean log-mel | Mean over all frames of the 64-bin log-mel used everywhere else (`features.acoustic.log_mel`) |

A fixed offset below the peak (35 dB) was tried first and found no pauses in noisy real LibriSpeech files, so the per-recording threshold replaced it before any run.

## Figure families (each saved as PNG and SVG, with a two-line interpretation printed on it and listed in `interpretations.json`)

| File | Content |
|---|---|
| `01_class_generator_counts` | Recordings by generator in B train; class totals per partition |
| `02_duration_histograms` | Duration by generator, share of recordings of 4 s or less |
| `03_paired_waveform_logmel_<stem>` (3 files) | The same sentence as genuine, MeloTTS and VITS: waveforms and log-mels on one amplitude and one colour scale |
| `04_level_and_silence` | RMS, noise floor, low-energy share, leading and trailing silence, clipping |
| `05_spectral_balance` | Average log-mel spectrum per generator, high-band energy share |
| `06_mfcc_distributions` | MFCC 1-12 means per generator over the R2-S1 crop bank |
| `07_format_audit` | Container, codec, sample rate, channels (from header inspection) |
| `08_speaker_group_sizes` | Admitted utterances per training speaker vs recordings the cohort uses |
| `09_exclusion_table` (+ `exclusions_table.csv`) | Admission, materialisation, repair and B-train exclusions |

The numbers in each interpretation are computed from the data, with the genuine-vs-synthetic AUC of a single column as the common indicator
(0.5 = no separation; folded so direction does not matter). These indicators are descriptive; the shortcut test itself is R2-S3.

## Guarantees

- The measurement refuses to overwrite finished outputs and resumes by skipping part files whose plan hash still matches.
- The figure step refuses outputs whose SHA-256 differs from `r2_s2_manifest.json`, and fails if any component other than genuine, melo, vits is present.
- SVG and CSV outputs are byte-identical across reruns (tested).
