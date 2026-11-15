# R4-N1 AST-A calibration candidate

Scope: AST only. The owner reports the supplied R4-N1 procedure fully approved; this records that authorization, not an independent reviewer signature. No LR, BiGRU or WavLM calibration, B source change, shared inference routing change or A lock is included.

## Method and reproducibility

Recipe `configs/r4_n1_ast_calibration_a.json` was saved before fitting. It copies the B WavLM numerical settings: clipping epsilon 1e-7; temperature bounds 0.01–50; positive-slope affine with u bounds -4–4, intercept bounds -5–5 and regularisation 0.01. This differs from the previous AST-B Platt-only exception: A uses the newly approved identity/temperature/affine LOSO procedure, equally for all models when their owners implement it.

`scripts/r4_n1_ast_calibrate.py` invokes the A-only evaluation module. It verifies input and export hashes, canonical role membership, labels, components, speaker role separation, finite scores and chunk-to-recording aggregation; test assignments are withheld by the existing split loader. Calibration LOSO uses per-speaker NLL and the simplest fit within one standard error of the best. Failed/bound-hitting fits are excluded. The fit is written before operating scores are loaded. Operating acceptance requires strict log-loss improvement and non-worsening Brier. The retained scores feed the existing contract threshold selector. Outputs use exclusive creation; preserve failed/incomplete directories rather than reusing them.

Selected upstream model: `runs/r4/r4_u1_ast_ft_A_20261115T102637Z`, epoch 8. Source bundle `bundles/r4_u1_ast_partial_A/`. New candidate run: `runs/r4/r4_n1_ast/candidate_20261115_approved_recipe/`; new bundle: `bundles/r4_n1_ast_A_candidate_20261115/`. The model components are copied and hash-verified; model weights are unchanged. The explicit A loader performs mean chunk probability, then the accepted recording transform, then the saved decision rule. It refuses threshold overrides and non-reference precision.

## Actual CPU result

LOSO selected temperature. Full calibration fit: **T = 0.292069762645965**. Operating log loss: **0.01327159736 → 0.01195557158**; Brier: **0.00348113840 → 0.00310454706**; 10-bin equal-width ECE: **0.00707028816 → 0.00346242496**. The predeclared candidate acceptance rule passed. This is not formal reviewer acceptance or promotion.

Operating threshold: **score >= 0.9702610119277397** on calibrated probabilities; macro F1 **0.99639942058**; genuine FPR **0.00384615385** (one of 260 genuine recordings). Only one observed score-induced partition ties for the best F1. This count is not the continuous width of a threshold interval and should not imply uniquely determined population performance.

The run contains frozen prospective settings, LOSO folds/fits/exclusions, operating metrics, per-component panels, per-speaker false alarms, 1,000-resample speaker bootstrap, reliability bins and SVG, operating scores/decisions and local verification. Bootstrap intervals are descriptive and do not include refitting uncertainty. The Brier-difference interval includes zero; passing the declared point-estimate rule is not statistical confirmation. Development roles have already been viewed in earlier model evaluation; this is not independent test evidence.

## Verification and recipient instructions

Seven focused unittest cases passed, including an end-to-end runner fixture covering all five components and full-recording aggregation. All 1,300 deployed operating scores and decisions were recomputed exactly from the bundle's saved calibration metadata and compared with the saved decisions. Source syntax and notebook code cells were checked. Broader pytest suites were not run because pytest is unavailable locally. Torch/transformers are unavailable locally; actual CUDA FP16 parity was executed manually on Colab and its receipt inspected locally.

Recipient receipt: `reports/r4/r4_n1_ast_parity/1b6e0a5518c741319e6f1df99bc46367.json`, SHA-256 `3e0fd2efe027eabf7f97be5f6123098ff0158280c3c13a12b823173aa4df4883`. It passed on 163 chunks covering all chunks of 50 selection recordings, all five components; maximum raw score difference 4.034882294945419e-7; zero recording decision flips. Bundle, calibration and runner hashes match the local candidate. This is sampled parity, not exhaustive inference verification or formal approval.

Transfer both `configs/r4_n1_ast_*.json` files; `src/echoproof/evaluation/r4_n1_ast.py`; `src/echoproof/models/r4_n1_ast_bundle.py`; both `scripts/r4_n1_ast_*.py`; the tests/notebook; the entire new candidate run; and the entire candidate bundle to matching paths in the chosen Drive project workspace. Existing selected AST-A exports, canonical audio shards and R4-U1 AST data module must be available there. No need to upload another dataset copy.

Open `notebooks/R4_N1_AST_Colab_Parity.ipynb`, set PROJECT_ROOT to that execution workspace and SHARD_DIR to its canonical shards, use a T4, then run cells. It verifies source hashes from the parity-source manifest and writes a uniquely named report. Python source may match exact bytes or a pinned LF equivalent; notebook executable cells are hashed without mutable outputs/metadata. Source edits fail verification. The A sampler includes Chatterbox; the B sampler is not used. It selects 50 stratified selection chunks and includes every chunk of their recordings before comparing calibrated recording decisions. Any flips are listed and require review.

Remaining: broader software checks in the supported environment, independent review, and coordination with the other three A-model owners before any separate A lock. Generated runs, bundles and receipts remain ignored and are shared through Drive; this document records their paths and the inspected receipt identity without copying ignored outputs into Git. No automatic promotion or lock write.

## Local sync check (16 November 2026, added by the project owner's session)

- `runs/r4/r4_n1_ast/candidate_20261115_approved_recipe` is present and complete; the recipe's input hashes (AST evidence and selection, source bundle, split, cohort and chunk files, R4-S1 manifest, B lock after line-ending normalisation) match the local files.
- The parity receipt `reports/r4/r4_n1_ast_parity/1b6e0a5518c741319e6f1df99bc46367.json` is present and its sha256 equals `3e0fd2ef...`; it records `approved: false`, 163 chunks, no flips, largest raw difference 4.0e-7.
- Operating figures recomputed from the saved files: temperature T 0.2921, threshold 0.9703, macro F1 0.9964, 1 genuine false alarm (`speaker:6476`) and 2 missed TTS recordings of 1,300; VITS is the weakest generator panel (F1 0.9942). The Brier-difference bootstrap interval includes zero.
- **Not present locally:** `bundles/r4_n1_ast_A_candidate_20261115`. Its hashes in the receipt cannot be rechecked until it is synced.
- `tests/test_r4_n1_ast.py` and `tests/test_r4_n1_ast_parity_runner.py` pass locally under pytest (the document records only unittest). See `docs/r4_n1_calibration_summary_a.md`.
