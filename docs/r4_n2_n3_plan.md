# R4-N2 and R4-N3: plan (16 November 2026)

**Status (16 November 2026 update): R4-N2 code and tests are implemented (see the decision log); the acceptance checklist `docs/r4_n2_app_acceptance.md` exists with nothing ticked; R4-N3 is not started.** The plan below is the original plan. Scope from `REVISED_PLAN.md` (R4-N2: deploy the production B bundle in the app and run the app acceptance list; R4-N3: fresh-environment install script and README tested on a clean Colab VM) and `IMPLEMENTATION_PLAN.md` section 12 (G12). Owners in the plan: Nouman / Usman (N2), Nouman / Muneeb (N3); nobody has started either.

## Starting point (checked)

- `app/main.py` loads `load_inference_bundle(ECHOPROOF_BUNDLE)`; **the default is still the old R2 LR path** (`runs/r2/r2_m1_20261106T140611Z/model`). The production selection in `locks/B_lock_gr3.json` is the calibrated B WavLM bundle `bundles/r3_n1_wavlm_head_B_calibrated` (`bundle:r3-n1-wavlm-head-B-calibrated`).
- Already present: 25 MiB / 60 s limits and failure messages in `analyze`, the scope sentence `NOT_PROOF`, score-type labelling in `present()`, the mock banner, waveform and log-mel figures, chunk table, JSON download, a spinner. The inference code uses no temporary files (no `tempfile` use found).
- Missing: a root `README.md`, any setup script, a requirements lock for the app, a bundle retrieval step, a busy state, accessibility text for the figures, an explicit uncalibrated-score warning, a retention statement, automated app tests, a CLI-versus-app parity test, an acceptance checklist.
- Bundles are gitignored and live on Drive (WavLM bundle about 380 MB). No download location is defined.
- `app/main.py` and `app_support.py` are hashed in the B lock; additive edits need a decision-log amendment (precedent: 14 November).

## R4-N2: app deployment and acceptance

1. **Production bundle by default.** Default `ECHOPROOF_BUNDLE` to `bundles/r3_n1_wavlm_head_B_calibrated`; at start-up verify `bundle.json` against `locks/B_lock_gr3.json` and fail closed with a clear message on a mismatch or a missing bundle (distinct from invalid user audio). Show model version and score type. No mock path remains in the final demo.
2. **Score-type warning and scope.** Show an explicit warning when the score type is not `calibrated_probability`; show a scope and retention statement before upload (English; genuine versus text-to-speech; audio is processed in memory and not kept; a hosted deployment would send audio to a server); keep the "not proof" wording. Disclose the locked policy's known cost (the B WavLM threshold flags 49 of 260 genuine operating recordings).
3. **Privacy and resources.** Confirm no temporary files are written on success, error or timeout (test with decoders for WAV, FLAC, OGG and MP3, including the MP3 path); no caching of user audio across sessions (`cache_resource` only for the bundle; the per-session result is cleared when another file is chosen or on reset); no logging of audio or filenames.
4. **Busy state.** One active inference per worker; a second request shows a busy and retry message; the Analyze button is disabled while running.
5. **Accessibility.** Descriptive labels, text alternatives for the waveform and log-mel figures, outcome shown as text and not by colour alone, keyboard-only walkthrough of upload, analyze, error and download.
6. **Parity.** The same fixture bytes through the library `analyze`, `echoproof predict` and the app code path give identical score, type, threshold and decision; any divergence is a release blocker (`REVISED_PLAN.md` risk list).
7. **Automated tests.** Streamlit's `AppTest` for: start-up verification, upload then analyze, each failure message, busy state, result clearing; plus unit tests for the new `app_support` helpers. Keep the existing `tests/test_r1_n2_inference.py` green.
8. **Acceptance checklist.** `docs/r4_n2_app_acceptance.md`: one row per G12 item and per section 12 flow step, each with evidence (test name or measured value) and a person who ticked it. Measured latency: reuse the B lock's local figures and re-measure on the demo laptop (Subhan's HP, CPU).
9. **Recording.** Decision-log entry listing the additive edits to `app/main.py` and `app_support.py`; no change to `inference.py`, `cli.py` or any bundle.

Needs a human: the keyboard and screen-reader walkthrough, the demo-laptop latency run, and sign-off.

## R4-N3: fresh install and README

1. **Pinned environment.** `requirements/app.lock.txt` from the working environment (extras `app`, `neural`, `pretrained`, `analysis`; transformers pinned to 5.22.0 as the bundles were verified with it), with Python version noted. Reuse the P1-13 `locks/colab-baseline-*` export for the Colab case.
2. **Setup script.** `scripts/setup_env.ps1` (Windows, the demo laptop) and `scripts/setup_env.sh` (Linux and Colab): create a virtual environment, install the lock, install the package editable with no dependencies, check imports and versions, print what failed.
3. **Checksum-verified bundle retrieval.** `scripts/fetch_bundles.py --source <folder|zip> [--all]`: copies the production bundle (and optionally all eight) from a transfer location and verifies every component against `locks/B_lock_gr3.json` and `locks/A_lock.json`; refuses on any mismatch; never downloads anything unpinned (the WavLM and AST weights are inside the bundles, so no model download at run time).
4. **Install check.** `scripts/verify_install.py`: load the production bundle, run `echoproof predict` on a small bundled fixture clip, compare with a stored expected score and decision, start the app headless with `AppTest`, exit non-zero on any failure.
5. **README.** Root `README.md`: what the project does, scope and limitations, requirements, install (Windows and Colab), fetching bundles, running the app, CLI `predict` example, running the tests, repository map, what is locked and where the evidence is, how to cite the locks.
6. **Clean-VM test.** `notebooks/R4_N3_Colab_Fresh_Install.ipynb`: fresh Colab runtime, clone or sync the repository, run setup, fetch, verify; output kept as evidence. Muneeb's independent reproduction (R6-REPRO) reuses it.
7. **Decision needed:** where the bundles are hosted for retrieval (a shared Drive folder with a checksum manifest is the lowest-effort choice; a release asset or Git LFS needs an account decision).

Needs a human: the clean-VM run, the hosting decision, the demo-laptop install.

## Order and dependencies

1. R4-N2 items 1, 2, 4, 6, 7 (code and tests, can start now), then 3 and 5, then the acceptance document.
2. R4-N3 items 1 to 5 in parallel with the late N2 items (the install check needs the final app), then the clean-VM test.
3. GR4 closes when: both locks exist (done), the app runs the production B bundle with the acceptance list ticked, latency is measured, and the fresh install works.
4. R5-S1 (corruption copies, Subhan) is independent of N2 and N3 and can run in parallel; R5-N1 follows the loading decision in `docs/r5_bundle_loading_options.md`.

## Risks

- Streamlit's upload handling keeps files in memory, not on disk, but the claim is tested, not assumed.
- WavLM on a CPU laptop: the B lock recorded warm p95 within 5 s on the named laptop; cold start is longer (disclosed in the lock).
- The transformers pin must match on the demo laptop (5.22.0), or the loader warns.
- Schedule: by the plan these were due D10 to D11; today is D13, so the clean-VM test and sealed run compete for the last days.
