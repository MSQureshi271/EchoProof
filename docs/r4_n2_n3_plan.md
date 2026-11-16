# R4-N2 and R4-N3: plan (16 November 2026)

**Status (16 November 2026 update): R4-N2 implementation and tests are complete, but human acceptance checks and sign-off remain outstanding; `docs/r4_n2_app_acceptance.md` is still unticked. R4-N3 implementation and fresh-environment Colab verification are complete. The Colab run passed on Python 3.13.18/Linux, including production B-bundle verification, CLI prediction, and Streamlit AppTest. Evidence: `reports/r4_n3/colab_fresh_install_20261116T140532Z.json`.** Scope from `REVISED_PLAN.md` (R4-N2: deploy the production B bundle in the app and run the app acceptance list; R4-N3: fresh-environment install script and README tested on a clean Colab VM) and `IMPLEMENTATION_PLAN.md` section 12 (G12). Owners in the plan: Nouman / Usman (N2), Nouman / Muneeb (N3).

**## Starting point (checked — 16 November 2026)**

- `app/main.py` now uses the R4-N2 locked production B-bundle loading path. The selected bundle is `bundles/r3_n1_wavlm_head_B_calibrated` (`bundle:r3-n1-wavlm-head-B-calibrated`), verified against `locks/B_lock_gr3.json`.
- R4-N2 implementation includes the production bundle integration, score-type and scope messaging, privacy/resource handling, busy state, accessibility support, and automated app/helper tests.
- Human acceptance remains outstanding: keyboard-only and screen-reader walkthrough, demo-laptop latency measurement, completion of `docs/r4_n2_app_acceptance.md`, and sign-off.
- R4-N3 files are implemented: root `README.md`, `requirements/app.lock.txt`, Windows/Linux setup scripts, checksum-verified bundle retrieval, install verifier, deterministic fixture and expected prediction, reproduction test, and Colab notebook.
- Fresh-environment Colab verification passed on Python 3.13.18/Linux. Production B-bundle verification, CLI prediction, and Streamlit AppTest passed. Evidence: `reports/r4_n3/colab_fresh_install_20261116T140532Z.json`.
- Remaining R4-N3 decisions/checks: formally record the permanent bundle-hosting location and verify the Windows demo-laptop installation.
- `app/main.py` and `app_support.py` are hashed in the B lock; additive edits require a decision-log amendment.

## R4-N2: app deployment and acceptance

**Implementation status: complete. Human acceptance: pending.**

1. **Production bundle by default — implemented.** The app uses the locked production B WavLM bundle at `bundles/r3_n1_wavlm_head_B_calibrated` (`bundle:r3-n1-wavlm-head-B-calibrated`). Bundle verification uses `locks/B_lock_gr3.json`; missing or invalid bundles fail closed. The old R2 LR default is no longer the production app path.
2. **Score-type warning and scope — implemented.** The app identifies the score type, warns when it is not `calibrated_probability`, and presents the scope, retention and "not proof" information. The known operating cost remains: the B WavLM threshold flags 49 of 260 genuine operating recordings.
3. **Privacy and resources — implementation present.** The app processes uploaded audio in memory and does not intentionally persist user audio or filenames. The required human acceptance checks across WAV, FLAC, OGG and MP3, including error and timeout paths, must still be recorded.
4. **Busy state — implemented.** The app provides a busy state and disables the Analyze action while inference is running.
5. **Accessibility — implementation present.** Descriptive labels, text alternatives and text-based outcomes are implemented. The keyboard-only and screen-reader walkthrough remains a human acceptance task.
6. **Parity — automated verification implemented.** The R4-N3 verification successfully exercised the CLI and Streamlit AppTest against the production B bundle and deterministic fixture. The recorded fixture score was `0.9997738974292909`, with score type `calibrated_probability`, threshold `0.4040371283027592`, and assessment `synthetic`.
7. **Automated tests — implemented.** Streamlit AppTest and helper tests are present. The fresh-environment Colab verification passed, including the Streamlit AppTest.
8. **Acceptance checklist — pending human completion.** Complete `docs/r4_n2_app_acceptance.md` with evidence and the person responsible for each check. Record the keyboard-only and screen-reader walkthrough, privacy/resource checks, and measured latency on the demo laptop (Subhan's HP, CPU).
9. **Decision-log recording — verify and close.** Ensure the decision log records the additive changes to `app/main.py` and `app_support.py`, consistent with the B-lock requirements.

**Remaining R4-N2 work:** human accessibility walkthrough, privacy/resource acceptance evidence, demo-laptop latency measurement, completed acceptance checklist, and sign-off. Do not mark R4-N2 accepted until these are recorded.

## R4-N3: fresh install and README

**Implementation status: complete. Fresh-environment Colab verification: passed.**

1. **Pinned environment — implemented and verified.** `requirements/app.lock.txt` records the pinned application environment, including the Transformers version used for bundle verification. The Colab environment ran on Python 3.13.18.
2. **Setup scripts — implemented.** `scripts/setup_env.ps1` supports Windows; `scripts/setup_env.sh` supports Linux and Colab. The Linux/Colab setup completed successfully in the isolated environment.
3. **Checksum-verified bundle retrieval — implemented and verified.** `scripts/fetch_bundles.py` retrieves the production bundle and verifies it against the relevant lock record. The Colab verification confirmed the production B bundle.
4. **Install check — implemented and passed.** `scripts/verify_install.py` successfully loaded the production bundle, ran the CLI fixture prediction, checked the expected result, and ran Streamlit AppTest.
5. **README — implemented.** The root `README.md` documents the project, requirements, setup, bundle retrieval, app and CLI usage, tests, repository structure and evidence.
6. **Clean-VM test — passed.** `notebooks/R4_N3_Colab_Fresh_Install.ipynb` completed fresh-environment verification on Colab. Evidence is stored at `reports/r4_n3/colab_fresh_install_20261116T140532Z.json`. The verified result was:
   - Bundle: `bundles/r3_n1_wavlm_head_B_calibrated`
   - Assessment: `synthetic`
   - Score: `0.9997738974292909`
   - Score type: `calibrated_probability`
   - Streamlit AppTest: passed
7. **Permanent bundle hosting decision — pending.** Formally record the long-term shared bundle location and retrieval arrangement. The successful Colab transfer does not, by itself, settle the permanent hosting decision.

**Remaining R4-N3 work:** formally record the permanent bundle-hosting decision and verify the Windows setup on the demo laptop. The clean-VM Colab verification is complete and must not be listed as outstanding.

## Order and dependencies

1. R4-N2 implementation and automated testing are complete. Proceed with the remaining human acceptance checks, record their evidence in `docs/r4_n2_app_acceptance.md`, verify the decision-log entry, and obtain sign-off.
2. R4-N3 implementation and Colab fresh-environment verification are complete. Record the permanent bundle-hosting decision and perform the Windows demo-laptop installation check.
3. GR4 closes when both locks are available, the app runs the production B bundle, the R4-N2 acceptance checklist is completed and signed off, latency is measured, and the fresh-install and platform checks are satisfied.
4. R5-S1 (corruption copies, Subhan) is independent of N2 and N3 and can run in parallel. R5-N1 follows the loading decision in `docs/r5_bundle_loading_options.md`.

## Risks

- Streamlit upload handling is expected to keep uploaded audio in memory rather than on disk. The required acceptance checks across success, error and timeout paths must be recorded.
- WavLM on a CPU laptop: the B lock recorded warm p95 within 5 seconds on the named laptop; cold start is longer, as disclosed in the lock. The demo-laptop measurement remains outstanding.
- The Transformers pin must match on the demo laptop (`5.22.0`), or the loader warns.
- Permanent bundle hosting has not yet been formally decided. The current Colab transfer arrangement should not be treated as a permanent hosting decision.
- Schedule: the original D10–D11 targets have passed. The remaining work is human acceptance, Windows verification, hosting documentation and GR4 sign-off.