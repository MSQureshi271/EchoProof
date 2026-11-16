# R4-N2: app acceptance checklist

**Purpose.** This checklist is the acceptance list for R4-N2 (deploy the production B bundle in the Streamlit app). It holds one row per requirement from `IMPLEMENTATION_PLAN.md` section 12 (user-flow steps, resource limits and privacy, exit gate G12), from the GR4 items in `REVISED_PLAN.md`, and from `docs/r4_n2_n3_plan.md`, plus the items that need a person.

**Status: open. Nothing is ticked.** Every row starts with Status "open" and blank evidence, and no row has been checked against the app.

**Who ticks.** The Owner column is "to assign" until someone takes the row. A row is ticked by the person who checked its evidence, who writes their name and the date in "Ticked by / date". Ticking a row means the evidence exists and was read, not that the work is planned.

**How evidence is cited.** Each row needs one of: a named automated test (the test file and test name), a measured value (the value, its unit, the machine, and the command that produced it), or a manual check (who did it, when, and what was observed). Put that reference in "Evidence reference", using a repository path where one exists. A row that cannot be met stays "open" and the gap is written in "Evidence reference" rather than left blank.

**Sources.** `IMPLEMENTATION_PLAN.md` section 12 (lines 495 to 548). `REVISED_PLAN.md` R4-N2, R4-N3 and GR4. `docs/r4_n2_n3_plan.md`. `locks/B_lock_gr3.json` and `locks/A_lock.json` for the locked bundles.

## A. User-flow steps (IMPLEMENTATION_PLAN.md section 12)

| ID | Requirement | Source | Evidence needed | Owner | Status | Evidence reference | Ticked by / date |
|---|---|---|---|---|---|---|---|
| A1 | Show a brief supported-scope and retention statement and the file restrictions before upload. | IMPLEMENTATION_PLAN.md section 12, user flow step 1; docs/r4_n2_n3_plan.md R4-N2 item 2 | Automated test that the scope and retention text and the file restrictions render before the uploader, named by test. | to assign | open |  |  |
| A2 | Accept WAV and MP3 upload, optionally record after explicit browser microphone permission, and validate decoded content even when a component requests 16 kHz. | IMPLEMENTATION_PLAN.md section 12, user flow step 2 | Automated tests for WAV and MP3 upload and for 16 kHz validation, named by test; manual check that recording does not start before permission is granted. | to assign | open |  |  |
| A3 | Show duration, playback, waveform and log-mel spectrogram with labelled axes, a readable colour scale and text alternatives, plus a clear Analyze action and a progress state. | IMPLEMENTATION_PLAN.md section 12, user flow step 3 | Automated test that the duration, playback, both figures and the Analyze control render; manual check of axis labels and colour scale on the figures. | to assign | open |  |  |
| A4 | Return understandable messages for corrupt, silent, insufficient, too-large, too-long, unsupported and timed-out audio, and show a model loading failure as distinct from invalid user audio. | IMPLEMENTATION_PLAN.md section 12, user flow step 4 | One named automated test per message class, plus a test that a missing or mismatched bundle produces a different message from invalid audio. | to assign | open |  |  |
| A5 | Show "Likely genuine speech" or "Likely TTS-generated speech", the score type, calibrated confidence only when supported, warnings and model version, with inconclusive following the saved policy and not UI thresholds. | IMPLEMENTATION_PLAN.md section 12, user flow step 5; docs/r4_n2_n3_plan.md R4-N2 item 2 | Automated test on a fixture result showing the label, score type and model version; test that calibrated confidence is hidden when the score type is not calibrated; comparison of the displayed label with the locked decision-policy output for the same fixture (values recorded). | to assign | open |  |  |
| A6 | Offer a JSON and text summary download containing the result, scope, model version, coverage and settings, with no raw audio in the report. | IMPLEMENTATION_PLAN.md section 12, user flow step 6 | Automated test that the downloaded JSON has the listed fields and contains no audio bytes; the same check for the text summary. | to assign | open |  |  |
| A7 | Explain that visuals summarise the signal and that assessment is limited to validated conditions, avoiding accusations, identity claims or certainty of authenticity. | IMPLEMENTATION_PLAN.md section 12, user flow step 7 | Manual check of the rendered wording against the step 7 requirement, with the page text recorded. | to assign | open |  |  |

## B. Resource limits and privacy (IMPLEMENTATION_PLAN.md section 12)

| ID | Requirement | Source | Evidence needed | Owner | Status | Evidence reference | Ticked by / date |
|---|---|---|---|---|---|---|---|
| B1 | Cap uploads at 25 MiB. | IMPLEMENTATION_PLAN.md section 12, resource limits; docs/r4_n2_n3_plan.md starting point | Named test: a file of exactly 25 MiB is accepted and one byte more gets the too-large message. | to assign | open |  |  |
| B2 | Cap decoded duration at 60 seconds. | IMPLEMENTATION_PLAN.md section 12, resource limits | Named test: a 60 s fixture is accepted and a 60.1 s fixture gets the too-long message. | to assign | open |  |  |
| B3 | Bound sample count, channels, decoder runtime and memory independently of file headers. | IMPLEMENTATION_PLAN.md section 12, resource limits | Named test with a header that understates the size of the decoded content, showing the bound is enforced; measured decoder runtime and peak memory on the bounded fixture (values, machine and command recorded). | to assign | open |  |  |
| B4 | Test expansion attacks, malformed files and misleading extensions using benign fixtures. | IMPLEMENTATION_PLAN.md section 12, resource limits | Fixture list with file names, expected message per fixture, and the named test run result for each. | to assign | open |  |  |
| B5 | Invoke decoders with fixed argument arrays, never user-controlled shell commands. | IMPLEMENTATION_PLAN.md section 12, resource limits | Code review of every decoder call site (file and line recorded) plus a test or search showing no shell string and no user-supplied value in an argument list. | to assign | open |  |  |
| B6 | Do not accept remote audio URLs in the MVP. | IMPLEMENTATION_PLAN.md section 12, resource limits | Named test that a URL input is rejected, and a search of the app code showing no URL fetch. | to assign | open |  |  |
| B7 | Use generated temporary names in an isolated temporary directory and leave no temporary files after success, error or timeout, covering WAV, FLAC, OGG and MP3 (including the MP3 path). | IMPLEMENTATION_PLAN.md section 12, privacy; docs/r4_n2_n3_plan.md R4-N2 item 3 | Named tests per format and per outcome (success, error, timeout) that list the temporary directory before and after the run and find nothing left over. | to assign | open |  |  |
| B8 | Do not cache user audio or results across sessions; `cache_resource` is used only for the model bundle, and the per-session result is cleared when another file is chosen or on reset. | IMPLEMENTATION_PLAN.md section 12, privacy; docs/r4_n2_n3_plan.md R4-N2 item 3 | Named test that a second session does not see the first session's result; code review of every cache decorator (file and line recorded) showing only the bundle is cached; named test for clearing on new file and on reset. | to assign | open |  |  |
| B9 | Do not log audio contents or sensitive filenames. | IMPLEMENTATION_PLAN.md section 12, privacy | Named test that captures log output during an analysis and finds no audio bytes and no filename; code review of logging calls (file and line recorded). | to assign | open |  |  |
| B10 | Allow one active inference per worker; a second request shows a busy and retry message, and the Analyze button is disabled while a run is in progress. | IMPLEMENTATION_PLAN.md section 12, concurrency; docs/r4_n2_n3_plan.md R4-N2 item 4 | Named busy-state test; manual check that the Analyze button is disabled during a run. | to assign | open |  |  |

## C. Accessibility (IMPLEMENTATION_PLAN.md section 12)

| ID | Requirement | Source | Evidence needed | Owner | Status | Evidence reference | Ticked by / date |
|---|---|---|---|---|---|---|---|
| C1 | Keyboard navigation reaches and operates every control (upload, record, Analyze, result, download). | IMPLEMENTATION_PLAN.md section 12, accessibility paragraph; docs/r4_n2_n3_plan.md R4-N2 item 5 | Manual keyboard-only check, recorded with the date and the controls tested; the person-run walkthrough is row E1. | to assign | open |  |  |
| C2 | Every control and input has a descriptive label. | IMPLEMENTATION_PLAN.md section 12, accessibility paragraph; docs/r4_n2_n3_plan.md R4-N2 item 5 | Automated check of label text for each control, or a recorded list from the page; manual review of the labels. | to assign | open |  |  |
| C3 | Loading and error states are visible: a progress state during analysis and a visible error message on failure. | IMPLEMENTATION_PLAN.md section 12, accessibility paragraph and user flow step 3 | Named test that the progress element appears during a run and the error element appears on a failing fixture; manual visual check. | to assign | open |  |  |
| C4 | Charts are legible: labelled axes, a readable colour scale, and text alternatives for the waveform and log-mel figures. | IMPLEMENTATION_PLAN.md section 12, accessibility paragraph; docs/r4_n2_n3_plan.md R4-N2 item 5 | Manual check of the axis labels and colour scale on both figures; named test that the text alternatives are present for each figure. | to assign | open |  |  |
| C5 | Outcomes and warnings are shown as text, not by colour alone. | IMPLEMENTATION_PLAN.md section 12, accessibility paragraph; docs/r4_n2_n3_plan.md R4-N2 item 5 | Named test that the outcome and warning text is present in the result; manual check that the outcome is readable with colour removed. | to assign | open |  |  |

## D. Gate G12 and GR4 items

| ID | Requirement | Source | Evidence needed | Owner | Status | Evidence reference | Ticked by / date |
|---|---|---|---|---|---|---|---|
| D1 | Upload, record, analyze and export work end to end. | IMPLEMENTATION_PLAN.md section 12, exit gate G12; REVISED_PLAN.md GR4 | Named end-to-end test for upload, analyze and download; manual record-then-analyze run with the date and file used. | to assign | open |  |  |
| D2 | CLI, UI and offline scores match. | IMPLEMENTATION_PLAN.md section 12, exit gate G12; docs/r4_n2_n3_plan.md R4-N2 item 6; REVISED_PLAN.md risk list ("App and CLI scores diverge") | Named parity test: the same fixture bytes through the library analyze, `echoproof predict` and the app code path give identical score, score type, threshold and decision (values recorded for each path). | to assign | open |  |  |
| D3 | No mock output remains in the app. | IMPLEMENTATION_PLAN.md section 12, exit gate G12; docs/r4_n2_n3_plan.md R4-N2 item 1 | Search of the app code for the mock banner and mock paths (result recorded); named test that the mock banner is absent from the page. | to assign | open |  |  |
| D4 | Privacy cleanup and bounds pass. | IMPLEMENTATION_PLAN.md section 12, exit gate G12 | All rows B1 to B10 ticked with their evidence, plus the named test run result (pass count and fail count recorded). | to assign | open |  |  |
| D5 | A fresh environment runs with the pinned bundle. | IMPLEMENTATION_PLAN.md section 12, exit gate G12; REVISED_PLAN.md GR4; docs/r4_n2_n3_plan.md R4-N3 | `scripts/verify_install.py` exit code recorded from a fresh environment, and the clean-VM run kept as evidence (row E4). | to assign | open |  |  |
| D6 | Latency is measured, and any miss against the target is disclosed. | IMPLEMENTATION_PLAN.md section 12, exit gate G12; REVISED_PLAN.md GR4; docs/r4_n2_n3_plan.md R4-N2 item 8 | Measured warm and cold timings on the demo laptop (values, machine and transformers version recorded, see E3), compared with the B lock's warm p95 target of 5 s; any miss written out in the evidence reference. | to assign | open |  |  |
| D7 | The production bundle is the locked B WavLM bundle (`bundles/r3_n1_wavlm_head_B_calibrated`, `bundle:r3-n1-wavlm-head-B-calibrated`), verified against `locks/B_lock_gr3.json` at start-up, and the app fails closed on a mismatch or a missing bundle. | REVISED_PLAN.md GR4; docs/r4_n2_n3_plan.md R4-N2 item 1; locks/B_lock_gr3.json | Named start-up verification test that passes on the locked bundle and fails closed on a changed or missing component; recorded bundle hashes matching the lock. | to assign | open |  |  |
| D8 | Eight bundles are locked: four B and four A. | REVISED_PLAN.md GR4 and R4-N1; locks/B_lock_gr3.json; locks/A_lock.json | Both lock files present, with the bundle hashes they record (values or file references recorded). | to assign | open |  |  |

## E. Items that need a person

| ID | Requirement | Source | Evidence needed | Owner | Status | Evidence reference | Ticked by / date |
|---|---|---|---|---|---|---|---|
| E1 | A person completes a keyboard-only walkthrough of upload, analyze, error and download. | docs/r4_n2_n3_plan.md R4-N2 item 5 and "Needs a human" | Written record of the walkthrough: person, date, device, and the steps passed or failed. | to assign | open |  |  |
| E2 | A person completes a screen-reader walkthrough of the same flow. | docs/r4_n2_n3_plan.md R4-N2 item 5 and "Needs a human"; IMPLEMENTATION_PLAN.md section 12, accessibility paragraph | Written record: person, date, screen reader name and version, and the labels or outcomes read out and any gaps found. | to assign | open |  |  |
| E3 | Demo-laptop latency run on Subhan's HP EliteBook 840 G7 (CPU) for cold and warm analysis. | docs/r4_n2_n3_plan.md R4-N2 item 8 and "Needs a human"; IMPLEMENTATION_PLAN.md section 12, exit gate G12 | Measured timings_ms values (decode, features, inference, total) for cold and warm runs, with the command, the laptop model and the transformers version. | to assign | open |  |  |
| E4 | Clean-VM run (fresh Colab runtime) of the setup, bundle fetch, verify, CLI predict and app steps. | REVISED_PLAN.md R4-N3; docs/r4_n2_n3_plan.md R4-N3 item 6 and "Needs a human" | Saved notebook output (notebooks/R4_N3_Colab_Fresh_Install.ipynb or its saved output path) recorded with the run date. | to assign | open |  |  |
| E5 | Sign-off that the acceptance list is complete. | docs/r4_n2_n3_plan.md "Needs a human"; REVISED_PLAN.md R4-N2 ("Acceptance checklist ticked") | Written sign-off with the signer's name, the date, and the list of rows still open (if any). | to assign | open |  |  |

## How to use

1. Work from the top. Read the row's requirement and its source, then produce the evidence named in "Evidence needed".
2. Record the evidence in "Evidence reference": a test name with its file path, a measured value with its unit and machine, or a manual check with the person and date.
3. Only when the evidence is recorded and read, change Status from "open" to "ticked", and write the name and date in "Ticked by / date".
4. If a row cannot be met, keep it "open" and write the gap in "Evidence reference". Do not change the requirement text to fit the app. A latency miss goes in D6 with the measured value.
5. Rows in sections A to D use automated tests or code review where possible. Rows in section E need a person and cannot be ticked by code alone.
6. GR4 closes when D1 to D8 and E3 to E5 are ticked, and the fresh install (D5 and E4) runs.
