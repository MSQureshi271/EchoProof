# Audit of teammates' R2 work (R2-M1, R2-M2, R2-U1, R2-N1, plus R2-N2 and R3-M1 found in the folder)

Audit date: 8 November 2026. Auditor: Claude, working for Subhan (data and features). Scope: everything that exists locally under the project root after your sync,
checked against `REVISED_PLAN.md`, the teammates' own reports, and independent recomputation. Nothing was modified except this file.

**How to read this.** Section 1 is the verdict. Section 2 lists what is wrong or missing, ranked. Section 3 holds the per-task detail, and section 4 the evidence
behind the verdicts. Section 5 is a checklist of what to ask each teammate to fix.

---

## 1. Verdict

| Task | Owner | Plan "done when" met? | Verified here | Open items |
|---|---|---|---|---|
| R2-M1 LR-B | Muneeb | **Yes** on content, one deliverable missing from disk | Independently recomputed: selection macro F1 0.6923, AUC 0.8203, per-generator AUCs; all 5 output hashes and 4 input hashes match | `reports/ledger.csv` not synced (the plan's "ledger rows for all 8 trials" cannot be seen) |
| R2-M2 BiGRU diagnostics | Muneeb / Usman | **Yes** | Report is internally consistent (mask sums equal frame lengths, padding difference 2.2e-8, 32 clips, training accuracy 0.969, no checkpoint kept) | None blocking. Ran on a T4 with torch 2.11, a different stack from U1 (torch 2.14) |
| R2-U1 WavLM cache | Usman | **Yes** | All 14 pack hashes and both indices match the manifest; counts add up (21,470 + 19,876 + 5,933 = 47,279); parity on 50 items passed | Independent reviewer not recorded. Trained-head parity is deferred to R3-U1 (as planned). Test-genuine extraction is not reflected in the lock (see 2.4) |
| R2-N1 LR integration | Nouman | **Yes** (provisional) | CLI, evaluator and app scores are identical in the report; app and CLI no longer use the mock | Evidence is stale against the current `inference.py`; verified on one synthetic clip only; no documentation page; the `r2_n1` notebook still has a zip-sync cell |
| R2-N2 bundle and lock (not in your list) | Nouman / Muneeb | Yes on code and tests | Bundle/lock tests pass; negative integrity tests are in the report | `bundles/protocol_B_lr/` was not synced; `B_lock.json` is provisional and LR-only |
| R3-M1 BiGRU-B (not in your list, ahead of schedule) | Muneeb | Baseline plus one alternative done | Independently recomputed selection macro F1 0.8611; all output and checkpoint hashes match | Ledger not synced; independent review pending |

**Overall:** the science and the numbers hold up. The audit found no fabricated or inconsistent result. The problems are sync gaps, stale evidence hashes, one
test-access wording issue, and thin documentation. None of them invalidates a result, but items 2.1 to 2.5 should be fixed before GR3.

---

## 2. Findings, ranked

### 2.1 `reports/ledger.csv` is missing locally (affects R2-M1, R3-M1)
R2-M1's plan row says "Ledger rows for all 8 trials". Both `docs/r2_m1_lr.md` and `docs/r3_m1_training.md` say the runs append to `reports/ledger.csv`.
The file does not exist in `reports/`. The R2-M1 evidence JSON contains all 8 trials (all converged), and the R3-M1 report embeds its 2 ledger rows, so the
information exists, but the shared ledger itself is not in the synced folder. **Ask Muneeb to sync `reports/ledger.csv`.** The R3 docs also warn that the ledger
is not concurrency-safe, so only one notebook should write to it at a time.

### 2.2 `bundles/` is empty locally (affects R2-N2, the lock)
`locks/B_lock.json` points at `bundles/protocol_B_lr` and the R2-N2 Colab report verified it there. The folder is empty on this machine. The bundle's two
components do exist in `runs/r2/r2_m1_20261106T140611Z/model/` and their hashes match the lock exactly, so nothing is lost, but `bundle.json` and the packaged
copy cannot be inspected or round-tripped locally. **Ask Nouman to sync `bundles/protocol_B_lr/`.** The app currently loads the legacy `runs/.../model` folder
(`DEFAULT_BUNDLE_PATH` in `app/main.py`), not the verified bundle; it should switch once the bundle is local.

### 2.3 Source files used for the evidence differ from the local copies
Hashes recorded in the teammates' reports were compared with the files now on disk:

| File | Evidence says | Local file | Reading |
|---|---|---|---|
| `src/echoproof/models/lr_detector.py` | `02a64ecc…` (R2-M1 and R2-N2, "exact raw match" on Colab) | `080459b1…` (the version R2-S1 used; git shows it unchanged) | **The Colab copy that produced R2-M1 and N1/N2 differs from the local one.** The local tests still pass, but the code that made the LR-B model is not the code on this machine |
| `src/echoproof/inference.py` | `cf377e9f…` (R2-N1 evidence, 7 Nov) | `252ad577…` (modified since, +89 lines in git) | R2-N1's evidence predates the latest `inference.py`; R2-N2 re-ran the N1 tests on 8 Nov (18 passed), so the current code is probably fine, but the N1 report does not describe it |
| `chunks.py`, `cohort.py` | R2-S1 and R2-U1 give different hashes | Equal once Windows line endings (CRLF) are normalised | Benign. Colab hashes the LF copy, Windows the CRLF copy |

**Ask Muneeb to sync the `lr_detector.py` that M1 used** (or commit it), and add `.gitattributes` with `* text eol=lf` so hashes stay stable across machines.

### 2.4 Test-access wording in the lock does not mention R2-U1's test extraction
`locks/B_lock.json` says `test_partition_accessed: false` and that no test identifiers appear in the evidence. That is true for R2-M1. But R2-U1 extracted WavLM
embeddings for **5,933 B-test genuine chunks** (`include_test_genuine: true`, a stored index `test_genuine_index.json`). This is allowed by the plan row ("plus all val
and B-test genuine chunks") and was extraction only, with no scoring or fitting, and the U1 doc says so. But the lock validator hard-codes `False`, so the
lock cannot record it. The plan also asks the lock to hold a test-access log. **The final lock should list this exception.** I did not find any test-partition
scoring in any teammate's output (the R3-M1 and R2-M1 prediction tables contain only selection, calibration and operating rows).

### 2.5 `B_lock.json` is a provisional LR-only record under the GR3 file name
The plan's lock (`locks/B_lock.json`) is the GR3 lock for **all four bundles**, written at the end of Day 9, after which nothing in B may change. The file present
is `lock:protocol_b_lr_v1`, created 8 Nov, covering only the LR, and says "NOT_DEPLOYMENT_APPROVED" itself. `generate_protocol_b_lock` is written around the LR
(its limitation text is LR-specific). Risks: someone reads it as the GR3 lock, or GR3 overwrites it without keeping a record. **Suggest renaming it to a
provisional name until GR3, and making the generator accept any bundle.**

### 2.6 Model results the team should be aware of (not errors)
Selection validation, recording level, threshold 0.5:

| Model | Macro F1 | AUC all | AUC known (Melo, VITS) | AUC unseen XTTS | Genuine recall |
|---|---|---|---|---|---|
| R2-M1 LR (C 0.01) | 0.692 | 0.820 | 0.967 | **0.526** | 0.836 |
| R3-M1 BiGRU (selected: lower learning rate, epoch 10) | 0.861 | 0.951 | 0.985 | 0.883 | 0.882 |

The LR is close to chance on the unseen generator (it detects 14% of XTTS). The lock's limitations say this honestly. The BiGRU is far better, and the choice
between its two runs (0.8611 vs 0.8598) is within noise, so the selection is effectively a tie resolved by the lower learning rate, which is also the one with
the lower genuine false-positive rate (0.118 vs 0.31).

### 2.7 Cross-check with R2-S3: the models' scores are not explained by duration
Using my duration-matched selection set (810 of 2,000 recordings, duration AUC 0.49 after matching):

| Model | AUC all selection | AUC duration-matched | Macro F1 all / matched | XTTS AUC all / matched |
|---|---|---|---|---|
| R2-M1 LR | 0.820 | 0.814 | 0.692 / 0.744 | 0.526 / 0.529 |
| R3-M1 BiGRU | 0.951 | 0.949 | 0.861 / 0.891 | 0.883 / 0.886 |

Neither model loses accuracy when duration is controlled. This does not control for leading silence (the strongest shortcut in R2-S3); the models' scores correlate
only weakly with it (Spearman 0.15 for the LR, -0.01 for the BiGRU). A leading-silence-matched evaluation would be the next robustness check.

### 2.8 Documentation gaps
- No documentation page exists for **R2-N1** or **R2-N2** (the other tasks have one).
- `docs/decision_log.md` has no entry after GR1 (nothing for GR2, the provisional lock, the R2-U1 test-genuine exception, or the R2-S3 shortcut finding).
  `docs/task_board.md` still shows its 28 October snapshot.
- `docs/r2_u1_wavlm_cache.md` has a stale "Current verification status": it says no test pass or GPU run is claimed, but 47 tests passed and the GPU run completed.
- It also cites `reports/r1/shard_rename_20261104T080826Z.json`, which is not in `reports/r1/`.

### 2.9 Smaller points
- `notebooks/r2_n1_colab_verification.ipynb` contains a "Code Archive Synchronization (zip if exists)" cell with `echoproof_sync.zip`; you asked for zip logic to be removed.
  Older R1 notebooks still contain it too. Other zip mentions found are only Python's `zip()` or a sentence saying there is no zip extraction.
- R2-N1 verified the app on one 6-second synthetic waveform, which scored 0.99996 "synthetic". The report states it checks plumbing and parity, not detection quality.
- Colab runtime drift: M2 used Python 3.13.16 with torch 2.11.0; U1 used torch 2.14.1; M1/N1/N2 used Python 3.13.18. Each report records its own versions, but no single environment lock covers R2.
- No teammate work is committed. `git status` shows all of it untracked; the last commit (21caba8) is mine.

### 2.10 Local test suite
`pytest tests`: **344 passed, 19 skipped, 2 failed** (3 min). The two failures are in `tests/test_p1_13_environment.py` (dependency-closure tests) because the
local `.venv` has no `pip` module; they are about environment tooling, not R2 code. The 19 skips are tests that need `transformers` (R1-U1 and R1-U2), which is not
installed locally; those passed on Colab according to their reports. All new R2 and R3 test files pass locally.

---

## 3. Per-task detail

### R2-M1, LR-B (Muneeb)
Plan: 8-trial grid on B train, select on B selection validation (recording macro F1 at 0.5), majority and random references, convergence, standardised coefficients, adapter.
- **Grid:** 4 values of C (0.01, 0.1, 1, 10) by 2 class-weight settings = 8 trials, all `converged: true` (29 to 66 iterations of 2,000).
- **Selection:** order 0 (C 0.01, no class weight): macro F1 0.6923, the highest of the eight, and also the lowest EER (0.2533). Differences between trials are small (0.676 to 0.692).
- **References:** majority class (all synthetic) macro F1 0.429; random scores AUC 0.510 (seed 42).
- **Coefficients:** 67 standardised coefficients saved; the largest are `delta_00_std` (+1.30), `voiced_fraction` (+1.26), `zcr_std` (+0.78).
- **Model saved:** `runs/r2/r2_m1_20261106T140611Z/model/{weights.npz, lr_model.json}`; reload difference 0.0.
- **Inputs:** all four R2-S1 files match my hashes (crop bank 954fcb5f, mel stats d94c122a, train features 9bfb3c57, validation features 2ee91836); split hash 0854de6b and plan hash adfca8c9 match.
- **Scope:** only the selection role was scored (2,000 recordings, 10,029 chunks); no test and no Chatterbox rows.

### R2-M2, BiGRU diagnostics (Muneeb / Usman)
32 B-train recordings (16 genuine, 8 Melo, 8 VITS), seed 42, T4 GPU, 12 epochs, training accuracy 0.969 (target 0.95); frame lengths equal mask sums for every
clip (the one short clip has 192 frames and mask sum 192); the output changes by at most 2.2e-8 when padded beside a longer clip (tolerance 1e-5). The report
states that no checkpoint was kept and that no generalisation is claimed. Input and source hashes match.

### R2-U1, WavLM embedding cache (Usman)
Scope checked against the plan: B-train crop bank 21,470 (the existing R2-S1 bank, not a new draw), all B validation chunks 19,876 (selection 10,029,
calibration 4,712, operating 5,135), B-test genuine chunks 5,933 (matches the chunk manifest). Components in the cache: genuine, melo, vits, xtts, none from Chatterbox.
Mean and mean+std keys are unique (47,279 each). Cache size 460 MB of packs plus 32 MB of indices, extraction 1,399 s on a T4. The pinned
`microsoft/wavlm-base-plus` revision and weight hash are recorded. Parity: 50 development items, embedding difference at most 7.7e-6, untrained-head score difference at
most 1.2e-7 (tolerance 1e-4). Hashes of all 14 local packs and both indices match the manifest. The `independent_reviewer` field is null.

### R2-N1, LR integration (Nouman)
CLI (`python -m echoproof.cli predict`), evaluator and app give the same score (0.999958552) bit for bit; CLI error codes 2 and 4 are tested; the production path
loads `load_lr_bundle` and `is_mock` is false; eight regression suites passed on Colab. See 2.3 and 2.9 for the stale-hash and synthetic-audio caveats.

### R2-N2, bundle and lock (Nouman / Muneeb)
`bundle.py` (285 lines) and `lock.py` (483 lines) with 21 tests (10 + 11) passing. The report shows five negative integrity tests passing (corrupt component,
metadata tamper, missing component, lock component hash tamper, evidence tamper), packaged-versus-legacy scoring identical, and threshold override leaving scores
unchanged. See 2.2, 2.4 and 2.5.

### R3-M1, BiGRU-B baseline (Muneeb)
Two experiments on B train, both seed 42: baseline (learning rate 0.001, stopped at epoch 6, best epoch 3, 8.9 min) and lower learning rate (0.0003, 13 epochs, best
epoch 10, 15.5 min), T4 GPU, about 369 MB peak memory. The experiment contract was written before fitting; selection was frozen before calibration and operating audio were
scored. The 6 prediction tables and both checkpoints match their recorded hashes. Selection macro F1 0.8611 (lower learning rate) against 0.8598.

---

## 4. What was checked

- **Hashes:** every output file named in the R2-M1, R3-M1, R2-U1 and lock records was re-hashed locally (all match except the source files listed in 2.3).
- **Recomputation:** selection macro F1 and AUC for R2-M1 and R3-M1 recomputed from the saved prediction tables with scikit-learn; per-generator AUCs for the LR.
- **Scope:** component and partition lists in each cache and prediction table checked for Chatterbox and test rows; calibration and operating tables contain 960 and 1,040 recordings and no Chatterbox.
- **Counts:** plan counts (21,470 / 19,876 / 5,933) reproduced from the frozen chunk manifest.
- **Tests:** full local suite run (see 2.10).
- **Not checked:** the Colab runs themselves (not reproducible here), the Drive copy of the project, the content of the notebooks beyond their saved outputs and zip mentions,
  and whether the teammates' unsynced files differ in any way other than the ones listed.

## 5. Requests to teammates

| To | Request |
|---|---|
| Muneeb | Sync `reports/ledger.csv` and the `lr_detector.py` that R2-M1 used; confirm which 8 and 2 ledger rows belong to R2-M1 and R3-M1 |
| Nouman | Sync `bundles/protocol_B_lr/`; rerun the R2-N1 parity check against the current `inference.py`; add short docs for R2-N1 and R2-N2; remove the zip cell from `r2_n1_colab_verification.ipynb`; switch the app to the verified bundle |
| Usman | Record the R2-U1 independent reviewer (Muneeb); add `shard_rename_20261104T080826Z.json` to `reports/r1/` or correct the doc's reference; refresh the "current verification status" section |
| All | Decide and log in `docs/decision_log.md`: the R2-U1 test-genuine extraction exception, the provisional status and name of `B_lock.json`, and the shortcut findings of R2-S3; add `.gitattributes` for LF endings; commit the work |

---

## 6. Re-audit after the second sync (8 November 2026)

You synced the files requested in section 5. Each finding from section 2 was rechecked against disk.

| Finding | Status now | Evidence |
|---|---|---|
| 2.1 `reports/ledger.csv` missing | **Fixed** | File present with 10 rows: 8 R2-M1 trials and 2 R3-M1 runs, all `completed`, no duplicate run ids. Every R2-M1 row's macro F1 and FPR equals the R2-M1 report's trial, and both R3-M1 rows equal the rows in the R3-M1 report |
| 2.2 `bundles/` empty | **Fixed** | `bundles/protocol_B_lr/{bundle.json, model/}` present. `bundle.json` and both components match the hashes in `locks/B_lock.json`. The verified bundle loads through `load_lr_bundle` (not mock), and `load_lock_record` validates the lock against it |
| 2.3 `lr_detector.py` differs from the evidence | **Fixed** | Local hash is now `02a64ecc…`, the value in the R2-M1 and R2-N2 evidence. The change from the old version is one line: the ledger `notes` text now also records `error=`. Harmless |
| 2.3 `inference.py` differs from R2-N1 evidence | **Still open** | Local `252ad577…` against the R2-N1 report's `cf377e9f…`. The R2-N2 report re-ran the N1 tests later, but the N1 evidence itself was not regenerated |
| 2.4 Lock does not mention test-genuine extraction | **Partly addressed** | `docs/decision_log.md` now has an "R2-U1 implementation record" that states the extraction-only test exception, never extracting Chatterbox or test TTS, and not scoring test examples. `locks/B_lock.json` is unchanged (still created 8 Nov 01:49, `test_partition_accessed: false`, no mention) |
| 2.5 Provisional lock under the GR3 name | **Still open** | Same file, same `lock:protocol_b_lr_v1` id |
| 2.8 Documentation | **Mostly open** | Still no doc page for R2-N1 or R2-N2. The R2-U1 doc's "current verification status" still says no local test pass or GPU run is claimed (file last edited 6 Nov). `reports/r1/shard_rename_20261104T080826Z.json` is still absent. Decision log gained only the R2-U1 entry; nothing on GR2, the provisional lock or R2-S3 |
| 2.9 Zip cell in `r2_n1_colab_verification.ipynb` | **Still open** | `echoproof_sync.zip` still referenced once |
| 2.9 App loads the legacy folder | **Still open** | `DEFAULT_BUNDLE_PATH` is still `runs/r2/r2_m1_20261106T140611Z/model`; the verified bundle is now available locally to switch to |
| `.gitattributes` / commits | **Still open** | No `.gitattributes`; all teammate work is still uncommitted |

### New problem introduced by this sync: `docs/task_board.md`
The working copy of `docs/task_board.md` now differs from the committed version in a way that removes information:
- The "Latest baseline milestone: P1-14 … recorded on 30 October 2026" line is gone.
- P0-09 and P0-10 changed from "S/M/U/N assigned by role" and "G0 closed by user team attestation" to "Pending actual assignment" and "Pending actual review".
- The paragraph under "Next-phase handoff queue" describing the Phase 1 evidence is gone, and two queue rows were added.
- It differs from both the first and the latest commits.

This looks like an older or alternative copy overwrote the committed text. It is not a teammate result, but the status changes (G0 reverting from closed to pending) contradict `phase_0_gate_checklist.md`, which records G0 as closed by the team's attestation.
**Ask whoever edited it to confirm whether the reversal of P0-09 and P0-10 was intended, or restore the committed version.**

### Tests after the sync
The R2-N1, LR detector, bundle, lock, R2-M1 workflow and R1-N2 inference suites were re-run locally: **86 passed, 1 skipped**. The skip is the existing matplotlib-absent test that no longer applies.

### Remaining requests (short list)
1. Nouman: regenerate the R2-N1 parity evidence against the current `inference.py`; add docs for R2-N1 and R2-N2; remove the zip cell; point the app at `bundles/protocol_B_lr`.
2. Usman: refresh the R2-U1 doc's status text; add or correct the `shard_rename` reference.
3. Whoever edited `docs/task_board.md`: confirm or revert.
4. Team: add the R2-U1 test-genuine exception to the lock record (or rename the file as provisional), log GR2 and the R2-S3 findings, add `.gitattributes`, commit.
