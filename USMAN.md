# Usman's R2-U1 and R2-U2 work: review report

Prepared 11 November 2026. Scope: the AST warm-up (R2-U2) and WavLM embedding cache (R2-U1) tasks owned by Usman (Muhammad Usman Murtaza), with Muneeb as the plan's reviewer. Everything below was checked against the files in this repository and the evidence bundles Usman supplied. Where I could not check something, it is listed in section 8 and not presented as verified.

---

## 1. Bottom line

| Task | Plan | What exists now | My verdict |
|---|---|---|---|
| **R2-U1** WavLM cache | D4 to D5, cache of embeddings for B train, all validation and B-test genuine chunks, parity on 50 chunks | Complete: 47,279 cached chunks, all 14 packs and both indices match their manifest, parity passed | **Delivered, on time, verified.** Independent review still not recorded |
| **R2-U2** AST frozen warm-up | D5, 1 to 3 epochs on B train, checkpoint plus ledger row | A usable checkpoint exists (best epoch 2, selection macro F1 0.7914) and is adopted as the frozen-AST comparator | **Delivered, about three days late, and by an unplanned route.** The checkpoint is sound. Acceptance rests on reported, not documented, team approval, and the handoff package has gaps |

The single most important finding: **the original R2-U2 run was not a training failure.** It trained two epochs and reached the same result that was later accepted. It was rejected only because the publication guard saw an epoch-mean gradient norm of `Infinity`. The re-run ("Option B") reproduced the original model **bit for bit** (section 4.4). The three days between the first run and the adoption went on making the reporting finite and writing the acceptance trail, not on improving the model.

---

## 2. What the plan asked for

| Row | Plan text (REVISED_PLAN.md, section 6) | Days | Depends on | "Done when" |
|---|---|---|---|---|
| R2-U1 | Extract pooled embeddings (mean and mean+std) for B train crop banks, plus **all validation** and **B-test genuine** chunks. Do not extract Chatterbox. Atomic writes, keys from audio hash, chunk coordinates, revision and pooling. Check cached against direct scores on 50 chunks. | D4 to D5 (7 to 8 Nov) | GR1, R1-U1 | Cache inventory, with extraction time and size logged |
| R2-U2 | Train the 2-class AST head with a frozen backbone in eval mode for 1 to 3 epochs on B train. Save the warm-up checkpoint as the "frozen AST" comparison. | D5 (8 Nov) | GR1, R1-U2, R1-M1 | Warm-up checkpoint and ledger row |

Both feed Usman's next tasks: R3-U1 (WavLM head, on the cache) and R3-U2 (AST partial fine-tuning, starting from the warm-up checkpoint).

---

## 3. R2-U1: WavLM embedding cache

### 3.1 What Usman did

- Wrote the extractor and builder (`src/echoproof/features/r2_u1.py`, `scripts/r2_u1_wavlm_cache.py`), a Colab notebook, a fixture test suite (`tests/test_r2_wavlm_cache.py`) and the guide `docs/r2_u1_wavlm_cache.md`.
- Ran the extraction on a Colab T4 and produced `data/features/r2_u1_B/`: 14 embedding packs (one per cohort shard), a development index and a separate test-genuine index, and a manifest.
- Produced four evidence reports in `reports/r2/`: build, tests, Colab summary and an independent integrity check.
- Committed the files on 7 November through five GitHub uploads.

### 3.2 What the reports say

| Item | Value |
|---|---|
| Cached chunks (logical rows) | **47,279**, with unique mean and mean+std keys for every one |
| By partition | train 21,470 (the R2-S1 crop bank); validation 19,876 (selection 10,029, calibration 4,712, operating 5,135); test 5,933 (genuine only) |
| By component | genuine 19,012; MeloTTS 11,663; VITS 11,946; XTTS 4,658; **ChatterboxTTS none** |
| Size | 492,294,268 bytes (packs 460,037,508; indices 32,256,760) |
| Extraction time | 1,398.8 s over all packs on a T4; 171.9 s staging the shards; 1,421.7 s for the session |
| Model | frozen `microsoft/wavlm-base-plus`, fp32, no TF32, last hidden state, 768 dimensions, pooling over true frames only |
| Parity (50 development chunks) | embedding difference at most 4.7e-6 (mean) and 7.7e-6 (mean+std); score difference at most 4.1e-8 and 1.2e-7; tolerance 1e-4; **passed** |
| Status in the manifest | `extracted_pending_independent_review`; `independent_reviewer: null` |

### 3.3 My independent check

I re-hashed all 14 packs against the manifest (14 of 14 match, 460,037,508 bytes), confirmed both index hashes, and confirmed the counts add up (21,470 + 19,876 + 5,933 = 47,279). The four R2-U1 reports in Usman's own evidence bundle are byte-identical to the ones in `reports/r2/`. The R2-U1 files have not changed since the 7 November upload; Muneeb's later commit only stripped saved outputs from the notebook.

### 3.4 Changes from the plan

| Plan | Actual | Assessment |
|---|---|---|
| "Check cached against direct scores on 50 chunks" | Checked with an **untrained, seeded diagnostic head** (no trained head exists yet). Trained-head parity is explicitly deferred to R3-U1. | Reasonable and openly stated. It tests the cache path, not detector accuracy |
| Plan budget assumed four WavLM crops per training recording | Used the R2-S1 K=5 crop bank as published (21,470 crops) | Recorded in the decision log (6 November). Keeps R2-S1 and R2-U1 consistent |
| "All validation and B-test genuine chunks" | Done as written, with the test-genuine part behind an explicit `--include-test-genuine` option, **separately indexed** and excluded from default readers | This is the plan's own instruction, but it contradicts the contract's normal sealing of test data. See below |

**Test exposure.** The cache contains embeddings of 5,933 Protocol B test-genuine chunks. The decision log records it as an acknowledged, extraction-only exception: nothing was scored, and no Chatterbox or test TTS audio was extracted. The manifest and the by-component counts agree. It should be disclosed in the report (the page 6 draft does).

### 3.5 Open items (all carried over from my earlier audit)

1. **No independent reviewer is recorded** (`independent_reviewer: null`).
2. **The guide's "Current verification status" paragraph is out of date.** It still says no GPU run is claimed, though the Colab run was done and verified.
3. **`reports/r1/shard_rename_20261104T080826Z.json` is still missing from the repository.** The guide cites it. It exists in Usman's evidence bundle and only needs copying.

---

## 4. R2-U2: AST frozen warm-up

### 4.1 Timeline

| When (UTC) | Event |
|---|---|
| 7 Nov | R2-U2 fixture suite: **68 tests passed** on a T4 (report 15:09) |
| 7 Nov | First training attempt fails at once: a **missing test-report path** (a launch-configuration error, 15:04) |
| 7 Nov | Second attempt trains two epochs (1,044.6 s), then the publication guard raises **"non-finite training metrics; do not publish this warm-up"** |
| 8 Nov 12:30 | Usman runs a **diagnostic** on the final two training recordings (8 numerical conditions) |
| 9 Nov 20:07 | The failed run is **archived** as `R2_U2_Unaccepted` (37 files, 690.6 MB, hashes verified) |
| 10 Nov 12:32 | New **software suite: 17 tests passed** on Colab (Option B instrumentation) |
| 10 Nov 13:07 to 13:30 | **Option B** run: fresh two-epoch warm-up with the new instrumentation; experiment checks pass |
| 10 Nov 20:37 | Adoption commit `0e4c0d5`; official layout `c2112f7` (20:53); handoff note `2c99434` (21:02) |

Planned for D5 (8 November); adopted on D8 (11 November).

### 4.2 What the original run did

The recipe (`configs/r2_u2_ast_b.json`) was chosen by Usman under the user's authority, not prescribed by the plan: 2 epochs, AdamW, learning rate 1e-4, weight decay 0.01, constant schedule, microbatch 4 with 4-step accumulation (effective batch 16), FP16 autocast with GradScaler, gradient clipping 1.0, seed 42, natural class mix (1,510 genuine, 3,020 TTS), one seeded 4-second crop per recording per epoch, only the classifier group trainable, backbone frozen and in eval mode. Training consumed only B-train audio; validation used only the selection role.

The run completed. Its history:

| Epoch | Train loss | Selection macro F1 | EER | Genuine FPR | Epoch gradient norm |
|---|---|---|---|---|---|
| 1 | 0.4679 | 0.7350 | 0.210 | 0.476 | **Infinity** |
| 2 | 0.3327 | **0.7914** | 0.170 | 0.148 | **Infinity** |

The `Infinity` came from the shared Trainer averaging gradient norms across all attempts, including FP16 attempts whose gradients overflowed and were skipped by the GradScaler. The publication guard correctly refused to publish a run with a non-finite recorded metric.

### 4.3 What Usman then did (Option B)

1. **Diagnosed** the final step. Evidence supports a scale-sensitive FP16 backward overflow in the reconstructed final two-recording group; the first historical event could not be localised (the report says so).
2. **Wrote a numerical extension** of the Trainer (`r2_u2_numerical_trainer.py`, 245 lines). It averages only finite norms of applied updates, records scaler skips separately, observes real optimizer calls and scaler transitions, and fails closed on norm-only overflow or unexplained skips. The shared `trainer.py` was **not modified** (checked: its last change is Muneeb's 7 November commit, and its hash matches the manifest).
3. **Fixed a launcher lifecycle bug** (metadata was written before a constructor that needs an empty directory).
4. **Re-ran** a fresh two-epoch warm-up with a pre-declared policy (`option_b_policy.json`: no arbitrary skip threshold, reject unexplained updates).
5. **Compared** it with the original and **adopted** it.

### 4.4 My independent comparison of the two runs

| Artifact | Original run | Option B run | Result |
|---|---|---|---|
| `predictions_recordings.parquet` | sha `64647cec…` | sha `64647cec…` | **Byte-identical** |
| `predictions_chunks.parquet` | sha `5a3a369a…` | sha `5a3a369a…` | **Byte-identical** |
| History fields (loss, all metrics, lr, skipped steps) | | | **Identical** for both epochs |
| Epoch gradient norm | Infinity, Infinity | 4.052, 3.323 | Now finite (the only reporting change) |
| `best.pt` model tensors | 203 | 203 | **All 203 bit-identical** |
| `best.pt` file hash | `f9a92af9…` | `6a3883e5…` | Differ (checkpoints also hold provenance and telemetry) |
| Epoch time | 397 s, 374 s | 473 s, 440 s | Option B is about 19% slower (instrumentation) |

So the accepted model **is** the model the original run produced. Option B changed how it is reported, not what was learned. Usman's own acceptance document says this ("Option B improved numerical evidence and reporting, not F1"), and my comparison confirms it.

### 4.5 What the Option B reports say

- **Software gate:** 17 of 17 tests passed on Colab (T4, torch 2.14.1+cu130), no skips or failures.
- **Run:** 568 optimizer attempts, 566 applied updates, **2 skipped by the scaler**. Both skips are the final two-recording group of each epoch (attempts 284 and 568); the loss scale fell 65,536 to 32,768 to 16,384. Epoch norms are 4.05 and 3.32 (applied updates only).
- **Freeze proof:** the classifier changed; all 14 other parameter groups (12 transformer blocks, embeddings, final norm) are unchanged. The backbone stayed frozen.
- **Reload parity:** reloading `best.pt` reproduces all 10,029 selection chunks and 2,000 recordings with maximum difference 0.0.
- **Coverage:** 6,530 recordings' PCM hashes verified (4,530 train + 2,000 selection); no calibration, operating, test or Chatterbox audio used.
- **Status:** `experiment_checks_passed_pending_team_review`, `research_accepted: false` as written at execution time. The later adoption is a separate record.

### 4.6 The adopted checkpoint

| Item | Value |
|---|---|
| Path | `runs/r2/option_b_1b3757e898da4baaad2bde49fa3cea01/best.pt` |
| SHA-256 | `6a3883e59fc48ec18697fcdfdb59641adbd7dee6e965c1f80ad6f5287ea0bd57` (I re-hashed it: matches) |
| Selected epoch | 2 of 2 |
| Selection role (2,000 recordings, fixed 0.5) | macro F1 **0.7914**, accuracy 0.8245, EER 0.170, genuine FPR 0.148 |
| Training cost | 15.2 minutes, 862 MB peak GPU memory |

**My analysis of its saved selection predictions** (not in Usman's reports):

| Panel | Value |
|---|---|
| Known generators (genuine + MeloTTS + VITS) | ROC-AUC **0.982**, EER 0.075, macro F1 0.924 |
| Unseen XTTS (genuine + XTTS) | ROC-AUC **0.780**, EER 0.311, macro F1 0.663 |
| Recall at 0.5 | MeloTTS 0.988, VITS 0.962, **XTTS 0.496** |

For comparison on the same role: the BiGRU selected by Muneeb has XTTS-panel AUC 0.904 and the handcrafted LR is near chance on XTTS. The frozen AST is therefore a **comparator, not a contender**: excellent on generators it saw, and about half of XTTS gets through at 0.5. That is the expected role for a frozen backbone, and R3-U2's partial fine-tuning exists to improve on it.

**Muneeb's three speakers.** The handoff note leaves open his observation about speakers 1578, 302 and 7190 (the BiGRU's false-alarm hot spots). For this model they do not behave alike: 1578 and 7190 have no false alarms, 302 has 8 of 20, and the worst speaker is **3235 (19 of 20 genuine recordings flagged)**, then 4853 (10), 5652 (9) and 3947 (6). 74 false alarms in total from 15 of 25 speakers. The speaker pattern is model-specific, so it should not be attributed to the data without more evidence.

---

## 5. Changes from the plan, and work beyond it

### 5.1 Changes from the plan

| # | Plan | Actual |
|---|---|---|
| 1 | R2-U2 finishes on D5 (8 Nov) | Adopted on D8 (11 Nov). R3-U2 (D6 to D9) could not start in the meantime |
| 2 | Warm-up checkpoint plus a ledger row | Checkpoint delivered. The **per-run** ledger row exists, but it is **not in the main `reports/ledger.csv`** (see finding F4) |
| 3 | Shared project Drive and shared trainer | Runs were made on a **second Colab account** ("Account B", workspace `Colab_R2_U1_Workspace`) using a **subclass of the shared Trainer** (the shared `trainer.py` itself is untouched) |
| 4 | Training settings left to the implementer | Usman chose and documented them (section 4.2), under the user's authority, not as a team decision |
| 5 | Plan row unchanged | Usman added a status paragraph to `REVISED_PLAN.md` (7 lines) and an adoption entry to the decision log. The plan's R2-U2 row itself was not edited |
| 6 | Done = checkpoint + ledger row | Done = checkpoint **plus** an acceptance record, a manifest of 112 bound files, and an unfinished recipient handoff |

### 5.2 Work Usman did that the plan did not ask for

- A **diagnostic** of the overflow (8 conditions, with its own safety tests), and an archive of the failed run (37 files, 690.6 MB, hash-verified).
- The **numerical Trainer extension**, scope and state checks, prospective acceptance checks, and a launcher that writes metadata exclusively before fitting.
- An **isolated experiment tree** (`experiments/r2_u2/`: 25 files in this repository; the placement inventory records 70 files in the original tree on Usman's Drive, which also holds the large runs and reports), a prospective policy, a **17-test** Colab software suite, and the 8 numerical regression cases with a separate copy-equivalence test file in the official `tests/`.
- An **adoption package**: decision record, evidence manifest, historical comparison, repository verification, a read-only **handoff verifier** (`verify_handoff.py`) and its tests.
- A retrospective **"Council" review** (section 6, F2) and a 214-line acceptance guide.
- Official-layout publication: copies of the run into `runs/r2/…` and of the reports into `reports/r2/r2_u2_option_b/`, with a 24-file hash inventory. I verified all 24 copies and all 11 run artifacts against their recorded hashes: no mismatches.

---

## 6. Findings and discrepancies

Ordered by importance. Each says what I checked.

**F1. The re-run was not needed to obtain the model (informational, positive).** Section 4.4: predictions byte-identical, history identical, 203 of 203 tensors bit-identical. The warm-up checkpoint existed on 7 November.

**F2. Acceptance rests on reported approval, and the "Council" is not a human review.**
- `decision.json` records `accepted_on_user_report_of_team_approval`; `independent_team_signoff_artifact`, the approval date and the message ID are all `null`. Usman's document says no signed message from Muneeb exists.
- The "five-advisor / five-peer Council" in the decision log and `council_review.md` is a set of AI-assistant reviews done in a conversation (the file says they were not five model providers and are not signed or archived). It supports the engineering reading, but it is not team approval, and the decision log wording ("previously completed five-advisor/five-peer Council review") could be misread as one.
- The plan's reviewer is Muneeb. A written confirmation from Muneeb is the missing piece.

**F3. Usman's own handoff verifier fails on this checkout.** `python experiments/r2_u2/acceptance/verify_handoff.py --project-root .` returns `verification_failed: Artifact identity mismatch: docs/r2_u2_acceptance.md`. I compared all 112 files the manifest binds:

| Result | Files |
|---|---|
| Match exactly | 27 |
| Differ only in line endings (Windows CRLF checkout) | 65 |
| Not present in this repository copy (the experiment's original `runs/` and `reports/` folders; official copies exist elsewhere) | 18 |
| Genuinely different | **2**: the acceptance guide, and `lr_adapter.py` |

`lr_adapter.py` is Muneeb's later commit (`97977c9`), after the source snapshot, so it is not Usman's. The acceptance guide is Usman's, and the guide says to check its historical identity "using original commit `0e4c0d5`". I tested that: **the manifest's hash for the guide matches no committed version** (three commits, each as-is, LF and CRLF). The most likely explanation (an inference) is that the guide was edited after the manifest was written and before it was committed. As shipped, the verification tool and its stated remedy both fail.

**F4. The ledger claim is not true in this checkout.** The guide says the run's ledger row "is mirrored verbatim into `reports/ledger.csv`". `reports/ledger.csv` has **no** `option_b_…` row (0 matches). The plan's done-criterion ("ledger row") is therefore unmet in the main ledger here. The per-run row exists in the run folder. The guide also says to add the row in the receiving project, so the mirroring may have been done only on Usman's own copy.

**F5. The recipient handoff is unverified.** The three outstanding actions in Usman's own guide are all still open: written team sign-off, a durable shared artifact link with permissions (the only location named is Usman's own Drive), and the CPU model-load and fresh-optimizer check (the pinned AST snapshot may have been lost with Colab scratch). The decision record says R3-U2 may not start until they are done.

**F6. The official layout wraps the experiment tree rather than replacing it.** `scripts/r2_u2_ast_warmup.py` (15 lines) just runs `experiments/r2_u2/scripts/launch.py`; the official notebooks execute the hash-bound experiment paths; `src/echoproof/training/r2_u2_*.py` are copies guarded against drift. That keeps provenance honest, but anyone who has only the "official" folders cannot run it. R3-U2 needs its own runner with its own warm-up and cosine schedule semantics; the Option B acceptance covers a constant schedule only, and the tests already print a scheduler-order warning for the cosine fixture.

**F7. Shared infrastructure is deliberately left unfixed.** The Trainer's norm averaging, norm-based skip counting and "completed" ledger semantics are unchanged for the BiGRU and all other models. That is a defensible scope decision, but the same `Infinity` issue can recur in other FP16 runs, and the team should know it exists.

**F8. R2-U1 loose ends from my earlier audit remain** (section 3.5).

**F9. Housekeeping.** `R2_U2_Unaccepted/` (690 MB) sits untracked in the project root. The repository has no `.gitattributes`, which is why 65 manifest "mismatches" are line endings only. Both recur until someone decides.

---

## 7. What to do next

| Owner | Action |
|---|---|
| Muneeb | Confirm in writing that the Option B candidate is approved (F2), or state that Usman's attestation is accepted team practice |
| Usman | Mirror the `option_b_…` row into the main `reports/ledger.csv` (F4) and name the shared artifact link (F5) |
| Usman | Fix or retire the manifest binding for the acceptance guide (F3), so `verify_handoff.py` can pass; copy `shard_rename_20261104T080826Z.json` into `reports/r1/`; refresh the R2-U1 guide's status text (F8) |
| Usman | Run the CPU model-load check where the pinned AST snapshot exists, and keep the receipt (F5) |
| Team | Decide whether R2-U2 is "done": the model is, the handoff is not. Record that, then let R3-U2 start from `best.pt` with a new optimizer and its own contract (F6) |
| Team | Add `.gitattributes` (LF), decide whether `R2_U2_Unaccepted/` belongs in the repo, and note the shared-Trainer limitation for later FP16 runs (F7, F9) |

---

## 8. How this was checked, and what was not

**Checked directly**
- Re-hashed: all 14 R2-U1 packs and both indices; the 11 Option B run artifacts; all 24 official inventory copies; `best.pt`; both prediction files.
- Compared the original and Option B runs: prediction bytes, history fields, and all 203 model tensors (loaded with weights-only deserialisation).
- Compared all 112 manifest hashes, and the acceptance guide against its three committed versions.
- Tested claims against the files: checkpoint hash, ledger mirroring, shared-Trainer hash and authorship, recipe equality (identical in content; hash differs only by CRLF).
- Ran Usman's tests locally: official suite **18 passed, 8 skipped** (7 need `transformers`, 1 needs CUDA); experiment suite **20 passed, 1 skipped**. Ran the read-only verifier (failed, F3).
- Computed selection-role panels and per-speaker false alarms from the saved predictions.

**Not checked, and why**
- **The Colab runs themselves** (the 68-test and 17-test suites, both training runs, the diagnostic): no GPU and no `transformers` here. I relied on the recorded reports and checked their hashes and internal consistency.
- **That the team approved**, or that Muneeb reviewed: no evidence exists in the repository.
- **Whether teammates can reach the artifacts**, and whether the pinned AST snapshot still exists.
- **The 18 absent experiment files**, which live on Usman's Drive.

## Appendix: key files

| Area | Files |
|---|---|
| R2-U1 code and docs | `src/echoproof/features/r2_u1.py`, `scripts/r2_u1_wavlm_cache.py`, `docs/r2_u1_wavlm_cache.md`, `notebooks/R2_U1_Colab_WavLM_Cache.ipynb`, `tests/test_r2_wavlm_cache.py` |
| R2-U1 outputs | `data/features/r2_u1_B/`, `reports/r2/r2_u1_{build,colab,tests,verify}_*.json` |
| R2-U2 official | `src/echoproof/training/r2_u2_*.py`, `scripts/r2_u2_ast_warmup.py`, `scripts/r2_u2_verify.py`, `configs/r2_u2_ast_b.json`, `notebooks/R2_U2_Colab_*.ipynb`, `tests/test_r2_u2_*.py`, `docs/r2_u2_acceptance.md` |
| R2-U2 outputs | `runs/r2/option_b_1b3757e898da4baaad2bde49fa3cea01/`, `reports/r2/r2_u2_option_b/`, `reports/r2/r2_u2_option_b_acceptance.json` |
| R2-U2 experiment and acceptance | `experiments/r2_u2/` (scripts, tests, configs, `acceptance/`) |
| Failed run archive | `R2_U2_Unaccepted/` (original run, failure report, diagnostic) |
| Decision trail | `docs/decision_log.md` ("R2-U1 implementation record", "R2-U2 Option B adoption"), `REVISED_PLAN.md` (status paragraph) |
| Usman's commits | 7 Nov: five R2-U1 uploads (`951b0a7` to `86a776b`); 11 Nov: `0e4c0d5`, `c2112f7`, `2c99434` |
