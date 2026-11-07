# EchoProof decision log

Version 1.0 · 28 October 2026 · Maintainer D

Source: [Phase 0 implementation plan](../IMPLEMENTATION_PLAN.md). Owners A–D map provisionally to S/M/U/N respectively. Actual names and calendar kickoff have not been supplied. G0 was closed by the user's team-understanding attestation; no individual review dates are inferred. Dates use project-relative weeks until kickoff is confirmed.

## Recorded decisions

| ID | Decision | Rationale / evidence | Owner / reviewer |
|---|---|---|---|
| DEC-001 | English recording-level genuine=0 versus TTS=1; exclude VC/mixed/replay claims | Supplied MVP and guide scope | D / A |
| DEC-002 | LR, two-layer BiGRU, frozen WavLM/head, partially adapted AST | Required approaches; both pretrained models are Transformers | B/C / D |
| DEC-003 | Group relatives and source speakers before crops/augmentation | Avoid content/identity leakage; disclose target/reference overlap | A / D |
| DEC-004 | Separate A/B fits; freeze B before A sees B's final family | Prevent indirect model-selection contamination; minimum eight fits | D / C |
| DEC-005 | Select specific B artifact using development macro F1 → EER → FPR → latency | Fixed comparison and production objective | D / B |
| DEC-006 | Mean chunk probability, recording calibration, separate threshold fitting | Match prediction unit and prevent fit-role leakage | D / B |
| DEC-007 | Maximize operating-validation macro F1; no version-1 FPR constraint | Avoid unsupported guarantees; ties defined in research contract | D / A |
| DEC-008 | PhonemeDF-only scope: genuine recordings and four TTS families | User decision supersedes the earlier two-source default; evaluate known and held-out generators | N / S |
| DEC-009 | WAV/MP3, 25 MiB/60 seconds, transient audio, shared inference | Bounded and reviewable MVP | D / C |
| DEC-010 | Five-second warm p95 target on ten-second clip; no accuracy guarantee | Plan's engineering target; measured qualification remains pending | D / C |
| DEC-011 | Reserve 25% compute; bound trials; reduce scale before integrity | Eight-week feasibility; no invented GPU-hour availability | C / B |
| DEC-012 | Model-score wording unless recording calibration supports probability | Prevent confidence overclaims under distribution shift | D / A |

## Open evidence and assignments

The following entries are now defined as decisions: where current information supports a choice, the chosen policy is recorded; where the choice depends on missing measurements or external confirmation, the decision rule and responsible owner are specified. “Pending evidence” means the policy is set but cannot be operationally closed yet.

| ID | Needed decision/evidence | Owner / reviewer | Deadline | Default and consequence |
|---|---|---|---|---|
| OPEN-001 | Pair by shared stem provisionally; verify transcript/audio and identity metadata in audit. Treat each genuine/TTS stem pair as one indivisible group; unknown source speakers stay unknown. The user reports XTTSv2 and ChatterboxTTS on Drive; verify each generator and its relationships. | S / N | Phase 2, before split | Do not infer transcript equality, source-speaker identity, or other generator families from names/directories. Unseen-generator feasibility remains pending verification of generator families and group support. |
| OPEN-002 | Closed by user decision: no second dataset or external-evaluation objective | S/N / M | Closed | PhonemeDF-only scope; generator families are not independent datasets. |
| OPEN-003 | Train LR and BiGRU on Colab first. Record runtime type, GPU, available session/storage, actual batch timings and persistent artifact destination. Seek separate GPU-machine time only if baseline evidence justifies it and access is available; budget no paid compute by default. | U / M | Before neural training and at Phase 1 profiling | Availability, numeric budget, GPU feasibility and backup are pending Colab/session inspection; stop/resume checkpoints and external persistence are required. |
| OPEN-004 | Benchmark on the actual selected demo device, batch 1, warm model, end-to-end ten-second audio, repeated runs, reporting p50/p95 and component times. Retain the plan's five-second warm p95 target as a target only. | N / U | Before candidate qualification | No demo device is known yet. Colab training hardware is not automatically the demo target. |
| OPEN-005 | Closed by user decision: two-source training and evaluation are outside scope | N / S | Closed | Use group-safe within-PhonemeDF protocols A and B. |
| OPEN-006 | Use placeholders A=S, B=M, C=U, D=N for roles. Actual names remain intentionally unspecified. Project week 1 day 1 is the planning anchor; replace with calendar dates when known. | N / all | Before calendar scheduling | Do not fabricate names or imply individual review signatures. G0 is closed by team attestation. |
| OPEN-007 | Split validation groups into selection, calibration and operating roles if supports allow. Estimate group-level uncertainty and achievable FPR precision before fitting. If inadequate, enlarge development data or predeclare group-aware cross-fitting. Never reuse final test for calibration or threshold choice. | N/S / M | Before split lock and any model selection | Exact group counts are unknown until pairing/source audit; no numerical precision claim can be made yet. |

Policy decisions for OPEN-001–007 are defined above. Their evidence-dependent closure remains tracked: notably source metadata, Colab/GPU resources, demo hardware, calendar kickoff, and validation-group support. The paired directory layout and Colab-first choice came from the user; they do not by themselves verify audio labels or benchmark quality.

### P1-02 evidence update — 29 October 2026

The [Colab run](../reports/phase1/p1_02/20261028T190956Z-8525fb2f/hardware.json) observed Python 3.13.16, a usable Tesla T4 through PyTorch 2.11.0+cu130, and a successful small CUDA backward check. The [environment review](environment.md) records Drive/storage findings and the missing dataset transfer. OPEN-003 now has an observed **single-run** GPU and persistent report path, but still lacks a numeric session/compute/storage budget and model throughput. OPEN-004 still lacks a named demo device; the T4 used for inventory has not been designated as one. OPEN-001 cannot proceed to paired-data audit on the selected Drive root until the missing XTTSv2 folder and 11,047 absent transcript filenames are reconciled. `pip check` reported missing `jedi` for IPython, assigned to P1-03. These updates do not close the affected OPEN entries.

**Follow-up decision, 29 October:** the user confirmed that the Google Drive dataset export was ongoing during the inventory. Accept the measured Colab environment and persistent report path as sufficient to proceed to P1-03. Treat missing paired files in that run as a transfer-progress snapshot; verify the finished copy in Phase 2 before data admission and training. `pip check` repair is a P1-03 task. OPEN-003 remains open only for measured compute/storage budget and session availability; OPEN-004 remains open for a named demo device. Do not alter the original Colab evidence or represent the transfer as complete before a new inventory.

## Compute and review controls

### P1-12 baseline planning inputs — 29 October 2026

The user selected 1,000 paired utterances / 2,000 candidate recordings, one four-second pilot crop each, and two planned three-hour Colab sessions (six hours total). Three-hour sessions are an expectation, not an official runtime entitlement. The latest user-reported free Drive space is 10.13 GB. A 25% reserve leaves 4.5 committed hours and 7.5975 GB of additional storage. The [baseline budget inputs](../configs/baseline_budget.json) make the candidate split counts, proposed epoch/overhead allowances and optional trials explicit. Complete pairing, eligibility, exact grouped splits and throughput remain unverified until their respective checks. OPEN-003 awaits benchmark/budget review and recovery readiness; OPEN-004 remains open for a named demo device. See the [baseline next steps](baseline_next_steps.md).

The project lasts eight weeks with four role owners. Eight final model/protocol combinations are mandatory. Begin neural searches with one baseline and one targeted alternative per model/protocol; LR allows eight inexpensive regularization/weighting trials. Three-seed repeats are optional, budget permitting. Count failed runs, frozen-encoder extraction, validation and final scoring. Preserve 25% of the measured available budget for failures/integration. Actual GPU hours, RAM/VRAM, storage and money limits remain OPEN-003; no paid purchase is implied.

Use W1 day 2 for G0 review, then W1 days 3/5 and two 20-minute reviews each subsequent week. Review saved evidence, risks, next gates and budget consumption. Complete B searches by end W5 where feasible, lock B early W6, and reserve remaining W6 capacity for predeclared A fits. Missed feasibility requires scope reduction, not hidden leakage or undocumented extra trials.

## Amendment record format

### P1-14 baseline handoff disposition — 30 October 2026

N owns the prepared [readiness checklist](phase1_gate_checklist.md), [handoff guide](phase1_handoff.md) and [evidence index](../reports/phase1/evidence_manifest.json); S/M/U independent review dates/findings remain pending. No research/final-test exposure was introduced by fixture or documentation work.

OPEN-003 has evidenced baseline Colab hardware, numeric provisional capacity, exact baseline dependency reproduction and fresh-VM artifact recovery. It remains open for reviewed current availability, audited-data throughput, actual training checkpoint continuation and the deferred four-model study. The six-hour allocation and 10.13 GB free-space snapshot are initial planning inputs, not an updated remaining balance. No paid compute is authorized or assumed. OPEN-004 remains pending: training on Tesla T4 does not designate a demo target. OPEN-001/002/007 still require verified source eligibility, relationships, additional data and feasible validation groups. The original full G1 and research protocol requirements remain in force; the baseline milestone supports progressing to audit, subject to later training prerequisites.

Evidence: [P1-13 review](p1_13_review.md) and [P1-14 completion review](../reports/phase1/p1_14/review.md). No source/model/dependency change or experiment refit is made by this documentation handoff. Any later source/dependency change must establish its own updated provenance and applicable rechecks. Independent reviewer disposition: not supplied.

For each change append: ID, date, author/owner, old/new policy, reason, evidence consulted, development/test exposure status, affected configs/manifests/bundles, refit/relock requirement, reviewer and actual disposition. Preserve superseded entries. Record an actual review only when it occurs. Post-test redesign cannot inherit an untouched-test claim.

## Skill application

- [data-scientist](../.agents/data-ai/data-scientist.md): problem formulation, research questions, validation responsibilities, uncertainty and honest interpretation.
- [data-engineer](../.agents/data-ai/data-engineer.md): data quality, provenance, grouping, immutable originals and unknown metadata.
- [ml-engineer](../.agents/data-ai/ml-engineer.md): shared inference, product acceptance, model lifecycle and graceful failures.
- [mlops-engineer](../.agents/data-ai/mlops-engineer.md): experiment versioning, compute reserve, ownership, locks and recovery controls.

These are the supplied Markdown role/skill guides. Their generic example SLAs, four-hour training/50 ms inference targets, p<0.05 checklist and example success figures are not EchoProof requirements or results. The project-specific plan governs. Context was supplied by the implementation plan; no unavailable context-manager service or empirical analysis was simulated.

## ChatterboxTTS Drive upload update

The user reports successful upload of an additional TTS dataset in `dataset/tts/ChatterboxTTS/`. This is a user-reported transfer update; counts, completeness, generator provenance, permissions, transcript/stem matching, speaker relationships and audio quality remain unverified. Phase 2 should inventory and audit it alongside XTTSv2 and genuine inputs. The existing 1,000-pair XTTSv2 baseline budget and historical reports remain unchanged; recheck current Drive free space and budget before expanding the cohort. The second reported generator is a candidate for unseen-generator evaluation, subject to audit and protocol feasibility.


## Phase 2 scheduling closure ? 3 November 2026

The project owner directed Phase 2 to be closed for project scheduling and Phase 3 to proceed. The owner states that the Phase 2 environment can be reproduced and will review the Phase 2 documents together. This decision does not change historical run results or constitute an independent Gate 2 review. See [Phase 2 closure](phase2_closure.md).

## R2-U1 implementation record — 6 November 2026

- **Authority:** the user requested implementation of R2-U1, owned by Usman with Muneeb as reviewer, under the existing revised plan. This records engineering interpretation; it does not record teammate approval, a new model fit, or a completed Colab run.
- **Crop bank:** consume the published R2-S1 K=5 bank exactly. The historical compute-budget assumption of four WavLM crops does not override that artifact. Preserve historical reports; record actual extraction timing in the R2 evidence.
- **Test access:** the task explicitly permits B-test genuine embeddings. Expose that as an acknowledged, logged extraction-only scope, separately indexed and excluded from default readers. Never extract Chatterbox or test TTS, score test examples, or sample them for parity. The general B-before-A firewall remains unchanged.
- **Parity:** no trained WavLM head exists in R2. Verify both embedding views directly and compare scores from an explicitly untrained deterministic diagnostic head on 50 development intervals. This is cache-path engineering evidence, not detector performance. Trained-head waveform/cache parity remains an R3-U1 obligation. Freeze absolute/relative tolerances at 1e-4; record the diagnostic definition and parameter identity.
- **Publication:** a complete cache requires validated atomic packs, exact expected coverage, separate development/test indexes, successful parity, and a final manifest. Missing, stale or corrupt artifacts fail verification rather than silently becoming training inputs.
- **Exposure/status:** this implementation task has not run pretrained extraction or examined final-test outcomes. Local Python execution is unavailable; fixture/GPU execution and Muneeb's actual review remain pending. See [the execution guide](r2_u1_wavlm_cache.md).

## R3-M1 experiment settings — 7 November 2026

Before R3-M1 training, use the baseline and one lower-learning-rate alternative specified in [the R3-M1 training guide](r3_m1_training.md). This records the implementation settings, including selection and tie rules, and does not assert teammate review or a completed model fit. The runner freezes the resolved configuration and input and source hashes before fitting. Final-test data, Chatterbox audio, and calibration or operating outcomes do not inform either checkpoint choice. Independent review by Usman remains pending.
