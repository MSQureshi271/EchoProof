# Phase 1 baseline readiness and full G1 checklist

Recorded: 30 October 2026 (Asia/Karachi). Owner D/N; reviewers A/S, B/M and C/U. These are role placeholders.

**Gate state: baseline readiness recorded; independent team review pending. Full G1 remains pending.** P1-14's baseline documentation and evidence consolidation are delivered. This record permits a handoff to data auditing; research training still requires the later data, preprocessing, split and feature gates. No independent signature is inferred from the agent's technical review.

Read the [handoff guide](phase1_handoff.md), [P1-13 review](p1_13_review.md), [execution history](phase1_execution.md) and [central evidence index](../reports/phase1/evidence_manifest.json). Report status strings are preserved as emitted. Failed historical runs are retained.

## Baseline technical checklist

| Check | Status | Evidence |
|---|---|---|
| G0 understanding and role mapping | Recorded by user attestation | [G0 record](phase_0_gate_checklist.md); A=S, B=M, C=U, D=N |
| P1-01 workspace preservation and scope | Evidenced; S review pending | Execution record; raw inputs preserved, no audit inferred |
| P1-02 interpreter/GPU/storage | Evidenced | Original Colab inventory/restart report; P1-13 fresh-runtime recovery |
| P1-03 editable package, imports and CLI | Evidenced | Package setup report; P1-13 installed CLI and dependency checks |
| P1-04 strict configuration and layout | Evidenced | Five configuration tests in fresh Colab runtime |
| P1-05 seven metadata contracts and model interfaces | Evidenced | Nine contract tests, including JSON/Parquet and relationship checks |
| P1-06 seeds, atomic artifacts and readback | Evidenced | Five tests, new storage probe, 31 historical files recovered |
| P1-07 bounded audio/fixtures | Evidenced | Four tests and WAV/MP3 probe; mono/16 kHz/padding/invalid input checks |
| P1-08 LR/BiGRU fixture paths | Evidenced | Four tests and both model probes; reload/padding parity and finite gradients |
| Baseline P1-12 profile | Evidenced, estimates provisional | Eleven main paths: five warm-ups and 100 measured iterations each; T4 FP32, batch four |
| Baseline P1-12 capacity | Provisionally fits declared allowance | 18.07 minutes estimated; 2.238 GB additional storage; limits 4.5 hours/7.5975 GB |
| Baseline P1-13 reproduction | Evidenced | 41 tests without skips, CPU probes and CUDA BiGRU optimizer update; same GPU/driver |
| Drive/source/lock integrity | Verified on supplied copies | 70 pins, seven export artifacts, 48 source hashes, 31 historical recovery hashes; 45 artifacts across ten runs |
| P1-14 documentation/evidence handoff | Delivered | This checklist, handoff guide, completion review and hashed evidence index |

Selected reproduction: `reports/phase1/p1_13/reproduce-20261029T192537Z-816edf2c/verification.json`. Selected lock: `locks/colab-baseline-20261029T192113Z-93db044e/requirements-lock.txt`; SHA-256 `d60c8db107fe2a94bc6ec93c493db516ab0401a3465fce38115f0738c6179c8b`.

The observed environment is Python 3.13.16, torch 2.11.0+cu130, CUDA build 13.0, Tesla T4, driver 580.82.07 and FFmpeg 6.1.1-3ubuntu5. The matching Colab torch installation was retained and verified. Installation of that torch wheel from an empty environment and offline dependency installation were not established.

## Full G1 checklist

- [x] G0 team understanding and S/M/U/N mapping recorded; independent P1-01 review is tracked separately.
- [x] Package, installed CLI and dependency checks pass in the selected fresh Colab runtime.
- [x] Baseline exact dependency/runtime/decoder provenance and lock are saved.
- [x] Configuration and shared contracts pass their implemented checks.
- [x] WAV decoding, mono conversion, 16 kHz resampling and MP3 fixture checks pass.
- [x] LR handcrafted pipeline and BiGRU temporal forward/backward/reload checks pass on generated fixtures.
- [ ] P1-09 pinned pretrained revisions, licenses, compatible processors and offline asset verification.
- [ ] P1-10 actual frozen WavLM encoder, masked pooling and direct/cache parity.
- [ ] P1-11 matched AST forward/backward probe, intended trainable layers and memory evidence.
- [x] Existing Drive artifacts survive a fresh-runtime readback with valid checksums.
- [x] Baseline 100-iteration profiling and its provisional limitations are recorded.
- [ ] Full four-model/eight-fit A/B feasibility, pretrained profiling and offline reproduction.
- [ ] OPEN-003 closed for the complete research study, including actual training recovery and reviewed capacity.
- [ ] OPEN-004 named demo hardware and ten-second end-to-end latency qualification.
- [x] Baseline fresh-runtime checks and an evidence index reference real supplied artifacts.
- [ ] Independent reviewers record actual review dates, findings and dispositions.

The user reports XTTSv2 and ChatterboxTTS uploads. Their generator provenance, relationships, eligibility and group support must be audited before determining unseen-generator Protocol B feasibility; an eligible independent external benchmark also remains unverified. No pretrained asset lock is currently present; its reserved config path is a future deliverable.

## Independent reviewer ledger

The task plan requires independent review. The table records assignments and open work, without inventing approvals or requesting another execution permission.

| Task/scope | Owner | Assigned reviewer | Actual review date | Finding/disposition |
|---|---|---|---|---|
| P1-01 entry inventory | N | S | Not supplied | Pending |
| P1-02 runtime decision | U | M | Not supplied | Pending |
| P1-03 package/bootstrap | M | U | Not supplied | Pending |
| P1-04 configuration/layout | M/N | S | Not supplied | Pending |
| P1-05 contracts | S/N | M | Not supplied | Pending |
| P1-06 reproducibility/storage | M/U | N | Not supplied | Pending |
| P1-07 fixture/audio | S | M | Not supplied | Pending |
| P1-08 baseline adapters | M/S | U | Not supplied | Pending |
| P1-12 baseline budget/assumptions | U/N | M | Not supplied | Pending |
| P1-13 baseline reproduction | M | U | Not supplied | Pending |
| P1-14 baseline handoff | N | S/M/U | Not supplied | Pending |

For each actual review, record: reviewer placeholder, date, evidence paths/lock hash, checks inspected, findings, required actions and accepted/revise disposition. A consolidated review may cover multiple named tasks if it identifies them explicitly. The agent's dated evidence reviews are documented technical checks and do not fill these team attestations.

## Handoff conditions

- [x] Source originals and earlier failed/successful evidence remain preserved.
- [x] Baseline limitations, deferred work and resource assumptions are explicit.
- [ ] Full uploaded corpus admission audit, verified labels/methods/permissions and identity relationships.
- [ ] Duplicate/near-duplicate dispositions and group-aware partitions before segmentation/augmentation.
- [ ] Training-only transformations, feasible validation roles and representative-data reprofile.
- [ ] Research training, checkpoint continuation, recording aggregation/calibration and final evaluation.

Proceed next with Phase 2 acquisition/provenance/inventory work. Record the independent team review alongside that handoff. Training begins only after its relevant later gates are satisfied. Deferred full-G1 work is revisited at the baseline development/hardware review; it is not marked passed here.
