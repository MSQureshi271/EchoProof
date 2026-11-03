# EchoProof Phase 2 — Executable acquisition and source-verification plan

**Current decision:** The project uses PhonemeDF genuine audio and its four TTS families. External evaluation and two-source acquisition are outside scope.

**Project scheduling status, 3 November 2026:** Phase 2 closed by project owner; see [closure and handoff](../docs/phase2_closure.md). Technical gate evidence remains in the linked review.

Prepared: 30 October 2026. Status: implementation plan; tasks below have not been executed by writing this document. Governing requirements: [IMPLEMENTATION_PLAN.md, Phase 2](../IMPLEMENTATION_PLAN.md), [research contract](../docs/research_contract.md), [Phase 1 handoff](../docs/phase1_handoff.md), and the user's latest dataset clarification.

## 1. Outcome and confirmed starting point

Produce reproducible PhonemeDF inventories, provenance and license records, adapters, immutable raw manifests and count reconciliation. Every admitted recording must be traceable to original bytes and documented source metadata.

The user reports that **all downloaded audio comes from PhonemeDF**. Genuine and all four TTS folders are uploaded to Google Drive. **The user excluded MLAAD-tiny; no external dataset is selected.** Four TTS folders from PhonemeDF do not constitute independent dataset sources.

```text
/content/drive/MyDrive/PAI&AIES Project/
  dataset/
    genuine/
      wav/                 # PhonemeDF genuine candidates
      transcripts/         # corresponding text supplied with the current copy
    tts/
      XTTSv2/              # PhonemeDF synthetic candidates
      ChatterboxTTS/       # PhonemeDF synthetic candidates; pairing not yet audited
      MeloTTS/             # uploaded per user; inventory pending
      VITS/                # uploaded per user; inventory pending
```

These are reported storage locations, not a completed file/provenance audit. The user reports all four TTS folders have matching filenames and correspond to the genuine transcripts. Inventory and pairing checks must establish actual counts, coverage and speaker mappings. Existing local counts and publisher totals are reference observations; the Drive inventory determines actual availability.

| Item | Current status | Phase 2 implication |
|---|---|---|
| PhonemeDF genuine and transcripts | User reports uploaded | Inventory and verify origin/content mapping |
| PhonemeDF XTTSv2 | User reports uploaded | Verify TTS method, release, identities and pairing |
| PhonemeDF ChatterboxTTS | User reports uploaded | Verify TTS versus similarly named VC component; preserve proposed held-out role |
| PhonemeDF MeloTTS/VITS | Uploaded to Drive per user, 31 October 2026 | Inventory and admission pending; local copies not established |
| Exact downloaded PhonemeDF release | User attests Zenodo v1.0.1 | Archive checksums have not been independently matched |
| Baseline environment | P1-13 reproduced; P1-14 documentation delivered | Reuse checked code/lock, track intentional implementation changes |
| Available Drive capacity | Last report 10.13 GB, before the new upload | Obtain a fresh account quota snapshot; do not reuse it as current free space |

## 2. Applied skills and implementation principles

Use [data-engineer](../.agents/data-ai/data-engineer.md) for immutable ingestion, lineage, resumable processing, schema validation and reconciliation; [data-scientist](../.agents/data-ai/data-scientist.md) for admissible labels, independence, language controls and protocol boundaries; and [mlops-engineer](../.agents/data-ai/mlops-engineer.md) for lock provenance, resource accounting, restart recovery and artifact manifests. Project context supplies their context-assessment inputs; no unavailable context-manager service is assumed. Generic example SLAs, significance thresholds and sample performance figures in those guides are not requirements here.

Implement with existing Python libraries and standard-library file/network facilities where practical. Do not introduce distributed infrastructure for this corpus. Prefer one bounded file at a time, stable ordering, atomic checkpoints and append-only run histories. Originals are read-only. File exclusion means a manifest disposition, never deletion from `dataset/`.

## 3. Boundaries, gates and protocol decisions

### 3.1 Phase boundaries

| Work | Owner phase |
|---|---|
| Source/release/license evidence, download inventory, byte hashing, metadata mapping, representative decode checks | Phase 2 |
| Full streamed decode, canonical PCM hashes, quality measurements, exact/near-duplicate resolution | Phase 3; early overlapping execution must retain this ownership |
| Connected groups, approximately 70/15/15 allocation, validation roles, leakage assertions and sealed final sets | Phase 4 |
| Four-second windows/padding/masks, training-fitted transforms and model feature caches | Phase 5 |
| LR/BiGRU research fitting and development review | Baseline model phases |
| WavLM/AST engineering qualification | Resume P1-09–11 after the baseline development/hardware review |

Before splits, permit administrative coverage, provenance and deterministic integrity inspection. Do not select acoustic quality thresholds using the entire corpus, train shortcut classifiers, or conduct outcome-guided EDA. Those decisions use appropriate development partitions later.

### 3.2 Preserve ChatterboxTTS's proposed role

The research contract proposes MeloTTS/VITS training, XTTSv2 unseen-family validation and ChatterboxTTS final unseen-family assessment for Protocol B. The user reports all four families uploaded; their inventory, identity relationships, admission and protocol feasibility remain to be established. Auditing ChatterboxTTS metadata and deterministic integrity is permitted; using it for detector training, hyperparameter selection, quality-rule selection or error-guided development would expose the proposed final family.

Keep the initial 1,000-pair XTTSv2 baseline budget as a candidate engineering/development study. Do not silently add ChatterboxTTS or relabel this pilot as the completed four-family A/B study. Before real fitting, Phase 4 must record the pilot's role, source-group exclusions and any necessary protocol amendment. Genuine originals used in development cannot later serve as final controls alongside synthetic relatives from those same source groups. All verified family variants must contribute relationships before group assignment, even if only one family enters the pilot.

With only XTTSv2 and ChatterboxTTS, the original separate unseen-validation and final-unseen family roles are not both available alongside distinct training families. Options are to acquire the missing verified families or predeclare a narrower protocol with explicit limitations and revised selection rules. Phase 2 reports feasibility; it does not invent a replacement scientific protocol or use final tests to choose one.

### 3.3 Gate states

- `primary_audit_ready`: primary admission evidence sufficient to begin Phase 3; known limitations attached. This is an intermediate milestone, not G2 passed.
- `needs_attention`: required files, checksums, mappings, restrictions or processing checks are unresolved; identify affected scope and owner.
- `g2_ready_for_review`: all Phase 2 PhonemeDF technical criteria exist; independent review pending.
- `g2_passed`: N records the actual independent review and disposition against the full gate.

Full G2 requires verified PhonemeDF labels and methods, understood identity limits, sufficient current storage and documented restrictions.

## 4. Work packages and dependencies

S = data/features (A); M = baseline/training (B); U = compute/pretrained (C); N = evaluation/application (D). Names are placeholders. Estimates below are person-hours for implementation/review, excluding downloads and full scans; they are not allocations from the earlier six-hour experiment budget.

| Task | Work package | Owner / reviewer | Depends on | Effort |
|---|---|---|---|---:|
| P2-01 | Reconcile entry state and protocol boundaries | S/N / M | Phase 1 handoff | 1–2 h |
| P2-02 | Source/release/license evidence registry | S / N | 01 | 2–4 h |
| P2-03 | Strict audit configuration and schemas | S/M / N | 01–02 | 3–5 h |
| P2-04 | Full Drive inventory and byte hashing | S / U | 03 | 3–5 h |
| P2-05 | PhonemeDF adapter and raw manifest | S / M | 02–04 | 4–6 h |
| P2-06 | Pairing, transcripts and identity evidence | S / N | 05 | 3–5 h |
| P2-07 | Representative decode and admission check | S / M | 05–06 | 2–4 h |
| P2-08 | Reconciliation and source cards | S / N | 04–07 | 2–3 h |
| P2-09 | Current capacity and acquisition selection | U/S / M | 02, 08 | 1–2 h |
| P2-10 | Resumable acquisition and safe extraction | S/U / N | 03, 09 | 4–6 h |
| P2-11 | Combined family/identity coverage and handoff | S/N / M | 06, 08 | 2–4 h |
| P2-12 | Colab recovery and verification suite | M/U / S | 03–12 for full scope | 3–5 h |
| P2-13 | Evidence index, G2 review and Phase 3 handoff | N / S/M/U | 08, 12–13 | 2–3 h |

Execute tasks for the uploaded PhonemeDF data; serialize Drive-heavy scans. P2-11 is removed from scope.

## 5. Proposed files and executable interfaces

These paths and commands are **to implement**, not existing CLI capabilities. Add them incrementally through the tasks below. Existing available commands are listed in [Phase 1 handoff](../docs/phase1_handoff.md). No command in this plan authorizes an automatic full dataset download merely by opening a notebook.

```text
configs/phase2.yaml
configs/data_sources.json
src/echoproof/data/acquisition.py
src/echoproof/data/inventory.py
src/echoproof/data/audit_contracts.py
src/echoproof/data/adapters/base.py
src/echoproof/data/adapters/phonemedf.py
src/echoproof/data/pairing.py
src/echoproof/data/admission.py
src/echoproof/data/reconciliation.py
scripts/p2_colab_verify.py
notebooks/P2_Data_Audit_Colab.ipynb
tests/test_p2_*.py
docs/datasets/phonemedf.md
docs/data_acquisition.md
docs/phase2_gate_checklist.md
data/manifests/phase2/<snapshot-id>/
reports/phase2/<run-id>/
reports/phase2/evidence_manifest.json
```

Do not put transcripts or audio content into ordinary report logs. Manifest artifacts may include relative paths and transcript hashes; protect any original metadata containing personal information. Store downloaded source documentation separately from processing results and record its hash. No absolute private download URLs with tokens in published records.

Every data command takes `--project-root`, `--config`, explicit input snapshot IDs and a new output run directory. Emit JSON status, resolved settings, code/environment/lock hashes, start/end UTC, source IDs, actual counts, failures and artifact manifest. Use exit codes 0 completed requested scope, 2 invalid configuration/input, 3 missing dependency/source, 4 failed verification/runtime check. Distinguish interrupted, sampled and exhaustive runs.

## 6. Stage A — Establish sources and reproducible inputs

### P2-01 — Entry reconciliation

**Implementation handoff:** [collector](../scripts/p2_01_entry.py), [Colab notebook](../notebooks/P2_01_Colab_Entry_Reconciliation.ipynb), [tests](../tests/test_p2_01_entry.py) and [execution record](../docs/phase2_execution.md) are prepared. Three local tests passed and the local report verifies seven saved lock artifacts; ChatterboxTTS is missing only in the observed local copy. Actual Drive path verification and independent M review remain pending. Run the standalone notebook on Drive; no scientific-package setup is needed.

**Inputs:** latest user clarification; Phase 1 lock/reproduction and gate records; research contract.

1. Record PhonemeDF as the user-reported origin for genuine and four TTS folders; no external source is selected. Keep source verification separate from user attestation.
2. Verify the active project root, required folders and prior lock files; report missing paths without creating empty stand-in dataset directories.
3. Carry forward independent Phase 1 review obligations; do not rewrite gate records as approved.
4. Create `reports/phase2/<run-id>/entry.json` with source status, intended scope, original capacity snapshot age and ChatterboxTTS's protected proposed role.
5. Record unresolved release/archive history, identities and metadata locations. Search supplied metadata/documentation first; ask the user only for missing acquisition history that cannot be recovered from files.

**Acceptance:** one source with genuine and four TTS components, preserved originals, explicit current unknowns and no model fitting.

### P2-02 — Capture publisher and permission evidence

**Execution handoff:** P2-01 Drive entry checks passed in run `20261030T123926Z-6181b135`. A [provisional registry](../configs/data_sources.json), preparation artifacts under `reports/phase2/p2_02-preparation-20261030/`, [source-evidence guide](../docs/source_evidence.md), [collector](../scripts/p2_02_source_evidence.py) and [Colab notebook](../notebooks/P2_02_Source_Evidence_Colab.ipynb) are prepared. Two collector tests passed. Publisher pages were reviewed, but raw document capture, actual term review and local acquisition/release linkage remain pending; run the metadata-only notebook before claiming this task's full evidence closure. Entry Python 3.13.18 differs from the earlier reproduced 3.13.16; requalify numerical runtime work separately.

1. Retrieve the relevant source documentation before further audio acquisition. Resolve immutable release/repository revisions and record final URL, retrieval UTC, publisher identifiers, content hash and local evidence path.
2. Compare archive/component names with the Drive organization. Record any renaming/reorganization history; do not equate a folder label with verified origin.
3. Preserve publisher checksums with their algorithms. Compute local SHA-256 separately. An archive MD5 cannot be compared with an extracted WAV SHA-256; absence of the original archive is an explicit provenance gap.
4. Review dataset/component licenses, attribution, research use, redistribution of audio/metadata and any relevant inherited source conditions. Keep restrictions and unresolved questions separate from a legal conclusion. Record license evidence per component; a model's license does not establish the dataset's terms.
5. Build a generator registry with observed name, normalized family, version, method (`genuine`, `tts`, `vc`, `unknown`), aliases, evidence reference, verification state and planned data role. Exclude VC from this binary TTS study; ambiguous methods remain quarantined.

**Outputs:** `configs/data_sources.json`, versioned source-document copies, `source_inventory.json`, `license_inventory.csv`, generator registry. Only PhonemeDF is in scope.

**Acceptance:** every claimed mapping has a source citation/version; no invented revision or TTS label. If local release matching is unresolved, use an explicit local snapshot identifier and `publisher_revision_verified: false` in the companion provenance table.

### P2-03 — Configuration and schema implementation

Create a dedicated strict Phase 2 config loader. Do not force these settings into the fixture-only Phase 1 schema or weaken its validation. Validate unknown keys, duplicate YAML keys, relative path traversal, output/source collisions, numeric limits and Boolean flags.

Minimum config fields: schema version; source registry path; five audio roots; transcript root; output/staging roots; worker count (start one); block size; checkpoint interval; hash policy; header/decode limits; evidence-sample seed/count; retry/time limits; acquisition byte ceiling; separately supplied current Drive and scratch capacities. Unknown capacities remain null and block large acquisitions. Source roots must not overlap output roots.

Define these auxiliary tables before writing adapters:

| Table | Minimum fields |
|---|---|
| File inventory | Source/component, relative path, type, size, scan timestamps, mtime observation, SHA-256 or failure, processing state |
| Provenance | Source revision/local snapshot, publisher match state, archive origin/checksums, document/license references |
| Mapping evidence | Raw label/method/language/identity values, normalized values, verification flags and evidence |
| Pair relationships | Endpoint IDs, relation type, evidence, candidate/verified/rejected state, reviewer/disposition |
| Dispositions | Recording/file ID, reason code, stage, retained/excluded/pending, evidence and date |
| Reconciliation | Source/component/class/method/language, observed stage, denominator, count, unknown/unexecuted reason |

Reuse `Recording` from `src/echoproof/contracts.py`. It requires a real SHA-256 and a nonempty dataset revision field. Unreadable/unhashed items stay in the inventory until representable; never invent dummy hashes. Use a named local snapshot revision when publisher revision is unverified and document the distinction in provenance. Unknown identities are null. `label_verified`/`method_verified` cannot be inferred from path names. Never treat `retained` alone or `validate_supervised()` alone as the full scientific admission gate: source permissions, language and identity requirements require companion checks.

**Acceptance:** malformed configuration fails before scanning; schema round trips preserve unknown values and evidence; fixture metadata cannot mix with research IDs.

## 7. Stage B — Inventory and admit the uploaded PhonemeDF components

### P2-04 — Resumable full file inventory and hashes

1. Walk the configured roots in stable relative-path order. Record WAVs, transcripts, metadata, unexpected extensions, empty files and inaccessible paths. Detect case/Unicode-normalization collisions without silently renaming originals.
2. Stream SHA-256 bytes with bounded memory. Record file size/stat before and after reading; a changed file is unstable and retried or quarantined, not committed as a verified hash.
3. Save durable batches to unique temporary artifacts, flush and atomically commit checkpoint metadata. Log completed/pending/error counts and bytes/time. A resume selects the same source/config snapshot and validates committed chunks before reuse.
4. Reuse a prior content hash only under the declared immutable-input policy; size/mtime is a cache hint, not proof against modification. Gate verification supports a full rehash, and suspected changes invalidate affected caches.
5. Separate file identity from content equivalence: recording ID derives from source namespace, component and stable original-relative identity; byte hashes link exact-content candidates. Equal hashes must not collapse two distinct inventory rows silently.
6. Record output hashes and a deterministic content-only inventory digest excluding timestamps/run IDs. Repeating unchanged input should reproduce that digest even when processing resumes.

**Command to implement:** `data inventory --source phonemedf`, with `--resume <run-id>` for an interrupted scan. **Outputs:** file inventory, transcript inventory, hash ledger, checkpoint journal, scan summary.

**Acceptance:** every discovered path appears once as completed or explicit error; interrupted/restarted scan matches uninterrupted output; originals unchanged; neither peak memory nor console output grows with total audio bytes.

### P2-05 — PhonemeDF adapter and immutable raw manifest

1. Implement the adapter against inspected real publisher metadata, not imagined CSV columns. Preserve source fields in companion records.
2. Map genuine/XTTSv2/ChatterboxTTS/MeloTTS/VITS using evidence from P2-02. Verify Chatterbox TTS versus VC independently. Folder labels remain candidate values until supported.
3. Populate `Recording` objects with real original paths/hashes and documented values. Keep generator version, language, source speaker, output voice and reference identities null when unavailable.
4. Encode unknown or conflicting mappings as explicit pending/quarantined dispositions. Preserve identifiable malformed rows and reason codes. Do not make a bad row disappear to obtain a clean manifest.
5. Write a new immutable raw snapshot with schema/config/source/input/environment hashes. Use existing envelope Parquet/JSON helpers deliberately; document physical layout rather than assuming arbitrary columns exist in the envelope.

**Command:** `data adapt --source phonemedf --inventory <inventory>`. **Outputs:** `raw_recordings.parquet`, mapping/provenance tables, disposition ledger and snapshot manifest.

**Acceptance:** byte-identical inputs and mappings yield stable IDs; duplicate stems across components remain distinct recordings; unsupported VC/unknown methods cannot pass the admission selector. Zero final partition or crop assignment is created.

### P2-06 — Verify transcript, pairing and identity relationships

1. Index genuine WAVs, transcript files and each TTS component independently by source-relative identity. Build a full outer join of candidate stems: complete XTTS pairs, complete Chatterbox pairs, complete three-audio groups, missing transcripts, missing genuine files and orphan synthetic files.
2. Report multiplicities explicitly. Do not use an inner join that hides unpaired rows. Do not require both generators for every usable primary recording unless the later experiment explicitly selects a common cohort.
3. Hash original transcript bytes. If normalizing text for comparison, save the normalization version and a separate normalized hash; retain raw text unchanged. Handle encodings, BOMs, whitespace-only files and malformed content explicitly.
4. Verify that text/utterance relationships are supported by publisher lineage or controlled manual checks. Filename equality and text equality alone are not proof of shared spoken identity. Log review coverage and unresolved relationships; no blanket claim that listening to a sample validates all audio.
5. Distinguish source utterance/speaker, synthetic output voice and cloning reference. Parse speaker-like filename fields only after verifying the naming convention. Reference recordings are not automatically eligible genuine training examples.
6. Export verified edges and candidate edges separately. Phase 3/4 must resolve or conservatively group unresolved leakage risks; do not split first to meet class counts.

**Command:** `data pair --source phonemedf --manifest <snapshot>`. **Outputs:** pairing coverage, transcript diagnostics, identity coverage and relationship-edge table.

**Acceptance:** every raw audio row has a pairing disposition; all three components share verified relation IDs where supported; identical basenames in unrelated sources cannot create cross-source links automatically.

### P2-07 — Representative decoding and source admission

1. Select a reproducible administrative sample from each available component and structural stratum, including suspicious headers, short/long files and unmatched cases. Record the exact IDs and selection rule; this is not a training pilot.
2. Inspect headers for all inventoried audio where feasible; keep header-readable and fully decoded counts separate. Fully decode selected files in bounded blocks, checking finite samples, header/frame consistency, sample rate/channels and duration.
3. Do not impose the Phase 1 fixture decoder's 10 MB/30-second/two-channel defaults as dataset eligibility rules. Implement an audit decoder with explicit resource bounds and `resource_limit` versus `corrupt_audio` outcomes. Preserve native-rate metadata; no resampling or training segmentation is required for admission.
4. Record header/decode successes and failures with filenames and reasons. Silence/clipping can be measurements/flags; choosing exclusion thresholds belongs to later development-based cleaning.
5. Admit a component for further audit only when its source/method/label evidence is sufficient, required usage conditions are understood and basic format checks are workable. Sampled success never means every file decoded successfully.

**Command:** `data inspect --manifest <snapshot> --pairing <pairing-run> --scope representative`. **Outputs:** sample selection, header inventory, decode results, admission decision.

**Acceptance:** corrupt/truncated/nonfinite fixtures fail correctly; long valid recordings are distinguished from decoder budget failures; sampling coverage is explicit; full integrity work is scheduled for Phase 3.

### P2-08 — Reconcile counts and write primary source card

Build independent tables for files, audio recordings, transcripts, candidate/verified relationships and independent identity counts. Break down source/component, class, generator and language including unknowns. Do not sum duplicated genuine controls once per generator and call that unique corpus size.

Use the lifecycle `publisher inventory -> downloaded -> hashed/header inspected -> decoded -> in-scope -> deduplicated -> partitioned -> sampled`. Annotate each stage's scope. Deduplication, partitions and final pilot selection remain `not_run` until their phases; do not output zero to mean unknown. Publisher totals may include VC and unavailable families; reconcile against the selected component scope. Any unexplained discrepancy remains an issue, not an automatically accepted tolerance.

Write `docs/datasets/phonemedf.md` with origin/revision evidence, actual Drive counts/bytes, identity assumptions, language evidence, restrictions, quality limitations, generator roles and unresolved items. Produce machine-readable reconciliation and primary admission summary.

**Acceptance:** stage totals reconcile within the stated denominator or list every discrepancy; ChatterboxTTS is included in administrative coverage; no guide number is presented as a local measurement.

## 8. Stage C — Capacity and required components

### P2-09 — Resource plan before acquisition

Obtain current Drive account usage/free space from the user's storage view and separately measure temporary Colab disk. Record capture time and units. Linux filesystem free space on a Drive mount does not establish account quota. Confirm available audit/download work time separately from the six-hour baseline experiment allowance.

Calculate peak additional usage: incomplete download + retained archive + expanded files + metadata/checkpoints + caches + reserve. Existing originals consume current quota already; do not count them twice. Scratch space does not increase Drive quota. Use publisher byte sizes when verified; unknown expanded size requires a cautious bounded sample or explicit pending estimate. Retain the 25% reserve; do not treat compressed size alone as sufficient capacity.

Prioritize documentation and already uploaded PhonemeDF components. No external dataset is selected for acquisition. Keep original uploaded components in place. No automatic full clone, full-corpus duplicate copy or purchase. If capacity fails, report the resource need or predeclared reduced scope; do not delete originals or claim acquisition complete.

**Outputs:** `acquisition_plan.json`, capacity worksheet, selected sources/revisions/files, byte ceilings and deferred list. **Acceptance:** each planned download/extraction fits recorded storage locations and reserve or is explicitly blocked.

### P2-10 — Download and extract selected assets safely

Implement resumable retrieval with bounded retries/backoff, timeouts and rate-limit handling. Record HTTP status, redirects, content length, ETag/Last-Modified where available, and immutable source revision. Resume ranges only if the server validates the expected object; a changed object or ignored range response starts a new partial file rather than appending incompatible bytes. Save `.part` files and verify publisher checksum plus local SHA-256 before final promotion.

Preflight archive entries before extraction: reject absolute/traversal/drive-prefixed names, escaping symlinks/hardlinks, device files, duplicate/case-colliding targets and excessive entry/expanded-byte counts. Extract into a new staging directory with containment and disk-budget checks, then publish only after verification. Do not overwrite the existing uploaded dataset. Keep an archive-member-to-final-path inventory and a failure cleanup/recovery policy limited to the run's owned temporary files.

For existing uploads, import available acquisition history instead of redownloading unnecessarily. If original archives are unavailable, document that publisher archive checksums were not verified; byte hashes of extracted recordings do not repair that missing comparison.

**Commands:** `data acquire --plan <plan> --metadata-only`, then explicitly `data acquire --plan <plan> --fetch-audio`; `data extract --download <verified-download>`. **Acceptance:** corrupt downloads, range mismatches, archive escapes and collisions fail without touching originals; interruption resumes to the same verified result.

### P2-11: Removed from scope

External dataset acquisition and evaluation are not project objectives.

## 9. Stage D — Integration, verification and gate review

### P2-12 — Combined coverage and downstream handoff

Combine admitted PhonemeDF metadata with namespaced IDs, original paths, generator registries and relationship evidence. Produce a feasibility matrix for Protocols A and B.

Carry all shared-utterance/identity edges to Phase 3/4, including ChatterboxTTS relationships. Provide unresolved identity/method/label items with owners and closure requirements. Defer full duplicate disposition to Phase 3 and actual connected-component partitions to Phase 4.

Resolve the two subset sizes without conflating them: the master plan's 500–1,000-recording integration cohort is a later group-selected smoke/integration set; the user-selected baseline budget is 1,000 pairs/2,000 recordings with one four-second crop each. Neither is selected in Phase 2. After full group assignment, sample reproducibly within allowed partitions, keep related recordings together, preserve validation roles and use the same recording IDs across models within each protocol. The optional 20,000–40,000 target is conditional on measured capacity, not a current acquisition requirement.

**Outputs:** `phase3_handoff.json`, coverage/relationship summaries, protocol feasibility matrix, unresolved-issue ledger and deferred subset specification. **Acceptance:** no split/chunk/feature/model artifacts masquerade as Phase 2 deliverables; unavailable families remain visible.

### P2-13 — Colab runner and meaningful checks

Build a notebook with selectable stages for configuration, PhonemeDF inventory, adapter, pairing, inspection, reconciliation and handoff.

Use the current kernel interpreter and fresh subprocesses for package import. Reuse the Phase 1 lock as a compatibility starting point. New Phase 2 source files intentionally differ from the P1-13 source snapshot; preserve that snapshot and record new source hashes, instead of altering its expected hashes or treating the old reproduction command as approval of new code. Any dependency addition needs a new environment export/reproduction. CPU is sufficient for inventory/hash/metadata tasks; do not reserve GPU time for them by default.

Use synthetic file trees and tiny generated audio for automated checks:

| Test area | Required behaviour |
|---|---|
| Configuration | Missing roots, duplicate/unknown keys and output/input overlap fail |
| Inventory | Missing permissions, zero bytes, changing files, duplicate names and stable ordering handled explicitly |
| Resume | Interrupted checkpoint, changed config/source and tampered completed artifacts cannot silently reuse stale state |
| Adapter | Correct field mappings, unknown labels, TTS/VC distinction, duplicate IDs and schema round trips |
| Pairing | Orphans, one-to-many matches, missing transcripts, Unicode/encoding issues and distinct source/target/reference identities |
| Decoder | Valid long/stereo files, truncated/nonfinite/corrupt inputs and resource-limit classification |
| Acquisition | Changed range objects, checksum failure, failed retries, traversal/symlink/collision archives and expanded-byte limits |
| Reconciliation | Conservation of counts, sampled versus full denominators and unknown versus zero |
| End-to-end | Small PhonemeDF fixture flow completes and preserves grouping constraints |

Test mock HTTP/archive fixtures locally without bulk downloads. In Colab, verify real Drive persistence by interrupting a bounded audit task, replacing the runtime and resuming against its journal. Record recovered hashes and equal content digest. Save actual test counts, skips and errors; required skipped checks cannot be labeled ready. Do not use synthetic test rows in research inventories.

**Outputs:** runner/notebook, test modules, `reports/phase2/<run-id>/verification.json`, checksum and environment/source/lock references. **Acceptance:** requested scope passes and restart recovery is demonstrated.

### P2-14 — G2 review and closure

Write `docs/phase2_gate_checklist.md` and a dated completion review; index artifact paths/hashes in a new Phase 2 evidence manifest. Keep the Phase 1 evidence snapshot historical. Maintain separate technical status and independent N review; do not fabricate names, dates or approvals.

Full G2 checklist:

- [ ] Actual primary inventory and immutable raw manifests, including all three reported components.
- [ ] Verified source/method/label mappings and understood unresolved identity limits.
- [ ] Publisher/local provenance, acquisition history, licenses and usage restrictions recorded.
- [ ] Source cards, tested adapters and count reconciliation with explicit stage denominators.
- [ ] Current storage fits declared acquisition/processing scope with reserve.
- [ ] Required checks and restart evidence pass; no missing source is disguised as an empty pass.
- [ ] Phase 3 handoff and deferred subset/partition work are explicit.
- [ ] Actual independent review findings/disposition recorded.

Issue `g2_ready_for_review` once the PhonemeDF checklist is satisfied; record the actual independent review before passing G2.

## 10. Planned execution examples

The `data` subcommands below must first be implemented by their assigned tasks. They are not available in the current CLI. Replace angle-bracket paths with actual printed output paths, and run from the project root:

```text
python -m echoproof.cli data validate --config configs/phase2.yaml
python -m echoproof.cli data inventory --config configs/phase2.yaml --source phonemedf
python -m echoproof.cli data inventory --config configs/phase2.yaml --source phonemedf --resume <interrupted-run-id>
python -m echoproof.cli data adapt --config configs/phase2.yaml --source phonemedf --inventory <inventory-path>
python -m echoproof.cli data pair --config configs/phase2.yaml --source phonemedf --manifest <raw-snapshot>
python -m echoproof.cli data inspect --config configs/phase2.yaml --manifest <raw-snapshot> --pairing <pairing-run> --scope representative
python -m echoproof.cli data reconcile --config configs/phase2.yaml --source phonemedf --manifest <raw-snapshot> --pairing <pairing-run> --inspection <inspection-run>
python -m echoproof.cli data acquire --config configs/phase2.yaml --plan <acquisition-plan> --metadata-only
python -m echoproof.cli data acquire --config configs/phase2.yaml --plan <acquisition-plan> --fetch-audio
python -m echoproof.cli data extract --config configs/phase2.yaml --download <verified-download>
python -m echoproof.cli data handoff --config configs/phase2.yaml --primary <primary-snapshot>
python -m unittest discover -s tests -p "test_p2_*.py" -v
```

Bind notebook commands as argument lists using `sys.executable`, `cwd=PROJECT_ROOT` and explicit paths. This handles the space/ampersand in the Drive directory. Never shell-interpolate publisher filenames. Running the notebook should report its task, scope, dependencies and intended byte limits before an explicit acquisition stage.

## 11. Completion evidence and failure responses

Each task handoff reports implemented files, exact commands, source/revision/input hashes, scope, actual processed counts/bytes, tests, runtime, elapsed work, peak storage estimate, unresolved issues and reviewer status. Emit `not_run` or a structured missing reason for later lifecycle stages. Preserve failed runs and use new snapshot IDs for corrections.

| Condition | Required response |
|---|---|
| User's folder differs from expected name | Resolve config against actual location; retain original name and provenance |
| Release or checksum cannot be established | Record local snapshot and confidence gap; do not assign publisher verification |
| Chatterbox component could be VC | Quarantine until method evidence resolves it |
| Orphan or conflicting transcript/identity | Keep explicit disposition and relationship uncertainty; do not invent a match |
| Drive quota insufficient | Reduce selected acquisition/staging scope or secure documented capacity; preserve originals |
| Header sample passes but full scan not run | Report only the completed scope; schedule exhaustive integrity in Phase 3 |
| Source changes during hashing/resume | Invalidate affected work and create a reconciled snapshot |
| Two families do not support original B | Record unmet protocol requirements; acquire missing families or explicitly amend before fitting |
| Pilot requested before group assignment | Defer cohort selection/crops until Phase 4/5 |

## 12. Primary-source references and planning verification

Consulted during this planning pass; download and hash exact documentation/revisions in P2-02 before treating them as reproducible acquisition evidence:

- [PhonemeDF authors' repository](https://github.com/Vamshi-Nallaguntla/PhonemeDF).
- [PhonemeDF release record 19522979](https://zenodo.org/records/19522979): the page identifies v1.0.1 and separately lists real, XTTSv2 and ChatterboxTTS archives alongside other TTS/VC assets. Publisher archive checksums are labeled MD5. This supports the audit targets, not verification of the user's extracted files.

No audio dataset was downloaded, relabeled, admitted, split or trained as part of creating this plan. The immediate implementation task is **P2-01**, followed by source evidence and strict audit configuration before the first full Drive inventory.
