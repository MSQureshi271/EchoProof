# Phase 1 — Executable engineering foundation plan

**Version:** 1.1 · **Prepared:** 28 October 2026 · **Window:** project week 1, maintained throughout delivery

**Status:** P1-01 is evidenced. P1-02 Colab evidence has been reviewed for progression; independent sign-off remains pending. P1-03 technical checks passed in a clean Colab rerun and await M/U review. Other commands below describe interfaces to implement unless explicitly provided. Paths describe intended outputs unless identified as existing source documents.

## 1. Objective, authority, and skill application

Implement Phase 1 of [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md): a reproducible Python environment, importable package, shared data/inference contracts, meaningful fixture-based model probes, persistent artifacts, and a measured resource plan. Preserve the [research contract](../docs/research_contract.md), [decision log](../docs/decision_log.md), and [G0 checklist](../docs/phase_0_gate_checklist.md).

Use these supplied guides for their relevant practices:

The user requests considering applicable `.agents/data-ai` guidance for this and future tasks, whether or not it is explicitly repeated.

| Skill | Application in this phase |
|---|---|
| [ml-engineer](../.agents/data-ai/ml-engineer.md) | Environment, reusable representations/adapters, finite predictions, checkpoint consistency |
| [data-engineer](../.agents/data-ai/data-engineer.md) | Typed manifests, provenance, referential integrity, path validation and durable artifacts |
| [mlops-engineer](../.agents/data-ai/mlops-engineer.md) | Dependency/model versioning, experiment evidence, resource allocation and recovery |
| [data-scientist](../.agents/data-ai/data-scientist.md) | Honest smoke-test interpretation, split units, training-only fitting and uncertainty boundaries |
| [machine-learning-engineer](../.agents/data-ai/machine-learning-engineer.md) | Hardware-specific profiling, full-path timing and deployment-interface consistency |

Their generic throughput, accuracy, significance, uptime and latency examples do not supersede EchoProof requirements. No production platform, auto-retraining service, feature store, database, Kubernetes cluster or hosted endpoint is required here. No unavailable context-manager tool should be simulated.

### Phase boundary

**Current execution amendment:** the team has copied the project to Google Drive and will train LR and BiGRU on Colab first. The user reports raw inputs at `dataset/genuine/wav/`, `dataset/genuine/transcripts/`, `dataset/tts/XTTSv2/`, and newly uploaded `dataset/tts/ChatterboxTTS/`. Preserve it in place; use configurable Drive roots and disposable `/content` working space. Matching stems link paired recordings for later verified grouping. Full pairing, speaker metadata and audio validation remain Phase 2/4 obligations. Both reported TTS families require provenance, eligibility, pairing and grouping audits before assessing the strict unseen-generator protocol.

Execute baseline tasks first: P1-02–08, a baseline-only P1-12 budget, then baseline P1-13/14 reproduction and review. P1-09–11 and pretrained portions of P1-12–14 are deferred until the team reviews baseline development results and hardware needs. Baseline readiness is an intermediate milestone, **not full G1 completion**; retain the original full G1 model requirements. Define the GPU-progression review before inspecting baseline results: leakage checks, improvement over the majority reference on grouped development data, error/shortcut analysis, and measured compute feasibility. No particular accuracy floor has been supplied. Never use final tests to decide whether to continue.

Implement the minimum actual paths needed to prove compatibility: waveform decoding/resampling; handcrafted and temporal features; pretrained processors; four forward probes; AST backward probe; schema/config validation; seeding; artifact writing; and profiling. Share these modules with later phases rather than building throwaway notebook-only logic.

Do not acquire full research datasets, create research splits, train final detectors, calibrate confidence, select a production model, or implement the full UI. Fixture-only labels test software behavior and establish no detector accuracy. Later Phases 3–5 refine audited preprocessing and feature policies without treating these fixture choices as research evidence.

## 2. Owners, order, and execution conventions

A/S = data/features; B/M = LR/BiGRU/training; C/U = pretrained/compute; D/N = evaluation/application. S/M/U/N are the user's placeholders; actual names and kickoff remain OPEN-006. Each task requires evidence and an independent reviewer. The [P1-01 execution record](../docs/phase1_execution.md) captures the entry inventory.

| Task | Deliverable | Owner / reviewer | Dependencies | Effort estimate |
|---|---|---|---|---|
| P1-01 | Entry review and workspace inventory | D / A | G0 documentation | 1–2 h |
| P1-02 | Hardware/runtime decision | C / B | 01 | 2–3 h |
| P1-03 | Package and environment bootstrap | B / C | 01–02 | 3–5 h |
| P1-04 | Layout, config and CLI | B/D / A | 03 | 3–5 h |
| P1-05 | Typed shared contracts | A/D / B | 04 | 4–6 h |
| P1-06 | Reproducibility and artifact primitives | B/C / D | 04–05 | 3–5 h |
| P1-07 | Fixture generation and waveform probe | A / B | 05–06 | 3–5 h |
| P1-08 | LR/BiGRU feature and model probes | B/A / C | 07 | 4–6 h |
| P1-09 | Pinned pretrained assets | C / D | 03, 06 | 2–4 h plus downloads |
| P1-10 | WavLM forward and cache probe | C / B | 07, 09 | 3–5 h |
| P1-11 | AST forward/backward probe | C / B | 07, 09 | 4–6 h |
| P1-12 | Baseline budget first; pretrained budget later | C/D / B | 08 initially; 10–11 for full scope | 3–5 h plus measurements |
| P1-13 | Baseline reproduction first; full model reproduction later | B / C | Completed scope of 08–12 | 3–5 h |
| P1-14 | Baseline readiness review; full G1 review later | D / A/B/C | Completed scope of 01–13 | 2–3 h |

These are planning estimates, not logged work. Parallelize human work after interfaces agree: A handles contracts/fixtures, B package and classical/recurrent probes, C assets/pretrained probes, D CLI/result contracts and evidence review. Serialize GPU benchmarks to avoid contention. Task 09 can overlap tasks 05–08. Allow download/GPU availability slack rather than promising completion from estimates alone.

### Commands and failure semantics

Use Colab Python cells for the primary training workflow; PowerShell commands below are local Windows alternatives. The Colab kernel supplies its own interpreter: record its version rather than forcing Python 3.11 or copying a Windows virtual environment to Drive. Use `sys.executable -m pip` or `%pip` for that kernel when P1-03 begins. Locally use `.venv-echoproof` to avoid replacing another environment. Preserve unrelated files. Check command return codes; existing paths must be inspected before changes.

Standard Python/pip/PowerShell commands are directly executable after their prerequisites. Every `python -m echoproof.cli ...` command below is an **interface to implement in P1-04 and the named task**. It is not currently asserted to exist. CLI help must work before probes; unsupported commands must exit nonzero, not print a fake success.

Implement exit codes: 0 verified success; 2 invalid config/input; 3 dependency or asset unavailable; 4 failed runtime assertion/probe. Emit concise console summaries and machine-readable evidence. A skipped pretrained/GPU gate is not success. Use a unique run directory for each invocation; never overwrite a previous run's measurements.

## 3. P1-01 — Establish entry conditions and safe workspace inventory

**Owner/reviewer:** D/A. **Inputs:** Phase 0 documents and main plan.

1. Read applicable repository instructions and inspect existing package/config files before implementation. Identify any name/path conflicts without altering unrelated projects.
2. Record the user's closure of G0 and the S/M/U/N role placeholders. Actual names, kickoff and individual walkthrough dates remain unspecified. Keep the G0 attestation distinct from the independent P1-01 review.
3. Create `docs/phase1_execution.md` with task status, owner, reviewer, evidence path, date, blockers, and observed deviations.
4. Register remaining OPEN-003 GPU/budget and OPEN-004 latency-device decisions as P1-02/P1-12 responsibilities. Dataset eligibility remains Phase 2 work.
5. Confirm where code, fixtures, weights, cache and run evidence may be stored. Record paths without secrets or private environment values.

**Read-only starting commands:**

```powershell
Get-Location
Get-Command py, python, ffmpeg -ErrorAction SilentlyContinue
Get-ChildItem -Force -Name
```

**Acceptance:** inventory exists, preservation decisions are explicit, owner roles have an assignment status, and no pre-existing code is represented as inspected unless actually read. The [current record](../docs/phase1_execution.md) is evidenced and awaits A/S review.

## 4. P1-02 — Select hardware, interpreter and storage

**Owner/reviewer:** U/M (C/B). **Outputs:** `docs/environment.md`, initial `reports/phase1/hardware.json`, immutable per-run evidence under `reports/phase1/p1_02/`.

1. Open [the P1-02 notebook](../notebooks/P1_02_Colab_Runtime_Setup.ipynb) in Colab and follow [the run guide](../notebooks/README.md). Mount the already-populated Drive folder and set its exact project path. Do not re-upload or copy the full dataset into `/content` for this task.
2. Collect observed Linux/Python/package versions, CPU/RAM, runtime disk, NVIDIA metadata when available, torch CUDA capability and a tiny disposable tensor forward/backward check. Retain Colab's working PyTorch build initially; defer installations and dependency locking to P1-03/13. Package import errors are recorded and block compatibility readiness where applicable.
3. Check the copied project and three dataset directories; enumerate filenames and byte totals without decoding or reading transcript text. Record paired/unpaired/ambiguous stems; this is storage/structural inventory, not a label, language or acoustic audit. When Drive export is still running, preserve the interim inventory and verify the finished copy at the Phase 2 data gate, before dataset admission.
4. Choose persistent project-relative locations for manifests, features, checkpoints, reports and future pretrained snapshots on Drive. Reserve `/content/echoproof` for disposable staging. Verify a small Drive file survives a new Python process and, in the optional restart section, a fresh Colab runtime. Drive mount filesystem free space is not authoritative account quota; enter current Drive free space manually or leave quota pending.
5. Enter a conservative storage allowance and planned session-work budget if known. Record unknown limits explicitly; Colab GPU type, runtime duration and available resources are not guaranteed. Keep the 25% compute reserve; actual training throughput/storage sizing is P1-12 work. [Colab resource and persistence FAQ](https://research.google.com/colaboratory/faq.html).
6. Save versioned evidence and the environment summary. The selected interpreter is the observed Colab kernel, provisional until P1-03 compatibility checks. CPU is valid for LR; if GPU was requested for BiGRU but is unavailable, capture the result and retry a GPU runtime or record a deliberate CPU pilot. A separate GPU machine and the eventual demo device remain unspecified until actually chosen.

The notebook is self-contained and requires no `echoproof` package or dependency installations. Each run creates its own directory. It creates canonical summary files only if absent, preserving existing human edits and older evidence. Review the newest per-run files when rerunning. P1-02 is evidence-complete only after the actual Colab output is reviewed; preparing the notebook does not close it.

Optional local Windows inventory commands (do not run these in Colab):

```powershell
py -0p
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, OSArchitecture, TotalVisibleMemorySize
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
Get-PSDrive -PSProvider FileSystem
# Only if NVIDIA tooling is available:
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

**Acceptance:** actual Colab interpreter/device and project/storage paths are recorded and persistence check succeeds. An unfinished Drive export is recorded as an interim data snapshot and checked again at Phase 2, before training. P1-03 resolves package consistency; P1-12 owns numerical throughput and storage feasibility; candidate qualification names demo hardware. Independent role review remains traceable. An unavailable GPU does not block a CPU LR pilot, but it cannot satisfy later GPU feasibility. The 12–16 GB advanced-model GPU remains a later planning target.

## 5. P1-03 — Create an installable Python package and environment

**Owner/reviewer:** B/C. **Outputs:** `pyproject.toml`, package skeleton, provisional dependency manifest.

**Implementation handoff:** [Colab package setup notebook](../notebooks/P1_03_Colab_Package_Setup.ipynb), [setup script](../scripts/p1_03_colab_setup.py), and [package setup notes](../docs/phase1_setup.md) are prepared. Sync these files, `pyproject.toml`, and `src/echoproof/` to the copied Drive project. Run the notebook on the selected Colab kernel and record its observed Python version. The notebook records actual install/import/CLI/FFmpeg checks under a new `reports/phase1/p1_03/<run-id>/` folder. Its saved result and independent M/U review determine final P1-03 status; local preparation alone does not.

1. On Colab, retain the kernel and its working torch build; install only required compatible project dependencies using that kernel, and record changes/restart requirements. Do not copy a Windows environment to Colab. Use `.venv-echoproof` only for local Windows development.
2. Implement `src/echoproof/__init__.py`, `__main__.py`, and `cli.py`. Choose setuptools package discovery under `src`, a project version, Python constraint, runtime dependencies and test extras. Use an entry point `echoproof = echoproof.cli:main` and support `python -m echoproof.cli` equally. Standard project metadata/build-system declarations belong in `pyproject.toml`. [Python packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
3. On Colab verify the preinstalled PyTorch build first; use the official installer only if replacement is necessary and documented. Start with the baseline dependency subset (NumPy, pandas, PyArrow, SciPy, scikit-learn, librosa, SoundFile, PyYAML and pytest). Add Transformers/huggingface_hub when pretrained work begins and Streamlit when application work begins. Retain Matplotlib/Seaborn for analysis. Check compatibility rather than upgrading all preinstalled libraries.
4. Install FFmpeg through the chosen platform mechanism and record its exact version/build; verify encoder/decoder support required for MP3. It is an external executable, not satisfied merely by a similarly named Python package.
5. Run import checks and `pip check`. Dependency conflicts block asset probes. Freeze exact versions only after probes pass in P1-13.

```powershell
# Local Windows alternative. In Colab, run the linked P1-03 notebook instead.
# Substitute the interpreter chosen in P1-02 if different.
py -3.11 -m venv .venv-echoproof
.\.venv-echoproof\Scripts\python.exe -m pip install --upgrade pip
# Execute the recorded official hardware-specific torch installation command here,
# using this environment's python.exe -m pip rather than global pip.
.\.venv-echoproof\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv-echoproof\Scripts\python.exe -m pip check
.\.venv-echoproof\Scripts\python.exe -m echoproof.cli --help
ffmpeg -version
```

**Acceptance:** import and CLI help work without manipulating `sys.path`; dependency checks pass; package metadata does not silently replace the selected torch build; decoder location/version is documented.

## 6. P1-04 — Layout, validated configuration and CLI dispatch

**Owner/reviewer:** B/D with A reviewing. **Dependency:** P1-03.

**Implementation handoff:** the configuration validator, CLI dispatch, storage policy, model interface stubs and [Colab verification notebook](../notebooks/P1_04_Colab_Verification.ipynb) are prepared. The notebook invokes the P1-03 setup script in the same runtime before P1-04 checks; a separate P1-03 notebook run is unnecessary. Sync both setup scripts and run the notebook, then review its evidence before recording P1-04 acceptance. Actual fixture registration and loading remain P1-07 work; `validate-config` itself does not read audio.

Create the directories described in Phase 1 of the main plan: `configs`, `src/echoproof/{data,features,models,training,evaluation}`, `app`, `tests/fixtures`, `data/{manifests,splits,processed,features}`, `runs`, `bundles`, `reports`, and `docs`. Keep the existing `dataset/` tree as raw input; a new `data/raw/` copy is unnecessary. Add configurable roots for the three input folders and separate Drive persistence from `/content` staging. Do not create empty “completed” implementations for later phases. Add a fixtures README and artifact-storage policy. Exclude `dataset/`, environments, private audio, caches, weights, temporary files and large runs from Git while retaining small schemas/configs and selected evidence summaries. Merge ignore rules rather than replacing unrelated rules.

Implement:

- `config.py`: safe YAML loading, typed validated fields, explicit relative-path resolution, unknown-key rejection, finite numeric checks, label mapping validation and resolved-config export.
- `cli.py`: subcommand parser, structured errors, common config/run-root/device options, and dry validation through `validate-config`.
- `configs/phase1.yaml`: fixture settings, device policy, seed, storage roots, pretrained asset-lock path, probe batch sizes and timing limits.
- Model-specific config stubs containing declared interfaces/defaults, marked experimental until later training. Do not assign synthetic hashes to absent research manifests.

Minimum Phase 1 configuration:

```yaml
schema_version: 1
mode: fixture_probe
seed: 42
labels: {genuine: 0, tts: 1}
device: cpu
audio:
  sample_rate: 16000
  crop_samples: 64000
  mono_policy: mean_channels
paths:
  fixtures: tests/fixtures/generated
  run_root: runs/phase1
  asset_lock: configs/pretrained_assets.json
probe:
  ast_microbatch: 2
  warmup_batches: 5
  measured_batches: 100
  precision: fp32
```

`mode=fixture_probe` bypasses research split requirements only for registered generated fixtures. It must refuse research audio/manifests. Every emitted metric is tagged `software_probe`, never research performance.

**Command:** `python -m echoproof.cli validate-config --config configs/phase1.yaml` using the environment interpreter.

**Acceptance:** valid config exports deterministically; wrong labels, unknown fields, invalid dimensions and escaping output paths fail. CLI implementations are connected incrementally as their tasks complete.

## 7. P1-05 — Implement shared schemas and adapter contracts

**Owner/reviewer:** A/D with B reviewing. **Outputs:** `contracts.py`, `docs/contracts.md`, contract tests and small schema fixtures.

**Implementation handoff:** the shared contracts, documentation, fixture metadata, and [Colab verification notebook](../notebooks/P1_05_Colab_Verification.ipynb) are prepared. The notebook performs the P1-03 setup in the same runtime and requires all contract checks to pass without skips. Local Python lacks NumPy and PyArrow, so their checks await Colab. Review the resulting evidence before marking P1-05 complete.

Use typed dataclasses plus explicit validation, or a deliberately pinned schema library if justified. Keep core tensor contracts separate from serialized metadata. Version every persisted schema. Define these contracts before models diverge:

| Type | Required information | Invariants |
|---|---|---|
| Recording | ID, relative original path, dataset/revision, nullable verified label/method, family/version, language, source utterance/speaker, target voice/reference, transcript hash, duration/rate/channels/codec, checksums, quality flags, retention/exclusion, license | Unique ID; valid relative path; unknown identities null; retained supervised records require verified label and method |
| SplitAssignment | Protocol, recording/group ID, partition, validation role, family policy, version/hash | One assignment per recording/protocol; validation role only for validation; no orphan IDs |
| Chunk | ID, parent recording/group/split, start/end samples and seconds, valid samples, preprocessing hash | Positive interval; offset units agree; valid length ≤ input length; parent exists |
| FeatureBatch | Tensor/array, chunk/recording IDs, optional labels, lengths/masks, schema hash | Batch dimensions agree; finite valid values; mask/length consistency |
| PredictionRow | Run/bundle/protocol/recording IDs, raw/calibrated TTS score, decision rule, prediction, status, duration/chunk count | Successful scores finite in [0,1]; failures retain status and null score |
| InferenceResult | Assessment, score type, warnings, model version, coverage, timings, failure reason | No score for invalid input; coverage ≤ decoded duration; JSON-safe fields |

Retained unknown-label records may be audit candidates but cannot enter supervised fitting. IDs are namespaced by source. Never use a literal shared `unknown` as a graph identity. Preserve an explicit decision-rule variant for all-negative threshold policies from Phase 0; do not encode infinity as invalid JSON or silently clamp thresholds.

Define a `ModelAdapter` protocol with immutable metadata, model-specific feature preparation, `score_chunks` returning ordered class-1 scores, and load/save responsibilities. Require a mapping back to every chunk ID; shape compatibility alone does not establish correct ordering. `predict_recording` receives bytes and a validated bundle specification; in Phase 1 define its result/error contract without pretending a production bundle exists.

**Tests:** valid records round-trip through JSON/Parquet; duplicate IDs, escaped paths, invalid class labels, orphan chunks, mismatched lengths, nonfinite scores and failure-with-score are rejected. Confirm null identities survive serialization.

**Acceptance:** all owners can consume the documented contracts; no independent duplicate result schema is introduced in the app.

## 8. P1-06 — Reproducibility, evidence logging and durable artifacts

**Owner/reviewer:** B/C with D reviewing. **Outputs:** `training/seeding.py`, `artifacts.py`, `environment.py`, unit tests.

**Implementation handoff:** seeding, runtime capture, run artifacts, the `doctor` and `storage-probe` commands, and [Colab verification notebook](../notebooks/P1_06_Colab_Verification.ipynb) are prepared. The notebook performs P1-03 setup in the same runtime and exercises actual Drive storage. Local tests lack NumPy/PyTorch; review the Colab report before accepting P1-06. The user reports that the XTTSv2 Drive upload is complete; Phase 2 still verifies the data and pairs.

Implement Python/NumPy/torch seeding, DataLoader generator and worker seeds, and named seeds for fixtures/crops/augmentation. Put Windows worker entry code behind a main guard; start at zero workers, then verify worker seeding where supported. Record deterministic settings and unsupported operations; do not promise cross-device bitwise identity.

Create unique run directories with `resolved_config.yaml`, `environment.json`, `events.jsonl`, `metrics.json`, `artifact_manifest.json` and final `status.json`. Capture UTC time, task/run ID, schema/config hashes, source revision when available, dependency-lock status, CPU/GPU/runtime/decoder versions, seeds, precision, batch settings and elapsed time. A missing Git revision is explicitly unavailable, not fabricated.

Use canonical serialization for hashes; document excluded volatile fields. Write temporary files in the destination directory and replace atomically after successful close. Hash completed artifacts and refuse accidental overwrite. Test interrupted writes do not leave a completed-success marker. Output storage must support a new process reading and validating saved files; do not claim durability based only on objects remaining in memory.

**Commands to implement:**

```text
python -m echoproof.cli doctor --config configs/phase1.yaml
python -m echoproof.cli storage-probe --config configs/phase1.yaml
```

**Acceptance:** two fresh seeded fixture runs produce equivalent outputs within declared tolerance; config changes alter the hash; a separate process reloads/checksums the storage probe; no secrets or uploaded audio enter logs. Checkpoint resume design is documented here; full optimizer/scheduler resume belongs to later trainer implementation.

## 9. P1-07 — Generate fixtures and prove the waveform path

**Owner/reviewer:** A/B. **Outputs:** fixture generator, `data/audio.py`, audio probe evidence and tests.

**Implementation handoff:** generated fixture and waveform modules, CLI commands, [probe notes](../docs/audio_probe.md), and a [Colab verification notebook](../notebooks/P1_07_Colab_Verification.ipynb) are prepared. The notebook performs setup and the WAV/MP3 checks in one runtime. Actual Colab evidence and B/M review remain pending.

Generate deterministic, tiny waveforms in code: one-second and four-second mono tones/noise mixtures, a non-16-kHz input, unequal-length examples, stereo channels, silence, empty bytes and corrupt bytes. Save provenance as algorithm/version/seed/checksum. Mark them synthetic test signals, not examples of genuine or TTS speech. Keep research dataset paths out of this phase.

Implement SoundFile decoding to finite float32, channel validation/mean conversion, a tested resampler to 16 kHz, valid sample lengths and right-padding for fixture probes. Preserve original fixture bytes. No denoising or peak normalization. Assert duration/resampling tolerance, finite amplitudes, shape, original-versus-output sample rate and channel metadata. Add the MP3 encode/decode round trip through FFmpeg with fixed argument arrays and bounded duration/time; record version and lossy tolerances.

```text
python -m echoproof.cli fixtures create --config configs/phase1.yaml
python -m echoproof.cli probe audio --config configs/phase1.yaml
```

**Acceptance:** WAV and tested MP3 paths work, corrupt/empty input produces structured errors, short input has correct valid length, stereo handling is explicit, and fixtures are reproducible. Silence may decode successfully while receiving a quality flag; distinguish decoding from eligibility.

## 10. P1-08 — Prove LR and BiGRU representations

**Owner/reviewer:** B/A with C reviewing. **Outputs:** reusable feature modules, minimal model adapters, probe reports.

**Implementation handoff:** the handcrafted/log-mel extractors, fixture LR/BiGRU adapters, probe commands, [probe notes](../docs/baseline_fixture_probes.md), and [Colab verification notebook](../notebooks/P1_08_Colab_Verification.ipynb) are prepared. They operate on generated signals and dummy labels only. Colab evidence and independent C/U review remain pending.

### LR fixture probe

Implement ordered summary features for MFCCs, deltas, centroid, bandwidth, rolloff, flatness, ZCR, RMS and voicing. Define handling of unvoiced/short inputs, missing features and padding. Return named columns with a schema hash. Fit an imputer/scaler/LR pipeline on a tiny generated fixture table with explicit **software-only dummy labels**, then call `predict_proba` and resolve the class-1 column from `classes_`. Save/reload the pipeline and compare scores. Do not compute or present detector accuracy from these labels.

### BiGRU fixture probe

Generate log-mel sequences with 64 bins, FFT 512, window 400 and hop 160; explicitly define centering/log floor and derive valid frame counts. The fixture-only normalization statistics must be tagged disposable and never reused for research training.

Instantiate the intended two-layer bidirectional GRU with hidden size 128 per direction, dropout 0.3, packed sequences, masked pooling and two-logit head. Run unequal-length samples alone and in batches in eval mode. Use a disposable backward step with dummy labels to verify intended gradients and finite loss; this is a software check, not a trained detector.

```text
python -m echoproof.cli probe lr --config configs/phase1.yaml
python -m echoproof.cli probe bigru --config configs/phase1.yaml
```

**Acceptance:** finite feature values after declared imputation, feature-order persistence, finite `[B,2]` logits, class-1 scores, correct lengths and eval-mode padding invariance. Baseline runs on CPU. Every artifact is labeled fixture-only and excluded from future model selection.

## 11. P1-09 — Pin, retrieve and verify pretrained assets

**Deferred stage:** start P1-09–11 after the baseline review; these tasks remain required for full G1. No pretrained weights need downloading in P1-02.

**Owner/reviewer:** C/D. **Outputs:** `configs/pretrained_assets.json`, `docs/pretrained_assets.md`, checksum inventory.

Register `microsoft/wavlm-base-plus` and `MIT/ast-finetuned-audioset-10-10-0.4593`. Resolve each repository revision to an immutable commit, record model/processor configs and license references, and download the matching required files to a persistent approved path. Record actual resolution date and local hashes. Do not invent revision hashes or treat `main` as a final pin.

Use the Hub download API's revision parameter for exact snapshots. Cached files must not be modified in place. [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

Implement two explicit steps so subsequent commands cannot silently fetch a changed latest version:

```text
python -m echoproof.cli assets resolve --config configs/phase1.yaml
python -m echoproof.cli assets fetch --lock configs/pretrained_assets.json
python -m echoproof.cli assets verify --lock configs/pretrained_assets.json --offline
```

Resolve records concrete commits; fetch requires those commits. Resume interrupted transfers safely and verify all required model/processor files. Use only trusted checkpoint formats and avoid arbitrary remote-code execution. Preserve license notices and document usage/distribution uncertainties.

**Acceptance:** matching model/processor assets load offline from the lock, checksums verify, license sources are recorded, and missing assets fail clearly. Network unavailability blocks G1 pretrained evidence; it does not justify substituting random weights without disclosure.

## 12. P1-10 — Frozen WavLM forward and cache compatibility

**Owner/reviewer:** C/B. **Outputs:** WavLM adapter/probe, valid-length pooling helper, report.

1. Load the exact local snapshot and processor; use 16 kHz valid waveforms from P1-07.
2. Freeze encoder parameters, use eval/no-gradient mode, and inspect the pinned processor's supported attention-mask behavior.
3. Convert waveform valid lengths through the encoder's convolutional length mapping; mean-pool only valid hidden frames. Verify single/equal-length extraction first, then variable-length batches if supported without contamination.
4. Instantiate a disposable head `D→128→2`, using GELU/dropout between linear layers; check score shapes and label order without asserting training performance.
5. Save a pooled embedding with a key containing audio checksum, coordinates, valid length, processor/checkpoint revision and pooling config. Reload and compare with direct extraction.

**Command:** `python -m echoproof.cli probe wavlm --config configs/phase1.yaml`.

**Acceptance:** actual pretrained encoder runs, frozen parameters stay unchanged, finite pooled embeddings match the model dimension, cached/direct results agree, and masks operate in hidden-frame units. Document tested tolerances and any single-item extraction restriction. Full encoder extraction cost is measured separately from head cost.

## 13. P1-11 — AST forward and backward memory feasibility

**Owner/reviewer:** C/B. **Outputs:** AST adapter/probe, parameter/gradient and memory evidence.

1. Load pinned extractor/model; preserve checkpoint grid/normalization (initially 128 mel bins and 1,024 frames). Feed valid fixture waveforms to its extractor rather than BiGRU feature tensors.
2. Replace only the task head with two outputs. Capture loading diagnostics; unexpected encoder/positional reinitialization blocks the task.
3. Verify finite `[B,2]` logits at microbatch 2. Short waveforms still occupy the matched grid; do not assume reduced attention memory.
4. Run a disposable head-only backward step, then a separate disposable partial-adaptation probe with the final two Transformer blocks, appropriate final normalization and head trainable. Construct optimizer groups after freezing/unfreezing.
5. Record trainable names/counts, nonzero intended gradients, frozen parameter checksums, finite loss and changed intended parameters after an AdamW step. Keep frozen submodules deterministic when invoking train mode.
6. Reset/synchronize accelerator memory statistics and time the actual forward/backward/optimizer path. Record precision, peak allocated/reserved VRAM and process memory. OOM at batch 2 is evidence: retry batch 1 with a documented reason and accumulation feasibility, never suppress the failure.
7. Destroy or clearly label disposable probe weights so they cannot become research checkpoints.

**Command:** `python -m echoproof.cli probe ast --config configs/phase1.yaml`.

**Acceptance:** actual matched pretrained AST forwards and backpropagates through intended backbone layers; memory evidence identifies hardware and settings. CPU success can establish API compatibility but cannot establish a feasible GPU training budget. If no practical training hardware exists, G1's compute readiness remains blocked. Neither probe replaces the required later head warm-up/partial-fine-tuning experiment.

## 14. P1-12 — Measure throughput and establish a feasible budget

**Owner/reviewer:** C/D with B reviewing. **Outputs:** `reports/phase1/benchmark.json`, `reports/phase1/compute_budget.csv`, updated environment/decision records.

**Baseline implementation handoff:** [profiling/budget notebook](../notebooks/P1_12_Colab_Verification.ipynb), [runner](../scripts/p1_12_colab_verify.py), [candidate budget](../configs/baseline_budget.json), and [next steps](../docs/baseline_next_steps.md) are prepared. User-selected planning inputs are 1,000 paired utterances / 2,000 recordings, one four-second crop each, two expected three-hour sessions, and 10.13 GB free Drive space. The template proposes 20 epochs and explicit overhead allowances; these are planning assumptions. The notebook records actual hardware and new timings. The supplied Colab report records six passing tests and complete fixture profiling on Tesla T4; the recomputed budget is provisionally within capacity (18.07 minutes and 2.238 GB). Baseline technical evidence is recorded in `docs/phase1_execution.md`; independent B/M review remains pending. Proceed to baseline P1-13; the full pretrained portion stays deferred. Canonical summary files are created only if absent; new per-run files are authoritative after reruns.

Implement an isolated benchmark runner with fixed fixtures, five warm-up batches and approximately 100 measured batches per applicable path. Record actual counts if the time budget truncates a run and mark the estimate provisional. Synchronize accelerator timings; reset peak memory per path. Exclude one-time download from warm timing while reporting download/load separately. Serialize runs on shared devices.

Measure LR extraction/scoring, BiGRU features/forward/backward, WavLM processor/encoder/pooling/head, and AST extraction/forward/backward/optimizer. Include p50/p95, sample/batch counts, batch size, precision, worker/thread count, hardware and input duration. Benchmark cold loading separately. Fixture timings are engineering estimates, not real-data throughput or final app latency claims. Repeat with representative real data in week 4.

Run a baseline-only first pass with LR/BiGRU on the actual Colab runtime. Budget current single-family grouped development separately from the future four-model A/B study. The user now reports a ChatterboxTTS upload; its eligibility and Protocol B feasibility must be verified before the future eight-fit study. Re-profile whenever Colab assigns different hardware. Session limits and Drive quota cannot be inferred from local free disk.

```text
python -m echoproof.cli benchmark --config configs/phase1.yaml --models lr bigru --device auto
python -m echoproof.cli budget --config configs/phase1.yaml --benchmark <new-run>/benchmark.json --plan configs/baseline_budget.json
```

The full four-model benchmark interface is deferred with P1-09–11. The budget command preserves the future eight-fit A/B requirement as unestimated work; it cannot qualify that study from baseline-only measurements. For the notebook's current-session budget, a null expected device is filled with the actually measured device name and recorded in `budget_inputs.json`; capacity does not transfer automatically to a different device.

Budget command requirements:

- Accept explicit candidate recording/crop counts, epochs, effective batches, validation cadence, hardware availability and storage budget. Missing values remain unresolved; no invented availability.
- Estimate training steps as `ceil(training_crops/effective_batch)` and include validation/extraction overhead. Distinguish microbatch timing from optimizer-step timing with accumulation.
- Enumerate at least eight final fits: four models × A/B. Treat final runs as selected development runs when reused, avoiding double counting; count any actual refit separately.
- Include baseline plus one neural alternative per model/protocol, LR's bounded eight-candidate grid, AST warm-up and adaptation, failed trials, final scoring, and optional seeds separately.
- Keep planned committed usage ≤75% of recorded available compute; reserve 25% for reruns/integration. If the total exceeds capacity, reduce declared shared cohort or optional searches before starting expensive work.
- Estimate raw float32 storage using 64,000 bytes per second at 16 kHz mono; count cache/checkpoint/optimizer/environment overhead separately. Prefer pooled WavLM embeddings over hidden-sequence caches.
- Reserve sequential schedule space for B locking before A development exposes B's held-out family.

Close OPEN-003/004 only with actual recorded hardware/budget decisions. **Acceptance:** numeric estimates trace to measurements and assumptions, available capacity is concrete, and infeasibility has a documented action. Never claim five-second application latency from head-only inference timing.

## 15. P1-13 — Freeze and reproduce the environment

For the baseline milestone, export a Colab-specific package/runtime snapshot and re-run baseline checks in a fresh Colab runtime. Use the kernel's actual Python version and preserve accelerator-build provenance. Recreate Drive mounts/paths after reconnecting; do not copy `.venv-echoproof` from Windows. Pretrained/offline probes below apply when that deferred stage begins. Record baseline-only reproduction without claiming full G1.

**Owner/reviewer:** B/C. **Outputs:** platform-specific dependency locks, install instructions, verification report.

**Baseline implementation handoff:** use the [P1-13 notebook](../notebooks/P1_13_Colab_Environment_Reproduction.ipynb), [runner](../scripts/p1_13_colab_verify.py) and [guide](../docs/environment_reproduction.md). Export from the successful GPU runtime into a new `locks/colab-baseline-<id>/`, then replace the Colab VM and reproduce from that exact export. The runner preserves baseline dependency/source/artifact provenance, rejects same-VM reuse, and requires passing baseline tests/probes without skips plus a CUDA BiGRU optimizer step. Actual Colab export/reproduction and independent M/U review remain pending. A different assigned GPU requires budget reprofiling.

**Subsequent evidence review:** [P1-13 review](../docs/p1_13_review.md) verifies the successful export and fresh-runtime reproduction: 41 tests passed without skips, CPU/CUDA probes passed, the 70-package lock matched, and supplied source/recovery/run hashes verified. Baseline technical reproduction is evidenced; proceed to P1-14. Independent M/U sign-off and the deferred full-model scope remain pending. The assigned GPU/driver match the exported runtime.

After probes pass, export exact installed dependencies excluding editable local-path entries into a checked lock file. Record Python/pip/build-tool versions and the explicit torch source/wheel provenance, since package names alone may not reproduce an accelerator build. Record FFmpeg and driver separately. Maintain separate CPU/CUDA or OS locks if necessary rather than claiming one freeze is portable.

Implement `export-environment` to generate the filtered lock and environment inventory without secrets, and `verify-environment` to compare required packages/runtime with the lock. Hash the finalized lock into later run records. Do not claim earlier provisional probes used the final lock; recheck them in the clean environment.

```text
python -m echoproof.cli export-environment --output locks/colab-baseline-<new-id>
```

Fresh PowerShell verification, using P1-02's selected Python and recorded torch index/install instructions:

```powershell
py -3.11 -m venv .venv-echoproof-verify
# Install the recorded torch build using its documented source first.
.\.venv-echoproof-verify\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv-echoproof-verify\Scripts\python.exe -m pip install --no-deps -e .
.\.venv-echoproof-verify\Scripts\python.exe -m pip check
.\.venv-echoproof-verify\Scripts\python.exe -m pytest -m "not pretrained and not gpu"
.\.venv-echoproof-verify\Scripts\python.exe -m echoproof.cli verify-environment --lock requirements-lock.txt
.\.venv-echoproof-verify\Scripts\python.exe -m echoproof.cli probe all --config configs/phase1.yaml --offline
```

Use exact locked build requirements or document them when build isolation installs tooling. Do not delete existing environments to make these commands work. The verification environment can be on the named training machine if local hardware cannot execute the pretrained probes; record that distinction.

**Acceptance:** a second environment loads assets offline, runs all required probes and confirms contract/persistence parity; selected torch build is preserved. Failed/skipped tests are not relabeled passing.

## 16. Verification matrix and run evidence

Register pytest markers `pretrained` and `gpu`. Default lightweight tests require no network/large weights. Pretrained tests use locked local files; GPU tests state device requirements. Explicit G1 verification treats unavailable required probes as blockers even if a developer's routine test run legitimately skips them.

| Test module to implement | Meaningful checks |
|---|---|
| `test_config.py` | Invalid keys/labels, stable hashes, relative path resolution |
| `test_contracts.py` | Null IDs, round trips, orphan chunks, dimensions and failure score rules |
| `test_audio.py` | Decode/resample lengths, stereo, nonfinite/corrupt/empty handling |
| `test_features.py` | Named feature order, unvoiced missing values, finite transforms, valid frames |
| `test_reproducibility.py` | Seeds, loader workers where supported, deterministic fixture generation |
| `test_artifacts.py` | Atomic writes, checksums, fresh-process reload, overwrite prevention |
| `test_lr_probe.py` | Class-1 mapping, pipeline save/reload parity |
| `test_bigru_probe.py` | Packed lengths, padding invariance, intended gradients |
| `test_wavlm_probe.py` | Frozen behavior, hidden-length pooling, direct/cache parity |
| `test_ast_probe.py` | Compatible grid, head-only replacement, intended gradients/updates |
| `test_cli.py` | Help, failure exit codes, run reports and no fake successful stubs |

Use development fixtures to set explicit floating-point tolerances per dtype/device. Avoid demanding exact CPU/GPU equality. Keep test durations bounded. No speculative research accuracy tests belong in Phase 1.

Each required probe must emit: task/run ID, UTC timestamps, input/checkpoint hashes, config/environment references, input/output shapes, status, finite-value checks, actual assertions performed, timings and failure details. A central `reports/phase1/evidence_manifest.json` indexes these run artifacts by path and hash. Large weights stay out of the report directory and Git.

## 17. P1-14 — Documentation, gate review and handoff

**Owner/reviewer:** D with A/B/C. Write `docs/phase1_gate_checklist.md`, update `docs/phase1_execution.md`, and document setup/run order in the project README or a focused linked setup document without overwriting unrelated content.

**Baseline P1-14 delivery:** [readiness checklist/reviewer ledger](../docs/phase1_gate_checklist.md), [setup and Phase 2 handoff](../docs/phase1_handoff.md), [completion review](../reports/phase1/p1_14/review.md), and [hashed evidence index](../reports/phase1/evidence_manifest.json) are prepared from supplied runs. Baseline technical readiness is recorded; independent S/M/U review remains pending. Full G1 and the deferred pretrained/eight-fit scope remain pending. Next implement the Phase 2 corpus audit before later preprocessing, split, feature and training work. No extra Colab run is required for this documentation delivery.

Required instructions cover: interpreter setup, hardware-specific torch install, FFmpeg, editable package installation, fixture generation, asset resolution/fetch/offline verification, unit/probe commands, evidence locations, environment reproduction and common failures. Explain that research training commands and calibrated production inference are later work.

### G1 acceptance checklist

**Intermediate baseline readiness:** complete P1-02–08, LR/BiGRU profiling, Colab environment reproduction, and Drive recovery checks; record pending advanced-model tasks explicitly. This enables later audited baseline training without claiming full G1. The checklist below remains the full gate.

- [ ] G0 team attestation and S/M/U/N placeholder mapping are recorded; independent P1-01 review is still documented separately.
- [ ] Installable package and CLI work in an isolated selected environment.
- [ ] Dependency checks pass; exact platform/runtime/decoder provenance is saved.
- [ ] Configs and seven shared data/model/result contracts validate and serialize correctly.
- [ ] Fixture WAV decoding, mono conversion and 16 kHz resampling succeed; MP3 decoder check documented.
- [ ] LR handcrafted features/pipeline execute and reload on software-only fixtures.
- [ ] BiGRU temporal features and packed forward/backward probe succeed.
- [ ] Frozen WavLM actual pretrained processor/encoder forward and pooling/cache checks succeed.
- [ ] Matched pretrained AST forwards and backpropagates through declared trainable layers; peak memory recorded.
- [ ] Model revisions and licenses are recorded; verified assets reload offline.
- [ ] Artifact storage survives fresh-process reload with valid checksums.
- [ ] Approximately 100-batch profiling or explicit provisional limitations are reported.
- [ ] Eight-fit budget, optional trials, storage and 25% reserve fit measured available resources or show a blocker.
- [ ] Named training/demo devices and benchmark settings resolve OPEN-003/004 where evidence supports it.
- [ ] Clean-environment tests/probes pass and evidence manifests reference real artifacts.
- [ ] Independent reviewer records actual review date, findings and disposition.

**Gate states:** baseline readiness recorded; full G1 ready for review; full G1 passed; or blocked with task/owner/recovery action. Deferred pretrained evidence is explicitly pending at the baseline milestone. Full G1 cannot pass until its required evidence exists.

### Failure response table

| Failure | Action |
|---|---|
| Conflicting existing Python/package layout | Isolate EchoProof paths; integrate deliberately; preserve unrelated files |
| Unavailable CUDA / wrong torch build | Check actual driver/build pairing; record CPU compatibility separately; use reserved capable hardware |
| Missing processor dependency | Inspect pinned extractor traceback; add compatible dependency, rerun checks and relock |
| Asset download/network failure | Resume from pinned revision, verify hashes; do not substitute unpinned/random weights |
| AST positional/encoder mismatch | Restore matched config/grid; allow only task-head replacement |
| AST OOM | Record failed settings; reduce microbatch and assess accumulation; reprofile before budgeting |
| Padding changes predictions | Correct valid-length mapping/packing; use single-item pretrained extraction until verified |
| Artifact reload/checksum failure | Fix durability/path policy; block reproduction claims |
| Budget exceeds availability | Cut optional trials, reduce shared sample scale, secure documented hardware; preserve eight fits and protocol separation |

### Phase 2 and later handoff

Phase 2 receives the working environment, recording schema, bounded decoder, provenance/hash helpers and persistent storage policy. Phase 4 receives split-assignment contracts, not fabricated research partitions. Phase 5 receives tested starter representations to refine against actual data. Model phases receive compatible adapters and measured feasibility; they still owe real training, calibration, metrics and proper bundles. Phase 12 receives the shared inference-result interface, not a claimed working detector application.

## 18. Completion report template

At execution completion, report tasks completed, files added/modified, actual test/probe counts, environment/device, checkpoint revisions, evidence manifest location, measured resource summary, open blockers and G1 disposition. Identify fixture-only results and unmet requirements explicitly. Do not substitute library benchmark figures for EchoProof measurements.

This plan is complete when its tasks are implementable and traceable. Phase 1 itself is complete only when the evidence and actual reviews above exist.
