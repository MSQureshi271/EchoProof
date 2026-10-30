# EchoProof baseline setup, recovery and Phase 2 handoff

Prepared for P1-14 on 30 October 2026. Start with the [readiness checklist](phase1_gate_checklist.md). Baseline engineering evidence is recorded; independent review and the deferred full G1 remain pending. The current code provides fixture/configuration/contract/profiling tools. Research fitting, calibrated recording inference and the upload application belong to later phases.

## Selected environment and durable files

Use the Colab kernel's interpreter via `sys.executable`; the reproduced runtime is Python 3.13.16 on Linux. Do not copy a Windows virtual environment to Drive. The project root is `/content/drive/MyDrive/PAI&AIES Project`; `/content/echoproof` is disposable scratch.

Keep `dataset/` immutable. Genuine WAVs are under `dataset/genuine/wav/`, transcripts under `dataset/genuine/transcripts/`, and synthetic candidates under `dataset/tts/XTTSv2/` and `dataset/tts/ChatterboxTTS/`. Matching stems were reported for XTTSv2; ChatterboxTTS pairing has not been established. The completed upload is user-reported; the full corpus is not admitted by Phase 1 fixture checks.

Persist valuable artifacts under project `runs/`, small evidence under `reports/`, and selected environment exports under `locks/`. Generated fixtures and model probe files are disposable engineering outputs. P1-13 verified checksummed recovery across a VM replacement; this does not establish resuming an interrupted research optimizer. Drive is the verified persistent destination; no independent backup service is claimed.

Selected lock directory: `locks/colab-baseline-20261029T192113Z-93db044e/`. Preserve it with its sibling metadata and manifest, not just its requirements file. Its 70 exact pins include build tools and transitive baseline dependencies. It is specific to the recorded Python/platform/CUDA build. See [environment reproduction](environment_reproduction.md) for source provenance and limitations.

## Colab setup and recovery order

1. Sync the source/configuration/scripts/tests/package definition and required notebooks to Drive. Preserve source versions and historical evidence.
2. For an initial package setup, use P1-03 in its notebook/runtime. For reproducing the selected baseline, use Stage B of the P1-13 notebook directly; a separate P1-03 notebook is unnecessary.
3. Select a GPU runtime and mount Drive. In the P1-13 notebook, set `STAGE = 'reproduce'` and the exact saved `EXPORT_DIR`; run the settings cell and Stage B. A new environment export is needed only when intentionally establishing a changed environment.
4. Stage B checks a different VM identity, source hashes and recovered files before installation. It retains matching packages, installs differing exact pins and EchoProof editable with `--no-deps --no-build-isolation`, then verifies everything in new processes.
5. Review the printed verification report. Require passing dependency/runtime checks, tests without skips, CPU probes and the CUDA BiGRU step. A changed GPU requires P1-12 reprofiling. Refresh actual quota/work availability before substantial work.

The earlier export used torch 2.11.0+cu130. A matching build was retained during the successful reproduction; its original archive URL/hash was unavailable. The exported `torch_source.json` records distribution fingerprints and `https://download.pytorch.org/whl/cu130` as the inferred reinstall channel. If the build differs, the runner attempts the exact version from that channel; failure remains a blocker rather than substituting an unpinned build. A CPU/Windows setup needs a separate tested lock.

FFmpeg is checked independently from pip. The verified Colab binary reports `6.1.1-3ubuntu5`; the lock does not install this system binary. If it is unavailable or differs, resolve the runtime/decoder and repeat audio verification before accepting a new environment. Exact Python/platform/decoder mismatches stop P1-13 verification.

## Existing commands

Run from the project root in the selected environment. In notebooks, execute subprocesses with `sys.executable` and an argument list, using `cwd=PROJECT_ROOT`; avoid unquoted shell paths containing spaces or `&`. The following are CLI equivalents, not a substitute for the fresh-runtime reproduction report:

```text
python -m pip check
python -m echoproof.cli --version
python -m echoproof.cli validate-config --config configs/phase1.yaml
python -m echoproof.cli doctor --config configs/phase1.yaml
python -m echoproof.cli fixtures create --config configs/phase1.yaml
python -m echoproof.cli storage-probe --config configs/phase1.yaml
python -m echoproof.cli probe audio --config configs/phase1.yaml
python -m echoproof.cli probe lr --config configs/phase1.yaml
python -m echoproof.cli probe bigru --config configs/phase1.yaml
python -m echoproof.cli benchmark --config configs/phase1.yaml --models lr bigru --device auto
python -m echoproof.cli budget --config configs/phase1.yaml --benchmark <new-benchmark-run>/benchmark.json --plan configs/baseline_budget.json
python -m echoproof.cli verify-environment --lock locks/colab-baseline-20261029T192113Z-93db044e/requirements-lock.txt
```

Placeholders in angle brackets must be replaced. Benchmark and budget commands create unique outputs; canonical summaries are created only if absent. Prefer the report's referenced per-run files after reruns. The checked-in fixture config uses CPU for baseline probes; the benchmark's `--device auto` selects and records available hardware.

P1-13 runs the following meaningful unittest modules separately: `test_p1_04_config.py`, `test_p1_05_contracts.py`, `test_p1_06_reproducibility.py`, `test_p1_07_audio.py`, `test_p1_08_models.py`, `test_p1_12_budget.py`, and `test_p1_13_environment.py`. To repeat one:

```text
python -m unittest discover -s tests -p test_p1_05_contracts.py -v
```

The accepted Colab run completed 41 tests without skips. No additional Colab notebook is required for this documentation handoff. Existing fixture LR/BiGRU models and normalization statistics must not become research baselines without real training-data fitting.

## Pretrained assets: deferred workflow

P1-09–11 are deferred. `configs/pretrained_assets.json` is a reserved config path and is not present. No asset fetch/resolution/offline CLI is implemented; do not use the planning examples as available commands. When that stage starts, first implement the asset lock, explicit checkpoint revisions and licenses, compatible processor/extractor settings, SHA-256 verification and offline resolution. Then run actual frozen WavLM forward/cache checks and matched AST forward/backward checks. Extend profiling and reproduction before claiming the full gate. Unpinned downloads, random encoders and a frozen AST head-only fallback cannot satisfy those outstanding requirements.

## Evidence and review

The [central index](../reports/phase1/evidence_manifest.json) records paths, sizes, SHA-256 hashes, run/report statuses, original manifest verification and selected evidence. Its sibling `evidence_manifest.sha256` checks the index bytes. It is a dated snapshot rather than a live index; new evidence requires regeneration. Keep earlier failed reports and original statuses. Hash equality establishes consistency of supplied files, not scientific validity or a signed attestation.

See [P1-14 completion review](../reports/phase1/p1_14/review.md) and the [independent reviewer ledger](phase1_gate_checklist.md). N coordinates the handoff; S reviews data/interfaces, M reviews baseline/resource assumptions, and U reviews runtime/reproduction. Record actual review dates and findings when supplied.

## Resource envelope and common failures

The candidate pilot is 1,000 pairs/2,000 recordings, one four-second crop each. Candidate counts are 1,400 training/300 validation/300 final-scoring recordings, subject to audited groups and validation-role feasibility. The provisional budget covers eight LR candidates and two 20-epoch BiGRU fits, reusing selected models for scoring. It estimates 18.07 compute minutes, 2.238 GB additional storage and a 7.07-minute longest job. These are fixture estimates, excluding full audit/transfer/setup/troubleshooting costs.

The user allocated two expected three-hour sessions and reported 10.13 GB free Drive space. With 25% reserve, the original committed limits are 4.5 hours and 7.5975 GB. These values are not a current remaining balance or a Colab entitlement. Actual remaining work time, quota and representative-data throughput must be refreshed before training. Store only the necessary pilot caches; copying the complete original corpus is not budgeted.

| Symptom | Action |
|---|---|
| `No module named echoproof` | Check project root and editable installation in this runtime; use a fresh Python subprocess |
| Missing `jedi` in Stage A | Updated P1-13 repairs it and records the command; require successful `pip check` |
| Same-VM or changed-source rejection | Replace the VM, or intentionally establish a new export after reviewing source changes |
| Checksum mismatch | Inspect the named file/copy, preserve evidence and restore intended bytes; do not rewrite the recorded expected hash to hide a mismatch |
| GPU/decoder/package mismatch | Review actual hardware/build, repeat the applicable verification and reprofile/relock as needed |
| Failed/skipped tests or probe | Keep the report, resolve the cause and create a new verification run |
| Incomplete paired dataset | Complete the audit; fixture checks cannot admit or label speech |

## Phase 2 receives

S leads an inventory of every WAV/transcript/TTS candidate across both reported generator folders, source/provenance/permissions and verified class/generator/identity metadata. Matching stems are candidate relationships; verify completeness and content before treating them as pairs. Measure durations/rates/channels, identify unreadable files, and hash originals without overwriting them. Record missing or ambiguous metadata explicitly.

Pass `Recording` metadata and relationship evidence to cleaning/duplicate review. Form connected groups covering genuine/synthetic relatives, utterance/content duplicates and verified speaker relationships. Establish approximately 70/15/15 group-aware partitions and feasible validation roles before segmentation or augmentation. Record achieved proportions and seal held-out data. Fit imputation, scaling and learned normalization using training data only.

Only then finalize the shared eligible pilot, processed windows/masks, feature caches, recording aggregation and real-data budget, and implement/train research LR/BiGRU. Additional verified generator families and an eligible external source remain necessary for the wider research protocols. Baseline model-selection and hardware progression use grouped development evidence; final tests remain sealed.
