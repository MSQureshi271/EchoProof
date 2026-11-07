# R2-U1: Protocol B frozen WavLM embedding cache

## Objective and authority

Implement the R2-U1 row of `REVISED_PLAN.md`, Section 6. Owner: Usman; reviewer: Muneeb. This task produces representations for R3-U1, not a trained detector. R2-U2 AST warm-up is a separate task.

Required scope:

- Training: the **existing** `data/features/r2_s1_B/crop_bank_B_v1.parquet`, not a newly sampled bank. The R2-S1 manifest records K=5, with one crop for recordings no longer than four seconds.
- Validation: every canonical chunk for every B validation role (selection, calibration, operating).
- Test: **genuine recordings only**, through an explicit extraction-only option. No test scores, metrics, fitting, or parity sampling.
- Never extract ChatterboxTTS or any other test TTS audio. Do not prepare Protocol A.
- Store mean (768 values) and mean plus population standard deviation (1536 values) embeddings.
- Use content keys containing canonical audio identity, sample coordinates, pinned encoder identity, and pooling. Writes must be atomic and interrupted runs resumable.
- Check cached against direct representations and diagnostic scores on 50 development intervals, then publish an inventory with measured extraction time and storage size.

The R2-S1 report records 4,530 training recordings, 21,470 training crops, 4,000 validation recordings, and 19,876 validation chunks. These are reference counts, not a substitute for deriving and checking the current plan. Genuine-test chunk counts must be derived from the frozen manifest.

## Phases and evidence gates

1. **Preflight / plan:** verify frozen file hashes, split identity, leakage evidence, the published crop bank, canonical windows, roles, and allowed components. Read metadata only.
2. **Software verification:** run the new fixture suite and existing R1-U1 suite. Generated fixtures establish software behavior, not detection accuracy. Missing dependencies or skipped tests do not establish a passing gate.
3. **Colab setup:** use a GPU runtime; verify the actual GPU and software versions. Retrieve the pinned checkpoint and processor. Stage required canonical shards on Colab scratch; preserve persistent input files.
4. **Extraction / resume:** validate decoded canonical PCM hashes and lengths, extract bounded batches at true lengths, and commit complete shard packs atomically. Reuse only fully validated packs. Corrupt or incompatible packs stop the run.
5. **Independent verification:** validate coverage, indexes, hashes, dimensions, finite values, both pooling views, and development-only parity. Publish the final manifest only after checks pass.
6. **Review:** give Muneeb the actual Colab evidence and output manifest. A software pass, a real GPU pass, and an independent teammate review are separate statuses.

## Required local or Colab project inputs

```text
data/manifests/cohort_v1.parquet
data/splits/protocol_B_v1.parquet
data/processed/chunks_v1.parquet
data/processed/cohort_v1/shard_*.tar
data/features/r2_s1_B/crop_bank_B_v1.parquet
data/features/r2_s1_B/r2_s1_manifest.json
reports/r1/split_report.json
reports/r1/leakage.json
reports/r1/r1_u1_colab_20261104T135216Z.json
reports/r1/ast_probe.json
reports/r2/r2_s1_colab_20261105T171342Z.json
src/echoproof/
scripts/r2_u1_wavlm_cache.py
tests/test_r2_wavlm_cache.py
tests/test_r1_u1_wavlm.py
```

The raw `dataset/` folder is unnecessary: this task consumes the prepared canonical shards. The repaired XTTS recording is in `shard_14.tar`; see `reports/r1/shard_rename_20261104T080826Z.json`.

Preflight requires a verified `reports/r2/r2_s1_colab_*.json` whose embedded build manifest exactly matches the published R2-S1 manifest. The filename above is the matching report present in this copy. The R1 reports supply the recorded checkpoint/runtime evidence; the canonical paths are read from the cohort manifest rather than the separate shard index.

## Numerical contract

Reuse `features/pretrained.py` and its pinned `microsoft/wavlm-base-plus` commit and weight checksum. Use the final hidden states, frozen encoder, evaluation/inference mode, FP32 inference, mono 16 kHz inputs and the pinned processor (`do_normalize=false`). Convert canonical int16 PCM to float32 using division by 32768; do not resample or normalize it again.

Each crop/window is embedded independently. Pool valid hidden frames, not crops or recordings. Mean-plus-std concatenates the mean with the **population** standard deviation (`ddof=0`); no crop averaging or learned normalization is introduced. Equal-length bucketing avoids zero-padding, which changes WavLM's temporal group-normalization statistics. Preserve input ordering after bucketing.

Record the processor settings, encoder weights/revision, source fingerprints, numerical policy, and library versions in the extraction identity. A different identity must not silently reuse an older cache generation. The R1 checks used Transformers 5.22.0; the runner checks that version rather than silently upgrading the environment.

## Test boundary and parity interpretation

The research contract normally seals final-test access. R2-U1 explicitly requests genuine-test extraction. The runner therefore requires `--include-test-genuine` for the complete task, records that narrow exception, and separates test references from development references. It does not train, select a model, or calculate test scores. Default readers exclude test references.

No trained WavLM head exists at this stage. A deterministic, explicitly **untrained diagnostic head** is used only to compare the cached and direct numerical paths. Its definition/seed and parameter fingerprint are recorded. Also compare the embeddings themselves for both pooling modes, with absolute and relative tolerances of 1e-4. Any parity failure blocks publication. Parity samples come from development only.

This is engineering evidence, **not** detector accuracy, calibration, or equivalence of the eventual R3 adapter. R3-U1 must repeat direct-versus-cached scoring with its trained head. No threshold or hyperparameter is selected here.

## Execution

Use `notebooks/R2_U1_Colab_WavLM_Cache.ipynb`. Set its project root explicitly; no teammate-specific Drive location is assumed. Sync the new source, runner, tests and notebook before opening it. The notebook must be executed by the user in Colab; preparing it locally does not establish GPU success.

CLI entry point: `scripts/r2_u1_wavlm_cache.py` (`plan`, `build`, `verify`). Read its `--help` for the exact options. The full task uses `--include-test-genuine`; a development-only run must not be reported as the complete R2-U1 deliverable.

The notebook requests resume automatically. After a hard runtime termination, a `.writer.lock` can remain in the output folder. Confirm the previous writer has stopped before manually removing **that lock only** in your own output copy; do not remove it while another run is active. Complete packs are validated on resume. Abandoned temporary files are not accepted as completed packs. A corrupt or incompatible generation requires inspection and a different output directory or an explicit repair; it is never silently overwritten.

Do not install or upgrade a large dependency stack blindly. The notebook checks required imports and the recorded model-library version before extraction. Resolve a missing or incompatible runtime explicitly, then rerun the checks.

## Deliverables and R3 handoff

- `src/echoproof/features/r2_u1.py`: validated planner, content identity, atomic packs, verification and reader.
- `scripts/r2_u1_wavlm_cache.py`: executable orchestration and evidence export.
- `tests/test_r2_wavlm_cache.py`: generated fixtures and adversarial failure/resume tests.
- `notebooks/R2_U1_Colab_WavLM_Cache.ipynb`: staged Colab execution and checks, with no fabricated saved outputs.
- `data/features/r2_u1_B/`: generated packs, separate development/genuine-test indexes, and final committed manifest after successful execution.
- `reports/r2/`: actual run evidence, with input/source fingerprints, device/versions, counts, extraction/reuse timings, size and parity results, or an explicit failure.

R3 must read only the training rows for fitting and selection validation for model choice. Calibration and operating representations are available for their later designated operations; availability does not authorize using them to choose a head. Genuine-test representations remain excluded by default.

Storage reporting separates pack bytes and index bytes. Their sum excludes the small final manifest, temporary files, staged input shards, and downloaded model files; the evidence states that accounting boundary. Do not interpret it as total Drive or Colab usage.

The historical AST GPU budget assumed four WavLM crops; R2-U1 uses the published five-crop bank. Actual R2 extraction timing supersedes that planning assumption. Existing historical evidence is preserved.

## Current verification status

At implementation time, the project-local data hashes and R1 evidence were inspected. The local `python` command resolves to an unavailable Windows Store launcher. No local Python test pass, pretrained extraction, real GPU parity, or teammate approval is claimed. The Colab run is the required execution gate.

The new suite declares **34 acceptance tests**. They cover metadata-only preflight and its evidence chain, forbidden scopes, the exact crop bank and canonical intervals, PCM identity, pooling dimensions, both diagnostic parity paths, atomic failure, partial/completed resume, corrupt packs/indexes, and stale writer locks. These are implemented tests, not recorded passes. The notebook runs them alongside the existing R1-U1 tests and refuses to extract research audio if tests fail or are skipped.

