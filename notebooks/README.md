# P1-02 on Google Colab

## R2-U1 WavLM cache

Use [R2_U1_Colab_WavLM_Cache.ipynb](R2_U1_Colab_WavLM_Cache.ipynb) and the [R2-U1 execution guide](../docs/r2_u1_wavlm_cache.md). This task reuses the published five-crop Protocol B training bank, embeds all B validation roles, and supports explicitly acknowledged genuine-test extraction only. No Chatterbox audio is extracted. The notebook separates preflight, fixture tests, GPU setup, shard staging, resumable extraction, parity checks and evidence export. Set the project path explicitly and sync the new source, runner and tests first. Preparing the notebook does not establish a successful Colab run.

For P2-07 through P2-09, use the single [P2_07_09_Colab_Audit.ipynb](P2_07_09_Colab_Audit.ipynb) and [execution guide](../docs/p2_07_09_execution.md).

For P2-06, use [P2_06_Colab_Pairing.ipynb](P2_06_Colab_Pairing.ipynb) and the [pairing guide](../docs/p2_06_pairing.md). The notebook also inspects the P2-05 Parquet output directly.

For P2-05, follow the sync instructions in the [adapter guide](../docs/p2_05_adapter.md), then run [P2_05_Colab_Adapter.ipynb](P2_05_Colab_Adapter.ipynb) on CPU.

For P2-04 use [P2_04_Colab_Inventory.ipynb](P2_04_Colab_Inventory.ipynb) and follow the [Drive inventory guide](../docs/p2_04_inventory.md), including the recovery instructions.

For P2-03, sync the files listed in the [audit contracts guide](../docs/p2_03_audit_contracts.md) and run [P2_03_Colab_Verification.ipynb](P2_03_Colab_Verification.ipynb). Share its `verification.json` for review.

For P2-02 publisher evidence, sync `scripts/p2_02_source_evidence.py` and `configs/data_sources.json`, then run [P2_02_Source_Evidence_Colab.ipynb](P2_02_Source_Evidence_Colab.ipynb) on CPU. It captures small publisher documents/metadata only. Share the printed source inventory, manifest and documents; review terms and local acquisition linkage separately. See [source evidence](../docs/source_evidence.md).

For Phase 2 entry reconciliation, use [P2_01_Colab_Entry_Reconciliation.ipynb](P2_01_Colab_Entry_Reconciliation.ipynb) and the [execution guide](../docs/phase2_execution.md). Sync its collector and current plan/docs first. It runs on CPU with only the standard library, verifies reported Drive roots and saved lock artifacts, and records unavailable external data and pending provenance. It does not require a separate P1-03 notebook. Share the printed `entry.json` and manifest for review.

Use [P1_02_Colab_Runtime_Setup.ipynb](P1_02_Colab_Runtime_Setup.ipynb) to inspect the runtime and project already copied to Drive. It is self-contained; no package installation, downloaded weights or sibling script is required. It records environment information, filename/size inventory, a small numpy/torch check and persistent-storage evidence. It does not train audio models or validate dataset contents.

## Run instructions

1. Copy the **new notebook** from this workspace to the `notebooks` folder of your project on Drive. Also sync the updated `plans/PHASE1.md` and this guide. Your earlier directory copy will not automatically include these new files unless a sync tool does so.
2. Open the notebook with Google Colab, or use Colab's **File → Upload notebook** option. Keep the Drive project folder as the source of data and saved reports.
3. Choose a Python runtime. For the BiGRU hardware inventory request a GPU through the runtime settings if offered; a CPU runtime is valid for LR inventory. Record the hardware actually assigned rather than assuming a particular GPU. GPU allocation and runtime lifetime vary. [Colab FAQ](https://research.google.com/colaboratory/faq.html).
4. Run the Drive-mount cell and grant access to your Drive. Edit `PROJECT_ROOT` in the next cell. The default is `/content/drive/MyDrive/PAI & AIES Project`; include any additional parent folders in your actual Drive location. Confirm the root contains `IMPLEMENTATION_PLAN.md` and `dataset/`.
5. Set `REQUEST_GPU=False` for an intentional CPU run, or keep it `True` when checking an assigned GPU. Leave unknown Drive quota and work/storage allowances as `None`. Do not use Linux filesystem free space as Drive account quota. The work-hours value is a chosen work allowance, not a promised session length.
6. Run cells through **Inspect the summary and verify saved checksums**, in order. The definition cell can be long; it does not start collection itself. Directory enumeration may take time for many Drive files. It opens no WAV or TXT contents.
7. Read `status`, `issues`, missing packages, device and output paths. A warning is captured evidence to resolve, not a passing check. This notebook intentionally does not upgrade packages or replace Colab's working torch build.
8. Copy the printed run-relative path and save the notebook. For stronger persistence evidence, perform the optional restart sequence exactly as described in the notebook: remount Drive, rerun configuration and definitions, then verify the old path. Do not regenerate the run before this check.

## Expected outputs

Each collection creates a new folder:

```text
<Drive project>/reports/phase1/p1_02/<UTC timestamp>-<unique suffix>/
  hardware.json
  environment.md
  persistence_probe.txt
  artifact_manifest.json
  restart_verification-<suffix>.json   # only after an actual fresh-runtime check
```

On the first run it also creates `docs/environment.md` and `reports/phase1/hardware.json` if those files do not exist. Existing canonical files are preserved. **For a rerun, always inspect/share the latest versioned run files.** No raw data is edited, moved or copied.

The report proposes persistent Drive locations for manifests/features, runs/checkpoints, bundles and later model assets; those future artifacts are not created here. `/content/echoproof` is disposable scratch. Persistent originals remain under:

```text
dataset/genuine/wav/<stem>.wav
dataset/genuine/transcripts/<stem>.txt
dataset/tts/XTTSv2/<stem>.wav
```

Stems link pair candidates. Enumeration does not prove correct transcripts, audio authenticity, speaker independence, language or license. The known-family baseline can proceed only after the later audit/split gates. The original unseen-generator protocol still needs more verified families.

## How to interpret the result

| Observation | Action |
|---|---|
| `collected_pending_review` | Inventory checks found no immediate issue; U/M review the saved evidence before accepting P1-02 |
| `needs_attention` | Read `issues` and detailed fields in `hardware.json`; resolve the affected path/runtime/dependency problem |
| GPU requested but CUDA false | Verify GPU runtime selection and reconnect if needed; choose an explicit CPU pilot only when intended |
| Missing packages | Record them for P1-03; do not install the entire future stack for this inventory |
| `pip check` errors | Inspect package conflicts before planning dependency changes; do not force upgrades in bulk |
| Zero/missing/duplicate stems | Confirm copied folder names/case and transfer completion; preserve evidence; audit rather than deleting originals |
| No quota or session estimate | Keep those fields pending; Colab limits cannot be inferred as fixed guarantees |
| Existing canonical files | Use the new run's `hardware.json` and `environment.md` to avoid stale evidence |
| Restart verification fails | Confirm remount, the original run path and Drive copy completion; retain the failed check details |

P1-02 does not close OPEN-003/004 in full: throughput budgeting, final demo device selection, and any separate GPU machine remain pending. It supplies observed evidence for those later decisions. Full G1 and baseline model training remain separate milestones.

## What to provide after running

Provide the latest run's **`hardware.json` and `environment.md`**, and the restart verification file if completed. These contain the evidence needed to choose P1-03 dependencies and update the execution record. You may instead paste the summary, issues, Python/torch versions, CUDA availability and filename counts. No recordings or transcript contents are needed.

The notebook is prepared locally. Actual Colab success, allocated GPU and storage persistence can only be recorded after your run.

## P1-03 package setup notebook

After P1-02, sync [P1_03_Colab_Package_Setup.ipynb](P1_03_Colab_Package_Setup.ipynb), `pyproject.toml`, the whole `src/echoproof/` folder, `docs/phase1_setup.md`, and `scripts/p1_03_colab_setup.py` to the Drive project. The notebook calls the setup script from that project. It requires no audio files to run and does not change the dataset.

Open the notebook in Colab, mount Drive, check that `PROJECT_ROOT` points at `/content/drive/MyDrive/PAI&AIES Project` or your actual copied folder, and run the cells in order. The default `ALLOW_PACKAGE_INSTALL=True` installs missing `setuptools`/`wheel` and the observed missing `jedi` dependency if needed, then installs only the local EchoProof package with `--no-deps --no-build-isolation`. It checks that the PyTorch version did not change, loads baseline imports, exercises the module and console CLI, runs `pip check`, and checks FFmpeg MP3 capability. If Colab asks for a restart, reconnect and rerun to capture the final state.

Each run saves `package_setup.json`, `package_setup.md`, and `artifact_manifest.json` under `reports/phase1/p1_03/<run-id>/`. Share the latest two report files for M/U review. `ready_for_review` is a software check result, not a claim that an independent team review occurred. Exact version locking and clean-environment reproduction belong to P1-13.

## P1-04 configuration verification

Sync `src/echoproof/`, `configs/`, `tests/test_p1_04_config.py`, both `scripts/p1_03_colab_setup.py` and `scripts/p1_04_colab_verify.py`, and [P1_04_Colab_Verification.ipynb](P1_04_Colab_Verification.ipynb) into the Drive project. Open the P1-04 notebook, confirm `PROJECT_ROOT`, and run every cell. Its verifier first performs the guarded P1-03 dependency repair and editable package installation in the same Colab runtime, then runs P1-04 checks. You do not need to open or run the P1-03 notebook separately. Share `reports/phase1/p1_04/<run-id>/verification.json`; a passing run is `ready_for_review` and still needs independent A/S review. The checks use temporary configurations and do not read the research dataset.

## P1-05 contract verification

Sync `src/echoproof/contracts.py`, `tests/test_p1_05_contracts.py`, `tests/fixtures/contracts/`, `scripts/p1_03_colab_setup.py`, `scripts/p1_05_colab_verify.py`, and [P1_05_Colab_Verification.ipynb](P1_05_Colab_Verification.ipynb) to the project on Drive. Open the P1-05 notebook, check `PROJECT_ROOT`, and run its cells. It performs P1-03 setup in the same runtime before contract checks; opening the P1-03 notebook is unnecessary. The result is `needs_attention` if a required check is skipped. Share the resulting `reports/phase1/p1_05/<run-id>/verification.json` for review. The test metadata is fabricated and does not open the research dataset.

## P1-06 reproducibility and storage verification

Sync the whole `src/echoproof/` folder, `configs/`, `tests/test_p1_06_reproducibility.py`, `scripts/p1_03_colab_setup.py`, `scripts/p1_06_colab_verify.py`, and [P1_06_Colab_Verification.ipynb](P1_06_Colab_Verification.ipynb) to Drive. Open the P1-06 notebook, confirm `PROJECT_ROOT`, and run all cells. It installs the local package and repairs missing P1-03 prerequisites before checking P1-06 in the same runtime. It creates one evidence report under `reports/phase1/p1_06/` and a separate storage probe under `runs/phase1/`. Share the printed `verification.json`; the report includes the storage run path and check output. A skipped test produces `needs_attention`. See [reproducibility notes](../docs/reproducibility.md) for what the checks establish.

## P1-07 generated audio verification

Sync `src/echoproof/`, `configs/`, `tests/test_p1_07_audio.py`, `scripts/p1_03_colab_setup.py`, `scripts/p1_07_colab_verify.py`, and [P1_07_Colab_Verification.ipynb](P1_07_Colab_Verification.ipynb) to Drive. Run all notebook cells with the correct `PROJECT_ROOT`. The notebook performs P1-03 setup in the same runtime and runs the fixture and audio probes. Share the printed `reports/phase1/p1_07/<run-id>/verification.json`; the audio probe's run directory is in that report. See [audio probe notes](../docs/audio_probe.md) for fixture provenance and limits.

## P1-08 baseline fixture verification

Sync `src/echoproof/`, `configs/`, `tests/test_p1_08_models.py`, `scripts/p1_03_colab_setup.py`, `scripts/p1_08_colab_verify.py`, and [P1_08_Colab_Verification.ipynb](P1_08_Colab_Verification.ipynb) to the Drive project. Confirm `PROJECT_ROOT` and run every cell. Setup, tests, LR and BiGRU probes run in one CPU runtime; the first feature extraction can take time. Share the printed `reports/phase1/p1_08/<run-id>/verification.json`, and the `metrics.json` from each model run referenced in it. Probe models and normalization are disposable, as described in [baseline fixture notes](../docs/baseline_fixture_probes.md).

## P1-12 baseline profiling and capacity budget

Sync `pyproject.toml`, `src/echoproof/`, `configs/`, `tests/test_p1_12_budget.py`, `scripts/p1_03_colab_setup.py`, `scripts/p1_12_colab_verify.py`, and [P1_12_Colab_Verification.ipynb](P1_12_Colab_Verification.ipynb). Use a Colab GPU runtime if profiling intended BiGRU GPU training; `DEVICE='auto'` records and uses the actual allocation, with CPU fallback. Check the candidate plan and capacity assumptions in the notebook, then run its cells. It performs setup, budget tests, both baseline benchmarks and the capacity calculation in the same runtime. Share the printed `verification.json` and the referenced benchmark/budget files. The [next-steps guide](../docs/baseline_next_steps.md) explains statuses, assumptions and how to revise a budget without repeating a matching benchmark.


## P1-13 baseline environment reproduction

For baseline environment reproduction, use [P1_13_Colab_Environment_Reproduction.ipynb](P1_13_Colab_Environment_Reproduction.ipynb) and the [guide](../docs/environment_reproduction.md). Sync the updated source/scripts/configurations/tests before export. Stage A saves the working GPU environment and historical artifact hashes. Save its exact export directory on Drive, replace the Colab VM, and use Stage B in the same notebook to install exact baseline pins, verify Drive recovery and rerun checks. Same-VM reuse or skipped checks cannot establish reproduction. No separate P1-03 notebook is required. Share the final reproduction report and exported lock directory for review.
