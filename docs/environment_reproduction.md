# P1-13 baseline environment reproduction

P1-13 saves the successful Colab environment and verifies the baseline software after replacing the VM. This implementation follows `.agents/data-ai/mlops-engineer.md` and `ml-engineer.md`. Actual Colab export/reproduction and independent M/U review remain pending.

## Sync before exporting

Copy `pyproject.toml`, `src/echoproof/`, `configs/`, all `tests/test_p1_*.py`, `scripts/p1_13_colab_verify.py`, and `notebooks/P1_13_Colab_Environment_Reproduction.ipynb` to the existing Drive project. Preserve `runs/phase1/`. Both stages use the same notebook in separate VMs. A separate P1-03 notebook is unnecessary. Do not edit source/config/test/script files between stages: their hashes must match.

## Stage A: working environment export

Open the notebook in the successful P1-12 GPU runtime, mount Drive, set `PROJECT_ROOT` to `/content/drive/MyDrive/PAI&AIES Project` and keep `STAGE = 'export'`. Run Stage A. If that runtime was reclaimed, restore and rerun P1-12 first; the subsequent export describes that restored environment rather than proving the earlier package state.

Before exporting, Stage A installs `jedi` only if its package metadata is missing, records that repair, checks that the PyTorch version is unchanged, and runs `pip check`. This repeats the narrow Colab repair from P1-03 without requiring its notebook. Failed installation or dependency checks stop export. The exact repaired dependency versions enter the new lock.

The runner creates an immutable `locks/colab-baseline-<id>/` directory and `reports/phase1/p1_13/export-<id>/verification.json`. Require `exported_for_reproduction`, save the exact `export_dir`, and confirm its files are visible in Drive before disconnecting.

| Export file | Purpose |
|---|---|
| `requirements-lock.txt` | Exact baseline dependency closure and build-tool versions |
| `installed-packages.txt` | Broader diagnostic inventory; not the installation input |
| `environment.json` | Python, platform, GPU/CUDA, driver, FFmpeg, VM identity and lock hash |
| `torch_source.json` | Build/channel, installed metadata fingerprints and inferred official reinstall index |
| `source_files.json` | Source, script, test, config and project-definition hashes |
| `recovery_files.json` | Verified historical run manifests and their listed artifact hashes |
| `snapshot_scope.json` | Included roots, excluded direct origins and limitations |
| `artifact_manifest.json` | Export checksums |

The lock follows installed dependencies, active platform markers and dependency extras. It includes baseline libraries, PyTorch, profiling packages, pip/setuptools/wheel and the earlier `jedi` repair. Editable EchoProof is excluded. Direct/local baseline origins stop export for explicit resolution, except a recorded official PyTorch wheel origin; they are not silently converted into index pins. The broader inventory excludes local/direct distributions and lists their names without URLs or credentials. `pip check` must pass before export.

The original Colab PyTorch wheel URL/archive hash may be unavailable. The tool records the installed version, channel, installer and METADATA/WHEEL/RECORD hashes, explicitly labeling the reinstall index as inferred. The observed `2.11.0+cu130` CPython 3.13 Linux wheel is listed on the [official CUDA 13.0 index](https://download.pytorch.org/whl/cu130/torch/). Pins and installed distribution fingerprints do not establish an archived wheel hash or offline package installation.

## Stage B: fresh VM reproduction

Use **Disconnect and delete runtime**, reconnect with a GPU, and remount Drive. Set `STAGE = 'reproduce'` and paste the Stage A `EXPORT_DIR`. Rerun the settings cell and Stage B. An ordinary kernel restart retains the VM boot identity and is rejected.

Before package changes, the runner verifies the export, a different VM boot identity, exact Python version, OS/architecture, source files and historical Drive artifacts. Matching package versions are retained. Differing exact pins install without dependency resolution, using PyPI for ordinary packages and the recorded official PyTorch channel for a differing torch build. It installs EchoProof editable with `--no-deps --no-build-isolation`. Downloads/cache use temporary VM storage; the broader Colab inventory is not installed wholesale. Installation failures retain their logs and cannot pass verification.

The notebook avoids importing scientific packages into its kernel before installation. Fresh subprocesses perform:

1. Exact package/Python/platform/FFmpeg/CUDA checks and PyTorch METADATA/WHEEL comparison.
2. `pip check`, installed CLI and configuration validation.
3. P1-04/05/06/07/08/12/13 unit checks, rejecting failures, skips and zero-test discovery.
4. Storage, audio, LR and BiGRU fixture probes with the unchanged CPU configuration.
5. A CUDA BiGRU forward/backward/AdamW step on generated tensors, checking finite gradients and a head update.
6. Final environment, source and historical artifact verification.

Later probe environment records carry the supplied dependency-lock file hash through `ECHOPROOF_DEPENDENCY_LOCK`. Earlier runs are preserved without retroactive lock claims. The existing CPU probes check score/padding/reload parity with their documented tolerances. Historical artifacts are recovered and hash-checked; interrupted research training continuation is later work. The CUDA check establishes operator compatibility, without asserting research performance or cross-device determinism.

## Reports and next step

Share `verification.json` and `verification.sha256` from the printed `reports/phase1/p1_13/reproduce-<id>/` plus the exported lock directory. New probes create unique directories under `runs/phase1`; command stdout in the report identifies them. `ready_for_review` requires a fresh VM, exact locked runtime/packages, preserved files, dependency checks, all selected tests without skips, and successful CPU/CUDA probes.

A changed GPU sets `budget_reprofile_required`, so P1-12 must be remeasured before relying on its T4 budget. A driver change is recorded; CUDA failures still block acceptance. Independent M/U review and P1-14 handoff follow this evidence. Audited data, grouped splitting and representative-data profiling precede training. WavLM/AST and full G1 remain deferred.

| Failure | Action |
|---|---|
| Same VM | Replace the runtime, remount and rerun Stage B |
| Source/historical hash mismatch | Inspect named files and restore the intended copy; intentional edits require a new reviewed export |
| Python/platform changed | Select a compatible runtime or establish and profile a new environment |
| Exact wheel unavailable/dependency conflict | Preserve logs and resolve an explicit replacement environment; keep version checks strict |
| FFmpeg changed | Review decoder differences and establish a new snapshot with audio checks |
| Changed GPU | Reprofile P1-12 |
| Failed/skipped test or probe | Resolve the reported cause and preserve the failed report |

The six-hour allocation and 10.13 GB free-space snapshot remain planning inputs. Refresh availability before substantial work. P1-12's compute estimate did not include all reproduction, audit or troubleshooting time.

## CLI equivalents

The notebook handles installation and the full suite. The commands below export or verify the lock/runtime/files only:

```text
python -m echoproof.cli export-environment --project-root . --output locks/colab-baseline-<new-id>
python -m echoproof.cli verify-environment --project-root . --lock locks/colab-baseline-<id>/requirements-lock.txt
```

Each export belongs to its recorded Colab Linux/Python/CUDA build. Establish CPU or Windows locks separately. Add pretrained/offline model probes when that deferred stage begins.
