# EchoProof

EchoProof is a reproducible research and demonstration project for distinguishing genuine English speech recordings from text-to-speech (TTS) output. The demo loads the locked, calibrated Protocol B WavLM-head model. Its score is evidence from that model, not proof of how audio was made or who made it.

## Scope and limits

The validated scope is English speech, genuine human recordings versus TTS output. It was not validated for other languages, music, noisy or heavily compressed audio, voice conversion, or other manipulation. A calibrated score is not a guarantee of correctness. The model's locked operating policy has a documented false-positive cost; see `locks/B_lock_gr3.json` and the app's scope text. Local app processing keeps audio in memory; a hosted app sends audio to its host for analysis.

## Requirements

- Python 3.11.x or 3.13.x (the lock was installed and exercised with Python 3.11.9; the P1-13 Colab baseline recorded 3.13.16).
- Git and PowerShell on Windows, or Bash on Linux/Colab.
- Several GB of free disk space for the production bundle and environment; the bundle is roughly 380 MB and PyTorch adds substantial platform-specific dependencies.
- A local copy of the project bundle from the shared Drive folder `PAI&AIES Project/bundles/r3_n1_wavlm_head_B_calibrated`.

Application dependencies, including app, neural, pretrained and analysis packages, are pinned in [`requirements/app.lock.txt`](requirements/app.lock.txt). `transformers==5.22.0` is required for the bundle's verified environment. Model weights are included in the bundle; inference does not fetch model files from Hugging Face.

## Fresh install

Run commands from the repository root.

### Windows PowerShell

```powershell
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, invoke tools directly with `.venv\Scripts\python.exe` and `.venv\Scripts\streamlit.exe`.

### Linux or Colab

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

The setup scripts create a virtual environment, install the pinned requirements, install EchoProof editable without resolving additional dependencies, then check required imports and version pins. They stop and report a useful error if a step fails.

## Retrieve and verify model bundles

Copy the production bundle folder from the project Drive to a local transfer folder, then run:

```bash
python scripts/fetch_bundles.py --source "/path/to/transfer/bundles"
```

`--source` accepts a folder or ZIP. The script selects bundles by their locked `bundle.json` SHA-256, validates every component's size and SHA-256, and places the production Protocol B bundle at `bundles/r3_n1_wavlm_head_B_calibrated`. It does not download from the internet. To retrieve all candidates recorded in both B and A locks, use `--all`; this includes R4-N1 A candidates and is not needed to run the production app.

The active production bundle is **only** Protocol B WavLM from `locks/B_lock_gr3.json`. `locks/A_lock.json` is used only when `--all` is requested.

## Run prediction and app

CLI prediction prints a JSON result:

```bash
echoproof predict --bundle bundles/r3_n1_wavlm_head_B_calibrated --audio path/to/recording.wav
```

Or, before activating the environment, use `python -m echoproof predict` with the same arguments. Supported audio formats and input limits are shown by the app and CLI.

Start Streamlit from the repository root:

```bash
streamlit run app/main.py
```

The app verifies the bundle against the Protocol B lock on startup and refuses analysis if verification fails. `ECHOPROOF_BUNDLE` is a development override; an override is not verified against the production lock.

## Verification and tests

Run the installation check after setup and bundle retrieval:

```bash
python scripts/verify_install.py
```

It verifies and loads the production bundle, invokes `echoproof predict` against `tests/fixtures/r4_n3/fixture.audio` (a synthetic software signal), compares the output with `tests/fixtures/r4_n3/expected_prediction.json`, and exercises the Streamlit app through `AppTest`. The expectation file must contain a score, decision, score type, threshold, and `"status": "verified"` recorded from an independently executed prediction with this exact locked bundle. Do not guess these values. The checked-in record is intentionally pending while the bundle is unavailable; until populated, `verify_install.py` exits nonzero.

Run the existing software suite with:

```bash
python -m pytest -q
```

Production-bundle parity tests skip when the bundle or exact Transformers version is absent. A skipped test is not production verification.

## Repository map

| Path | Contents |
|---|---|
| `src/echoproof/` | Inference, CLI, audio, bundle loading and project workflows |
| `app/` | Streamlit demo |
| `scripts/` | Reproduction, setup, retrieval and verification tools |
| `requirements/` | Pinned application environment |
| `locks/` | B and A bundle manifests and checksums |
| `bundles/` | Locally retrieved model artifacts; not committed to Git |
| `tests/` | Software tests and fixtures |
| `notebooks/` | Colab and research workflows |
| `docs/` | Plans, decisions and evidence guides |
| `reports/` | Generated run reports and reproduction evidence |

## Locks and evidence

`locks/B_lock_gr3.json` records the production B bundle identity, manifest and component hashes, threshold, score type and policy measurements. `locks/A_lock.json` records Protocol A candidates separately; it does not select the production demo model. Cite the lock file and its bundle ID when referring to a particular model artifact. For implementation and acceptance context, see [`docs/r4_n2_n3_plan.md`](docs/r4_n2_n3_plan.md), [`docs/r4_n2_app_acceptance.md`](docs/r4_n2_app_acceptance.md), the relevant entries in [`docs/decision_log.md`](docs/decision_log.md), and the reports referenced from the locks.

For a Colab reproduction, open [`notebooks/R4_N3_Colab_Fresh_Install.ipynb`](notebooks/R4_N3_Colab_Fresh_Install.ipynb), mount Drive, set the project root, and run all cells in order. The notebook saves command output and environment details under `reports/r4_n3/`. A notebook that has not been run in a fresh Colab runtime is a workflow, not evidence of successful reproduction.
