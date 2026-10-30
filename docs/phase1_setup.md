# EchoProof package setup

**Current baseline handoff:** P1-13 fresh-runtime reproduction is evidenced, and P1-14 setup/recovery instructions are in the [handoff guide](phase1_handoff.md). See the [readiness checklist](phase1_gate_checklist.md) for scope and pending reviews. The paragraphs below retain the earlier package-bootstrap context.

The package scaffold supports P1-03: an importable `echoproof` module and a CLI entry point. Data, feature, model, training and application commands are added in later tasks. Unknown CLI commands fail rather than producing a placeholder success.

For the observed Colab runtime, use [the P1-03 notebook](../notebooks/P1_03_Colab_Package_Setup.ipynb) and [its run instructions](../notebooks/README.md). The notebook mounts the Drive project, checks the existing PyTorch build, repairs the observed missing `jedi` dependency if needed, installs the local package with `--no-deps`, checks imports and records a new environment report. It does not install another PyTorch build.

`pyproject.toml` declares the baseline libraries. Optional groups separate neural, analysis, pretrained and application packages. Initial editable setup uses the available Colab baseline libraries. The subsequently verified baseline lock is `locks/colab-baseline-20261029T192113Z-93db044e/requirements-lock.txt`; use the P1-13 runner for exact dependency comparison/recreation. The package definition alone is not an environment lock.

The observed P1-02 Colab run used Python 3.13.16, PyTorch 2.11.0+cu130 and FFmpeg 6.1.1. Those observations belong to one runtime and must be checked again when P1-03 runs. The Windows environment is separate and may need a compatible interpreter and platform-specific PyTorch installation.

P1-03 acceptance requires a successful editable installation, `import echoproof` in a fresh Python process, module and console CLI help/version, baseline imports, `pip check`, and recorded FFmpeg version. A fresh process is needed because the editable install's `.pth` entry may not become visible to an already-running notebook kernel. The notebook saves results under `reports/phase1/p1_03/<run-id>/`. This does not validate dataset contents or start model training.

P1-04 adds `python -m echoproof.cli validate-config --config configs/phase1.yaml`, which prints sorted resolved JSON without creating artifacts. Run from the project root, or pass `--project-root` explicitly. Relative paths resolve against that root; `--run-root` may select a path under `runs/` and `--device` can select `cpu`, `cuda`, or `auto`. Validation does not check GPU availability or require the input directories to exist. The fixture path must remain under `tests/fixtures/generated`; P1-07 will implement fixture registration and actual data loading. See the [Colab verification instructions](../notebooks/README.md#p1-04-configuration-verification).
