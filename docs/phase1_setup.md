# EchoProof package setup

The package scaffold supports P1-03: an importable `echoproof` module and a CLI entry point. Data, feature, model, training and application commands are added in later tasks. Unknown CLI commands fail rather than producing a placeholder success.

For the observed Colab runtime, use [the P1-03 notebook](../notebooks/P1_03_Colab_Package_Setup.ipynb) and [its run instructions](../notebooks/README.md). The notebook mounts the Drive project, checks the existing PyTorch build, repairs the observed missing `jedi` dependency if needed, installs the local package with `--no-deps`, checks imports and records a new environment report. It does not install another PyTorch build.

`pyproject.toml` declares the baseline libraries needed by future code. Optional groups separate neural, analysis, pretrained and application packages so later work can add them deliberately. The editable package install in Colab uses the preinstalled baseline libraries. P1-13 will freeze exact versions after probes and clean reproduction; this scaffold does not claim a locked environment.

The observed P1-02 Colab run used Python 3.13.16, PyTorch 2.11.0+cu130 and FFmpeg 6.1.1. Those observations belong to one runtime and must be checked again when P1-03 runs. The Windows environment is separate and may need a compatible interpreter and platform-specific PyTorch installation.

P1-03 acceptance requires a successful editable installation, `import echoproof`, module and console CLI help/version, baseline imports, `pip check`, and recorded FFmpeg version. The notebook saves results under `reports/phase1/p1_03/<run-id>/`. This does not validate dataset contents or start model training.
