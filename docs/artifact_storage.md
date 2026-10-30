# Artifact storage policy

The existing `dataset/` contains immutable source inputs; it is excluded from Git. Phase 1 validation reads configuration only. The configured persistent `runs/` lives under the project on Google Drive, while `/content/echoproof` is temporary Colab staging. Recreate scratch data after a runtime reset and save valuable artifacts to Drive. Do not store credentials or private recording contents in logs.

`data/manifests`, `data/splits`, `data/processed`, `data/features`, `runs`, and `bundles` are excluded from Git because they may contain private or large artifacts. Keep small schemas, configs, documentation, and deliberately selected report summaries under version control. Review summaries for sensitive information before adding them. Model weights, environments, cached data and generated fixtures are excluded. Later tasks implement artifact manifests, checksums and recovery; this policy does not claim those operations are complete.

The default asset lock path is reserved for P1-09; no checkpoint revision, license, or hash is asserted yet. Model configuration files describe experimental interfaces only.

## P1-14 evidence and recovery update

P1-06 implemented immutable run directories, atomic artifact writes and SHA-256 manifests; P1-13 verified fresh-VM Drive readback of 31 historical files and produced four new checksummed probe runs. See the [review](p1_13_review.md). These establish the recorded artifact recovery; no independent backup service or interrupted research-training continuation is claimed.

Preserve the selected `locks/colab-baseline-20261029T192113Z-93db044e/` directory, all failed/successful verification reports and original run manifests. [The central evidence index](../reports/phase1/evidence_manifest.json) is a dated snapshot of project-relative paths, byte sizes and hashes, with its own SHA-256 sidecar. It excludes itself and its sidecar from its file list to avoid self-reference. New artifacts and intentionally edited documentation require an updated index; do not change prior run/lock files to match a later snapshot. See [setup and recovery](phase1_handoff.md) for executable commands and storage boundaries.
