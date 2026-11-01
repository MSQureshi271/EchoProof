# P2-04 Drive inventory

The inventory walks the configured genuine, transcript, XTTSv2, ChatterboxTTS, MeloTTS and VITS roots. It streams SHA-256 in bounded blocks, records each path independently, detects normalized case collisions, and checks file metadata before and after hashing. It does not decode or modify audio, determine labels, select samples, or admit recordings.

Sync `src/echoproof/`, `configs/phase2.yaml`, `configs/data_sources.json`, `tests/test_p2_04_inventory.py`, and `notebooks/P2_04_Colab_Inventory.ipynb` to the Drive project. Open the notebook on a CPU runtime, confirm the project root and run the fixture checks before starting the inventory. PyYAML must be available in the runtime. The run may take a long time due to Drive I/O. It prints the run ID before scanning; save it for recovery.

The 31 October update adds MeloTTS and VITS. Sync the updated files before starting a new run. A checkpoint created with the earlier four-root configuration cannot be resumed under the new configuration; begin a new run so all six roots are covered.

The equivalent CLI is `python -m echoproof.cli data inventory --project-root <project> --source phonemedf`. Resume using `--resume <run-id>`. A resume requires the same project path, source registry, config, and scanner/loader code. The SQLite checkpoint stores individually checksummed rows and uses full synchronous commits. A forced runtime termination may leave `running.lock`; after confirming no process is running, remove only that run's lock before resuming. A run is not safe for concurrent execution.

The supplied config has `immutable_inputs: false`: resume rehashes all encountered files, including checkpointed ones. This avoids trusting size or modification time as proof of content. A new run supplies a full rehash. `immutable_inputs: true` permits reusing completed checkpoint hashes only if the team can guarantee inputs are immutable; changing this setting invalidates existing resume fingerprints. Any suspected modification requires a fresh full rehash.

Each run under `data/manifests/phase2/<run-id>/` includes `checkpoint.sqlite` (checkpoint journal), `file_inventory.jsonl`, `transcript_inventory.jsonl`, `hash_ledger.jsonl`, `collisions.jsonl` and `scan_summary.json`. The summary records output hashes, counts, bytes, elapsed time, and a deterministic digest excluding scan timestamps and run IDs. Errors, unstable files, missing roots, symlinks and name collisions yield `needs_attention`; investigate them without changing the original evidence. The report does not prove source permissions, quality, pairing or speaker separation.

Share the final `scan_summary.json`, `collisions.jsonl`, and the run ID for review. The inventory itself can contain filenames and should stay in the project Drive unless needed for review. The Phase 1 scientific runtime reproduction does not qualify the current runtime for model training.
