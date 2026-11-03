# P2-07 to P2-09: single Colab workflow

Use `notebooks/P2_07_09_Colab_Audit.ipynb` on a CPU runtime. Sync the notebook, `src/echoproof/`, `configs/phase2.yaml`, `configs/data_sources.json`, and `tests/test_p2_07_09_quality.py` to Drive, together with the existing P2-05 snapshot and P2-06 pairing run. The notebook requires PyYAML, PyArrow, numpy and soundfile. It runs fixture checks first and stops if a check is skipped or fails; it does not install or upgrade packages automatically.

## P2-07

Run cells in order. The inspection reads the header of every candidate audio file and commits progress every 100 newly processed files. This can take hours on Drive. Save the printed run ID. The deterministic sample uses the configured `sample_seed` and a total `sample_count` of 100, distributed across the five audio components and their available header-error, unmatched, under-four-second, and four-second-or-longer strata. Rare strata receive priority through round-robin selection. Selection metadata records exact IDs and strata. Each selected file is decoded at its native sample rate with bounded blocks; decoding is limited to 120 seconds by the current config, 20 million frames, and 32 channels. A resource limit is not classified as corrupt audio. RIFF size, frame counts and finite sample values are checked, and peak/RMS observations are recorded without using them as exclusion thresholds.

To resume, make sure the former runtime has stopped; if the run's `running.lock` remains, remove only that lock. Set `RESUME_P207` to the printed run ID in the setup cell and rerun. Resume requires the same project path, configuration, code and Python/library versions. Completed header checkpoints are reused; this does not rehash audio or establish that inputs have remained unchanged since inspection. Use a fresh run when inputs may have changed. Do not run the same checkpoint concurrently.

Outputs are under `data/manifests/phase2_inspection/<run-id>/`: checkpoint, header inventory, exact selection, sample decode results, admission record and `inspection_summary.json`. `ready_for_review` means no observed header/decode/resource-limit issue, not that every audio file was fully decoded or admitted. A `needs_attention` result still allows the next cell to produce a reconciliation that explicitly carries the issue. User-reported identity review is recorded, but no reviewer details or verified relationship edges are fabricated.

## P2-08

The reconciliation cell checks input hashes and creates a versioned report in `reports/phase2/p2_08-<run-id>/`: `reconciliation.json`, `phonemedf.md` and an artifact manifest. Counts distinguish inventory files, candidate audio records, header coverage, sampled decodes and pending stages. Publisher synthetic totals include voice conversion and cannot be directly compared with the selected TTS subset. Admission, full deduplication, partitions and training-pilot selection stay pending or not run. `phonemedf.md` is the run-specific source card; the stable project entry is `docs/datasets/phonemedf.md`.

If reconnecting, set `INSPECTION` in the P2-08 cell to the completed P2-07 run directory. The cell validates its inputs before producing a new report. Do not treat matching transcript text as evidence that distinct recordings are duplicates.

## P2-09

Enter current Drive account total, used and free values in decimal GB from Google Drive's storage view, plus the remaining work hours allocated to the audit. Leave unknown entries as `None`: the report will remain `needs_attention`. The notebook independently measures free space on `/content`; it does not infer Drive quota from that measurement. The 10.13 GB value reported earlier predates later uploads and is not current.

No download is selected by default. If considering a later acquisition, set the compressed download, expanded, staging and cache byte estimates. The worksheet conservatively compares their sum plus 25% reserve to both storage locations; provide a nonzero expanded-size estimate whenever a download is planned. This is a planning estimate, not authorization to acquire files. The output `reports/phase2/p2_09-<run-id>/` contains `acquisition_plan.json`, a human-readable worksheet and an artifact manifest.

## Review

Share the P2-07 `inspection_summary.json` and `admission.json`, the P2-08 `reconciliation.json` and `phonemedf.md`, and the P2-09 `acquisition_plan.json`. Preserve the associated manifests and checkpoint. These stages produce no training partitions, segments or model results. The selected inventory was edited following reported duplicate cleanup; retained audio hashes have not been independently recalculated since that cleanup.
