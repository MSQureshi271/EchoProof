# R5-N1 sealed runner (Option 2: loader registry plus runner)

**Status: implemented and tested on synthetic fixtures; the `run` stage has not been verified against real data.** Option 2 of `docs/r5_bundle_loading_options.md` is the design; the 16 November decision-log entry records the owner-reported choice. Before a real run, resolve the clean-only scope change-control and GR4 prerequisites recorded there.

Files (all new): `src/echoproof/evaluation/r5_loaders.py`, `scripts/r5_n1_sealed.py`, `tests/test_r5_loaders.py`, `tests/test_r5_sealed_runner.py`.

## What it does

**Registry (`r5_loaders.py`).** One entry for each of the eight bundles named in `locks/B_lock_gr3.json` and `locks/A_lock.json`, and nothing else; `candidate_bundles` refuses if a lock names a bundle the registry lacks or the registry names one a lock lacks, or if a model version differs.

| Key | Loader | Device and precision |
|---|---|---|
| B:LR | shared router `load_inference_bundle` | CPU |
| B:WavLM | shared router on CPU; on CUDA the function the router calls (`load_wavlm_bundle`) with a device | CPU or CUDA |
| B:BiGRU | `load_calibrated_bigru_bundle(..., allow_pending_review=True)` | CPU or CUDA |
| B:AST | `load_ast_platt_candidate` | CUDA fp16 only |
| A:WavLM | `load_wavlm_bundle_a` | CPU or CUDA |
| A:AST | `load_ast_a_bundle` | CUDA fp16 only |
| A:LR | `load_lr_bundle_a` | CPU |
| A:BiGRU | `load_bigru_bundle_a` | CPU or CUDA |

The B BiGRU bundle's calibration review is "pending". Its loader refuses that unless told otherwise, so the registry passes `allow_pending_review=True` explicitly (constant `ALLOW_PENDING_REVIEW_BIGRU_B`) and still refuses any review status other than `pending` or `approved`. An AST bundle without a CUDA device is refused, never run on CPU. A threshold override is refused for all eight bundles at the registry, and the A loaders refuse it themselves as well.

**Verification.** `verify_b_lock` and `verify_a_lock` check the lock file hash (the B lock's recorded hash is of the CRLF form; LF and CRLF checkouts are both accepted), the lock identity, the split hash, the candidate set, and for A (deep mode) every evidence hash through `r4_n1_a_lock.verify_record`. `A_LOCK_SHA256` is pinned in the module from the committed file (`8613ff1e22e7bfa4...`, matching `docs/r4_n1_a_lock.md`). `verify_bundle` re-checks each bundle: `bundle.json` hash, every component hash, version, id, preprocessing hash and decision rule against the lock record. After loading, `check_loaded` confirms the object is not a mock and has the locked id, version and rule.

**Runner (`r5_n1_sealed.py`).**

- `plan`: verifies both locks, all eight bundles, both split files (bytes hashed, never parsed; the A file against its lock), the audio index and shards, and the device policy; prints exactly what `run` would score; reads no test rows and writes nothing. It also lists, for information, shared source files that differ from the hashes recorded in each lock.
- `run --confirm-sealed-run`: order is confirmation, run-folder check, verified plan, device preflight (all before anything is written), then `test_access_log.jsonl` is created and its first event flushed, and only then `splits.load_assignments(..., sealed=True)` is called. It verifies the recomputed split row digest against the locked split hash, keeps only the test partition, scores every test recording with each bundle through the unchanged shared `predict_recording`, and runs the integrity checks per bundle before anything else: missing, duplicate and unexpected recordings, count mismatch, any failed inference result or unreadable audio (counted, never dropped), non-finite or out-of-range scores, and decisions that disagree with the locked threshold. It writes `predictions_<protocol>_<model>.parquet` per bundle and `integrity_report.json` (also written when the run stops early, with status `aborted`). The run folder defaults to `reports/final/results/r5_n1_sealed_run` and is never overwritten.

## How to run

For a Drive-backed Colab GPU runtime, use `notebooks/R5_N1_Clean_Colab.ipynb`.
It defaults to tests and full preflight only. The sealed run, interrupted-run
recovery and analysis have separate switches; execution requires the notebook's
three prerequisite confirmations. Its output lives directly on Drive. The
notebook has not yet been executed end to end on Colab.

```
.venv/Scripts/python.exe scripts/r5_n1_sealed.py plan                  # add --quick to skip re-hashing the A evidence, --load-check to load every bundle
.venv/Scripts/python.exe scripts/r5_n1_sealed.py run --confirm-sealed-run --out reports/final/results/r5_n1_sealed_run
```

Run `plan` on the machine that will run `run` (GPU and the cohort shards present); it must end with "Ready to run on this machine: YES". On a laptop without shards or CUDA it correctly says NO while the verification lines can still be YES (checked on 16 November: all locks, bundles and split hashes verify; this machine has no cohort shards and no CUDA).

## What it will not do

It computes no metric, bootstrap, ROC, PR or reliability plot (hook: `METRICS_HOOK` / the `metrics_hook` argument, called only after every integrity check passed). It does not score the corruption panel. It edits no locked file, adds no route to `inference.py`, `cli.py` or the app, overrides no threshold, falls back to no other device for AST, and never overwrites completed predictions.

## Analysis of a completed clean run

`scripts/r5_n1_clean_analysis.py` is separate from scoring. After the completed run folder and all locked evidence are available, run:

```
python scripts/r5_n1_clean_analysis.py --run reports/final/results/r5_n1_sealed_run
```

It reads no audio and no split rows. Before calculating any metric it checks the complete access log and integrity report, the locks and their evidence, eight bundle identities, prediction file hashes, recording coverage and labels, locked decisions and panel support. It writes a new `analysis_clean_v1` folder containing `metrics.json`, `comparison.csv`, twelve PNG figures (four figure types per panel), and output hashes. It refuses an existing output folder. If plotting fails after the folder is created, the folder is incomplete and must be inspected; use a new output path rather than assuming it is complete. Analysis dependencies include the `analysis` extra (`matplotlib`) as well as numpy, scipy, scikit-learn and pyarrow.

Panels are A genuine plus all four known families, B genuine plus Melo/VITS, and B genuine plus unseen Chatterbox. Each model uses its own locked threshold. The 1,000 paired speaker-cluster bootstrap resamples (seed 42, 95% percentile intervals) cover macro F1, FPR, TTS recall and ROC-AUC; paired differences use the same speaker draws. Skipped single-class replicates are counted. These intervals do not quantify training-seed variability. A/B recordings and model-specific coverage are checked for agreement. Reliability bins use ten equal-width bins; LR-A and BiGRU-A raw model scores remain identified as such and their probability-loss and reliability values do not claim calibration. EER is descriptive, not a new threshold. Protocol D remains not measured.

The analysis code and figures have been exercised on synthetic predictions only. Run the full verification on the actual persistent environment after the legitimate sealed scoring has completed. `reports/r1/split_report.json`, cited in the output provenance, supplies the expected panel supports; the split hash in that report is checked against the lock. The report is supporting metadata, not itself a locked byte hash.

## Interrupted run recovery

The runner logs each recording before scoring and flushes a hash-chained journal entry after scoring. The run identity binds locks, source hashes, split file hashes, bundle identities, shard/index hashes, selected devices, package versions, Python and accelerator details. Computing shard hashes on Drive may take substantial time; they are checked again on recovery. Journals and access logs contain test identities and predictions and must remain in the persistent run directory. They are not a substitute for independent monitoring or immutable storage.

For a genuine interruption, after confirming the original process stopped, use the **same environment and run folder**:

```
python scripts/r5_n1_sealed.py run --confirm-sealed-run --out reports/final/results/r5_n1_sealed_run --resume --resume-reason "Colab disconnected; previous worker stopped"
```

Recovery logs the reason before reopening the sealed split, checks the run identity and reuses complete journaled predictions. A stale `active_run.claim` after a killed process requires `--recover-stale-claim`, only after independently confirming the previous process is no longer running; the flag cannot determine this itself. Concurrent recovery could corrupt the run. An incomplete log, damaged journal, changed identity, integrity failure or a recording with a start event but no saved result fails closed for manual review. In the last case it is unknown whether scoring finished: the runner does not silently rescore it. A partial result, changed environment or a failed run requires a documented team decision before further test access. Resume is a further read of the sealed split, not proof of a single physical read.

Journaling and start logging are flushed per recording; this improves recoverability but adds Drive I/O and affects runtime. Atomic rename protects final prediction files from partial writes, subject to the mounted filesystem's guarantees. No recovery path has yet been tested on real Colab shards or GPUs.

## Open items

1. Team confirmation and change-control assessment for the clean-only Protocol D omission; Option 2 is recorded as owner-approved in the decision log.
2. The B lock records only the split row digest, not the split file hash; the runner checks the digest at run time and writes the B file hash to the access log. Recording that hash in a lock amendment is the owner's call.
3. The cohort index and shard hashes are recorded in no lock; the plan prints the index hash and the log stores it, but nothing verifies it against a reference.
4. The working tree is dirty (`app_support.py` differs from the hash recorded in both locks, and R4-N2 files are uncommitted). The plan reports this as information. The owner should decide whether the sealed run needs a clean, committed tree.
5. The access log is a file the runner writes; filesystem access is not monitored (same limitation as the locks).
6. `run` has not been executed on real data, on a GPU, or with real shards. First real use needs `plan` on the Colab machine, and a decision on how the shards reach it.
7. B WavLM runs on CPU through the router unless a CUDA device is given, in which case it uses the router's own loader function with a device; the equivalence is tested on spec, rule, score type and calibration only with the adapter stubbed.
8. The CLI `evaluate` command named in the plan does not exist; a thin wrapper around this runner can be added later as a recorded additive edit.
