# R5-N1 sealed runner (Option 2: loader registry plus runner)

**Status: implemented and tested on synthetic fixtures; the `run` stage has never been executed against real data.** No locked or shared file was edited. Option 2 of `docs/r5_bundle_loading_options.md` is the design; the decision-log entry that records the choice is still to be written by the project owner.

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

```
.venv/Scripts/python.exe scripts/r5_n1_sealed.py plan                  # add --quick to skip re-hashing the A evidence, --load-check to load every bundle
.venv/Scripts/python.exe scripts/r5_n1_sealed.py run --confirm-sealed-run --out reports/final/results/r5_n1_sealed_run
```

Run `plan` on the machine that will run `run` (GPU and the cohort shards present); it must end with "Ready to run on this machine: YES". On a laptop without shards or CUDA it correctly says NO while the verification lines can still be YES (checked on 16 November: all locks, bundles and split hashes verify; this machine has no cohort shards and no CUDA).

## What it will not do

It computes no metric, bootstrap, ROC, PR or reliability plot (hook: `METRICS_HOOK` / the `metrics_hook` argument, called only after every integrity check passed). It does not score the corruption panel (R5-S1 copies are not part of this runner). It edits no locked file, adds no route to `inference.py`, `cli.py` or the app, overrides no threshold, falls back to no other device for AST, and never overwrites a run folder. It does not make the sealed run repeatable: after `run` the test partition has been read once, and a second run needs a new folder and a recorded reason.

## Open items

1. Decision-log entry for Option 2 (who confirmed it); the paper recommends it but records nothing.
2. The B lock records only the split row digest, not the split file hash; the runner checks the digest at run time and writes the B file hash to the access log. Recording that hash in a lock amendment is the owner's call.
3. The cohort index and shard hashes are recorded in no lock; the plan prints the index hash and the log stores it, but nothing verifies it against a reference.
4. The working tree is dirty (`app_support.py` differs from the hash recorded in both locks, and R4-N2 files are uncommitted). The plan reports this as information. The owner should decide whether the sealed run needs a clean, committed tree.
5. The access log is a file the runner writes; filesystem access is not monitored (same limitation as the locks).
6. `run` has not been executed on real data, on a GPU, or with real shards. First real use needs `plan` on the Colab machine, and a decision on how the shards reach it.
7. B WavLM runs on CPU through the router unless a CUDA device is given, in which case it uses the router's own loader function with a device; the equivalence is tested on spec, rule, score type and calibration only with the adapter stubbed.
8. The CLI `evaluate` command named in the plan does not exist; a thin wrapper around this runner can be added later as a recorded additive edit.
