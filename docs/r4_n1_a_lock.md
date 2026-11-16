# R4-N1: the Protocol A lock record (`locks/A_lock.json`)

Code: `src/echoproof/evaluation/r4_n1_a_lock.py` (build and verify), `scripts/r4_n1/a_lock.py` (dry run by default; `--write`, `--allow-dirty`, `--verify`), `tests/test_r4_n1_a_lock.py`. Modelled on `scripts/r3_all/gr3_lock.py`; no B file is edited and `locks/B_lock_gr3.json` is not touched.

**Status: written on 16 November 2026 (`locks/A_lock.json`, sha256 `8613ff1e22e7bfa4...`), built from a clean working tree at git head `cd99944`; committed in `70d11ac`.** `scripts/r4_n1/a_lock.py --verify locks/A_lock.json` re-checks it against the files.

## What the record binds

For each of the four bundles (WavLM-A calibrated, AST-A candidate, LR-A and BiGRU-A raw-with-threshold): bundle id, model version, `bundle.json` hash, every component hash and size, the serialized decision rule, bundle flags (all `promoted: false`, `locked: false`), the calibration outcome (mapping, parameters, acceptance, score type), the operating-role result at the rule (macro F1, genuine FPR, confusion counts, threshold plateau), the replay or parity evidence (kind, recordings, tolerance, largest difference, flips), and a hashed reference to every evidence file (runs, reports, recipes, source bundles). The record also holds the A split hash and role counts, the R4-S1 manifest hash, the calibration procedure and its status, the code state (git head, dirty flag and the dirty entries, hashes of `src/echoproof/**` and `scripts/r4_*.py`), a partially verified test-access audit (test partition not accessed), the review status (none recorded), and ten limitations.

Every fact is read from the files and re-verified when the record is built and again when it is written: bundle components load through the verified loader, hashes match, parity reports belong to the same `bundle.json`, thresholds equal the bundle's rule, no run reports test data read, and the Protocol B lock is byte-unchanged (its recorded hash is of the CRLF form; an LF checkout is accepted). Any missing file or mismatch stops the build.

## What it deliberately does not do

- It selects no deployment model (`production_selection.selected` is null). Production remains the B WavLM bundle of `locks/B_lock_gr3.json`; using an A bundle is an R4-N2 decision.
- It does not mark any bundle promoted or locked, and it records no independent review (none exists).
- It does not claim equal evidence for the four models: WavLM-A, LR-A and BiGRU-A were replayed on 1,500 recordings (all operating plus 100 each of selection and calibration); AST-A has a sampled CUDA fp16 parity (all chunks of 50 selection recordings, receipt `approved: false`), AST-A is verified on CUDA fp16 only, and no A bundle has latency or memory measurements.

## Dry run (16 November 2026, dirty tree)

| Model | Rule | Mapping, accepted | Operating macro F1 / genuine FPR | Replay |
|---|---|---|---|---|
| WavLM-A | `>= 0.625407` | affine, yes | 1.0000 / 0 of 260 | 1,500 recordings, 0 flips, max difference 2.6e-6 |
| AST-A | `>= 0.970261` | temperature, yes | 0.9964 / 1 of 260 | 50 recordings (163 chunks), 0 flips, 4.0e-7 |
| LR-A | `>= 0.591080` | identity, no | 0.8912 / 47 of 260 | 1,500 recordings, 1 explained flip, 3.9e-7 |
| BiGRU-A | `>= 0.610877` | identity, no | 0.9741 / 19 of 260 | 1,500 recordings, 1 explained flip, 1.2e-7 |

## How it was written

1. The working tree was committed and merged with the teammates' push first (the lock records `git_head`; a dirty tree is refused unless `--allow-dirty`).
2. `.venv/Scripts/python.exe scripts/r4_n1/a_lock.py` (dry run, summary reviewed), then `... --write`. It refuses to overwrite. Re-check any time with `--verify locks/A_lock.json`.
3. Tests: `tests/test_r4_n1_a_lock.py` (7 tests; they hash the 345 MB AST model and take about a minute).

## What the lock does not cover (next)

It is a development lock: it does not authorise any use of the A bundles beyond the sealed evaluation (R5). How the sealed evaluation loads the eight bundles is analysed in `docs/r5_bundle_loading_options.md`; the app and fresh-install work (R4-N2, R4-N3) is planned in `docs/r4_n2_n3_plan.md`.

## Open

Independent reviews, R4-N2 (app routing for A bundles), latency and memory for A bundles, and R5 on the sealed test partition. Policy items beyond the calibration recipe have no user-reported decision for A yet.
