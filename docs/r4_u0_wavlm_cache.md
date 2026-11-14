# R4-U0: Protocol A WavLM embeddings the B cache lacks

Code: `src/echoproof/features/r4_u0.py` (planner, extraction, verification, combined reader), `scripts/r4_u0_wavlm_cache.py` (`plan`, `build`, `verify`).
Tests: `tests/test_r4_u0_wavlm_cache.py`. Colab: `notebooks/R4_U0_Colab_WavLM_Cache.ipynb` (GPU). Owner: Usman, with Subhan. Reviewer: to be recorded.

**Status: extracted and verified on Colab (Tesla T4, 14 November 2026); independent review pending.** Evidence: `reports/r4/r4_u0_colab_20261114T224150Z_92e256a0.json`, manifest `data/features/r4_u0_A/r4_u0_manifest.json` (sha256 `8217a743…`).

## What it does

R4-S1 listed the Protocol A rows the B WavLM cache (`data/features/r2_u1_B`, R2-U1) does not hold (`data/features/r4_s1_A/wavlm_cache_gap_A_v1.json`, sha256 `0a489193…`).
R4-U0 extracts exactly those rows into a separate cache, `data/features/r4_u0_A/`, and publishes a manifest only after every check passes.

| Gap rows (18,848) | Count |
|---|---|
| Chatterbox train crops | 7,222 |
| XTTS train crops | 7,090 |
| Chatterbox validation chunks | 4,536 (selection 2,302, calibration 1,065, operating 1,169) |

The B cache is content-addressed (PCM hash, interval, extraction identity, pooling), and the A crops and chunks of recordings shared with B are identical (R4-S1 cross-check, maximum difference 0.0), so those rows are reused, not recomputed.
The A set is 35,782 crops + 24,412 chunks = 60,194 rows; the B cache holds 41,346 of them and the gap is the other 18,848.
A metadata-only `plan` run on the real manifests confirms this: plan sha256 `98ad5eb5c4a2…`, 18,848 rows, coverage 41,346 + 18,848 = 60,194, 15 shards, no test rows.

## Why a new module and a separate cache

`r2_u1.py` is a frozen B source (its hash is in the GR3 lock, and a test checks it still matches) and refuses Chatterbox. `r4_u0.py` reuses its protocol-free helpers (shard packs, atomic writes, content keys, the cached-versus-direct parity check) and restates the planner, plan validation, manifest and reader. The B cache is only read; the new cache is separate so no B artefact changes.

## Checks the run must pass

1. **Evidence chain:** the R4-S1 manifest is verified and matches a verified Colab report; input, crop-bank and gap-file hashes match; the A split hash equals the frozen value; the R1 leakage report passes; the B cache manifest and development index are consistent.
2. **Independent gap:** the gap is re-derived from the A crop bank, the validation chunks and the B development index, and must equal the gap file row for row (and its summary). Any A test assignment is refused, not skipped; test chunks are never planned.
3. **Coverage:** B cache and the new cache partition the A rows exactly (60,194), and the combined reader returns every one of them.
4. **Parity with the cache:** 50 stratified gap rows re-extracted directly must equal the cached vectors within 1e-4, for both poolings.
5. **Parity with B (new):** 50 rows the B cache already holds are re-extracted in this run and must equal B's stored vectors within 1e-4. This shows A and B embeddings can be used together even if the Colab environment differs from the one that built B. Any difference in the extraction identity is recorded in the manifest (`identity_vs_b`), not hidden.
6. Pinned checkpoint and processor, frozen eval mode, fp32, TF32 off, equal-length batches, canonical PCM hash and length checked per recording, packs written atomically and resumable, nothing overwritten.

## Reading the result

`read_embeddings_a(root, partition=None, pooling="mean" | "mean_std", a_dir=None)` returns `(rows, matrix)` for every A train crop and A validation chunk, in the R4-S1 crop-bank and chunk order, with the A item ids, labels, groups and roles, taking each vector from whichever cache holds it and verifying keys and pack hashes. `partition="test"` is refused. It also refuses an A cache without a passed B parity gate.

## Running it

Sync `src/`, `scripts/r4_u0_wavlm_cache.py`, `tests/` and `notebooks/R4_U0_Colab_WavLM_Cache.ipynb` to Drive, open the notebook on a Colab **GPU** runtime and run it top to bottom (about 10 minutes of extraction on a T4 at B's rate, plus staging 15 shards). It writes `reports/r4/r4_u0_{tests,build,verify,colab}_*.json` and `data/features/r4_u0_A/`.
Share the `r4_u0_colab_*.json` report and `r4_u0_manifest.json` for review and for the decision-log entry.

## Limits

- Local tests use a fake extractor; they establish software behaviour only. The real extraction and both parity checks happen on Colab.
- No A test row is extracted. Any sealed-test extraction needed later is a separate logged step.
- AST-A (R4-U1) reads audio directly; this cache serves WavLM-head-A only.
- Review by a teammate is a separate status and is not claimed here.

## Result (Colab run, reviewed 15 November 2026)

| Item | Value |
|---|---|
| Rows extracted | 18,848 (Chatterbox 11,758; XTTS 7,090); train 14,312, validation 4,536 (selection 2,302, calibration 1,065, operating 1,169) |
| Coverage | 41,346 B-cache rows + 18,848 new rows = 60,194 A rows, each exactly once; the combined reader returned all 60,194 at both poolings (mean 768, mean+std 1,536; float32, finite), 0 test rows, 60,194 unique item ids |
| Extraction identity | **identical to B's** (`identity_vs_b: equal`): transformers 5.22.0 (pinned), torch 2.14.1+cu130, numpy 2.4.6, huggingface-hub 1.33.0, same source hashes for `pretrained.py` and `r2_u1.py` |
| Parity, new rows (cached vs direct, 50) | max abs difference 1.8e-6 (mean), 2.0e-6 (mean+std); tolerance 1e-4 |
| Parity against B's stored vectors (50 B rows re-extracted) | max abs difference 4.7e-6 (mean), 7.7e-6 (mean+std); tolerance 1e-4 |
| Time | extraction 560 s over 15 shard packs; whole run 1,052 s |
| Size | 196,456,079 bytes (packs plus index) |
| Plan sha256 | `2d173ba5a041…` (the Colab plan, with the observed Colab identity; the local metadata-only plan `98ad5eb5c4a2…` used the default identity, so the two differ by design while their 18,848 rows are the same) |
| Tests on Colab | 70 passed, 0 skipped |
| Pack and manifest hashes | all 15 packs and the manifest on this machine equal the manifest/report values |

The new vectors reproduce B's stored vectors to about 1e-5, so A-cache and B-cache embeddings can be used together through `read_embeddings_a`.

## Notes recorded with the run

- **`contracts.py` on Drive is older than the repository.** The Colab source hash of `src/echoproof/contracts.py` (`20483ebc…`) differs from the repository and the GR3 lock (`c17aa041…`). The only difference between the two versions (commit `f2c910c`, 9 November) is a docstring on `ModelAdapter`, with no code change. The same older copy was used by the R2-U1 and R4-S1 Colab runs. No result is affected, but the Drive copy should be refreshed from git.
- **`r4_u0.py` was changed after the run.** `read_embeddings_a` re-read each NPZ array on every row access, which made the full read take over half an hour locally. It now loads each pack array once; behaviour and outputs are unchanged (the 23 R4-U0 tests still pass, and the full union read now takes about 70 s). The build, verification and parity code the run used is untouched, but the file's hash differs from the one in the Colab report (`261d3833…` then, now `1ad072ee…`).
- Review by a teammate is not recorded here.
