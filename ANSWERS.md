# How the dataset was broken down: speakers, planned stems, protocols and chunks

Every figure below comes from a report in `reports/r1/` or from the code in `src/echoproof/data/`. Each claim names its source. Where the evidence doesn't say something, the text says so.

Sources used:
- `reports/r1/admission.json` (R1-S1)
- `reports/r1/split_report.json` (R1-S2)
- `reports/r1/materialize.json` (R1-S3)
- `reports/r1/repair_20261104T074150Z.json` (R1-S3 repair)
- `reports/r1/leakage.json` (R1-S4)
- `src/echoproof/data/cohort.py`, `splits.py` and `chunks.py`

---

## 1. Speaker: definition, count and extraction

**Definition.** A speaker is the first number in a LibriSpeech-style file stem `speaker-chapter-utterance`. The stem `103-1240-0001` belongs to speaker `103`, chapter `1240`, utterance `0001`.
- `cohort.py:21` defines the pattern `^(\d+)-(\d+)-(\d{4})$`.
- `parse_stem` (`cohort.py:28`) returns `(speaker, chapter, utterance)`.
- `speaker_group` (`cohort.py:43`) turns it into the grouping key `speaker:<id>`.
- `grouping_unit` in `admission.json` is `"speaker"`.
- Speaker identity is read from the file name only. The code doesn't check it against the audio.
- Each of the five recordings made from a stem (`genuine`, `xtts`, `chatterbox`, `melo`, `vits`) gets the speaker of that stem.
- The code doesn't state, and I haven't checked, whose voice each TTS system actually imitated. The grouping follows the stem.

**Total.** There are **251 speakers** (`admission.json: "speakers": 251`).
- They come from 585 chapters.
- No chapter is shared by two speakers (`chapters_shared_by_multiple_speakers: []`).
- There are 28,539 stems in total (`stems_admitted: 28,539`, `stems_excluded: {}`).
- Utterances per speaker: minimum 26 (speaker 7312), median 114, maximum 166 (speaker 211).

**Extraction.** The R1-S1 `admission()` function (`cohort.py:54`) works as follows:
1. It reads the Phase 2 coverage rows (one `candidate_stem` per row).
2. It rejects any stem that doesn't match the pattern. `format_errors` is empty and `format_consistent` is true.
3. It requires exactly one file for each of the five components, plus one transcript, per stem.
4. It excludes a stem from every component if any component header isn't `ok`.
5. It groups the surviving stems by `parse_stem(stem)[0]`.

None were excluded, so all 28,539 stems were admitted.

**Why speaker is the grouping unit.** Speaker is the independence unit for splitting (`splits.py` docstring). A speaker's recordings land in only one partition (train, validation or test), so the model can't be tested on a voice it trained on.

---

## 2. `planned_stems`

**`planned_stems: 3510` is not the number of recordings used to train.** It is the size of the sampled cohort across all partitions: train, validation and test combined.

It is also not the 28,539 admitted stems. R1-S2 draws a fixed number of stems per speaker (`sample_stems`, `splits.py:70`). The report shows `per_speaker: {train: 10, validation: 20, test: 20}`.

| Partition | Speakers | Stems per speaker | Stems |
|---|---|---|---|
| train | 151 | 10 | 1,510 |
| validation (selection 25, calibration 12, operating 13) | 50 | 20 | 1,000 |
| test | 50 | 20 | 1,000 |
| **Total** | **251** | | **3,510** |

The 60/20/20 split is by speakers: 151, 50 and 50 (`speaker_counts` in `split_report.json`). The validation speakers are split 50/25/25 into selection, calibration and operating roles.

Each stem produces five recordings (genuine plus four TTS), so 3,510 × 5 = **17,550 recordings**. This matches `materialize.json: "recordings": 17550` and `split_report.json: protocols.A.recordings`.

Of those, **only the train partition trains the model**. In Protocol B that is 4,530 recordings (1,510 genuine plus 3,020 TTS). In Protocol A it is 7,550. The validation and test partitions are held out.

---

## 3. Protocols A and B

Both are filters over the same 3,510 stems. They differ in which TTS components are allowed in each partition (`PROTOCOL_A` and `PROTOCOL_B`, `splits.py:27-36`).

| | Train | Validation | Test |
|---|---|---|---|
| **A** | genuine, xtts, chatterbox, melo, vits | the same five | the same five |
| **B** | genuine, melo, vits | genuine, melo, vits + **xtts** (`unseen_validation`) | genuine, melo, vits + **chatterbox** (`unseen_final`) |

Recordings: A has **17,550** and B has **12,530** (`split_report.json`).
- A: 3,510 × 5.
- B: 1,510 × 3 + 1,000 × 4 + 1,000 × 4.

- **Protocol A (all four TTS families known).** A model trains on every TTS family and is tested on the same families, using new speakers. It answers: can the model detect families it has seen, when the voice is new?
- **Protocol B (two TTS families held out).** The model trains only on Melo and VITS. XTTS appears only in validation and counts as an unseen family there, and is used to select the model. Chatterbox appears only in the final test (`unseen_final`). It answers: can the model detect a TTS system it has never seen?

**Why they were made.** My source for the intent is the code and `REVISED_PLAN.md`, not the original research contract.
- B tests generalisation to an unseen generator. A tests the easier in-family case.
- The plan's hard rule H7 says no Protocol A training and no extraction of Chatterbox train or validation data may happen until the B bundles are locked at GR3. It calls this the "firewall in the research contract". The firewall keeps Chatterbox from leaking into B's selection.
- B uses XTTS for validation and Chatterbox for the final test, so model selection never sees the final-test generator.

---

## 4. Chunks, and why 150,612 isn't comparable to 142,695

**What a chunk is.** A chunk is a fixed-position evaluation window inside one recording (`chunks.py`).
- It is 4 s long (64,000 samples at 16 kHz) with a 2 s stride.
- If the recording doesn't end on a window boundary, one extra full 4 s window is added that ends at the last sample.
- A recording of 4 s or less becomes a single chunk with its true length. Audio is never repeated to fill a window.
- A chunk ID is `chunk:{protocol}:{recording_id}:{start}`.
- A long recording therefore produces many chunks.

An example from `eval_windows` for a 13 s clip: starts at 0, 2, 4, 6 and 8 s, plus a final window at 9 s. That is 6 chunks from one recording.

**Why the chunk total is larger than the recording count.** Two things combine.
1. **Chunks are per window, not per file.** The average recording is about 10.9 s (`split_report.json` storage estimate: 53.22 hours over 17,550 files), so most files give several chunks.
2. **Chunks are counted separately for each protocol.** The same recording appears in both A and B (B's recordings are a subset of A's). `leakage.json` shows the chunk counts per protocol:

| Protocol | Assignment rows | Chunks |
|---|---|---|
| A | 17,550 | 87,082 |
| B | 12,530 | 63,530 |
| **Total** | **30,080** | **150,612** |

That is about 5.0 chunks per assignment row.

**The comparison to 28,539 × 5 = 142,695 doesn't hold.**
- 28,539 is the pool of admitted stems. Only 3,510 were sampled (section 2). The recordings actually used are 17,550, not 142,695.
- The two totals count different things. 142,695 would be every recording in the full pool, while 150,612 is windows over a 17,550-recording sample, counted once for each protocol. The 7,910 gap between them is a coincidence, not a meaningful quantity.
- The numbers in the report are consistent with each other. 30,080 assignment rows × about 5.0 chunks per row gives 150,612.

**150,605 versus 150,612.**
- `materialize.json` reports `"chunks": 150605`. That was before the repair, when the one recording that hit a transient Drive read error (`phonemedf:xtts:103-1240-0001`) was excluded.
- The repair report shows `chunks_before: 150605`, `chunks_after: 150612`. The recovered recording contributes 7 chunks, which is the difference.
- `leakage.json` is the post-repair figure. Its protocol sums 87,082 + 63,530 equal 150,612.
- The same repair raised `retained` from 17,549 to 17,550.

**Training doesn't use these chunks as its samples.**
- The chunk manifest is fixed windows for evaluation. The plan's contract line says `mode ∈ {eval}`.
- It also says "Training crops are random per epoch and seeded, so they are not stored" (`REVISED_PLAN.md:132`).
- Training draws one random 4 s crop per recording per epoch (R1-M1).
- Train-partition recordings do have chunk rows, as part of the manifest for all partitions.

---

## 5. Shards: what they are, how they are made, and how many

**What a shard is.** A shard is an uncompressed `.tar` archive that holds many recordings as individual FLAC files. The cohort's 17,550 recordings live in these archives on Drive (`data/processed/cohort_v1/`) rather than as 17,550 loose files.
- The audio inside is the canonical form: 16 kHz mono, stored as FLAC (`SETTINGS` in `chunks.py`: `int16_flac_canonical`).
- The tar itself doesn't compress anything. Only the FLAC inside is compressed.
- I haven't verified why shards were chosen over loose files. The plan's use of them is for storage on Drive and for reading many small files efficiently.

**How they are made.** `write_shards` (`cohort.py:205`) is called from `materialize` in the R1-S3 step (`scripts/r1_data_lock.py materialize`).
1. Each recording is read, decoded, resampled once if needed and converted to mono 16-bit PCM. It is hashed (`pcm_sha256`) and encoded as FLAC.
2. Each FLAC blob is appended to the current tar as a member named `<recording_id>.flac`, with `:` replaced by `__`. For example, `phonemedf__xtts__103-1240-0001.flac`.
3. When the shard's running size reaches the threshold of **256 MiB** (`shard_bytes`), it is closed and the next one is opened. Names are numbered `shard_00.tar`, `shard_01.tar`, and so on. A shard can end slightly above 256 MiB, because the size is checked before each new member is added.
4. The function refuses to overwrite an existing shard file (`CohortError: refusing to overwrite`).
5. The manifest records where each recording went, in `canonical_path = shard.tar::recording_id`. Loading goes through this path (`read_from_shard`), and the manifest is the source of truth.

**Count: 15 shard files, 17,550 recordings.** I counted the `canonical_path` values in the local `data/manifests/cohort_v1.parquet`:

| Shards | Recordings each |
|---|---|
| `shard_00` – `shard_12` (13 shards) | 1,305 – 1,406 |
| `shard_13` | 94 |
| `shard_14` | 1 |
| **Total** | **17,550** |

- `shard_13.tar` is the last shard from the original run. It is partly filled, because the data ran out.
- `shard_14.tar` holds a single recording, `phonemedf:xtts:103-1240-0001`. It is the repair shard from the transient Drive read error (`repair_shards: ["shard_14.tar"]` in the repair report). It was created as `repair_00.tar` and you renamed it to `shard_14.tar`.
- The manifest was updated to point at that name.
- Total size of the original shards: 3.511 GB (`materialize.json: shard_gb`). The repair shard adds 298,868 bytes.
- The shard count isn't fixed by design. It follows from the 256 MiB threshold and the data size, and the counts above come from the manifest.

**The stale informational index.** `data/processed/cohort_v1/index.parquet` still describes only the original run: 17,549 recordings and 14 shards. It was written during materialize and not updated by the repair. It is informational only. Loaders use the manifest's `canonical_path`, so this doesn't affect later phases.

---

## Summary

| Quantity | Value | Source |
|---|---|---|
| Admitted stems | 28,539 | `admission.json` |
| Speakers | 251 | `admission.json` |
| Planned stems | 3,510 | `split_report.json` |
| Recordings, Protocol A | 17,550 | `split_report.json` |
| Recordings, Protocol B | 12,530 | `split_report.json` |
| Chunks (post-repair), A + B | 87,082 + 63,530 = 150,612 | `leakage.json`, repair report |
| Shard files | 15 (`shard_00`–`shard_14`) | `data/manifests/cohort_v1.parquet` |
| Leakage violations | 0 | `leakage.json` |
