# R3-S1: corruption suite builder

Status: **specification version 2 (10 November 2026) built, tested, and verified in Colab on a real validation sample** (`reports/r3/spec_v2/r3_s1_colab_20261110T152638Z.json`). Nouman's review is pending. Version 1 was also run in Colab (verified); its evidence is kept unchanged. Owner: Subhan. Reviewer: Nouman. Code: `src/echoproof/data/corruption.py` (transformations), `src/echoproof/data/corruption_sample.py` (validation-sample run), `scripts/r3_s1_corruption_sample.py`, `notebooks/R3_S1_Colab_corruption_sample.ipynb`. Tests: `tests/test_r3_s1_corruption.py` (27), `tests/test_r3_s1_sample.py` (10), `tests/test_r3_s1_mp3_probe.py` (21). Version 2 acceptance check: `scripts/r3_s1_v2_acceptance.py`.

This task builds and checks the transformations. It does **not** generate the stored corrupted copies of the locked test recordings; that is R5-S1, after GR4. No test data, calibration data, operating data or ChatterboxTTS audio is read here.

## Frozen specification (version 2)

Specification hash: `d68c4ed61a29ad010bfb10b8ac8995dad1c02af72a914828a050613c6fc805e0` (`corruption.SPEC_SHA256`; pinned in the tests, so a deliberate change must update the test and the decision log together). Master seed 42.

**History.** Version 1 (hash `1c3546287e922731df2faa7ab67f86dfb2287533b31b3751ac422ce022b85b55`) had `mp3_64` where version 2 has `mp3_32`. A measurement showed 64 kbps changes 16 kHz speech far less than the noise conditions do (see below), so on 10 November 2026 the project owner amended it. Only `mp3_32` differs: `mp3_128`, `noise_20` and `noise_10` produce byte-identical output in both versions (checked on the fixture). The condition count stays four, so R5-S1 is unaffected. The plan's R3-S1 row names 64 and 128 kbps; the decision log records the amendment and `REVISED_PLAN.md` itself has not been edited.

| Condition | Transformation |
|---|---|
| `mp3_32` | MP3 through ffmpeg/libmp3lame, constant 32 kbps, 16 kHz mono, decoded back to 16-bit PCM (version 1: `mp3_64`, 64 kbps) |
| `mp3_128` | Same at 128 kbps |
| `noise_20` | Additive noise at 20 dB SNR |
| `noise_10` | Additive noise at 10 dB SNR |

- **SNR** is `10*log10(mean(signal^2) / mean(noise^2))` over the whole recording (global power). Measured SNR must be within 0.5 dB of the target. The check is skipped only when the required noise is below about 8 least-significant bits, where 16-bit rounding dominates, and the report counts those cases.
- **Noise colour** is decided once per parent recording from `sha256(seed|parent_id|noise-color)`: even first byte is white, odd is pink. Both noise conditions use the same colour for a recording. White is standard Gaussian. Pink is white shaped by 1/sqrt(f) in the rFFT domain. Both are made zero-mean and scaled to unit power before the SNR scaling. The noise seed is `sha256(seed|parent_id|condition)`.
- **MP3 alignment.** ffmpeg adds a fixed 1,105-sample delay (576 encoder plus 529 decoder), and the tag that lets it trim this itself is not written when output goes to a pipe. The builder drops those 1,105 samples explicitly, keeps the input length, and then checks that the decoded length is within one 1,152-sample frame of the input and that the lag-0 correlation with the input is at least 0.9. A misaligned or wrong-length decode raises an error instead of being used.
- **Level.** If noise or the codec pushes the peak above 0.999, the whole output is scaled by one gain. One gain leaves the SNR unchanged. The gain is recorded per variant.
- **Storage.** Every output is quantised to 16-bit PCM and returned as the values a decoder would read back. `pcm_sha256` hashes that PCM, so scoring a variant in memory equals scoring it after storage.
- **Identity.** Parent IDs are kept. A variant is `<parent_id>@<condition>`. Lengths equal the parent's.
- **ffmpeg safety.** Fixed argument lists, no shell, audio over pipes, 60 s timeout. Inputs shorter than 4,096 samples, silent, non-finite, non-mono or beyond full scale are rejected.

## What the tests establish (local, 58 tests including the probe)

Exact four-condition spec and pinned hash; determinism; per-recording seeding; SNR accuracy for both colours at both levels (within 0.1 dB on speech-like signals); white spectrum flat and pink slope near -1; zero-mean unit-power noise; the level guard preserves SNR; outputs are exactly representable 16-bit values; inputs are not modified; every rejected input; MP3 length, correlation, bitrate and determinism; higher bitrate is closer to the original; ffmpeg is called with fixed lists and no shell; a shifted or truncated decode is rejected; missing, failing and timed-out ffmpeg give clear errors. A mutation check (removing pink shaping, offsetting the SNR by 3 dB, disabling the MP3 delay trim, disabling the level guard, and leaving `mp3_32` at 64 kbps) made the tests fail each time. The sample runner is tested on a miniature cohort: selection-role-only scope, no Chatterbox, repeat-hash identity, an injected failure marks the report failed, finished outputs are never overwritten, and no audio is written.

## Validation-sample run (Colab; versions 1 and 2 done)

`notebooks/R3_S1_Colab_corruption_sample.ipynb` (now version 2; it writes to `reports/r3/spec_v2/` so version 1's files in `reports/r3/` are not touched or reloaded) applies the four conditions to 8 recordings per component (32 recordings, 128 variants) drawn by hashed order from the selection role. It writes `reports/r3/r3_s1_corruption_sample.json`, `reports/r3/r3_s1_corruption_sample_variants.csv` and an evidence file `reports/r3/r3_s1_colab_<timestamp>.json`, and it independently recomputes one variant per condition from the shards. The report includes the effect on the R2-S2 measurement columns (noise should raise the noise floor; MP3 should remove high-band energy), the ffmpeg build, and a synthetic fixture digest. The fixture digest is compared with the project machine's value, informationally: a different ffmpeg build can change MP3 bytes.

## Result of the version 1 Colab validation sample (10 November 2026)

This is the version 1 run (`mp3_64`, `mp3_128`, `noise_20`, `noise_10`). Evidence: `reports/r3/r3_s1_colab_20261110T114146Z.json` (verified: 37 tests passed on Colab, 128 variants built, all repeat hashes identical, independent recomputation matched, selection role only). Noise SNRs were exactly 10.000 and 20.000 dB on all 64 noise variants; the level guard fired on 5 of 128 variants (minimum gain 0.94). MP3 round trips aligned (correlation at least 0.9964) at 64.7 and 129.3 kbps effective.

## MP3 strength measurement (10 November 2026)

The two frozen MP3 conditions changed the audio little, so 11 MP3 settings were measured on the same seeded sample of 32 selection recordings (8 each of genuine, MeloTTS, VITS, XTTS; 20 speakers). Code: `src/echoproof/data/mp3_probe.py`, `mp3_probe_run.py`, `scripts/r3_s1_mp3_probe.py`, `notebooks/R3_S1_Colab_mp3_probe.ipynb`; 20 tests in `tests/test_r3_s1_mp3_probe.py`. It is signal-level only: each variant is compared with its own original, and no model scores anything. LAME scales the signal by a fixed factor of about 0.95 (-0.45 dB), which caps a raw SNR near 26 dB, so the comparison with the noise conditions uses the SNR after removing that level change.

Evidence: Colab run `reports/r3/r3_s1_mp3_colab_20261110T135522Z.json` (verified: 57 tests passed, 352 rows with no errors, independent recomputation matched, output hashes match); outputs `r3_s1_mp3_probe_shards.json`, `_rows.csv`, `.png/.svg`. An earlier local run on the 16 genuine and XTTS recordings available on this machine (`r3_s1_mp3_probe_local.*`, files verified against the manifest hash) gave the same values.

| Setting | SNR after removing the level change, median [range] | Energy kept above 4 kHz / 7 kHz (median) | Compared with the noise conditions |
|---|---|---|---|
| 16 kHz, 128 kbps (frozen `mp3_128`) | 33.8 dB [21.4, 49.4] | 86% / 30% | 14 dB milder than noise_20 |
| 16 kHz, 64 kbps (frozen `mp3_64`) | 30.7 dB [20.7, 37.2] | 87% / 29% | 11 dB milder than noise_20 |
| 44.1 kHz, 128 kbps | 28.4 dB [17.9, 33.8] | 87% / 43% | milder |
| 44.1 kHz, 64 kbps | 25.7 dB [15.4, 31.2] | 88% / 45% | milder |
| **16 kHz, 32 kbps** | **20.0 dB [10.0, 24.4]** | 92% / 30% | **equal to noise_20 (median)** |
| 44.1 kHz, 32 kbps | 17.4 dB [6.6, 23.4] | 88% / 33% | between noise_20 and noise_10 |
| 8 kHz, 32 kbps | 17.4 dB [0.6, 27.0] | 0.1% / 0% | band-limited to 4 kHz |
| 16 kHz, 24 kbps | 17.3 dB [7.5, 21.3] | 62% / 0% | between |
| 16 kHz, 16 kbps | 14.7 dB [4.2, 18.6] | 46% / 0% | between |
| 8 kHz, 16 kbps | 14.2 dB [-0.2, 18.9] | 0.1% / 0% | band-limited |
| 16 kHz, 8 kbps | 11.2 dB [-3.1, 14.1] | 0% / 0% | near noise_10; some recordings below 0 dB |

What this shows:
- The frozen conditions are 11 to 14 dB milder than `noise_20` on the median, and they differ from each other by about 3 dB. They are not uniformly mild: the worst single recordings (21.4 and 20.7 dB) reach about the `noise_20` level.
- `16 kHz, 32 kbps` is the one setting that matches `noise_20` while keeping most of the spectrum. Its worst recording (10.0 dB) is as harsh as `noise_10`.
- Upsampling to 44.1 kHz before encoding makes the same bitrates about 5 dB harsher and keeps more 7 to 8 kHz energy.
- The 7 to 8 kHz band loses about 70% of its energy at 16 kHz, but it is a negligible share of total energy, which is why the earlier ratio-based measurement showed almost no change. The per-recording range of this column is very wide, so the SNR is the more reliable figure.
- Settings at 8 kHz are bandwidth limits (telephony-style), a different corruption from codec loss. 8 kbps is below any realistic use.
- **The class effect is component-specific, not a genuine-versus-synthetic rule.** Median SNR after removing the level change, frozen conditions at 64 / 128 kbps, and 32 kbps:

| Component | 64 kbps | 128 kbps | 32 kbps |
|---|---|---|---|
| XTTS | 33.8 | 38.4 | 22.6 |
| Melo | 30.7 | 32.8 | 21.1 |
| Genuine | 27.4 | 29.0 | 17.9 |
| VITS | 25.7 | 26.4 | 18.8 |

  XTTS is damaged least and Melo next; genuine and VITS are damaged about equally (the probability that a genuine recording is more damaged than a VITS one is 0.48 to 0.56). Against XTTS it is 0.75 to 0.84, against Melo 0.62 to 0.75. With 8 recordings per component these are indicative, not established. R5 should read any change in detector output per component.
- **Reproducibility.** All 176 pairings of the 16 genuine and XTTS recordings with the 11 settings gave identical results on Windows ffmpeg 9.0.2 and Colab ffmpeg 6.1.1, with identical alignment lags. The frozen conditions' raw SNRs also agree with the Colab R3-S1 table to within 0.0005 dB on all 32 recordings.

This measurement led the project owner to amend the specification to version 2 (`mp3_64` replaced by `mp3_32`); see the next section and the decision log. The measurement above describes version 1's `mp3_64` and `mp3_128` and the candidates, and is unchanged.

## Version 2 acceptance check (10 November 2026)

The round-trip check rejects an MP3 decode whose lag-0 correlation with the input is below 0.9, to catch misalignment. A harsher codec setting could make correct decodes fall below that floor and stop R5-S1, so version 2 was checked on real audio before being relied on. `scripts/r3_s1_v2_acceptance.py` encodes and decodes at 32 and 128 kbps with the specification's own arguments, without applying the check, so the margin is visible. Scope: the 1,000 selection-role genuine and XTTS recordings (all available locally, each verified against the cohort manifest hash; Melo and VITS are not available locally), no model scoring, no audio written. Evidence: `reports/r3/r3_s1_v2_acceptance_local.json` and `_rows.csv`.

| | `mp3_32` (new) | `mp3_128` |
|---|---|---|
| Recordings below the 0.9 correlation floor | 0 of 1,000 | 0 of 1,000 |
| Lag-0 correlation, minimum / 5th percentile | 0.934 / 0.974 | 0.975 / 0.995 |
| Alignment peak at the expected position | 1,000 of 1,000 | 1,000 of 1,000 |
| Length difference after alignment (limit 1,152) | 47 to 612 | 47 to 612 |
| SNR after removing the level change, median [minimum] | **20.7 dB [8.4]** | 35.7 dB [12.8] |
| Median by component | genuine 19.2, XTTS 22.1 | genuine 29.7, XTTS 37.5 |
| Effective bitrate | 32.2 to 34.0 kbps | 128.5 to 135.4 kbps |

What this shows:
- Version 2 meets the stated aim: one MP3 condition at about the strength of `noise_20` (median 20.7 dB over 1,000 recordings; 20.0 dB on the 32-recording Colab sample), while `mp3_128` stays mild.
- The existing check works unchanged: every round trip passes with its alignment peak exactly at the expected position. I had expected harsher decodes might need a different check; the data did not support changing it, so the specification keeps the version 1 check.
- The margin is small (minimum 0.934 against a floor of 0.9). R5-S1 fails loudly, not silently, on any recording that trips it, and lowering the floor would be a specification change with a new hash. The worst genuine recording at 32 kbps (8.4 dB) is harsher than `noise_10`.

## Result of the version 2 Colab validation sample (10 November 2026)

Evidence: `reports/r3/spec_v2/r3_s1_colab_20261110T152638Z.json`, with `r3_s1_corruption_sample.json` and `_variants.csv` beside it. Verified: 37 tests passed on Colab, all 128 variants built, every repeat hash identical, the independent recomputation matched, selection role only, no Chatterbox. Output hashes match; the specification hash in the report equals the code's (`d68c4ed6…`); the corruption module hash equals the local file's. Only `contracts.py` differs from the local copy (a docstring, behaviour identical).

- **Unchanged conditions stayed identical on real audio.** On the same 32 recordings, the PCM hashes of `mp3_128`, `noise_20` and `noise_10` are identical between the version 1 and version 2 runs (96 of 96). `mp3_32` differs from version 1's `mp3_64` for all 32.
- **`mp3_32` behaves as designed.** SNR after removing the level change: median 20.0 dB, range 10.0 to 24.4 (genuine 17.9, Melo 21.1, VITS 18.8, XTTS 22.6). Raw SNR median 19.3 dB. Correlation at least 0.954, effective bitrate 32.2 to 33.6 kbps, length difference 57 to 575 samples. These values agree with the probe's to within 0.0005 dB.
- **Noise conditions** are exactly 10.000 and 20.000 dB on all 64 variants (19 pink, 13 white recordings, none unchecked). The level guard fired on 5 of 128 variants, the same recordings as in version 1 (minimum gain 0.94).
- **Cross-build reproducibility.** The Windows acceptance run (ffmpeg 9.0.2) and this Colab run (6.1.1) give identical decoded audio for `mp3_32` and `mp3_128` on the 16 shared genuine and XTTS recordings (differences below 1e-14 dB).
- **The R2-S2 measurement columns do not capture the codec damage.** `mp3_32` leaves the measured noise floor and high-band ratios essentially unchanged (mean noise-floor change +0.05 dB), while the noise conditions raise the noise floor by 11 to 23 dB. The codec's effect shows in SNR and band energy, not in those columns, so a measured-feature shift is not a reliable sign of MP3 strength.

## Limits and open points

- **`mp3_128` is still mild** (median 33.8 dB on the Colab sample, 35.7 dB over 1,000 recordings). It is kept as the mild codec reference; `mp3_32` carries the codec stress. The plan text naming 64 kbps is superseded by the decision log.
- **Review pending.** Nouman has not reviewed version 2. The acceptance check covered genuine and XTTS only; the four-component Colab run covers Melo and VITS on 8 recordings each.
- **Choice made after looking at audio.** `mp3_32` was chosen after measuring candidates on validation audio, against the rule "one codec condition at about the `noise_20` strength". No model score informed it, but it is still a result-informed choice and is recorded as such.
- **ffmpeg builds.** MP3 output was identical between ffmpeg 6.1.1 and 9.0.2 for all 11 settings on 16 real recordings, and for the synthetic fixture. This is not guaranteed in general. Generate the final R5-S1 copies on one machine, record its build with `scripts/r3_s1_corruption_sample.py spec`, and hash every output.
- **Loud recordings** receive a gain below 1 under the noise conditions (5 of 128 variants in the sample).
- **Sample size.** 32 recordings (8 per component, 20 speakers). Enough to check the builder and rank the settings, not to estimate effects on the full test set or to establish class differences.
- **Not claimed:** anything about detector robustness. Frozen models are only run on corrupted test copies in R5.
