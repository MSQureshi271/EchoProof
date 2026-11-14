# R3-N2: local latency of the calibrated LR, BiGRU and WavLM bundles

Status: measured on the local development machine on 14 November 2026. These are the figures the owner chose to use. The Colab latency reports by Nouman (`reports/r3/r3_n2_{lr,bigru}_latency_*`) were run on a different device and are not used here. AST is excluded (no calibrated AST bundle exists). The result is a development measurement, not the GR3 deployment qualification, and the demo device is not named (see "Open").

Harness: `scripts/r3_n2_latency_local.py` (summary tables: `scripts/r3_n2_latency_local_summary.py`). It follows the pattern of Nouman's notebooks `notebooks/R3_N2_{LR,BiGRU}_Latency.ipynb`.

## Method

- **Path measured.** The production `analyze()` call, batch 1 (one recording per call), from audio bytes to decision, with the exact bundles below. Stage times come from wrapping the production callables for the length of one call (decode, chunking, feature extraction or encoder, model prediction, aggregation, calibration, decision) and restoring them afterwards. A derived `other` stage is the total minus the listed ones.
- **Parity.** The first `analyze` of each clip is the reference. Every timed call must return the same score (tolerance 1e-10) and decision.
- **Clips.** The plan's 1, 4, 10, 30 and 60 s. Each is the first N seconds of one in-memory concatenation of six explicit genuine fixtures (`dataset/genuine/wav/`: 19-198-0002, 19-198-0001, 19-198-0003, 911-128684-0002, 89-218-0004, 19-198-0037), cut to length, 16 kHz mono PCM16, nothing written to disk. The fixtures serve timing only, so the scores they produce say nothing about accuracy.
- **Repetitions.** LR and BiGRU: 5 warm-up calls, 100 timed calls and 10 fresh processes per clip. WavLM: 3 warm-up calls, 30 timed calls (1, 4, 10 s), 20 (30 s) and 10 (60 s), and 5 fresh processes. A 60 s WavLM call takes 13 s, so p95 for WavLM at 30 and 60 s rests on few calls.
- **Cold interpreter.** Each fresh process loads the bundle and makes one prediction. This is a cold Python process, not a cold machine, a cold disk cache or a cold virtual machine.
- **Device.** Intel Core i5-10310U (4 cores, 8 logical processors, 1.7 GHz base; Windows reports a 2.2 GHz maximum), 15.8 GB RAM, Windows 11, on AC power, Balanced plan, CPU only. Python 3.14.4, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, librosa 1.0.0, torch 2.15.0+cpu (4 threads, fp32), transformers 5.22.0.

## Bundles

| Model | Bundle | `bundle.json` sha256 (first 16) | Score type |
|---|---|---|---|
| LR | `runs/r3/r3_n1_lr/selected_lr_v1/handoff/bundle` | `421bcb5f4add7076` | calibrated probability |
| BiGRU | `runs/r3/r3_n1_bigru/selected_bigru_v1/handoff/bundle` | `6627f6f95486c61e` | calibrated probability (review status pending; loaded with the explicit override, which approves nothing) |
| WavLM | `bundles/r3_n1_wavlm_head_B_calibrated` | `491310e8e563c4cc` | calibrated probability |

## Results: warm end-to-end latency (p50 / p95)

| Clip | Chunks | LR | BiGRU | WavLM |
|---|---:|---|---|---|
| 1 s | 1 | 9.8 / 10.1 ms | 17.7 / 24.8 ms | 245 / 284 ms |
| 4 s | 1 | 28.0 / 29.8 ms | 68.9 / 91.6 ms | 674 / 709 ms |
| **10 s** | 4 | **109 / 117 ms** | **284 / 371 ms** | **2.11 / 2.41 s** |
| 30 s | 14 | 383 / 416 ms | 1.06 / 1.28 s | 6.40 / 6.77 s |
| 60 s | 29 | 818 / 888 ms | 2.22 / 2.43 s | 13.28 / 13.48 s |
| Timed calls per clip | | 100 | 100 | 30, 30, 30, 20, 10 |

- **Against the contract's five-second target on the ten-second benchmark:** all three are under it warm on this machine. Highest p95 is WavLM at 2.41 s, which has about half the budget to spare. This is not the formal qualification (the demo device is not named and the contract's declared repetitions apply).
- **Cost scales with the number of 4 s chunks, not the clip length:** a 10 s clip is 4 chunks, 30 s is 14 and 60 s is 29 (stride 2 s).
- **Parity:** exact in every call, zero decision flips, for all three models and all five clips.

## Where the time goes (p50 share of the total)

| Model | Dominant stage | Share | Everything after the model (aggregation, calibration, decision) |
|---|---|---:|---|
| LR | feature extraction (handcrafted features, librosa) | 94 to 99% | under 0.1 ms |
| BiGRU | recurrent prediction | 89 to 97% (feature extraction 2 to 6%) | under 0.1 ms |
| WavLM | encoder embedding | 99.1 to 99.9% (the head takes under 1 ms) | under 0.2 ms |

Decode costs 0.3 to 14 ms across all models and clips. Calibration and the threshold decision are negligible for all three: the calibrated bundles add nothing measurable to latency.

## Cold interpreter (p50, fresh subprocess, process start to first result)

| Model | Bundle load | First prediction (1 s / 10 s / 60 s clip) | Process start to first result (10 s clip) |
|---|---:|---|---:|
| LR | 2.7 s | 1.65 / 1.79 / 2.48 s | 5.4 s |
| BiGRU | 4.9 s | 0.02 / 0.25 / 1.80 s | 6.2 s |
| WavLM | 13.6 to 18.8 s | 0.13 / 1.94 / 12.48 s | 17.9 s |

- **First-use cost is dominated by loading, not scoring.** LR's first prediction is slower than its warm calls (1.65 s against 9.8 ms for a 1 s clip) because libraries import and compile on first use.
- **If the five-second target is read as including a cold start, all three miss it** on a 10 s clip (LR 5.4 s, BiGRU 6.2 s, WavLM 17.9 s). It is met warm. The application should therefore load the model once at start-up and not per request. The contract's wording does not settle which reading applies; that is for the team.
- **Cold figures vary more than warm ones** (WavLM bundle load ranged from 13.6 to 18.8 s between clips within one run, and 33 s in a first, superseded run), because they depend on the disk cache and what else the machine is doing.

## Reliability of the numbers on this machine

- **The first runs were slower, by up to 2x.** The first LR, BiGRU and WavLM runs were slower than later runs on the same code: BiGRU 10 s 629 ms against 284 ms, LR 60 s 1.80 s against 0.82 s, WavLM 10 s 4.45 s against 2.11 s. Short WavLM clips were affected, 30 s and 60 s much less (6.89 against 6.40 s, 14.19 against 13.28 s). Scores and decisions were identical; only time differed.
- **Cause not established.** The machine was meant to be idle. Google Drive's sync client was running (the harness writes into the synced project folder), and I did not run a controlled test. The per-group CPU clock and load snapshots recorded in the final LR and BiGRU runs show an idle machine (load 1 to 7%) at group boundaries, which does not rule out brief spikes in between.
- **What was done about it.** Both sets were kept. The superseded reports stay in `reports/r3/` untouched: `r3_n2_local_lr_latency_20261114T122201Z.json`, `r3_n2_local_bigru_latency_20261114T130410Z.json`, `r3_n2_local_wavlm_latency_20261114T123817Z.json`, and the mixed summary `r3_n2_local_latency_summary.md`. The figures above come from `r3_n2_local_lr_latency_20261114T135646Z.json`, `r3_n2_local_bigru_latency_20261114T140417Z.json`, `r3_n2_local_wavlm_latency_20261114T133632Z.json` and the summary `r3_n2_local_latency_summary_final.md`. The final LR and BiGRU runs agree with independent idle spot checks (10 s and 60 s, 30 calls, `reports/r3/r3_n2_local_spotchecks/`) to within 4 to 8% (LR 10 s: 109 against 113 ms; 60 s: 818 against 821 ms; BiGRU 10 s: 284 against 263 ms; 60 s: 2,221 against 2,106 ms).
- **How to read them.** Treat the figures as accurate to roughly ±10% on this laptop when idle, and up to 2x worse when it is not. Anyone reproducing them should keep the machine idle (pausing Drive sync), run twice and compare.

## Open

- **Device naming (OPEN-004).** On 14 November 2026 the project owner named this machine "Subhan's HP" (HP EliteBook 840 G7 Notebook PC, user-reported) as the demo device, and reports that the team reviewed and accepted the name (user-reported; there is no team record in the repository). It was named after the timing runs, not before, which the contract asks for. The original wording follows: the demo device was not named when these runs were made. The contract asks to name device, precision, threads, warm-up and repetitions before timing. This record gives them for the machine measured; whether this machine is the demo device is a team decision, and the Colab and local figures are on different hardware.
- **Formal qualification, the choice of cold or warm reading, and the AST measurement** are outside this record.
- **BiGRU review status is pending** in its bundle (Nouman's record); the benchmark does not approve it.
- **No GPU figures.** CPU only.

## Files

| What | Path | In git |
|---|---|---|
| Harness | `scripts/r3_n2_latency_local.py` | yes |
| Summary tables | `scripts/r3_n2_latency_local_summary.py` | yes |
| Final reports (raw samples included) | `reports/r3/r3_n2_local_{lr,bigru,wavlm}_latency_*.json` (the three named above) | no (`/reports/` is ignored; travels by Drive) |
| Final summary | `reports/r3/r3_n2_local_latency_summary_final.md` | no |
| Superseded first runs and spot checks | `reports/r3/` (named above), `reports/r3/r3_n2_local_spotchecks/` | no |
