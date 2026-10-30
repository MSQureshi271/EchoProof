# EchoProof P1-02 Colab environment review

Reviewed 29 October 2026 from [Colab hardware evidence](../reports/phase1/p1_02/20261028T190956Z-8525fb2f/hardware.json), [Colab summary](../reports/phase1/p1_02/20261028T190956Z-8525fb2f/environment.md), and the [restart record](../reports/phase1/p1_02/20261028T190956Z-8525fb2f/restart_verification-99dd0717.json). This file documents analysis of the supplied run. It is not another runtime measurement or an independent M review.

## Working Colab runtime

| Field | Observed result | Interpretation |
|---|---|---|
| Python | 3.13.16, `/usr/bin/python3` | Use this kernel for the first Colab environment; P1-03 still checks package compatibility |
| PyTorch | 2.11.0+cu130, CUDA available | Tiny tensor forward/backward check succeeded on CUDA; retain the working install for now |
| GPU | Tesla T4, 15,360 MiB; driver 580.82.07 | Actual assigned device for this run; future Colab allocations may differ |
| CPU and RAM | Intel Xeon 2.00 GHz, 2 logical CPUs, 13,605,818,368 bytes RAM | Record resource limits; this does not establish model training throughput |
| Temporary disk | 75,237,453,824 bytes free under `/content/echoproof` | Temporary runtime capacity, not Google Drive account quota |
| FFmpeg | 6.1.1, present | Decoder is available for later format checks |
| Baseline packages | All packages on the notebook's baseline availability list were present | Import/version inventory only; P1-03 must validate project imports and locking |
| Package consistency | `pip check` returned 1: `ipython 7.34.0 requires jedi, which is not installed` | Resolve in P1-03 and recheck; this does not show a PyTorch failure |

Persistent project path in the run: `/content/drive/MyDrive/PAI&AIES Project`. Scratch path: `/content/echoproof`. The report's small marker passed a separate Python-process readback. The subsequent fresh-runtime record says `all_saved_hashes_match: true` with operator confirmation. This supports persistence of the **report artifacts** across a remount; model checkpoint recovery is a later task. Drive free quota, planned work hours, session lifetime, training throughput, and demo device were not supplied.

## Dataset export in progress

The Colab report counted 28,539 genuine WAVs totaling 11,589,325,052 bytes, 17,492 transcript TXT files totaling 3,273,350 bytes, and **no** `dataset/tts/XTTSv2` directory at the selected Drive root. It found zero complete triples and correctly returned `needs_attention`.

A separate read-only inventory of the local workspace found 28,539 genuine WAVs, 28,539 transcript files totaling 5,326,896 bytes, and 28,539 XTTSv2 WAVs totaling 9,561,750,812 bytes. The user confirmed that the Drive export was still running when the Colab inventory was captured. The Drive counts are therefore an interim transfer snapshot. At that time 11,047 transcript filenames and the XTTSv2 directory were not yet visible at the selected Drive root. Local counts do not prove transcripts and recordings match in content.

**Transfer handoff:** after export completes, rerun the notebook or the Phase 2 inventory and compare counts, byte totals and paired stems. Preserve the original `hardware.json` as an accurate snapshot of the earlier run. Source labels, licenses, transcript content and speaker identities still require Phase 2 auditing. Paired training starts only after the complete copy has passed the audit and group-safe split checks.

## Decision status

### Updated planning inputs for P1-12

The user later reported that both genuine and XTTSv2 uploads are complete; Phase 2 still verifies the finished copy. The selected candidate pilot is 1,000 pairs / 2,000 recordings with one four-second crop each. The user allocated two expected three-hour sessions and most recently reported 10.13 GB free Drive capacity. These are user-supplied planning inputs rather than measured session guarantees or a new Drive inventory. Apply the 25% reserve and recheck the actual runtime and free space during P1-12.

**P1-02 runtime and storage review: accepted for proceeding to P1-03.** The observed Colab kernel, CUDA device, temporary space, persistent project path and fresh-runtime report recovery establish the environment needed for package setup. The original notebook status `needs_attention` accurately describes its in-progress dataset snapshot and `pip check` result; it is not being rewritten. The missing `jedi` dependency belongs to P1-03. Drive account quota and numeric training budget belong to P1-12; the demo device belongs to later candidate qualification. The user accepted the Colab-first progression. Independent U/M reviewer sign-off has not been recorded. Full G1 and baseline training remain separate gates.

### P1-12 observed baseline capacity

Subsequent baseline reproduction is evidenced in the [P1-13 review](p1_13_review.md): the selected exact lock is `locks/colab-baseline-20261029T192113Z-93db044e/requirements-lock.txt`, with SHA-256 `d60c8db107fe2a94bc6ec93c493db516ab0401a3465fce38115f0738c6179c8b`. The fresh Colab run retained the matching Tesla T4/PyTorch/CUDA/driver/decoder state and passed 41 tests plus CPU/CUDA probes. Independent review and the baseline P1-14 handoff remain pending.

The 29 October Colab [benchmark](../reports/phase1/benchmark.json) records a Tesla T4, FP32 and batch size four. Its [budget](../runs/phase1/20261029T183114Z-bef3f3b89096/budget.json) estimates 0.3011 hours and 2.238 GB additional storage against the user-reported six-hour allocation and 10.13 GB free Drive snapshot. With the 25% reserve applied, all capacity checks pass provisionally. These fixture measurements qualify proceeding to P1-13; audited-data throughput, fresh-runtime reproduction and independent review remain pending. Reprofile if the assigned hardware changes.

## ChatterboxTTS Drive upload update

The user reports successful upload of an additional TTS dataset in `dataset/tts/ChatterboxTTS/`. This is a user-reported transfer update; counts, completeness, generator provenance, permissions, transcript/stem matching, speaker relationships and audio quality remain unverified. Phase 2 should inventory and audit it alongside XTTSv2 and genuine inputs. The existing 1,000-pair XTTSv2 baseline budget and historical reports remain unchanged; recheck current Drive free space and budget before expanding the cohort. The second reported generator is a candidate for unseen-generator evaluation, subject to audit and protocol feasibility.
