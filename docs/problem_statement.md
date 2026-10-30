# EchoProof problem statement

Version 1.0 · 28 October 2026 · Owner D · Reviewers A, B, C

Authority: [implementation plan, Phase 0](../IMPLEMENTATION_PLAN.md). This is the product definition for an eight-week, four-developer research MVP. No performance or dataset verification is claimed here.

## Problem and objective

Text-to-speech systems can produce recordings that resemble genuine human speech. EchoProof will investigate whether acoustic classifiers can distinguish verified genuine English speech from text-conditioned synthetic speech, particularly when the synthesis generator differs from those used for training.

Deliver a reproducible comparison of four complete detection approaches and an application that makes their selected result usable. Success requires defensible experiments and a working application, including when generalization is weak or the baseline wins.

## Users and workflow

The initial users are the project team, instructors, and demonstration participants assessing consented or licensed recordings. A user uploads a supported file, reviews its duration and visualizations, requests analysis, and receives a recording-level assessment with a clearly described score and downloadable summary.

Input: a recording of English speech, initially WAV or MP3 through a tested decoder. Initial limits are 25 MiB and 60 decoded seconds. Browser recording follows working upload support. Validate decoded content instead of trusting file extensions. Reject overlong audio rather than silently analyzing a prefix.

Training labels are `0 = genuine` and `1 = tts`. TTS includes documented text-based voice cloning. Voice conversion is a different task and is excluded. The unit of prediction is the original recording; four-second windows are internal computation units.

Output: “Likely genuine speech” or “Likely TTS-generated speech,” a TTS-oriented model score, model version, quality warnings, analyzed coverage, waveform, log-mel spectrogram, and JSON/text summary. Report calibrated probability only after recording-level calibration has been evaluated. An optional inconclusive state requires a validation-selected policy. Invalid or insufficient audio receives no invented score.

## Boundaries

Validated scope is genuine versus TTS English speech. Singing, voice conversion, edited or mixed genuine/synthetic recordings, other languages, replay-only attacks, streaming detection, and synthetic-region localization are outside the initial claim. Unsupported content is not necessarily automatically detectable.

An assessment does not establish speaker identity, statement truth, legal authenticity, or universal resistance to new synthesis systems. Waveforms and spectrograms describe the signal; they do not prove the classifier's reasoning. Benchmark calibration does not establish confidence for every uploaded domain.

## Research questions

| ID | Question | Comparison and evidence |
|---|---|---|
| RQ1 | Do temporal acoustic representations help? | LR handcrafted summaries versus two-layer BiGRU sequences on identical within-protocol recordings |
| RQ2 | Do pretrained speech representations transfer? | Frozen WavLM/head versus BiGRU, especially unseen-family validation and final evaluation |
| RQ3 | Does task adaptation justify its cost? | Frozen AST warm-up versus partial AST adaptation; compare adapted AST with frozen WavLM as complete approaches |
| RQ4 | How stable are results under distribution shift? | Locked known/unseen-generator, external-source, noise, and compression results, including genuine false positives |

Measure recording-level accuracy, TTS precision/recall, macro F1, ROC-AUC, EER, confusion matrix, genuine FPR, and latency. Include calibration, independent-group uncertainty, training/extraction time, parameters, and memory. WavLM and AST are both Transformers; their comparison does not isolate architecture alone. More complexity is not assumed to improve performance.

## Product acceptance and engineering targets

| Scenario | Observable acceptance condition |
|---|---|
| Valid WAV/MP3 | Actual inference, complete declared coverage, finite score, correct model version, plots and export |
| Genuine or TTS recording | Honest likely-class wording; no requirement that every example be classified correctly |
| Corrupt/empty file | Clear validation error and no score |
| Silent/insufficient speech | Quality/insufficiency response; proposed minimum usable speech is one second, frozen after development validation |
| Stereo or alternate sample rate | Documented mono conversion and resampling to 16 kHz, consistent with offline inference |
| Oversize/overlong input | Clear rejection before excessive resource use; no silent truncation |
| Unavailable model or timeout | Distinct failure state, no mock result, temporary-file cleanup |
| CLI versus UI | Same recording and bundle produce equivalent scores within documented numerical tolerance |
| Private upload | Transient processing, no raw audio in logs or shared result caches; hosted transfer notice |

The warm end-to-end p95 target is at most five seconds for a ten-second clip on hardware named before benchmarking. Include decoding, feature extraction, model inference, and postprocessing. This is a target, not a measured SLA. No arbitrary minimum accuracy or guaranteed FPR is imposed. If the target is missed, record the miss; the project may still demonstrate its research outcome but cannot claim the target was met.

## Completion definition

The MVP requires four genuinely trained approaches, including actual AST backbone adaptation, shared leakage-safe protocols, two documented data sources, reproducible EDA/features/training, sealed final evaluation, a versioned inference bundle, working upload/visualization/export flow, and the report/demo package. The guide calls for at least ten substantive report pages; plan for twelve. Missing required experiments remain unmet requirements.

See [research contract](research_contract.md) for selection and data boundaries, [decision log](decision_log.md) for unresolved evidence, and [task board](task_board.md) for owners and deadlines.
