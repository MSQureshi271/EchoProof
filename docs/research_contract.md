# EchoProof research contract

Version 1.0 · Predeclared 28 October 2026 · Owner D · Reviewers A/B/C

This document fixes the initial experimental policy before model results exist. It implements [Phase 0](../IMPLEMENTATION_PLAN.md) and the [problem statement](problem_statement.md). Actual split manifests, device profiles, and fitted artifacts are later-phase outputs. Changes follow the decision-log procedure below.

## Units, scope, and lineage

Use `genuine=0`, `tts=1`; larger scores indicate TTS. Main metrics are recording-level, after arithmetic-mean aggregation of deterministic chunk probabilities. Segments are never independent evaluation subjects. Independent source groups control splitting and uncertainty estimation.

Group genuine/synthetic relatives, verified common source utterances, exact/confirmed near-duplicates, derivative parents, and source speakers before generating segments or augmentations. Track target voices and cloning references separately. Only claim output-voice/reference independence where verified grouping supports it. If those constraints create giant components, preserve core source/utterance integrity and disclose the narrower claim. Missing identity values remain unknown, not invented or globally grouped.

Raw recordings remain immutable. Each derivative records its parent and transformation. Fit imputation, scaling, learned normalization, and feature selection using the applicable training partition only. Pretrained fixed processor settings are versioned; they are not fitted on EchoProof test data. Group-safe integrity inspection may precede splitting; outcome-driven EDA and quality choices use development data.

## Protocol definitions

| Protocol | Fit/development boundary | Final assessment |
|---|---|---|
| A: known generators | Four verified TTS families with source-group train/validation separation | Held-out source groups containing known families |
| B: strict unseen generator | Initial mapping: MeloTTS and VITS training, XTTS v2 unseen validation; Chatterbox TTS absent from every development source | Genuine and Chatterbox TTS recordings from held-out source groups |
| C: external transfer | No external-test fitting or threshold tuning | Verified English genuine and TTS subset of MLAAD-tiny, subject to eligibility audit |
| D: corruption | Transformation policy chosen on development data | Paired noise/compression versions of final recordings, with frozen models and thresholds |

The generator mapping is the predeclared starting design, conditional on verified metadata. Any correction precedes training and is versioned. Verify aliases and forks across sources. Unknown architecture-family independence supports only an “unseen dataset system ID” claim. External-source shift does not automatically mean unseen-generator shift.

Target approximately 70/15/15 train/validation/test allocation subject to whole-group constraints. Report achieved proportions and class/group supports. Group integrity takes precedence over exact percentages. Within each protocol all four models use the same eligible recording IDs; A and B require separate fitted systems and training-local transformations. Budget at least eight final fits.

Default data roles: PhonemeDF for primary development and MLAAD-tiny for sealed external assessment. Verify English genuine/TTS controls, provenance, and permissions before accepting either benchmark. If two-source training is required, reserve a group-disjoint external development subset before tuning and preserve a sealed external remainder; document this as a protocol change.

## Validation responsibilities

| Role | Permitted use | Forbidden use |
|---|---|---|
| Training | Weights, training transformations, class strategy estimates | Final evaluation |
| Selection validation | Hyperparameters, checkpoints, architecture comparisons | Fitting model weights |
| Calibration validation | Fit recording-level calibration after aggregation | Model/architecture tuning |
| Operating validation | Calibration assessment, operating threshold and optional abstention policy | Weight fitting |
| Final tests | Locked performance, uncertainty, error reporting | Selection, threshold tuning, recalibration |

Allocate disjoint validation groups for selection/calibration/operating roles, initially 50/25/25 of validation subject to support. If inadequate, enlarge development allocation or document nested group-aware cross-fitting before fitting. Never fill the gap using final tests. Phase 4 must include a precision worksheet with independent-group counts, genuine FPR uncertainty, and attainable claims. One final held-out family does not establish universal unseen-generator performance.

## Selection algorithm

1. Select checkpoints/hyperparameters on recording-level selection-validation macro F1 at fixed raw-score threshold 0.5. Use deterministic evaluation windows and mean chunk probability aggregation. Never optimize the threshold per epoch.
2. Rank deployable Protocol B candidates by unseen-generator selection-validation macro F1, then lower EER, lower genuine FPR at 0.5, and lower warm end-to-end p95 latency. For an exact remaining tie, use stable lexicographic candidate ID. Known-generator development results are a reported sanity check, not an undisclosed override.
3. A candidate must pass bundle loading, finite complete prediction, input limits, parity checks, memory feasibility, and the five-second target on the declared ten-second benchmark to meet the deployment qualification target. Name device, precision, threads, warm-up and measurement repetitions before timing. No measured hardware is assumed now.
4. If none qualifies, record the engineering target miss and use the highest-ranked technically functioning B candidate for a clearly labeled research demonstration. Do not claim deployment qualification. If none functions, application completion remains unmet.
5. Do not impose an unsupported accuracy floor. Report quality and uncertainty even if performance is modest. No selection is allowed from final-test tables.

The chosen artifact is model + protocol + seed + config/checkpoint identity, not just a family name. All models, including nonselected models, retain their independently locked operating policies for final comparisons.

## Calibration and decision policy

Fix aggregation first. Fit a low-capacity recording-level calibrator on calibration groups only; evaluate reliability, Brier score and log loss on operating groups. Preserve raw scores. Describe probabilities as conditional on measured evaluation domains; otherwise label output “model score.”

Select the operating threshold by maximizing macro F1 on operating validation. For reproducible search, evaluate all distinct score-induced decision partitions, including all-positive and all-negative possibilities; save threshold representation and equality behavior. Tie-break by lower genuine FPR, then proximity to 0.5, then the smaller threshold. Persist exact semantics, including an explicit all-negative rule if required; do not approximate a threshold above 1 by clamping to 1. Ordinary decisions use `score >= threshold` for TTS.

An FPR-constrained policy is not part of version 1.0; it requires a recorded pre-result amendment and appropriate sample-size evidence. Optional abstention also requires a development-selected rule. Always report full-set binary metrics; additionally report coverage and error among accepted cases. Low confidence is not a general out-of-distribution detector.

## Sealing and change control

Complete and freeze B weights, hyperparameters, processors, aggregation, calibrator, thresholds, quality rules, and production selection before any A training or outcome-guided analysis exposes B's final family. Generic fixture-based engineering may proceed earlier. A findings must not feed back into B tuning.

Each lock records code/config/data/split hashes, checkpoint revisions, trained parameter sets, selected epoch, seed, processor and feature settings, decision policy and dependency environment. Ordinary training commands must not access final-test inputs; sealed evaluation has a logged explicit entry point.

All final metrics follow recording aggregation. EER uses a test ROC sweep as a descriptive statistic; its crossing threshold must never replace the operating threshold. Use independent-group paired bootstrap, not chunk bootstrap. Report common denominators and processing failures; never silently exclude a model's failed recordings.

Changes require a decision-log entry describing rationale, evidence used, affected artifacts, and whether final outcomes were exposed. Before test exposure, issue a new version and refit/relock affected components. After exposure, label further work exploratory and obtain new untouched data for new unbiased final claims. Mechanical repairs still require prediction-impact checks. No automated retraining or hidden fallback model is authorized by this contract.

## Research integrity and stopping conditions

Audit class-correlated source, codec, duration, silence, RMS, sample rate, and recording-channel artifacts using training/development diagnostics. Include a nuisance-only baseline and matched-condition checks. Leakage-safe groups alone do not remove acquisition shortcuts.

Block training on prohibited group leakage or ambiguous task labels; block external claims on missing English genuine/TTS controls; block release on offline/UI score divergence; mark AST incomplete without actual partial backbone adaptation. Reduce optional trials and shared cohort scale before weakening these controls. No significance threshold, performance gain, or advanced-model superiority is required to declare an honest completed study.
