# Phase 0 — Dated G0 gate checklist

Recorded: 28 October 2026 · Version 1.0 · Gate owner D

**Gate G0: closed by team attestation.** The user confirmed that the team understands every G0 concept. Individual review dates and signatures were not supplied; this records the team's attestation, not separate interviews.

## Deliverable verification

- [x] [Problem statement](problem_statement.md): input/output, labels, supported scope, exclusions, research questions and measurable engineering targets.
- [x] [Research contract](research_contract.md): selection rule, B production artifact, validation roles, operating policy, group boundaries and final-test sealing.
- [x] [Decision log](decision_log.md): recorded policies, unresolved evidence, role owners/reviewers, deadlines, compute controls, review cadence and change procedure.
- [x] [Task board](task_board.md): phase tasks, dependencies, status, evidence and next-phase handoff.
- [x] This dated gate record distinguishes completed documents from pending operational evidence.
- [x] No trained model, acquired dataset, measured latency, statistical significance or team approval is claimed.

## G0 walkthrough: expected answers

| Check | Expected answer |
|---|---|
| Positive class and false positive | TTS=1; genuine speech flagged TTS is a false positive |
| Prediction unit | Recording after mean chunk-probability aggregation |
| Independence unit | Verified source groups/connected components; never overlapping chunks |
| Selection data | Selection validation, especially B unseen-generator development; no final tests |
| Calibration and operating data | Separate groups for score mapping and threshold selection/reliability |
| Final-test responsibilities | A known-family, B held-out-family, C external and D paired corruption; no tuning |
| Information boundary | Freeze B before A exposure to B's final family; no A-to-B tuning feedback |
| Meaning of confidence | Validated recording-level calibration only; otherwise model score; no universal authenticity claim |
| Model requirements | LR, BiGRU, frozen WavLM/head, actual partial AST adaptation; both latter models are Transformers |
| Unmet conditions | Leakage, absent eligible external controls, missing model/adaptation, incomplete scoring or UI parity; disclose target misses |

## Actual review record

| Role | Assigned person | Walkthrough evidence/date | Status |
|---|---|---|---|
| A — Data/features | S (placeholder) | Team understanding confirmed by user; individual date not supplied | Covered by team attestation |
| B — Baseline/recurrent | M (placeholder) | Team understanding confirmed by user; individual date not supplied | Covered by team attestation |
| C — Pretrained/compute | U (placeholder) | Team understanding confirmed by user; individual date not supplied | Covered by team attestation |
| D — Evaluation/app | N (placeholder) | Team understanding confirmed by user; individual date not supplied | Covered by team attestation |

- [x] Role placeholders S, M, U, N recorded. Actual names and calendar kickoff remain to be supplied.
- [x] Team understanding of every G0 concept confirmed by user; individual review record remains unavailable.
- [ ] Any changed policy is versioned in the decision log before affected experiments.

G0 is closed on the user's team attestation. Later-phase metadata, hardware and dataset checks remain tracked obligations.
