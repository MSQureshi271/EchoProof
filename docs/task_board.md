# EchoProof Phase 0 task board

Snapshot: 28 October 2026 · Maintainer D · Source: [implementation plan](../IMPLEMENTATION_PLAN.md)

Status means documentation delivery only unless specified. A = data/features; B = LR/BiGRU/training; C = pretrained models/compute; D = evaluation/application. Actual individual assignments remain OPEN-006 in the [decision log](decision_log.md).

| ID | Work item | Owner / reviewer | Due | Dependency | Status / evidence |
|---|---|---|---|---|---|
| P0-01 | Define problem, labels, scope, exclusions, user outputs | D / A | W1 D1 | Plan | Done: [problem statement](problem_statement.md) |
| P0-02 | Predeclare four research questions and metric units | D / B/C | W1 D1 | P0-01 | Done: problem statement and research contract |
| P0-03 | Define group integrity and A/B/C/D boundaries | A / D | W1 D1 | P0-01 | Done: [research contract](research_contract.md) |
| P0-04 | Fix selection ranking and deployable artifact identity | D / B | W1 D2 | P0-02/03 | Done: research contract selection algorithm |
| P0-05 | Separate selection, calibration, operating and final data | D / A | W1 D2 | P0-03 | Done: research contract validation roles |
| P0-06 | Freeze threshold policy and test-access/change rules | D / C | W1 D2 | P0-04/05 | Done: research contract |
| P0-07 | Record unknowns, owners, deadlines and defaults | D / all | W1 D2 | P0-01–06 | Done: [decision log](decision_log.md) |
| P0-08 | Set resource controls, reviews and completion gates | C/D / B | W1 D2 | P0-07 | Done: decision log and gate checklist |
| P0-09 | Confirm actual people and kickoff | D / all | W1 D2 | OPEN-006 | Pending actual assignment |
| P0-10 | Team walkthrough of G0 knowledge checks | All / D records | W1 D2 | P0-01–09 | Pending actual review; documentation is ready |

## Next-phase handoff queue

| Work item | Owner / reviewer | Due | Evidence required |
|---|---|---|---|
| Resolve metadata/identity mapping | A / D | End W1 | Sample mapping and provenance review |
| Reserve and profile compute | C / B | End W1 | Hardware, AST probe, numeric budget and persistent storage |
| Define latency hardware | D / C | End W1 | Named benchmark device/settings |
| Clarify two-source course interpretation | D / A | End W1 | Recorded interpretation; retain declared default meanwhile |
| Verify external English benchmark | A/D / B | End W2 | Both classes, method/language metadata, permissions |
| Validate split and calibration support | A/D / B | End W2 | Group manifests, leakage assertions, precision worksheet |

No queued implementation task is marked performed by this documentation pass. At each review move tasks through pending → active → evidenced → reviewed; record links and actual reviewer/date rather than marking completion from intention.
