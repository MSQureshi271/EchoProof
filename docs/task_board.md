# EchoProof Phase 0 task board

Latest baseline milestone: P1-14 documentation/evidence handoff recorded on 30 October 2026. See the [Phase 1 checklist](phase1_gate_checklist.md), [handoff](phase1_handoff.md) and [completion review](../reports/phase1/p1_14/review.md). The table below remains the Phase 0 snapshot.

Snapshot: 28 October 2026 · Maintainer D · Source: [implementation plan](../IMPLEMENTATION_PLAN.md)

Status means documentation delivery only unless specified. A = S (data/features); B = M (LR/BiGRU/training); C = U (pretrained/compute); D = N (evaluation/application). S/M/U/N are placeholders; actual names and calendar kickoff remain open in the [decision log](decision_log.md).

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
| P0-09 | Assign role placeholders and record kickoff | D / all | W1 D2 | OPEN-006 | S/M/U/N assigned by role; calendar kickoff remains open |
| P0-10 | Team understanding of G0 knowledge checks | All / D records | W1 D2 | P0-01–09 | G0 closed by user team attestation; see [gate record](phase_0_gate_checklist.md) |

## Next-phase handoff queue

The current Phase 1 baseline technical evidence is recorded: P1-02–08, baseline P1-12/13 and P1-14 documentation. Independent task reviews remain pending in the checklist ledger. The next implementation scope is Phase 2 corpus/provenance/eligibility auditing; research training waits for cleaning/split/feature gates. P1-09–11 and full G1 remain deferred. The queued metadata/demo/validation-support obligations below are not closed by the baseline milestone.

| Work item | Owner / reviewer | Due | Evidence required |
|---|---|---|---|
| Resolve metadata/identity mapping | A / D | End W1 | Sample mapping and provenance review |
| Reserve and profile compute | C / B | End W1 | Hardware, AST probe, numeric budget and persistent storage |
| Define latency hardware | D / C | End W1 | Named benchmark device/settings |
| Validate split and calibration support | A/D / B | End W2 | Group manifests, leakage assertions, precision worksheet |

No queued implementation task is marked performed by this documentation pass. At each review move tasks through pending → active → evidenced → reviewed; record links and actual reviewer/date rather than marking completion from intention.