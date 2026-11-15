# R4-U2: pretrained-model and compute report draft

Owner: Muhammad Usman Murtaza. This delivers the page 8 draft and compute contribution to page 9 required by REVISED_PLAN.md, R4-M2 / R4-U2. Peer review and integration with Muneeb's training-procedure section remain pending. Pagination depends on the team's final report format.

Run `python scripts/r4_u2_report.py` from the repository with the saved local evidence available. It executes no model or dataset pipeline. It writes:

- `reports/final/r4_u2_pages_8_9_draft.md`: report text, model/parameter descriptions, adaptation comparison, bounded compute table, limitations and review checklist.
- `reports/final/r4_u2_facts.json`: source SHA-256 identities, selected AST-A result, per-history timing facts and explicit accounting limits.

Both generated files follow the existing ignored reports directory convention; share them through Drive. The generator and this guide can travel in Git. Do not change ignore rules or force-add runtime artifacts for this task.

Git publication also includes lightweight snapshots `docs/r4_u2_pages_8_9_draft.md` and `docs/r4_u2_facts.json`, copied from those generated outputs. Refresh these together after regenerating. Source verification uses the exact committed Git blob if checkout bytes differ, and records the commit-qualified identity; it does not normalize historical source hashes.

Verification checks the AST-A report's linked selection/adaptation/evidence/parity hashes, checkpoint and prediction identities, the bundle components, frozen group checksum equality and recorded change flags. The exact Colab launcher is locally preserved as `scripts/r4_u1_ast (2).py`; the generator verifies it against the run contract rather than substituting the older local launcher. Head parameter counts are derived from the recorded architecture, while AST counts come from the resolved run configuration.

The timing table covers recorded epoch loops, including validation. It separately shows the interrupted A attempt. It excludes unmeasured setup and does not claim complete GPU billing, energy or project cost. Earlier extraction and feasibility reports are available, but no consolidated total should be published without stage-by-stage reconciliation and overlap checks. Their absence does not prevent delivery of an accurately bounded draft.

No run acceptance, calibration promotion, protocol amendment or final test result is established by this report task.
