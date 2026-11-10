# R2-U2 Option B adoption and R3-U2 handoff

Recorded 11 November 2026. Owner: Muhammad Usman Murtaza (Usman); plan reviewer:
Muneeb. Authority: `REVISED_PLAN.md`, Section 6, R2-U2 and R3-U2.

## Official file placement

The accepted deliverables are physically present in the normal project folders:

| Deliverable | Official location |
|---|---|
| Numerical Trainer and helpers | `src/echoproof/training/r2_u2_*.py` |
| Training and verification CLI | `scripts/r2_u2_ast_warmup.py`, `scripts/r2_u2_verify.py` |
| Frozen training recipe | `configs/r2_u2_ast_b.json` |
| Colab workflows | `notebooks/R2_U2_Colab_AST_Warmup.ipynb`, `notebooks/R2_U2_Colab_Software_Tests.ipynb` |
| Software tests | `tests/test_r2_u2_numerical.py`, `tests/test_r2_u2_official.py` |
| Accepted checkpoints, predictions, history, update log, baseline and run receipts | `runs/r2/option_b_1b3757e898da4baaad2bde49fa3cea01/` |
| Acceptance records and matched successful Colab evidence | `reports/r2/r2_u2_option_b/` |
| Official output inventory and per-file SHA-256 | `reports/r2/r2_u2_option_b_acceptance.json` |

For R3, the official initializer is
`runs/r2/option_b_1b3757e898da4baaad2bde49fa3cea01/best.pt`, SHA-256
`6a3883e59fc48ec18697fcdfdb59641adbd7dee6e965c1f80ad6f5287ea0bd57`.
The original `experiments/r2_u2/` tree is preserved in full. Its accepted run and
24 official output/report copies were checked byte-for-byte by SHA-256; the
placement inventory records every source and destination. Failed experiments
remain in that preserved tree and are not represented as accepted outputs.

All copied JSON, checkpoint and ledger fields retain their original values,
including historical experiment paths and execution-time pending-review flags.
The separate adoption decision establishes acceptance; file placement does not
rewrite that evidence. The matched selection predictions live alongside the
official initializer. `runs/` and `reports/` remain Git-ignored runtime storage:
copy these official folders to the same relative locations in the Drive project
without replacing conflicting existing files or the team's main ledger.

The historical verifier still checks its original experiment paths. Its archived
manifest is not silently redirected. For the official copies, use the separate
placement inventory's hashes; recipient model-load and access checks remain
required. No original experiment file was moved, edited or deleted.

## Official implementation and frozen provenance

The official package contains the actual accepted numerical Trainer in
`src/echoproof/training/r2_u2_numerical_trainer.py`, scope and state/prediction
checks in `r2_u2_scope.py`, and prospective numerical checks in
`r2_u2_acceptance.py`. These are checked copies of the frozen experiment modules;
only the Trainer helper import is adapted to the package. The small
`r2_u2_contract.py` adapter reuses the frozen identity/config helpers and rejects
any copy divergence before emitting provenance. The copies are not independently
maintained variants: new numerical changes require new experiment evidence and
an explicit synchronization. Shared Trainer and all R3 code remain unchanged.

`configs/r2_u2_ast_b.json` is the actual frozen recipe, byte-identical to the
accepted experiment recipe. It replaces the former reference-only JSON index.
The original eight numerical regression cases are available at
`tests/test_r2_u2_numerical.py`, adapted only for package imports/project root;
`tests/test_r2_u2_official.py` checks copy equivalence and rejection of drift.
The official numerical tests require the prepared ML environment; the archived
17-test receipt remains evidence for the frozen experiment, not a claim that a
new local ML test run occurred.

The official notebooks are `notebooks/R2_U2_Colab_AST_Warmup.ipynb` and
`notebooks/R2_U2_Colab_Software_Tests.ipynb`. They reuse the complete accepted
Colab workflows, with saved execution output removed from the official copies.
Their code cells still execute the hash-bound experiment paths deliberately.
The incomplete standalone verification notebook has been removed. The two
small CLI boundaries `scripts/r2_u2_ast_warmup.py` and `scripts/r2_u2_verify.py`
remain: they execute the accepted launch/verification implementation at its
original path, preserving its ROOT calculations and source-bound software gate.
They contain no alternative training algorithm. Fresh research training remains
subject to the existing explicit authorization and source/runtime gates.

R3 consumers may import the official NumericalTrainer for the **unchanged warm-up
contract only**. It is not the R3 adaptation Trainer: R3 requires a new optimizer,
trainable groups and its own scheduler/provenance contract. The exact accepted
initializer and model-weights-only recipient procedure below remain authoritative.
No new AST training or inference is required to publish these copies.

The original acceptance manifest binds the pre-integration version of this guide.
It is an immutable historical evidence snapshot; its document checksum is not
updated to disguise subsequent integration edits. Use original commit 0e4c0d5
when checking that historical document identity. Experiment sources, artifacts
and model identities are unchanged; the corrective commit records the current
integration. The original experiment tree remains the immutable reference.

## Decision and precise status

Usman has authorized repository-side adoption of
`option_b_1b3757e898da4baaad2bde49fa3cea01`, reporting that the team, including
Muneeb, approved the candidate. The exact statement and its evidentiary limits
are recorded in [decision.json](../experiments/r2_u2/acceptance/decision.json).
This is **accepted on the user's reported team approval**; no independent signed
Muneeb message, review URL or approval date has been supplied. The recording date
is not an invented date of the team's meeting or approval.

Repository adoption is documented. **Recipient handoff is not yet verified:**
separate team sign-off evidence, recipient access to the durable artifacts, and
the model-load/optimizer compatibility check remain outstanding. This does not
authorize new GPU training or claim completion of R3-U2, GR3, calibration or final
evaluation. The original failed run remains unaccepted.

The authoritative artifact identity and evidence index is
[manifest.json](../experiments/r2_u2/acceptance/manifest.json). The experiment's
original `research_accepted=false` and pending-review status remain correct
historical observations. This additive decision supersedes them only for this
candidate's subsequent adoption; it does not edit them or the prospective policy.

## Exact initializer and matched baseline

- Run: `experiments/r2_u2/runs/option_b_1b3757e898da4baaad2bde49fa3cea01/`.
- Initializer: **best.pt**, selected epoch 2; SHA-256
  `6a3883e59fc48ec18697fcdfdb59641adbd7dee6e965c1f80ad6f5287ea0bd57`.
- Selection: 2,000 recordings, 10,029 chunks; mean chunk probability aggregation,
  `genuine=0`, `tts=1`, fixed threshold 0.5. Macro F1 0.791368914146525,
  accuracy 0.8245, EER 0.17, genuine FPR 0.148.
- Keep `predictions_chunks.parquet` and `predictions_recordings.parquet` from this
  exact run as the frozen comparison. Their hashes, split/config identities and
  input hashes are in the manifest. R3 comparisons must match recording IDs,
  labels, chunk identities and evaluation rules, not merely sample counts.
- This is selection evidence, not independent test performance or a production
  model decision. Calibration/operating/test roles cannot be repurposed for tuning.

The run's per-run ledger is original evidence. Its single row is mirrored
verbatim into `reports/ledger.csv`, following the project's selected-model
convention. Bind the specific row, not the entire mutable team ledger. A
`completed` row establishes completed training, not acceptance authority; this
decision and manifest supply the latter. No teammate rows are rewritten.

## Engineering causes, fixes and scope

| Issue | Cause and actual isolated change | Verification and limits |
|---|---|---|
| Infinite epoch gradient norm | Shared Trainer averaged pre-clipping norms across attempts, including true FP16 non-finite events. `numerical_trainer.py` averages only finite applied-update norms and records skips separately. | Matching 17-test Colab suite, update logs and two finite epoch means. Overflows were not removed. |
| Incorrect general skip inference | Shared Trainer inferred skips from norm finiteness, which cannot distinguish actual GradScaler skips from norm-only reduction overflow. Extension observes optimizer post-step calls, scaled/unscaled element finiteness and scaler transitions. | Real AMP skip, norm-only overflow, unsafe/non-AMP rejection and accounting fixtures; actual 568 attempts, 566 applied, two skips. |
| Rejected-gradient handling | After unscale has detected nonfinite gradients, extension lets GradScaler skip without clipping those rejected gradients. Finite applied gradients retain original clipping. | Finite-path equivalence and final partial-group tests. Denominator remains 2 for each final group, never 16. |
| Launcher lifecycle failure | Launcher wrote contract/baseline before a constructor requiring an empty fresh directory. Launcher now initializes Trainer first, writes metadata exclusively before fit, and records initialization failures. | Three lifecycle regressions: empty initialization/metadata before fit; initialization-failure report; existing run rejection without overwrites. Earlier failed launcher artifacts remain untouched. |
| Misleading completion status | Shared Trainer's ledger records training completion before external verification/adoption. Experiment separately checks outputs and forbids automatic promotion. | This adoption record explicitly separates completion, experimental checks, acceptance and recipient readiness. Shared status semantics were not redesigned. |
| Stale launch software flag | Notebook initializes `software_gate_passed=false` and never updates that field in its launch report. | The hash-linked actual software report passed 17/17, zero skipped/failed/errors; launcher checks report, sources, log and JUnit. Disclose the stale field; never rewrite historical JSON. Future notebook cleanup is separate work. |

Affected implementations are `experiments/r2_u2/scripts/numerical_trainer.py` and
`launch.py`, with regression tests in `tests/test_numerical.py`,
`test_launch.py` and `test_contract.py` under that experiment. Exact executed
bytes are bound by the run contract and source manifest. Tests comprise eight
numerical tests, seven launch/contract-policy tests and two identity tests.

**These fixes do not modify shared `src/echoproof/training/trainer.py`.** Its
norm aggregation, norm-based skip count, attempted-update scheduler behavior,
non-strict historical metric serialization and training-completion ledger
semantics remain shared-infrastructure concerns. They are not automatically fixed
for BiGRU, other models or future R3 runners. The extension also deliberately
retains the legacy attempted-update clock for nonconstant schedules. The software
suite's deliberate skipped-step/cosine fixture emits a scheduler-order warning;
the actual R2-U2 constant schedule does not advance a scheduler. R3's planned
warm-up/cosine schedule needs its own explicit semantics and tests; do not silently
reuse the R2 constant-schedule acceptance contract.

Shared Trainer and existing R3 sources, numerical precision, initial model,
optimizer, loss scaling defaults, clipping, batches, seed, data, crop mathematics,
split assignments and historical hashes remain unchanged. No checkpoint is
rewritten, and no old source-hash gate is bypassed. The newly added acceptance
tools live outside the experiment's source-discovery globs, preserving its
verified source manifest and 17-test receipt.

## Numerical and historical interpretation

Both final two-recording groups overflowed (attempts 284 and 568). Neither caused
an optimizer update. Scale transitions were 65536 to 32768 to 16384. Successful
update mean norms were 4.051598231278544 and 3.32315197239495. There is no subsequent
successful update proving stability at scale 16384. The prospective contract
explicitly permits that disclosed outcome without an arbitrary skip threshold.

Historical diagnostics located scale-sensitive FP16 backward matrix-multiply
overflow for the final group. The first historical event's precise operation was
not independently localized. The new logs establish both skips, not a universal
claim that all possible nonfinite gradients have the same cause.

[historical_comparison.json](../experiments/r2_u2/acceptance/historical_comparison.json)
records separately computed full-file hashes and raw tensor-storage comparisons.
Serialized checkpoint hashes differ because checkpoints also contain provenance,
telemetry and other state. Matching raw storage blobs must not be described as
matching whole checkpoint files or a new independent tensor deserialization.
Both saved prediction files and epoch losses/validation metrics match the
historical run exactly. Option B improved numerical evidence and reporting, not
F1 or the historical artifact's acceptance status.

The Colab execution checked finite model/optimizer tensors, AdamW counters,
nonnegative second moments, frozen-backbone fingerprints, changed classifier,
PCM coverage and exact best-checkpoint reload parity. Repository checks verify
their source/artifact bindings; they do not claim to have rerun that inference.
Muneeb's observations about speakers 1578, 302 and 7190 remain deferred evaluation
questions, not established causes or grounds for split changes.

## Evidence and Council provenance

- [Repository verification](../experiments/r2_u2/acceptance/repository_verification.json):
  artifact/source/input hashes, historical comparison and actual local check status.
- [Council record](../experiments/r2_u2/acceptance/council_review.md): retrospective
  summary of the five independent advisor/five anonymized peer reviews performed
  in this conversation. No new Council or teammate review is fabricated.
- Software: `experiments/r2_u2/reports/software_70fc2d647510481d93297f01ba76665c/`.
- Launch: `experiments/r2_u2/reports/launch_a69b822ab05d4f65b8c4ce706abef3a0/`.
- Historical failed archive: `D:/EchoProof/EchoProof-private/R2_U2_Unaccepted`.

## Durable delivery and recipient procedure

The observed producer-side locator is Account B Drive:
`/content/drive/MyDrive/Colab_R2_U1_Workspace/experiments/r2_u2/runs/option_b_1b3757e898da4baaad2bde49fa3cea01/`.
This identifies the producer's location; **it is not a verified shared URL or
evidence that teammates can access it**. Obtain an exact durable shared folder
link/ID and confirm recipient read permission. Preserve the directory and its
original evidence; no upload or permission change has been performed here.

Transfer the small `experiments/r2_u2/acceptance/` directory and this guide to the
existing project copy. Retain the candidate run, source snapshot, configs, frozen
inputs and linked reports. The manifest enumerates exact required files; do not
re-upload datasets if already present. The shared source snapshot remains
`software_sources/shared_sources.zip`; never overwrite historical Drive sources
to make hashes pass. Use the existing verified scratch-source procedure or a
separate checkout containing those exact bytes if the Drive sources differ.

The receiving project also needs the exact candidate row from the manifest in
its main `reports/ledger.csv`. Merge only that row after checking for an existing
matching run ID; never replace a teammate's ledger with this machine's copy. The
main ledger is ignored by Git here, so this runtime row will not travel in an
ordinary source commit. The manifest preserves it as reviewable small metadata.
All shared Python sources in the original snapshot, not only Trainer, are bound
by the handoff manifest. Verify against an isolated matching source copy instead
of modifying another developer's current source tree.

First verify the bundle, with the receiver's actual project root:

```text
python -B experiments/r2_u2/acceptance/verify_handoff.py --project-root <PROJECT_ROOT>
```

This reads and hashes artifacts; it writes nothing. It requires the exact
approved sources, inputs and evidence. It rejects missing/mismatched files and
missing/duplicated/changed ledger rows. Successful identity verification alone
still reports the model-load check as pending.

In the already prepared Account B environment, use an **existing local** AST
snapshot directory ending in revision
`f826b80d28226b62986cc218e5cec390b1096902`:

```text
python -B experiments/r2_u2/acceptance/verify_handoff.py --project-root <PROJECT_ROOT> --model-snapshot <EXISTING_PINNED_AST_SNAPSHOT>
```

The check requires recorded torch 2.14.1+cu130 and transformers 5.22.0 but uses
CPU only. It performs no downloads, audio access, forward/backward passes,
optimizer steps, checkpoint saving or resume. It verifies pretrained identity,
loads `best.pt` strictly on CPU, checks finite model weights/group fingerprints,
sets the last two blocks/final norm/head trainable, and creates a fresh empty
AdamW optimizer with backbone 1e-5, head 1e-4 and weight decay 0.01. It rechecks
the checkpoint hash. It does not establish future R3 gradient/mode/scheduler
correctness or authorize training. The snapshot may have been lost with Colab
scratch; absence is a real prerequisite to resolve, not permission to substitute
a model or silently download one.

Save stdout to a **new uniquely named** recipient receipt (exclusive creation,
never overwrite an earlier report) or retain it in the executed notebook. Record
receiver, date, accessible artifact locator and manifest hash alongside it. If
the check fails, preserve its error and do not start R3.

R3 must start a new run and optimizer, loading **only** checkpoint `model` weights.
Do not call warm-up resume or restore its optimizer/scaler/scheduler/RNG state.
The real R3-U2 runner/recipe remains a separate task: effective batch 16,
declared epoch budget, warm-up/cosine schedule, partial-adaptation evidence and
safe resume need implementation/review. Full selection comparison must stay
paired; calibration, operating and final panels are separate later work. No B
lock, production selection or held-out exposure is changed by this handoff.

## Minimum outstanding team actions

1. Attach a written team/Muneeb sign-off or explicitly confirm that Usman's
   recorded attestation is the sufficient approval evidence under team practice.
2. Supply a durable recipient-accessible artifact link and confirm permissions.
3. Execute and retain the CPU recipient check in the prepared environment, then
   implement/review the R3 runner under its own frozen contract before training.

No additional R2-U2 GPU training is required to resolve these handoff items.


## Corrective integration verification and checkout limitation

The official equivalence guard permits only exact accepted bytes or the same
source after CRLF-to-LF canonicalization. This is a copy-equivalence check, not a
waiver of the experiment's raw-byte research provenance gates. Nine files in
this Windows checkout differ from the original manifest only by line endings.
The original launch gate correctly rejects such raw-byte differences; retrieve
the exact approved bytes from the retained Drive source set before reproducing
that run. Do not amend historical manifests or claim those raw hashes pass here.
The original accepted scope.py hash itself includes CRLF and remains unchanged.
No experiment files were edited during this correction.
