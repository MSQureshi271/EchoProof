# R2-U2 Option B isolated experiment

## Adoption update: 11 November 2026

Usman reports team approval, including Muneeb, of candidate
`option_b_1b3757e898da4baaad2bde49fa3cea01`. The additive authoritative
[adoption and handoff guide](../../docs/r2_u2_acceptance.md) and
[evidence manifest](acceptance/manifest.json) designate its exact `best.pt`.
Recipient access, separate team sign-off evidence and CPU load verification remain
pending; this is not a completed R3 launch. Historical run statuses and policy
files are intentionally unchanged. The sections below describe the original
experimental authorization/execution contract, not a request to rerun it.

The user authorized a prospective fresh two-epoch Option B experiment. This does
not accept the historical failed run or authorize official R2/R3 promotion.
Shared Trainer and R3 code remain unchanged. The exact frozen recipe remains in
configs/historical_recipe.json. The prospective contract is configs/option_b_policy.json.

## Account B execution

Upload only the updated contents of experiments/r2_u2 to the existing private
Drive project, retaining old reports/runs. Include software_sources, configs,
scripts, tests, manifest and notebooks. No dataset upload is needed.

1. Open notebooks/R2_U2_Colab_Software_Tests.ipynb on the prepared T4.
   Set PROJECT_ROOT; run all cells. The changed release requires 17 passing tests,
   zero skips/errors/failures, unchanged verification inputs and linked log/XML
   hashes. Previous 10/10 evidence remains historical and is intentionally stale
   for these new sources. No automatic package installation is performed.
2. Open notebooks/R2_U2_Colab_Option_B_Experiment.ipynb in the same prepared
   environment. Set PROJECT_ROOT and SOFTWARE_REPORT to the NEW verification.json.
   Set AUTHORIZE_FRESH_EXPERIMENT=True deliberately, then run cells in order.
3. Return the launch ZIP and the printed run's experiment_result.json for review.

The source snapshot preserves Git-pinned shared source bytes (commit c7195db),
including current data/model dependencies; experiment changes are uncommitted
and identified by their hashes. Existing Drive shared files are never replaced.
Both notebooks verify the uploaded manifest and stage code into /content scratch.
Frozen metadata/reports and canonical shards are read from existing Drive. Only
plan shards are staged, with checksums. Only assigned B train and selection audio
is decoded, with canonical PCM verification. The AST snapshot is pinned and
checked against original processor, weights and initial parameter fingerprints.

## Prospective checks

Every expected attempt is tied to actual optimizer calls, scaler state, group
sizes, recording and crop identities. The final two examples are normalized by 2.
Only applied updates contribute finite pre-clipping norm count/mean/min/max;
scaler skips retain separate explicit overflow events. No arbitrary skip threshold
is imposed. All-skipped epochs, norm-only overflow, unexplained updates/skips,
unsafe states, invalid metrics or provenance mismatches fail closed. The default
scaler, FP16 autocast, FP32 parameters, clipping, optimizer, batches and constant
schedule remain unchanged. Nonconstant scheduler experiments still preserve the
legacy attempt clock; changing that clock needs separate approval.

The launcher accepts only a fresh UUID run under experiments/r2_u2/runs/option_b_*.
No research resume is offered in this initial release. Do not rerun launch to
repair an interruption; preserve evidence and inspect whether a child writer is
still active. Fixture checkpoint/resume tests remain available.

Outputs include a frozen contract, initial baseline, best/latest checkpoints,
resolved config, full history and updates.jsonl, norm summaries inside latest,
selection predictions, an isolated per-run ledger and checksummed experiment
result. Readable summaries are also written beneath experiments/r2_u2/reports/.
Passing checks sets experiment_checks_passed_pending_team_review, with
research_accepted=false. A final scaler skip is disclosed without claiming that
the final backed-off scale was demonstrated stable. No official publication
manifest, lock update, historical status rewrite or R3 initializer promotion occurs.

The local extension subclasses the shared Trainer rather than duplicating it.
Scope/freeze/prediction helpers are extracted from the historical R2-U2 module;
the old launch/hash gate and publication code are not reused or bypassed. Current
sources receive new software evidence, and frozen historical reports remain
unchanged references to past work.

## Deferred evaluation

Muneeb's observation about speakers 1578, 302 and 7190 is a deferred development
evaluation question. No speaker-specific cause of numerical overflow is established.
Do not alter splits, sampling, training settings or acceptance based on it.

No GPU execution was performed locally. Standard-library tests and static checks
cannot establish a full ML software pass. Review actual Colab evidence before
interpreting or adopting the experimental result.
