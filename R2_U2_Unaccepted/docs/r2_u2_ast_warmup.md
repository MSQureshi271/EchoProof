# R2-U2: Protocol B frozen AST head warm-up

## Objective and boundaries

Authority: `REVISED_PLAN.md`, Section 6, R2-U2. Owner: Muhammad Usman Murtaza (Usman); reviewer: Muneeb. Dependencies: GR1 data lock, R1-U2 AST wrapper/feasibility, and R1-M1 shared trainer. Train the two-class head for 1–3 epochs on B training audio while the pretrained backbone remains frozen and in evaluation mode. Preserve a warm-up checkpoint as the frozen-AST comparison and append its experiment-ledger row. R3-U2 subsequently performs partial adaptation; completing this warm-up does not establish adaptation or final model quality.

Use the existing `ASTDetector`, `Trainer`, `CropDataset`, `ChunkDataset` and shared ledger schema. The head is the wrapper's complete `classifier.*` group, including classifier normalization; it is not limited to the two newly initialized dense tensors.

Audio scope is retained, assigned B training recordings (genuine, Melo, VITS), plus **selection-validation only** (genuine, Melo, VITS, XTTS). Calibration, operating, test and Chatterbox audio are not consumed. Reading split/chunk metadata to establish those boundaries does not authorize audio access. A retained cohort row need not have a B assignment: XTTS for train speaker 103 is intentionally outside B training.

Training consumes canonical audio directly, using one seeded four-second crop per recording per epoch. It does not consume the WavLM embedding cache, WavLM/LR five-crop bank, or BiGRU mel statistics. Validation uses the complete canonical windows for selection recordings, aggregated to recording scores by the shared trainer.

## Explicit baseline recipe

The frozen run recipe is `configs/r2_u2_ast_b.json`. Its `decisions` distinguish plan requirements, measured recommendations and newly selected implementation settings. The user authorized resolving open settings during implementation; this is not a record of teammate approval.

| Setting | Value | Basis |
|---|---|---|
| Epochs | 2 | Within the prescribed 1–3; R1-U2's recorded budget uses two |
| Optimizer | AdamW | Shared trainer / project methodology |
| Head learning rate; weight decay | 1e-4; 0.01 | Explicit warm-up baseline choice, using the AST adaptation proposal as a starting point |
| Schedule | Constant | Explicit choice for a two-epoch baseline; R3's learning-rate warm-up/cosine schedule is separate |
| Train microbatch; accumulation | 4; 4 | Probe recommends head batch four; accumulation four gives effective batch 16, chosen for consistency with R3 |
| Evaluation microbatch | 16 | R1-U2 probe |
| Precision | FP16 autocast with GradScaler | R1-U2 T4 probe; no automatic precision fallback |
| Seed; workers | 42; 0 | Explicit reproducibility recipe; epoch-dependent crops use the shared seed derivation |
| Class handling | Natural composition; no rebalancing | Explicit choice: 1,510 genuine and 3,020 TTS recordings are not balanced classes |
| Gradient clipping | 1.0 | Explicit use of the shared trainer setting, after unscaling |
| Selection criterion | All selection recordings, macro F1 at threshold 0.5; first epoch wins exact ties | Shared trainer checkpoint rule; distinct from later production-model ranking on unseen XTTS |
| Early-stop patience | 3 | Does not shorten the two-epoch recipe |

The probe's accumulation two accompanies **partial-stage microbatch eight**, not head microbatch four. Preserve the checkpoint's processor: 16 kHz input, normalization enabled, 128 mel bins, 1,024-frame grid, mean -4.2677393, standard deviation 4.5689974. Do not replace its normalization with train-fitted BiGRU statistics or change the grid.

Pin AST to `MIT/ast-finetuned-audioset-10-10-0.4593`, commit `f826b80d28226b62986cc218e5cec390b1096902`, weight SHA-256 `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`. Use Transformers 5.22.0 as in R1. The actual Colab PyTorch/GPU are recorded and tested afresh: the prior AST probe used torch 2.11.0+cu130, whereas the recent WavLM run used 2.14.1+cu130. Prior feasibility is not proof of present compatibility.

## Implementation phases and gates

1. **Metadata preflight:** validate the recipe's frozen file hashes, split identity, leakage evidence, R1 AST/trainer evidence, safe canonical paths, permitted components, speaker/PCM boundaries and complete selection windows. No audio or model download.
2. **Software gate:** run `tests/test_r2_ast_warmup.py`, `tests/test_r1_u2_ast.py` and `tests/test_m1_trainer.py`. Require a source-bound report with exit code zero, passed tests and zero skipped tests. Fixture tests use generated audio/tiny models; they are not research performance results.
3. **Colab staging:** check runtime imports, actual GPU, and version; copy only metadata-plan shards to scratch using streaming SHA-256. Persistent inputs remain unchanged.
4. **Warm-up:** load the pinned model/processor with its loading audit, set the declared label mapping (0 genuine, 1 TTS), and train only `classifier.*`. Verify canonical PCM hashes and lengths. Preserve initial backbone evidence before fitting and across resume.
5. **Publication:** require unchanged backbone parameters/buffers, frozen evaluation mode during actual forwards, changed head, finite results and saved tensors, complete valid selection predictions, successful best-checkpoint reload/prediction comparison, and a linked completed ledger row. Save actual checkpoint and artifact hashes.
6. **Independent integrity check and handoff:** verify the committed artifact manifest, then give Muneeb the actual Colab report and manifest. Successful software tests, a real GPU run, artifact integrity and Muneeb's approval are separate claims.

Skipped **tests** block training. FP16 optimizer steps skipped by GradScaler are recorded separately; they must not be described as skipped tests or hidden. Publication still requires finite recorded epoch results, a changed head and checkpoint/prediction verification. No automatic retry with a different recipe, precision or determinism policy is performed after an error.

## Required project inputs

```text
pyproject.toml
configs/r2_u2_ast_b.json
src/echoproof/
scripts/r2_u2_ast_warmup.py
tests/test_r2_ast_warmup.py
tests/test_r1_u2_ast.py
tests/test_m1_trainer.py
notebooks/R2_U2_Colab_AST_Warmup.ipynb
data/manifests/cohort_v1.parquet
data/splits/protocol_B_v1.parquet
data/processed/chunks_v1.parquet
data/processed/cohort_v1/shard_*.tar  # only the plan's named shards are staged
reports/r1/split_report.json
reports/r1/leakage.json
reports/r1/ast_probe.json
reports/r1/r1_m1_colab_20261103T182916Z.json
reports/ledger.csv                 # existing rows preserved; real run appends its row
```

The recipe pins hashes of the three existing metadata files and four R1 evidence reports. Metadata hashes were checked against the existing R2-S1 frozen evidence during implementation; R2-U2 does not depend on loading R2-S1 features. `dataset/` is unnecessary. Model weights are retrieved and checked in Colab scratch by existing pinned retrieval code.

## Execution and recovery

Use `notebooks/R2_U2_Colab_AST_Warmup.ipynb` in your **own writable Drive project copy**. Set `PROJECT_ROOT`, the persistent run name and the explicit resume option. A new run and an interrupted run must not silently select different directories. Execute cells in order; the notebook does not install packages.

The CLI supports `plan`, `train`, and `verify`; inspect `scripts/r2_u2_ast_warmup.py --help` for options. `plan` prints metadata only. Training requires staged canonical shards, a passing source-bound test report and a Colab GPU. Verification reads generated artifacts and provenance, not research audio or remote model weights.

Keep `baseline.json` and the complete run directory after an interruption. `latest.pt` contains full warm-up continuation state. Resume refuses changed configuration/source/input/runtime identities. A `.writer.lock` may survive a terminated runtime; confirm the old writer has stopped before manually removing **that lock only** in your own output copy. Do not remove an active writer's lock or silently overwrite another run.

The shared trainer checkpoints at epoch boundaries, not after each batch. An interruption before the first completed checkpoint requires inspection and a deliberately chosen new run. It also writes best predictions/history before committing `latest.pt`; if termination leaves those files ahead of the committed checkpoint, R2-U2 refuses resume and preserves the files for inspection. It does not silently truncate history, guess which weights to retain, or publish an inconsistent run.

After training finishes, publication can be retried with resume: it must preserve the original initial baseline, recheck outputs, and reuse the matching completed ledger row. A recovery row, if needed after a crash between checkpoint completion and ledger writing, must be explicitly identified as recovered rather than a second training run. The manifest links the specific ledger row by content rather than hashing the entire mutable team ledger.

## Deliverables and R3 handoff

The actual Colab run—not local implementation—must produce:

- A persistent run directory with `best.pt` (selected frozen-AST comparison and R3-U2 initializer), `latest.pt` (full R2-U2 resume state), initial `baseline.json`, resolved configuration, epoch history and status. `training_mode_receipt.json` and `head_gradient_receipt.json` preserve actual training observations across publication recovery.
- Complete best-checkpoint selection chunk and recording prediction files, preserving IDs, labels and the probability-of-TTS convention.
- `r2_u2_manifest.json` linking checkpoint/artifact checksums, scope, configuration, sources, inputs, model/processor identity, initial/final frozen-state evidence, reload verification and the actual ledger row.
- A completed row in `reports/ledger.csv`, plus actual R2-U2 test/train/verification/final Colab JSON reports under `reports/r2/`. Failures remain visible and do not establish completion.

`best.pt` serves **both** the frozen comparison and R3-U2 initialization. `latest.pt` continues this warm-up; it is not the selected comparison by filename alone. In R3-U2, load the selected model weights, unfreeze the prescribed last two blocks/final norm/head, and rebuild the optimizer as the R3 task requires. Do not reuse the warm-up optimizer as a substitute for that change.

## Council review and current status

Five independent advisors reviewed the recipe; five anonymous reviewers examined their condensed responses in reordered A–E form. Batching respected the three available child-agent slots. The primary agent synthesized the result. Agreement: reuse the shared components; label new recipe choices; preserve best/latest roles and original baseline; prove scope, freeze behavior, reload and ledger linkage. Review additions: distinguish test skips from scaler skips; preserve class mapping; verify current runtime compatibility; verify actual tensors/predictions; recover safely after training but before publication. No additional training trial or protocol expansion was authorized by this review.

Local implementation checks on 7 November parsed 63 Python files and all seven code cells in the unexecuted 14-cell notebook, validated recipe JSON/explicit trainer fields, checked all seven pinned input/evidence hashes, and confirmed the historical AST/pretrained/trainer source identities. Generated in-memory metadata and temporary fixtures exercised the actual planner, complete-suite evidence gate, ledger/status recovery and persisted-observation validation. The temporary fixtures were removed; no research audio was decoded or research outputs generated.

The new suite declares **34 acceptance tests**, including generated PCM identity/scope cases, metadata/evidence failures, actual tiny AST/shared-Trainer frozen-backbone and resume cases, artifact corruption, label mapping and interrupted publication. Source identity covers the runtime and test dependencies; adding a separate later-stage module does not invalidate an unchanged warm-up. These are implemented tests, not recorded full-suite passes.

No actual R2-U2 checkpoint, training result or reviewer approval is claimed until the Colab gates execute successfully. The system `python` command is an unavailable Windows Store launcher, but an existing uv-managed CPython 3.12.14 was found offline without installation. It supports the standard-library checks above; NumPy, PyTorch, Transformers, pytest and the other research dependencies are absent in that interpreter. Those checks cannot establish an ML acceptance-test pass. Run the software gate in the prepared Colab runtime.
