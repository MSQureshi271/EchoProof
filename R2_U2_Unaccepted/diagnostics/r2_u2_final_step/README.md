# R2-U2 final-step diagnostic (not training or acceptance)

Copy this entire small folder into your own Account B project at
`Colab_R2_U1_Workspace/diagnostics/r2_u2_final_step/`.
Required new files: `diagnose.py`, `test_safety.py`, and
`R2_U2_Final_Step_Diagnostic.ipynb`. Keep this README as the instructions.
Do not overwrite original source, notebook, reports, data, or checkpoints.

Open the NEW diagnostic notebook in Google Colab on Account B; select Tesla T4.
Use a fresh runtime/process, with the same already prepared library versions as
the failed run. No cell installs packages. Set PROJECT_ROOT in the first cell;
the default is `/content/drive/MyDrive/Colab_R2_U1_Workspace`.
The default evidence location is `runs/r2/r2_u2_ast_b_baseline_v1` beneath it.
If your originals are instead under the evidence export, set RUN_RELATIVE to
`R2_U2_Colab_Evidence/runs/r2/r2_u2_ast_b_baseline_v1`. Choose their actual existing
location; do not move or duplicate them just for this diagnostic.

Run the new notebook's cells in order. Do not run the training notebook.
The diagnostic requires the original project src package, frozen metadata and R1
reports, config, source-bound R2-U2 test report, original run directory and
canonical shards in `data/processed/cohort_v1/`. Those existing workspace inputs
are checked before replay. Missing/changed inputs or a mismatched recorded
runtime stop execution; send the resulting report/error instead of improvising
dependency changes. The frozen source hashes must still match the failed run.

What it does:

1. Validates known checkpoint/history/baseline SHA-256s, source/input identity,
   software evidence, runtime versions, model/optimizer tensors and scaler state.
2. Uses the original Trainer loader with metadata-only items to reconstruct the
   complete epoch-two order (including DataLoader's base-seed consumption).
3. Reads only the final two B-training recordings and their epoch-two crops;
   verifies canonical PCM identities. Hashes their entire source shards before
   and after for integrity. No validation/test audio is read.
4. Loads the pinned original AST in scratch and the existing checkpoint in memory.
   Checks backbone identity and stateless classifier structure. Reconstructs the
   original train/eval modes and deterministic flags.
5. Runs eight small forward/backward conditions: original FP16/32768 twice;
   instrumented FP16/32768; FP16/16384; FP16/1; FP32/32768; FP32/1; and the same
   two crops with a deliberately counterfactual denominator of 16. The last
   condition is NOT a full 16-example batch or a proposed training change.
6. Checks module boundaries and the first observed non-finite ATen output in the
   instrumented backward. Checks repeat/instrumentation parity against original
   accumulation. Fused-kernel internals are not visible at this boundary.
7. Verifies original files and in-memory model weights/buffers are unchanged.
   Writes one new, unique JSON under `diagnostics/r2_u2_final_step/results/`.

No optimizer step, scaler update, Trainer construction/fit/resume, ledger append,
checkpoint save, policy amendment or publication is performed. The disposable
optimizer exists only to use the real GradScaler unscale kernel; its step method
raises if called. Gradients are created, unscaled/clipped for inspection, then
cleared in memory. Original checkpoints are never written. Model downloads and
temporary runtime files go only into a new `/content/echoproof-r2u2-diagnostic-*`
scratch folder. A new UUID report is created even on most diagnostic failures.
Abrupt runtime termination can prevent final checks/report writing; that is not
a passing diagnostic. Do not run other project writers concurrently.

Download the JSON using the final notebook cell and return it for analysis.
It includes recording IDs, crop hashes, complete reconstructed order, numerical
summaries, library source hashes, runtime settings, comparison results and limits.
It contains no audio or full model weights. Also retain any notebook error text
if execution stops before a report is created. Do not publish or accept the model
based on this diagnostic's completion message.

The original pre-step RNG state and original per-step log were not saved. The
final-skip/pre-attempt-weight premise is a strong conditional inference from
matched control flow, counters, scaler state, frozen backbone and a buffer-free
head, not a historical pre-step snapshot. RNG consumption is tested in replay;
fresh CUDA kernel/runtime state is an explicit limitation. No finding can explain
the first historical overflow automatically or recover lost gradient averages.

Local check (standard-library only): `python -B test_safety.py` from this folder.
These tests check guards and report decisions, not GPU numerical correctness.
GPU execution has intentionally not been performed by Codex.
