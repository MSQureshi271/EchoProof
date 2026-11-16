# How should the sealed evaluation load the eight bundles? (decision paper, 16 November 2026)

**Status: analysis and recommendation only. No decision is recorded and no file was changed for either option.** The project owner decides; a recorded decision-log entry is needed either way.

## The question

R5-N1 scores every test recording with all eight locked bundles (four Protocol B bundles in `locks/B_lock_gr3.json`, four Protocol A bundles in `locks/A_lock.json`). The shared router `echoproof.inference.load_inference_bundle` is hashed in the B lock, so the choice is between:

- **Option 1: route the bundles in the shared code.** Edit `src/echoproof/inference.py` (and `cli.py` for a CLI `evaluate`) so that `load_inference_bundle` also finds the A bundles, then use the shared entry points.
- **Option 2: a dedicated loader registry and sealed runner.** Add new files only (a registry that maps each locked bundle to its verified loader, plus a sealed-evaluation script) that call the unchanged shared `predict_recording` / `analyze`.

## Facts that frame it (checked in the repository)

1. **There is no `cli evaluate` and no sealed-evaluation code today.** `cli.py` has `predict`, `benchmark`, `budget`, environment commands and `data`; nothing scores a manifest. The plan's "`cli evaluate --suite sealed --lock ...`" must be written under either option. This removes the usual advantage of reusing an existing evaluator.
2. **The shared router already cannot load most of the eight bundles.** It routes only `logistic_regression` (to the legacy LR loader, which accepts a threshold override) and `wavlm_head` (to the B loader, which refuses `wavlm_head:A`). The B BiGRU (`bigru:B`, `load_calibrated_bigru_bundle`, review status pending, blocked unless explicitly allowed), the B AST (`ast_partial:B`, its own Platt loader), and all four A bundles have no route. Option 1 therefore means adding about six routes, not one.
3. **The A loaders differ in what they need.** `load_ast_a_bundle` requires CUDA and fp16 and refuses everything else; `load_wavlm_bundle_a`, `load_lr_bundle_a` and `load_bigru_bundle_a` take a device; every A loader refuses a threshold override. The router's signature is `(bundle_path, threshold)`: no device, no precision.
4. **The shared files are hashed in both locks.** `inference.py`, `cli.py` and `app_support.py` appear in `locks/B_lock_gr3.json` (`code.src_echoproof_sha256`) and in `locks/A_lock.json`. The locks record provenance (what built the bundles); nothing enforces equality at run time. There is precedent for an additive shared edit (decision log, 14 November: `load_inference_bundle` routing by family, confirmed by the owner, teammates' confirmation reported).
5. **The shared scoring path is `analyze` / `predict_recording`.** Both options use it unchanged for decoding, chunking, scoring, calibration and the decision, so the choice affects how a bundle is found and verified, not how a recording is scored.
6. **Time.** By `REVISED_PLAN.md` the sealed run was due on D12 (15 November); today is D13 and R4-N2, R4-N3 and R5-S1 are open.

## How to decide

The sealed evaluation is the only moment the test partition is read and no model choice may follow it (H9). So the decision rule should be: **pick the option that makes the sealed run easiest to audit and least able to surprise us, then the one that is cheapest.** Concretely, score each option on:

| Criterion | Why it matters |
|---|---|
| Locked files touched | Anything hashed in the locks must be re-recorded; B behaviour must not change |
| Blast radius | Which other paths (production app, CLI predict) can break |
| Auditability of the sealed run | The run should hash its own code, verify every bundle against the lock, log test access |
| Handles the real loader differences | Device, fp16, pending-review gates, no threshold override |
| Effort and schedule | Days remaining |
| Future reuse | Whether an A bundle could later be served by the app |
| Drift risk | Whether the scoring path used for R5 can diverge from the one the app and CLI use |

## Comparison

| Criterion | Option 1: shared routing | Option 2: dedicated registry and runner |
|---|---|---|
| Locked files touched | `inference.py` (all eight routes) and `cli.py` for `evaluate`; both change the hashes recorded in the B and A locks; needs a recorded amendment and a note that the locks describe earlier code | None. New files only; both lock records stay exactly true |
| Blast radius | High: the production path (`analyze`, app, CLI predict) imports the router; a routing mistake can break the B WavLM app flow. Needs B regression and parity re-runs | Low: production path untouched; a mistake can only affect the sealed runner |
| Auditability | Weaker: the evaluator runs the same file that was edited after the locks were written, so provenance rests on a recorded amendment | Stronger: the runner can verify both lock files, every bundle hash, the test-manifest hash and its own source hashes before reading any test row, and write the test-access log |
| Loader differences | Awkward: the router has no device or precision argument; AST-A's CUDA-only rule and the BiGRU pending-review gate need new parameters in a shared signature | Natural: the registry declares per bundle the loader, device and any explicit allow-pending flag, and fails closed |
| Effort | Larger: six routes, signature change, tests for the shared path, regression of B, plus the evaluator itself | Smaller: one registry, the runner, tests; no regression of B needed |
| Future reuse | Best: the app and CLI could later serve A or other B bundles with the same entry point | Weaker: the app still loads only what the router knows; later reuse needs option 1 anyway (a small step then) |
| Drift risk | Lowest: one loader table | Real but controllable: two tables. Mitigate with a test that every bundle in both locks resolves in the registry and that, where the router has a route, the registry returns an equal scorer, spec and rule |
| Fit with the plan's wording | Matches "`cli evaluate`" | Differs in the command name only; a thin `cli evaluate` wrapper can be added later as one additive edit |

## Which is better and which is worse

- **Option 2 is better for the sealed run.** It changes nothing that the two locks describe, it keeps the production B path out of the risk, and it is the only option that lets the run verify the locks and its own code before the first test row is read. It is also smaller, which matters on D13. Its one weakness (a second loader table) is testable.
- **Option 1 is worse for this task.** It would edit hashed shared code after both locks were written, change a signature to carry device and precision just for evaluation, and put the app's production path in the blast radius of the sealed run's needs. Its real advantage (one entry point for the app and the CLI) matters only if an A bundle will be served by the app, and no current plan item does that.
- **Option 1 is not wrong.** If the team decides that A bundles, or the B BiGRU and AST bundles, should be loadable from the CLI and app for the demo, option 1 is the right place to do it, after R5, as a separate recorded change.

## Recommendation

1. Use **Option 2** for R5-N1: a registry module (for example `src/echoproof/evaluation/r5_loaders.py`) and a sealed runner script that verifies `locks/B_lock_gr3.json`, `locks/A_lock.json`, every bundle and the test manifest against their hashes, writes the test-access log first, then scores through the unchanged shared `predict_recording`.
2. Do not add A routing to the app or `cli.py` now. Record a decision-log entry that says so.
3. If a CLI `evaluate` is still wanted, add it afterwards as a thin wrapper around the runner (one additive edit, recorded as an amendment).
4. Run the sealed evaluation of AST-A and AST-B on a GPU; both are verified on CUDA fp16 only.

## What would change the recommendation

- A decision to demo or serve an A bundle (or the B BiGRU or AST) from the app or CLI: then option 1 is needed, as a separate change.
- Evidence that the registry's loaders return different scores from the shared router for a bundle the router can load: then fix the registry before any sealed run.

## To record

A decision-log entry stating the chosen option, who confirmed it, and (for option 1) the amendment to the locked source hashes.
