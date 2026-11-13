"""Standalone, non-updating R2-U2 forensic replay. Never call fit/step/save.

Only new diagnostic reports and Colab scratch downloads may be written.
Run through the companion notebook, in a fresh process, with original sources.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import traceback
import uuid

RUN_NAME = "r2_u2_ast_b_baseline_v1"
KNOWN = {
    "best.pt": "f9a92af91416334735fb83883462a8cb1aa38e740341100c0a707aa3ecd57339",
    "latest.pt": "4fc4f6f6e6d5ee6f6565558bbc733c7a89a3280709fef3de7c2abc3e4e8b0d29",
    "baseline.json": "eac7cbba160ac14f9ff26341a0df92ea5ef5d28e8de41467d167d8beea609f70",
    "history.jsonl": "4da3c364e707a0b2eaedbc9f94fa7bf44f986e8434b32cc75a7064bd2286a1e7",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def inside(root, path):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(root), f"Path outside selected project: {path}")
    return path


def premise(latest, steps, n, batch, accumulation):
    """Pure check: conditional inference from matched default scaler control flow."""
    attempts = math.ceil(math.ceil(n / batch) / accumulation) * latest["epoch"]
    s = latest["scaler"]
    require(latest["epoch"] == 2 and latest["finished"] is True, "Not the finished two-epoch run")
    require(s["growth_interval"] == 2000 and s["growth_factor"] == 2.0
            and s["backoff_factor"] == 0.5 and s["scale"] == 16384.0,
            "Unexpected scaler policy/state")
    require(attempts < s["growth_interval"] and s["_growth_tracker"] == 0,
            "Final-skip inference is not supported")
    require(attempts == 568 and len(steps) == 4 and set(steps) == {566.0}
            and latest["skipped_steps"] == 2, "Update accounting differs from investigated run")
    return {"expected_attempts": attempts, "optimizer_steps": steps,
            "final_attempt_skipped": "strong conditional inference, not a historical event log",
            "pre_attempt_scale": s["scale"] / s["backoff_factor"],
            "basis": "Default initial scale 65536, no growth possible, two backoffs; tracker zero at finished checkpoint",
            "assumptions": ["Recorded original code and standard recorded PyTorch implementation executed",
                            "No unrecorded external state mutation"]}


def conclusion(cases):
    by = {c["name"]: c for c in cases}
    first = by["original"]
    if not first["scaled_nonfinite"]:
        return "Original-condition replay did not reproduce non-finite parameter gradients; historical cause remains unresolved."
    if not by["original_repeat"]["matches_original"] or not by["instrumented"]["matches_original"]:
        return "Non-finite gradients observed, but repeat/instrumentation parity failed; do not infer an exact historical mechanism."
    if not all(by[name]["rng_unchanged"] for name in ("original", "original_repeat", "instrumented")):
        return "Replayed calculation consumes RNG; historical pre-attempt RNG was not saved. Findings describe this diagnostic replay, not an exact historical reconstruction."
    if by["fp32_scale1"]["scaled_nonfinite"]:
        return "Non-finiteness also occurs in the FP32 reference; investigate inputs/model operations beyond FP16 scaling."
    if not by["fp16_scale1"]["scaled_nonfinite"]:
        return "Evidence supports scale-sensitive FP16 backward overflow in this reconstructed final group; first historical event remains unresolved."
    return "Non-finiteness persists at FP16 scale 1 but not FP32 scale 1; FP16 computation is implicated, not merely a large scale."


def execute(root, run, scratch, report, protect):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
    from echoproof.features.pretrained import pinned_snapshot, PRETRAINED
    from echoproof.models.ast_detector import ASTDetector, BACKBONE
    from echoproof.training import r2_u2
    from echoproof.training.datasets import CropDataset, crop_start, collate
    from echoproof.training.seeding import seed_everything
    from echoproof.training.trainer import Trainer, TrainConfig

    require(Path(r2_u2.__file__).resolve() == root / "src/echoproof/training/r2_u2.py",
            "Imported a different project copy")
    baseline = read(run / "baseline.json")
    require(r2_u2.source_hashes(root) == baseline["provenance"]["source_sha256"],
            "Original source/config hashes differ; restore the original matching workspace")
    for name in baseline["provenance"]["source_sha256"]:
        protect(inside(root, name))
    for name in baseline["recipe"]["input_sha256"]:
        protect(inside(root, name))
    test_path = inside(root, baseline["tests"]["file"])
    protect(test_path)
    require(sha(test_path) == baseline["tests"]["sha256"], "Original test report missing/changed")
    r2_u2._test_gate(root, test_path, r2_u2.source_hashes(root))
    recipe = r2_u2.load_recipe(root)
    require(recipe == baseline["recipe"], "Recipe differs")
    plan = r2_u2.preflight(root, recipe)
    require(plan.sha256 == baseline["provenance"]["plan_sha256"], "Plan differs")
    environment = r2_u2._runtime()
    report["environment"] = environment
    report["torch_build_configuration"] = torch.__config__.show()
    report["original_environment"] = baseline["provenance"]["environment"]
    require(environment == report["original_environment"],
            "Runtime differs from original Account B run; no exact replay attempted. Send this report for review; do not auto-install packages")
    require(environment["gpu"] == "Tesla T4", "This diagnostic targets the recorded T4")
    cfg = TrainConfig(**recipe["train"])
    require(cfg.num_workers == 0 and cfg.class_strategy == "none" and cfg.scheduler == "constant"
            and cfg.batch_size == 4 and cfg.accumulation_steps == 4, "Unsupported historical recipe")
    latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    r2_u2._finite_tensors(latest)
    r2_u2._finite_tensors(best)
    steps = [float(s["step"]) for s in latest["optimizer"]["state"].values()]
    report["premise"] = premise(latest, steps, len(plan.train_rows), cfg.batch_size, cfg.accumulation_steps)
    require(best["epoch"] == latest["best_epoch"] == 2
            and latest["config"] == recipe["train"]
            and best["config_hash"] == latest["config_hash"], "Checkpoint identities differ")
    require(set(best["model"]) == set(latest["model"])
            and all(torch.equal(v, latest["model"][k]) for k, v in best["model"].items()), "Best/latest weights differ")
    require(all(torch.all(s["exp_avg_sq"] >= 0).item() for s in latest["optimizer"]["state"].values()),
            "Negative AdamW second moment")
    report["checkpoint_validation"] = {"all_tensors_finite": True, "best_latest_equal": True,
                                        "optimizer_second_moments_nonnegative": True}
    del best

    # Use the ORIGINAL loader implementation, with a metadata-only dataset.
    # DataLoader's iterator consumes a base seed before RandomSampler's randperm.
    class MetadataDataset:
        def __len__(self):
            return len(plan.train_rows)

        def __getitem__(self, index):
            row = plan.train_rows[index]
            return {"recording_id": row["recording_id"], "chunk_id": str(index), "label": row["label"],
                    "waveform": np.zeros(1, dtype=np.float32), "valid_samples": 1}

    facade = object.__new__(Trainer)  # __init__ writes files; deliberately NEVER called.
    facade.cfg, facade.train_ds, facade.sample_weights = cfg, MetadataDataset(), None
    batches = [b["recording_ids"] for b in Trainer._train_loader(facade, latest["epoch"])]
    order = [rid for b in batches for rid in b]
    require(len(order) == len(set(order)) == len(plan.train_rows)
            and set(order) == {r["recording_id"] for r in plan.train_rows}, "Invalid reconstructed permutation")
    start = ((len(batches) - 1) // cfg.accumulation_steps) * cfg.accumulation_steps
    group_ids = batches[start:]
    ids = [rid for b in group_ids for rid in b]
    require(len(group_ids) == 1 and len(ids) == 2, "Expected final two-recording group not found")
    report["loader"] = {"method": "original Trainer._train_loader with metadata-only items; original collate",
                        "epoch": 2, "batch_count": len(batches), "attempt_index_1based": 568,
                        "ordered_microbatches_sha256": digest(batches), "final_microbatches": group_ids,
                        "order": order, "normalization_denominator": len(ids),
                        "limitation": "Reconstructed from source, seed, version and frozen plan; historical per-step ID log does not exist"}
    rows = {r["recording_id"]: r for r in plan.train_rows}
    shard_dir = root / "data/processed/cohort_v1"
    for shard in sorted({rows[rid]["canonical_path"].split("::")[0] for rid in ids}):
        protect(shard_dir / shard)
    loader = r2_u2.CheckedWaveformLoader(shard_dir, [rows[rid] for rid in ids])
    try:
        ds = CropDataset([(rid, rows[rid]["label"]) for rid in ids], loader, seed=cfg.seed)
        ds.set_epoch(2)
        items = [ds[i] for i in range(len(ids))]
    finally:
        loader.close()
    report["inputs"] = [{"recording_id": rid, "label": rows[rid]["label"],
                         "canonical_path": rows[rid]["canonical_path"], "pcm_sha256": rows[rid]["pcm_sha256"],
                         "n_samples": rows[rid]["n_samples"],
                         "start_sample": crop_start(cfg.seed, 2, rid, rows[rid]["n_samples"]),
                         "valid_samples": item["valid_samples"], "chunk_id": item["chunk_id"],
                         "padded_float32_crop_sha256": hashlib.sha256(item["waveform"].tobytes()).hexdigest()}
                        for rid, item in zip(ids, items)]
    batch = collate(items)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(cfg.seed)
    snapshot = pinned_snapshot("ast", cache_dir=scratch / "model_cache")
    require(sha(snapshot / "preprocessor_config.json") == baseline["provenance"]["model"]["processor_file_sha256"],
            "Processor file differs")
    require(sha(snapshot / PRETRAINED["ast"]["weights_file"]) == baseline["provenance"]["model"]["weights_sha256"],
            "Pretrained weights differ")
    model = ASTDetector.from_dir(snapshot, device="cuda", seed=cfg.seed)
    require(model.digests() == baseline["initial_group_sha256"], "Initial model reference differs")
    model.model.config.id2label = {0: "genuine", 1: "tts"}
    model.model.config.label2id = {"genuine": 0, "tts": 1}
    model.load_state_dict(latest["model"], strict=True)
    model.set_stage("head")
    r2_u2.validate_head_stage(model)
    report["freeze_proof"] = r2_u2._state_proof(model, baseline)
    facade.model, facade.device, facade.class_weights = model, torch.device("cuda"), None
    Trainer._set_train_mode(facade)
    backbone = getattr(model.model, BACKBONE)
    require(not any(m.training for m in backbone.modules()), "Backbone not in eval")
    require(not list(model.model.classifier.buffers()), "Head has buffers: pre-attempt state inference needs review")
    for module in model.model.classifier.modules():
        if not list(module.children()):
            require(isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Identity))
                    or isinstance(module, torch.nn.Dropout) and module.p == 0,
                    f"Unreviewed/stochastic head leaf: {type(module).__name__}")
    require(all(p.dtype == torch.float32 for p in model.parameters()), "Non-FP32 parameters")
    report["model_modes"] = {n: m.training for n, m in model.named_modules()}
    report["model_config"] = model.model.config.to_dict()
    report["runtime_flags"] = {"deterministic": torch.are_deterministic_algorithms_enabled(),
                               "cudnn_deterministic": torch.backends.cudnn.deterministic,
                               "cudnn_benchmark": torch.backends.cudnn.benchmark,
                               "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                               "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                               "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    report["library_sources"] = {name: {"file": inspect.getsourcefile(obj), "sha256": sha(inspect.getsourcefile(obj))}
                                 for name, obj in {"GradScaler": torch.amp.GradScaler,
                                                   "AST": type(model.model),
                                                   "head": type(model.model.classifier),
                                                   "feature_extractor": type(model.feature_extractor)}.items()}
    report["premise"]["pre_attempt_weights"] = (
        "Conditional on the final-skip inference: no optimizer update, frozen backbone, stateless Linear/LayerNorm head; "
        "subsequent evaluation does not change weights. This is not a saved historical pre-step snapshot.")

    def tensor_hash(t):
        return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def model_hash():
        return digest({n: tensor_hash(t) for n, t in model.state_dict().items()})

    def stats(t):
        with _disable_current_modes(), torch.no_grad():
            t = t.detach()
            finite = torch.isfinite(t)
            good = t[finite].double()
            return {"shape": list(t.shape), "dtype": str(t.dtype),
                    "nan": int(torch.isnan(t).sum()), "inf": int(torch.isinf(t).sum()),
                    "finite_min": float(good.min()) if good.numel() else None,
                    "finite_max": float(good.max()) if good.numel() else None,
                    "finite_l2_fp64": float(torch.linalg.vector_norm(good)) if good.numel() else None}

    def leaves(x):
        if isinstance(x, torch.Tensor):
            yield x
        elif isinstance(x, (list, tuple)):
            for v in x:
                yield from leaves(v)
        elif isinstance(x, dict):
            for v in x.values():
                yield from leaves(v)

    class Trace(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.operations = 0
            self.first_nonfinite = None

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            self.operations += 1
            if self.first_nonfinite is None:
                with _disable_current_modes():
                    values = [v for v in leaves(output) if v.is_floating_point()]
                    if any(not torch.isfinite(v).all().item() for v in values):
                        self.first_nonfinite = {"index": self.operations, "operation": str(func),
                                                "inputs": [stats(v) for v in leaves((args, kwargs or {})) if v.is_floating_point()],
                                                "outputs": [stats(v) for v in values]}
            return output

    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())

    def restore_rng():
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])

    def rng_hash():
        return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(),
            torch.get_rng_state().numpy().tobytes(), [s.cpu().numpy().tobytes() for s in torch.cuda.get_rng_state_all()]))).hexdigest()

    initial_model_hash = model_hash()
    report["initial_diagnostic_model_sha256"] = initial_model_hash
    report["cases"] = []
    reference = None
    parameters = [p for p in model.parameters() if p.requires_grad]

    def case(name, precision, scale, denominator, instrument=False):
        nonlocal reference
        print(f"Diagnostic condition: {name}", flush=True)
        restore_rng()
        before_rng = rng_hash()
        model.zero_grad(set_to_none=True)
        facade.cfg = TrainConfig(**{**recipe["train"], "precision": precision})
        # No optimizer state is loaded, and optimizer.step is disabled below.
        optimizer = torch.optim.AdamW(model.param_groups(head_lr=cfg.lr, backbone_lr=cfg.lr, weight_decay=cfg.weight_decay))
        def forbidden(*args, **kwargs):
            raise RuntimeError("Optimizer updates are forbidden in this diagnostic")
        optimizer.step = forbidden
        facade.scaler = torch.amp.GradScaler("cuda", init_scale=scale, enabled=True)
        hooks, forward, captures = [], [], {}
        trace = Trace()
        result = {"name": name, "precision": precision, "scale": scale, "denominator": denominator,
                  "classification": "original-condition reconstruction" if name.startswith("original") else "exploratory/instrumented comparison"}
        if instrument:
            def hook(n):
                def observe(module, inputs, output):
                    forward.append({"module": n, "sequence": len(forward) + 1,
                                    "inputs": [stats(v) for v in leaves(inputs) if v.is_floating_point()],
                                    "outputs": [stats(v) for v in leaves(output) if v.is_floating_point()]})
                return observe
            for n, module in model.named_modules():
                if not list(module.children()) or n == "model.classifier":
                    hooks.append(module.register_forward_hook(hook(n)))
        try:
            if denominator == len(ids) and not instrument:
                # Actual original accumulation method, not a rewritten approximation.
                total, den = Trainer._accumulate(facade, [batch])
                result["loss_sum"], result["loss_denominator"] = total, den
            else:
                inputs, labels = Trainer._move(facade, batch)
                with Trainer._autocast(facade):
                    logits = model(**inputs)
                loss_sum = F.cross_entropy(logits.float(), labels, reduction="sum")
                require(torch.isfinite(loss_sum).item(), "Non-finite diagnostic loss")
                result["loss_sum"], result["loss_denominator"] = float(loss_sum.detach()), denominator
                result["logits"] = stats(logits)
                if instrument:
                    with trace:
                        facade.scaler.scale(loss_sum / denominator).backward()
                else:
                    facade.scaler.scale(loss_sum / denominator).backward()
            result["rng_unchanged"] = before_rng == rng_hash()
            result["scaled_gradients"] = {n: stats(p.grad) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
            require(len(result["scaled_gradients"]) == 4, "Missing classifier gradient")
            result["scaled_nonfinite"] = any(s["nan"] or s["inf"] for s in result["scaled_gradients"].values())
            captures = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            facade.scaler.unscale_(optimizer)  # Original unscale kernel; NEVER step/update.
            result["unscaled_gradients"] = {n: stats(p.grad) for n, p in model.named_parameters() if p.requires_grad}
            # Read-only private state, version-bound and independently cross-checked.
            found = facade.scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
            result["scaler_found_inf"] = sum(float(v) for v in found.values()) > 0
            require(result["scaler_found_inf"] == result["scaled_nonfinite"], "Scaler and elementwise evidence disagree")
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip))
            result["original_clip_return"] = norm if math.isfinite(norm) else str(norm)
            result["post_clip_gradients"] = {n: stats(p.grad) for n, p in model.named_parameters() if p.requires_grad}
            require(not optimizer.state, "An optimizer update/state allocation unexpectedly occurred")
            result["optimizer_state_remained_empty"] = True
            require(all(p.grad is None for p in backbone.parameters()), "Backbone received gradients")
            if reference is None:
                reference = (captures, result["loss_sum"])
            if name in ("original_repeat", "instrumented"):
                result["matches_original"] = (result["loss_sum"] == reference[1] and all(
                    torch.all((v == reference[0][n]) | (torch.isnan(v) & torch.isnan(reference[0][n]))).item()
                    for n, v in captures.items()))
            result["model_weights_unchanged"] = model_hash() == initial_model_hash
            require(result["model_weights_unchanged"], "Diagnostic mutated model weights/buffers")
            if instrument:
                result["forward_module_boundaries"] = forward
                result["first_nonfinite_forward_boundary"] = next(
                    (event for event in forward if any(s["nan"] or s["inf"] for s in event["outputs"])), None)
                result["backward_first_observed_nonfinite"] = trace.first_nonfinite
                result["backward_dispatch_operations"] = trace.operations
                result["trace_limit"] = "First observed ATen output, not internal fused-kernel arithmetic; instrumentation parity is checked separately"
            report["cases"].append(result)
        finally:
            for handle in hooks:
                handle.remove()
            model.zero_grad(set_to_none=True)
            require(model_hash() == initial_model_hash, "Model state changed, including during a failed condition")

    scale = report["premise"]["pre_attempt_scale"]
    case("original", "fp16", scale, 2)
    case("original_repeat", "fp16", scale, 2)
    case("instrumented", "fp16", scale, 2, instrument=True)
    case("fp16_scale16384", "fp16", 16384, 2)
    case("fp16_scale1", "fp16", 1, 2)
    case("fp32_scale32768", "fp32", scale, 2)
    case("fp32_scale1", "fp32", 1, 2)
    case("counterfactual_denominator16", "fp16", scale, 16)
    report["conclusion"] = conclusion(report["cases"])
    by_case = {c["name"]: c for c in report["cases"]}
    report["controlled_comparisons"] = {
        "original_nonfinite_reproduced": by_case["original"]["scaled_nonfinite"],
        "fp16_scale16384_nonfinite": by_case["fp16_scale16384"]["scaled_nonfinite"],
        "fp16_scale1_nonfinite": by_case["fp16_scale1"]["scaled_nonfinite"],
        "fp32_scale32768_nonfinite": by_case["fp32_scale32768"]["scaled_nonfinite"],
        "fp32_scale1_nonfinite": by_case["fp32_scale1"]["scaled_nonfinite"],
        "counterfactual_denominator16_nonfinite": by_case["counterfactual_denominator16"]["scaled_nonfinite"],
        "interpretation_limit": "Changing denominator tests loss-amplitude sensitivity only; it does not isolate group size from example composition or establish a valid alternative recipe",
    }
    report["limitations"] = [
        "No historical per-step gradient/ID log or pre-event RNG snapshot exists; reconstruction is conditional",
        "Fresh-process CUDA kernel state/backend choices cannot be certified identical from saved metadata alone",
        "RNG consumption is tested; the historical pre-event RNG state cannot be reconstructed exactly",
        "The denominator16 condition deliberately underweights this same two-recording group; it is NOT a full-batch reproduction or proposed recipe",
        "FP32/scaling comparisons are diagnostic counterfactuals, not new accepted research runs",
        "This cannot identify the first historical overflow or recover lost finite-only epoch norm averages",
        "No research acceptance decision follows from this diagnostic",
    ]
    report["model_weights_unchanged"] = model_hash() == initial_model_hash
    report["status"] = "diagnostic_completed_not_research_acceptance"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-dir", default=f"runs/r2/{RUN_NAME}")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    require((root / "pyproject.toml").is_file(), "Select the existing project root")
    run = inside(root, args.run_dir)
    require(run.name == RUN_NAME and run.is_dir(), "Select the original evidence run directory")
    output_root = root / "diagnostics/r2_u2_final_step/results"
    require(not output_root.resolve().is_relative_to(run) and not run.is_relative_to(output_root.resolve()),
            "Output overlaps original run")
    require(output_root.resolve().is_relative_to(root), "Output symlink leaves project")
    scratch = Path("/content") / ("echoproof-r2u2-diagnostic-" + uuid.uuid4().hex)
    require(sys.platform == "linux" and Path("/content").is_dir(), "GPU diagnostic is Colab-only; local tests use test_safety.py")
    scratch.mkdir()
    (scratch / "tmp").mkdir()
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      HF_HOME=str(scratch / "hf"), TORCH_HOME=str(scratch / "torch"),
                      XDG_CACHE_HOME=str(scratch / "cache"), NUMBA_CACHE_DIR=str(scratch / "numba"),
                      MPLCONFIGDIR=str(scratch / "matplotlib"),
                      TMPDIR=str(scratch / "tmp"), TMP=str(scratch / "tmp"), TEMP=str(scratch / "tmp"))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / ("diagnostic_" + uuid.uuid4().hex + ".json")
    protected = {}

    def protect(path):
        path = inside(root, path)
        require(path.is_file(), f"Missing original input: {path}")
        if str(path) not in protected:
            print(f"Hashing protected input: {path.relative_to(root)}", flush=True)
            protected[str(path)] = sha(path)

    report = {"schema_version": 1, "task": "R2-U2 final-step diagnostic", "status": "started",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "research_accepted": False, "optimizer_updates_authorized": False,
              "run_directory": str(run), "diagnostic_script_sha256": sha(__file__),
              "diagnostic_files": {p.name: sha(p) for p in Path(__file__).parent.iterdir()
                                   if p.is_file() and p.suffix in (".py", ".ipynb", ".md")},
              "scratch": str(scratch)}
    exit_code = 0
    try:
        for p in run.rglob("*"):
            if p.is_file():
                protect(p)
        for name, expected in KNOWN.items():
            require(protected.get(str(run / name)) == expected, f"Wrong original artifact: {name}")
        for p in (root / "reports").rglob("*"):
            if p.is_file() and (p.name == "ledger.csv" or p.name.startswith("r2_u2_")):
                protect(p)
        execute(root, run, scratch, report, protect)
    except BaseException as exc:
        exit_code = 1
        report.update(status="diagnostic_failed_or_interrupted", error=str(exc), traceback=traceback.format_exc())
    finally:
        changes = []
        for name, before in protected.items():
            try:
                if sha(name) != before:
                    changes.append(name)
            except OSError:
                changes.append(name)
        report["protected_input_sha256"] = protected
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["original_files_unchanged"] = not changes
        report["changed_or_missing_originals"] = changes
        if changes:
            exit_code = 1
            report["status"] = "integrity_failure_do_not_interpret"
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        print(json.dumps({"report": str(output), "status": report["status"],
                          "original_files_unchanged": not changes, "research_accepted": False}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
