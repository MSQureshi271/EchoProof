"""Transparent baseline budget arithmetic; no scientific Python dependency."""
import csv
import io
import math
from pathlib import Path
import json

FIELDS = {
    "schema_version", "scope", "train_recordings", "validation_recordings", "scoring_recordings",
    "train_crops", "validation_crops", "scoring_crops", "canonical_audio_seconds", "epochs",
    "validate_every_epochs", "available_compute_hours", "max_session_hours", "free_drive_gb",
    "expected_bigru_device", "environment_overhead_gb", "other_storage_gb", "extra_bigru_seed_fits",
    "extra_lr_grid_runs", "final_lr_refits", "final_bigru_refits", "failed_lr_grid_equivalents",
    "failed_bigru_fit_equivalents", "bigru_checkpoint_copies_per_fit", "additional_original_copy_gb",
}
COUNTS = {"train_recordings", "validation_recordings", "scoring_recordings", "train_crops", "validation_crops",
          "scoring_crops", "epochs", "validate_every_epochs", "extra_bigru_seed_fits", "extra_lr_grid_runs",
          "final_lr_refits", "final_bigru_refits", "failed_lr_grid_equivalents", "failed_bigru_fit_equivalents",
          "bigru_checkpoint_copies_per_fit"}
OPTIONAL_ZERO = {"extra_bigru_seed_fits", "extra_lr_grid_runs", "final_lr_refits", "final_bigru_refits",
                 "failed_lr_grid_equivalents", "failed_bigru_fit_equivalents"}


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != FIELDS:
        raise ValueError("budget plan has unknown or missing fields")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1 or plan["scope"] != "baseline_known_generator":
        raise ValueError("only schema 1 baseline_known_generator budgets are implemented")
    for name in FIELDS - {"schema_version", "scope", "expected_bigru_device"}:
        value = plan[name]
        if value is None:
            continue
        minimum = 0 if name in OPTIONAL_ZERO or name in {"environment_overhead_gb", "other_storage_gb", "additional_original_copy_gb"} else 1
        if name in COUNTS:
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum} or null")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < (0 if minimum == 0 else 1e-12):
            raise ValueError(f"{name} must be a finite nonnegative/positive number or null")
    if plan["expected_bigru_device"] is not None and (not isinstance(plan["expected_bigru_device"], str) or not plan["expected_bigru_device"].strip()):
        raise ValueError("expected_bigru_device must be a device name or null")
    for part in ("train", "validation", "scoring"):
        if plan[f"{part}_crops"] is not None and plan[f"{part}_recordings"] is not None and plan[f"{part}_crops"] < plan[f"{part}_recordings"]:
            raise ValueError(f"{part}_crops cannot be less than {part}_recordings")
    return plan


def estimate_budget(plan, benchmark):
    validate_plan(plan)
    if benchmark.get("schema_version") != 1 or benchmark.get("scope") != "baseline_fixture_benchmark":
        raise ValueError("incompatible benchmark")
    missing = sorted(name for name in FIELDS if plan[name] is None)
    result = {"schema_version": 1, "scope": "baseline_known_generator", "status": "needs_inputs" if missing else "provisional",
              "missing_inputs": missing, "reserve_fraction": 0.25, "inputs": plan, "rows": [],
              "advanced_scope": {"status": "deferred", "future_final_fits": [f"{model}/{protocol}" for model in ("lr", "bigru", "wavlm", "ast") for protocol in ("A", "B")],
                                 "requirements": "Additional verified generator families, baseline review, pretrained timings, AST warm-up/adaptation and capacity; lock B before A development."},
              "limitations": ["Fixture timings are provisional and must be remeasured on representative audited data.",
                              "LR fitting is extrapolated linearly from synthetic rows; actual convergence can be nonlinear.",
                              "Features are assumed cached once; warm fixture file reads do not characterize real-data Drive I/O, cleaning or transfer times."]}
    if missing:
        return result
    paths = benchmark["paths"]
    required = ("drive_read_decode", "lr_features", "lr_fit_grid", "lr_scoring", "bigru_features", "bigru_forward", "bigru_train_step")
    if any(name not in paths or not paths[name].get("measured_count") for name in required):
        raise ValueError("benchmark is missing required baseline measurements")
    for name in required:
        rate = paths[name].get("p95_seconds")
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"invalid benchmark timing: {name}")
    batch = benchmark["batch_size"]
    accumulation = benchmark["accumulation_steps"]
    if type(batch) is not int or batch < 1 or type(accumulation) is not int or accumulation < 1:
        raise ValueError("invalid benchmark batching")
    effective = batch * accumulation
    train, val, scoring = (plan[f"{key}_crops"] for key in ("train", "validation", "scoring"))
    total_crops = train + val + scoring
    bigru_fits = 2 + plan["extra_bigru_seed_fits"] + plan["final_bigru_refits"] + plan["failed_bigru_fit_equivalents"]
    lr_grids = 1 + plan["extra_lr_grid_runs"] + plan["failed_lr_grid_equivalents"]
    lr_fits = lr_grids * 8 + plan["final_lr_refits"]
    validation_passes = math.ceil(plan["epochs"] / plan["validate_every_epochs"])
    steps_per_epoch = math.ceil(train / effective)
    rows = result["rows"]
    def add(component, seconds, basis):
        rows.append({"component": component, "estimated_seconds": seconds, "basis": basis})
    def batches(count):
        return math.ceil(count / batch)
    profiling_seconds = benchmark.get("profiling_wall_seconds")
    if type(profiling_seconds) not in (int, float) or not math.isfinite(profiling_seconds) or profiling_seconds < 0:
        raise ValueError("invalid profiling wall time")
    add("profiling_already_consumed", profiling_seconds, "this benchmark's measured wall time, charged once")
    add("decode_cache_once", batches(total_crops) * paths["drive_read_decode"]["p95_seconds"], "one shared warm read/canonicalization pass")
    add("lr_feature_cache", batches(total_crops) * paths["lr_features"]["p95_seconds"], "one cached feature pass")
    add("bigru_feature_cache", batches(total_crops) * paths["bigru_features"]["p95_seconds"], "one cached mel pass")
    fit_rows = benchmark["lr_fit_rows"]
    if type(fit_rows) is not int or fit_rows < 2:
        raise ValueError("invalid LR benchmark training rows")
    add("lr_fits", paths["lr_fit_grid"]["p95_seconds"] * train / fit_rows * (lr_fits / 8), "8-candidate grid timing; linear row-count extrapolation")
    add("lr_validation", lr_fits * batches(val) * paths["lr_scoring"]["p95_seconds"], "one validation pass per fit")
    epoch_seconds = steps_per_epoch * paths["bigru_train_step"]["p95_seconds"]
    add("bigru_training", bigru_fits * plan["epochs"] * epoch_seconds, "ceil(train/effective_batch) optimizer steps per epoch")
    add("bigru_validation", bigru_fits * validation_passes * batches(val) * paths["bigru_forward"]["p95_seconds"], "validation cadence including last epoch")
    add("lr_final_scoring", batches(scoring) * paths["lr_scoring"]["p95_seconds"], "one selected LR recording cohort pass")
    add("bigru_final_scoring", batches(scoring) * paths["bigru_forward"]["p95_seconds"], "one selected BiGRU recording cohort pass")
    loads = benchmark.get("loads", {})
    for model, fits in (("lr", lr_fits), ("bigru", bigru_fits)):
        load_seconds = loads.get(f"{model}_reload", {}).get("p95_seconds")
        if type(load_seconds) not in (int, float) or not math.isfinite(load_seconds) or load_seconds <= 0:
            raise ValueError(f"missing or invalid {model} reload measurement")
        cold_seconds = benchmark.get("cold_process_loads", {}).get(model, {}).get("wall_seconds")
        if type(cold_seconds) not in (int, float) or not math.isfinite(cold_seconds) or cold_seconds <= 0:
            raise ValueError(f"missing or invalid {model} cold-process measurement")
        add(f"{model}_reloads", (fits + 1) * max(load_seconds, cold_seconds), "one process/model load per fit plus final scoring; filesystem cache uncontrolled")
    write_seconds = loads.get("bigru_checkpoint_write", {}).get("p95_seconds")
    if type(write_seconds) not in (int, float) or not math.isfinite(write_seconds) or write_seconds <= 0:
        raise ValueError("missing or invalid checkpoint write measurement")
    add("bigru_checkpoint_writes", bigru_fits * plan["epochs"] * 2 * write_seconds, "upper bound of best and latest checkpoint writes each epoch")
    seconds = sum(row["estimated_seconds"] for row in rows)
    checkpoint_bytes = (bigru_fits * plan["bigru_checkpoint_copies_per_fit"] * benchmark["bigru_checkpoint_optimizer_bytes"] + lr_fits * benchmark["lr_pipeline_bytes"])
    storage = {"canonical_float32_bytes": math.ceil(plan["canonical_audio_seconds"] * 64000),
               "lr_cache_bytes": total_crops * benchmark["lr_feature_count"] * 8,
               "mel_cache_bytes": total_crops * benchmark["mel_frames"] * 64 * 4,
               "checkpoint_optimizer_bytes": checkpoint_bytes,
               "environment_bytes": math.ceil(plan["environment_overhead_gb"] * 1e9),
               "other_bytes": math.ceil(plan["other_storage_gb"] * 1e9),
               "additional_original_copy_bytes": math.ceil(plan["additional_original_copy_gb"] * 1e9)}
    compute_limit = plan["available_compute_hours"] * 3600 * 0.75
    storage_limit = plan["free_drive_gb"] * 1e9 * 0.75
    hardware_matches = plan["expected_bigru_device"] == benchmark["bigru_device_name"]
    fits_compute, fits_storage = seconds <= compute_limit, sum(storage.values()) <= storage_limit
    epoch_fits_session = epoch_seconds <= plan["max_session_hours"] * 3600 * 0.75
    row_costs = {row["component"]: row["estimated_seconds"] for row in rows}
    longest_job = max(epoch_seconds * plan["epochs"] + validation_passes * batches(val) * paths["bigru_forward"]["p95_seconds"] + plan["epochs"] * 2 * write_seconds,
                      row_costs["lr_fits"] / lr_fits,
                      *(row_costs[name] for name in ("decode_cache_once", "lr_feature_cache", "bigru_feature_cache", "lr_final_scoring", "bigru_final_scoring")))
    fits_session = longest_job <= plan["max_session_hours"] * 3600 * 0.75
    within = hardware_matches and fits_compute and fits_storage and fits_session
    result.update({"status": "provisional_within_capacity" if within else "needs_revision", "estimated_compute_hours": seconds / 3600,
                   "committable_compute_hours": compute_limit / 3600, "estimated_storage_bytes": sum(storage.values()), "storage_breakdown": storage,
                   "committable_storage_bytes": storage_limit, "steps_per_epoch": steps_per_epoch, "effective_batch": effective,
                   "epoch_seconds": epoch_seconds, "validation_passes_per_fit": validation_passes,
                   "fits": {"lr": lr_fits, "bigru": bigru_fits, "selected_final_models_reused": 2},
                   "longest_planned_job_seconds": longest_job,
                   "checks": {"hardware_matches": hardware_matches, "compute_fits": fits_compute, "storage_fits": fits_storage, "epoch_fits_session": epoch_fits_session, "longest_job_fits_session": fits_session},
                   "benchmark_provisional": benchmark.get("provisional", True),
                   "action": "Review assumptions, then reprofile on audited data before training." if within else "Reduce candidate cohort/trials, increase explicit capacity, or reprofile on the intended device.",
                   "session_note": "Longest job must fit 75% of a planned session; unexpected disconnections still require later checkpoint resume implementation."})
    return result


def budget_csv(report):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["component", "estimated_seconds", "basis"])
    writer.writeheader()
    writer.writerows(report["rows"])
    if not report["rows"]:
        writer.writerow({"component": "unresolved", "estimated_seconds": "", "basis": "Missing: " + ", ".join(report["missing_inputs"])})
    return stream.getvalue()


def run_budget(config, project_root, benchmark_file, plan_file):
    from .artifacts import atomic_write_new, create_run, digest, make_manifest, write_json
    benchmark = json.loads(Path(benchmark_file).read_text(encoding="utf-8"))
    plan = json.loads(Path(plan_file).read_text(encoding="utf-8"))
    report = estimate_budget(plan, benchmark)
    report.update({"benchmark_file": str(benchmark_file), "benchmark_hash": digest(benchmark)})
    run_id, folder = create_run(config.paths.run_root)
    report["run_id"] = run_id
    write_json(folder / "budget.json", report)
    atomic_write_new(folder / "compute_budget.csv", budget_csv(report).encode("utf-8"))
    write_json(folder / "artifact_manifest.json", make_manifest(folder, ["budget.json", "compute_budget.csv"]))
    write_json(folder / "status.json", {"status": report["status"], "run_id": run_id})
    summaries = Path(project_root) / "reports/phase1"
    summaries.mkdir(parents=True, exist_ok=True)
    if not (summaries / "compute_budget.csv").exists():
        atomic_write_new(summaries / "compute_budget.csv", budget_csv(report).encode("utf-8"))
    return {"status": report["status"], "run_dir": str(folder), "budget_file": str(folder / "budget.json"), "missing_inputs": report["missing_inputs"]}
