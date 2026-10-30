"""Budget limits, accumulation arithmetic and benchmark sampling checks."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

from echoproof.budget import budget_csv, estimate_budget, validate_plan

ROOT = Path(__file__).resolve().parents[1]


def complete_plan():
    plan = json.loads((ROOT / "configs/baseline_budget.json").read_text(encoding="utf-8"))
    plan.update(train_recordings=17, validation_recordings=5, scoring_recordings=5,
                train_crops=17, validation_crops=5, scoring_crops=5, canonical_audio_seconds=108,
                epochs=3, validate_every_epochs=2, available_compute_hours=3, max_session_hours=3,
                free_drive_gb=10, expected_bigru_device="cpu", environment_overhead_gb=1, other_storage_gb=0.5)
    return plan


def measured_benchmark():
    paths = {name: {"p95_seconds": 1.0, "measured_count": 100} for name in
             ("drive_read_decode", "lr_features", "lr_fit_grid", "lr_scoring", "bigru_features", "bigru_forward", "bigru_train_step")}
    paths["bigru_train_step"]["p95_seconds"] = 2.0
    return {"schema_version": 1, "scope": "baseline_fixture_benchmark", "paths": paths,
            "batch_size": 4, "accumulation_steps": 2, "lr_fit_rows": 256, "bigru_device_name": "cpu",
            "lr_feature_count": 67, "mel_frames": 397, "bigru_checkpoint_optimizer_bytes": 1000,
            "lr_pipeline_bytes": 100, "provisional": True,
            "profiling_wall_seconds": 10,
            "cold_process_loads": {"lr": {"wall_seconds": 1}, "bigru": {"wall_seconds": 1}},
            "loads": {"lr_reload": {"p95_seconds": 0.1}, "bigru_reload": {"p95_seconds": 0.1}, "bigru_checkpoint_write": {"p95_seconds": 0.1}}}


class BudgetTests(unittest.TestCase):
    def test_accumulation_epochs_validation_and_reserved_capacity(self):
        report = estimate_budget(complete_plan(), measured_benchmark())
        self.assertEqual(report["steps_per_epoch"], 3)
        self.assertEqual(report["effective_batch"], 8)
        rows = {r["component"]: r["estimated_seconds"] for r in report["rows"]}
        self.assertEqual(rows["bigru_training"], 36)
        self.assertEqual(rows["bigru_validation"], 8)
        self.assertEqual(report["committable_compute_hours"], 2.25)
        self.assertEqual(report["status"], "provisional_within_capacity")
        self.assertEqual(len(report["advanced_scope"]["future_final_fits"]), 8)

    def test_unknown_inputs_are_not_zero_and_csv_explains(self):
        plan = complete_plan()
        plan["available_compute_hours"] = None
        report = estimate_budget(plan, measured_benchmark())
        self.assertEqual(report["status"], "needs_inputs")
        self.assertIn("available_compute_hours", report["missing_inputs"])
        self.assertNotIn("estimated_compute_hours", report)
        self.assertIn("unresolved", budget_csv(report))

    def test_over_capacity_and_wrong_hardware_require_revision(self):
        for key, value in (("available_compute_hours", 0.0001), ("free_drive_gb", 0.001), ("expected_bigru_device", "different_gpu")):
            plan = complete_plan()
            plan[key] = value
            report = estimate_budget(plan, measured_benchmark())
            self.assertEqual(report["status"], "needs_revision")

    def test_invalid_inputs_and_measurements_rejected(self):
        for key, value in (("epochs", 1.5), ("free_drive_gb", float("nan")), ("extra_bigru_seed_fits", -1)):
            plan = complete_plan()
            plan[key] = value
            with self.assertRaises(ValueError):
                validate_plan(plan)
        benchmark = measured_benchmark()
        benchmark["paths"]["bigru_train_step"]["p95_seconds"] = float("inf")
        with self.assertRaises(ValueError):
            estimate_budget(complete_plan(), benchmark)

    def test_extra_trials_and_float32_storage_are_counted(self):
        plan = complete_plan()
        plan.update(extra_bigru_seed_fits=1, failed_bigru_fit_equivalents=1, final_lr_refits=1)
        report = estimate_budget(plan, measured_benchmark())
        self.assertEqual(report["fits"]["bigru"], 4)
        self.assertEqual(report["fits"]["lr"], 9)
        self.assertEqual(report["storage_breakdown"]["canonical_float32_bytes"], 108 * 64000)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "numpy", "psutil")), "profiling dependencies unavailable")
    def test_timing_records_actual_counts_and_truncation(self):
        import time
        import torch
        from echoproof.benchmark import _measure
        report = _measure(lambda: time.sleep(0.001), warmup=1, count=3, limit=2, device=torch.device("cpu"), samples_per_iteration=4)
        self.assertEqual(report["measured_count"], 3)
        self.assertGreater(report["p95_seconds"], 0)
        self.assertFalse(report["truncated"])
        truncated = _measure(lambda: None, warmup=1, count=3, limit=1e-12, device=torch.device("cpu"), samples_per_iteration=4)
        self.assertEqual(truncated["measured_count"], 0)
        self.assertTrue(truncated["truncated"])


if __name__ == "__main__":
    unittest.main()
