"""Run baseline profiling and explicit capacity planning in one Colab notebook."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid


def verify(project_root, *, plan_overrides=None, benchmark_file=None, device="auto", batch_size=4, accumulation_steps=1):
    root = Path(project_root).resolve()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = root / "reports/phase1/p1_12" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    checks = {}
    def command(name, argv, timeout=9000):
        print("Running", name, "...", flush=True)
        try:
            result = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout)
            checks[name] = {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            return result
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks[name] = {"argv": argv, "returncode": None, "error": str(exc)}
            return None
    try:
        spec = importlib.util.spec_from_file_location("echoproof_p1_03_setup", root / "scripts/p1_03_colab_setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        setup, setup_path = module.setup(root, allow_install=True)
        checks["setup"] = {"status": setup["status"], "issues": setup["issues"], "evidence_dir": str(setup_path)}
    except Exception as exc:
        checks["setup"] = {"status": "needs_attention", "error": str(exc)}
    if importlib.util.find_spec("psutil") is None:
        command("install_profiling_dependency", [sys.executable, "-m", "pip", "install", "psutil"], timeout=180)
    tests = command("tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_p1_12_budget.py", "-v"])
    if tests:
        checks["tests"]["skipped"] = "skipped=" in tests.stderr or "... skipped" in tests.stderr
    common = ["--config", "configs/phase1.yaml", "--project-root", str(root)]
    budget_result = None
    benchmark_data = None
    try:
        if benchmark_file is None:
            timed = command("benchmark", [sys.executable, "-m", "echoproof.cli", "benchmark", *common,
                             "--models", "lr", "bigru", "--device", device, "--batch-size", str(batch_size),
                             "--accumulation-steps", str(accumulation_steps)])
            if timed is None or timed.returncode != 0:
                raise RuntimeError("benchmark failed; inspect its saved output")
            summary = json.loads(timed.stdout)
            benchmark_file = summary["benchmark_file"]
        else:
            checks["benchmark"] = {"returncode": 0, "reused_file": str(benchmark_file),
                                   "note": "Operator-selected prior timings; verify that the intended hardware and settings still match."}
        benchmark_data = json.loads(Path(benchmark_file).read_text(encoding="utf-8"))
        plan = json.loads((root / "configs/baseline_budget.json").read_text(encoding="utf-8"))
        plan.update(plan_overrides or {})
        if plan.get("expected_bigru_device") is None:
            plan["expected_bigru_device"] = benchmark_data["bigru_device_name"]
            checks["observed_device_assumption"] = plan["expected_bigru_device"]
        plan_path = folder / "budget_inputs.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        budget_run = command("budget", [sys.executable, "-m", "echoproof.cli", "budget", *common,
                             "--benchmark", str(benchmark_file), "--plan", str(plan_path)])
        if budget_run is not None and budget_run.returncode == 0:
            budget_summary = json.loads(budget_run.stdout)
            checks["budget"]["result"] = budget_summary
            budget_result = json.loads(Path(budget_summary["budget_file"]).read_text(encoding="utf-8"))
    except Exception as exc:
        checks["planning_error"] = str(exc)
    tests_ok = checks.get("tests", {}).get("returncode") == 0 and not checks.get("tests", {}).get("skipped", True)
    status = "needs_attention"
    if checks["setup"].get("status") == "ready_for_review" and tests_ok and budget_result:
        if budget_result["status"] == "needs_inputs":
            status = "needs_budget_inputs"
        elif budget_result["status"] == "needs_revision":
            status = "needs_budget_revision"
        elif benchmark_data.get("truncated"):
            status = "provisional_measurements"
        elif budget_result["status"] == "provisional_within_capacity":
            status = "ready_for_review"
    report = {"schema_version": 1, "task": "P1-12", "run_id": run_id, "python": sys.version,
              "status": status, "checks": checks, "benchmark_file": str(benchmark_file) if benchmark_file else None,
              "capacity_status": budget_result["status"] if budget_result else None,
              "scope": "Baseline-only provisional engineering budget; pretrained budgets and real-data reprofile remain pending"}
    data = json.dumps(report, indent=2) + "\n"
    (folder / "verification.json").write_text(data, encoding="utf-8")
    (folder / "verification.sha256").write_text(hashlib.sha256(data.encode("utf-8")).hexdigest() + "  verification.json\n", encoding="utf-8")
    print("P1-12 status:", status)
    print("Evidence:", folder)
    if budget_result:
        print("Budget status:", budget_result["status"])
        print("Estimated hours:", budget_result.get("estimated_compute_hours"))
        print("Estimated additional storage GB:", budget_result.get("estimated_storage_bytes", 0) / 1e9 if "estimated_storage_bytes" in budget_result else None)
        print("Missing inputs:", budget_result["missing_inputs"])
    if status == "needs_attention":
        print(json.dumps(checks, indent=2))
    return report, folder
