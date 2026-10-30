"""Bootstrap P1-03 prerequisites, verify P1-04, and save Colab evidence."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid


def verify(project_root):
    root = Path(project_root).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("Project root must contain pyproject.toml")
    checks = {}
    # P1-04 notebook is independently runnable. Reuse P1-03's guarded installer
    # so jedi and the editable package are present in this kernel's interpreter.
    setup_path = root / "scripts/p1_03_colab_setup.py"
    try:
        if not setup_path.is_file():
            raise FileNotFoundError("Sync scripts/p1_03_colab_setup.py to the Drive project")
        spec = importlib.util.spec_from_file_location("echoproof_p1_03_setup", setup_path)
        setup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup_module)
        setup_report, setup_folder = setup_module.setup(root, allow_install=True)
        checks["p1_03_setup"] = {"status": setup_report["status"], "issues": setup_report["issues"],
                                 "evidence_dir": str(setup_folder), "torch_before": setup_report["before"].get("torch_distribution"),
                                 "torch_after": setup_report.get("torch_distribution_after")}
    except Exception as exc:
        checks["p1_03_setup"] = {"status": "needs_attention", "error": str(exc)}

    commands = {
        "validate_config": [sys.executable, "-m", "echoproof.cli", "validate-config", "--project-root", str(root), "--config", "configs/phase1.yaml"],
        "tests": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_p1_04_config.py", "-v"],
        "pip_check": [sys.executable, "-m", "pip", "check"],
    }
    for name, command in commands.items():
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180)
            checks[name] = {"argv": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks[name] = {"argv": command, "returncode": None, "error": str(exc)}
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = root / "reports/phase1/p1_04" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    success = checks.get("p1_03_setup", {}).get("status") == "ready_for_review" and all(
        checks.get(name, {}).get("returncode") == 0 for name in commands)
    report = {"task": "P1-04", "run_id": run_id, "python": sys.version, "checks": checks,
              "status": "ready_for_review" if success else "needs_attention",
              "scope": "Configuration and CLI only; no dataset audit or model training"}
    data = json.dumps(report, indent=2) + "\n"
    (folder / "verification.json").write_text(data, encoding="utf-8")
    (folder / "verification.sha256").write_text(hashlib.sha256(data.encode("utf-8")).hexdigest() + "  verification.json\n", encoding="utf-8")
    print("P1-04 status:", report["status"])
    print("Evidence:", folder)
    for name, result in checks.items():
        print(name, "status/exit:", result.get("status", result.get("returncode")))
        if result.get("status") not in (None, "ready_for_review") or result.get("returncode") != 0:
            print(result.get("stderr") or result.get("error") or result.get("stdout"))
    return report, folder
