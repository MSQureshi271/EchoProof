"""Bootstrap the Colab environment and record P1-05 contract checks."""
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
    checks = {}
    try:
        script = root / "scripts/p1_03_colab_setup.py"
        spec = importlib.util.spec_from_file_location("echoproof_p1_03_setup", script)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Missing setup script: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        setup_report, setup_path = module.setup(root, allow_install=True)
        checks["setup"] = {"status": setup_report["status"], "issues": setup_report["issues"], "evidence_dir": str(setup_path)}
    except Exception as exc:
        checks["setup"] = {"status": "needs_attention", "error": str(exc)}

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_p1_05_contracts.py", "-v"]
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180)
        checks["contracts"] = {"argv": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                               "skipped": "skipped=" in result.stderr or "... skipped" in result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks["contracts"] = {"argv": command, "returncode": None, "error": str(exc), "skipped": True}

    passed = checks["setup"]["status"] == "ready_for_review" and checks["contracts"]["returncode"] == 0 and not checks["contracts"]["skipped"]
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = root / "reports/phase1/p1_05" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {"task": "P1-05", "run_id": run_id, "python": sys.version,
              "status": "ready_for_review" if passed else "needs_attention", "checks": checks,
              "scope": "Contract validation with fabricated metadata; no research audio or training"}
    data = json.dumps(report, indent=2) + "\n"
    (folder / "verification.json").write_text(data, encoding="utf-8")
    (folder / "verification.sha256").write_text(hashlib.sha256(data.encode("utf-8")).hexdigest() + "  verification.json\n", encoding="utf-8")
    print("P1-05 status:", report["status"])
    print("Evidence:", folder)
    if not passed:
        print(json.dumps(checks, indent=2))
    return report, folder
