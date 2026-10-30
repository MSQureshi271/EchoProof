"""Run P1-07 setup, tests and fixture audio probes in one Colab runtime."""
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
        spec = importlib.util.spec_from_file_location("echoproof_p1_03_setup", root / "scripts/p1_03_colab_setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        setup, setup_folder = module.setup(root, allow_install=True)
        checks["setup"] = {"status": setup["status"], "issues": setup["issues"], "evidence_dir": str(setup_folder)}
    except Exception as exc:
        checks["setup"] = {"status": "needs_attention", "error": str(exc)}
    common = ["--project-root", str(root), "--config", "configs/phase1.yaml"]
    commands = {
        "tests": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_p1_07_audio.py", "-v"],
        "fixtures": [sys.executable, "-m", "echoproof.cli", "fixtures", "create", *common],
        "audio_probe": [sys.executable, "-m", "echoproof.cli", "probe", "audio", *common],
    }
    for name, command in commands.items():
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=300)
            checks[name] = {"argv": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            if name == "tests":
                checks[name]["skipped"] = "skipped=" in result.stderr or "... skipped" in result.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks[name] = {"argv": command, "returncode": None, "error": str(exc)}
    passed = (checks["setup"].get("status") == "ready_for_review" and
              all(checks[name].get("returncode") == 0 for name in commands) and not checks["tests"].get("skipped", True))
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = root / "reports/phase1/p1_07" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "task": "P1-07", "run_id": run_id, "python": sys.version,
              "status": "ready_for_review" if passed else "needs_attention", "checks": checks,
              "scope": "Generated software audio fixtures only; no research audio or model training"}
    data = json.dumps(report, indent=2) + "\n"
    (folder / "verification.json").write_text(data, encoding="utf-8")
    (folder / "verification.sha256").write_text(hashlib.sha256(data.encode("utf-8")).hexdigest() + "  verification.json\n", encoding="utf-8")
    print("P1-07 status:", report["status"])
    print("Evidence:", folder)
    if not passed:
        print(json.dumps(checks, indent=2))
    return report, folder
