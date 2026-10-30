"""P1-13 export/recreation orchestrator. Uses subprocesses so kernel imports stay clean."""
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def _module(root):
    # Loading the shared stdlib helpers does not require an editable installation.
    import importlib.util
    root = Path(root).resolve()
    package = "echoproof_p113_helpers"
    spec = importlib.util.spec_from_file_location(package, root / "src/echoproof/__init__.py",
                                                submodule_search_locations=[str(root / "src/echoproof")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    import importlib
    return importlib.import_module(package + ".environment_lock")


def _run(root, argv, timeout=1200, source=False, lock=None):
    env = os.environ.copy()
    if source:
        env["PYTHONPATH"] = str(root / "src")
    # Preserve deterministic CUDA settings before any torch subprocess import.
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if lock:
        env["ECHOPROOF_DEPENDENCY_LOCK"] = str(lock)
    print("Running:", " ".join(map(str, argv)), flush=True)
    try:
        process = subprocess.run(list(map(str, argv)), cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
        return {"argv": list(map(str, argv)), "returncode": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": list(map(str, argv)), "returncode": None, "error": str(exc)}


def _folder(root, stage):
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = root / "reports/phase1/p1_13" / (stage + "-" + run_id)
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _save(folder, report):
    data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    (folder / "verification.json").write_bytes(data)
    (folder / "verification.sha256").write_text(hashlib.sha256(data).hexdigest() + "  verification.json\n", encoding="utf-8")
    print("P1-13 status:", report["status"], "\nEvidence:", folder)
    return report, folder


def export(project_root):
    root = Path(project_root).resolve()
    folder = _folder(root, "export")
    lock_dir = root / "locks" / ("colab-baseline-" + folder.name.removeprefix("export-"))
    checks = {}
    report = {"schema_version": 1, "task": "P1-13", "stage": "export", "checks": checks,
              "status": "needs_attention",
              "export_dir": str(lock_dir), "scope": "baseline; fresh-runtime reproduction pending"}
    try:
        def version(name):
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None
        # Mirror P1-03's narrow Colab repair before freezing the package state.
        # Stage A may be opened in a runtime that has not run the setup notebook.
        torch_before = version("torch")
        checks["jedi_before"] = version("jedi")
        if checks["jedi_before"] is None:
            checks["repair_jedi"] = _run(root, [sys.executable, "-m", "pip", "install", "--index-url",
                                               "https://pypi.org/simple", "jedi"])
            if checks["repair_jedi"]["returncode"] != 0:
                return _save(folder, report)
        checks["torch_version_unchanged"] = version("torch") == torch_before
        if not checks["torch_version_unchanged"]:
            report["error"] = "PyTorch version changed during the jedi repair; review before export"
            return _save(folder, report)
        checks["pip_check"] = _run(root, [sys.executable, "-m", "pip", "check"])
        if checks["pip_check"]["returncode"] != 0:
            return _save(folder, report)
        checks["export"] = _run(root, [sys.executable, "-m", "echoproof.cli", "export-environment",
                                     "--project-root", root, "--output", lock_dir], source=True)
        if checks["export"]["returncode"] == 0:
            report["status"] = "exported_for_reproduction"
    except Exception as exc:
        report["error"] = str(exc)
    return _save(folder, report)


def reproduce(project_root, export_dir, install=True):
    root = Path(project_root).resolve()
    helpers = _module(root)
    folder = _folder(root, "reproduce")
    checks = {}
    report = {"schema_version": 1, "task": "P1-13", "stage": "reproduce", "status": "needs_attention",
              "checks": checks, "scope": "baseline fixtures; pretrained/full G1 pending"}
    def finish():
        return _save(folder, report)
    try:
        export_path = Path(export_dir)
        if not export_path.is_absolute():
            export_path = root / export_path
        exported, pins, expected = helpers.inspect_export(export_path)
        report.update(export_dir=str(exported), lock_sha256=expected["dependency_lock"])
        # Fail before package mutations if the runtime, source or historical files differ.
        preflight = {"fresh_runtime": bool(helpers.boot_id() and helpers.boot_id() != expected["boot_id"]),
                     "python_matches": __import__("platform").python_version() == expected["python_version"],
                     "platform_matches": __import__("platform").system() == expected["system"] and __import__("platform").machine() == expected["machine"],
                     "sources": helpers.verify_files(root, json.loads((exported / "source_files.json").read_text())),
                     "recovery": helpers.verify_files(root, json.loads((exported / "recovery_files.json").read_text()))}
        checks["preflight"] = preflight
        if not preflight["fresh_runtime"] or not preflight["python_matches"] or not preflight["platform_matches"] or preflight["sources"]["mismatches"] or preflight["recovery"]["mismatches"]:
            return finish()
        if install:
            # Install only differing exact pins; retain an already matching Colab torch build.
            def differs(name, version):
                try:
                    return metadata.version(name) != version
                except metadata.PackageNotFoundError:
                    return True
            if differs("torch", pins["torch"]):
                origin = json.loads((exported / "torch_source.json").read_text())
                command = [sys.executable, "-m", "pip", "install", "--no-deps", "--index-url", origin["reinstall_index"], "torch==" + pins["torch"]]
                checks["install_torch"] = _run(root, command, timeout=2400)
                if checks["install_torch"]["returncode"] != 0:
                    return finish()
            other = [f"{name}=={version}" for name, version in pins.items() if name != "torch" and differs(name, version)]
            if other:
                requirements = folder / "requirements-to-install.txt"
                requirements.write_text("\n".join(other) + "\n", encoding="utf-8")
                checks["install_pins"] = _run(root, [sys.executable, "-m", "pip", "install", "--no-deps", "--index-url", "https://pypi.org/simple", "-r", requirements], timeout=2400)
                if checks["install_pins"]["returncode"] != 0:
                    return finish()
            checks["install_project"] = _run(root, [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."])
            if checks["install_project"]["returncode"] != 0:
                return finish()
        common = ["--project-root", root, "--config", "configs/phase1.yaml"]
        commands = {"environment": [sys.executable, "-m", "echoproof.cli", "verify-environment", "--project-root", root, "--lock", exported / "requirements-lock.txt"],
                    "pip_check": [sys.executable, "-m", "pip", "check"],
                    "installed_cli": [sys.executable, "-m", "echoproof.cli", "--version"],
                    "validate_config": [sys.executable, "-m", "echoproof.cli", "validate-config", *common]}
        for task in ("04_config", "05_contracts", "06_reproducibility", "07_audio", "08_models", "12_budget", "13_environment"):
            commands["tests_" + task] = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_p1_" + task + ".py", "-v"]
        commands.update({"storage_probe": [sys.executable, "-m", "echoproof.cli", "storage-probe", *common],
                         **{name + "_probe": [sys.executable, "-m", "echoproof.cli", "probe", name, *common] for name in ("audio", "lr", "bigru")}})
        cuda_code = """
import json
import torch
from echoproof.models.bigru import BiGRU
from echoproof.training.seeding import seed_everything
seed_everything(42)
torch.set_num_threads(1)
assert torch.cuda.is_available(), 'Assigned CUDA GPU is required'
model = BiGRU().cuda().train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(4, 397, 64, device='cuda')
lengths = torch.tensor([397, 300, 200, 100])
before = model.head.weight.detach().clone()
logits = model(x, lengths)
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1], device='cuda'))
assert logits.shape == (4, 2) and torch.isfinite(loss)
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
optimizer.step()
torch.cuda.synchronize()
assert not torch.equal(before, model.head.weight)
print(json.dumps({'scope':'generated_tensor_compatibility', 'device':torch.cuda.get_device_name(0),
                  'logit_shape':list(logits.shape), 'loss':loss.item(), 'finite_gradients':True, 'head_updated':True}))
"""
        commands["cuda_bigru_step"] = [sys.executable, "-c", cuda_code]
        for name, command in commands.items():
            checks[name] = _run(root, command, lock=exported / "requirements-lock.txt")
            if name.startswith("tests_"):
                log = checks[name].get("stderr", "") + checks[name].get("stdout", "")
                checks[name]["skipped"] = "skipped=" in log or "... skipped" in log
                checks[name]["tests_executed"] = bool(__import__("re").search(r"Ran [1-9]\d* tests?", log))
            if name == "environment" and checks[name]["returncode"] == 0:
                report["environment_verification"] = json.loads(checks[name]["stdout"])
        passed = all(checks[name]["returncode"] == 0 and (not name.startswith("tests_") or (not checks[name]["skipped"] and checks[name]["tests_executed"])) for name in commands)
        # Recheck the lock/runtime and source after all probes, not only before them.
        checks["environment_after"] = _run(root, commands["environment"], lock=exported / "requirements-lock.txt")
        passed = passed and checks["environment_after"]["returncode"] == 0
        report["status"] = "ready_for_review" if passed else "needs_attention"
        report["probe_device"] = "CPU fixture parity plus CUDA BiGRU forward/backward/AdamW step on generated tensors"
        report["limitations"] = ["Fresh-runtime artifact readback is verified; interrupted training/optimizer continuation is later work.",
                                 "CPU fixture parity does not establish CUDA training determinism or research performance.",
                                 "Wheel hashes and offline package installation are not established by version pins."]
    except Exception as exc:
        report["error"] = str(exc)
    return finish()
