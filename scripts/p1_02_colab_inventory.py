"""P1-02 evidence collector. Standard library plus optional existing torch/numpy.

The Colab notebook embeds this module so it can run without a package install.
No datasets or pretrained models are downloaded or modified.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid


def command(args, timeout=30):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout.strip()[:16000],
                "stderr": p.stderr.strip()[:4000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_inventory(folder, extension):
    result = {"path": str(folder), "exists": folder.is_dir(), "files": 0,
              "bytes": 0, "nested_directories": 0, "errors": [], "stems": collections.Counter()}
    if not result["exists"]:
        return result
    try:
        with os.scandir(folder) as entries:
            for item in entries:
                try:
                    if item.is_symlink():
                        result["errors"].append("Skipped symbolic link: " + item.name)
                    elif item.is_dir():
                        result["nested_directories"] += 1
                    elif Path(item.name).suffix.lower() == extension:
                        result["files"] += 1
                        result["bytes"] += item.stat().st_size
                        result["stems"][Path(item.name).stem] += 1
                except OSError as exc:
                    result["errors"].append(str(exc))
    except OSError as exc:
        result["errors"].append(str(exc))
    return result


def dataset_inventory(project):
    folders = {"genuine": (project / "dataset/genuine/wav", ".wav"),
               "transcripts": (project / "dataset/genuine/transcripts", ".txt"),
               "tts_xttsv2": (project / "dataset/tts/XTTSv2", ".wav")}
    scans = {key: file_inventory(path, ext) for key, (path, ext) in folders.items()}
    sets = {key: set(value["stems"]) for key, value in scans.items()}
    common = set.intersection(*sets.values())
    union = set.union(*sets.values())
    missing = {key: {"count": len(union - stems), "sample": sorted(union - stems)[:10]}
               for key, stems in sets.items()}
    for scan in scans.values():
        counter = scan.pop("stems")
        scan["duplicate_stems"] = sorted(k for k, n in counter.items() if n > 1)[:10]
        scan["has_duplicate_stems"] = any(n > 1 for n in counter.values())
    structurally_ready = bool(common) and all(
        scan["exists"] and not scan["errors"] and not scan["nested_directories"]
        and not scan["has_duplicate_stems"] for scan in scans.values()
    ) and all(item["count"] == 0 for item in missing.values())
    return {"folders": scans, "complete_stem_triples": len(common),
            "paired_sample": sorted(common)[:3], "missing_by_folder": missing,
            "observed_bytes": sum(s["bytes"] for s in scans.values()),
            "structural_check_passed": structurally_ready,
            "scope": "Flat-directory filenames and sizes only; labels, speech, language, speaker IDs and licenses not audited.",
            "visible_training_family": "XTTSv2; other families have not been audited",
            "unseen_generator_protocol": "pending additional verified families and protocol revision"}


def hardware_inventory(scratch):
    cpu_model = None
    mem_total = None
    mem_available = None
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    if Path("/proc/meminfo").exists():
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        mem_total = int(memory["MemTotal"].split()[0]) * 1024
        mem_available = int(memory["MemAvailable"].split()[0]) * 1024
    disk = shutil.disk_usage(scratch)
    packages = {}
    for name in ["pip", "torch", "numpy", "pandas", "pyarrow", "scipy", "scikit-learn",
                 "librosa", "soundfile", "matplotlib", "seaborn", "PyYAML", "pytest",
                 "transformers", "huggingface-hub", "streamlit"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"platform": platform.platform(), "architecture": platform.machine(),
            "python": sys.version, "python_executable": sys.executable,
            "cpu_model": cpu_model, "logical_cpu_count": os.cpu_count(),
            "ram_total_bytes": mem_total, "ram_available_bytes": mem_available,
            "scratch_disk": {"total_bytes": disk.total, "free_bytes": disk.free},
            "packages": packages,
            "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]),
            "ffmpeg": command(["ffmpeg", "-version"]),
            "pip_check": command([sys.executable, "-m", "pip", "check"], timeout=120)}


def tensor_probe():
    result = {"purpose": "Disposable runtime arithmetic/backward check, not audio training or performance evaluation."}
    try:
        import numpy as np
        value = float(np.dot(np.array([1., 2.]), np.array([3., 4.])))
        result["numpy"] = {"ok": value == 11., "dot_result": value}
    except Exception as exc:
        result["numpy"] = {"ok": False, "error": str(exc)}
    try:
        import torch
        result["torch"] = {"ok": False, "version": torch.__version__,
                           "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(),
                           "threads": torch.get_num_threads()}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            result["torch"]["gpu_name"] = torch.cuda.get_device_name(0)
        # fork_rng restores caller RNG state after this disposable check.
        with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
            torch.manual_seed(42)
            layer = torch.nn.Linear(8, 2).to(device)
            x = torch.ones(2, 8, device=device)
            y = layer(x)
            loss = y.square().mean()
            loss.backward()
            ok = bool(torch.isfinite(y).all().item() and
                      all(p.grad is not None and torch.isfinite(p.grad).all().item()
                          for p in layer.parameters()))
            if device == "cuda":
                torch.cuda.synchronize()
            result["torch"].update(ok=ok, device=device, output_shape=list(y.shape),
                                   loss=float(loss.detach().cpu()))
    except Exception as exc:
        result.setdefault("torch", {})["ok"] = False
        result["torch"]["error"] = str(exc)
    return result


def positive_optional(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value <= 0):
        raise ValueError(name + " must be a finite positive number or None")
    return value


def run_inventory(project_root, scratch_root, *, request_gpu=False, drive_free_gib=None,
                  extra_storage_budget_gib=None, planned_work_hours=None):
    project = Path(project_root).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not project.is_dir() or not (project / "IMPLEMENTATION_PLAN.md").is_file():
        raise ValueError("Set PROJECT_ROOT to the copied project folder containing IMPLEMENTATION_PLAN.md")
    # Keep temporary space separate from the original project/dataset.
    if scratch == project or project in scratch.parents or scratch in project.parents:
        raise ValueError("SCRATCH_ROOT must be separate from the persistent project")
    for key, value in [("drive_free_gib", drive_free_gib),
                       ("extra_storage_budget_gib", extra_storage_budget_gib),
                       ("planned_work_hours", planned_work_hours)]:
        positive_optional(value, key)
    scratch.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = stamp + "-" + uuid.uuid4().hex[:8]
    output = project / "reports/phase1/p1_02" / run_id
    output.mkdir(parents=True, exist_ok=False)
    hardware = hardware_inventory(scratch)
    tensors = tensor_probe()
    dataset = dataset_inventory(project)

    probe_bytes = ("EchoProof persistence probe " + run_id).encode()
    marker = output / "persistence_probe.txt"
    atomic_text(marker, probe_bytes.decode())
    expected_hash = hashlib.sha256(probe_bytes).hexdigest()
    child = command([sys.executable, "-c",
                     "import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())",
                     str(marker)])
    persistence_ok = child["returncode"] == 0 and child["stdout"] == expected_hash
    storage = {"project_root": str(project), "raw_input": str(project / "dataset"),
               "scratch_root": str(scratch), "run_root": str(project / "runs"),
               "manifests": str(project / "data/manifests"),
               "features": str(project / "data/features"),
               "bundles": str(project / "bundles"),
               "future_model_snapshots": str(project / "model_assets"),
               "evidence_root": str(project / "reports/phase1"),
               "drive_free_gib_user_reported": drive_free_gib,
               "extra_storage_budget_gib_user_selected": extra_storage_budget_gib,
               "quota_note": "Mounted filesystem disk usage is not Google Drive account quota.",
               "fresh_process_read_verified": persistence_ok,
               "restart_verified": False,
               "probe_relative_path": str(marker.relative_to(project)),
               "probe_sha256": expected_hash}
    capacity = {"planned_work_hours_user_selected": planned_work_hours,
                "committed_hours_ceiling": planned_work_hours * .75 if planned_work_hours else None,
                "reserve_hours": planned_work_hours * .25 if planned_work_hours else None,
                "session_limit_hours": None, "gpu_access_guaranteed": False,
                "measured_training_throughput": None,
                "stage": "Colab LR/BiGRU first; dedicated GPU conditional on development review"}
    issues = []
    if not dataset["structural_check_passed"]:
        issues.append("Dataset directories, stem pairing or enumeration need review before data admission.")
    if not persistence_ok:
        issues.append("Fresh-process persistence check failed.")
    if not all(tensors.get(k, {}).get("ok") for k in ["numpy", "torch"]):
        issues.append("Existing numpy/torch imports or arithmetic failed; resolve in P1-03.")
    if request_gpu and not tensors.get("torch", {}).get("cuda_available", False):
        issues.append("GPU requested but torch CUDA unavailable; retry GPU runtime or explicitly plan CPU pilot.")
    if hardware["pip_check"]["returncode"] != 0:
        issues.append("pip check did not pass; inspect captured diagnostics before dependency installation.")
    if (drive_free_gib is not None and extra_storage_budget_gib is not None
            and extra_storage_budget_gib > drive_free_gib):
        issues.append("Chosen additional storage allowance exceeds reported remaining Drive quota.")
    missing_packages = [name for name in ["numpy", "torch", "pandas", "pyarrow", "scipy",
                        "scikit-learn", "librosa", "soundfile", "PyYAML", "pytest"]
                        if hardware["packages"][name] is None]
    report = {"schema_version": 1, "task": "P1-02", "run_id": run_id,
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "evidence_kind": "runtime_inventory", "hardware": hardware,
              "runtime_probe": tensors, "dataset": dataset, "storage": storage,
              "capacity": capacity, "missing_baseline_packages_for_p1_03": missing_packages,
              "issues": issues, "status": "needs_attention" if issues else "collected_pending_review",
              "human_reviewed": False, "demo_device": None,
              "pending": ["P1-03 dependency compatibility and lock",
                          "P1-12 measured training budget", "Drive quota/session budget if not supplied",
                          "Fresh-runtime persistence confirmation", "Named demo device",
                          "Dataset content/provenance audit and independent reviewer disposition"],
              "artifact_scope": "No audio decoded, transcripts opened, dataset split or model trained."}
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    atomic_text(output / "hardware.json", payload)
    device = tensors.get("torch", {}).get("device", "unavailable")
    summary = f"""# EchoProof Colab environment — P1-02

Run: `{run_id}` · Status: **{report['status']}**, pending U/M review.

- Project: `{project}`
- Python: `{platform.python_version()}`; use the actual Colab kernel; dependency validation is P1-03.
- Disposable arithmetic device: `{device}`; requested GPU: `{request_gpu}`.
- Dataset matched filename triples: {dataset['complete_stem_triples']}; structural check: {dataset['structural_check_passed']}.
- Raw input stays in `{project / 'dataset'}`; no content audit performed.
- Temporary workspace: `{scratch}`; persistent reports/checkpoints/features use project-relative Drive locations.
- Separate-process Drive readback: {persistence_ok}; fresh-runtime verification is still pending.
- Drive free quota entered by user (GiB): {drive_free_gib}; planned extra allowance (GiB): {extra_storage_budget_gib}.
- Planned work hours (not a guaranteed Colab session): {planned_work_hours}; reserve: 25%.
- Demo hardware is unselected; this runtime does not establish the five-second app target.

## Issues

{chr(10).join('- ' + issue for issue in issues) or '- No immediate inventory issue detected; later checks remain pending.'}

## Follow-up

Inspect `hardware.json` in `reports/phase1/p1_02/{run_id}/` for package versions, missing packages, command errors and pending items. This is an inventory, not full G1 or baseline training completion. Resolve reported issues before proceeding with affected work. Existing torch/numpy were tested without installation changes. Advanced models are deferred under the baseline-first amendment.
"""
    atomic_text(output / "environment.md", summary)
    # First-run canonical summaries only; never overwrite existing reports or human notes.
    canonical = {}
    for target, content in [(project / "reports/phase1/hardware.json", payload),
                            (project / "docs/environment.md", summary)]:
        if target.exists():
            canonical[str(target)] = "preserved existing; use this run's versioned snapshot"
        else:
            atomic_text(target, content)
            canonical[str(target)] = "created"
    manifest = {"run_id": run_id, "canonical_summaries": canonical,
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [output / "hardware.json", output / "environment.md", marker]}}
    atomic_text(output / "artifact_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"run_id": run_id, "status": report["status"], "device": device,
                      "matched_stems": dataset["complete_stem_triples"], "issues": issues,
                      "evidence_directory": str(output), "canonical_summaries": canonical}, indent=2))
    return report, output


def verify_saved_run(project_root, relative_run):
    """Read-only verification usable after a fresh Colab runtime remount."""
    project = Path(project_root).resolve()
    folder = (project / relative_run).resolve()
    if project not in folder.parents:
        raise ValueError("Run directory must stay inside project")
    manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (folder / name).resolve()
        if path.parent != folder or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Checksum verification failed: " + name)
    return {"run_id": manifest["run_id"], "all_saved_hashes_match": True,
            "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
