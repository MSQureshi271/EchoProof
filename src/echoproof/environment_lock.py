"""Platform-specific baseline locks and provenance; no scientific imports at module load."""
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit

from .artifacts import atomic_write_new, file_digest, make_manifest, verify_manifest, write_json

ROOT_PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "librosa", "soundfile",
                 "PyYAML", "torch", "joblib", "psutil", "pip", "setuptools", "wheel", "jedi")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def boot_id():
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.is_file() else None


def torch_distribution_provenance():
    import hashlib
    dist = metadata.distribution("torch")
    return {"installer": (dist.read_text("INSTALLER") or "unknown").strip(),
            "original_official_wheel": official_torch_origin(dist),
            "metadata_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name in ("METADATA", "WHEEL", "RECORD")
                                if (text := dist.read_text(name)) is not None}}


def official_torch_origin(dist):
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    value = json.loads(raw)
    url = urlsplit(value.get("url", ""))
    if (url.scheme != "https" or url.hostname not in ("download.pytorch.org", "download-r2.pytorch.org")
            or url.username or url.password or url.query or url.fragment or not url.path.endswith(".whl")
            or "archive_info" not in value):
        return None
    archive = value["archive_info"]
    sha256 = archive.get("hashes", {}).get("sha256")
    if not sha256 and str(archive.get("hash", "")).startswith("sha256="):
        sha256 = archive["hash"].split("=", 1)[1]
    return {"url": url.geturl(), "sha256": sha256 if isinstance(sha256, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha256) else None}


def dependency_closure(roots=ROOT_PACKAGES):
    # pip's vendored parser is available even before optional project packages install.
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.markers import default_environment
    selected, visited, pending = {}, set(), [(name, frozenset()) for name in roots]
    while pending:
        name, extras = pending.pop()
        key = (normalize(name), extras)
        if key in visited:
            continue
        visited.add(key)
        dist = metadata.distribution(name)
        direct = dist.read_text("direct_url.json")
        if direct and not (normalize(name) == "torch" and official_torch_origin(dist)):
            # Do not turn local/editable/VCS installs into misleading index pins.
            raise ValueError(f"Baseline dependency {name} has a direct/local origin; resolve explicitly before export")
        selected[normalize(name)] = dist.version
        for text in dist.requires or ():
            req = Requirement(text)
            environments = [dict(default_environment(), extra=extra) for extra in ({""} | set(extras))]
            if req.marker and not any(req.marker.evaluate(env) for env in environments):
                continue
            if req.url:
                raise ValueError(f"Dependency {req.name} uses a direct reference; explicit resolution is required")
            installed = metadata.version(req.name)
            if not req.specifier.contains(installed, prereleases=True):
                raise ValueError(f"Installed {req.name} does not satisfy the baseline dependency graph")
            pending.append((req.name, frozenset(req.extras)))
    return dict(sorted(selected.items()))


def parse_lock(text):
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)", line)
        if not match or normalize(match[1]) in pins:
            raise ValueError("Lock must contain unique exact name==version pins only")
        pins[normalize(match[1])] = match[2]
    if not pins:
        raise ValueError("Empty dependency lock")
    return pins


def checked_path(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path escapes project")
    return path


def snapshot_files(root):
    root = Path(root).resolve()
    files = {}
    # Preserve source/config/test provenance used by the reproduction checks.
    for directory in ("src/echoproof", "tests", "scripts", "configs"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".json") and "__pycache__" not in path.parts and "fixtures" not in path.parts:
                files[path.relative_to(root).as_posix()] = file_digest(path)
    files["pyproject.toml"] = file_digest(root / "pyproject.toml")
    return files


def recovery_snapshot(root):
    root = Path(root).resolve()
    files = {}
    for manifest in sorted((root / "runs/phase1").glob("*/artifact_manifest.json")):
        verify_manifest(manifest.parent)
        names = ["artifact_manifest.json", *json.loads(manifest.read_text())["files"]]
        for name in names:
            path = manifest.parent / name
            files[path.relative_to(root).as_posix()] = file_digest(path)
    if not files:
        raise ValueError("No historical run manifests available for Drive recovery")
    return files


def verify_files(root, files):
    errors = []
    for name, expected in files.items():
        path = checked_path(root, name)
        if not path.is_file() or file_digest(path) != expected:
            errors.append(name)
    return {"checked_files": len(files), "mismatches": errors}


def export_environment(root, output):
    from .environment import capture_environment
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new export directory; exports are immutable")
    pins = dependency_closure()
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120)
    if check.returncode:
        raise ValueError("pip check failed; resolve dependencies before exporting")
    environment = capture_environment(root)
    if environment["gpu"] is None or not boot_id():
        raise ValueError("The baseline Colab CUDA export requires an assigned GPU and Linux boot identity")
    torch_version = pins["torch"]
    build = re.search(r"\+(cu\d+|cpu)$", torch_version)
    if not build:
        raise ValueError("Cannot infer a PyTorch wheel channel from its version")
    lock = "# EchoProof baseline dependency closure; exact platform/runtime in environment.json\n" + "".join(f"{name}=={version}\n" for name, version in pins.items())
    inventory, excluded = {}, []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        if normalize(name) == "echoproof" or dist.read_text("direct_url.json"):
            excluded.append(normalize(name))
        else:
            inventory[normalize(name)] = dist.version
    sources, recovery = snapshot_files(root), recovery_snapshot(root)
    identity = boot_id()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_new(output / "requirements-lock.txt", lock.encode())
    environment.update({"dependency_lock": file_digest(output / "requirements-lock.txt"),
                        "python_version": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "boot_id": identity})
    write_json(output / "environment.json", environment)
    write_json(output / "source_files.json", sources)
    write_json(output / "recovery_files.json", recovery)
    torch_provenance = torch_distribution_provenance()
    write_json(output / "torch_source.json", {
        "version": torch_version, "wheel_channel": build[1],
        "installed_distribution": torch_provenance,
        "reinstall_index": "https://download.pytorch.org/whl/" + build[1],
        "origin_status": ("Original official wheel origin retained from distribution metadata; reinstall channel inferred from build." if torch_provenance.get("original_official_wheel") else "Original Colab wheel URL/hash unavailable; reinstall channel inferred from installed build. Exact wheel availability must be established by fresh-runtime installation."),
        "reference": "https://download.pytorch.org/whl/cu130/torch/"})
    atomic_write_new(output / "installed-packages.txt", ("".join(f"{n}=={v}\n" for n, v in sorted(inventory.items()))).encode())
    write_json(output / "snapshot_scope.json", {"scope": "baseline dependency closure, not all Colab packages",
               "roots": list(ROOT_PACKAGES), "excluded_from_full_inventory": sorted(set(excluded)),
               "pretrained": "deferred", "wheel_hashes": "not captured; version pins are not a wheel archive"})
    names = [p.name for p in output.iterdir() if p.is_file()]
    write_json(output / "artifact_manifest.json", make_manifest(output, names))
    return {"status": "exported_for_reproduction", "export_dir": str(output), "lock_sha256": environment["dependency_lock"],
            "pinned_packages": len(pins), "recovery_files": len(recovery)}


def inspect_export(folder):
    folder = Path(folder).resolve()
    verify_manifest(folder)
    pins = parse_lock((folder / "requirements-lock.txt").read_text())
    environment = json.loads((folder / "environment.json").read_text())
    if file_digest(folder / "requirements-lock.txt") != environment["dependency_lock"]:
        raise ValueError("Lock hash does not match environment inventory")
    return folder, pins, environment


def verify_environment(root, folder):
    from .environment import capture_environment
    folder, pins, expected = inspect_export(folder)
    current = capture_environment(root)
    mismatches = {}
    for name, version in pins.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            mismatches[name] = {"expected": version, "actual": installed}
    runtime = {"python_version": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
               "torch_cuda_build": current["torch_cuda_build"], "ffmpeg_version": current["ffmpeg_version"]}
    runtime_errors = {key: {"expected": expected[key], "actual": value} for key, value in runtime.items() if expected[key] != value}
    fresh = bool(boot_id() and expected["boot_id"] and boot_id() != expected["boot_id"])
    sources = verify_files(root, json.loads((folder / "source_files.json").read_text()))
    recovery = verify_files(root, json.loads((folder / "recovery_files.json").read_text()))
    expected_torch = json.loads((folder / "torch_source.json").read_text())["installed_distribution"]
    current_torch = torch_distribution_provenance()
    torch_metadata_matches = all(current_torch["metadata_sha256"].get(name) == value for name, value in expected_torch["metadata_sha256"].items() if name in ("METADATA", "WHEEL"))
    passed = not mismatches and not runtime_errors and fresh and current["cuda_available"] and torch_metadata_matches and not sources["mismatches"] and not recovery["mismatches"]
    return {"status": "ready_for_checks" if passed else "needs_attention", "scope": "baseline", "lock_sha256": expected["dependency_lock"],
            "fresh_runtime": fresh, "package_mismatches": mismatches, "runtime_mismatches": runtime_errors,
            "source_verification": sources, "drive_recovery": recovery, "environment": current,
            "torch_distribution": current_torch, "torch_metadata_matches": torch_metadata_matches,
            "budget_reprofile_required": current["gpu"] != expected["gpu"],
            "driver_changed": current["nvidia_driver"] != expected["nvidia_driver"]}
