"""Install the local EchoProof scaffold in Colab without resolving torch dependencies.

P1-03 only. Run from the Colab notebook; no datasets are opened or altered.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid


BASELINE_IMPORTS = {
    "numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow", "scipy": "scipy",
    "scikit-learn": "sklearn", "librosa": "librosa", "soundfile": "soundfile",
    "PyYAML": "yaml", "torch": "torch", "pytest": "pytest",
}


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(args, timeout=600):
    args = [str(item) for item in args]
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"argv": args, "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": args, "returncode": None, "error": str(exc)}


def setup(project_root, *, allow_install=False):
    project = Path(project_root).expanduser().resolve()
    if not (project / "IMPLEMENTATION_PLAN.md").is_file():
        raise ValueError("Project root must contain IMPLEMENTATION_PLAN.md")
    if not (project / "pyproject.toml").is_file() or not (project / "src/echoproof/cli.py").is_file() or not (project / "docs/phase1_setup.md").is_file():
        raise ValueError("Sync pyproject.toml, src/echoproof/, and docs/phase1_setup.md to the Drive project first")

    before = {"python": sys.version, "executable": sys.executable,
              "torch_distribution": version("torch"), "cuda_available": None,
              "cuda_build": None, "gpu_name": None}
    try:
        import torch
        before["cuda_available"] = torch.cuda.is_available()
        before["cuda_build"] = torch.version.cuda
        if before["cuda_available"]:
            before["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        before["torch_error"] = str(exc)

    installs = []
    if allow_install:
        setuptools_version = version("setuptools")
        setuptools_major = (int(re.match(r"\d+", setuptools_version).group())
                            if setuptools_version and re.match(r"\d+", setuptools_version) else 0)
        missing_build = (["setuptools>=68"] if setuptools_major < 68 else [])
        if version("wheel") is None:
            missing_build.append("wheel")
        if missing_build:
            installs.append(run([sys.executable, "-m", "pip", "install", *missing_build]))
        if version("jedi") is None:
            installs.append(run([sys.executable, "-m", "pip", "install", "jedi"]))
        if all(item["returncode"] == 0 for item in installs):
            installs.append(run([sys.executable, "-m", "pip", "install", "--no-deps",
                                 "--no-build-isolation", "-e", project]))
        importlib.invalidate_caches()

    imports = {}
    for dist, module in BASELINE_IMPORTS.items():
        try:
            importlib.import_module(module)
            imports[dist] = {"ok": True, "version": version(dist)}
        except Exception as exc:
            imports[dist] = {"ok": False, "version": version(dist), "error": str(exc)}
    # Editable installs register a .pth file at interpreter startup. The current
    # notebook kernel may not see it until restart, so verify in a fresh process.
    package_import = run([sys.executable, "-c", "import echoproof; print(echoproof.__version__)"], timeout=60)
    imports["echoproof"] = {"ok": package_import["returncode"] == 0,
                            "version": version("echoproof"),
                            "stdout_tail": package_import.get("stdout_tail", ""),
                            "stderr_tail": package_import.get("stderr_tail", ""),
                            "error": package_import.get("error")}

    module_help = run([sys.executable, "-m", "echoproof.cli", "--help"], timeout=60)
    module_version = run([sys.executable, "-m", "echoproof.cli", "--version"], timeout=60)
    console_path = shutil.which("echoproof")
    console_version = (run([console_path, "--version"], timeout=60) if console_path
                       else {"returncode": None, "error": "Console script not found"})
    unknown = run([sys.executable, "-m", "echoproof.cli", "not-implemented"], timeout=60)
    pip_check = run([sys.executable, "-m", "pip", "check"], timeout=120)
    ffmpeg = run(["ffmpeg", "-version"], timeout=30)
    mp3_decoder = run(["ffmpeg", "-hide_banner", "-h", "decoder=mp3"], timeout=30)
    mp3_encoder = run(["ffmpeg", "-hide_banner", "-h", "encoder=libmp3lame"], timeout=30)
    after_torch = version("torch")

    issues = []
    if sys.version_info < (3, 11):
        issues.append("Python does not meet requires-python >=3.11")
    if before["torch_distribution"] != after_torch:
        issues.append("Torch distribution changed during setup")
    if not allow_install:
        issues.append("Read-only diagnostic: editable package installation was not attempted")
    if any(item["returncode"] != 0 for item in installs):
        issues.append("Package installation command failed")
    if not all(item["ok"] for item in imports.values()):
        issues.append("One or more baseline imports failed")
    if any(item["returncode"] != 0 for item in (module_help, module_version, console_version)):
        issues.append("Module or console CLI failed")
    if unknown["returncode"] == 0:
        issues.append("Unknown CLI command incorrectly reported success")
    if pip_check["returncode"] != 0:
        issues.append("pip check reported unresolved dependency conflicts")
    if ffmpeg["returncode"] != 0 or mp3_decoder["returncode"] != 0 or mp3_encoder["returncode"] != 0:
        issues.append("FFmpeg or required MP3 codec capability was not confirmed")

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = project / "reports/phase1/p1_03" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "task": "P1-03", "run_id": run_id,
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "project_root": str(project), "platform": platform.platform(),
              "before": before, "torch_distribution_after": after_torch,
              "packages": {name: version(name) for name in [*BASELINE_IMPORTS,
                           "echoproof", "setuptools", "wheel", "jedi", "ipython", "pip"]},
              "install_commands": installs, "imports": imports,
              "module_help": module_help, "module_version": module_version,
              "console_script_path": console_path, "console_version": console_version,
              "unknown_command": unknown, "pip_check": pip_check,
              "ffmpeg_version": ffmpeg["stdout_tail"].splitlines()[0] if ffmpeg.get("stdout_tail") else None,
              "mp3_decoder": mp3_decoder["returncode"] == 0,
              "mp3_encoder": mp3_encoder["returncode"] == 0,
              "issues": issues, "status": "needs_attention" if issues else "ready_for_review",
              "scope": "Package setup only; no dataset audit, model training or environment lock"}
    summary = f"""# EchoProof P1-03 package setup

Run: `{run_id}` · Status: **{report['status']}**; pending M/U review.

- Python: `{platform.python_version()}`; torch before/after: `{before['torch_distribution']}` / `{after_torch}`.
- CUDA available: `{before['cuda_available']}`; GPU: `{before['gpu_name']}`.
- EchoProof package version: `{version('echoproof')}`; console script: `{console_path}`.
- Baseline imports passed: `{all(item['ok'] for item in imports.values())}`.
- `pip check` exit: `{pip_check['returncode']}`; FFmpeg: `{report['ffmpeg_version']}`.
- MP3 decoder/encoder confirmed: `{report['mp3_decoder']}` / `{report['mp3_encoder']}`.

## Issues

{chr(10).join('- ' + item for item in issues) or '- None in this run.'}

See `package_setup.json` for install commands, versions and detailed diagnostics. P1-13 freezes exact versions after later probes.
"""
    files = {"package_setup.json": json.dumps(report, indent=2, ensure_ascii=False) + "\n",
             "package_setup.md": summary}
    for name, content in files.items():
        (folder / name).write_text(content, encoding="utf-8")
    manifest = {"run_id": run_id, "files": {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files}}
    (folder / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary)
    print("Evidence folder:", folder)
    return report, folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--install", action="store_true",
                        help="Install missing build tools/jedi and local EchoProof without dependency resolution")
    args = parser.parse_args(argv)
    report, _ = setup(args.project_root, allow_install=args.install)
    return 0 if report["status"] == "ready_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
