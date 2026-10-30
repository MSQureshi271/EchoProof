"""Capture a limited, non-secret runtime inventory for run provenance."""
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys


def _command(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def capture_environment(project_root):
    import torch
    packages = {}
    for name in ("echoproof", "numpy", "torch", "PyYAML", "pyarrow", "scipy", "scikit-learn", "librosa", "soundfile", "joblib", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    ffmpeg = _command(["ffmpeg", "-version"])
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    driver = _command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    revision = _command(["git", "rev-parse", "HEAD"], cwd=project_root)
    try:
        import psutil
        ram_bytes = psutil.virtual_memory().total
    except ImportError:
        ram_bytes = None
    lock = os.environ.get("ECHOPROOF_DEPENDENCY_LOCK")
    if lock:
        from .artifacts import file_digest
        lock = file_digest(lock)
    return {"python": sys.version, "platform": platform.platform(), "cpu": platform.processor() or None,
            "cpu_count": os.cpu_count(), "ram_bytes": ram_bytes, "gpu": gpu,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if gpu else None,
            "cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
            "nvidia_driver": driver, "ffmpeg_path": shutil.which("ffmpeg"),
            "ffmpeg_version": ffmpeg.splitlines()[0] if ffmpeg else None, "packages": packages,
            "source_revision": revision, "dependency_lock": lock or "not_yet_frozen",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "unsupported_operations": "not_exercised_by_storage_probe"}
