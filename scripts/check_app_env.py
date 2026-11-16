#!/usr/bin/env python3
"""Check that all locked distributions and required application imports are present."""
from __future__ import annotations

import importlib
import importlib.metadata
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = ("echoproof", "numpy", "pandas", "pyarrow", "scipy", "sklearn", "librosa", "soundfile", "yaml",
           "torch", "transformers", "huggingface_hub", "matplotlib", "seaborn", "streamlit", "pytest")


def main() -> int:
    issues = []
    for line in (ROOT / "requirements/app.lock.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^;\s]+)", line)
        if not match:
            issues.append(f"Invalid unpinned requirement: {line}")
            continue
        name, expected = match.groups()
        try:
            actual = importlib.metadata.version(name)
            if actual != expected:
                issues.append(f"{name}: expected {expected}, installed {actual}")
        except importlib.metadata.PackageNotFoundError:
            issues.append(f"{name}: not installed (expected {expected})")
    for name in IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:
            issues.append(f"import {name} failed: {type(exc).__name__}: {exc}")
    if issues:
        print("Environment check failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print("All locked versions and application imports are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
