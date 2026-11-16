#!/usr/bin/env python3
"""Copy locked model bundles from a local folder or ZIP; never fetches network data."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ("B_lock_gr3.json", "WavLM")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(bundle: Path, entry: dict) -> None:
    if not bundle.is_dir() or not (bundle / "bundle.json").is_file():
        raise ValueError(f"Bundle missing bundle.json: {bundle}")
    manifest_path = bundle / "bundle.json"
    if sha256(manifest_path) != entry["bundle_json_sha256"]:
        raise ValueError(f"bundle.json checksum mismatch: {bundle}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_id") != entry.get("bundle_id"):
        raise ValueError(f"Bundle identity mismatch: expected {entry.get('bundle_id')!r}, got {manifest.get('bundle_id')!r}")
    expected_files = {"bundle.json", *entry["components"]}
    actual_files = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(f"Bundle file set differs from lock (missing={missing}, extra={extra}): {bundle}")
    for rel, metadata in entry["components"].items():
        part = bundle / rel
        if not part.is_file():
            raise ValueError(f"Missing locked component {rel}: {bundle}")
        if part.stat().st_size != metadata["bytes"] or sha256(part) != metadata["sha256"]:
            raise ValueError(f"Locked component mismatch: {bundle / rel}")


def read_locks(all_bundles: bool) -> list[tuple[str, str, dict]]:
    locks = [("B_lock_gr3.json", "WavLM", json.loads((ROOT / "locks/B_lock_gr3.json").read_text(encoding="utf-8")))]
    if all_bundles:
        locks.append(("A_lock.json", "", json.loads((ROOT / "locks/A_lock.json").read_text(encoding="utf-8"))))
        locks[0] = (locks[0][0], "", locks[0][2])
    out = []
    for lock_name, only_candidate, lock in locks:
        candidates = lock["candidates"]
        for name, entry in candidates.items():
            if only_candidate and name != only_candidate:
                continue
            out.append((lock_name, name, entry))
    return out


def prepare_source(source: Path, temp: Path) -> Path:
    source = source.resolve()
    if source.is_dir():
        return source
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError(f"--source must be an existing folder or ZIP file: {source}")
    with zipfile.ZipFile(source) as archive:
        base = temp.resolve()
        for member in archive.infolist():
            target = (base / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        archive.extractall(base)
    return base


def locate(source: Path, entry: dict) -> Path:
    expected = entry["bundle_json_sha256"]
    # Locate by the locked manifest digest, not by a user-supplied name/path.
    for manifest in source.rglob("bundle.json"):
        try:
            if sha256(manifest) == expected:
                candidate = manifest.parent
                verify(candidate, entry)
                return candidate
        except OSError:
            continue
    raise ValueError(f"No bundle with locked manifest {expected} found under {source}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local folder or ZIP containing bundle folders")
    parser.add_argument("--all", action="store_true", help="Retrieve every candidate in both B and A locks")
    args = parser.parse_args(argv)
    try:
        source_path = Path(args.source)
        with tempfile.TemporaryDirectory(prefix="echoproof-bundles-") as temporary:
            source = prepare_source(source_path, Path(temporary))
            records = read_locks(args.all)
            located = [(lock_name, candidate, entry, locate(source, entry))
                       for lock_name, candidate, entry in records]
            # Validate every source bundle before creating or replacing any destination.
            for lock_name, candidate, entry, origin in located:
                destination = ROOT / entry["bundle_path"]
                if destination.exists():
                    verify(destination, entry)
                    print(f"Verified existing {candidate}: {destination.relative_to(ROOT)}")
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging = destination.with_name(destination.name + ".fetch-staging")
                if staging.exists():
                    shutil.rmtree(staging)
                shutil.copytree(origin, staging)
                verify(staging, entry)
                staging.rename(destination)
                verify(destination, entry)
                print(f"Retrieved and verified {candidate} ({lock_name}): {destination.relative_to(ROOT)}")
        return 0
    except Exception as exc:
        print(f"ERROR: bundle retrieval failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
