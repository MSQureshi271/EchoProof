"""Unique run directories, canonical hashes and verified immutable artifacts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_new(path, data):
    """Reserve a destination exclusively, then atomically replace it with data."""
    path = Path(path)
    if not isinstance(data, bytes):
        raise TypeError("artifact content must be bytes")
    temporary = None
    reserved = False
    try:
        # Exclusive reservation prevents accidental replacement by another writer.
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        reserved = True
        os.close(descriptor)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".echoproof-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except BaseException:
        if reserved:
            path.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write_new(path, (canonical_json(value) + "\n").encode("utf-8"))


def create_run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    directory = root / run_id
    directory.mkdir(exist_ok=False)
    return run_id, directory


def make_manifest(directory, names):
    directory = Path(directory)
    if any(Path(name).name != name or name in ("artifact_manifest.json", "status.json") for name in names):
        raise ValueError("manifest entries must be simple artifact names")
    if len(set(names)) != len(names):
        raise ValueError("duplicate artifact name")
    return {"schema_version": 1, "algorithm": "sha256", "files": {name: file_digest(directory / name) for name in sorted(names)}}


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256" or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("invalid artifact manifest")
    for name, expected in manifest["files"].items():
        if not isinstance(name, str) or Path(name).name != name or name in ("artifact_manifest.json", "status.json") or file_digest(directory / name) != expected:
            raise ValueError(f"invalid or altered artifact: {name}")
    return {"verified_files": len(manifest["files"]), "manifest_sha256": file_digest(directory / "artifact_manifest.json")}
