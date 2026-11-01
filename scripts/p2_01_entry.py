"""Standard-library P2-01 entry reconciliation; no audio loading or installation."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid

LOCK = "locks/colab-baseline-20261029T192113Z-93db044e"
ROOTS = {"genuine_audio": "dataset/genuine/wav", "transcripts": "dataset/genuine/transcripts",
         "xttsv2": "dataset/tts/XTTSv2", "chatterbox_tts": "dataset/tts/ChatterboxTTS"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Path escapes project root")
    return path


def collect(project_root, observation_scope="local_workspace"):
    root = Path(project_root).resolve()
    if not root.is_dir() or observation_scope not in ("local_workspace", "colab_drive"):
        raise ValueError("Existing project root and valid observation scope required")
    issues, paths = [], {}
    for key, relative in ROOTS.items():
        path = contained(root, relative)
        state = "directory_present" if path.is_dir() else "missing_or_not_directory"
        paths[key] = {"path": relative, "observation": state, "contents_verified": False}
        if state != "directory_present":
            issues.append({"kind": "required_path", "path": relative, "reason": state})
    required = ["plans/PHASE2.md", "docs/research_contract.md", "docs/phase1_gate_checklist.md",
                "docs/p1_13_review.md", "reports/phase1/p1_13/reproduce-20261029T192537Z-816edf2c/verification.json"]
    evidence = {}
    for relative in required:
        path = contained(root, relative)
        if path.is_file():
            evidence[relative] = sha(path)
        else:
            issues.append({"kind": "entry_evidence", "path": relative, "reason": "missing"})
    lock_check = {"path": LOCK, "status": "needs_attention", "verified_files": 0}
    try:
        folder = contained(root, LOCK)
        manifest = folder / "artifact_manifest.json"
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if value.get("algorithm") != "sha256" or value.get("schema_version") != 1 or not value.get("files"):
            raise ValueError("Invalid lock manifest")
        for name, expected in value["files"].items():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Invalid lock entry")
            path = contained(root, LOCK + "/" + name)
            if sha(path) != expected:
                raise ValueError("Lock artifact mismatch: " + name)
        lock_check.update(status="verified", verified_files=len(value["files"]),
                          manifest_sha256=sha(manifest), lock_sha256=sha(folder / "requirements-lock.txt"))
    except (OSError, ValueError, TypeError) as exc:
        issues.append({"kind": "lock_integrity", "reason": str(exc)})
    # Only discover small metadata candidates at administrative roots, never scan audio trees.
    candidates = []
    for relative in ("dataset", "dataset/genuine", "dataset/tts", "docs/datasets"):
        path = contained(root, relative)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and (child.suffix.lower() in (".json", ".csv", ".tsv", ".pdf", ".md") or child.name.lower().startswith(("license", "readme"))):
                    candidates.append(child.relative_to(root).as_posix())
    return {"schema_version": 1, "task": "P2-01", "observation_scope": observation_scope,
            "status": "entry_ready_for_review" if not issues else "needs_attention", "project_root": str(root),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "runtime": {"python": sys.version, "platform": platform.platform()},
            "source_state": {"phonemedf": {"availability": "user_reported_uploaded_to_drive",
                             "components": ["genuine", "XTTSv2", "ChatterboxTTS"], "source_verified": False,
                             "publisher_revision": None, "publisher_revision_verified": False},
                             "mlaad_tiny": {"availability": "not_downloaded_user_confirmed", "eligible": False}},
            "paths": paths, "prior_lock": lock_check, "entry_evidence_hashes": evidence,
            "metadata_discovery": {"scope": "administrative root filenames only; no content verification",
                                   "candidates": candidates},
            "protocol_controls": {"chatterbox_tts": "proposed_B_final_family; administrative/integrity audit only before B lock",
                                  "pilot": "candidate XTTSv2 study; no cohort/splits/crops selected",
                                  "full_B": "original MeloTTS/VITS training families not reported acquired; feasibility pending",
                                  "external": "no second source admitted", "training_permitted_by_entry_check": False},
            "capacity": {"current_free_drive_gb": None, "previous_free_drive_gb": 10.13,
                         "previous_snapshot_current": False, "reason": "snapshot predates reported ChatterboxTTS upload",
                         "prior_work_allocation_hours": 6, "remaining_work_hours": None},
            "pending": ["publisher release/archive and reorganization history", "license and usage evidence",
                        "label/method verification", "complete counts/content pairing", "source/target/reference identities",
                        "current quota and work availability", "independent Phase 1 and P2-01 reviews"],
            "issues": issues, "next_task": "P2-02 source/release/license evidence",
            "scope_note": "Path and saved-lock checks only; no audio admission, scientific split or model fitting"}


def run(project_root, observation_scope="local_workspace"):
    report = collect(project_root, observation_scope)
    root = Path(project_root).resolve()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = root / "reports/phase2" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report["run_id"] = run_id
    report["collector_sha256"] = sha(__file__)
    (folder / "entry.json").write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    manifest = {"schema_version": 1, "algorithm": "sha256", "files": {"entry.json": sha(folder / "entry.json")}}
    (folder / "artifact_manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"task": "P2-01", "status": report["status"], "observation_scope": observation_scope,
                      "evidence_dir": str(folder), "issues": report["issues"]}, indent=2))
    return report, folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--scope", choices=("local_workspace", "colab_drive"), default="local_workspace")
    args = parser.parse_args()
    report, _ = run(args.project_root, args.scope)
    raise SystemExit(0 if report["status"] == "entry_ready_for_review" else 4)
