"""Phase 2 handoff and review evidence; no dataset modification."""
import hashlib, json, platform, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")


def verify_manifest(folder, filename, key=None):
    folder = Path(folder)
    manifest = read(folder / filename)
    entries = manifest[key] if key else manifest
    for name, expected in entries.items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError("Artifact mismatch: " + name)
    return manifest


def handoff(root, snapshot, pairing, inspection, reconciliation, capacity):
    root, snapshot, pairing, inspection, reconciliation, capacity = map(Path, (root, snapshot, pairing, inspection, reconciliation, capacity))
    snap = verify_manifest(snapshot, "snapshot_manifest.json", "artifacts_sha256")
    pair = verify_manifest(pairing, "pairing_summary.json", "artifacts_sha256")
    ins = verify_manifest(inspection, "inspection_summary.json", "artifacts_sha256")
    verify_manifest(reconciliation, "artifact_manifest.json")
    verify_manifest(capacity, "artifact_manifest.json")
    rec = read(reconciliation / "reconciliation.json")
    cap = read(capacity / "acquisition_plan.json")
    if pair["snapshot_manifest_sha256"] != digest(snapshot / "snapshot_manifest.json") or ins["pairing_summary_sha256"] != digest(pairing / "pairing_summary.json"):
        raise ValueError("Snapshot, pairing, and inspection do not match")
    for key, path in (("snapshot_manifest_sha256", snapshot / "snapshot_manifest.json"), ("pairing_summary_sha256", pairing / "pairing_summary.json"), ("inspection_summary_sha256", inspection / "inspection_summary.json")):
        if rec["inputs"][key] != digest(path): raise ValueError("Reconciliation input mismatch: " + key)
    out = root / "reports/phase2" / ("p2_12_14-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12])
    out.mkdir(parents=True, exist_ok=False)
    issues = list(rec["issues"])
    issues += ["Source admission remains pending", "Machine-readable relationship and identity evidence remains pending", "Full duplicate disposition and partitions are deferred", "Phase 3 processing storage estimate has not been established"]
    value = {"task":"P2-12", "status":"needs_attention", "components":rec["components"], "pairing":rec["pairing"], "unique_genuine_controls":rec["unique_genuine_controls"], "transcript_files":rec["transcript_files"], "protocols":{"A":"pending admission and grouped partition design", "B":"pending admission, grouped partition design and predeclared unseen-family protocol"}, "relationship_evidence":str(pairing / "candidate_edges.jsonl"), "pairing_directory":str(pairing), "deferred":{"deduplication":"Phase 3", "grouped_partitions":"Phase 4", "integration_cohort":"500?1000 recordings, not selected", "baseline_pilot":"1000 pairs / 2000 recordings, one four-second crop per recording; not selected"}, "issues":issues, "capacity_status":cap["status"], "inputs":{str(p):digest(p) for p in (snapshot / "snapshot_manifest.json", pairing / "pairing_summary.json", inspection / "inspection_summary.json", reconciliation / "reconciliation.json", capacity / "acquisition_plan.json")}}
    # Resolve actual pairing artifact names from its signed summary.
    value["relationship_artifacts"] = {k:str(pairing / k) for k in pair["artifacts_sha256"] if "edge" in k or "disposition" in k or "coverage" in k}
    value.pop("relationship_evidence")
    write(out / "phase3_handoff.json", value)
    return out


def verify(root, out):
    root, out = Path(root), Path(out)
    checks = []
    for name in ("test_p2_03_contracts.py", "test_p2_04_inventory.py", "test_p2_05_adapter.py", "test_p2_06_pairing.py", "test_p2_07_09_quality.py"):
        run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-p", name, "-v"], cwd=root, text=True, capture_output=True)
        output = run.stdout + run.stderr
        checks.append({"name":name, "returncode":run.returncode, "skips_detected":"skipped" in output.lower(), "output":output})
    sources = sorted((root / "src/echoproof/data").glob("*.py"))
    result = {"task":"P2-13", "status":"needs_attention", "fixture_checks":checks, "python":sys.version, "platform":platform.platform(), "source_hashes":{str(p.relative_to(root)):digest(p) for p in sources}, "drive_restart_recovery":"not_demonstrated_by_this_runner", "environment_reproduction":"not_demonstrated_for_phase2_sources", "note":"A successful fixture resume test does not establish recovery across Colab runtimes."}
    write(out / "verification.json", result)
    return result


def gate(out):
    out = Path(out)
    hand = read(out / "phase3_handoff.json")
    ver = read(out / "verification.json")
    checks = {"inspection_and_reconciliation": "pending" if hand["issues"] else "review_required", "source_admission_and_machine_readable_identity": "pending", "processing_capacity": "pending", "fixture_checks": "passed" if all(c["returncode"] == 0 and not c["skips_detected"] for c in ver["fixture_checks"]) else "needs_attention", "fresh_runtime_recovery": "pending", "phase2_environment_reproduction": "pending", "independent_review": "pending"}
    result = {"task":"P2-14", "status":"needs_attention", "gate":"not_passed", "checks":checks, "reviewer":"N", "independent_review":"not_recorded", "handoff_issues":hand["issues"]}
    write(out / "gate_review.json", result)
    names = ("phase3_handoff.json", "verification.json", "gate_review.json")
    write(out / "evidence_manifest.json", {name:digest(out / name) for name in names})
    return result
