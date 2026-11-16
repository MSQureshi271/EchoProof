"""R5-N1 sealed evaluation runner (Option 2: dedicated loader registry, shared scoring path, no locked file edited).

    python scripts/r5_n1_sealed.py plan                          # dry run: verifies everything, prints what would be scored, reads no test rows, writes nothing
    python scripts/r5_n1_sealed.py run --confirm-sealed-run      # the one sealed run (GPU needed for the two AST bundles)

`run` logs before opening test rows and journals every prediction. A matching interrupted run can be resumed with an explicit reason; ambiguous
in-flight scoring is refused for review. It computes no metric (see `METRICS_HOOK`). Existing completed runs are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import importlib.metadata
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from echoproof.artifacts import digest                      # noqa: E402
from echoproof.data import splits                           # noqa: E402  (looked up as `splits.load_assignments` at call time)
from echoproof.evaluation import r5_loaders as R            # noqa: E402

DEFAULT_OUT = "reports/final/results/r5_n1_sealed_run"
DEFAULT_SHARDS = "data/processed/cohort_v1"
INDEX_NAME = "index.parquet"
COHORT_MANIFEST = "data/manifests/cohort_v1.parquet"
SCHEMA_VERSION = 1
METRICS_HOOK = None   # later task: a callable (run_dir, integrity_report) -> None, called by `execute` only when every integrity check passed


class SealedRunError(RuntimeError):
    """The sealed run is refused (missing confirmation, existing run folder, unverified lock, bundle or device, or a failed integrity check)."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- code state -----------------------------------------------------------------------------------------------------------------------------------

def code_state(root) -> dict:
    root = Path(root)
    src = {p.relative_to(root).as_posix(): R.sha256_file(p) for p in sorted((root / "src/echoproof").rglob("*.py"))} if (root / "src/echoproof").is_dir() else {}
    own = {p.relative_to(root).as_posix(): R.sha256_file(p) for p in (Path(__file__),) if root in p.parents}
    state = {"src_echoproof_sha256": src, "runner_sha256": own, "python": sys.version.split()[0]}
    try:
        git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=True).stdout.strip()
        status = git("status", "--porcelain")
        state.update(git_head=git("rev-parse", "HEAD"), working_tree_dirty=bool(status), uncommitted_entries=status.splitlines())
    except Exception:
        state.update(git_head=None, working_tree_dirty=None, uncommitted_entries=[])
    return state


def provenance_diff(code: dict, records: dict) -> dict:
    """Informational: which shared source files differ from the hashes recorded in each lock (the locks record provenance; nothing enforces equality)."""
    out = {}
    for protocol, record in records.items():
        recorded = record.get("code", {}).get("src_echoproof_sha256", {})
        now = code["src_echoproof_sha256"]
        out[protocol] = {"changed": sorted(k for k in recorded if k in now and now[k] != recorded[k]), "missing_now": sorted(k for k in recorded if k not in now),
                         "new_since_lock": sorted(k for k in now if k not in recorded)}
    return out


# ---- plan -----------------------------------------------------------------------------------------------------------------------------------------

def _check(checks, name, ok, detail="", scope="verification"):
    checks.append({"name": name, "ok": bool(ok), "scope": scope, "detail": detail})
    return ok


def build_plan(root, *, shards=DEFAULT_SHARDS, device=None, deep: bool = True, load_check: bool = False, cuda: bool | None = None) -> dict:
    """Verify locks, bundles, split files and the audio source, and describe what would be scored. Reads no test rows (split files are hashed as bytes only)."""
    root = Path(root)
    checks, entries = [], []
    plan = {"schema_version": SCHEMA_VERSION, "created_utc": utc_now(), "checks": checks, "entries": entries}
    cuda = R.cuda_available() if cuda is None else cuda
    plan["cuda_available"] = cuda
    records = {}
    try:
        records["B"] = R.verify_b_lock(root)
        _check(checks, "Protocol B lock (locks/B_lock_gr3.json) is the locked file", True)
    except Exception as exc:
        _check(checks, "Protocol B lock (locks/B_lock_gr3.json) is the locked file", False, str(exc))
    try:
        records["A"] = R.verify_a_lock(root, deep=deep)
        _check(checks, "Protocol A lock (locks/A_lock.json) is the locked file" + (" and its evidence hashes verify" if deep else " (file hash only, --quick)"), True)
    except Exception as exc:
        _check(checks, "Protocol A lock (locks/A_lock.json) is the locked file", False, str(exc))
    pairs = []
    if len(records) == 2:
        try:
            pairs = R.candidate_bundles(records)
            _check(checks, "registry covers exactly the eight lock candidates", True)
        except Exception as exc:
            _check(checks, "registry covers exactly the eight lock candidates", False, str(exc))
    for spec, record in pairs:
        ok, detail = True, ""
        try:
            R.verify_bundle(root, spec, record)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        _check(checks, f"{spec.key}: bundle.json, components and decision rule match the lock", ok, detail)
        dev, dev_detail = None, ""
        try:
            dev = R.choose_device(spec, device, cuda=cuda)
        except Exception as exc:
            dev_detail = str(exc)
        _check(checks, f"{spec.key}: device policy resolves here ({dev or 'refused'})", dev is not None, dev_detail, scope="machine")
        if load_check and ok and dev is not None:
            loaded = None
            try:
                loaded = R.load_verified(root, spec, record, device=dev, cuda=cuda)[0]
                _check(checks, f"{spec.key}: loads through {spec.loader}", True, f"score_type={loaded.score_type}")
            except Exception as exc:
                _check(checks, f"{spec.key}: loads through {spec.loader}", False, f"{type(exc).__name__}: {exc}", scope="machine")
            finally:
                loaded = None
                _free(None)
        entries.append({"key": spec.key, "protocol": spec.protocol, "model": spec.model, "model_version": spec.model_version, "loader": spec.loader,
                        "bundle_path": record["bundle_path"], "bundle_id": record["bundle_id"], "bundle_json_sha256": record["bundle_json_sha256"],
                        "decision_rule": record["decision_rule"], "device": dev, "precision": spec.precision, "requires_cuda": spec.requires_cuda,
                        "allow_pending_review": spec.allow_pending_review, "via_shared_router": spec.via_router, "split_file": R.SPLIT_FILES[spec.protocol],
                        "partition": "test", "_spec": spec, "_record": record})
    split_plan = {}
    for protocol in ("B", "A"):
        path = root / R.SPLIT_FILES[protocol]
        locked_file_sha = records.get(protocol, {}).get("split", {}).get("split_file_sha256")
        info = {"file": R.SPLIT_FILES[protocol], "expected_split_hash": R.SPLIT_HASHES[protocol], "file_sha256": R.sha256_file(path) if path.is_file() else None,
                "locked_file_sha256": locked_file_sha}
        split_plan[protocol] = info
        _check(checks, f"split file for Protocol {protocol} is present", path.is_file(), R.SPLIT_FILES[protocol])
        if locked_file_sha is not None:
            _check(checks, f"split file for Protocol {protocol} matches the hash in its lock", info["file_sha256"] == locked_file_sha)
        else:
            info["note"] = "the B lock records only the row digest (split_hash); it is verified at run time before any recording is scored, and the file hash is written to the access log"
    plan["splits"] = split_plan
    shard_dir = (root / shards)
    index = shard_dir / INDEX_NAME
    present = sorted(p.name for p in shard_dir.glob("shard_*.tar")) if shard_dir.is_dir() else []
    plan["audio"] = {"shard_dir": str(shards), "index": str(Path(shards) / INDEX_NAME), "index_sha256": R.sha256_file(index) if index.is_file() else None,
                     "shards_present": present, "cohort_manifest": COHORT_MANIFEST,
                     "cohort_manifest_sha256": R.sha256_file(root / COHORT_MANIFEST) if (root / COHORT_MANIFEST).is_file() else None}
    _check(checks, "audio shard index is present", index.is_file(), str(Path(shards) / INDEX_NAME), scope="machine")
    _check(checks, "audio shards are present (shard_*.tar)", bool(present), f"{len(present)} found in {shards}", scope="machine")
    plan["code"] = code_state(root)
    plan["provenance_diff_vs_locks"] = provenance_diff(plan["code"], records) if records else {}
    plan["verified"] = all(c["ok"] for c in checks if c["scope"] == "verification") and len(entries) == len(R.REGISTRY)
    plan["ready"] = plan["verified"] and all(c["ok"] for c in checks)
    return plan


def public(plan: dict) -> dict:
    """The plan without live objects, for printing and for the access log."""
    return {**plan, "entries": [{k: v for k, v in e.items() if not k.startswith("_")} for e in plan["entries"]]}


def format_plan(plan: dict) -> str:
    lines = ["R5-N1 sealed evaluation: PLAN (dry run; no test row was read, nothing was written)", ""]
    lines.append("Checks")
    for c in plan["checks"]:
        lines.append(f"  [{'ok' if c['ok'] else 'FAIL'}] ({c['scope']}) {c['name']}" + (f" -- {c['detail']}" if c["detail"] else ""))
    lines += ["", "What `run --confirm-sealed-run` would score (every recording of the TEST partition of each protocol; counts are known only after the split is opened):"]
    for e in plan["entries"]:
        extra = []
        if e["allow_pending_review"]:
            extra.append("explicit allow_pending_review=True (calibration review pending)")
        if e["via_shared_router"]:
            extra.append("shared router")
        lines.append(f"  {e['key']:9s} {e['model_version']:22s} threshold {e['decision_rule']['threshold']:.6f}  device {e['device'] or 'REFUSED here'}"
                     + (f" {e['precision']}" if e["precision"] else "") + f"  loader {e['loader']}  bundle {e['bundle_path']}" + (f"  [{'; '.join(extra)}]" if extra else ""))
        lines.append(f"{'':10s}scores partition=test of {e['split_file']} through shared predict_recording; no threshold override; failures are counted, never dropped")
    lines += ["", "Split files: " + "; ".join(f"{p}: {s['file']} sha256 {str(s['file_sha256'])[:16]} (row digest to verify {s['expected_split_hash'][:16]})" for p, s in plan["splits"].items())]
    lines.append(f"Audio: {plan['audio']['shard_dir']} index sha256 {str(plan['audio']['index_sha256'])[:16]}, {len(plan['audio']['shards_present'])} shards present")
    lines.append(f"Code: git head {plan['code']['git_head']}, working tree dirty: {plan['code']['working_tree_dirty']}")
    for p, d in plan["provenance_diff_vs_locks"].items():
        lines.append(f"  shared sources vs the {p} lock record: {len(d['changed'])} changed, {len(d['missing_now'])} missing, {len(d['new_since_lock'])} new (informational; nothing enforces equality)")
    lines += ["", f"Locks, bundles and split files verified: {'YES' if plan['verified'] else 'NO'}", f"Ready to run on this machine: {'YES' if plan['ready'] else 'NO'}"]
    return "\n".join(lines)


# ---- audio source ----------------------------------------------------------------------------------------------------------------------------------

class ShardAudio:
    """Reads a recording's canonical FLAC bytes from the cohort tar shards (`index.parquet`: recording_id -> shard). Opens each tar once."""

    def __init__(self, shard_dir):
        import pyarrow.parquet as pq
        self.dir = Path(shard_dir)
        self.index = {r["recording_id"]: r["shard"] for r in pq.read_table(str(self.dir / INDEX_NAME), columns=["recording_id", "shard"]).to_pylist()}
        self._members: dict[str, dict] = {}

    def _shard(self, name):
        if name not in self._members:
            import tarfile
            with tarfile.open(self.dir / name, "r") as tar:
                self._members[name] = {m.name: (m.offset_data, m.size) for m in tar.getmembers()}
        return self._members[name]

    def __call__(self, recording_id: str) -> bytes:
        shard = self.index[recording_id]
        offset, size = self._shard(shard)[recording_id.replace(":", "__") + ".flac"]
        with open(self.dir / shard, "rb") as f:
            f.seek(offset)
            return f.read(size)


# ---- integrity checks -------------------------------------------------------------------------------------------------------------------------------

def label_of(recording_id: str) -> int:
    return 0 if recording_id.split(":")[1] == "genuine" else 1


def check_integrity(expected_ids, records, *, threshold=None) -> dict:
    """Denominators, failures and finiteness for one bundle, BEFORE any metric. Zero tolerance: any problem fails the bundle (nothing is dropped)."""
    expected = list(expected_ids)
    ids = [r["recording_id"] for r in records]
    counts = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    exp = set(expected)
    duplicates = sorted(i for i, n in counts.items() if n > 1)
    missing = sorted(exp - set(ids))
    unexpected = sorted(set(ids) - exp)
    failures = sorted((r["recording_id"], r.get("failure_reason") or r.get("status")) for r in records if r.get("status") != "success")
    nonfinite = sorted(r["recording_id"] for r in records if r.get("status") == "success"
                       and not (isinstance(r.get("score"), (int, float)) and not isinstance(r.get("score"), bool) and math.isfinite(r["score"]) and 0.0 <= r["score"] <= 1.0))
    mismatch = []
    if threshold is not None:
        for r in records:
            if r.get("status") == "success" and r["recording_id"] not in nonfinite:
                if (r.get("assessment") == "synthetic") != (r["score"] >= threshold):
                    mismatch.append(r["recording_id"])
    problems = {"missing": missing, "duplicates": duplicates, "unexpected": unexpected, "failures": [list(f) for f in failures], "nonfinite_scores": nonfinite,
                "decision_mismatch": sorted(mismatch), "count_mismatch": [] if len(records) == len(expected) else [[len(records), len(expected)]]}
    return {"expected": len(expected), "records": len(records), "scored_ok": len(records) - len(failures) - len(nonfinite), "problems": problems,
            "passed": not any(problems.values())}


def split_digest(protocol: str, ordered) -> str:
    """The split row digest as the split builder computed it. The stored `family_policy` column of the real split files holds `protocol_<X>` for every row,
    while the locked hash was taken over the per-row policy of the protocol filter (`splits.PROTOCOLS`); that policy is re-derived from (partition, role,
    component). Rows that already carry a real policy (synthetic fixtures) are used as stored."""
    rules = splits.PROTOCOLS[protocol]
    out = []
    for r in ordered:
        policy = r.family_policy
        if str(policy).startswith("protocol_"):
            if policy != f"protocol_{protocol}":
                raise SealedRunError(f"Protocol {protocol}: unexpected stored family policy {policy!r}")
            policy = rules[(r.partition, "*" if r.partition == "validation" else None)][r.recording_id.split(":")[1]]
        out.append([r.recording_id, r.group_id, r.partition, r.validation_role, policy])
    return digest(out)


# ---- the sealed run ----------------------------------------------------------------------------------------------------------------------------------

class AccessLog:
    """Append-only JSON lines, flushed and fsynced per event. Created exclusively, so an existing log is never overwritten."""

    def __init__(self, path, *, resume=False):
        self.path = Path(path)
        if resume:
            data = self.path.read_bytes()
            if not data or not data.endswith(b"\n"):
                raise SealedRunError("access log is incomplete; inspect it before recovery")
            try:
                self.events = [json.loads(line) for line in data.splitlines()]
            except (ValueError, UnicodeError) as exc:
                raise SealedRunError("access log is invalid; inspect it before recovery") from exc
            if self.events[0].get("event") != "sealed_run_start":
                raise SealedRunError("access log has no sealed start event")
        else:
            self.events = []
        self._f = open(self.path, "a" if resume else "x", encoding="utf-8")

    def event(self, kind, **data):
        self._f.write(json.dumps({"utc": utc_now(), "event": kind, **data}, sort_keys=True, allow_nan=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        self._f.close()


def _free(loaded):
    try:
        import gc
        del loaded
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_identity(root, plan):
    """Bind a continuation to the same code, locks, inputs, models and runtime."""
    root = Path(root)
    audio = plan["audio"]
    shard_root = root / audio["shard_dir"]
    shard_hashes = {name: R.sha256_file(shard_root / name) for name in audio.get("shards_present", [])}
    versions = {}
    for package in ("numpy", "scipy", "librosa", "scikit-learn", "soundfile", "torch", "transformers", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    lock_hashes = {p: R.sha256_file(root / path) for p, path in R.LOCKS.items() if (root / path).is_file()}
    code = plan.get("code", {})
    accelerator = None
    if plan.get("cuda_available"):
        import torch
        accelerator = {"cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                       "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    return {"entries": public(plan)["entries"], "splits": plan["splits"], "audio": audio,
            "shard_sha256": shard_hashes, "lock_sha256": lock_hashes, "versions": versions,
            "python": sys.version, "cuda_available": plan.get("cuda_available"),
            "accelerator": accelerator,
            "code": {k: code.get(k) for k in ("src_echoproof_sha256", "runner_sha256")}}


def journal_read(path, expected):
    """Validate an append-only, hash-chained prefix of the expected recording order."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise SealedRunError(f"incomplete journal {path}; manual inspection required")
    records, previous = [], "0" * 64
    try:
        for line in data.splitlines():
            item = json.loads(line)
            rec = item["record"]
            calculated = hashlib.sha256(json.dumps({"previous": previous, "record": rec}, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()
            if item["sha256"] != calculated or len(records) >= len(expected) or rec["recording_id"] != expected[len(records)]:
                raise ValueError("journal hash or recording order mismatch")
            previous = calculated
            records.append(rec)
    except (ValueError, KeyError, TypeError) as exc:
        raise SealedRunError(f"invalid journal {path}; manual inspection required") from exc
    return records


def journal_append(handle, record, previous):
    payload = {"previous": previous, "record": record}
    value = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()
    handle.write(json.dumps({"record": record, "sha256": value}, sort_keys=True, allow_nan=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return value


def _record(row, component, result):
    ok = result is not None and getattr(result, "status", None) == "success"
    return {"recording_id": row.recording_id, "group_id": row.group_id, "component": component, "label": 0 if component == "genuine" else 1,
            "family_policy": row.family_policy, "status": getattr(result, "status", "error") if result is not None else "error",
            "score": getattr(result, "score", None) if ok else None, "assessment": getattr(result, "assessment", None) if ok else None,
            "score_type": getattr(result, "score_type", None), "model_version": getattr(result, "model_version", None),
            "decoded_duration_seconds": getattr(result, "decoded_duration_seconds", None),
            "failure_reason": getattr(result, "failure_reason", None) if result is not None else "audio_or_scoring_error"}


def _write_predictions(path, records):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(records), str(path))


def execute(root, out_dir, *, confirm: bool, plan: dict | None = None, load=None, read_audio=None, predict=None, device=None, cuda=None,
            metrics_hook=None, progress_every: int = 500, resume: bool = False, resume_reason: str | None = None,
            recover_stale_claim: bool = False) -> dict:
    """The sealed run. Order: confirmation, run-folder check, preflight (verified plan, device policy), access log, only then the test rows."""
    root, out_dir = Path(root), Path(out_dir)
    if confirm is not True:
        raise SealedRunError("the sealed run needs the explicit --confirm-sealed-run flag; use `plan` for a dry run")
    if out_dir.exists() and not resume:
        raise SealedRunError(f"the run folder already exists and is never overwritten: {out_dir}")
    if resume and (not out_dir.is_dir() or not resume_reason or not resume_reason.strip()):
        raise SealedRunError("resume requires an existing run and a recorded reason")
    plan = plan if plan is not None else build_plan(root, device=device, cuda=cuda)
    if not plan.get("verified"):
        raise SealedRunError("the plan does not verify (run `plan` and fix every FAIL line); no test row was opened")
    if not plan.get("ready"):
        raise SealedRunError("this machine cannot run the plan (shards or device); no test row was opened")
    cuda = plan.get("cuda_available") if cuda is None else cuda
    for e in plan["entries"]:                                      # preflight: the device policy again, before the log and before any test row
        R.choose_device(e["_spec"], e["device"], cuda=cuda)
    if load is None:
        load = lambda e: R.load_verified(root, e["_spec"], e["_record"], device=e["device"], cuda=cuda)[0]
    if predict is None:
        from echoproof.inference import predict_recording as predict
    identity = run_identity(root, plan)
    if resume:
        try:
            saved_identity = json.loads((out_dir / "run_identity.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SealedRunError("missing or invalid run identity; cannot resume") from exc
        if saved_identity != identity:
            raise SealedRunError("run identity changed; refusing resume before test access")
        status_file = out_dir / "integrity_report.json"
        if status_file.exists():
            status = json.loads(status_file.read_text(encoding="utf-8"))["status"]
            if status != "interrupted":
                raise SealedRunError(f"cannot resume run with status {status}; inspect and record next action")
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
    claim = out_dir / "active_run.claim"
    if claim.exists() and recover_stale_claim and resume:
        # The caller must first verify that no previous worker is active.
        claim.unlink()
    try:
        claim_handle = claim.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise SealedRunError("active run claim exists; verify the prior worker stopped before recovering it") from exc
    claim_handle.write(json.dumps({"pid": os.getpid(), "created_utc": utc_now()}))
    claim_handle.close()
    if not resume:
        (out_dir / "run_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    try:
        log = AccessLog(out_dir / "test_access_log.jsonl", resume=resume)
    except BaseException:
        claim.unlink(missing_ok=True)
        raise
    summary = {"schema_version": SCHEMA_VERSION, "status": "aborted", "started_utc": utc_now(), "bundles": {}, "passed": False}
    try:
        if resume:
            log.event("sealed_run_resume", reason=resume_reason.strip(), stale_claim_recovered=recover_stale_claim,
                      note="continuation rereads the sealed split; no completed prediction is repeated")
        else:
            log.event("sealed_run_start", note="written BEFORE any test row is opened", plan=public(plan), out_dir=str(out_dir), code=plan.get("code"))
        rows_by_protocol, expected_by_protocol = {}, {}
        for protocol in sorted({e["protocol"] for e in plan["entries"]}):
            info = plan["splits"][protocol]
            log.event("opening_test_rows", protocol=protocol, split_file=info["file"], split_file_sha256=info["file_sha256"], via="splits.load_assignments(sealed=True)")
            rows = splits.load_assignments(root / info["file"], sealed=True)
            ordered = sorted(rows, key=lambda r: r.recording_id)
            got = split_digest(protocol, ordered)
            if got != info["expected_split_hash"] or any(r.split_hash != info["expected_split_hash"] for r in rows):
                raise SealedRunError(f"Protocol {protocol}: the split rows do not match the locked split hash")
            test = [r for r in ordered if r.partition == "test"]
            if not test or len({r.recording_id for r in test}) != len(test):
                raise SealedRunError(f"Protocol {protocol}: the test partition is empty or has duplicate recording ids")
            rows_by_protocol[protocol] = test
            expected_by_protocol[protocol] = [r.recording_id for r in test]
            by_component = {}
            for r in test:
                by_component[r.recording_id.split(":")[1]] = by_component.get(r.recording_id.split(":")[1], 0) + 1
            log.event("opened_test_rows", protocol=protocol, n_test=len(test), by_component=by_component, speakers=len({r.group_id for r in test}))
        # Verify every saved prefix before doing any further scoring, including later bundles.
        if resume:
            import pyarrow.parquet as pq
            for entry in plan["entries"]:
                journal = out_dir / f"journal_{entry['protocol']}_{entry['model']}.jsonl"
                saved = journal_read(journal, expected_by_protocol[entry["protocol"]])
                saved_ids = {r["recording_id"] for r in saved}
                pending = [event for event in log.events if event.get("event") == "recording_started"
                           and event.get("key") == entry["key"] and event.get("recording_id") not in saved_ids]
                if pending:
                    raise SealedRunError(f"{entry['key']}: scoring was interrupted before its result was durably saved; inspect before any retry")
                prediction_path = out_dir / f"predictions_{entry['protocol']}_{entry['model']}.parquet"
                if prediction_path.exists() and pq.read_table(str(prediction_path)).to_pylist() != saved:
                    raise SealedRunError(f"existing predictions differ from durable journal: {prediction_path.name}")
        if read_audio is None:
            read_audio = ShardAudio(root / plan["audio"]["shard_dir"])
        for e in plan["entries"]:
            test = rows_by_protocol[e["protocol"]]
            journal = out_dir / f"journal_{e['protocol']}_{e['model']}.jsonl"
            records = journal_read(journal, expected_by_protocol[e["protocol"]])
            log.event("bundle_start", key=e["key"], device=e["device"], n=len(test))
            loaded = None
            try:
                if len(records) < len(test):
                    loaded = load(e)
                previous = json.loads(journal.read_bytes().splitlines()[-1])["sha256"] if records else "0" * 64
                with journal.open("a", encoding="utf-8") as output:
                    for i, row in enumerate(test[len(records):], len(records) + 1):
                        component = row.recording_id.split(":")[1]
                        log.event("recording_started", key=e["key"], recording_id=row.recording_id)
                        try:
                            result = predict(read_audio(row.recording_id), loaded)
                        except Exception as exc:
                            result = None
                            log.event("recording_error", key=e["key"], recording_id=row.recording_id, error=f"{type(exc).__name__}")
                        record = _record(row, component, result)
                        previous = journal_append(output, record, previous)
                        records.append(record)
                        if progress_every and i % progress_every == 0:
                            print(f"  {e['key']}: {i}/{len(test)}", flush=True)
            finally:
                loaded = None
                _free(None)
            report = check_integrity(expected_by_protocol[e["protocol"]], records, threshold=e["decision_rule"]["threshold"])
            name = f"predictions_{e['protocol']}_{e['model']}.parquet"
            prediction_path = out_dir / name
            if prediction_path.exists():
                import pyarrow.parquet as pq
                existing = pq.read_table(str(prediction_path)).to_pylist()
                if existing != records:
                    raise SealedRunError(f"existing predictions differ from durable journal: {name}")
            else:
                temporary = prediction_path.with_suffix(".parquet.tmp")
                _write_predictions(temporary, records)
                os.replace(temporary, prediction_path)
            summary["bundles"][e["key"]] = {"predictions_file": name, "predictions_sha256": R.sha256_file(out_dir / name), "model_version": e["model_version"],
                                            "bundle_json_sha256": e["bundle_json_sha256"], "device": e["device"], "integrity": report}
            log.event("bundle_done", key=e["key"], passed=report["passed"], expected=report["expected"], records=report["records"])
        summary["passed"] = all(b["integrity"]["passed"] for b in summary["bundles"].values()) and len(summary["bundles"]) == len(plan["entries"])
        summary["status"] = "complete" if summary["passed"] else "integrity_failed"
        return summary
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        raise
    finally:
        summary["finished_utc"] = utc_now()
        (out_dir / "integrity_report.json").write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False), encoding="utf-8")
        log.event("sealed_run_end", status=summary["status"], passed=summary["passed"])
        log.close()
        claim.unlink(missing_ok=True)
        hook = metrics_hook if metrics_hook is not None else METRICS_HOOK
        if summary["passed"] and hook is not None:
            hook(out_dir, summary)       # later task: metrics, bootstrap, plots; called only after every integrity check passed


# ---- command line ------------------------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT))
    sub = parser.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--shards", default=DEFAULT_SHARDS)
    common.add_argument("--device", default=None, help="cpu or cuda for the non-AST bundles (AST is CUDA fp16 only)")
    p = sub.add_parser("plan", parents=[common], help="dry run: verify and print; reads no test rows, writes nothing")
    p.add_argument("--quick", action="store_true", help="skip re-hashing the A lock evidence files (file hash and bundles are still verified)")
    p.add_argument("--load-check", action="store_true", help="also load every bundle through its loader (no test data is touched)")
    r = sub.add_parser("run", parents=[common], help="the sealed run")
    r.add_argument("--confirm-sealed-run", action="store_true")
    r.add_argument("--out", default=DEFAULT_OUT)
    r.add_argument("--resume", action="store_true", help="continue an interrupted run without overwriting saved predictions")
    r.add_argument("--resume-reason", help="required explanation recorded before a resumed test read")
    r.add_argument("--recover-stale-claim", action="store_true", help="only after verifying the previous worker is no longer running")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.stage == "plan":
        plan = build_plan(root, shards=args.shards, device=args.device, deep=not args.quick, load_check=args.load_check)
        print(format_plan(plan))
        return 0 if plan["verified"] else 1
    if not args.confirm_sealed_run:
        print("REFUSED: the sealed run needs the explicit --confirm-sealed-run flag; use `plan` for a dry run", file=sys.stderr)
        return 2
    plan = build_plan(root, shards=args.shards, device=args.device)
    try:
        summary = execute(root, root / args.out, confirm=args.confirm_sealed_run, plan=plan, device=args.device,
                          resume=args.resume, resume_reason=args.resume_reason, recover_stale_claim=args.recover_stale_claim)
    except SealedRunError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"sealed run {summary['status']}: see {args.out}/integrity_report.json")
    return 0 if summary["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
