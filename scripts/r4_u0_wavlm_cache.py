"""R4-U0: extract the Protocol A WavLM rows the B cache lacks (`plan`, `build`, `verify`).

Run from a prepared Colab GPU environment; this runner never installs packages. `plan` opens metadata only. `build` downloads the pinned model
into the explicit model-cache directory, then reads only the shards the gap needs. No A test row is planned or extracted.
The pinned-extractor setup, test-report check and path guard are the R2-U1 runner's (`scripts/r2_u1_wavlm_cache.py`), loaded from that file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from echoproof.artifacts import file_digest, write_json  # noqa: E402

SOURCE_FILES = (
    "src/echoproof/features/r4_u0.py", "src/echoproof/features/r2_u1.py", "src/echoproof/features/pretrained.py",
    "src/echoproof/features/r4_s1.py", "src/echoproof/data/cohort.py", "src/echoproof/data/chunks.py", "src/echoproof/data/splits.py",
    "src/echoproof/contracts.py", "src/echoproof/artifacts.py", "scripts/r2_u1_wavlm_cache.py", "scripts/r4_u0_wavlm_cache.py",
    "tests/test_r4_u0_wavlm_cache.py", "tests/test_r2_wavlm_cache.py", "tests/test_r1_u1_wavlm.py",
)


def _r2_runner():
    spec = importlib.util.spec_from_file_location("r2_u1_runner", SOURCE_ROOT / "scripts/r2_u1_wavlm_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hashes(root):
    if (Path(root) / "src").resolve() != (SOURCE_ROOT / "src").resolve():
        raise ValueError("run the script inside the same project selected by --root")
    return {name: file_digest(Path(root) / name) for name in SOURCE_FILES}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("plan", "build", "verify"):
        item = sub.add_parser(action)
        item.add_argument("--root", required=True, help="existing project root; no Drive path is assumed")
        item.add_argument("--out-dir", help="inside the project; default data/features/r4_u0_A")
        if action == "build":
            item.add_argument("--shard-dir", required=True, help="staged canonical shards on Colab scratch")
            item.add_argument("--model-cache", help="explicit model cache; default sibling of --shard-dir")
            item.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            item.add_argument("--batch-size", type=int, default=16)
            item.add_argument("--resume", action="store_true")
            item.add_argument("--test-report", help="source-bound report produced by the notebook fixture-test cell")
    return p


def run(args):
    from echoproof.features import r4_u0
    r2 = _r2_runner()
    root = Path(args.root).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("--root must contain the EchoProof project")
    hashes = source_hashes(root)
    out_dir = r2.inside(root, args.out_dir, "data/features/r4_u0_A")
    report = {"task": "R4-U0", "action": args.action, "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes,
              "out_dir": str(out_dir.relative_to(root)), "test_rows": "none; A test is sealed"}
    if args.action == "plan":
        plan = r4_u0.preflight(root)
        report.update(plan_sha256=plan.sha256, counts=plan.counts, coverage=plan.provenance["coverage"], shards=plan.shards(),
                      status="planned_metadata_only", verified=False)
        return report
    if args.action == "verify":
        manifest = json.loads((out_dir / r4_u0.MANIFEST_NAME).read_text(encoding="utf-8"))
        plan = r4_u0.preflight(root, identity=manifest["identity"])
        verification = r4_u0.verify_cache(plan, out_dir)
        rows, _ = r4_u0.read_embeddings_a(root, a_dir=out_dir)
        report.update(plan_sha256=plan.sha256, counts=plan.counts, verification=verification, union_rows_read=len(rows),
                      status="cache_integrity_checked", verified=False,
                      note="integrity check and a full read of the A union; real GPU and parity evidence stays in the build report")
        return report
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    tests = r2.checked_test_report(args.test_report, root, hashes)
    r4_u0.preflight(root)                                   # frozen metadata is validated before any model download or audio read
    start = time.perf_counter()
    extractor, identity, environment = r2.observed_extractor(args, hashes)
    plan = r4_u0.preflight(root, identity=identity)
    b_index = json.loads((root / r4_u0.B_DIR / "development_index.json").read_text(encoding="utf-8"))["rows"]
    manifest = r4_u0.build(plan, Path(args.shard_dir).resolve(), out_dir, extractor, b_dir=root / r4_u0.B_DIR, b_index_rows=b_index,
                           resume=args.resume, batch_size=args.batch_size, fixture=False, progress=lambda m: print(m, flush=True))
    verification = r4_u0.verify_cache(plan, out_dir)
    rows, _ = r4_u0.read_embeddings_a(root, a_dir=out_dir)
    report.update(plan_sha256=plan.sha256, counts=plan.counts, coverage=plan.provenance["coverage"], environment=environment, tests=tests,
                  manifest=manifest, verification=verification, union_rows_read=len(rows),
                  seconds_this_session=round(time.perf_counter() - start, 3), verified=False, status="cache_built_pending_review")
    gate = (manifest.get("status") == "extracted_pending_independent_review" and manifest["verification"]["passed"] is True
            and verification["passed"] is True and manifest["parity"]["passed"] is True and manifest["parity"]["items"] == 50
            and manifest["b_parity"]["passed"] is True and manifest["b_parity"]["items"] == 50
            and manifest["test_rows_extracted"] == 0 and len(rows) == plan.provenance["coverage"]["a_rows"])
    report["verified"] = bool(tests["verified"] and args.device == "cuda" and gate)
    if report["verified"]:
        report["status"] = "colab_cache_verified_pending_independent_review"
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.root).resolve()
    if not (root / "pyproject.toml").is_file():
        print(json.dumps({"status": "failed", "error": "invalid project root"}), file=sys.stderr)
        return 2
    reports = root / "reports" / "r4"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_path = reports / f"r4_u0_{args.action}_{stamp}_{uuid.uuid4().hex[:8]}.json"
    code = 0
    try:
        report = run(args)
    except Exception as exc:
        report = {"task": "R4-U0", "action": args.action, "status": "failed", "verified": False, "error_type": type(exc).__name__,
                  "error": str(exc), "created_utc": datetime.now(timezone.utc).isoformat()}
        code = 1
    write_json(evidence_path, report)
    print(json.dumps({"status": report["status"], "verified": report.get("verified", False), "evidence": str(evidence_path),
                      "error": report.get("error")}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
