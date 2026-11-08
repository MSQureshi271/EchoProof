"""R2-U1 planning, pinned WavLM extraction and independent cache verification.

Run from a prepared Colab GPU environment; this runner never installs packages.
``plan`` opens metadata only. ``build`` may download pinned model files into the
explicit model-cache directory, then reads only the permitted canonical audio.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import uuid

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from echoproof.artifacts import file_digest, write_json  # noqa: E402

TRANSFORMERS_VERSION = "5.22.0"  # recorded R1-U1/U2 environment, not a latest-version claim
SOURCE_FILES = (
    "src/echoproof/features/r2_u1.py", "src/echoproof/features/pretrained.py",
    "src/echoproof/data/cohort.py", "src/echoproof/data/chunks.py",
    "src/echoproof/contracts.py", "src/echoproof/artifacts.py",
    "scripts/r2_u1_wavlm_cache.py", "tests/test_r2_wavlm_cache.py",
    "tests/test_r1_u1_wavlm.py",
)


def source_hashes(root):
    """Bind evidence to the files actually imported by this runner."""
    if (Path(root) / "src").resolve() != (SOURCE_ROOT / "src").resolve():
        raise ValueError("run the script inside the same project selected by --root")
    return {name: file_digest(Path(root) / name) for name in SOURCE_FILES}


def inside(root, value, default):
    path = Path(value) if value is not None else root / default
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"output/report path must stay inside the selected project: {path}")
    return path


def runtime_versions():
    return {name: importlib.metadata.version(name) for name in (
        "numpy", "pyarrow", "soundfile", "torch", "transformers", "huggingface-hub"
    )}


def observed_extractor(args, hashes):
    import torch
    import transformers
    from echoproof.features.pretrained import PRETRAINED, WavLMExtractor, pinned_snapshot
    from echoproof.features.r2_u1 import extraction_identity

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"R1 verified transformers=={TRANSFORMERS_VERSION}; found {transformers.__version__}. "
            "Prepare that environment and restart the runtime before continuing. No packages were changed."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("select a Colab GPU runtime; CUDA is not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_cache = Path(args.model_cache).resolve() if args.model_cache else Path(args.shard_dir).resolve().parent / "model-cache"
    snapshot = pinned_snapshot("wavlm", cache_dir=model_cache)
    config = json.loads((snapshot / "preprocessor_config.json").read_text(encoding="utf-8"))
    for key, expected in PRETRAINED["wavlm"]["expected_preprocessing"].items():
        if config.get(key) != expected:
            raise RuntimeError(f"pinned processor disagrees on {key}: {config.get(key)!r}")
    extractor = WavLMExtractor.from_pinned(device=args.device, batch_size=args.batch_size, cache_dir=model_cache)
    description = extractor.describe()
    if description["hidden_size"] != 768 or description["trainable_parameters"] != 0 or description["training_mode"]:
        raise RuntimeError(f"extractor is not the frozen R1 WavLM: {description}")
    if any(p.dtype != torch.float32 for p in extractor.encoder.parameters()):
        raise RuntimeError("WavLM extraction must use FP32 parameters")
    for key, expected in PRETRAINED["wavlm"]["expected_preprocessing"].items():
        if getattr(extractor.feature_extractor, key, None) != expected:
            raise RuntimeError(f"loaded processor disagrees on {key}")
    versions = runtime_versions()
    identity = extraction_identity(
        processor_config=config, versions=versions,
        source_sha256={key: hashes[key] for key in (
            "src/echoproof/features/r2_u1.py", "src/echoproof/features/pretrained.py"
        )}, hidden_size=extractor.dim,
    )
    environment = {
        "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
        "device": args.device, "cuda": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "extractor": description,
        "processor_file_sha256": file_digest(snapshot / "preprocessor_config.json"),
        "weights_sha256": file_digest(snapshot / PRETRAINED["wavlm"]["weights_file"]),
        "model_cache": str(model_cache),
    }
    return extractor, identity, environment


def checked_test_report(path, root, hashes):
    if path is None:
        return {"verified": False, "reason": "no source-bound fixture test report supplied"}
    test_path = inside(root, path, "reports/r2/unused.json")
    report = json.loads(test_path.read_text(encoding="utf-8"))
    if report.get("returncode") != 0 or report.get("skipped", True) or not report.get("passed_count", 0):
        raise RuntimeError("fixture test report must record passing tests, zero skips and exit code zero")
    if report.get("source_sha256") != hashes:
        raise RuntimeError("fixture tests did not run against the current source files")
    return {"verified": True, "path": str(test_path.relative_to(root)), "sha256": file_digest(test_path),
            "passed_count": report["passed_count"], "summary": report.get("summary")}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("plan", "build", "verify"):
        item = sub.add_parser(action)
        item.add_argument("--root", required=True, help="existing project root; no Drive path is assumed")
        item.add_argument("--out-dir", help="inside the project; default data/features/r2_u1_B")
        item.add_argument("--include-test-genuine", action="store_true",
                          help="acknowledge R2-U1 extraction-only access to B-test genuine; no test scores/metrics")
        if action == "build":
            item.add_argument("--shard-dir", required=True, help="staged canonical shards on Colab scratch")
            item.add_argument("--model-cache", help="explicit model cache; default sibling of --shard-dir")
            item.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            item.add_argument("--batch-size", type=int, default=16)
            item.add_argument("--resume", action="store_true")
            item.add_argument("--test-report", help="source-bound report produced by the notebook fixture-test cell")
    return p


def run(args):
    """Run one phase and return its evidence; failure evidence is written by main."""
    from echoproof.features import r2_u1

    root = Path(args.root).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("--root must contain the EchoProof project")
    hashes = source_hashes(root)
    out_dir = inside(root, args.out_dir, "data/features/r2_u1_B")
    report = {"task": "R2-U1", "action": args.action, "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashes, "include_test_genuine": args.include_test_genuine,
              "score_parity_kind": "untrained diagnostic head; not detector accuracy",
              "trained_head_parity": "pending R3-U1", "out_dir": str(out_dir.relative_to(root))}
    if args.action == "plan":
        plan = r2_u1.preflight(root, include_test_genuine=args.include_test_genuine)
        report.update(plan_sha256=plan.sha256, counts=plan.counts, shards=plan.shards(),
                      status="planned_metadata_only", verified=False)
        return report
    if args.action == "verify":
        manifest_path = out_dir / "r2_u1_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plan = r2_u1.preflight(root, identity=manifest["identity"], include_test_genuine=args.include_test_genuine)
        verification = r2_u1.verify_cache(plan, out_dir)
        report.update(plan_sha256=plan.sha256, counts=plan.counts, verification=verification,
                      status="cache_integrity_checked", verified=False,
                      note="integrity check only; real GPU/test/parity evidence remains in the build report")
        return report
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    tests = checked_test_report(args.test_report, root, hashes)
    # Validate frozen metadata before downloading a checkpoint or opening any audio.
    r2_u1.preflight(root, include_test_genuine=args.include_test_genuine)
    start = time.perf_counter()
    extractor, identity, environment = observed_extractor(args, hashes)
    plan = r2_u1.preflight(root, identity=identity, include_test_genuine=args.include_test_genuine)
    manifest = r2_u1.build(plan, Path(args.shard_dir).resolve(), out_dir, extractor,
                           resume=args.resume, batch_size=args.batch_size, parity_n=50,
                           fixture=False, progress=lambda message: print(message, flush=True))
    verification = r2_u1.verify_cache(plan, out_dir)
    report.update(plan_sha256=plan.sha256, counts=plan.counts, environment=environment,
                  tests=tests, manifest=manifest, verification=verification,
                  seconds_this_session=round(time.perf_counter() - start, 3),
                  verified=False, status="cache_built_pending_review")
    # The manifest's parity/verification fields are the publication gate. A fixture
    # or CPU run is never elevated into real GPU evidence here.
    parity = manifest.get("parity", {})
    cache_gate = (
        manifest.get("status") == "extracted_pending_independent_review"
        and manifest.get("full_R2_U1_scope") is True
        and manifest.get("verification", {}).get("passed") is True
        and verification.get("passed") is True
        and parity.get("passed") is True and parity.get("items") == 50
        and parity.get("scope") == "development only"
    )
    report["verified"] = bool(tests["verified"] and args.device == "cuda" and args.include_test_genuine and cache_gate)
    report["cache_manifest_created_utc"] = manifest.get("created_utc")
    report["parity_note"] = "parity belongs to the committed cache generation; resume may reuse previously verified parity"
    if report["verified"]:
        report["status"] = "colab_cache_verified_pending_independent_review"
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.root).resolve()
    # Only write evidence under a recognised project. Incorrect roots are reported
    # to stderr without creating a directory at an arbitrary location.
    if not (root / "pyproject.toml").is_file():
        print(json.dumps({"status": "failed", "error": "invalid project root"}), file=sys.stderr)
        return 2
    reports = root / "reports" / "r2"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_path = reports / f"r2_u1_{args.action}_{stamp}_{uuid.uuid4().hex[:8]}.json"
    code = 0
    try:
        report = run(args)
    except Exception as exc:
        report = {"task": "R2-U1", "action": args.action, "status": "failed", "verified": False,
                  "error_type": type(exc).__name__, "error": str(exc),
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        code = 1
    write_json(evidence_path, report)
    print(json.dumps({"status": report["status"], "verified": report.get("verified", False),
                      "evidence": str(evidence_path), "error": report.get("error")}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
