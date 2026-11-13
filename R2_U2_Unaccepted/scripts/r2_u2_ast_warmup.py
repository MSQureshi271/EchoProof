"""R2-U2 metadata plan, frozen AST head warm-up, and artifact integrity check.

No command installs packages. Research training requires a source-bound passing
test report and a prepared Colab GPU. See docs/r2_u2_ast_warmup.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
import uuid

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))


def inside(root, value):
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"output/input reference must be inside the selected project: {path}")
    return path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("plan", "train", "verify"))
    result.add_argument("--root", required=True)
    result.add_argument("--run-dir", help="persistent directory inside project; required for train/verify")
    result.add_argument("--shard-dir", help="canonical shards staged on Colab scratch; required for train")
    result.add_argument("--model-cache", help="download cache on Colab scratch")
    result.add_argument("--test-report", help="passing source-bound notebook test report inside project")
    result.add_argument("--resume", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.root).resolve()
    if root != SOURCE_ROOT or not (root / "pyproject.toml").is_file():
        raise ValueError("--root must select this script's own project copy with pyproject.toml")
    from echoproof.artifacts import write_json
    report = {"task": "R2-U2", "action": args.action, "verified": False,
              "created_utc": datetime.now(timezone.utc).isoformat()}
    exit_code = 0
    try:
        from echoproof.training import r2_u2
        if args.action == "plan":
            recipe = r2_u2.load_recipe(root)
            plan = r2_u2.preflight(root, recipe)
            print(json.dumps({"task": "R2-U2", "status": "metadata_only",
                              "counts": plan.counts, "plan_sha256": plan.sha256,
                              "required_shards": plan.shards()}, indent=2))
            return 0
        if not args.run_dir:
            raise ValueError("--run-dir is required")
        run_dir = inside(root, args.run_dir)
        if args.action == "verify":
            report.update(r2_u2.verify_run(root, run_dir))
        else:
            if not args.shard_dir or not args.test_report:
                raise ValueError("--shard-dir and --test-report are required for training")
            report.update(r2_u2.run_warmup(
                root, Path(args.shard_dir).resolve(), run_dir, inside(root, args.test_report),
                resume=args.resume,
                model_cache=Path(args.model_cache).resolve() if args.model_cache else None))
            if report.get("verified") is not True:
                raise RuntimeError("warm-up did not establish all publication gates")
    except BaseException as exc:
        report.update(verified=False, status="failed", error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        exit_code = 1
    reports = root / "reports/r2"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / ("r2_u2_" + args.action + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                      + "_" + uuid.uuid4().hex[:8] + ".json")
    write_json(path, report)
    print(json.dumps({"status": report.get("status"), "report": str(path),
                      "verified": report.get("verified"), "error": report.get("error")}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
