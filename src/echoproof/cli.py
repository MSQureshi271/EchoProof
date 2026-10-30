"""EchoProof CLI for Phase 1 configuration and reproducibility checks."""

import argparse
import json
from pathlib import Path
import sys
from . import __version__


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, json.dumps({"status": "error", "type": "invalid_input", "message": message}) + "\n")


def main(argv=None):
    parser = Parser(prog="echoproof", description="EchoProof speech synthesis detection research CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
    def config_options(item):
        item.add_argument("--config", required=True, help="YAML file inside the project")
        item.add_argument("--project-root", default=".", help="Project root (default: current directory)")
        item.add_argument("--run-root", help="Override run root, constrained to project runs/")
        item.add_argument("--device", choices=("cpu", "cuda", "auto"))

    for command, help_text in (("validate-config", "Validate and print resolved fixture settings as JSON"),
                               ("doctor", "Inspect the selected runtime"),
                               ("storage-probe", "Check seeded fixture runs and persistent artifacts")):
        item = sub.add_parser(command, help=help_text)
        config_options(item)
    for command, actions in (("fixtures", ("create",)), ("probe", ("audio", "lr", "bigru"))):
        parent = sub.add_parser(command)
        nested = parent.add_subparsers(dest="action", required=True, parser_class=Parser)
        for action in actions:
            config_options(nested.add_parser(action))
    benchmark = sub.add_parser("benchmark", help="Profile LR and BiGRU on generated fixtures")
    config_options(benchmark)
    benchmark.add_argument("--models", nargs="+", default=["lr", "bigru"])
    benchmark.add_argument("--batch-size", type=int, default=4)
    benchmark.add_argument("--accumulation-steps", type=int, default=1)
    benchmark.add_argument("--lr-fit-rows", type=int, default=256)
    budget = sub.add_parser("budget", help="Estimate a baseline budget using measured timings")
    config_options(budget)
    budget.add_argument("--benchmark", required=True)
    budget.add_argument("--plan", default="configs/baseline_budget.json")
    export = sub.add_parser("export-environment", help="Export an immutable baseline lock directory")
    export.add_argument("--project-root", default=".")
    export.add_argument("--output", required=True, help="New directory for the lock and provenance")
    verify = sub.add_parser("verify-environment", help="Compare a fresh Colab runtime with a baseline lock")
    verify.add_argument("--project-root", default=".")
    verify.add_argument("--lock", required=True, help="Exported requirements-lock.txt with sibling provenance")
    args = parser.parse_args(argv)
    if args.command in ("export-environment", "verify-environment"):
        try:
            from .environment_lock import export_environment, verify_environment
            root = Path(args.project_root).resolve()
            if args.command == "export-environment":
                output = Path(args.output)
                result = export_environment(root, output if output.is_absolute() else root / output)
            else:
                lock = Path(args.lock)
                lock = lock if lock.is_absolute() else root / lock
                if lock.name != "requirements-lock.txt":
                    raise ValueError("--lock must select the exported requirements-lock.txt")
                result = verify_environment(root, lock.parent)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return 0 if result["status"] != "needs_attention" else 4
        except Exception as exc:
            print(json.dumps({"status": "error", "type": "environment_failed", "message": str(exc)}), file=sys.stderr)
            return 4
    try:
        from .config import ConfigError, load_config
    except ImportError as exc:
        print(json.dumps({"status": "error", "type": "dependency_unavailable", "message": str(exc)}), file=sys.stderr)
        return 3
    try:
        root = Path(args.project_root).resolve()
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = root / config_path
        config = load_config(config_path, root, run_root=args.run_root, device=args.device)
    except ConfigError as exc:
        print(json.dumps({"status": "error", "type": "invalid_config", "message": str(exc)}), file=sys.stderr)
        return 2
    if args.command == "validate-config":
        print(config.to_json(), end="")
        return 0
    try:
        if args.command == "doctor":
            from .environment import capture_environment
            result = capture_environment(root)
        elif args.command == "storage-probe":
            from .reproducibility import storage_probe
            result = storage_probe(config, root)
        elif args.command == "fixtures":
            from .data.fixtures import create_fixtures
            fixture_dir, manifest = create_fixtures(config.paths.fixtures, config.seed)
            result = {"status": "success", "fixture_dir": str(fixture_dir), "manifest": manifest}
        elif args.command == "benchmark":
            from .benchmark import run_benchmark
            result = run_benchmark(config, root, models=args.models, batch_size=args.batch_size,
                                   accumulation_steps=args.accumulation_steps, lr_fit_rows=args.lr_fit_rows)
        elif args.command == "budget":
            from .budget import run_budget
            def project_path(value):
                path = Path(value)
                return path if path.is_absolute() else root / path
            result = run_budget(config, root, project_path(args.benchmark), project_path(args.plan))
        elif args.action == "audio":
            from .data.audio_probe import probe_audio
            result = probe_audio(config, root)
        else:
            from .models.fixture_probes import run_probe
            result = run_probe(args.action, config, root)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except ImportError as exc:
        print(json.dumps({"status": "error", "type": "dependency_unavailable", "message": str(exc)}), file=sys.stderr)
        return 3
    except Exception as exc:
        print(json.dumps({"status": "error", "type": "probe_failed", "message": str(exc)}), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
