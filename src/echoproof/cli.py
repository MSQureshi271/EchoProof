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
    predict = sub.add_parser("predict", help="Run inference on an audio recording using a model bundle")
    predict.add_argument("--bundle", required=True, help="Path to model bundle directory")
    predict.add_argument("--audio", required=True, help="Path to audio file")
    predict.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    data = sub.add_parser('data', help='Dataset audit operations')
    data_actions = data.add_subparsers(dest='data_action', required=True, parser_class=Parser)
    inventory = data_actions.add_parser('inventory')
    inventory.add_argument('--source', required=True, choices=['phonemedf'])
    inventory.add_argument('--project-root', default='.')
    inventory.add_argument('--config', default='configs/phase2.yaml')
    inventory.add_argument('--resume', help='Existing inventory run ID')
    adapt = data_actions.add_parser('adapt')
    adapt.add_argument('--source', required=True, choices=['phonemedf'])
    adapt.add_argument('--project-root', default='.')
    adapt.add_argument('--config', default='configs/phase2.yaml')
    adapt.add_argument('--inventory', required=True, help='Completed inventory run directory')
    pair = data_actions.add_parser('pair')
    pair.add_argument('--source', required=True, choices=['phonemedf'])
    pair.add_argument('--project-root', default='.')
    pair.add_argument('--config', default='configs/phase2.yaml')
    pair.add_argument('--manifest', required=True, help='P2-05 snapshot directory')
    inspect = data_actions.add_parser('inspect')
    inspect.add_argument('--source', default='phonemedf', choices=['phonemedf'])
    inspect.add_argument('--project-root', default='.')
    inspect.add_argument('--config', default='configs/phase2.yaml')
    inspect.add_argument('--manifest', required=True)
    inspect.add_argument('--pairing', required=True)
    inspect.add_argument('--scope', default='representative', choices=['representative'])
    inspect.add_argument('--resume')
    reconcile = data_actions.add_parser('reconcile')
    reconcile.add_argument('--source', default='phonemedf', choices=['phonemedf'])
    reconcile.add_argument('--project-root', default='.')
    reconcile.add_argument('--config', default='configs/phase2.yaml')
    reconcile.add_argument('--manifest', required=True)
    reconcile.add_argument('--pairing', required=True)
    reconcile.add_argument('--inspection', required=True)
    capacity = data_actions.add_parser('capacity')
    capacity.add_argument('--project-root', default='.')
    capacity.add_argument('--config', default='configs/phase2.yaml')
    capacity.add_argument('--drive-free-gb', type=float)
    capacity.add_argument('--drive-total-gb', type=float)
    capacity.add_argument('--drive-used-gb', type=float)
    capacity.add_argument('--remaining-hours', type=float)
    capacity.add_argument('--scratch-path')
    capacity.add_argument('--planned-download-bytes', type=int, default=0)
    capacity.add_argument('--expanded-bytes', type=int, default=0)
    capacity.add_argument('--staging-bytes', type=int, default=0)
    capacity.add_argument('--cache-bytes', type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == 'data':
        try:
            from .phase2_config import load_phase2_config
            root = Path(args.project_root).resolve()
            config_path = Path(args.config)
            if not config_path.is_absolute(): config_path = root / config_path
            config = load_phase2_config(config_path, root)
            if args.data_action == 'inventory':
                from .data.inventory import run_inventory
                result = run_inventory(root, config, config_path, root / config['source_registry'], args.resume)
            elif args.data_action == 'adapt':
                from .data.phonemedf_adapter import create_snapshot
                selected = Path(args.inventory)
                result = create_snapshot(root, config, config_path, selected if selected.is_absolute() else root / selected)
            elif args.data_action == 'pair':
                from .data.pairing import run_pairing
                selected = Path(args.manifest)
                result = run_pairing(root, config, selected if selected.is_absolute() else root / selected)
            else:
                from .data.phase2_quality import inspect as inspect_data, reconcile as reconcile_data, capacity as capacity_data
                def project_path(value):
                    path = Path(value)
                    return path if path.is_absolute() else root / path
                if args.data_action == 'inspect':
                    result = inspect_data(root, config, project_path(args.manifest), project_path(args.pairing), args.resume)
                elif args.data_action == 'reconcile':
                    result = reconcile_data(root, config, project_path(args.manifest), project_path(args.pairing), project_path(args.inspection))
                else:
                    result = capacity_data(root, drive_free_gb=args.drive_free_gb, drive_total_gb=args.drive_total_gb,
                        drive_used_gb=args.drive_used_gb, remaining_hours=args.remaining_hours, scratch_path=args.scratch_path,
                        planned_download_bytes=args.planned_download_bytes, expanded_bytes=args.expanded_bytes,
                        staging_bytes=args.staging_bytes, cache_bytes=args.cache_bytes)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return 0 if result['status'] in ('complete','ready_for_review') else 4
        except Exception as exc:
            print(json.dumps({'status':'error','type':args.data_action+'_failed','message':str(exc)}), file=sys.stderr)
            return 4
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
    if args.command == "predict":
        if args.threshold < 0.0 or args.threshold > 1.0:
            print(json.dumps({"status": "error", "type": "invalid_input", "message": "--threshold must be between 0.0 and 1.0"}), file=sys.stderr)
            return 2
        bundle_path = Path(args.bundle)
        if not bundle_path.exists():
            print(json.dumps({"status": "error", "type": "bundle_error", "message": f"Bundle path does not exist: {args.bundle}"}), file=sys.stderr)
            return 2
        try:
            from .inference import load_lr_bundle, predict_recording
            bundle = load_lr_bundle(bundle_path, threshold=args.threshold)
        except Exception as exc:
            print(json.dumps({"status": "error", "type": "bundle_error", "message": f"Failed to load bundle: {exc}"}), file=sys.stderr)
            return 4

        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(json.dumps({"status": "error", "type": "audio_not_found", "message": f"Audio file does not exist: {args.audio}"}), file=sys.stderr)
            return 2
        try:
            audio_bytes = audio_path.read_bytes()
        except Exception as exc:
            print(json.dumps({"status": "error", "type": "audio_read_error", "message": f"Failed to read audio file: {exc}"}), file=sys.stderr)
            return 2

        try:
            result = predict_recording(audio_bytes, bundle)
        except Exception as exc:
            print(json.dumps({"status": "error", "type": "inference_failed", "message": str(exc)}), file=sys.stderr)
            return 4

        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
        return 0 if result.status == "success" else 4
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
