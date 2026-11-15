"""Run a predeclared Protocol A R4-N1 calibration model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echoproof.evaluation.r4_n1 import R4N1Error, run_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("lr", "bigru"), required=True, help="Protocol A model to calibrate")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: current directory)")
    parser.add_argument("--config", type=Path, default=Path("configs/r4_n1_calibration_a.json"), help="Predeclared R4-N1 recipe")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional output override; defaults to the model-specific R4-N1 run directory; existing paths are refused")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else args.root / args.config
    try:
        result = run_model(args.root, config, args.model, args.run_dir)
    except (R4N1Error, FileExistsError, OSError, ValueError) as exc:
        parser.exit(2, f"R4-N1 failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

