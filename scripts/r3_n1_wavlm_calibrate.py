"""R3-N1 for the selected WavLM head (WAVLM_CALIBRATION.md). Stages run in order, each in the same new run folder under runs/r3/r3_n1_wavlm/.

  audit      create a new run folder, freeze a copy of the recipe, verify the saved exports and the locked manifests, recompute the quoted summaries
  fit        read the calibration role only; compare identity, temperature and affine by leave-one-speaker-out; fit and freeze the selected mapping
  operating  open the operating role once; apply the acceptance rule; choose the threshold; write diagnostics
  package    build the calibrated bundle from the frozen run (runs on this machine; needs no model library)

Nothing here loads a model, trains, reads audio, touches the test partition or modifies an input.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.artifacts import create_run, file_digest  # noqa: E402
from echoproof.evaluation import r3_n1_wavlm as W  # noqa: E402

RECIPE = "configs/r3_n1_wavlm_calibration_b.json"


def run_dir_of(root: Path, recipe: dict, value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    base = (root / recipe["output_root"]).resolve()
    if base not in path.resolve().parents:
        raise SystemExit(f"--run-dir must be a folder under {base}")
    if not path.is_dir():
        raise SystemExit(f"run folder not found: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["audit", "fit", "operating", "package"])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--recipe", default=RECIPE)
    parser.add_argument("--run-dir", default=None, help="the run folder created by `audit` (required for the later stages)")
    parser.add_argument("--owner", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    recipe = W.load_recipe(root / args.recipe)
    if args.command == "audit":
        run_id, run_dir = create_run(root / recipe["output_root"])
        shutil.copyfile(root / args.recipe, run_dir / "recipe.json")
        W.write(run_dir / "environment.json", {**W.environment(recipe), "owner": args.owner, "run_id": run_id, "recipe_file_sha256": file_digest(root / args.recipe)})
        result = W.audit(root, recipe, run_dir)
        print("audit passed:", run_dir)
        print("raw summary at 0.5:", json.dumps(result["raw_summary_at_0_5"]))
        env = result["environment"]
        if env["transformers_matches_pin"] is False:
            print("WARNING: transformers", env["transformers"], "is not the pinned", env["transformers_pin"])
        return
    if not args.run_dir:
        raise SystemExit("--run-dir is required")
    run_dir = run_dir_of(root, recipe, args.run_dir)
    if file_digest(run_dir / "recipe.json") != file_digest(root / args.recipe):
        raise SystemExit("the recipe differs from the frozen copy in the run folder")
    if args.command == "fit":
        result = W.fit_stage(root, recipe, run_dir)
        print("selected mapping:", result["selected_mapping"], result["selected_parameters"])
        print("selection:", json.dumps(result["selection"]["means"]), "best:", result["selection"]["best"], "limit:", result["selection"]["limit"])
    elif args.command == "operating":
        from echoproof.evaluation.r3_n1_wavlm_operating import operating_stage
        result = operating_stage(root, recipe, run_dir)
        print("calibration accepted:", result["threshold"]["calibration_accepted"], "|", result["threshold"]["acceptance"]["reason"])
        print("deployed rule:", result["threshold"]["decision_rule"], "| score type:", result["threshold"]["score_type"])
    else:
        from echoproof.models.r3_n1_wavlm_bundle import package_stage
        result = package_stage(root, recipe, run_dir)
        print("bundle written:", result)


if __name__ == "__main__":
    main()
