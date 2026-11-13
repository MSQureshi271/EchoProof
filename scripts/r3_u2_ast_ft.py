"""R3-U2: AST partial fine-tuning from the R2-U2 warm-up checkpoint (Protocol B). Run on a Colab GPU.

Trains the baseline and one alternative for the declared budget (the last two blocks, the final norm and the head), writes adaptation evidence per run,
freezes the selection, then exports raw validation predictions and a paired warm-up versus adapted comparison. Reads Protocol B training audio and the
three validation roles only: no test and no Chatterbox audio. Run directories are never reused; `--resume` continues an interrupted run only if the
frozen experiment contract is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.artifacts import file_digest  # noqa: E402

RECIPE = "configs/r3_u2_ast_ft_b.json"
SOURCES = ("scripts/r3_u2_ast_ft.py", "configs/r3_u2_ast_ft_b.json", "src/echoproof/training/r3_u2_data.py", "src/echoproof/training/r3_u2_train.py",
           "src/echoproof/training/r3_m1_data.py", "src/echoproof/training/r3_m1_selection.py", "src/echoproof/training/r3_m1_export.py",
           "src/echoproof/training/trainer.py", "src/echoproof/training/datasets.py", "src/echoproof/training/seeding.py",
           "src/echoproof/models/ast_detector.py", "src/echoproof/features/pretrained.py", "src/echoproof/features/r2_s1.py",
           "src/echoproof/evaluation/aggregate.py", "src/echoproof/evaluation/metrics.py")


def freeze_contract(output: Path, contract: dict, resume: bool):
    path = output / "experiment_contract.json"
    if resume:
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != contract:
            raise SystemExit("resume refused: the experiment contract or provenance changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(contract, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
    return path


def run(args):
    import numpy as np
    import torch
    from echoproof.features.pretrained import pinned_snapshot
    from echoproof.training import r3_u2_train as T
    from echoproof.training.r3_u2_data import build_data

    root = args.root.resolve()
    if (root / "src").resolve() != (ROOT / "src").resolve():
        raise SystemExit("run the script inside the project selected by --root")
    if not torch.cuda.is_available():
        raise SystemExit("R3-U2 requires a CUDA runtime (fp16 training)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    recipe = T.load_recipe(root / RECIPE)
    checkpoint = root / (args.checkpoint or recipe["init"]["checkpoint"])
    if not checkpoint.is_file():
        raise SystemExit(f"warm-up checkpoint not found: {checkpoint}")
    if file_digest(checkpoint) != recipe["init"]["checkpoint_sha256"]:
        raise SystemExit("the warm-up checkpoint differs from the pinned SHA-256")
    snapshot = pinned_snapshot("ast", cache_dir=args.model_cache)
    data = build_data(root, args.shard_dir, seed=recipe["train"]["seed"])
    try:
        contract = {"task": T.TASK, "protocol": "B", "recipe_sha256": T.recipe_hash(recipe), "recipe_status": recipe["status"],
                    "split_sha256": data["split_sha256"], "inputs": data["input_sha256"], "init_checkpoint_sha256": recipe["init"]["checkpoint_sha256"],
                    "counts": data["counts"], "sources": {p: file_digest(root / p) for p in SOURCES},
                    "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                                    "transformers": __import__("transformers").__version__}}
        output = args.output.resolve()
        contract_path = freeze_contract(output, contract, args.resume)
        summaries = {}
        for name in T.EXPERIMENTS:
            summaries[name] = T.run_experiment(name, recipe, data, checkpoint=checkpoint, snapshot=snapshot, output=output, ledger=args.ledger,
                                               owner=args.owner, resume=args.resume, provenance={"contract_sha256": file_digest(contract_path)},
                                               progress=lambda *m: print(*m, flush=True))
            print(name, "best epoch", summaries[name]["best_epoch"], "macro F1", summaries[name]["best_macro_f1"], flush=True)
        evidence = T.finish(output, data, summaries, recipe, snapshot=snapshot, device="cuda", contract_path=contract_path)
        print("selected:", evidence["selection"]["selected"], "| evidence:", output / "evidence.json")
        return evidence
    finally:
        data["loader"].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new run folder on Drive")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--checkpoint", default=None, help="warm-up best.pt relative to --root (default: the recipe's)")
    parser.add_argument("--model-cache", default=None, help="Hugging Face cache folder for the pinned AST")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
