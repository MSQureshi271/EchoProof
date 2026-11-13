"""R3-U2: package the selected fine-tuned AST as a verified inference bundle and check the adapter against the exports. Run on a Colab GPU.

  package  convert the selected best.pt (named by the frozen selection.json) to safetensors with the pinned config and extractor, verify the conversion
           and the frozen/unfrozen checksums, and write the bundle (a new folder only)
  parity   score 50 deterministic selection-role chunks from the raw shard audio with the bundle adapter and compare with the exported probabilities:
           the fp16 CUDA adapter is the gate; fp32 on CUDA and on CPU are measured and reported

Reads Protocol B selection-role audio only (verified against the cohort manifest). No training, calibration or lock update.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.artifacts import file_digest, write_json  # noqa: E402

RECIPE = "configs/r3_u2_ast_ft_b.json"
BUNDLE = "bundles/r3_u2_ast_partial_B"
SOURCES = ("scripts/r3_u2_ast_bundle.py", "src/echoproof/models/ast_adapter.py", "src/echoproof/models/r3_u2_bundle.py",
           "src/echoproof/models/ast_detector.py", "src/echoproof/training/r3_u2_parity.py", "src/echoproof/training/r3_u2_data.py",
           "src/echoproof/features/pretrained.py")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def environment():
    import numpy
    import torch
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
            "transformers": transformers.__version__, "platform": platform.platform()}


def selection_of(run_dir: Path):
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    winner = selection["selected"]
    return selection, winner, run_dir / f"{run_dir.name}-{winner}" / "best.pt"


def cmd_package(root, args):
    import torch
    from echoproof.features.pretrained import pinned_snapshot
    from echoproof.models.r3_u2_bundle import package_ast_detector
    from echoproof.training import r3_u2_train as T

    run_dir = (root / args.run_dir).resolve()
    selection, winner, checkpoint = selection_of(run_dir)
    recipe = T.load_recipe(root / RECIPE)
    if selection["recipe_sha256"] != T.recipe_hash(recipe):
        raise SystemExit("the recipe changed since the selection was frozen")
    if file_digest(checkpoint) != selection["checkpoint_sha256"][winner]:
        raise SystemExit("the selected checkpoint changed since the selection was frozen")
    meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
    snapshot = pinned_snapshot("ast", cache_dir=args.model_cache)
    bundle = package_ast_detector(checkpoint, snapshot, root / args.bundle, bundle_id=f"bundle:r3-u2-ast-partial-B-{winner}", protocol="B",
                                  expected_checkpoint_sha256=selection["checkpoint_sha256"][winner], split_sha256=selection["split_sha256"],
                                  expected_epoch=selection["selected_epoch"], expected_config_hash=meta["config_hash"],
                                  recipe_sha256=selection["recipe_sha256"], warmup_checkpoint_sha256=selection["init_checkpoint_sha256"],
                                  pretrained_group_sha256=recipe["init"]["pretrained_group_sha256"],
                                  unfrozen_blocks=recipe["model"]["unfrozen_blocks"], experiment=winner)
    print("bundle written:", root / args.bundle, "| experiment:", winner, "| epoch:", selection["selected_epoch"])
    return bundle


def cmd_parity(root, args):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from echoproof.models.ast_adapter import ASTBundleAdapter
    from echoproof.training import r3_u2_parity as P
    from echoproof.training.r3_u2_data import build_data

    if not torch.cuda.is_available():
        raise SystemExit("the parity gate needs a CUDA runtime")
    torch.backends.cuda.matmul.allow_tf32 = False                      # fp32 must really be fp32
    torch.backends.cudnn.allow_tf32 = False
    run_dir = (root / args.run_dir).resolve()
    selection, winner, checkpoint = selection_of(run_dir)
    bundle_dir = root / args.bundle
    data = build_data(root, args.shard_dir, seed=42)
    try:
        dataset = data["validation"]["selection"]
        indices = P.select_chunks(dataset, data["validation_components"], args.n)
        chunk_ids = [dataset.chunks[i].chunk_id for i in indices]
        waves = P.waves_for(dataset, indices)
    finally:
        data["loader"].close()
    exported = {r["chunk_id"]: r["tts_probability"] for r in pq.read_table(run_dir / "predictions_chunks_selection.parquet").to_pylist()}
    expected = np.array([exported[c] for c in chunk_ids])
    scores = {}
    for label, device, precision in (("fp16", "cuda", "fp16"), ("fp32_cuda", "cuda", "fp32"), ("fp32_cpu", "cpu", "fp32")):
        adapter = ASTBundleAdapter.load(bundle_dir, device=device, precision=precision, batch_size=8)
        scores[label] = adapter.score_chunks(waves)
        if label == "fp16":
            metadata = dict(adapter.metadata)
        del adapter
        torch.cuda.empty_cache()
    report = P.parity_check(chunk_ids, expected, scores_fp16=scores["fp16"], others={k: v for k, v in scores.items() if k != "fp16"})
    manifest = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    report.update({"task": "R3-U2", "kind": "adapter_parity", "created_utc": stamp(), "owner": args.owner, "experiment": winner,
                   "selected_epoch": selection["selected_epoch"], "checkpoint_sha256": selection["checkpoint_sha256"][winner],
                   "bundle_components": manifest["components"], "adapter_metadata": metadata,
                   "components": sorted({data["validation_components"][dataset.chunks[i].recording_id] for i in indices}),
                   "short_chunks": int(sum(dataset.chunks[i].valid_samples < P.FULL_CHUNK for i in indices)),
                   "environment": {**environment(), "gpu": torch.cuda.get_device_name(0)},
                   "sources": {n: file_digest(root / n) for n in SOURCES if (root / n).exists()}})
    path = root / "reports/r3" / f"r3_u2_ast_parity_{stamp()}.json"
    write_json(path, report)
    print("parity", "PASSED" if report["passed"] else "FAILED", path)
    print("fp16 CUDA:", report["fp16"])
    for name, value in report["reported_not_gated"].items():
        print(name, value)
    if not report["passed"]:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["package", "parity"])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--run-dir", required=True, help="the R3-U2 run folder, relative to --root (runs/r3/<RUN_ID>)")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--bundle", default=BUNDLE)
    parser.add_argument("--shard-dir", default=None, help="canonical cohort shards (parity)")
    parser.add_argument("--model-cache", default=None, help="Hugging Face cache folder for the pinned AST (package)")
    parser.add_argument("--n", type=int, default=50, help="parity sample size")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if (root / "src").resolve() != (ROOT / "src").resolve():
        raise SystemExit("run the script inside the project selected by --root")
    if args.command == "parity" and not args.shard_dir:
        raise SystemExit("--shard-dir is required for parity")
    {"package": cmd_package, "parity": cmd_parity}[args.command](root, args)


if __name__ == "__main__":
    main()
