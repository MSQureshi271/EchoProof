"""R4-U1b: AST-A on Protocol A. Run on a Colab GPU.

  warmup    head-only warm-up on A training crops (the B R2-U2 recipe); writes warmup_result.json in its run folder
  finetune  partial fine-tune from that warm-up (the B-selected baseline): proofs, epoch-0 reproduction, training, adaptation evidence, frozen selection,
            raw validation exports and the paired warm-up versus adapted comparison
  package   convert the selected best.pt to a verified safetensors bundle (a new folder only)
  parity    score 50 selection-role chunks (all five components) from raw shard audio with the bundle adapter and compare with the exports

Reads Protocol A training audio and the three validation roles only: no test audio. Run folders are never reused; `--resume` continues an interrupted run
only if its frozen contract is unchanged. `warmup` and `finetune` refuse a recipe whose status does not record the team's agreement.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.artifacts import file_digest, write_json  # noqa: E402

WARMUP_RECIPE = "configs/r4_u1_ast_warmup_a.json"
FT_RECIPE = "configs/r4_u1_ast_ft_a.json"
BUNDLE = "bundles/r4_u1_ast_partial_A"
SOURCES = ("scripts/r4_u1_ast.py", "src/echoproof/training/r4_u1_ast_data.py", "src/echoproof/training/r4_u1_ast_train.py",
           "src/echoproof/training/r3_u2_train.py", "src/echoproof/training/r3_m1_data.py", "src/echoproof/training/r3_m1_selection.py",
           "src/echoproof/training/r3_m1_export.py", "src/echoproof/training/trainer.py", "src/echoproof/training/datasets.py",
           "src/echoproof/training/seeding.py", "src/echoproof/models/ast_detector.py", "src/echoproof/features/pretrained.py",
           "src/echoproof/features/r4_s1.py", "src/echoproof/features/r2_s1.py", "src/echoproof/evaluation/aggregate.py",
           "src/echoproof/evaluation/metrics.py")
PACKAGE_SOURCES = ("scripts/r4_u1_ast.py", "src/echoproof/models/ast_adapter.py", "src/echoproof/models/r3_u2_bundle.py",
                   "src/echoproof/models/ast_detector.py", "src/echoproof/training/r3_u2_parity.py", "src/echoproof/training/r4_u1_ast_data.py",
                   "src/echoproof/features/pretrained.py")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def environment():
    import numpy
    import torch
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "platform": platform.platform()}


def contract_changes(stored, now, prefix=""):
    """Field-level differences between two contracts as (path, stored value, current value)."""
    out = []
    for key in sorted(set(stored) | set(now)):
        a, b = stored.get(key, "<absent>"), now.get(key, "<absent>")
        path = prefix + key
        if isinstance(a, dict) and isinstance(b, dict):
            out.extend(contract_changes(a, b, path + "."))
        elif a != b:
            out.append((path, a, b))
    return out


# Only these may differ on a resumed run, and each difference is recorded with both values: the kernel/platform string of a new Colab VM, and the
# orchestration script itself (its resume check was made field-level on 15 November 2026 after a refused resume; it does not touch the
# training numerics). The training and data modules, recipe, data hashes, library versions and the GPU may not differ.
TOLERATED_ON_RESUME = ("environment.platform", "sources.scripts/r4_u1_ast.py")


def freeze_contract(run_dir: Path, contract: dict, resume: bool) -> Path:
    """The contract sits beside the run folder (the Trainer needs the folder itself empty on a fresh start)."""
    path = run_dir.parent / (run_dir.name + "_contract.json")
    if resume:
        if not path.is_file():
            raise SystemExit(f"resume refused: no frozen contract at {path}")
        changes = contract_changes(json.loads(path.read_text(encoding="utf-8")), contract)
        blocking = [c for c in changes if c[0] not in TOLERATED_ON_RESUME]
        if blocking:
            raise SystemExit("resume refused: the experiment contract or provenance changed: "
                             + "; ".join(f"{p}: stored {str(a)[:80]!r}, now {str(b)[:80]!r}" for p, a, b in blocking))
        tolerated = [c for c in changes if c[0] in TOLERATED_ON_RESUME]
        if tolerated:
            log = run_dir.parent / (run_dir.name + "_resume_log.jsonl")
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"resumed_utc": stamp(), "tolerated_changes": [list(c) for c in tolerated]}, sort_keys=True) + "\n")
            print("resuming with tolerated differences (recorded in", log.name + "):", tolerated, flush=True)
    else:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(contract, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
    return path


def common(args):
    import numpy as np
    import torch
    root = args.root.resolve()
    if (root / "src").resolve() != (ROOT / "src").resolve():
        raise SystemExit("run the script inside the project selected by --root")
    if not torch.cuda.is_available():
        raise SystemExit("AST-A requires a CUDA runtime (fp16 training)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return root, np, torch


def cmd_warmup(args):
    from echoproof.features.pretrained import pinned_snapshot
    from echoproof.training import r4_u1_ast_train as T
    from echoproof.training.r4_u1_ast_data import build_data
    root, np, torch = common(args)
    warm_recipe = T.load_warmup_recipe(root / WARMUP_RECIPE, root=root, require_agreed=True)
    ft_recipe = T.load_finetune_recipe(root / FT_RECIPE, root=root)
    pins = ft_recipe["init"]["pretrained_group_sha256"]
    snapshot = pinned_snapshot("ast", cache_dir=args.model_cache)
    data = build_data(root, args.shard_dir, seed=warm_recipe["train"]["seed"])
    try:
        contract = {"task": T.WARMUP_TASK, "protocol": "A", "recipe_sha256": T.recipe_hash(warm_recipe), "recipe_status": warm_recipe["status"],
                    "split_sha256": data["split_sha256"], "inputs": data["input_sha256"], "counts": data["counts"],
                    "sources": {p: file_digest(root / p) for p in SOURCES}, "environment": environment()}
        run_dir = (root / args.run_dir).resolve()
        contract_path = freeze_contract(run_dir, contract, args.resume)
        result = T.run_warmup(warm_recipe, data, snapshot=snapshot, run_dir=run_dir, ledger=root / args.ledger, owner=args.owner, pins=pins,
                              resume=args.resume, provenance={"contract_sha256": file_digest(contract_path)},
                              progress=lambda *m: print(*m, flush=True))
        print("warm-up best epoch", result["best_epoch"], "selection macro F1", result["best_macro_f1"], "| result:", run_dir / "warmup_result.json")
        return result
    finally:
        data["loader"].close()


def cmd_finetune(args):
    from echoproof.features.pretrained import pinned_snapshot
    from echoproof.training import r4_u1_ast_train as T
    from echoproof.training.r4_u1_ast_data import build_data
    root, np, torch = common(args)
    recipe = T.load_finetune_recipe(root / FT_RECIPE, root=root, require_agreed=True)
    warm_recipe = T.load_warmup_recipe(root / WARMUP_RECIPE, root=root)
    warm = T.load_warmup_result((root / args.warmup_dir).resolve(), warm_recipe)
    snapshot = pinned_snapshot("ast", cache_dir=args.model_cache)
    data = build_data(root, args.shard_dir, seed=recipe["train"]["seed"])
    try:
        if warm["split_sha256"] != data["split_sha256"]:
            raise SystemExit("the warm-up ran under a different split hash")
        contract = {"task": T.TASK, "protocol": "A", "recipe_sha256": T.recipe_hash(recipe), "recipe_status": recipe["status"],
                    "split_sha256": data["split_sha256"], "inputs": data["input_sha256"], "counts": data["counts"],
                    "warmup": {"result_sha256": file_digest(root / args.warmup_dir / "warmup_result.json"),
                               "checkpoint_sha256": warm["best_checkpoint_sha256"], "best_epoch": warm["best_epoch"],
                               "best_macro_f1": warm["best_macro_f1"], "recipe_sha256": warm["recipe_sha256"]},
                    "sources": {p: file_digest(root / p) for p in SOURCES}, "environment": environment()}
        run_dir = (root / args.run_dir).resolve()
        contract_path = freeze_contract(run_dir, contract, args.resume)
        summary = T.run_finetune(recipe, warm, data, snapshot=snapshot, run_dir=run_dir, ledger=root / args.ledger, owner=args.owner,
                                 resume=args.resume, provenance={"contract_sha256": file_digest(contract_path)},
                                 progress=lambda *m: print(*m, flush=True))
        print("fine-tune best epoch", summary["best_epoch"], "macro F1", summary["best_macro_f1"], flush=True)
        evidence = T.finish(run_dir, data, summary, recipe, warm, snapshot=snapshot, device="cuda", contract_path=contract_path)
        print("evidence:", run_dir / "evidence.json")
        return evidence
    finally:
        data["loader"].close()


def cmd_package(args):
    import torch
    from echoproof.features.pretrained import pinned_snapshot
    from echoproof.models.r3_u2_bundle import package_ast_detector
    from echoproof.training import r4_u1_ast_train as T
    root = args.root.resolve()
    run_dir = (root / args.run_dir).resolve()
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    recipe = T.load_finetune_recipe(root / FT_RECIPE, root=root)
    if selection["recipe_sha256"] != T.recipe_hash(recipe):
        raise SystemExit("the recipe changed since the selection was frozen")
    checkpoint = run_dir / "best.pt"
    if file_digest(checkpoint) != selection["checkpoint_sha256"]["baseline"]:
        raise SystemExit("the selected checkpoint changed since the selection was frozen")
    meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
    snapshot = pinned_snapshot("ast", cache_dir=args.model_cache)
    package_ast_detector(checkpoint, snapshot, root / args.bundle, bundle_id="bundle:r4-u1-ast-partial-A-baseline", protocol="A",
                         expected_checkpoint_sha256=selection["checkpoint_sha256"]["baseline"], split_sha256=selection["split_sha256"],
                         expected_epoch=selection["selected_epoch"], expected_config_hash=meta["config_hash"], recipe_sha256=selection["recipe_sha256"],
                         warmup_checkpoint_sha256=selection["init_checkpoint_sha256"],
                         pretrained_group_sha256=recipe["init"]["pretrained_group_sha256"], unfrozen_blocks=recipe["model"]["unfrozen_blocks"],
                         experiment="baseline")
    print("bundle written:", root / args.bundle, "| epoch:", selection["selected_epoch"])


def cmd_parity(args):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from echoproof.models.ast_adapter import ASTBundleAdapter
    from echoproof.training import r3_u2_parity as P
    from echoproof.training.r4_u1_ast_data import build_data, select_chunks

    if not torch.cuda.is_available():
        raise SystemExit("the parity gate needs a CUDA runtime")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.root.resolve()
    run_dir = (root / args.run_dir).resolve()
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    bundle_dir = root / args.bundle
    data = build_data(root, args.shard_dir, seed=42)
    try:
        dataset = data["validation"]["selection"]
        indices = select_chunks(dataset, data["validation_components"], args.n)
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
    report.update({"task": "R4-U1b", "kind": "adapter_parity", "created_utc": stamp(), "owner": args.owner, "selected_epoch": selection["selected_epoch"],
                   "checkpoint_sha256": selection["checkpoint_sha256"]["baseline"], "bundle_components": manifest["components"],
                   "adapter_metadata": metadata,
                   "components": sorted({data["validation_components"][dataset.chunks[i].recording_id] for i in indices}),
                   "short_chunks": int(sum(dataset.chunks[i].valid_samples < P.FULL_CHUNK for i in indices)), "environment": environment(),
                   "sources": {n: file_digest(root / n) for n in PACKAGE_SOURCES if (root / n).exists()}})
    path = root / "reports/r4" / f"r4_u1_ast_parity_{stamp()}.json"
    write_json(path, report)
    print("parity", "PASSED" if report["passed"] else "FAILED", path)
    print("fp16 CUDA:", report["fp16"])
    for name, value in report["reported_not_gated"].items():
        print(name, value)
    if not report["passed"]:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["warmup", "finetune", "package", "parity"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--run-dir", required=True, help="run folder relative to --root (a new folder; for package/parity the fine-tune run)")
    parser.add_argument("--warmup-dir", default=None, help="the A warm-up run folder relative to --root (finetune)")
    parser.add_argument("--shard-dir", type=Path, default=None, help="canonical cohort shards on local disk")
    parser.add_argument("--ledger", default="reports/ledger.csv")
    parser.add_argument("--bundle", default=BUNDLE)
    parser.add_argument("--model-cache", default=None, help="Hugging Face cache folder for the pinned AST")
    parser.add_argument("--n", type=int, default=50, help="parity sample size")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command in ("warmup", "finetune", "parity") and not args.shard_dir:
        raise SystemExit("--shard-dir is required")
    if args.command == "finetune" and not args.warmup_dir:
        raise SystemExit("--warmup-dir is required for finetune")
    {"warmup": cmd_warmup, "finetune": cmd_finetune, "package": cmd_package, "parity": cmd_parity}[args.command](args)


if __name__ == "__main__":
    main()
