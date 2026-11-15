"""R4-U1a: WavLM head for Protocol A on the A embeddings (R2-U1 B cache plus R4-U0 gap cache).

Subcommands (each requires --owner; nothing is overwritten, every output directory must be new or empty):
  check-data  read the A union, prove it matches the A split / crop bank / chunk manifest / cohort, write alignment evidence
  train       check-data, train the B-selected mean_std head for the full budget, freeze the selection, export validation predictions
              (refuses a recipe whose status does not record the team's agreement)
  parity      compare direct waveform scoring against the cached embeddings for the selected head (needs the pinned WavLM and the cohort shards)
  package     build the verified inference bundle from the selected checkpoint and the pinned encoder

Reads the training and validation partitions only. The sealed A test partition is never opened. Training needs no GPU; parity and packaging run on Colab.
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

RECIPE = "configs/r4_u1_wavlm_head_a.json"
OUTPUT = "reports/r4/r4_u1_wavlm_A"
SOURCES = ("scripts/r4_u1_wavlm_head.py", "src/echoproof/training/r4_u1_wavlm_data.py", "src/echoproof/training/r4_u1_wavlm_train.py",
           "src/echoproof/features/r4_u0.py", "src/echoproof/training/r3_u1_data.py", "src/echoproof/training/r3_u1_train.py",
           "src/echoproof/training/r3_u1_parity.py", "src/echoproof/models/wavlm_head.py", "src/echoproof/models/wavlm_adapter.py",
           "src/echoproof/models/r3_u1_bundle.py", "src/echoproof/evaluation/aggregate.py", "src/echoproof/evaluation/metrics.py")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def split_sha(root):
    return json.loads((root / "reports/r1/split_report.json").read_text(encoding="utf-8"))["protocols"]["A"]["split_sha256"]


def environment():
    import numpy
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__, "platform": platform.platform()}


def sources(root):
    return {n: file_digest(root / n) for n in SOURCES if (root / n).exists()}


def cmd_check(root, args, *, require_agreed=False):
    from echoproof.training import r4_u1_wavlm_data as D
    from echoproof.training.r4_u1_wavlm_train import load_recipe
    recipe = load_recipe(root / args.recipe, root=root, require_agreed=require_agreed)
    print("reading the A embeddings (mean_std) ...", flush=True)
    data = D.load_embeddings_a(root)
    report = D.check_alignment_a(root, data)
    evidence = {"task": "R4-U1a", "kind": "alignment", "created_utc": stamp(), "owner": args.owner, "split_sha256": split_sha(root),
                "recipe_file_sha256": file_digest(root / args.recipe), "recipe_status": recipe["status"], "report": report,
                "data_identity": data.identity, "model_scores_used": False, "sources": sources(root)}
    path = root / "reports/r4" / f"r4_u1_wavlm_alignment_{stamp()}.json"
    write_json(path, evidence)
    print("alignment passed:", path)
    return data, recipe


def cmd_train(root, args):
    from echoproof.training.r4_u1_wavlm_train import finish, train
    output = root / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to reuse a non-empty output directory: {output}")
    data, recipe = cmd_check(root, args, require_agreed=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "environment.json", {"created_utc": stamp(), "owner": args.owner, "device": args.device, **environment(),
                                             "recipe_file_sha256": file_digest(root / args.recipe), "sources": sources(root)})
    print("training mean_std ...", flush=True)
    summary = train(data, recipe, output / "run_mean_std", owner=args.owner, split_sha256=split_sha(root), device=args.device,
                    run_id=f"r4_u1_wavlm_head_A_{stamp()}", ledger_path=root / args.ledger, progress=lambda *m: print(*m, flush=True))
    print("best epoch", summary["best_epoch"], "macro F1", summary["best_macro_f1"], flush=True)
    evidence = finish(output, data, summary, recipe, split_sha256=split_sha(root), device=args.device)
    print("selected:", evidence["selection"]["selected"], "evidence:", output / "evidence.json")


def cmd_parity(root, args):
    import torch
    from echoproof.features.pretrained import WavLMExtractor
    from echoproof.models.wavlm_head import WavLMHead
    from echoproof.training import r3_u1_parity as P, r4_u1_wavlm_data as D
    output = root / args.output
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    ckpt_path = output / "run_mean_std" / "best.pt"
    if file_digest(ckpt_path) != selection["checkpoint_sha256"]["mean_std"]:
        raise SystemExit("selected checkpoint changed since the selection was frozen")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    a = ckpt["architecture"]
    head = WavLMHead(a["in_dim"], hidden=a["hidden"], dropout=a["dropout"]).eval()
    head.load_state_dict(ckpt["model"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False                      # the cache was built with TF32 off
    torch.backends.cudnn.allow_tf32 = False
    data = D.load_embeddings_a(root)
    chosen = D.parity_rows(root, data, args.n)
    waves = P.waves_from_shards(chosen, root / args.shards)            # hash- and length-verified against the cohort manifest
    cached = data.roles["selection"].X[[r["position"] for r in chosen]]
    extractor = WavLMExtractor.from_pinned(device=args.device, batch_size=8, cache_dir=args.model_cache)
    report = P.parity_check(chosen, waves, cached, head, embed_fn=lambda w: extractor.embed(w, pooling="mean_std"), pooling="mean_std")
    report.update({"task": "R4-U1a", "kind": "parity", "created_utc": stamp(), "checkpoint_sha256": selection["checkpoint_sha256"]["mean_std"],
                   "encoder": extractor.describe(), "environment": environment(), "components_sampled": sorted({r["component"] for r in chosen})})
    path = root / "reports/r4" / f"r4_u1_wavlm_parity_{stamp()}.json"
    write_json(path, report)
    print("parity", "PASSED" if report["passed"] else "FAILED", path)
    if not report["passed"]:
        raise SystemExit(1)


def cmd_package(root, args):
    import torch
    from echoproof.features.pretrained import pinned_snapshot
    from echoproof.models.r3_u1_bundle import package_wavlm_head
    output = root / args.output
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    snapshot = pinned_snapshot("wavlm", cache_dir=args.model_cache)
    ckpt_path = output / "run_mean_std" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    destination = root / args.bundle
    spec = package_wavlm_head(ckpt_path, snapshot, destination, bundle_id="bundle:r4-u1-wavlm-head-A-mean_std", protocol="A",
                              expected_checkpoint_sha256=selection["checkpoint_sha256"]["mean_std"], split_sha256=split_sha(root),
                              expected_epoch=selection["selected_epoch"], expected_config_hash=ckpt["config_hash"])
    print("bundle written:", destination, spec if not hasattr(spec, "bundle_id") else spec.bundle_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check-data", "train", "parity", "package"])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--owner", required=True)
    parser.add_argument("--recipe", default=RECIPE)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--ledger", default="reports/ledger.csv")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--shards", default="data/processed/cohort_v1", help="canonical cohort shard folder (parity)")
    parser.add_argument("--model-cache", default=None, help="Hugging Face cache folder for the pinned WavLM (parity, package)")
    parser.add_argument("--n", type=int, default=50, help="parity sample size")
    parser.add_argument("--bundle", default="bundles/r4_u1_wavlm_head_A")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if (root / "src").resolve() != (ROOT / "src").resolve():
        raise SystemExit("run the script inside the project selected by --root")
    {"check-data": cmd_check, "train": cmd_train, "parity": cmd_parity, "package": cmd_package}[args.command](root, args)


if __name__ == "__main__":
    main()
