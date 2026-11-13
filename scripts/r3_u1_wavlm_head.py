"""R3-U1: WavLM head on the cached R2-U1 embeddings (Protocol B).

Subcommands (each requires --owner; nothing is overwritten, every output directory must be new or empty):
  check-data  read the cache, prove it matches the split / chunk manifest / cohort, write alignment evidence
  train       check-data, train both poolings for the full budget, freeze the selection, export validation predictions
  parity      compare direct waveform scoring against the cached embeddings for the selected head (needs the pinned WavLM and the cohort shards)
  package     build the verified inference bundle from the selected checkpoint and the pinned encoder

Reads the training and validation partitions only. The sealed test partition is never opened. Training needs no GPU; parity and packaging run on Colab.
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

RECIPE = "configs/r3_u1_wavlm_head_b.json"
CACHE = "data/features/r2_u1_B"
MEMO = "data/features/r3_u1_memo"
OUTPUT = "reports/r3/r3_u1_B"
SOURCES = ("scripts/r3_u1_wavlm_head.py", "src/echoproof/training/r3_u1_data.py", "src/echoproof/training/r3_u1_train.py",
           "src/echoproof/training/r3_u1_parity.py", "src/echoproof/models/wavlm_head.py", "src/echoproof/models/wavlm_adapter.py",
           "src/echoproof/models/r3_u1_bundle.py", "src/echoproof/evaluation/aggregate.py", "src/echoproof/evaluation/metrics.py")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def split_sha(root):
    return json.loads((root / "reports/r1/split_report.json").read_text(encoding="utf-8"))["protocols"]["B"]["split_sha256"]


def environment():
    import numpy
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
            "platform": platform.platform()}


def load_all(root, args):
    from echoproof.training import r3_u1_data as D
    datas = {}
    for pooling in ("mean", "mean_std"):
        print(f"reading {pooling} embeddings ...", flush=True)
        datas[pooling] = D.load_embeddings(root / args.cache, pooling, memo_dir=root / args.memo)
    return datas


def cmd_check(root, args):
    from echoproof.training import r3_u1_data as D
    from echoproof.training.r3_u1_train import load_recipe
    recipe = load_recipe(root / args.recipe)
    datas = load_all(root, args)
    reports = {p: D.check_alignment(root, d) for p, d in datas.items()}
    if not all(r["passed"] for r in reports.values()):
        raise SystemExit("alignment failed")
    evidence = {"task": "R3-U1", "kind": "alignment", "created_utc": stamp(), "owner": args.owner, "split_sha256": split_sha(root),
                "recipe_file_sha256": file_digest(root / args.recipe), "reports": reports,
                "data_identity": {p: d.identity for p, d in datas.items()}, "model_scores_used": False,
                "sources": {n: file_digest(root / n) for n in SOURCES if (root / n).exists()}}
    path = root / "reports/r3" / f"r3_u1_alignment_{stamp()}.json"
    write_json(path, evidence)
    print("alignment passed:", path)
    return datas, recipe


def cmd_train(root, args):
    from echoproof.training.r3_u1_train import finish, train_variant
    output = root / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to reuse a non-empty output directory: {output}")
    datas, recipe = cmd_check(root, args)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "environment.json", {"created_utc": stamp(), "owner": args.owner, "device": args.device, **environment(),
                                             "recipe_file_sha256": file_digest(root / args.recipe),
                                             "sources": {n: file_digest(root / n) for n in SOURCES if (root / n).exists()}})
    summaries = {}
    for pooling, data in datas.items():
        print(f"training {pooling} ...", flush=True)
        summaries[pooling] = train_variant(data, recipe, output / f"run_{pooling}", owner=args.owner, split_sha256=split_sha(root),
                                           device=args.device, run_id=f"r3_u1_{pooling}_{stamp()}",
                                           ledger_path=root / args.ledger,
                                           progress=lambda *m: print(*m, flush=True))
        print(pooling, "best epoch", summaries[pooling]["best_epoch"], "macro F1", summaries[pooling]["best_macro_f1"], flush=True)
    evidence = finish(output, datas, summaries, recipe, split_sha256=split_sha(root), device=args.device)
    print("selected:", evidence["selection"]["selected"], "evidence:", output / "evidence.json")


def cmd_parity(root, args):
    import torch
    from echoproof.features.pretrained import WavLMExtractor
    from echoproof.models.wavlm_head import WavLMHead
    from echoproof.training import r3_u1_data as D, r3_u1_parity as P
    output = root / args.output
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    pooling = selection["selected"]
    ckpt_path = output / f"run_{pooling}" / "best.pt"
    if file_digest(ckpt_path) != selection["checkpoint_sha256"][pooling]:
        raise SystemExit("selected checkpoint changed since the selection was frozen")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    a = ckpt["architecture"]
    head = WavLMHead(a["in_dim"], hidden=a["hidden"], dropout=a["dropout"]).eval()
    head.load_state_dict(ckpt["model"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False                      # the cache was built with TF32 off
    torch.backends.cudnn.allow_tf32 = False
    rows = P.development_rows(root / args.cache)
    chosen = P.select_items(rows, args.n)                                # selection role, every component except Chatterbox
    waves = P.waves_from_shards(chosen, root / args.shards)              # hash- and length-verified against the cohort manifest
    data = D.load_embeddings(root / args.cache, pooling, memo_dir=root / args.memo)
    index = {i: k for k, i in enumerate(data.roles["selection"].item_ids)}
    cached = data.roles["selection"].X[[index[r["item_id"]] for r in chosen]]
    extractor = WavLMExtractor.from_pinned(device=args.device, batch_size=8, cache_dir=args.model_cache)
    report = P.parity_check(chosen, waves, cached, head, embed_fn=lambda w: extractor.embed(w, pooling=pooling), pooling=pooling)
    report.update({"task": "R3-U1", "kind": "parity", "created_utc": stamp(), "checkpoint_sha256": selection["checkpoint_sha256"][pooling],
                   "encoder": extractor.describe(), "environment": environment()})
    path = root / "reports/r3" / f"r3_u1_parity_{stamp()}.json"
    write_json(path, report)
    print("parity", "PASSED" if report["passed"] else "FAILED", path)
    if not report["passed"]:
        raise SystemExit(1)


def cmd_package(root, args):
    from echoproof.features.pretrained import pinned_snapshot
    from echoproof.models.r3_u1_bundle import package_wavlm_head
    output = root / args.output
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    pooling = selection["selected"]
    snapshot = pinned_snapshot("wavlm", cache_dir=args.model_cache)
    import torch
    ckpt = torch.load(output / f"run_{pooling}" / "best.pt", map_location="cpu", weights_only=True)
    destination = root / args.bundle
    spec = package_wavlm_head(output / f"run_{pooling}" / "best.pt", snapshot, destination, bundle_id=f"bundle:r3-u1-wavlm-head-B-{pooling}",
                              protocol="B", expected_checkpoint_sha256=selection["checkpoint_sha256"][pooling], split_sha256=split_sha(root),
                              expected_epoch=selection["selected_epoch"], expected_config_hash=ckpt["config_hash"])
    print("bundle written:", destination, spec if not hasattr(spec, "bundle_id") else spec.bundle_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check-data", "train", "parity", "package"])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--owner", required=True)
    parser.add_argument("--recipe", default=RECIPE)
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--memo", default=MEMO)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--ledger", default="reports/ledger.csv")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--shards", default="data/processed/cohort_v1", help="canonical cohort shard folder (parity)")
    parser.add_argument("--model-cache", default=None, help="Hugging Face cache folder for the pinned WavLM (parity, package)")
    parser.add_argument("--n", type=int, default=50, help="parity sample size")
    parser.add_argument("--bundle", default="bundles/r3_u1_wavlm_head_B")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if (root / "src").resolve() != (ROOT / "src").resolve():
        raise SystemExit("run the script inside the project selected by --root")
    {"check-data": cmd_check, "train": cmd_train, "parity": cmd_parity, "package": cmd_package}[args.command](root, args)


if __name__ == "__main__":
    main()
