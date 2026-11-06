"""R2-S2: training-partition EDA (Protocol B train only).

Step 1 needs the audio shards (run on Colab, see notebooks/R2_S2_Colab_eda.ipynb):
    python scripts/r2_s2_eda.py measure --root "/content/drive/MyDrive/PAI&AIES Project" --shard-dir /content/shards --workers 2

Step 2 needs no audio (run locally once the step-1 files are synced):
    python scripts/r2_s2_eda.py figures --root .

Measurement refuses to overwrite finished outputs; use --resume to continue an interrupted run.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="per-recording measurements from the shards (Colab)")
    m.add_argument("--root", default=str(ROOT))
    m.add_argument("--shard-dir", required=True)
    m.add_argument("--out-dir", default=None, help="default: <root>/data/eda/r2_s2_B_train")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--workers", type=int, default=1)
    m.add_argument("--resume", action="store_true")
    f = sub.add_parser("figures", help="build every figure family from the measurement files (local)")
    f.add_argument("--root", default=str(ROOT))
    f.add_argument("--data-dir", default=None, help="default: <root>/data/eda/r2_s2_B_train")
    f.add_argument("--out-dir", default=None, help="default: <root>/reports/eda")
    args = p.parse_args(argv)
    if args.command == "measure":
        from echoproof.eda import measure
        manifest = measure.build(args.root, args.shard_dir, args.out_dir, seed=args.seed, workers=args.workers, resume=args.resume,
                                 progress=lambda msg: print(msg, flush=True))
        print(json.dumps({k: manifest[k] for k in ("plan_sha256", "counts", "outputs", "seconds_total")}, indent=2, sort_keys=True))
    else:
        from echoproof.eda import figures
        report = figures.make_all(args.root, args.out_dir, args.data_dir)
        print(json.dumps({"figure_families": report["figure_families"], "figures": [f["figure"] for f in report["figures"]]}, indent=2))
        for item in report["figures"]:
            print(item["figure"])
            for line in item["interpretation"]:
                print("   ", line)


if __name__ == "__main__":
    main()
