"""R2-S3: shortcut probe (metadata-only logistic regression, Protocol B).

Step 1 needs the audio shards (Colab, notebooks/R2_S3_Colab_selection_measure.ipynb):
    python scripts/r2_s3_shortcut.py measure --root "/content/drive/MyDrive/PAI&AIES Project" --shard-dir /content/shards --workers 2

Step 2 needs no audio (local, after R2-S2 train measurements and the step-1 files are in data/eda/):
    python scripts/r2_s3_shortcut.py probe --root .
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
    m = sub.add_parser("measure", help="measure the selection-role validation recordings (Colab)")
    m.add_argument("--root", default=str(ROOT))
    m.add_argument("--shard-dir", required=True)
    m.add_argument("--out-dir", default=None, help="default: <root>/data/eda/r2_s3_B_selection")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--workers", type=int, default=1)
    m.add_argument("--resume", action="store_true")
    q = sub.add_parser("probe", help="fit the probe, write the memo, figure and matched ids (local)")
    q.add_argument("--root", default=str(ROOT))
    q.add_argument("--resamples", type=int, default=1000)
    args = p.parse_args(argv)
    if args.command == "measure":
        from echoproof.eda import selection
        manifest = selection.build(args.root, args.shard_dir, args.out_dir, seed=args.seed, workers=args.workers, resume=args.resume,
                                   progress=lambda msg: print(msg, flush=True))
        print(json.dumps({k: manifest[k] for k in ("plan_sha256", "counts", "outputs", "seconds_total")}, indent=2, sort_keys=True))
    else:
        from echoproof.eda import shortcut
        report = shortcut.run(args.root, n_resamples=args.resamples)
        print(json.dumps(report["conclusions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
