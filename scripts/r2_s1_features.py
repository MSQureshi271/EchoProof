"""R2-S1: Protocol B mel statistics, crop bank and LR feature tables.

    python scripts/r2_s1_features.py --root "/content/drive/MyDrive/PAI&AIES Project" --shard-dir /content/shards --workers 2

Refuses to overwrite finished outputs. Use --resume to continue an interrupted run (finished shards are skipped).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.features import r2_s1  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT), help="project root holding data/ and reports/ (Drive on Colab)")
    p.add_argument("--shard-dir", required=True, help="directory holding shard_*.tar (copy them from Drive to local disk first)")
    p.add_argument("--out-dir", default=None, help="default: <root>/data/features/r2_s1_B")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    manifest = r2_s1.build(args.root, args.shard_dir, args.out_dir, k=args.k, seed=args.seed, workers=args.workers, resume=args.resume,
                           progress=lambda msg: print(msg, flush=True))
    print(json.dumps({k: manifest[k] for k in ("plan_sha256", "counts", "mel_stats", "outputs", "seconds_total")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
