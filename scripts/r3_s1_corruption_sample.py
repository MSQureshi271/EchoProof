"""R3-S1: corruption suite builder.

`spec` prints the frozen specification, its hash and the local ffmpeg build (no data needed):
    python scripts/r3_s1_corruption_sample.py spec

`sample` applies the four conditions to a seeded sample of B selection-role recordings and writes a report plus a per-variant table
(needs the audio shards, so it runs on Colab: notebooks/R3_S1_Colab_corruption_sample.ipynb). It never touches test, calibration,
operating or Chatterbox audio and writes no audio:
    python scripts/r3_s1_corruption_sample.py sample --root "/content/drive/MyDrive/PAI&AIES Project" --shard-dir /content/shards
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
    sub.add_parser("spec", help="print the frozen specification and the ffmpeg build")
    s = sub.add_parser("sample", help="run the suite on a validation sample (Colab)")
    s.add_argument("--root", default=str(ROOT))
    s.add_argument("--shard-dir", required=True)
    s.add_argument("--out-dir", default=None, help="default: <root>/reports/r3")
    s.add_argument("--per-component", type=int, default=8)
    s.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.command == "spec":
        from echoproof.data import corruption
        print(json.dumps(corruption.spec_record(), indent=2, sort_keys=True))
    else:
        from echoproof.data import corruption_sample
        report = corruption_sample.run(args.root, args.shard_dir, args.out_dir, per_component=args.per_component, seed=args.seed,
                                       progress=lambda msg: print(msg, flush=True))
        print(json.dumps({k: report[k] for k in ("passed", "checks", "failures", "sample", "conditions", "seconds_total")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
