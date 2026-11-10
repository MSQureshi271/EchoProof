"""R3-S1 follow-up: measure how strongly candidate MP3 settings change validation audio (signal level only; nothing is scored by a model).

Local, against the original WAVs in `dataset/` (genuine and XTTS only; every file is checked against the cohort manifest hash first):
    python scripts/r3_s1_mp3_probe.py --source local --dataset-dir dataset

Colab, against the cohort shards (all four components; notebooks/R3_S1_Colab_mp3_probe.ipynb):
    python scripts/r3_s1_mp3_probe.py --source shards --root "/content/drive/MyDrive/PAI&AIES Project" --shard-dir /content/shards

Scope: B selection-role recordings only, the same seeded sample as the R3-S1 validation run. No test, calibration, operating or Chatterbox audio is
read, and no audio is written.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=("local", "shards"), required=True)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--shard-dir", default=None)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--out-dir", default=None, help="default: <root>/reports/r3")
    p.add_argument("--per-component", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    from echoproof.data import mp3_probe_run
    report = mp3_probe_run.run(args.root, args.out_dir, source=args.source, shard_dir=args.shard_dir, dataset_dir=args.dataset_dir,
                               per_component=args.per_component, seed=args.seed, progress=lambda msg: print(msg, flush=True))
    brief = {n: {"kbps": s["kbps"], "encode_rate": s["encode_rate"], "snr_gaincomp_db_median": (s["snr_gaincomp_db"] or {}).get("median"),
                 "retained_7k_pct_median": (s["retained_7k_pct"] or {}).get("median"), "relative_to_noise": s["relative_to_noise"]}
             for n, s in report["summary"].items()}
    print(json.dumps({"sample": report["sample"], "summary": brief, "seconds_total": report["seconds_total"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
