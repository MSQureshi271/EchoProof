"""Train the selected B BiGRU configuration on Protocol A."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from echoproof.training.r4_m1_bigru import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--shard-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.shard_dir, args.output, resume=args.resume), indent=2))
