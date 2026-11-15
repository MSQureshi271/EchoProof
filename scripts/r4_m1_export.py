"""Export frozen LR-A or BiGRU-A validation scores for R4-N1."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from echoproof.training.r4_m1_export import export

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--shard-dir', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', choices=('lr', 'bigru'), required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.shard_dir, args.run_dir, args.output, args.model), indent=2))
