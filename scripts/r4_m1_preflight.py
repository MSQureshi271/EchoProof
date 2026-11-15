"""Read-only Protocol A preflight for the R4-M1 training handoff."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from echoproof.training.r4_m1_data import build_data, canonical_json_sha256, file_sha256

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard-dir', type=Path, required=True)
    args = parser.parse_args()
    data = build_data(args.root, args.shard_dir)
    try:
        print(json.dumps({'task': 'R4-M1 preflight', 'status': 'passed', 'protocol': 'A',
                          'recipe_sha256': file_sha256(args.root/'configs/r4_m1_recipes.json'),
                          'b_lock_file_sha256': file_sha256(args.root/data['recipe']['b_lock']['path']),
                          'b_lock_canonical_json_sha256': canonical_json_sha256(args.root/data['recipe']['b_lock']['path']),
                          'split_sha256': data['split_sha256'], 'counts': data['counts'],
                          'input_sha256': data['input_sha256'],
                          'test_waveforms_or_features_read': False,
                          'scope_note': 'Cohort and chunk manifest metadata may include test rows; sealed assignments are filtered. Shard existence checked; decoded audio checked lazily during training.'}, indent=2))
    finally:
        data['loader'].close()
