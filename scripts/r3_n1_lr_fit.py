"""Fit the selected LR's recording-level temperature using calibration data only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from echoproof.artifacts import file_digest
from echoproof.evaluation.calibrate import apply_temperature, fit_temperature
from echoproof.evaluation.metrics import compute_metrics

CONTRACT = ROOT/'configs/r3_n1_lr_contract.json'
CALIBRATION_FILE = 'predictions_recordings_calibration.parquet'


def validate_calibration(rows, counts):
    if len(rows) != counts['recordings']:
        raise ValueError('calibration recording coverage mismatch')
    ids, groups = set(), set()
    for row in rows:
        rid, group = row['recording_id'], row['group_id']
        if not isinstance(rid, str) or not rid or rid in ids or not isinstance(group, str) or not group:
            raise ValueError('invalid or duplicate calibration identity')
        if row['role'] != 'calibration':
            raise ValueError('non-calibration role supplied to temperature fit')
        if row['component'] not in {'genuine', 'melo', 'vits', 'xtts'} or row['label'] not in (0, 1) or row['label'] != (row['component'] != 'genuine'):
            raise ValueError('invalid calibration component or label')
        if type(row['chunk_count']) is not int or row['chunk_count'] < 1:
            raise ValueError('invalid calibration chunk count')
        ids.add(rid)
        groups.add(group)
    if len(groups) != counts['groups'] or sum(r['chunk_count'] for r in rows) != counts['chunks']:
        raise ValueError('calibration group or chunk coverage mismatch')


def fit(export_dir, output):
    import pyarrow.parquet as pq

    c = json.loads(CONTRACT.read_text(encoding='utf-8'))
    export_dir = Path(export_dir).resolve()
    root = (ROOT/c['output_root']).resolve()
    output = Path(output).resolve()
    if not export_dir.is_relative_to(root) or export_dir == root or not output.is_relative_to(root) or output == root:
        raise ValueError('export and fit output must be separate directories under the LR run root')
    if export_dir == output or output.exists():
        raise FileExistsError('fit output already exists or equals export; refusing to overwrite')
    evidence_path = export_dir/'evidence.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    if (evidence.get('task') != 'R3-N1-LR prediction export' or evidence.get('status') != 'complete'
            or evidence.get('contract_sha256') != file_digest(CONTRACT)
            or evidence.get('selected_bundle_sha256') != c['bundle_sha256']
            or evidence.get('split_sha256') != c['split_sha256']
            or evidence.get('feature_manifest_sha256') != c['feature_manifest_sha256']):
        raise ValueError('export evidence is incompatible with LR contract')
    calibration_path = export_dir/CALIBRATION_FILE
    calibration_sha = file_digest(calibration_path)
    if evidence.get('outputs', {}).get(CALIBRATION_FILE) != calibration_sha:
        raise ValueError('calibration predictions checksum mismatch')
    rows = pq.read_table(calibration_path).to_pylist()
    validate_calibration(rows, c['calibration'])
    labels, scores = [r['label'] for r in rows], [r['raw_score'] for r in rows]
    temperature = fit_temperature(labels, scores, epsilon=c['epsilon'], bounds=tuple(c['temperature_bounds']))
    calibrated = apply_temperature(scores, temperature, epsilon=c['epsilon'])
    raw_metrics = compute_metrics(labels, scores, .5)
    calibrated_metrics = compute_metrics(labels, calibrated, .5)
    result = {'task': 'R3-N1-LR temperature fit', 'status': 'calibration_fit_complete',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'method': c['calibration_method'],
              'temperature': temperature, 'epsilon': c['epsilon'], 'bounds': c['temperature_bounds'],
              'source_contract_sha256': file_digest(CONTRACT), 'source_export_evidence_sha256': file_digest(evidence_path),
              'source_calibration_predictions_sha256': calibration_sha, 'selected_bundle_sha256': c['bundle_sha256'],
              'fit_recordings': len(rows), 'fit_groups': len({r['group_id'] for r in rows}),
              'fit_chunk_count': sum(r['chunk_count'] for r in rows),
              'fit_role': 'calibration', 'operating_predictions_read': False,
              'test_data_read': False, 'model_retrained': False, 'threshold_selected': False,
              'fit_metrics_in_sample': {'raw': raw_metrics, 'temperature': calibrated_metrics},
              'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
                                (Path(__file__).resolve(), ROOT/'src/echoproof/evaluation/calibrate.py',
                                 ROOT/'src/echoproof/evaluation/metrics.py')}}
    output.mkdir(parents=True, exist_ok=False)
    (output/'temperature.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(fit(args.export_dir, args.output), indent=2))
