"""Run the Protocol B LR grid on the verified R2-S1 feature handoff."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from echoproof.evaluation.aggregate import aggregate_recordings
from echoproof.evaluation.metrics import compute_metrics
from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH
from echoproof.models.lr_detector import LRDetector, run_lr_grid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(feature_dir):
    root = Path(feature_dir)
    manifest = json.loads((root / 'r2_s1_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('protocol') != 'B' or not manifest.get('verification', {}).get('passed'):
        raise ValueError('R2-S1 manifest is not a verified Protocol B build')
    if manifest['inputs']['feature_schema_hash'] != FEATURE_SCHEMA_HASH:
        raise ValueError('feature schema mismatch')
    arrays = []
    for key in ('train_features', 'validation_features'):
        info = manifest['outputs'][key]
        path = root / info['file']
        if path.resolve().parent != root.resolve() or sha(path) != info['sha256']:
            raise ValueError(f'{key} path or checksum mismatch')
        with np.load(path, allow_pickle=False) as z:
            table = {k: z[k] for k in z.files}
        if str(table['feature_schema_hash']) != FEATURE_SCHEMA_HASH or str(table['plan_sha256']) != manifest['plan_sha256']:
            raise ValueError(f'{key} provenance mismatch')
        if table['feature_names'].tolist() != list(FEATURE_NAMES):
            raise ValueError('feature order mismatch')
        n = len(table['y'])
        fields = ('recording_id', 'group_id', 'component', 'crop_id' if key == 'train_features' else 'chunk_id')
        if table['X'].shape != (n, len(FEATURE_NAMES)) or any(len(table[f]) != n for f in fields):
            raise ValueError('feature rows are misaligned')
        if np.isinf(table['X']).any() or set(np.unique(table['y'])) != {0, 1}:
            raise ValueError('invalid features or labels')
        missing_columns = set(np.flatnonzero(np.isnan(table['X']).any(axis=0)).tolist())
        allowed_missing = {FEATURE_NAMES.index('voiced_f0_mean'), FEATURE_NAMES.index('voiced_f0_std')}
        if not missing_columns <= allowed_missing:
            raise ValueError('NaN outside the permitted voicing columns')
        if len(set(table[fields[-1]].tolist())) != n:
            raise ValueError('duplicate crop or chunk ID')
        metadata = {}
        for rid, group, label, component in zip(table['recording_id'], table['group_id'], table['y'], table['component']):
            identity = (group, int(label), component)
            if rid in metadata and metadata[rid] != identity:
                raise ValueError('recording metadata differs between rows')
            metadata[rid] = identity
        allowed = {'genuine', 'melo', 'vits'} | ({'xtts'} if key == 'validation_features' else set())
        if not set(table['component']) <= allowed or any(int(y) != (c != 'genuine') for y, c in zip(table['y'], table['component'])):
            raise ValueError('invalid Protocol B component or label')
        arrays.append(table)
    train, val = arrays
    if len(val['role']) != len(val['y']) or set(val['role']) != {'selection', 'calibration', 'operating'}:
        raise ValueError('invalid validation roles')
    if set(train['group_id']) & set(val['group_id']):
        raise ValueError('speaker crosses train and validation')
    roles = {}
    for group, role in zip(val['group_id'], val['role']):
        if group in roles and roles[group] != role:
            raise ValueError('speaker crosses validation roles')
        roles[group] = role
    mask = val['role'] == 'selection'
    selection = {k: v[mask] for k, v in val.items() if getattr(v, 'ndim', 0) and len(v) == len(mask) and k != 'feature_names'}
    # The authoritative aggregator validates IDs, labels and recording metadata.
    aggregate_recordings(selection['chunk_id'], selection['recording_id'], selection['y'], np.full(mask.sum(), 0.5))
    return manifest, train, selection


def run(feature_dir, output, ledger):
    manifest, train, val = load_inputs(feature_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    split_hash = manifest['inputs']['split_sha256']
    train_input = {'X': train['X'], 'y': train['y'], 'recording_ids': train['recording_id']}
    selection_input = {'X': val['X'], 'y': val['y'], 'recording_ids': val['recording_id']}
    try:
        result = run_lr_grid(train_input, selection_input, protocol='B', split_hash=split_hash,
                             run_prefix=output.name, ledger_path=ledger, owner='Muneeb', metrics_fn=compute_metrics)
    except Exception as exc:
        failure = {'task': 'R2-M1', 'status': 'failed', 'split_sha256': split_hash,
                   'error': f'{type(exc).__name__}: {exc}', 'ledger': str(ledger)}
        (output / 'evidence.json').write_text(json.dumps(failure, indent=2), encoding='utf-8')
        raise
    trials = result['trials']
    report = {'task': 'R2-M1', 'protocol': 'B', 'seed': 42, 'threshold': 0.5,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__},
              'source_sha256': {p.name: sha(p) for p in [Path(__file__), Path(__file__).resolve().parents[1] / 'src/echoproof/models/lr_detector.py', Path(__file__).resolve().parents[1] / 'src/echoproof/evaluation/metrics.py']},
              'split_sha256': split_hash, 'feature_plan_sha256': manifest['plan_sha256'],
              'feature_schema_hash': FEATURE_SCHEMA_HASH, 'input_manifest_sha256': sha(Path(feature_dir) / 'r2_s1_manifest.json'),
              'input_files': manifest['outputs'], 'trials': [{k: v for k, v in t.items() if k != 'model'} for t in trials]}
    if len(trials) != 8 or any(t['status'] != 'completed' or not t['diagnostics']['converged'] for t in trials):
        report['status'] = 'needs_attention'
        (output / 'evidence.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        raise RuntimeError('one or more trials failed or did not converge; see evidence.json')
    selected = result['selected_model']
    selected.save(output / 'model')
    scores = selected.predict_proba_features(val['X'])
    loaded = LRDetector.load(output / 'model')
    parity = float(np.max(np.abs(scores - loaded.predict_proba_features(val['X']))))
    if parity > 1e-12:
        raise RuntimeError('model reload parity failed')
    rec = aggregate_recordings(val['chunk_id'], val['recording_id'], val['y'], scores)
    component_by_id = dict(zip(val['recording_id'], val['component']))
    with (output / 'predictions_chunks.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['chunk_id', 'recording_id', 'label', 'component', 'raw_score'])
        w.writerows(zip(val['chunk_id'], val['recording_id'], val['y'], val['component'], scores))
    with (output / 'predictions_recordings.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['recording_id', 'label', 'component', 'raw_score', 'chunks'])
        w.writerows((rid, label, component_by_id[rid], score, count) for rid, label, score, count in zip(rec.recording_ids, rec.labels, rec.scores, rec.chunk_counts))
    with (output / 'coefficients.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['feature', 'standardized_coefficient']); w.writerows(selected.coefficient_table())
    _, first = np.unique(train['recording_id'], return_index=True)
    majority = int(np.bincount(train['y'][first], minlength=2).argmax())
    rng = np.random.default_rng(42)
    report.update(status='complete', selected_order=result['selected'], selected_config=result['selected_config'],
                  reload_max_abs=parity, selection_recordings=len(rec.labels), selection_chunks=len(scores),
                  references={'majority_class': majority, 'majority_metrics': compute_metrics(rec.labels, [float(majority)] * len(rec.labels), 0.5),
                              'random_seed': 42, 'random_metrics': compute_metrics(rec.labels, rng.random(len(rec.labels)), 0.5)})
    report['selection_panels'] = {}
    for name, components in [('known', {'genuine', 'melo', 'vits'}), ('unseen_xtts', {'genuine', 'xtts'})]:
        indices = [i for i, rid in enumerate(rec.recording_ids) if component_by_id[rid] in components]
        if len({rec.labels[i] for i in indices}) == 2:
            report['selection_panels'][name] = compute_metrics([rec.labels[i] for i in indices], [rec.scores[i] for i in indices], 0.5)
    report['outputs'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    (output / 'evidence.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True, help='new, unique run directory')
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--report', help='optional new evidence JSON path')
    args = parser.parse_args()
    if args.report and Path(args.report).exists():
        parser.error('report already exists')
    try:
        print(json.dumps(run(args.features, args.output, args.ledger), indent=2))
    finally:
        evidence = Path(args.output) / 'evidence.json'
        if args.report and evidence.exists():
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open('x', encoding='utf-8') as f:
                f.write(evidence.read_text(encoding='utf-8'))
