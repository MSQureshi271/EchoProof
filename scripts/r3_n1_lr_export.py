"""Export frozen LR selection, calibration and operating predictions from verified B features."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from echoproof.artifacts import file_digest
from echoproof.bundle import load_bundle
from echoproof.evaluation.aggregate import aggregate_recordings
from echoproof.models.lr_detector import LRDetector
from r2_m1_lr import load_inputs

CONTRACT = ROOT/'configs/r3_n1_lr_contract.json'


def validate_roles(table, contract):
    roles = ('selection', 'calibration', 'operating')
    fields = ('chunk_id', 'recording_id', 'group_id', 'component', 'role', 'y')
    n = len(table['y'])
    if any(len(table[k]) != n for k in fields) or len(set(table['chunk_id'])) != n:
        raise ValueError('duplicate or misaligned validation rows')
    if set(table['role']) != set(roles):
        raise ValueError('unexpected validation roles')
    groups, recordings = {}, {}
    for cid, rid, group, component, role, label in zip(*(table[k] for k in fields)):
        if not all(isinstance(x, (str, np.str_)) and x for x in (cid, rid, group, component, role)):
            raise ValueError('invalid row identity')
        if component not in {'genuine', 'melo', 'vits', 'xtts'} or label not in (0, 1) or int(label) != (component != 'genuine'):
            raise ValueError('invalid component or label')
        identity = (group, component, role, int(label))
        if rid in recordings and recordings[rid] != identity:
            raise ValueError('inconsistent recording metadata')
        recordings[rid] = identity
        if group in groups and groups[group] != role:
            raise ValueError('speaker crosses validation roles')
        groups[group] = role
    for role in roles:
        mask = table['role'] == role
        observed = {'chunks': int(mask.sum()), 'recordings': len(set(table['recording_id'][mask])),
                    'groups': len(set(table['group_id'][mask]))}
        if any(observed[k] != contract[role][k] for k in observed):
            raise ValueError(f'{role} coverage mismatch: {observed}')


def export(feature_dir, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    c = json.loads(CONTRACT.read_text(encoding='utf-8'))
    bundle, reference = ROOT/c['bundle'], ROOT/c['reference_evidence']
    if file_digest(bundle/'bundle.json') != c['bundle_sha256'] or file_digest(reference) != c['reference_evidence_sha256']:
        raise ValueError('model bundle or original evidence differs from contract')
    ref_evidence = json.loads(reference.read_text(encoding='utf-8'))
    if file_digest(reference.parent/'predictions_chunks.csv') != ref_evidence['outputs']['predictions_chunks.csv']:
        raise ValueError('original LR selection predictions differ from evidence')
    load_bundle(bundle)
    for name, sha in c['model_components_sha256'].items():
        if file_digest(bundle/'model'/name) != sha:
            raise ValueError('selected LR model differs from contract')
    feature_dir = Path(feature_dir)
    if file_digest(feature_dir/'r2_s1_manifest.json') != c['feature_manifest_sha256']:
        raise ValueError('feature manifest differs from contract')
    manifest, _, _ = load_inputs(feature_dir)  # Original R2 validation, including train/validation speaker disjointness.
    info = manifest['outputs']['validation_features']
    if info['sha256'] != c['validation_features_sha256'] or manifest['inputs']['split_sha256'] != c['split_sha256']:
        raise ValueError('validation features or split differ from contract')
    with np.load(feature_dir/info['file'], allow_pickle=False) as archive:
        table = {k: archive[k] for k in archive.files}
    validate_roles(table, c)
    model = LRDetector.load(bundle/'model')
    mask = table['role'] == 'selection'
    selection_scores = model.predict_proba_features(table['X'][mask])
    with (reference.parent/'predictions_chunks.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    saved = {r['chunk_id']: r for r in rows}
    ids = table['chunk_id'][mask]
    if len(saved) != len(rows) or set(saved) != set(ids):
        raise ValueError('saved selection coverage mismatch')
    diffs = []
    for cid, rid, label, score in zip(ids, table['recording_id'][mask], table['y'][mask], selection_scores):
        row = saved[cid]
        if row['recording_id'] != rid or int(row['label']) != label:
            raise ValueError('saved selection identity mismatch')
        diffs.append(abs(float(score)-float(row['raw_score'])))
    max_diff = max(diffs)
    if max_diff > 1e-12:
        raise ValueError(f'selected LR score parity failed: {max_diff}')
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/c['output_root']).resolve()) or output == (ROOT/c['output_root']).resolve():
        raise ValueError('output must be a new subdirectory under runs/r3/r3_n1_lr')
    output.mkdir(parents=True, exist_ok=False)
    for role in ('selection', 'calibration', 'operating'):
        mask = table['role'] == role
        ids, rids, labels = table['chunk_id'][mask], table['recording_id'][mask], table['y'][mask]
        scores = selection_scores if role == 'selection' else model.predict_proba_features(table['X'][mask])
        agg = aggregate_recordings(ids, rids, labels, scores)
        metadata = {rid: (str(group), str(component)) for rid, group, component in
                    zip(rids, table['group_id'][mask], table['component'][mask])}
        chunk = pa.table({'chunk_id': ids.tolist(), 'recording_id': rids.tolist(),
                          'group_id': table['group_id'][mask].tolist(), 'component': table['component'][mask].tolist(),
                          'role': [role]*len(ids), 'label': labels.tolist(), 'raw_score': scores.tolist()})
        recording = pa.table({'recording_id': agg.recording_ids, 'group_id': [metadata[r][0] for r in agg.recording_ids],
                              'component': [metadata[r][1] for r in agg.recording_ids], 'role': [role]*len(agg.recording_ids),
                              'label': agg.labels, 'raw_score': agg.scores, 'chunk_count': agg.chunk_counts})
        pq.write_table(chunk, output/f'predictions_chunks_{role}.parquet')
        pq.write_table(recording, output/f'predictions_recordings_{role}.parquet')
    report = {'task': 'R3-N1-LR prediction export', 'status': 'complete',
              'contract_sha256': file_digest(CONTRACT), 'selected_bundle_sha256': c['bundle_sha256'],
              'feature_manifest_sha256': c['feature_manifest_sha256'], 'split_sha256': c['split_sha256'],
              'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
                                (Path(__file__).resolve(), ROOT/'scripts/r2_m1_lr.py',
                                 ROOT/'src/echoproof/models/lr_detector.py', ROOT/'src/echoproof/evaluation/aggregate.py')},
              'selection_max_abs_difference': max_diff, 'selection_tolerance': 1e-12,
              'calibration_fitted': False, 'threshold_selected': False, 'model_retrained': False,
              'outputs': {p.name: file_digest(p) for p in sorted(output.glob('*.parquet'))}}
    (output/'evidence.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.feature_dir, args.output), indent=2))
