"""Export frozen R4-M1 model predictions for the three A validation roles."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from echoproof.evaluation.aggregate import aggregate_recordings
from echoproof.models.lr_detector import LRDetector
from echoproof.training.r4_m1_data import build_data, file_sha256
from echoproof.training.r3_m1_export import aggregate_role, score_dataset

ROLES = ('selection', 'calibration', 'operating')


def check_selection(rows, saved, tolerance):
    """Match IDs, labels and scores, independently of row ordering."""
    expected = {r['chunk_id']: r for r in saved}
    if len(expected) != len(saved) or len(rows) != len(saved):
        raise ValueError('saved selection coverage mismatch')
    differences = []
    seen = set()
    for cid, rid, label, score in rows:
        if cid in seen or cid not in expected: raise ValueError('unexpected selection chunk')
        seen.add(cid); old = expected[cid]
        if old['recording_id'] != rid or int(old['label']) != int(label): raise ValueError('selection identity mismatch')
        prior = old.get('raw_score', old.get('tts_probability'))
        if prior is None or not np.isfinite(score) or not np.isfinite(prior): raise ValueError('invalid selection score')
        differences.append(abs(float(score)-float(prior)))
    maximum = max(differences, default=float('inf'))
    if maximum > tolerance: raise ValueError(f'selection reload parity failed: {maximum}')
    return maximum


def verify_training(root, run_dir, model_type, data):
    root, run_dir = Path(root), Path(run_dir)
    recipe_sha = file_sha256(root/'configs/r4_m1_recipes.json')
    path = run_dir/('evidence.json' if model_type == 'lr' else 'training_evidence.json')
    evidence = json.loads(path.read_text(encoding='utf-8'))
    if evidence.get('protocol') != 'A' or evidence.get('split_sha256') != data['split_sha256']:
        raise ValueError('training protocol or split mismatch')
    if model_type == 'lr':
        if evidence.get('task') != 'R4-M1-LR' or evidence.get('status') != 'complete' or evidence.get('recipe_sha256') != recipe_sha or evidence.get('input_sha256') != data['input_sha256']:
            raise ValueError('incompatible LR training evidence')
        required = ('model/weights.npz', 'model/lr_model.json', 'predictions_chunks_selection.parquet', 'predictions_recordings_selection.parquet')
        for name in required:
            if file_sha256(run_dir/name) != evidence['outputs'][name]: raise ValueError(f'LR artifact mismatch: {name}')
        if not evidence['fit_diagnostics']['converged']: raise ValueError('LR did not converge')
        model = LRDetector.load(run_dir/'model')
        if model.meta['hyperparameters'] != data['recipe']['lr']['hyperparameters'] or model.meta.get('protocol') != 'A' or model.meta.get('split_sha256') != data['split_sha256']:
            raise ValueError('LR model metadata mismatch')
        return evidence, model
    if model_type != 'bigru': raise ValueError('unknown model')
    import torch
    from echoproof.models.bigru_waveform import WaveformBiGRU
    from echoproof.training.r4_m1_bigru import verify_resume
    if evidence.get('task') != 'R4-M1-BiGRU' or evidence.get('status') != 'training_complete': raise ValueError('incomplete BiGRU training')
    contract_path = run_dir/'experiment_contract.json'
    if file_sha256(contract_path) != evidence['contract_sha256']: raise ValueError('BiGRU contract mismatch')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    if contract['recipe_sha256'] != recipe_sha or contract['input_sha256'] != data['input_sha256'] or contract['split_sha256'] != data['split_sha256']:
        raise ValueError('BiGRU input provenance mismatch')
    for name, sha in contract['sources'].items():
        if file_sha256(root/name) != sha: raise ValueError(f'BiGRU source mismatch: {name}')
    if file_sha256(run_dir/'best.pt') != evidence['best_checkpoint_sha256'] or file_sha256(run_dir/'latest.pt') != evidence['latest_checkpoint_sha256']:
        raise ValueError('BiGRU checkpoint mismatch')
    for name, sha in evidence['selection_predictions_sha256'].items():
        if file_sha256(run_dir/name) != sha: raise ValueError('BiGRU selection predictions mismatch')
    latest = verify_resume(run_dir)
    checkpoint = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=True)
    if not latest['finished'] or checkpoint['epoch'] != evidence['best_epoch'] or checkpoint['config_hash'] != evidence['config_hash']:
        raise ValueError('BiGRU selection not frozen')
    if not torch.cuda.is_available(): raise RuntimeError('BiGRU export requires CUDA for selection parity')
    model = WaveformBiGRU(data['mel_stats']['mean'], data['mel_stats']['std'], hidden_size=128, dropout=.3)
    model.load_state_dict(checkpoint['model'], strict=True); model.to('cuda')
    return evidence, model


def export(root, shard_dir, run_dir, output, model_type):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, run_dir, output = (Path(p).resolve() for p in (root, run_dir, output))
    run_root = (root/'runs/r4').resolve()
    if run_dir.parent != run_root or output.parent != run_root or output.exists():
        raise ValueError('training and new output directories must be direct children of runs/r4')
    data = build_data(root, shard_dir)
    try:
        evidence, model = verify_training(root, run_dir, model_type, data)
        saved_name = 'predictions_chunks_selection.parquet' if model_type == 'lr' else 'predictions_chunks.parquet'
        saved = pq.read_table(run_dir/saved_name).to_pylist()
        def score(role):
            if model_type == 'bigru': return score_dataset(model, data['validation'][role], 'cuda', batch_size=64)
            table = data['lr_validation']; mask = table['role'] == role
            return list(zip(table['chunk_id'][mask].tolist(), table['recording_id'][mask].tolist(),
                            table['y'][mask].tolist(), model.predict_proba_features(table['X'][mask]).tolist()))
        # Confirm frozen selection parity before accessing later-role audio or scoring features.
        selection = score('selection')
        tolerance = 1e-12 if model_type == 'lr' else 1e-6
        selection_diff = check_selection(selection, saved, tolerance)
        output.mkdir(parents=True, exist_ok=False)
        counts = {}
        for role in ROLES:
            rows = selection if role == 'selection' else score(role)
            recordings = aggregate_role(data['validation'][role], rows)
            groups, components = data['validation_groups'], data['validation_components']
            chunks = [{'chunk_id': cid, 'recording_id': rid, 'group_id': groups[rid], 'component': components[rid],
                       'role': role, 'label': int(label), 'raw_score': float(value)} for cid, rid, label, value in rows]
            recs = [{'recording_id': rid, 'group_id': groups[rid], 'component': components[rid], 'role': role,
                     'label': int(label), 'raw_score': float(value), 'chunk_count': int(n)} for rid, label, value, n in recordings]
            for kind, content in [('chunks', chunks), ('recordings', recs)]:
                pq.write_table(pa.Table.from_pylist(content), output/f'predictions_{kind}_{role}.parquet')
            persisted = pq.read_table(output/f'predictions_chunks_{role}.parquet').to_pylist()
            persisted_recs = pq.read_table(output/f'predictions_recordings_{role}.parquet').to_pylist()
            if persisted != chunks or persisted_recs != recs:
                raise ValueError(f'{role} export readback mismatch')
            reaggregated = aggregate_recordings([r['chunk_id'] for r in persisted], [r['recording_id'] for r in persisted],
                                                [r['label'] for r in persisted], [r['raw_score'] for r in persisted])
            actual = {rid: (int(y), float(s), int(n)) for rid, y, s, n in zip(reaggregated.recording_ids, reaggregated.labels, reaggregated.scores, reaggregated.chunk_counts)}
            for r in persisted_recs:
                expected = actual.get(r['recording_id'])
                if expected is None or expected[0] != r['label'] or expected[2] != r['chunk_count'] or abs(expected[1]-r['raw_score']) > 1e-12:
                    raise ValueError(f'{role} recording aggregation mismatch')
            counts[role] = {'chunks': len(chunks), 'recordings': len(recs), 'groups': len({r['group_id'] for r in recs})}
            if counts[role]['chunks'] != data['counts']['validation_chunks'][role] or counts[role]['recordings'] != data['counts']['validation_recordings'][role]:
                raise ValueError(f'{role} export coverage mismatch')
        report = {'task': 'R4-M1 validation export', 'model': model_type, 'protocol': 'A', 'status': 'complete',
                  'training_evidence_sha256': file_sha256(run_dir/('evidence.json' if model_type == 'lr' else 'training_evidence.json')),
                  'selected_model_sha256': file_sha256(run_dir/('model/weights.npz' if model_type == 'lr' else 'best.pt')),
                  'recipe_sha256': file_sha256(root/'configs/r4_m1_recipes.json'), 'input_sha256': data['input_sha256'],
                  'split_sha256': data['split_sha256'], 'selection_reload_max_abs': selection_diff,
                  'selection_reload_tolerance': tolerance, 'counts': counts,
                  'calibration_fitted': False, 'threshold_selected': False, 'test_scored': False,
                  'source_sha256': file_sha256(Path(__file__)),
                  'outputs': {p.name: file_sha256(p) for p in sorted(output.glob('*.parquet'))}}
        (output/'evidence.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
        return report
    finally:
        data['loader'].close()
