"""Run the approved, isolated two-fit LR energy/activity-proxy ablation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
from echoproof.artifacts import digest, file_digest
from echoproof.evaluation.aggregate import aggregate_recordings
from echoproof.evaluation.metrics import compute_metrics
from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH
from echoproof.models.lr_detector import LRDetector
from echoproof.models.lr_subset import LRSubsetDetector, REMOVED_FEATURES, SUBSET_SCHEMA_HASH
from echoproof.training.ledger import append_ledger
from r2_m1_lr import load_inputs

EXP = ROOT/'experiments/r3_m2'
PLAN = ROOT/'configs/r3_m2_ablation.json'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')


def lock_snapshot(approved_locks=None):
    result = {}
    for p in sorted((ROOT/'locks').glob('*_lock.json')):
        name = p.relative_to(ROOT).as_posix()
        raw = p.read_bytes().replace(b'\r\n', b'\n')
        raw_hash = hashlib.sha256(raw).hexdigest()
        canonical = json.dumps(json.loads(raw), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        canonical_hash = hashlib.sha256(canonical).hexdigest()
        expected = (approved_locks or {}).get(name)
        result[name] = expected if expected == canonical_hash else raw_hash
    return result


def approval_plan_hash(path, approved_hash):
    """Accept only byte-identical content or equivalent LF/CRLF line endings."""
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    candidates = (raw, lf, lf.replace(b'\n', b'\r\n'))
    hashes = {hashlib.sha256(value).hexdigest() for value in candidates}
    return approved_hash if approved_hash in hashes else hashlib.sha256(raw).hexdigest()


def validate_approval(approval, plan_sha, locks):
    if (approval.get('approved_by') != 'Muneeb' or approval.get('approved') is not True
            or approval.get('scope') != 'r3_m2_fixed_feature_ablation_only'
            or approval.get('plan_sha256') != plan_sha or approval.get('lock_sha256_lf') != locks
            or approval.get('official_promotion_allowed') is not False):
        raise ValueError('explicit R3-M2 approval for this plan/current locks is required; M1 approval does not apply')


def prepare_output(output, ledger_path):
    """Recover an empty preflight directory only when no fit was recorded."""
    output, ledger_path = Path(output), Path(ledger_path)
    if ledger_path.exists():
        with ledger_path.open(newline='', encoding='utf-8') as f:
            if any(row.get('run_id') in {'r3_m2_full67_control', 'r3_m2_without_energy_activity60'}
                   for row in csv.DictReader(f)):
                raise FileExistsError('ablation ledger already records a fit; do not restart')
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError('ablation output is not empty; inspect evidence, do not restart')
    else:
        output.mkdir(parents=True, exist_ok=False)


def score(model, val):
    scores = model.predict_proba_features(val['X'])
    agg = aggregate_recordings(val['chunk_id'], val['recording_id'], val['y'], scores)
    components = dict(zip(val['recording_id'], val['component']))
    panels = {}
    for name, allowed in [('known', {'genuine', 'melo', 'vits'}), ('unseen_xtts', {'genuine', 'xtts'})]:
        indices = [i for i, rid in enumerate(agg.recording_ids) if components[rid] in allowed]
        panels[name] = compute_metrics([agg.labels[i] for i in indices], [agg.scores[i] for i in indices], .5)
    return scores, agg, compute_metrics(agg.labels, agg.scores, .5), panels


def run(feature_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq

    approval_path = EXP/'configs/authorization.json'
    if not approval_path.is_file():
        raise ValueError('R3-M2 fitting is disabled until Muneeb approves the isolated ablation; no authorization file exists')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    approval = json.loads(approval_path.read_text(encoding='utf-8'))
    validate_approval(approval, approval_plan_hash(PLAN, approval.get('plan_sha256')),
                      lock_snapshot(approval.get('lock_sha256_lf')))
    if (plan['removed_features'] != list(REMOVED_FEATURES) or plan['source_feature_schema_hash'] != FEATURE_SCHEMA_HASH
            or plan['max_fits'] != 2 or plan['max_cpu_minutes'] != 60):
        raise ValueError('unexpected ablation definition')
    hp = {'C': .01, 'class_weight': 'none', 'max_iter': 2000, 'seed': 42}
    if plan['hyperparameters'] != hp:
        raise ValueError('hyperparameters differ from the declared fixed baseline')
    reference = ROOT/'runs/r2/r2_m1_20261106T140611Z'
    evidence = json.loads((reference/'evidence.json').read_text(encoding='utf-8'))
    manifest, train, val = load_inputs(feature_dir)
    if file_digest(Path(feature_dir)/'r2_s1_manifest.json') != evidence['input_manifest_sha256']:
        raise ValueError('feature manifest differs from R2-M1')
    if manifest['outputs'] != evidence['input_files']:
        raise ValueError('feature artifacts differ from R2-M1')
    if (len(set(train['recording_id'])) != 4530 or len(train['y']) != 21470 or len(set(train['group_id'])) != 151
            or len(set(val['recording_id'])) != 2000 or len(val['y']) != 10029 or len(set(val['group_id'])) != 25):
        raise ValueError('unexpected cohort coverage')
    for rel, expected in evidence['outputs'].items():
        if file_digest(reference/rel) != expected:
            raise ValueError(f'original LR evidence changed: {rel}')
    saved = LRDetector.load(reference/'model')
    baseline_scores, _, baseline_metrics, baseline_panels = score(saved, val)
    with (reference/'predictions_chunks.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    by_id = {row['chunk_id']: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(val['chunk_id']):
        raise ValueError('saved LR selection chunk coverage differs')
    expected_scores = []
    for cid, rid, label in zip(val['chunk_id'], val['recording_id'], val['y']):
        row = by_id[cid]
        if row['recording_id'] != rid or int(row['label']) != label:
            raise ValueError('saved LR selection identities differ')
        expected_scores.append(float(row['raw_score']))
    baseline_max_abs = float(np.max(np.abs(baseline_scores-np.asarray(expected_scores))))
    if baseline_max_abs > 1e-12:
        raise ValueError('preserved LR chunk scores did not reproduce')
    with (reference/'coefficients.csv').open(newline='', encoding='utf-8') as f:
        stored_coefficients = [(row['feature'], float(row['standardized_coefficient'])) for row in csv.DictReader(f)]
    if stored_coefficients != saved.coefficient_table():
        raise ValueError('saved LR coefficient table differs from model weights')
    original_metrics = evidence['trials'][evidence['selected_order']]['metrics']
    if abs(baseline_metrics['macro_f1']-original_metrics['macro_f1']) > 1e-12:
        raise ValueError('preserved LR selection metric did not reproduce')
    output = EXP/'runs/ablation01'
    protected = {str(p.relative_to(ROOT)): file_digest(p) for p in (ROOT/'locks').glob('*_lock.json')}
    protected.update({str((reference/rel).relative_to(ROOT)): expected for rel, expected in evidence['outputs'].items()})
    resolved = {'task': 'R3-M2 LR ablation', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'plan': plan, 'plan_sha256': file_digest(PLAN), 'approval': approval,
                'reference_evidence_sha256': file_digest(reference/'evidence.json'), 'inputs': manifest['outputs'],
                'split_sha256': evidence['split_sha256'], 'subset_schema_hash': SUBSET_SCHEMA_HASH,
                'protected_files': protected,
                'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
                    [Path(__file__).resolve(), ROOT/'scripts/r2_m1_lr.py', ROOT/'src/echoproof/models/lr_subset.py',
                     ROOT/'src/echoproof/models/lr_detector.py', ROOT/'src/echoproof/features/acoustic.py']},
                'environment': {'python': platform.python_version(), **{name: version(name) for name in
                                ('numpy', 'scipy', 'scikit-learn', 'pyarrow', 'librosa')}}}
    prepare_output(output, EXP/'ledger.csv')
    write_json(output/'resolved_config.json', resolved)
    marker = output/'active_attempt.json'
    write_json(marker, {'started_utc': resolved['created_utc'], 'resolved_sha256': file_digest(output/'resolved_config.json')})
    started = time.monotonic()
    report = {'task': 'R3-M2 LR ablation', 'status': 'running', 'official_selection_changed': False,
              'baseline_saved': {'metrics': baseline_metrics, 'panels': baseline_panels,
                                 'chunk_score_max_abs': baseline_max_abs, 'coefficient_table_verified': True}, 'fits': [],
              'resolved_sha256': file_digest(output/'resolved_config.json'), 'review_status': 'pending',
              'scope': 'isolated fixed-hyperparameter energy/activity-proxy ablation; no automatic adoption'}
    try:
        for name, cls in [('full67_control', LRDetector), ('without_energy_activity60', LRSubsetDetector)]:
            if time.monotonic()-started >= plan['max_cpu_minutes']*60:
                raise RuntimeError('ablation time budget exhausted at fit boundary')
            tick = time.monotonic()
            directory = output/name
            directory.mkdir()
            ledger = {'run_id': 'r3_m2_'+name, 'owner': 'Muneeb', 'model': 'lr', 'protocol': 'B',
                      'seed': 42, 'split_hash': evidence['split_sha256'], 'device': 'cpu',
                      'config_hash': digest({'resolved_sha256': report['resolved_sha256'], 'variant': name}), 'status': 'failed'}
            try:
                model = cls.fit(train['X'], train['y'], train['recording_id'], **hp)
                if not model.meta['diagnostics']['converged']:
                    raise RuntimeError('LR did not converge within the fixed budget')
                model.save(directory/'model')
                scores, agg, metrics, panels = score(model, val)
                loaded = cls.load(directory/'model')
                parity = float(np.max(np.abs(scores-loaded.predict_proba_features(val['X']))))
                if parity > 1e-12:
                    raise RuntimeError('subset/full LR reload parity failed')
                pq.write_table(pa.table({'chunk_id': val['chunk_id'], 'recording_id': val['recording_id'],
                                         'label': val['y'], 'raw_score': scores}), directory/'predictions_chunks.parquet')
                pq.write_table(pa.table({'recording_id': agg.recording_ids, 'label': agg.labels,
                                         'raw_score': agg.scores, 'chunk_count': agg.chunk_counts}), directory/'predictions_recordings.parquet')
                coefs = dict(model.coefficient_table())
                with (directory/'coefficients.csv').open('w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['feature', 'included', 'standardized_coefficient', 'absolute_rank', 'group'])
                    ranks = {key: i for i, (key, _) in enumerate(model.coefficient_table(), 1)}
                    for feature in FEATURE_NAMES:
                        w.writerow([feature, feature in coefs, coefs.get(feature, ''), ranks.get(feature, ''),
                                    'energy_activity_proxy' if feature in REMOVED_FEATURES else 'retained_acoustic'])
                report['fits'].append({'id': name, 'metrics': metrics, 'panels': panels,
                                       'diagnostics': model.meta['diagnostics'], 'reload_max_abs': parity})
                ledger.update(status='completed', epochs_run=model.meta['diagnostics']['n_iter'],
                              val_macro_f1=metrics['macro_f1'], val_eer=metrics['eer'], val_fpr=metrics['fpr'])
            except Exception as exc:
                ledger['notes'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                ledger['train_minutes'] = (time.monotonic()-tick)/60
                append_ledger(EXP/'ledger.csv', ledger)
        if time.monotonic()-started > plan['max_cpu_minutes']*60:
            raise RuntimeError('fits completed but exceeded declared time budget; comparison needs review')
        if any(file_digest(ROOT/rel) != expected for rel, expected in protected.items()):
            raise ValueError('protected evidence changed during ablation')
        validate_approval(approval, approval_plan_hash(PLAN, approval.get('plan_sha256')),
                          lock_snapshot(approval.get('lock_sha256_lf')))
        report['status'] = 'experimental_ablation_complete'
        report['macro_f1_difference_reduced_minus_control'] = report['fits'][1]['metrics']['macro_f1']-report['fits'][0]['metrics']['macro_f1']
        lines = ['# R3-M2 LR feature-group ablation', '', '| Model | Macro F1 | FPR | XTTS recall | XTTS AUC |',
                 '|---|---:|---:|---:|---:|']
        for row in [{'id': 'saved_R2_M1', **report['baseline_saved']}, *report['fits']]:
            lines.append(f"| {row['id']} | {row['metrics']['macro_f1']:.6f} | {row['metrics']['fpr']:.6f} | {row['panels']['unseen_xtts']['tts_recall']:.6f} | {row['panels']['unseen_xtts']['roc_auc']:.6f} |")
        lines += ['', 'Advisory only. No official model changed. Group removal does not eliminate all energy/silence cues.',
                  'Fixed C=0.01; no feature-set-specific tuning, calibration/operating scoring or test access. Subhan review pending.']
        (output/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['artifacts'] = {str(p.relative_to(output)): file_digest(p) for p in output.rglob('*')
                               if p.is_file() and p != marker and p.name != 'evidence.json'}
        write_json(output/'evidence.json', report)
        marker.unlink()
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.features), indent=2))
