"""Isolated exploratory LR calibration comparison; never overwrites a run."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from echoproof.evaluation.calibrate import fit_temperature, apply_temperature

EPS = 1e-7
EXPORT = ROOT / 'runs/r3/r3_n1_lr/selected_lr_v1/export'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(y, p):
    p = np.clip(p, EPS, 1-EPS)
    return {'log_loss': float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
            'brier': float(np.mean((p-y)**2))}

def fit(method, y, p):
    if len(np.unique(y)) != 2:
        raise ValueError('both classes required in each training fold')
    if method == 'raw': return {}
    if method == 'temperature':
        return {'temperature': fit_temperature(y.tolist(), p.tolist(), epsilon=EPS, bounds=(.01, 50.))}
    x = logit(np.clip(p, EPS, 1-EPS))
    def objective(v):
        z = v[0]*x+v[1]
        return float(np.mean(np.logaddexp(0, z)-y*z))
    result = minimize(objective, [1., 0.], method='L-BFGS-B', bounds=[(1e-8, None), (None, None)], options={'maxiter': 2000, 'ftol': 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Platt fit failed: {result.message}')
    return {'slope': float(result.x[0]), 'intercept': float(result.x[1])}

def apply(method, params, p):
    if method == 'raw': return p.copy()
    if method == 'temperature': return np.asarray(apply_temperature(p.tolist(), params['temperature'], epsilon=EPS))
    return expit(params['slope']*logit(np.clip(p, EPS, 1-EPS))+params['intercept'])

def load(role, evidence):
    name = f'predictions_recordings_{role}.parquet'
    path = EXPORT / name
    if digest(path) != evidence['outputs'][name]: raise ValueError('prediction hash mismatch')
    rows = pq.read_table(path).to_pylist()
    expected = {'calibration': (960, 12), 'operating': (1040, 13)}[role]
    if (len(rows), len({r['group_id'] for r in rows})) != expected: raise ValueError('role coverage mismatch')
    if len({r['recording_id'] for r in rows}) != len(rows) or any(r['role'] != role for r in rows): raise ValueError('invalid role or identity')
    y = np.array([r['label'] for r in rows], dtype=int)
    p = np.array([r['raw_score'] for r in rows], dtype=float)
    groups = np.array([r['group_id'] for r in rows])
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not set(y).issubset({0, 1}): raise ValueError('invalid inputs')
    return rows, y, p, groups

def main(output):
    output = output.resolve()
    if EXP.resolve() not in output.parents or output.exists(): raise ValueError('output must be a new directory inside experiment')
    evpath = EXPORT / 'evidence.json'
    evidence = json.loads(evpath.read_text(encoding='utf-8'))
    if evidence['status'] != 'complete' or evidence['selected_bundle_sha256'] != 'bbf7983794fe0ce4f0e59231b77c31442c9c8dfd2bc88ce49a03a30216cd440d': raise ValueError('wrong export provenance')
    rows, y, p, groups = load('calibration', evidence)
    methods = ('raw', 'temperature', 'platt')
    oof = {m: np.empty(len(y)) for m in methods}
    folds = []
    for group in sorted(set(groups)):
        train, test = groups != group, groups == group
        for method in methods:
            params = fit(method, y[train], p[train])
            oof[method][test] = apply(method, params, p[test])
            folds.append({'speaker': group, 'method': method, **metrics(y[test], oof[method][test])})
    cv = {m: metrics(y, oof[m]) for m in methods}
    chosen = min(methods, key=lambda m: cv[m]['log_loss'])
    params = fit(chosen, y, p)
    # Selection and fitting precede access to operating data.
    op_rows, oy, op, og = load('operating', evidence)
    if set(groups) & set(og) or {r['recording_id'] for r in rows} & {r['recording_id'] for r in op_rows}: raise ValueError('role overlap')
    result = apply(chosen, params, op)
    output.mkdir(parents=True)
    with (output/'calibration_oof.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['recording_id', 'group_id', 'label', *methods])
        for i, r in enumerate(rows): w.writerow([r['recording_id'], r['group_id'], int(y[i]), *(oof[m][i] for m in methods)])
    with (output/'operating_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['recording_id', 'group_id', 'label', 'raw', 'selected'])
        for i, r in enumerate(op_rows): w.writerow([r['recording_id'], r['group_id'], int(oy[i]), op[i], result[i]])
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete', 'experimental_only': True,
              'source_export_evidence_sha256': digest(evpath), 'source_calibration_sha256': evidence['outputs']['predictions_recordings_calibration.parquet'],
              'source_operating_sha256': evidence['outputs']['predictions_recordings_operating.parquet'], 'script_sha256': digest(Path(__file__)),
              'plan_sha256': digest(EXP/'reports/plan.md'), 'cv': cv, 'speaker_folds': folds,
              'chosen_method': chosen, 'fit_parameters': params, 'operating_raw': metrics(oy, op), 'operating_selected': metrics(oy, result),
              'warning': 'Operating role previously inspected; results are exploratory, not independent test performance.',
              'outputs': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    (output/'evidence.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
