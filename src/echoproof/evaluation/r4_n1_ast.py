"""AST-A recording calibration. No model training, test scoring or automatic lock."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from echoproof.evaluation.calibrate import fit_temperature, apply_temperature
from echoproof.evaluation.affine_calibration import fit_affine, apply_affine
from echoproof.evaluation.threshold import select_threshold
from echoproof.evaluation.metrics import compute_metrics
from echoproof.evaluation.reliability import reliability_bins, expected_calibration_error
from echoproof.evaluation.bootstrap import paired_group_bootstrap

ORDER = ('identity', 'temperature', 'affine')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)


def mapping(method, params, scores, eps=1e-7):
    if method == 'identity':
        return np.asarray(scores, dtype=float)
    if method == 'temperature':
        return np.asarray(apply_temperature(list(scores), params['temperature'], epsilon=eps))
    if method == 'affine':
        return np.asarray(apply_affine(list(scores), params['a'], params['b'], epsilon=eps))
    raise ValueError('Unknown mapping')


def fit(method, y, scores, recipe):
    try:
        if method == 'identity':
            return {'ok': True, 'params': {}}
        if method == 'temperature':
            cfg = recipe['temperature']
            t = float(fit_temperature(list(y), list(scores), epsilon=recipe['epsilon'], bounds=tuple(cfg['bounds'])))
            lo, hi = cfg['bounds']; tol = cfg['bound_tolerance']
            return {'ok': math.isfinite(t) and lo+tol < t < hi-tol, 'params': {'temperature': t}}
        cfg = recipe['affine']
        result = fit_affine(list(y), list(scores), epsilon=recipe['epsilon'], u_bounds=tuple(cfg['u_bounds']),
                            b_bounds=tuple(cfg['b_bounds']), reg=cfg['regularisation'], start=tuple(cfg['start']))
        return {'ok': result['ok'], 'params': {k: result[k] for k in ('a', 'b', 'u')}, 'diagnostics': result}
    except (ValueError, FloatingPointError, RuntimeError) as exc:
        return {'ok': False, 'params': {}, 'error': str(exc)}


def quality(y, s):
    p = np.clip(s, 1e-7, 1-1e-7)
    return {'log_loss': float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'brier': float(np.mean((np.asarray(s)-y)**2)),
            'ece_10_equal_width': expected_calibration_error(list(y), list(s), 10)}


def loso(y, s, groups, recipe):
    folds = []; losses = {m: [] for m in ORDER}; excluded = {}
    for group in sorted(set(groups)):
        held = groups == group
        if set(y[~held]) != {0, 1} or set(y[held]) != {0, 1}:
            raise ValueError('LOSO class coverage failure')
        row = {'speaker': str(group), 'fits': {}}
        for method in ORDER:
            fitted = fit(method, y[~held], s[~held], recipe)
            row['fits'][method] = fitted
            if not fitted['ok']:
                excluded[method] = 'failed or bound-hitting fold fit'
                continue
            q = quality(y[held], mapping(method, fitted['params'], s[held], recipe['epsilon']))
            row[method] = q
            losses[method].append(q['log_loss'])
        folds.append(row)
    means = {m: float(np.mean(v)) for m, v in losses.items() if m not in excluded}
    best = min(means, key=lambda m: (means[m], ORDER.index(m)))
    se = float(np.std(losses[best], ddof=1)/np.sqrt(len(losses[best])))
    selected = next(m for m in ORDER if m in means and means[m] <= means[best]+se)
    return {'folds': folds, 'means': means, 'excluded': excluded, 'best': best, 'standard_error': se, 'selected': selected}


def accepted(method, fitted, raw, calibrated):
    return bool(method != 'identity' and fitted['ok'] and calibrated['log_loss'] < raw['log_loss']
                and calibrated['brier'] <= raw['brier'])


def validate_rows(rows, expected, role, count):
    if len(rows) != count or len({r['recording_id'] for r in rows}) != count:
        raise ValueError('Role size or duplicate recording failure')
    if {r['recording_id'] for r in rows} != set(expected):
        raise ValueError('Canonical membership differs')
    for row in rows:
        assignment = expected[row['recording_id']]
        if row['validation_role'] != role or row['group_id'] != assignment.group_id:
            raise ValueError('Role/speaker mismatch')
        if row['label'] not in (0, 1) or row['label'] != int(row['component'] != 'genuine'):
            raise ValueError('Invalid label')
        if not math.isfinite(row['raw_score']) or not 0 <= row['raw_score'] <= 1:
            raise ValueError('Invalid score')


def run(root, recipe_path, output):
    import platform
    import pyarrow.parquet as pq
    from echoproof.data.splits import load_assignments
    root, output = Path(root), Path(output)
    recipe = json.loads(Path(recipe_path).read_text())
    if recipe['protocol'] != 'A' or not recipe['status'].startswith('team_agreed'):
        raise ValueError('Approved A recipe required')
    for path, sha in recipe['input_hashes'].items():
        if digest(root/path) != sha:
            raise ValueError(f'Input changed: {path}')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'prospective.json', {'recipe': recipe, 'recipe_sha256': digest(recipe_path),
          'source_sha256': {str(Path(__file__).relative_to(root)): digest(__file__)},
          'python': platform.python_version(), 'numpy': np.__version__, 'status': 'candidate_not_approved'})
    base = root/recipe['input_directory']
    evidence = json.loads((base/'evidence.json').read_text())
    assignments = load_assignments(root/'data/splits/protocol_A_v1.parquet', sealed=False)
    roles = {}; speakers = {}
    # Read identity metadata without operating scores until the fit is frozen.
    for role, count in recipe['roles'].items():
        expected = {a.recording_id: a for a in assignments if a.partition == 'validation' and a.validation_role == role}
        p = base/f'predictions_recordings_{role}.parquet'
        if digest(p) != evidence['output_sha256'][p.name]:
            raise ValueError('Prediction hash mismatch')
        meta = pq.read_table(p, columns=['recording_id','group_id','label','component','validation_role']).to_pylist()
        validate_rows([{**r, 'raw_score': .5} for r in meta], expected, role, count)
        if {r['component'] for r in meta} != set(recipe['components']):
            raise ValueError('Component coverage mismatch')
        speakers[role] = {r['group_id'] for r in meta}
    for a in speakers:
        for b in speakers:
            if a != b and speakers[a] & speakers[b]:
                raise ValueError('Speaker role overlap')
    def load(role):
        rows = pq.read_table(base/f'predictions_recordings_{role}.parquet').to_pylist()
        expected = {a.recording_id:a for a in assignments if a.partition == 'validation' and a.validation_role == role}
        validate_rows(rows, expected, role, recipe['roles'][role])
        # Canonical chunk probabilities must aggregate to the stored raw recording score.
        cp = base/f'predictions_chunks_{role}.parquet'
        if digest(cp) != evidence['output_sha256'][cp.name]:
            raise ValueError('Chunk hash mismatch')
        chunks = pq.read_table(cp).to_pylist(); grouped = {}
        for c in chunks:
            grouped.setdefault(c['recording_id'], []).append(c['tts_probability'])
        for row in rows:
            values = grouped.pop(row['recording_id'])
            if len(values) != row['chunk_count'] or not np.isfinite(values).all() or not np.isclose(np.mean(values), row['raw_score'], atol=1e-12, rtol=0):
                raise ValueError('Recording aggregation differs')
        if grouped:
            raise ValueError('Unexpected chunks')
        return rows, np.array([r['label'] for r in rows]), np.array([r['raw_score'] for r in rows]), np.array([r['group_id'] for r in rows])
    _, y, s, groups = load('calibration')
    cv = loso(y, s, groups, recipe)
    method = cv['selected']; fitted = fit(method, y, s, recipe)
    if not fitted['ok']:
        method = 'identity'; fitted = fit(method, y, s, recipe)
    fit_record = {'cv': cv, 'method': method, 'fit': fitted, 'calibration_quality': quality(y, mapping(method, fitted['params'], s))}
    write(output/'fit.json', fit_record)
    rows, y, s, groups = load('operating')
    cal = mapping(method, fitted['params'], s)
    raw_q, cal_q = quality(y, s), quality(y, cal)
    keep = accepted(method, fitted, raw_q, cal_q)
    deployed = cal if keep else s
    rule, f1, fpr = select_threshold(list(y), list(deployed))
    def metrics(scores):
        return compute_metrics(list(y), list(scores), threshold=rule.threshold if rule.kind == 'threshold' else 1.0000001)
    # Count tied score-induced partitions with the selector's 1e-12 rounding convention.
    tied = []
    for t in np.unique(deployed):
        pred = deployed >= t; tp = int(sum(pred & (y==1))); fp = int(sum(pred & (y==0))); tn = int(sum(~pred & (y==0))); fn = int(sum(~pred & (y==1)))
        score = .5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))
        if round(score/1e-12) == round(f1/1e-12):
            tied.append(float(t))
    result = {'method': method, 'parameters': fitted['params'], 'accepted': keep,
              'score_type': 'calibrated_probability' if keep else 'model_score',
              'rule': rule.to_dict(), 'raw': raw_q, 'calibrated': cal_q, 'macro_f1': f1, 'genuine_fpr': fpr,
              'threshold_plateau': {'tied_score_partitions': len(tied), 'candidate_thresholds': tied},
              'fit_sha256': digest(output/'fit.json'), 'operating_metrics': metrics(deployed),
              'reliability': {'raw': reliability_bins(list(y),list(s)), 'calibrated': reliability_bins(list(y),list(cal))},
              'speakers': {str(g): {'genuine':int(sum((groups==g)&(y==0))), 'false_alarms':int(sum((groups==g)&(y==0)&(deployed >= rule.threshold))) if rule.kind=='threshold' else 0} for g in sorted(set(groups))},
              'bootstrap': {m: paired_group_bootstrap(list(groups), list(y), {'raw':list(s),'calibrated':list(cal)},
                  lambda yy, ss, key=m:quality(np.array(yy),np.array(ss))[key], n_resamples=recipe['bootstrap']['resamples'], seed=recipe['bootstrap']['seed']) for m in ('log_loss','brier')}}
    result['panels'] = {}
    for component in recipe['components']:
        if component == 'genuine': continue
        mask = np.array([r['component'] in ('genuine',component) for r in rows])
        result['panels'][component] = compute_metrics(list(y[mask]), list(deployed[mask]), threshold=rule.threshold)
    write(output/'operating.json', result)
    write(output/'decisions.json', [{'recording_id':r['recording_id'],'raw':float(s[i]),'calibrated':float(cal[i]),'retained':float(deployed[i]),'decision':int(rule.kind=='threshold' and deployed[i]>=rule.threshold)} for i,r in enumerate(rows)])
    return result
