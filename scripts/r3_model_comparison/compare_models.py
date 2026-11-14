import sys, json, hashlib
sys.path.insert(0, 'src')
import numpy as np, pyarrow.parquet as pq
from echoproof.evaluation.metrics import compute_metrics, interpolated_eer
from echoproof.evaluation.threshold import select_threshold
from echoproof.evaluation.affine_calibration import fit_affine, apply_affine
from echoproof.evaluation.reliability import expected_calibration_error
from scipy.special import logit, expit
from scipy.optimize import minimize_scalar

ROLES = ['selection', 'calibration', 'operating']
WRUN = 'runs/r3/r3_n1_wavlm/20261114T083533Z-e6ff5a288068/operating/operating_predictions.parquet'


def rows(p):
    return pq.read_table(p).to_pylist()


SRC = {
    'LR': lambda r: f'runs/r3/r3_n1_lr/selected_lr_v1/export/predictions_recordings_{r}.parquet',
    'BiGRU': lambda r: f'experiments/r3_m1/runs/batch01/selected/predictions_recordings_{r}.parquet',
    'AST': lambda r: f'runs/r3/r3_u2_20261112T134049Z/predictions_recordings_{r}.parquet',
    'WavLM': lambda r: f'reports/r3/r3_u1_B/predictions_recordings_{r}.parquet',
}
MODELS = list(SRC)
D, hashes = {}, {}
for m in MODELS:
    D[m] = {}
    for r in ROLES:
        D[m][r] = sorted(rows(SRC[m](r)), key=lambda x: x['recording_id'])
        hashes[f'{m}/{r}'] = hashlib.sha256(open(SRC[m](r), 'rb').read()).hexdigest()[:16]
for r in ROLES:
    ids = [x['recording_id'] for x in D['WavLM'][r]]
    labels = [x['label'] for x in D['WavLM'][r]]
    for m in MODELS:
        assert [x['recording_id'] for x in D[m][r]] == ids, (m, r)
        assert [x['label'] for x in D[m][r]] == labels, (m, r)
A = {m: {r: {'y': np.array([x['label'] for x in D[m][r]]), 's': np.array([x['raw_score'] for x in D[m][r]]),
             'g': np.array([x['group_id'] for x in D[m][r]]), 'c': np.array([x['component'] for x in D[m][r]]),
             'id': [x['recording_id'] for x in D[m][r]]} for r in ROLES} for m in MODELS}
wrun = {x['recording_id']: x['raw_score'] for x in rows(WRUN)}
wmatch = all(wrun[i] == s for i, s in zip(A['WavLM']['operating']['id'], A['WavLM']['operating']['s']))
EPS = 1e-7


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def br(y, p):
    return float(np.mean((p - y) ** 2))


def temp_fit(y, s):
    z = logit(np.clip(s, EPS, 1 - EPS))
    r = minimize_scalar(lambda lt: ll(y, expit(z / np.exp(lt))), bounds=(-3, 3), method='bounded')
    return float(np.exp(r.x))


def temp_apply(s, T):
    return expit(logit(np.clip(s, EPS, 1 - EPS)) / T)


out = {'hashes': hashes, 'wavlm_raw_equals_calibration_run_raw': bool(wmatch)}
cal = {}
for m in MODELS:
    c = A[m]['calibration']
    aff = fit_affine(c['y'].tolist(), c['s'].tolist())
    cal[m] = {'T': temp_fit(c['y'], c['s']), 'a': aff['a'], 'b': aff['b'], 'affine_ok': aff['ok']}
out['calibration_params'] = cal

res = {}
for m in MODELS:
    res[m] = {}
    for r in ROLES:
        a = A[m][r]
        y, s, c = a['y'], a['s'], a['c']
        d = compute_metrics(y.tolist(), s.tolist(), 0.5)
        res[m][r] = {'macro_f1': d['macro_f1'], 'auc': d['roc_auc'], 'eer': d['eer'], 'ap': d['average_precision'], 'fpr': d['fpr'],
                     'recall': d['tts_recall'], 'precision': d['tts_precision'], 'acc': d['accuracy'], 'brier': d['brier_score'],
                     'logloss': d['log_loss'], 'ece': expected_calibration_error(y.tolist(), s.tolist()),
                     'rec_melo': float(np.mean(s[c == 'melo'] >= 0.5)), 'rec_vits': float(np.mean(s[c == 'vits'] >= 0.5)),
                     'rec_xtts': float(np.mean(s[c == 'xtts'] >= 0.5))}
out['raw_at_0_5'] = res

common = {}
for m in MODELS:
    o = A[m]['operating']
    y, s, p = o['y'], o['s'], cal[m]
    t = temp_apply(s, p['T'])
    q = np.array(apply_affine(s.tolist(), p['a'], p['b']))
    common[m] = {'raw': {'logloss': ll(y, s), 'brier': br(y, s), 'ece': expected_calibration_error(y.tolist(), s.tolist())},
                 'temperature': {'logloss': ll(y, t), 'brier': br(y, t), 'ece': expected_calibration_error(y.tolist(), t.tolist()), 'T': p['T']},
                 'affine': {'logloss': ll(y, q), 'brier': br(y, q), 'ece': expected_calibration_error(y.tolist(), q.tolist()), 'a': p['a'], 'b': p['b']}}
out['common_calibration_operating'] = common

thr = {}
for m in MODELS:
    o = A[m]['operating']
    y, s, c = o['y'], o['s'], o['c']
    thr[m] = {}
    for src in ('operating', 'calibration', 'selection'):
        a = A[m][src]
        rule, f1, fpr = select_threshold(a['y'].tolist(), a['s'].tolist())
        pred = (s >= rule.threshold) if rule.threshold is not None else np.zeros_like(s, dtype=bool)
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        tn = int((~pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        f1t = 2 * tp / (2 * tp + fp + fn); f1g = 2 * tn / (2 * tn + fp + fn)
        thr[m][src] = {'threshold_raw': rule.threshold, 'macro_f1': (f1t + f1g) / 2, 'fpr': fp / (fp + tn), 'recall': tp / (tp + fn),
                       'fp': fp, 'fn': fn, 'rec_melo': float(np.mean(pred[c == 'melo'])), 'rec_vits': float(np.mean(pred[c == 'vits'])),
                       'rec_xtts': float(np.mean(pred[c == 'xtts']))}
out['thresholds_applied_to_operating'] = thr

conc = {}
for m in MODELS:
    o = A[m]['operating']
    y, s, g = o['y'], o['s'], o['g']
    pred = s >= thr[m]['operating']['threshold_raw']
    sp = {str(k): (int((pred & (y == 0) & (g == k)).sum()), int(((y == 0) & (g == k)).sum())) for k in np.unique(g)}
    fa = sum(v[0] for v in sp.values())
    top = sorted(sp.items(), key=lambda kv: -kv[1][0])
    conc[m] = {'false_alarms': fa, 'speakers_with_any': sum(1 for v in sp.values() if v[0] > 0),
               'speakers_ge_25pct': sum(1 for v in sp.values() if v[0] / v[1] >= 0.25),
               'top3_share': sum(v[1][0] for v in top[:3]) / max(fa, 1), 'top': [(k, v[0], v[1]) for k, v in top[:4]]}
out['false_alarm_concentration_operating_threshold'] = conc

dm = json.load(open('data/eda/r2_s3_shortcut/duration_matched_B_v1.json'))['ids']
dmset = set(dm['selection'])
dmr = {}
for m in MODELS:
    a = A[m]['selection']
    mask = np.array([i in dmset for i in a['id']])
    y, s = a['y'][mask], a['s'][mask]
    d = compute_metrics(y.tolist(), s.tolist(), 0.5)
    dmr[m] = {'n': int(mask.sum()), 'macro_f1': d['macro_f1'], 'auc': d['roc_auc'], 'eer': d['eer'], 'fpr': d['fpr'],
              'recall': d['tts_recall'], 'rec_xtts': float(np.mean(s[a['c'][mask] == 'xtts'] >= 0.5))}
out['duration_matched_selection_raw_0_5'] = dmr

rng = np.random.default_rng(42)
o = A['WavLM']['operating']
g, y = o['g'], o['y']
spk = np.unique(g)
idx = {k: np.where(g == k)[0] for k in spk}


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def stats(y, s):
    pred = s >= 0.5
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); tn = (~pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
    f1t = 2 * tp / max(2 * tp + fp + fn, 1); f1g = 2 * tn / max(2 * tn + fp + fn, 1)
    return {'auc': auc(y, s), 'macro_f1': float((f1t + f1g) / 2), 'fpr': float(fp / max(fp + tn, 1)), 'recall': float(tp / max(tp + fn, 1)),
            'eer': float(interpolated_eer(y, s))}


KEYS = ('auc', 'macro_f1', 'fpr', 'recall', 'eer')
boot = {m: [] for m in MODELS}
for _ in range(1000):
    ii = np.concatenate([idx[k] for k in rng.choice(spk, len(spk), replace=True)])
    for m in MODELS:
        boot[m].append(stats(y[ii], A[m]['operating']['s'][ii]))


def ci(v):
    v = np.array(v, float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


point = {m: stats(y, A[m]['operating']['s']) for m in MODELS}
out['bootstrap'] = {'resamples': 1000, 'seed': 42, 'unit': 'speaker (13 operating speakers)', 'point_raw_0_5': point,
                    'ci': {m: {k: ci([b[k] for b in boot[m]]) for k in KEYS} for m in MODELS},
                    'paired_vs_wavlm': {m: {k: {'diff': point['WavLM'][k] - point[m][k], 'ci': ci([bw[k] - bm[k] for bw, bm in zip(boot['WavLM'], boot[m])])}
                                            for k in KEYS} for m in MODELS if m != 'WavLM'}}

out['contract_ranking_selection_role'] = sorted(MODELS, key=lambda m: (-res[m]['selection']['macro_f1'], res[m]['selection']['eer'], res[m]['selection']['fpr']))

ag = {}
for i, m in enumerate(MODELS):
    for n in MODELS[i + 1:]:
        a, b = A[m]['operating']['s'], A[n]['operating']['s']
        ag[f'{m}~{n}'] = {'decision_agreement': float(((a >= 0.5) == (b >= 0.5)).mean()),
                          'rank_correlation': float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])}
out['agreement_operating'] = ag

ens = {}
for name, ms in {'mean of all four': MODELS, 'mean of WavLM and AST': ['WavLM', 'AST'], 'mean of LR, BiGRU, AST': ['LR', 'BiGRU', 'AST']}.items():
    s = np.mean([A[m]['operating']['s'] for m in MODELS if m in ms], axis=0)
    d = compute_metrics(y.tolist(), s.tolist(), 0.5)
    ens[name] = {'auc': d['roc_auc'], 'eer': d['eer'], 'macro_f1': d['macro_f1'], 'fpr': d['fpr'], 'recall': d['tts_recall']}
out['descriptive_unweighted_mean_ensembles_operating'] = ens

own = {}
lr = sorted(rows('runs/r3/r3_n1_lr/selected_lr_v1/operating/operating_predictions.parquet'), key=lambda x: x['recording_id'])
yl = np.array([x['label'] for x in lr]); v = np.array([x['chosen_score'] for x in lr])
own['LR, temperature scaling'] = {'logloss': ll(yl, v), 'brier': br(yl, v), 'ece': expected_calibration_error(yl.tolist(), v.tolist())}
bg = sorted(rows('runs/r3/r3_n1_operational_calibration_selected/20261113T132213Z-0b91eafbbec9/operating_predictions.parquet'), key=lambda x: x['recording_id'])
yb = np.array([x['label'] for x in bg])
for k, label in (('calibrated_platt', 'BiGRU, Platt'), ('calibrated_smoothed_isotonic', 'BiGRU, smoothed isotonic')):
    v = np.array([x[k] for x in bg])
    own[label] = {'logloss': ll(yb, v), 'brier': br(yb, v), 'ece': expected_calibration_error(yb.tolist(), v.tolist())}
out['teammate_calibrations_operating'] = own

json.dump(out, open(sys.argv[1], 'w'), indent=1, default=float)
print('done; WavLM raw equals the calibration run raw:', wmatch)
