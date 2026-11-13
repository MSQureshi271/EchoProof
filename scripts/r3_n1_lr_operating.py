"""Evaluate the frozen LR temperature and choose a development operating rule."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from echoproof.artifacts import file_digest
from echoproof.evaluation.calibrate import apply_temperature
from echoproof.evaluation.metrics import compute_metrics
from echoproof.evaluation.threshold import select_threshold
from r3_n1_lr_fit import validate_calibration

CONTRACT = ROOT/'configs/r3_n1_lr_contract.json'


def reliability(labels, scores):
    bins = []
    for i in range(10):
        idx = [j for j, x in enumerate(scores) if min(int(x*10), 9) == i]
        bins.append({'bin': i, 'lower': i/10, 'upper': (i+1)/10, 'count': len(idx),
                     'mean_score': sum(scores[j] for j in idx)/len(idx) if idx else None,
                     'positive_fraction': sum(labels[j] for j in idx)/len(idx) if idx else None})
    ece = sum(r['count']*abs(r['mean_score']-r['positive_fraction']) for r in bins if r['count'])/len(labels)
    return bins, ece


def validate_operating(rows, contract):
    # Reuse identity/component/count checks without changing the input or permitting another role.
    if any(row.get('role') != 'operating' for row in rows):
        raise ValueError('non-operating role supplied to evaluation')
    validate_calibration([dict(row, role='calibration') for row in rows], contract['operating'])


def evaluate(export_dir, fit_dir, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    c = json.loads(CONTRACT.read_text(encoding='utf-8'))
    root = (ROOT/c['output_root']).resolve()
    export_dir, fit_dir, output = (Path(p).resolve() for p in (export_dir, fit_dir, output))
    if any(p == root or not p.is_relative_to(root) for p in (export_dir, fit_dir, output)):
        raise ValueError('inputs and output must be under the LR run root')
    if output.exists() or output in (export_dir, fit_dir):
        raise FileExistsError('refusing to overwrite an existing evaluation')
    evidence_path, fit_path = export_dir/'evidence.json', fit_dir/'temperature.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    if (evidence.get('status') != 'complete' or evidence.get('task') != 'R3-N1-LR prediction export'
            or evidence.get('contract_sha256') != file_digest(CONTRACT)
            or evidence.get('selected_bundle_sha256') != c['bundle_sha256']
            or fit.get('task') != 'R3-N1-LR temperature fit' or fit.get('status') != 'calibration_fit_complete'
            or fit.get('source_contract_sha256') != file_digest(CONTRACT)
            or fit.get('source_export_evidence_sha256') != file_digest(evidence_path)
            or fit.get('selected_bundle_sha256') != c['bundle_sha256'] or fit.get('fit_role') != 'calibration'
            or fit.get('method') != c['calibration_method'] or fit.get('epsilon') != c['epsilon']
            or fit.get('bounds') != c['temperature_bounds']
            or not c['temperature_bounds'][0] <= fit.get('temperature', 0) <= c['temperature_bounds'][1]):
        raise ValueError('export or fit provenance does not match the contract')
    if c['calibration_acceptance'] != 'operating_log_loss_strictly_improves_and_brier_does_not_worsen; otherwise retain_raw_model_score':
        raise ValueError('unsupported calibration acceptance rule')
    def read_role(role):
        name = f'predictions_recordings_{role}.parquet'
        path = export_dir/name
        actual = file_digest(path)
        if actual != evidence.get('outputs', {}).get(name):
            raise ValueError(f'{role} prediction checksum mismatch')
        return pq.read_table(path).to_pylist(), actual
    calibration, calibration_sha = read_role('calibration')
    if calibration_sha != fit.get('source_calibration_predictions_sha256'):
        raise ValueError('fit used different calibration predictions')
    validate_calibration(calibration, c['calibration'])
    rows, operating_sha = read_role('operating')
    validate_operating(rows, c)
    for key in ('recording_id', 'group_id'):
        if {r[key] for r in calibration} & {r[key] for r in rows}:
            raise ValueError(f'calibration and operating overlap by {key}')
    labels, raw = [r['label'] for r in rows], [r['raw_score'] for r in rows]
    temperature = apply_temperature(raw, fit['temperature'], epsilon=c['epsilon'])
    raw_metrics, temp_metrics = compute_metrics(labels, raw, .5), compute_metrics(labels, temperature, .5)
    accepted = temp_metrics['log_loss'] < raw_metrics['log_loss'] and temp_metrics['brier_score'] <= raw_metrics['brier_score']
    chosen = temperature if accepted else raw
    rule, macro_f1, fpr = select_threshold(labels, chosen)
    decisions = [rule.decide(x) for x in chosen]
    confusion = {'tp': sum(y == 1 and d == 1 for y, d in zip(labels, decisions)),
                 'fp': sum(y == 0 and d == 1 for y, d in zip(labels, decisions)),
                 'tn': sum(y == 0 and d == 0 for y, d in zip(labels, decisions)),
                 'fn': sum(y == 1 and d == 0 for y, d in zip(labels, decisions))}
    raw_bins, raw_ece = reliability(labels, raw)
    temp_bins, temp_ece = reliability(labels, temperature)
    output.mkdir(parents=True, exist_ok=False)
    prediction_rows = [{**row, 'temperature_score': score, 'chosen_score': value, 'decision': rule.decide(value)}
                       for row, score, value in zip(rows, temperature, chosen)]
    pq.write_table(pa.Table.from_pylist(prediction_rows), output/'operating_predictions.parquet')
    with (output/'reliability_bins.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['scores', 'bin', 'lower', 'upper', 'count', 'mean_score', 'positive_fraction'])
        writer.writeheader()
        for name, bins in [('raw', raw_bins), ('temperature', temp_bins)]:
            writer.writerows({'scores': name, **row} for row in bins)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], '--', label='ideal')
    for name, bins in [('raw', raw_bins), ('temperature', temp_bins)]:
        valid = [b for b in bins if b['count']]
        ax.plot([b['mean_score'] for b in valid], [b['positive_fraction'] for b in valid], marker='o', label=name)
    ax.set(xlabel='Mean score', ylabel='Observed TTS fraction', title='LR operating reliability', xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(output/'reliability.png'); plt.close(fig)
    report = {'task': 'R3-N1-LR operating evaluation', 'status': 'complete',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'source_contract_sha256': file_digest(CONTRACT),
              'source_export_evidence_sha256': file_digest(evidence_path), 'source_fit_sha256': file_digest(fit_path),
              'source_operating_predictions_sha256': operating_sha, 'temperature': fit['temperature'],
              'acceptance_rule': c['calibration_acceptance'], 'temperature_accepted': accepted,
              'score_type': 'calibrated_probability' if accepted else 'model_score',
              'raw_fixed_0_5': raw_metrics, 'temperature_fixed_0_5': temp_metrics,
              'raw_equal_width_ece': raw_ece, 'temperature_equal_width_ece': temp_ece,
              'operating_rule': {'kind': rule.kind, 'threshold': rule.threshold, 'macro_f1': macro_f1, 'fpr': fpr, **confusion},
              'selection_note': 'Threshold and acceptance evaluated on operating role; not independent test performance.',
              'fit_on_operating': False, 'model_retrained': False, 'official_lock_changed': False,
              'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in
                                (Path(__file__).resolve(), ROOT/'src/echoproof/evaluation/threshold.py')},
              'outputs': {p.name: file_digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'evidence.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export-dir', required=True, type=Path)
    p.add_argument('--fit-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(evaluate(args.export_dir, args.fit_dir, args.output), indent=2))
