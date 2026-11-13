"""Analyze one completed LR calibration experiment without modifying its evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from echoproof.evaluation.calibrate import apply_temperature
from echoproof.evaluation.threshold import select_threshold


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run):
    run = run.resolve()
    if (EXP / 'runs').resolve() not in run.parents:
        raise ValueError('run must be inside the experiment runs directory')
    ev = json.loads((run / 'evidence.json').read_text(encoding='utf-8'))
    for name, expected in ev['outputs'].items():
        if sha(run / name) != expected:
            raise ValueError(f'run artifact hash mismatch: {name}')
    for name, expected in [('script_sha256', EXP/'scripts/run.py'), ('plan_sha256', EXP/'reports/plan.md')]:
        if ev[name] != sha(expected):
            raise ValueError(f'experiment provenance mismatch: {name}')
    export = ROOT/'runs/r3/r3_n1_lr/selected_lr_v1/export'
    if sha(export/'evidence.json') != ev['source_export_evidence_sha256']:
        raise ValueError('source export evidence mismatch')
    temp_path = ROOT/'runs/r3/r3_n1_lr/selected_lr_v1/fit/temperature.json'
    temperature = json.loads(temp_path.read_text(encoding='utf-8'))
    if temperature['source_calibration_predictions_sha256'] != ev['source_calibration_sha256']:
        raise ValueError('temperature calibration predictions mismatch')
    with (run/'operating_predictions.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r['label']) for r in rows]); raw = np.array([float(r['raw']) for r in rows])
    selected = np.array([float(r['selected']) for r in rows])
    if len(y) != 1040 or not np.isfinite(raw).all() or not np.isfinite(selected).all():
        raise ValueError('invalid operating predictions')
    methods = {'raw': raw, 'temperature': np.array(apply_temperature(raw.tolist(), temperature['temperature'])), ev['chosen_method']: selected}
    if ev['chosen_method'] != 'platt':
        raise ValueError('this initial analysis expects the selected Platt run')
    bins, threshold_rows = [], []
    for name, scores in methods.items():
        for i in range(10):
            mask = np.minimum((scores * 10).astype(int), 9) == i
            bins.append({'method': name, 'bin': i, 'count': int(mask.sum()), 'mean_score': float(scores[mask].mean()) if mask.any() else None, 'positive_fraction': float(y[mask].mean()) if mask.any() else None})
        rule, f1, fpr = select_threshold(y.tolist(), scores.tolist())
        for kind in ('fixed_0.5', 'operating_selected'):
            predictions = scores >= .5 if kind == 'fixed_0.5' else np.array([rule.decide(float(s)) for s in scores], dtype=bool)
            tp = int(((y == 1) & predictions).sum()); fp = int(((y == 0) & predictions).sum())
            tn = int(((y == 0) & ~predictions).sum()); fn = int(((y == 1) & ~predictions).sum())
            macro = ((2*tp/(2*tp+fp+fn))+(2*tn/(2*tn+fp+fn)))/2
            threshold_rows.append({'method': name, 'rule': kind, 'threshold': .5 if kind == 'fixed_0.5' else rule.threshold, 'macro_f1': macro, 'fpr': fp/(fp+tn), 'tts_recall': tp/(tp+fn), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})
    out = run/'analysis'
    out.mkdir(exist_ok=False)
    for filename, data in [('reliability_bins.csv', bins), ('threshold_comparison.csv', threshold_rows)]:
        with (out/filename).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)
    colors = {'raw': '#c2410c', 'temperature': '#2563eb', 'platt': '#15803d'}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="650" height="560" viewBox="0 0 650 560">', '<rect width="650" height="560" fill="white"/>', '<text x="70" y="25" font-family="sans-serif" font-size="18">Operating reliability (10 equal-width bins)</text>', '<path d="M 70 470 L 470 470 L 470 70" fill="none" stroke="black"/>', '<path d="M 70 470 L 470 70" stroke="#777" stroke-dasharray="5 5"/>']
    for name in methods:
        points = [(70+400*b['mean_score'], 470-400*b['positive_fraction']) for b in bins if b['method'] == name and b['count']]
        svg.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(colors[name], ' '.join(f'{x:.2f},{z:.2f}' for x,z in points)))
        svg.extend(f'<circle cx="{x:.2f}" cy="{z:.2f}" r="3" fill="{colors[name]}"/>' for x,z in points)
    for i, name in enumerate(methods): svg.append(f'<text x="490" y="{100+i*25}" fill="{colors[name]}" font-family="sans-serif">{name}</text>')
    svg += ['<text x="200" y="515" font-family="sans-serif">Mean predicted TTS probability</text>', '<text transform="translate(20 420) rotate(-90)" font-family="sans-serif">Observed TTS fraction</text>', '<text x="65" y="490" font-family="sans-serif">0</text><text x="460" y="490" font-family="sans-serif">1</text>', '<text x="50" y="475" font-family="sans-serif">0</text><text x="50" y="75" font-family="sans-serif">1</text>', '</svg>']
    (out/'operating_reliability.svg').write_text('\n'.join(svg), encoding='utf-8')
    report = {'source_evidence_sha256': sha(run/'evidence.json'), 'source_temperature_sha256': sha(temp_path), 'analysis_script_sha256': sha(Path(__file__)), 'thresholds': threshold_rows, 'ece': {name: sum(b['count']*abs(b['mean_score']-b['positive_fraction']) for b in bins if b['method']==name and b['count'])/len(y) for name in methods}, 'limitations': 'Operating thresholds are selected and evaluated on the same previously inspected role; exploratory only.', 'outputs': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'evidence.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run)
