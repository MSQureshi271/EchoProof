"""Create an exploratory report from the immutable initial experiment."""
import csv
import json
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[1]
RUN = EXP/'runs/initial'
report = json.loads((RUN/'evidence.json').read_text(encoding='utf-8'))
with (RUN/'calibration_oof.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
groups = sorted({r['group_id'] for r in rows})
rng = np.random.default_rng(42)
differences = []
for _ in range(1000):
    sample = rng.choice(groups, len(groups), replace=True)
    selected = [r for g in sample for r in rows if r['group_id'] == g]
    y = np.array([int(r['label']) for r in selected])
    def loss(method):
        p = np.clip(np.array([float(r[method]) for r in selected]), 1e-7, 1-1e-7)
        return np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))
    differences.append(loss('platt')-loss('temperature'))
folds = report['speaker_folds']
wins = sum(next(x['log_loss'] for x in folds if x['speaker']==g and x['method']=='platt') < next(x['log_loss'] for x in folds if x['speaker']==g and x['method']=='temperature') for g in groups)
lo, hi = np.quantile(differences, [.025, .975])
lines = ['# LR calibration: initial experiment', '', 'Exploratory only; no official promotion. The operating role was previously inspected.', '', '| Method | Speaker-held-out log loss | Brier |', '|---|---:|---:|']
for method, values in report['cv'].items(): lines.append(f"| {method} | {values['log_loss']:.6f} | {values['brier']:.6f} |")
lines += ['', f'Platt had lower log loss for {wins} of {len(groups)} held-out speakers. The 1,000-resample speaker bootstrap interval for the pooled log-loss difference (Platt minus temperature) was [{lo:.6f}, {hi:.6f}]. This interval is descriptive, based on only 12 speakers and fixed out-of-fold predictions; it does not include uncertainty from repeating the full fitting process.', '', f"Selected method: **{report['chosen_method']}**. Operating log loss: {report['operating_raw']['log_loss']:.6f} raw versus {report['operating_selected']['log_loss']:.6f} selected; Brier: {report['operating_raw']['brier']:.6f} versus {report['operating_selected']['brier']:.6f}.", '', 'Platt is monotonic here; the gain concerns scores, not ranking or the attainable threshold trade-off. Confidence may shift with deployment prevalence and unseen generators. Review before any adoption.']
(EXP/'reports/initial_results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print('\n'.join(lines))
