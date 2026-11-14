"""GR3 (R3-ALL) production-selection computation under the research contract's rule.

Rank deployable Protocol B candidates by unseen-generator (xtts) selection-validation macro F1 at the raw
threshold 0.5, then lower EER, lower genuine FPR at 0.5, then lower warm p95 latency, then lexicographic ID.
Reads raw recording scores only (selection role). Writes a JSON receipt; does not write any lock.

Usage: python scripts/r3_all/gr3_selection.py --out reports/r3/gr3/selection_<stamp>.json
"""
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, 'src')
import numpy as np, pyarrow.parquet as pq
from echoproof.evaluation.metrics import compute_metrics

SRC = {
    'LR': 'runs/r3/r3_n1_lr/selected_lr_v1/export/predictions_recordings_selection.parquet',
    'BiGRU': 'experiments/r3_m1/runs/batch01/selected/predictions_recordings_selection.parquet',
    'AST': 'runs/r3/r3_u2_20261112T134049Z/predictions_recordings_selection.parquet',
    'WavLM': 'reports/r3/r3_u1_B/predictions_recordings_selection.parquet',
}
# warm p95 on the 10 s clip, ms, local laptop (docs/r3_n2_latency_local.md); None where not measured. Tie-break only.
P95_MS = {'LR': 117, 'BiGRU': 371, 'WavLM': 2410, 'AST': None}


def panel(rows, generators):
    keep = [r for r in rows if r['component'] in ('genuine', *generators)]
    y = [r['label'] for r in keep]; s = [r['raw_score'] for r in keep]
    d = compute_metrics(y, s, 0.5)
    return {'n': len(keep), 'macro_f1': d['macro_f1'], 'eer': d['eer'], 'fpr': d['fpr'], 'tts_recall': d['tts_recall'], 'roc_auc': d['roc_auc']}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = {'rule': 'unseen-xtts selection macro F1 at raw 0.5, then lower EER, lower FPR at 0.5, lower warm p95, lexicographic ID',
           'candidates': {}}
    for m, p in SRC.items():
        rows = pq.read_table(p).to_pylist()
        comps = sorted({r['component'] for r in rows})
        out['candidates'][m] = {'source': p, 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest(), 'components': comps, 'n': len(rows),
                                'unseen_xtts': panel(rows, ['xtts']), 'known_melo_vits': panel(rows, ['melo', 'vits']),
                                'p95_ms_10s_local': P95_MS[m]}
    def key(m):
        u = out['candidates'][m]['unseen_xtts']; l = out['candidates'][m]['p95_ms_10s_local']
        return (-u['macro_f1'], u['eer'], u['fpr'], l if l is not None else float('inf'), m)
    ranking = sorted(out['candidates'], key=key)
    out['ranking'] = ranking
    out['tie_at_top'] = len(ranking) > 1 and key(ranking[0])[:3] == key(ranking[1])[:3]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(a.out).exists(): raise SystemExit(f'refusing to overwrite {a.out}')
    Path(a.out).write_text(json.dumps(out, indent=2), encoding='utf-8', newline='\n')
    for i, m in enumerate(ranking, 1):
        u = out['candidates'][m]['unseen_xtts']; k = out['candidates'][m]['known_melo_vits']
        print(f"{i}. {m:6s} xtts F1 {u['macro_f1']:.4f} EER {u['eer']:.4f} FPR {u['fpr']:.4f} | known F1 {k['macro_f1']:.4f} | n {u['n']}")
    print('tie at top:', out['tie_at_top'])

main()
