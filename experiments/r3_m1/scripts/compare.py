"""Verify all selection predictions and compare the declared BiGRU candidates."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import replay
from checkpoint_guard import check_run
from prepare_batch import sha
from train import EXPERIMENT, verify_contract, verify_epoch_files


def summarize_history(history):
    if [r['epoch'] for r in history] != list(range(1, 21)):
        raise ValueError('a complete 20-epoch history is required')
    scores = [r['val']['macro_f1'] for r in history]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
        raise ValueError('invalid selection scores')
    stopped = replay(history, patience=3)
    best = max(scores)
    return {'patience3': stopped, 'full20_best_epoch': scores.index(best)+1, 'full20_best_macro_f1': best,
            'gain_over_patience3': best - stopped['best_macro_f1'],
            'seconds': sum(r['seconds'] for r in history)}


def choose(candidates):
    if len(candidates) != 4 or len({c['id'] for c in candidates}) != 4:
        raise ValueError('four distinct candidates are required')
    if any(not math.isfinite(c['macro_f1']) or not 0 <= c['macro_f1'] <= 1 for c in candidates):
        raise ValueError('invalid candidate metric')
    return sorted(candidates, key=lambda c: (-c['macro_f1'], c['id'] != 'original_lower_lr', c['id']))[0]


def metrics_from_files(chunk_path, recording_path, expected):
    import pyarrow.parquet as pq
    from echoproof.evaluation.aggregate import aggregate_recordings
    from echoproof.evaluation.metrics import compute_metrics
    chunks = pq.read_table(chunk_path).to_pylist()
    records = pq.read_table(recording_path).to_pylist()
    identities = {r['chunk_id']: (r['recording_id'], r['label']) for r in chunks}
    if len(identities) != len(chunks) or identities != expected:
        raise ValueError('selection chunk identities differ from reference')
    agg = aggregate_recordings([r['chunk_id'] for r in chunks], [r['recording_id'] for r in chunks],
                               [r['label'] for r in chunks], [r['tts_probability'] for r in chunks])
    scores = dict(zip(agg.recording_ids, zip(agg.labels, agg.scores, agg.chunk_counts)))
    if len(records) != len(scores) or len({r['recording_id'] for r in records}) != len(records):
        raise ValueError('recording coverage mismatch')
    for row in records:
        label, score, count = scores[row['recording_id']]
        if (row['label'] != label or row['chunks'] != count or not math.isfinite(row['raw_score']) or
                abs(row['raw_score']-score) > 1e-12):
            raise ValueError('recording predictions differ from chunk means')
    metrics = compute_metrics(agg.labels, agg.scores, .5)
    panels = {}
    for name, components in [('known', {'genuine','melo','vits'}), ('unseen_xtts', {'genuine','xtts'})]:
        rows = [(y, s) for rid, (y, s, _) in scores.items() if rid.split(':')[1] in components]
        panels[name] = compute_metrics([r[0] for r in rows], [r[1] for r in rows], .5)
    return metrics, panels


def compare(contract_path):
    import pyarrow.parquet as pq
    import torch
    from echoproof.training.r3_m1_export import write_json
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    verify_contract(contract, contract_path)
    batch = EXPERIMENT / 'runs/batch01'
    if (batch / 'active_attempt.json').exists():
        raise ValueError('active or unclosed attempt; do not compare partial runs')
    reference = ROOT / 'runs/r3/r3_m1_20261107T144037Z'
    ref_chunks = pq.read_table(reference / (reference.name+'-lower_lr') / 'predictions_chunks.parquet').to_pylist()
    expected = {r['chunk_id']: (r['recording_id'], r['label']) for r in ref_chunks}
    candidates, trajectories, artifacts = [], {}, {}
    paths = [('original_'+name, reference / (reference.name+'-'+name)) for name in ('baseline','lower_lr')]
    paths += [(name, batch / name) for name in contract['configs']]
    for name, directory in paths:
        boundary = check_run(directory)
        history = [json.loads(s) for s in (directory/'history.jsonl').read_text().splitlines()]
        metrics, panels = metrics_from_files(directory/'predictions_chunks.parquet', directory/'predictions_recordings.parquet', expected)
        if metrics['macro_f1'] != boundary['best_macro_f1']:
            raise ValueError('best checkpoint metrics differ from saved predictions')
        if not name.startswith('original_'):
            latest = torch.load(directory/'latest.pt', map_location='cpu', weights_only=True)
            if not latest['finished'] or latest['epoch'] != 20 or latest['stop_reason'] != 'max_epochs':
                raise ValueError('new run is not complete')
            if latest['config'] != contract['configs'][name]:
                raise ValueError('checkpoint configuration differs from frozen contract')
            verify_epoch_files(directory, 20)
            for row in history:
                prefix = directory/'epoch_predictions'/f"epoch_{row['epoch']:02d}"
                actual, _ = metrics_from_files(Path(str(prefix)+'_chunks.parquet'), Path(str(prefix)+'_recordings.parquet'), expected)
                if actual['macro_f1'] != row['val']['macro_f1']:
                    raise ValueError('epoch history differs from saved scores')
            trajectories[name] = summarize_history(history)
            stop = trajectories[name]['patience3']
            if stop['stop_epoch'] is not None:
                snapshot = torch.load(directory/'patience3_best.pt', map_location='cpu', weights_only=True)
                if snapshot['epoch'] != stop['best_epoch'] or snapshot['metrics']['macro_f1'] != stop['best_macro_f1']:
                    raise ValueError('hypothetical early-stop checkpoint does not match history')
            original_name = 'baseline' if name == 'b01_baseline_full20' else 'lower_lr'
            old = [json.loads(s) for s in (reference/(reference.name+'-'+original_name)/'history.jsonl').read_text().splitlines()]
            trajectories[name]['original_prefix_max_abs_f1_difference'] = max(abs(a['val']['macro_f1']-b['val']['macro_f1']) for a,b in zip(history,old))
        candidates.append({'id': name, 'macro_f1': metrics['macro_f1'], 'metrics': metrics, 'panels': panels,
                           'best_epoch': boundary['best_epoch'], 'checkpoint': str((directory/'best.pt').relative_to(ROOT)),
                           'checkpoint_sha256': sha(directory/'best.pt')})
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.parquet','.json','.jsonl','.pt'):
                artifacts[str(path.relative_to(ROOT))] = sha(path)
    winner = choose(candidates)
    attempts = [json.loads(s) for s in (batch/'attempts.jsonl').read_text().splitlines()]
    result = {'task': 'R3-M1 batch01 comparison', 'status': 'comparison_complete', 'contract_sha256': sha(contract_path),
              'candidates': candidates, 'trajectories': trajectories, 'selected': winner, 'artifact_sha256': artifacts,
              'training_attempts': attempts, 'training_attempt_seconds': sum(a['elapsed_seconds'] for a in attempts),
              'review_status': 'pending', 'limitations': 'Result-informed single-seed selection; no statistical superiority or global optimum claimed.'}
    reports = EXPERIMENT/'reports'
    reports.mkdir(exist_ok=True)
    decision = reports/'final_selection.json'
    if decision.exists() and json.loads(decision.read_text()) != winner:
        raise ValueError('a different final decision already exists')
    write_json(decision, winner)
    write_json(reports/'batch01_comparison.json', result)
    lines = ['# Batch 01 comparison', '', '| Candidate | Best epoch | Macro F1 | XTTS F1 | FPR |', '|---|---:|---:|---:|---:|']
    for c in candidates:
        lines.append(f"| {c['id']} | {c['best_epoch']} | {c['macro_f1']:.6f} | {c['panels']['unseen_xtts']['macro_f1']:.6f} | {c['metrics']['fpr']:.6f} |")
    lines += ['', f"Selected: {winner['id']}. Independent review pending.", '', result['limitations']]
    for name, t in trajectories.items():
        lines += ['', f"{name}: hypothetical patience-3 stop {t['patience3']['stop_epoch']}; gain from full 20 epochs {t['gain_over_patience3']:.6f}."]
    (reports/'batch01_comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Self-contained SVG uses saved histories only; no plotting dependency needed.
    colors = ['#1765ad','#bd4520']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360" viewBox="0 0 760 360">', '<rect width="760" height="360" fill="white"/>', '<text x="50" y="22">Selection macro F1 at threshold 0.5 (y: 0 to 1; x: epoch 1 to 20)</text>', '<path d="M50 40 V300 H730" fill="none" stroke="black"/>']
    for i, name in enumerate(contract['configs']):
        history = [json.loads(s) for s in (batch/name/'history.jsonl').read_text().splitlines()]
        points = ' '.join(f"{50+(r['epoch']-1)*680/19:.2f},{300-r['val']['macro_f1']*260:.2f}" for r in history)
        svg += [f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="2"/>', f'<text x="50" y="{325+i*20}" fill="{colors[i]}">{name}</text>']
    svg.append('</svg>')
    (reports/'learning_curves.svg').write_text('\n'.join(svg), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(compare(a.contract)['selected'], indent=2))
