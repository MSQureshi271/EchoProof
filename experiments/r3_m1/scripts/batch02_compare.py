"""Verify the hidden64 run and compare it with all four preserved candidates."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch02_contract import ROOT, EXP, RUN_ID, ARCHITECTURE, read, verify_contract, validate_candidate_ids
from checkpoint_guard import check_run
from compare import metrics_from_files, summarize_history
from train import verify_epoch_files
from prepare_batch import sha


def choose(candidates):
    validate_candidate_ids(candidates)
    if any(not math.isfinite(c['macro_f1']) or not 0 <= c['macro_f1'] <= 1 for c in candidates):
        raise ValueError('invalid candidate metric')
    return sorted(candidates, key=lambda c: (-c['macro_f1'], c['id'] != 'original_lower_lr', c['id']))[0]


def verify_checkpoint_metadata(latest, best, resolved, contract, contract_hash):
    from echoproof.artifacts import digest
    if not latest['finished'] or latest['epoch'] != 20 or latest['stop_reason'] != 'max_epochs':
        raise ValueError('hidden64 run is incomplete')
    provenance = resolved['provenance']
    if (provenance.get('batch_contract_sha256') != contract_hash or provenance.get('architecture') != ARCHITECTURE
            or provenance.get('experiment') != RUN_ID):
        raise ValueError('checkpoint architecture/provenance differs from contract')
    config_hash = digest({'config': contract['configs'][RUN_ID], 'provenance': provenance})
    if resolved['config'] != contract['configs'][RUN_ID] or resolved['config_hash'] != config_hash:
        raise ValueError('resolved run configuration differs from contract')
    for checkpoint in (latest, best):
        if checkpoint['config'] != contract['configs'][RUN_ID] or checkpoint['config_hash'] != config_hash:
            raise ValueError('checkpoint configuration differs from contract')
        weights = checkpoint['model']
        if tuple(weights['bigru.gru.weight_ih_l0'].shape) != (192, 64) or tuple(weights['bigru.head.weight'].shape) != (2, 128):
            raise ValueError('checkpoint is not the declared hidden64 model')


def compare(contract_path):
    import pyarrow.parquet as pq
    import torch
    from echoproof.training.r3_m1_export import write_json
    contract_path = Path(contract_path)
    contract = read(contract_path)
    verify_contract(contract, contract_path)
    batch = EXP / 'runs/batch02'
    if (batch / 'active_attempt.json').exists():
        raise ValueError('active or unclosed training attempt')
    previous = read(EXP / 'reports/batch01_comparison.json')
    ref = ROOT / previous['selected']['checkpoint']
    expected_rows = pq.read_table(ref.parent / 'predictions_chunks.parquet').to_pylist()
    expected = {r['chunk_id']: (r['recording_id'], r['label']) for r in expected_rows}
    paths = [(c['id'], (ROOT / c['checkpoint']).parent) for c in previous['candidates']]
    paths.append((RUN_ID, batch / RUN_ID))
    candidates, artifacts = [], {}
    trajectory = None
    for name, directory in paths:
        boundary = check_run(directory)
        metrics, panels = metrics_from_files(directory / 'predictions_chunks.parquet', directory / 'predictions_recordings.parquet', expected)
        if metrics['macro_f1'] != boundary['best_macro_f1']:
            raise ValueError('saved predictions differ from best checkpoint')
        if name == RUN_ID:
            latest = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=True)
            best = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
            verify_checkpoint_metadata(latest, best, read(directory / 'resolved_config.json'), contract, sha(contract_path))
            history = [json.loads(s) for s in (directory / 'history.jsonl').read_text().splitlines()]
            verify_epoch_files(directory, 20)
            for row in history:
                prefix = directory / 'epoch_predictions' / f"epoch_{row['epoch']:02d}"
                actual, _ = metrics_from_files(Path(str(prefix)+'_chunks.parquet'), Path(str(prefix)+'_recordings.parquet'), expected)
                if actual['macro_f1'] != row['val']['macro_f1']:
                    raise ValueError('epoch predictions differ from history')
            trajectory = summarize_history(history)
        architecture = dict(ARCHITECTURE, hidden_size=64 if name == RUN_ID else 128)
        candidates.append({'id': name, 'macro_f1': metrics['macro_f1'], 'metrics': metrics, 'panels': panels,
                           'architecture': architecture, 'best_epoch': boundary['best_epoch'],
                           'checkpoint': (directory / 'best.pt').relative_to(ROOT).as_posix(),
                           'checkpoint_sha256': sha(directory / 'best.pt')})
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.pt', '.parquet'):
                artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    selected = choose(candidates)
    attempts = [json.loads(s) for s in (batch / 'attempts.jsonl').read_text().splitlines()]
    if any(a['experiment'] != RUN_ID for a in attempts) or not attempts or attempts[-1]['status'] != 'completed':
        raise ValueError('unexpected or incomplete attempt history')
    incumbent = next(c for c in candidates if c['id'] == previous['selected']['id'])
    challenger = next(c for c in candidates if c['id'] == RUN_ID)
    result = {'task': 'R3-M1 batch02 comparison', 'status': 'experimental_comparison_complete',
              'contract_sha256': sha(contract_path), 'candidates': candidates, 'experimental_preference': selected,
              'hidden64_trajectory': trajectory, 'gain_vs_batch01_selected': challenger['macro_f1']-incumbent['macro_f1'],
              'training_attempts': attempts, 'training_attempt_seconds': sum(a['elapsed_seconds'] for a in attempts),
              'artifact_sha256': artifacts, 'review_status': 'pending',
              'official_selection_changed': False, 'lock_changes': False,
              'limitations': 'Isolated post-lock, result-informed single-seed diagnostic. Advisory only; no automatic promotion, calibration/operating evaluation, statistical superiority, or unseen-generator guarantee.'}
    reports = EXP / 'reports/batch02'
    reports.mkdir(parents=True, exist_ok=True)
    decision = reports / 'experimental_preference.json'
    if decision.exists() and read(decision) != selected:
        raise ValueError('refusing to replace a different batch02 experimental preference')
    write_json(decision, selected)
    write_json(reports / 'comparison.json', result)
    lines = ['# Batch 02: hidden-size-64 comparison', '', '| Candidate | Hidden size | Best epoch | Full macro F1 | XTTS recall | Genuine FPR | XTTS AUC |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for c in candidates:
        lines.append(f"| {c['id']} | {c['architecture']['hidden_size']} | {c['best_epoch']} | {c['macro_f1']:.6f} | {c['panels']['unseen_xtts']['tts_recall']:.6f} | {c['metrics']['fpr']:.6f} | {c['panels']['unseen_xtts']['roc_auc']:.6f} |")
    lines += ['', f"Experimental preference: {selected['id']}. Advisory only; team review pending.", '', result['limitations'], '',
              'Choice follows full selection macro F1 at 0.5. XTTS diagnostics do not override that rule.', '',
              f"Hidden64 F1 difference from the batch01 selected checkpoint: {result['gain_vs_batch01_selected']:+.6f}."]
    (reports / 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(compare(args.contract)['experimental_preference'], indent=2))
