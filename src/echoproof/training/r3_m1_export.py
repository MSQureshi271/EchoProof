"""Freeze R3-M1 selection, then export raw validation predictions."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from echoproof.evaluation.aggregate import aggregate_recordings
from echoproof.training.r3_m1_data import file_sha256
from echoproof.training.r3_m1_selection import selection_metrics, selection_panels


def choose_winner(metrics):
    if set(metrics) != {'baseline', 'lower_lr'}:
        raise ValueError('both experiments are required')
    scores = {name: float(value) for name, value in metrics.items()}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError('invalid selection score')
    return 'lower_lr' if scores['lower_lr'] > scores['baseline'] else 'baseline'


def aggregate_role(dataset, rows):
    expected = {c.chunk_id: (c.recording_id, int(dataset.labels_by_recording[c.recording_id])) for c in dataset.chunks}
    if not expected or len(expected) != len(dataset.chunks):
        raise ValueError('empty or duplicate expected chunks')
    actual = {}
    for cid, rid, label, score in rows:
        if cid in actual or expected.get(cid) != (rid, label):
            raise ValueError('duplicate, unexpected or mislabelled chunk')
        actual[cid] = score
    if set(actual) != set(expected):
        raise ValueError('chunk coverage mismatch')
    agg = aggregate_recordings([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                               [r[3] for r in rows], expected_recording_ids={x[0] for x in expected.values()})
    return list(zip(agg.recording_ids, agg.labels, agg.scores, agg.chunk_counts))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def score_dataset(model, dataset, device, batch_size=64):
    import torch
    from torch.utils.data import DataLoader
    from echoproof.training.datasets import collate
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch['model_inputs'].items()}
            scores = torch.softmax(model(**inputs).float(), dim=1)[:, 1].cpu().tolist()
            rows.extend(zip(batch['chunk_ids'], batch['recording_ids'], batch['labels'].tolist(), scores))
    return rows


def finish(output, data, summaries):
    """Select using checkpoint metadata alone, freeze it, then score all validation roles."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from echoproof.models.bigru_waveform import WaveformBiGRU

    output = Path(output)
    if set(summaries) != {'baseline', 'lower_lr'}:
        raise ValueError('both completed runs are required')
    checkpoints, metrics, hashes = {}, {}, {}
    for name, summary in summaries.items():
        path = Path(summary['best_checkpoint'])
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if (checkpoint['config_hash'] != summary['config_hash'] or checkpoint['epoch'] != summary['best_epoch'] or
                checkpoint['metrics']['macro_f1'] != summary['best_macro_f1'] or
                summary['stop_reason'] not in {'early_stopping', 'max_epochs'}):
            raise ValueError(f'inconsistent or unfinished checkpoint: {name}')
        checkpoints[name], metrics[name], hashes[name] = checkpoint, checkpoint['metrics']['macro_f1'], file_sha256(path)
    winner = choose_winner(metrics)
    selection = {'task': 'R3-M1', 'protocol': 'B', 'selected': winner, 'selection_threshold': 0.5,
                 'selection_macro_f1': metrics, 'checkpoint_sha256': hashes, 'selected_epoch': summaries[winner]['best_epoch'],
                 'split_sha256': data['split_sha256'], 'tie_rule': 'baseline',
                 'contract_sha256': file_sha256(output / 'experiment_contract.json')}
    selection_path = output / 'selection.json'
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding='utf-8')) != selection:
            raise ValueError('existing selection differs; refusing to overwrite')
    else:
        write_json(selection_path, selection)
    # The winner is recorded before any calibration or operating audio is scored.
    model = WaveformBiGRU(data['mel_stats']['mean'], data['mel_stats']['std'], hidden_size=128, dropout=0.3)
    model.load_state_dict(checkpoints[winner]['model'], strict=True)
    model.to('cuda')
    products, role_counts, selection_panel_metrics = {}, {}, None
    for role in ('selection', 'calibration', 'operating'):
        dataset = data['validation'][role]
        rows = score_dataset(model, dataset, 'cuda')
        recordings = aggregate_role(dataset, rows)
        if role == 'selection':
            actual = selection_metrics([r[1] for r in recordings], [r[2] for r in recordings], 0.5)['macro_f1']
            if abs(actual - metrics[winner]) > 1e-12:
                raise ValueError('reloaded checkpoint selection metric does not match saved best')
            selection_panel_metrics = selection_panels(recordings, data['validation_components'])
        groups, components = data['validation_groups'], data['validation_components']
        chunk_table = pa.Table.from_pylist([{'chunk_id': cid, 'recording_id': rid, 'group_id': groups[rid],
                                             'label': label, 'component': components[rid], 'validation_role': role,
                                             'tts_probability': score} for cid, rid, label, score in rows])
        rec_table = pa.Table.from_pylist([{'recording_id': rid, 'group_id': groups[rid], 'label': label,
                                           'component': components[rid], 'validation_role': role, 'raw_score': score,
                                           'chunk_count': n} for rid, label, score, n in recordings])
        for kind, table in [('chunks', chunk_table), ('recordings', rec_table)]:
            path = output / f'predictions_{kind}_{role}.parquet'
            tmp = path.with_suffix('.parquet.tmp')
            pq.write_table(table, tmp)
            os.replace(tmp, path)
            products[path.name] = file_sha256(path)
        role_counts[role] = {'chunks': len(rows), 'recordings': len(recordings)}
    evidence = {'task': 'R3-M1', 'status': 'validation_predictions_complete', 'selection': selection,
                'selection_sha256': file_sha256(selection_path), 'input_sha256': data['input_sha256'],
                'output_sha256': products, 'counts': role_counts, 'selection_panels': selection_panel_metrics,
                'score_type': 'raw uncalibrated TTS score', 'note': 'Calibration and operating predictions are not selection metrics.'}
    write_json(output / 'evidence.json', evidence)
    return evidence
