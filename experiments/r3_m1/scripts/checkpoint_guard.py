"""Fail closed when trainer checkpoint artifacts disagree at a resume boundary.

This module never repairs or deletes artifacts. The experiment runner must call
``check_run`` immediately before resuming and refuse resume on GuardError.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


class GuardError(RuntimeError):
    """A checkpoint set cannot be safely resumed without investigation."""


def validate_state(latest, best, history, prediction_epochs):
    epoch = latest.get('epoch')
    if type(epoch) is not int or epoch < 1 or len(history) != epoch:
        raise GuardError('history length does not match latest checkpoint epoch')
    if [row.get('epoch') for row in history] != list(range(1, epoch + 1)):
        raise GuardError('history epochs are missing, duplicated or out of order')
    best_epoch = latest.get('best_epoch')
    if type(best_epoch) is not int or not 1 <= best_epoch <= epoch:
        raise GuardError('latest checkpoint has invalid best epoch')
    if latest.get('config_hash') != best.get('config_hash') or not latest.get('config_hash'):
        raise GuardError('best and latest checkpoint configuration hashes differ')
    if best.get('epoch') != best_epoch:
        raise GuardError('best checkpoint does not match latest best epoch')
    try:
        scores = [row['val']['macro_f1'] for row in history]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores):
            raise ValueError('invalid score')
        high = max(scores)
        first_best = scores.index(high) + 1
        recorded = best['metrics']['macro_f1']
    except (KeyError, TypeError, ValueError) as exc:
        raise GuardError('invalid history or checkpoint metrics') from exc
    if best_epoch != first_best or recorded != high or latest.get('best_metric') != high:
        raise GuardError('best metrics disagree with history or earliest-tie rule')
    if len(prediction_epochs) != 2 or any(not epochs or set(epochs) != {best_epoch} for epochs in prediction_epochs):
        raise GuardError('prediction files do not match best checkpoint epoch')
    return {'latest_epoch': epoch, 'best_epoch': best_epoch, 'best_macro_f1': high}


def check_run(run_dir):
    import pyarrow.parquet as pq
    import torch
    run_dir = Path(run_dir)
    try:
        latest = torch.load(run_dir / 'latest.pt', map_location='cpu', weights_only=True)
        best = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
        history = [json.loads(line) for line in (run_dir / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
        prediction_epochs = [pq.read_table(run_dir / f'predictions_{name}.parquet', columns=['epoch']).column('epoch').to_pylist()
                             for name in ('chunks', 'recordings')]
        return validate_state(latest, best, history, prediction_epochs)
    except GuardError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise GuardError(f'cannot verify checkpoint artifacts: {exc}') from exc
