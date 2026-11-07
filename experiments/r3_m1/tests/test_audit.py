import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'scripts/audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def history(scores):
    best = -1
    rows = []
    for index, value in enumerate(scores, 1):
        improved = value > best
        best = max(best, value)
        rows.append({'epoch': index, 'improved': improved, 'val': {'macro_f1': value}})
    return rows


def test_replay_stops_after_three_without_a_new_best():
    result = audit.replay(history([.5, .6, .55, .58, .59, .8]))
    assert result == {'best_epoch': 2, 'best_macro_f1': .6, 'stop_epoch': 5}


def test_replay_handles_ties_and_rejects_inconsistent_history():
    assert audit.replay(history([.5, .5, .5, .5]))['stop_epoch'] == 4
    rows = history([.5, .6])
    rows[1]['improved'] = False
    with pytest.raises(ValueError, match='improvement flag'):
        audit.replay(rows)


def test_coverage_rejects_duplicate_training_recordings():
    with pytest.raises(ValueError, match='duplicate'):
        audit.validate_coverage([('phonemedf:genuine:1-2-3', 0)] * 2, [])
