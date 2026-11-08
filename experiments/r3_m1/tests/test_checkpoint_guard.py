import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('checkpoint_guard', Path(__file__).resolve().parents[1] / 'scripts/checkpoint_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def state():
    latest = {'epoch': 2, 'best_epoch': 1, 'best_metric': .8, 'config_hash': 'hash'}
    best = {'epoch': 1, 'config_hash': 'hash', 'metrics': {'macro_f1': .8}}
    history = [{'epoch': 1, 'val': {'macro_f1': .8}}, {'epoch': 2, 'val': {'macro_f1': .7}}]
    return latest, best, history, [[1, 1], [1]]


def test_valid_boundary():
    assert guard.validate_state(*state())['best_epoch'] == 1


@pytest.mark.parametrize('damage', ['extra_history', 'best_ahead', 'predictions_ahead', 'different_config', 'missing_history', 'missing_predictions'])
def test_partial_epoch_is_refused(damage):
    latest, best, history, predictions = state()
    if damage == 'extra_history': history.append({'epoch': 3, 'val': {'macro_f1': .9}})
    if damage == 'best_ahead': best['epoch'] = 3
    if damage == 'predictions_ahead': predictions[0] = [3]
    if damage == 'different_config': best['config_hash'] = 'changed'
    if damage == 'missing_history': history.pop()
    if damage == 'missing_predictions': predictions[1] = []
    with pytest.raises(guard.GuardError):
        guard.validate_state(latest, best, history, predictions)
