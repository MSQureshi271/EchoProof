import json

import pytest

from echoproof.training.r4_m1_bigru import check_history, freeze_contract


def test_contract_new_and_resume_guard(tmp_path):
    run = tmp_path/'run'
    frozen = {'protocol': 'A', 'recipe': 'hash', 'source': {'train': 'hash'}}
    freeze_contract(run, frozen, False)
    assert json.loads((run/'experiment_contract.json').read_text()) == frozen
    freeze_contract(run, frozen, True)
    with pytest.raises(ValueError, match='contract or provenance changed'):
        freeze_contract(run, {**frozen, 'recipe': 'changed'}, True)
    with pytest.raises(FileExistsError):
        freeze_contract(run, frozen, False)


def test_resume_rejects_inconsistent_history():
    history = [{'epoch': 1, 'improved': True}, {'epoch': 2, 'improved': False},
               {'epoch': 3, 'improved': True}]
    check_history(history, 3, 3)
    with pytest.raises(ValueError, match='history and latest'):
        check_history(history, 2, 1)
    with pytest.raises(ValueError, match='history and best'):
        check_history(history, 3, 1)


def test_selection_only_guard():
    from types import SimpleNamespace
    from echoproof.training.r3_m1_selection import SelectionAggregation
    with pytest.raises(ValueError, match='selection validation role'):
        SelectionAggregation(SimpleNamespace(validation_role='operating'))
