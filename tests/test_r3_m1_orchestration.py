import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('r3_m1_train', Path(__file__).resolve().parents[1] / 'scripts/r3_m1_train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_contract_is_immutable_and_resume_requires_match(tmp_path):
    output = tmp_path / 'run'
    contract = {'split': 'frozen', 'learning_rates': module.EXPERIMENTS}
    module.freeze_contract(output, contract, False)
    module.freeze_contract(output, contract, True)
    with pytest.raises(ValueError, match='contract'):
        module.freeze_contract(output, {'split': 'changed'}, True)
    with pytest.raises(FileExistsError):
        module.freeze_contract(output, contract, False)


def test_resume_refuses_missing_contract(tmp_path):
    with pytest.raises(ValueError, match='contract'):
        module.freeze_contract(tmp_path / 'missing', {}, True)
