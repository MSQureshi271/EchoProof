import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('prepare_batch', Path(__file__).resolve().parents[1] / 'scripts/prepare_batch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_must_be_complete():
    with pytest.raises(ValueError, match='complete passed'):
        module.validate_audit({'task': 'R3-M1 audit', 'overall': 'passed', 'checks': {}})


def test_firewall_rejects_recorded_a_and_b_lock(tmp_path):
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'reports/ledger.csv').write_text('run_id,protocol\nx,B\n', encoding='utf-8')
    assert module.verify_firewall(tmp_path)['protocol_a_ledger_rows'] == 0
    (tmp_path / 'reports/ledger.csv').write_text('run_id,protocol\nx,A\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Protocol A'):
        module.verify_firewall(tmp_path)
    (tmp_path / 'locks').mkdir()
    (tmp_path / 'locks/B_lock.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='lock'):
        module.verify_firewall(tmp_path)
