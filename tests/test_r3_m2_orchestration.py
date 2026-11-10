"""Keep scientific execution separate from the earlier M1 exception."""
import importlib.util
import json
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r3_m2_ablation_runner', ROOT/'scripts/r3_m2_ablation.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_m1_approval_cannot_authorize_m2_and_state_changes_are_rejected():
    approval = {'approved_by': 'Muneeb', 'approved': True, 'scope': 'r3_m2_fixed_feature_ablation_only',
                'plan_sha256': 'plan', 'lock_sha256_lf': {'locks/B_lock.json': 'lock'},
                'official_promotion_allowed': False}
    runner.validate_approval(approval, 'plan', {'locks/B_lock.json': 'lock'})
    for key, value in [('scope', 'R3-M1 batch02'), ('plan_sha256', 'other'),
                       ('official_promotion_allowed', True), ('lock_sha256_lf', {})]:
        with pytest.raises(ValueError, match='M1 approval does not apply'):
            runner.validate_approval(dict(approval, **{key: value}), 'plan', {'locks/B_lock.json': 'lock'})


def test_no_approval_stops_before_feature_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'EXP', tmp_path)
    with pytest.raises(ValueError, match='no authorization file'):
        runner.run(tmp_path/'nonexistent-features')
    assert not list(tmp_path.iterdir())


def test_declared_plan_matches_bounded_ablation():
    plan = json.loads((ROOT/'configs/r3_m2_ablation.json').read_text())
    assert plan['removed_features'] == list(runner.REMOVED_FEATURES)
    assert plan['source_feature_schema_hash'] == runner.FEATURE_SCHEMA_HASH
    assert plan['max_fits'] == 2 and plan['max_cpu_minutes'] == 60
    assert plan['official_promotion_allowed'] is False


def test_approval_plan_allows_line_endings_but_not_content_changes(tmp_path):
    original = b'{\r\n  "max_fits": 2\r\n}\r\n'
    approved = hashlib.sha256(original).hexdigest()
    path = tmp_path/'plan.json'
    path.write_bytes(original.replace(b'\r\n', b'\n'))
    assert runner.approval_plan_hash(path, approved) == approved
    path.write_bytes(b'{\n  "max_fits": 3\n}\n')
    assert runner.approval_plan_hash(path, approved) != approved


def test_lock_format_equivalence_does_not_allow_content_changes(tmp_path, monkeypatch):
    lock_dir = tmp_path/'locks'
    lock_dir.mkdir()
    path = lock_dir/'B_lock.json'
    lock = {'protocol': 'B', 'value': 1}
    canonical = json.dumps(lock, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    approved = {'locks/B_lock.json': hashlib.sha256(canonical).hexdigest()}
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    path.write_text(json.dumps(lock, indent=2), encoding='utf-8')
    assert runner.lock_snapshot(approved) == approved
    path.write_text(json.dumps(dict(lock, value=2), indent=2), encoding='utf-8')
    assert runner.lock_snapshot(approved) != approved
    (lock_dir/'C_lock.json').write_text('{}', encoding='utf-8')
    assert set(runner.lock_snapshot(approved)) != set(approved)


def test_empty_preflight_recovery_never_restarts_an_attempt(tmp_path):
    output = tmp_path/'runs/ablation01'
    ledger = tmp_path/'ledger.csv'
    output.mkdir(parents=True)
    runner.prepare_output(output, ledger)
    (output/'active_attempt.json').write_text('{}')
    with pytest.raises(FileExistsError, match='not empty'):
        runner.prepare_output(output, ledger)
    (output/'active_attempt.json').unlink()
    ledger.write_text('run_id,status\nr3_m2_full67_control,failed\n')
    with pytest.raises(FileExistsError, match='ledger'):
        runner.prepare_output(output, ledger)
