"""Contract, advisory selection and checkpoint checks that do not need CUDA."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'scripts'))
sys.path.insert(0, str(EXP.parents[1] / 'src'))
import batch02_contract as contract
import batch02_compare as comparison
from echoproof.artifacts import digest


def test_matched_design_rejects_extra_training_changes():
    plan = contract.read(EXP / 'configs/batch02_plan.json')
    prior = contract.read(EXP / 'configs/batch01_resolved.json')
    assert contract.validate_design(plan, prior)['lr'] == .0003
    for key, value in [('lr', .001), ('seed', 7), ('dropout', .5)]:
        altered = copy.deepcopy(plan)
        altered['training_config'][key] = value
        with pytest.raises(ValueError, match='configuration'):
            contract.validate_design(altered, prior)
    altered = copy.deepcopy(plan)
    altered['architecture']['num_layers'] = 3
    with pytest.raises(ValueError, match='architecture'):
        contract.validate_design(altered, prior)


def test_selection_retains_declared_tie_rule_and_requires_all_candidates():
    candidates = [{'id': name, 'macro_f1': .8} for name in sorted(contract.PRIOR_IDS | {contract.RUN_ID})]
    assert comparison.choose(candidates)['id'] == 'original_lower_lr'
    new = next(c for c in candidates if c['id'] == contract.RUN_ID)
    new['macro_f1'] = .9
    assert comparison.choose(candidates)['id'] == contract.RUN_ID
    new['macro_f1'] = .7
    assert comparison.choose(candidates)['id'] != contract.RUN_ID
    with pytest.raises(ValueError, match='five|four prior'):
        comparison.choose(candidates[:-1])
    new['macro_f1'] = float('nan')
    with pytest.raises(ValueError, match='metric'):
        comparison.choose(candidates)


def test_approval_is_specific_to_plan_and_lock_state(tmp_path, monkeypatch):
    exp = tmp_path / 'experiments/r3_m1'
    (exp / 'configs').mkdir(parents=True)
    (tmp_path / 'locks').mkdir()
    (exp / 'configs/batch02_plan.json').write_text('{}')
    lock = tmp_path / 'locks/B_lock.json'
    lock.write_text('{}')
    monkeypatch.setattr(contract, 'EXP', exp)
    monkeypatch.setattr(contract, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='explicit Muneeb approval'):
        contract.verify_authorization()
    approval = {'approved': True, 'approved_by': 'Muneeb', 'restrictions': contract.RESTRICTIONS,
                'plan_sha256': contract.sha(exp / 'configs/batch02_plan.json'), 'lock_sha256': contract.lock_snapshot()}
    (exp / 'configs/batch02_authorization.json').write_text(json.dumps(approval))
    contract.verify_authorization()
    lock.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='ask Muneeb again'):
        contract.verify_authorization()


@pytest.mark.parametrize('plan_hash,locks,expected_message', [
    ('approved-plan', {}, 'approved lock missing: locks/B_lock.json'),
    ('approved-plan', {'locks/B_lock.json': 'changed'}, 'lock hash differs: locks/B_lock.json'),
    ('different-plan', {'locks/B_lock.json': 'approved-lock'}, 'plan hash differs:'),
    ('approved-plan', {'locks/B_lock.json': 'approved-lock', 'locks/A_lock.json': 'new'}, 'unapproved additional lock: locks/A_lock.json'),
])
def test_authorization_diagnostics_identify_exact_mismatch(plan_hash, locks, expected_message):
    approval = {'approved_by': 'Muneeb', 'approved': True, 'plan_sha256': 'approved-plan',
                'lock_sha256': {'locks/B_lock.json': 'approved-lock'}, 'restrictions': contract.RESTRICTIONS}
    assert contract.authorization_differences(approval, 'approved-plan', approval['lock_sha256']) == []
    differences = contract.authorization_differences(approval, plan_hash, locks)
    assert len(differences) == 1 and expected_message in differences[0]


def test_approved_lock_accepts_only_line_ending_transport_changes(tmp_path, monkeypatch):
    exp = tmp_path / 'experiments/r3_m1'
    (exp / 'configs').mkdir(parents=True)
    (tmp_path / 'locks').mkdir()
    (exp / 'configs/batch02_plan.json').write_bytes(b'{}')
    lock = tmp_path / 'locks/B_lock.json'
    lock.write_bytes(b'{"lock_id":"approved"}\r\n')
    monkeypatch.setattr(contract, 'ROOT', tmp_path)
    monkeypatch.setattr(contract, 'EXP', exp)
    approval = {'approved_by': 'Muneeb', 'approved': True, 'restrictions': contract.RESTRICTIONS,
                'plan_sha256': contract.sha(exp / 'configs/batch02_plan.json'),
                'lock_sha256': contract.lock_snapshot()}
    approval_path = exp / 'configs/batch02_authorization.json'
    approval_path.write_text(json.dumps(approval))
    approval_before = approval_path.read_bytes()
    lock.write_bytes(b'{"lock_id":"approved"}\n')
    runtime_hash = contract.sha(lock)
    assert runtime_hash != approval['lock_sha256']['locks/B_lock.json']
    contract.verify_authorization()
    assert contract.lock_snapshot()['locks/B_lock.json'] == runtime_hash
    assert approval_path.read_bytes() == approval_before
    # Even semantically equivalent JSON with different spaces is not exempted.
    for changed in (b'{"lock_id": "approved"}\n', b'{"lock_id":"new"}\n'):
        lock.write_bytes(changed)
        with pytest.raises(ValueError, match='lock hash differs'):
            contract.verify_authorization()
    lock.write_bytes(b'{"lock_id":"approved"}\n')
    (tmp_path / 'locks/A_lock.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError, match='unapproved additional lock'):
        contract.verify_authorization()


def checkpoint_fixture():
    cfg = contract.read(EXP / 'configs/batch02_plan.json')['training_config']
    provenance = {'batch_contract_sha256': 'contract-hash', 'architecture': contract.ARCHITECTURE, 'experiment': contract.RUN_ID}
    config_hash = digest({'config': cfg, 'provenance': provenance})
    resolved = {'config': cfg, 'config_hash': config_hash, 'provenance': provenance}
    checkpoint = {'finished': True, 'epoch': 20, 'stop_reason': 'max_epochs', 'config': cfg, 'config_hash': config_hash,
                  'model': {'bigru.gru.weight_ih_l0': np.zeros((192, 64)), 'bigru.head.weight': np.zeros((2, 128))}}
    return checkpoint, copy.deepcopy(checkpoint), resolved, {'configs': {contract.RUN_ID: cfg}}


def test_protected_snapshot_records_absent_bundles_and_detects_creation(tmp_path, monkeypatch):
    (tmp_path / 'locks').mkdir()
    (tmp_path / 'locks/B_lock.json').write_text(json.dumps({'bundle': {'bundle_path': 'bundles/lr'}}))
    monkeypatch.setattr(contract, 'ROOT', tmp_path)
    before = contract.protected_artifacts()
    assert before['bundles/lr/'] is None
    bundle = tmp_path / 'bundles/lr'
    bundle.mkdir(parents=True)
    (bundle / 'weights.npz').write_bytes(b'fixture')
    assert contract.protected_artifacts() != before


def test_checkpoint_uses_real_trainer_schema_and_rejects_wrong_width():
    latest, best, resolved, cfg = checkpoint_fixture()
    # Shared Trainer stores provenance in resolved_config.json, not in .pt files.
    comparison.verify_checkpoint_metadata(latest, best, resolved, cfg, 'contract-hash')
    best['model']['bigru.head.weight'] = np.zeros((2, 256))
    with pytest.raises(ValueError, match='hidden64'):
        comparison.verify_checkpoint_metadata(latest, best, resolved, cfg, 'contract-hash')


def test_incomplete_or_unrelated_checkpoint_cannot_enter_comparison():
    latest, best, resolved, cfg = checkpoint_fixture()
    latest['epoch'] = 19
    with pytest.raises(ValueError, match='incomplete'):
        comparison.verify_checkpoint_metadata(latest, best, resolved, cfg, 'contract-hash')
    latest['epoch'] = 20
    best['config_hash'] = 'another-run'
    with pytest.raises(ValueError, match='configuration'):
        comparison.verify_checkpoint_metadata(latest, best, resolved, cfg, 'contract-hash')
