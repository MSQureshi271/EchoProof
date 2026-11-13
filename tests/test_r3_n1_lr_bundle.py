"""Recording-level calibration must survive a verified bundle reload."""
import json
import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, write_bundle
from echoproof.contracts import BundleSpecification, ContractError, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.evaluation.calibrate import apply_temperature
from echoproof.inference import load_lr_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('accepted', [True, False])
def test_bundle_reload_applies_only_the_accepted_recording_rule(tmp_path, accepted):
    model = ROOT/'runs/r2/r2_m1_20261106T140611Z/model'
    spec = BundleSpecification(1, 'bundle:lr-fixture', 'logistic_regression:B', PREPROCESSING_HASH,
                               DecisionRule(1, 'threshold', .4), 'genuine', 'tts')
    meta = {'schema': 'r3_n1_lr_calibration_v1', 'method': 'temperature_on_log_odds', 'temperature': 2.0,
            'epsilon': 1e-7, 'temperature_accepted': accepted,
            'score_type': 'calibrated_probability' if accepted else 'model_score'}
    path = tmp_path/'bundle'
    write_bundle(path, spec, {'model/weights.npz': model/'weights.npz', 'model/lr_model.json': model/'lr_model.json',
                              'calibration/lr_calibration.json': json.dumps(meta).encode()},
                 extra={'stage': 'r3_n1_lr_development'})
    loaded = load_lr_bundle(path)
    assert loaded.score_type == meta['score_type']
    actual = loaded.calibrate(.7) if loaded.calibrate else .7
    assert actual == pytest.approx(apply_temperature([.7], 2.0)[0] if accepted else .7)
    assert loaded.spec.decision_rule.decide(actual) == (actual >= .4)
    with pytest.raises(ContractError, match='override'):
        load_lr_bundle(path, threshold=.5)
    with (path/'calibration/lr_calibration.json').open('ab') as f:
        f.write(b'tampered')
    with pytest.raises(BundleError, match='Checksum'):
        load_lr_bundle(path)


def test_calibrated_stage_with_missing_score_rule_fails_closed(tmp_path):
    model = ROOT/'runs/r2/r2_m1_20261106T140611Z/model'
    spec = BundleSpecification(1, 'bundle:lr-missing-rule', 'logistic_regression:B', PREPROCESSING_HASH,
                               DecisionRule(1, 'all_negative', None), 'genuine', 'tts')
    path = tmp_path/'bundle'
    write_bundle(path, spec, {'model/weights.npz': model/'weights.npz', 'model/lr_model.json': model/'lr_model.json'},
                 extra={'stage': 'r3_n1_lr_development'})
    with pytest.raises(ContractError, match='missing'):
        load_lr_bundle(path)


def test_handoff_packages_and_reloads_recording_scores(tmp_path, monkeypatch):
    script = ROOT/'scripts/r3_n1_lr_bundle.py'
    loader = importlib.util.spec_from_file_location('r3_n1_lr_package', script)
    runner = importlib.util.module_from_spec(loader); loader.loader.exec_module(runner)
    model = ROOT/'runs/r2/r2_m1_20261106T140611Z/model'
    source = tmp_path/'original'
    spec = BundleSpecification(1, 'bundle:source', 'logistic_regression:B', PREPROCESSING_HASH,
                               DecisionRule(1, 'threshold', .5), 'genuine', 'tts')
    write_bundle(source, spec, {'model/weights.npz': model/'weights.npz', 'model/lr_model.json': model/'lr_model.json'})
    contract = {'output_root': 'runs/r3/r3_n1_lr', 'bundle': 'original', 'bundle_sha256': file_digest(source/'bundle.json'),
                'model_components_sha256': {n: file_digest(source/'model'/n) for n in ('weights.npz', 'lr_model.json')},
                'calibration_acceptance': 'rule', 'operating': {'recordings': 2}}
    cp = tmp_path/'contract.json'; cp.write_text(json.dumps(contract))
    root = tmp_path/'runs/r3/r3_n1_lr'; fitdir = root/'fit'; evdir = root/'eval'
    fitdir.mkdir(parents=True); evdir.mkdir()
    fit = {'source_contract_sha256': file_digest(cp), 'selected_bundle_sha256': contract['bundle_sha256'],
           'status': 'calibration_fit_complete', 'temperature': 2.0, 'epsilon': 1e-7}
    fp = fitdir/'temperature.json'; fp.write_text(json.dumps(fit))
    raw = [.2, .8]; chosen = apply_temperature(raw, 2.0)
    pp = evdir/'operating_predictions.parquet'
    pq.write_table(pa.Table.from_pylist([{'recording_id': f'r{i}', 'raw_score': v, 'chosen_score': chosen[i],
                                          'decision': int(chosen[i] >= .5)} for i, v in enumerate(raw)]), pp)
    ev = {'task': 'R3-N1-LR operating evaluation', 'status': 'complete', 'source_contract_sha256': file_digest(cp),
          'source_fit_sha256': file_digest(fp), 'temperature': 2.0, 'acceptance_rule': 'rule',
          'temperature_accepted': True, 'score_type': 'calibrated_probability',
          'temperature_fixed_0_5': {'log_loss': .2, 'brier_score': .1},
          'raw_fixed_0_5': {'log_loss': .3, 'brier_score': .2},
          'outputs': {'operating_predictions.parquet': file_digest(pp)},
          'operating_rule': {'kind': 'threshold', 'threshold': .5}}
    (evdir/'evidence.json').write_text(json.dumps(ev))
    monkeypatch.setattr(runner, 'ROOT', tmp_path); monkeypatch.setattr(runner, 'CONTRACT', cp)
    result = runner.package(evdir, fitdir, root/'handoff')
    assert result['recording_score_max_abs'] == 0 and result['decision_flips'] == 0
    assert load_lr_bundle(root/'handoff/bundle').score_type == 'calibrated_probability'
