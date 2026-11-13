"""Check role isolation, frozen provenance and persistence of LR calibration."""
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from echoproof.artifacts import file_digest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r3_n1_lr_fit', ROOT/'scripts/r3_n1_lr_fit.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_calibration_fit_requires_only_calibration_and_refuses_overwrite(tmp_path, monkeypatch):
    contract = {'output_root': 'runs/r3/r3_n1_lr', 'bundle_sha256': 'bundle', 'split_sha256': 'split',
                'feature_manifest_sha256': 'features', 'calibration': {'recordings': 4, 'groups': 2, 'chunks': 4},
                'epsilon': 1e-7, 'temperature_bounds': [.01, 50.0], 'calibration_method': 'temperature_on_log_odds'}
    contract_path = tmp_path/'contract.json'
    contract_path.write_text(json.dumps(contract))
    export = tmp_path/'runs/r3/r3_n1_lr/export'
    export.mkdir(parents=True)
    rows = [{'recording_id': f'r{i}', 'group_id': f'g{i//2}', 'role': 'calibration',
             'component': 'genuine' if i % 2 == 0 else 'xtts', 'label': i % 2,
             'raw_score': [.05, .8, .35, .6][i], 'chunk_count': 1} for i in range(4)]
    path = export/runner.CALIBRATION_FILE
    pq.write_table(pa.Table.from_pylist(rows), path)
    evidence = {'task': 'R3-N1-LR prediction export', 'status': 'complete',
                'contract_sha256': file_digest(contract_path), 'selected_bundle_sha256': 'bundle',
                'split_sha256': 'split', 'feature_manifest_sha256': 'features',
                'outputs': {runner.CALIBRATION_FILE: file_digest(path)}}
    (export/'evidence.json').write_text(json.dumps(evidence))
    # No operating file is present; the calibration stage must still succeed.
    monkeypatch.setattr(runner, 'CONTRACT', contract_path)
    # ROOT is needed for the real source hashes; use a run root under the real workspace.
    original_root = runner.ROOT
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    # Source files are project files; make temporary source identities explicit.
    for source in ('scripts/r3_n1_lr_fit.py', 'src/echoproof/evaluation/calibrate.py', 'src/echoproof/evaluation/metrics.py'):
        target = tmp_path/source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('fixture')
    monkeypatch.setattr(runner, '__file__', str(tmp_path/'scripts/r3_n1_lr_fit.py'))
    output = tmp_path/'runs/r3/r3_n1_lr/fit'
    result = runner.fit(export, output)
    assert result['temperature'] > 0 and result['operating_predictions_read'] is False
    assert json.loads((output/'temperature.json').read_text()) == result
    with pytest.raises(FileExistsError):
        runner.fit(export, output)
    assert original_root == ROOT


def test_calibration_rejects_wrong_role_and_group_count():
    rows = [{'recording_id': 'r1', 'group_id': 'g1', 'role': 'operating',
             'component': 'genuine', 'label': 0, 'chunk_count': 1}]
    with pytest.raises(ValueError, match='non-calibration'):
        runner.validate_calibration(rows, {'recordings': 1, 'groups': 1, 'chunks': 1})
    rows[0]['role'] = 'calibration'
    with pytest.raises(ValueError, match='group or chunk'):
        runner.validate_calibration(rows, {'recordings': 1, 'groups': 2, 'chunks': 1})
