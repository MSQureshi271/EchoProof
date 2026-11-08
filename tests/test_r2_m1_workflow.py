import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH

spec = importlib.util.spec_from_file_location('r2_m1', Path(__file__).resolve().parents[1] / 'scripts/r2_m1_lr.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def handoff(root):
    rng = np.random.default_rng(12)
    manifest = {'protocol': 'B', 'verification': {'passed': True}, 'inputs': {'feature_schema_hash': FEATURE_SCHEMA_HASH, 'split_sha256': 'a' * 64}, 'plan_sha256': 'b' * 64, 'outputs': {}}
    for name, n in [('train_features', 24), ('validation_features', 24)]:
        labels = np.arange(n) % 2
        data = {'X': rng.normal(size=(n, 67)) + labels[:, None], 'y': labels,
                'recording_id': np.array([f'fixture:{name}:{i}' for i in range(n)]),
                'group_id': np.array([f'speaker:{name}:{i}' for i in range(n)]),
                'component': np.where(labels, 'melo', 'genuine'), 'feature_names': np.array(FEATURE_NAMES),
                'feature_schema_hash': np.array(FEATURE_SCHEMA_HASH), 'plan_sha256': np.array('b' * 64)}
        data['crop_id' if name == 'train_features' else 'chunk_id'] = np.array([f'chunk:{name}:{i}' for i in range(n)])
        if name == 'validation_features':
            data['role'] = np.repeat(['selection', 'calibration', 'operating'], 8)
        path = root / f'{name}.npz'
        np.savez(path, **data)
        manifest['outputs'][name] = {'file': path.name, 'sha256': mod.sha(path)}
    (root / 'r2_s1_manifest.json').write_text(json.dumps(manifest))


def test_end_to_end_and_selection_only(tmp_path):
    handoff(tmp_path)
    result = mod.run(tmp_path, tmp_path / 'run', tmp_path / 'ledger.csv')
    assert result['status'] == 'complete'
    assert len(result['trials']) == 8
    assert result['selection_recordings'] == 8
    assert result['reload_max_abs'] <= 1e-12
    assert len((tmp_path / 'ledger.csv').read_text().splitlines()) == 9
    assert (tmp_path / 'run/model/lr_model.json').exists()


def test_rejects_tampered_feature_file(tmp_path):
    handoff(tmp_path)
    with (tmp_path / 'train_features.npz').open('ab') as f:
        f.write(b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        mod.load_inputs(tmp_path)
