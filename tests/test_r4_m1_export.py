from types import SimpleNamespace
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from echoproof.training import r4_m1_export as module
from echoproof.training.r4_m1_data import file_sha256
from echoproof.training.r4_m1_lr import fit_selection


def test_parity_rejects_changed_or_missing_scores():
    saved = [{'chunk_id': 'c:1', 'recording_id': 'r:1', 'label': 0, 'raw_score': .4}]
    assert module.check_selection([('c:1', 'r:1', 0, .4)], saved, 1e-12) == 0
    with pytest.raises(ValueError, match='parity'):
        module.check_selection([('c:1', 'r:1', 0, .5)], saved, 1e-12)
    with pytest.raises(ValueError): module.check_selection([], saved, 1e-12)


def test_lr_export_full_handoff(tmp_path, monkeypatch):
    root = tmp_path; (root/'configs').mkdir(); (root/'runs/r4/train').mkdir(parents=True)
    params = {'C': .01, 'class_weight': 'none', 'max_iter': 2000, 'seed': 42}
    recipe = {'lr': {'hyperparameters': params}}
    (root/'configs/r4_m1_recipes.json').write_text(json.dumps(recipe))
    rng = np.random.default_rng(12)
    train = {'X': rng.normal(size=(40, 67)), 'y': np.array([0, 1]*20), 'recording_id': np.array([f'rec:t:{i}' for i in range(40)])}
    roles = ['selection', 'selection', 'calibration', 'calibration', 'operating', 'operating']
    val = {'X': rng.normal(size=(6, 67)), 'y': np.array([0, 1]*3), 'role': np.array(roles),
           'chunk_id': np.array([f'chunk:A:{i}' for i in range(6)]), 'recording_id': np.array([f'rec:v:{i}' for i in range(6)])}
    model, sel, scores, _, _ = fit_selection(train, val, params, 'split')
    model.save(root/'runs/r4/train/model')
    saved = [{'chunk_id': c, 'recording_id': r, 'label': int(y), 'raw_score': float(s)} for c,r,y,s in zip(sel['chunk_id'],sel['recording_id'],sel['y'],scores)]
    pq.write_table(pa.Table.from_pylist(saved), root/'runs/r4/train/predictions_chunks_selection.parquet')
    pq.write_table(pa.table({'placeholder': [1]}), root/'runs/r4/train/predictions_recordings_selection.parquet')
    names = ('model/weights.npz', 'model/lr_model.json', 'predictions_chunks_selection.parquet', 'predictions_recordings_selection.parquet')
    evidence = {'task':'R4-M1-LR','status':'complete','protocol':'A','split_sha256':'split',
                'recipe_sha256':file_sha256(root/'configs/r4_m1_recipes.json'),'input_sha256':{'manifest':'hash'},
                'fit_diagnostics':{'converged':True},'outputs':{n:file_sha256(root/'runs/r4/train'/n) for n in names}}
    (root/'runs/r4/train/evidence.json').write_text(json.dumps(evidence))
    datasets = {}
    for role in module.ROLES:
        ids = np.flatnonzero(val['role'] == role)
        chunks = [SimpleNamespace(chunk_id=val['chunk_id'][i], recording_id=val['recording_id'][i]) for i in ids]
        datasets[role] = SimpleNamespace(chunks=chunks, labels_by_recording={val['recording_id'][i]:int(val['y'][i]) for i in ids})
    loader = SimpleNamespace(close=lambda: None)
    info = {'recipe':recipe,'split_sha256':'split','input_sha256':{'manifest':'hash'},'lr_validation':val,'validation':datasets,
            'validation_groups':{r:f'speaker:{i}' for i,r in enumerate(val['recording_id'])},
            'validation_components':{r:('genuine' if y == 0 else 'melo') for r,y in zip(val['recording_id'],val['y'])},
            'counts':{'validation_chunks':{r:2 for r in module.ROLES},'validation_recordings':{r:2 for r in module.ROLES}},'loader':loader}
    monkeypatch.setattr(module, 'build_data', lambda *args: info)
    out = root/'runs/r4/export'
    result = module.export(root, root/'shards', root/'runs/r4/train', out, 'lr')
    assert result['status'] == 'complete' and len(result['outputs']) == 6
    assert result['selection_reload_max_abs'] == 0
    for role in module.ROLES:
        rows = pq.read_table(out/f'predictions_recordings_{role}.parquet').to_pylist()
        assert len(rows) == 2 and {row['role'] for row in rows} == {role}
    with pytest.raises(ValueError): module.export(root, root/'shards', root/'runs/r4/train', out, 'lr')
