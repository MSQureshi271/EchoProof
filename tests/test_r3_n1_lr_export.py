"""Safety checks for the LR calibration prediction export."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import pyarrow.parquet as pq
from echoproof.artifacts import file_digest
from echoproof.bundle import write_bundle
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.models.lr_detector import LRDetector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('r3_n1_lr_export', ROOT/'scripts/r3_n1_lr_export.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture
def fixture():
    table = {'chunk_id': np.array(['s0', 's1', 'c0', 'c1', 'o0', 'o1']),
             'recording_id': np.array(['sr0', 'sr1', 'cr0', 'cr1', 'or0', 'or1']),
             'group_id': np.array(['sg', 'sg', 'cg', 'cg', 'og', 'og']),
             'component': np.array(['genuine', 'xtts']*3),
             'role': np.array(['selection']*2+['calibration']*2+['operating']*2),
             'y': np.array([0, 1]*3)}
    contract = {r: {'chunks': 2, 'recordings': 2, 'groups': 1} for r in ('selection', 'calibration', 'operating')}
    return table, contract


def test_role_coverage_and_speaker_separation(fixture):
    table, contract = fixture
    exporter.validate_roles(table, contract)
    altered = {k: v.copy() for k, v in table.items()}
    altered['group_id'][2:] = 'sg'
    with pytest.raises(ValueError, match='speaker crosses'):
        exporter.validate_roles(altered, contract)
    wrong = {**contract, 'operating': dict(contract['operating'], recordings=3)}
    with pytest.raises(ValueError, match='coverage'):
        exporter.validate_roles(table, wrong)


def test_rejects_leaking_components_and_duplicate_chunks(fixture):
    table, contract = fixture
    altered = {k: v.copy() for k, v in table.items()}
    altered['component'] = altered['component'].astype('<U16')
    altered['component'][1] = 'chatterbox'
    with pytest.raises(ValueError, match='component'):
        exporter.validate_roles(altered, contract)
    altered = {k: v.copy() for k, v in table.items()}
    altered['chunk_id'][1] = altered['chunk_id'][0]
    with pytest.raises(ValueError, match='duplicate'):
        exporter.validate_roles(altered, contract)


def test_export_writes_separate_role_predictions_and_preserves_recording_scores(tmp_path, monkeypatch, fixture):
    table, roles = fixture
    table['chunk_id'] = np.array(['chunk:'+v for v in table['chunk_id']])
    table['recording_id'] = np.array(['recording:'+v for v in table['recording_id']])
    model_dir = ROOT/'runs/r2/r2_m1_20261106T140611Z/model'
    bundle = tmp_path/'bundle'
    spec = BundleSpecification(1, 'bundle:lr-export-fixture', 'logistic_regression:B', PREPROCESSING_HASH,
                               DecisionRule(1, 'threshold', .5), 'genuine', 'tts')
    write_bundle(bundle, spec, {'model/weights.npz': model_dir/'weights.npz', 'model/lr_model.json': model_dir/'lr_model.json'})
    features = tmp_path/'features'; features.mkdir()
    table['X'] = np.zeros((6, 67), dtype=np.float64)
    np.savez(features/'validation.npz', **table)
    (features/'r2_s1_manifest.json').write_text('fixture manifest')
    selection = LRDetector.load(bundle/'model').predict_proba_features(table['X'][:2])
    ref = tmp_path/'reference'; ref.mkdir()
    import csv
    with (ref/'predictions_chunks.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['chunk_id', 'recording_id', 'label', 'raw_score'])
        w.writerows(zip(table['chunk_id'][:2], table['recording_id'][:2], table['y'][:2], selection))
    (ref/'evidence.json').write_text(json.dumps({'outputs': {'predictions_chunks.csv': file_digest(ref/'predictions_chunks.csv')}}))
    contract = {**roles, 'bundle': 'bundle', 'bundle_sha256': file_digest(bundle/'bundle.json'),
                'reference_evidence': 'reference/evidence.json', 'reference_evidence_sha256': file_digest(ref/'evidence.json'),
                'model_components_sha256': {n: file_digest(bundle/'model'/n) for n in ('weights.npz', 'lr_model.json')},
                'feature_manifest_sha256': file_digest(features/'r2_s1_manifest.json'),
                'validation_features_sha256': file_digest(features/'validation.npz'), 'split_sha256': 'split',
                'output_root': 'runs/r3/r3_n1_lr'}
    cp = tmp_path/'contract.json'; cp.write_text(json.dumps(contract))
    manifest = {'outputs': {'validation_features': {'file': 'validation.npz', 'sha256': contract['validation_features_sha256']}},
                'inputs': {'split_sha256': 'split'}}
    monkeypatch.setattr(exporter, 'ROOT', tmp_path); monkeypatch.setattr(exporter, 'CONTRACT', cp)
    monkeypatch.setattr(exporter, 'load_inputs', lambda _: (manifest, None, None))
    for source in ('scripts/r3_n1_lr_export.py', 'scripts/r2_m1_lr.py', 'src/echoproof/models/lr_detector.py', 'src/echoproof/evaluation/aggregate.py'):
        p = tmp_path/source; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixture')
    monkeypatch.setattr(exporter, '__file__', str(tmp_path/'scripts/r3_n1_lr_export.py'))
    output = tmp_path/'runs/r3/r3_n1_lr/export'
    result = exporter.export(features, output)
    assert result['selection_max_abs_difference'] == 0 and len(result['outputs']) == 6
    assert pq.read_table(output/'predictions_recordings_calibration.parquet').num_rows == 2
    with pytest.raises(FileExistsError): exporter.export(features, output)
