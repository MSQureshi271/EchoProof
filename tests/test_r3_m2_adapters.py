"""Bundle/input validation and LR parity without depending on a deployed bundle."""
import json
from pathlib import Path

import numpy as np
import pytest

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.features.acoustic import SCHEMA
from echoproof.models.base import InferenceModelAdapter
from echoproof.models.bigru_adapter import MODEL_SCHEMA, validate_metadata, validate_waveforms
from echoproof.models.lr_adapter import LRBundleAdapter
from echoproof.models.lr_detector import LRDetector
from echoproof.models.r3_m2_bundle import package_lr

ROOT = Path(__file__).resolve().parents[1]


def test_input_validation_preserves_valid_lengths_and_rejects_invalid_chunks():
    waves = [np.zeros(100, dtype=np.float64), np.ones(64000, dtype=np.float32)]
    checked = validate_waveforms(waves)
    assert [len(w) for w in checked] == [100, 64000]
    assert all(w.dtype == np.float32 for w in checked)
    for bad in [np.zeros(0), np.zeros(64001), np.zeros((2, 100)), np.zeros(10, dtype=int),
                np.array([np.nan]), np.array([np.inf]), np.array([1e300])]:
        with pytest.raises(ValueError):
            validate_waveforms([bad])


def test_metadata_rejects_incompatible_architecture_frontend_and_labels():
    meta = {'schema': MODEL_SCHEMA, 'frontend': SCHEMA, 'preprocessing_hash': PREPROCESSING_HASH,
            'label_map': {'genuine': 0, 'tts': 1}, 'protocol': 'B', 'score_type': 'raw_uncalibrated',
            'architecture': {'hidden_size': 128, 'num_layers': 2, 'bidirectional': True, 'dropout': .3},
            'component_sha256': {'weights.pt': '0'*64, 'normalization.npz': '1'*64}}
    validate_metadata(meta)
    for key, value in [('frontend', {}), ('label_map', {'genuine': 1, 'tts': 0}),
                       ('architecture', dict(meta['architecture'], num_layers=3)), ('score_type', 'calibrated')]:
        with pytest.raises(ValueError):
            validate_metadata(dict(meta, **{key: value}))


def test_lr_staging_bundle_preserves_scores_and_rejects_tampering(tmp_path):
    source = ROOT/'runs/r2/r2_m1_20261106T140611Z/model'
    hashes = {name: file_digest(source/name) for name in ('weights.npz', 'lr_model.json')}
    bundle = tmp_path/'lr'
    package_lr(source, bundle, bundle_id='bundle:r3-m2-fixture-lr', expected_hashes=hashes)
    assert load_bundle(bundle).raw_metadata['extra']['threshold_status'] == 'selection_reference_only'
    adapter = LRBundleAdapter.load(bundle)
    assert isinstance(adapter, InferenceModelAdapter)
    assert adapter.metadata['score_type'] == 'raw_uncalibrated'
    assert adapter.metadata['protocol'] == 'B'
    rng = np.random.default_rng(42)
    waves = [rng.normal(0, .02, n).astype(np.float32) for n in (512, 1600)]
    np.testing.assert_array_equal(adapter.score_chunks(waves), LRDetector.load(source).score_chunks(waves))
    assert adapter.score_chunks([]).shape == (0,)
    with pytest.raises(ValueError, match='overwrite'):
        package_lr(source, bundle, bundle_id='bundle:r3-m2-fixture-lr', expected_hashes=hashes)
    with (bundle/'model/weights.npz').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(BundleError, match='Checksum'):
        LRBundleAdapter.load(bundle)
    assert {name: file_digest(source/name) for name in hashes} == hashes


def test_lr_adapter_rejects_wrong_preprocessing_even_with_valid_components(tmp_path):
    source = ROOT/'runs/r2/r2_m1_20261106T140611Z/model'
    hashes = {name: file_digest(source/name) for name in ('weights.npz', 'lr_model.json')}
    bundle = tmp_path/'lr'
    package_lr(source, bundle, bundle_id='bundle:r3-m2-schema-fixture', expected_hashes=hashes)
    path = bundle/'bundle.json'
    meta = json.loads(path.read_text())
    meta['preprocessing_hash'] = '0'*64
    path.write_text(json.dumps(meta))
    with pytest.raises(BundleError, match='preprocessing'):
        LRBundleAdapter.load(bundle)
