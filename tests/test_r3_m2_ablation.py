import json

import numpy as np
import pytest

from echoproof.features.acoustic import FEATURE_NAMES
from echoproof.models.lr_subset import LRSubsetDetector, REMOVED_FEATURES, RETAINED_FEATURES


def training_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 67))
    y = np.repeat([0, 1], 30)
    X[:, FEATURE_NAMES.index('mfcc_01_mean')] += y*2
    X[::3, FEATURE_NAMES.index('voiced_f0_mean')] = np.nan
    ids = [f'fixture:{i//3}' for i in range(60)]
    return X, y, ids


def test_removed_features_cannot_affect_predictions_or_training():
    X, y, ids = training_data()
    a = LRSubsetDetector.fit(X, y, ids)
    changed = X.copy()
    for name in REMOVED_FEATURES:
        changed[:, FEATURE_NAMES.index(name)] = np.arange(len(X))*1e6
    b = LRSubsetDetector.fit(changed, y, ids)
    np.testing.assert_array_equal(a.predict_proba_features(X), a.predict_proba_features(changed))
    np.testing.assert_allclose(a.arrays['coef'], b.arrays['coef'], atol=1e-12, rtol=0)
    assert len(RETAINED_FEATURES) == 60 and len(set(REMOVED_FEATURES)) == 7
    assert a.meta['diagnostics']['converged']


def test_reload_parity_and_inference_do_not_refit_statistics(tmp_path):
    X, y, ids = training_data()
    model = LRSubsetDetector.fit(X, y, ids)
    before = {k: v.copy() for k, v in model.arrays.items()}
    model.save(tmp_path/'model')
    restored = LRSubsetDetector.load(tmp_path/'model')
    np.testing.assert_allclose(model.predict_proba_features(X), restored.predict_proba_features(X), atol=1e-12, rtol=0)
    model.predict_proba_features(np.full((5, 67), 10000.0))
    for k in before:
        np.testing.assert_array_equal(model.arrays[k], before[k])
    meta_path = tmp_path/'model/lr_subset.json'
    meta = json.loads(meta_path.read_text())
    meta['feature_names'].reverse()
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='schema'):
        LRSubsetDetector.load(tmp_path/'model')


def test_subset_does_not_accept_already_reduced_or_infinite_features():
    for X in [np.zeros((2, 60)), np.full((2, 67), np.inf)]:
        with pytest.raises(ValueError, match='67-column'):
            LRSubsetDetector.select_columns(X)
