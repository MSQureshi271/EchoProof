import numpy as np
import pytest

from echoproof.models.lr_detector import LRDetector
from echoproof.training.r4_m1_lr import fit_selection


def data():
    rng = np.random.default_rng(42)
    y = np.array([0, 1] * 40)
    x = rng.normal(size=(80, 67)); x[:, 0] += y * 2
    train = {'X': x, 'y': y, 'recording_id': np.array([f'rec:train:{i}' for i in range(80)])}
    labels = np.array([0, 1, 0, 1, 0, 1])
    val = {'X': rng.normal(size=(6, 67)), 'y': labels,
           'recording_id': np.array([f'rec:val:{i}' for i in range(6)]),
           'chunk_id': np.array([f'chunk:A:{i}' for i in range(6)]),
           'role': np.array(['selection', 'selection', 'calibration', 'calibration', 'operating', 'operating'])}
    return train, val


def test_fit_uses_train_only_and_selection_only(tmp_path):
    train, val = data()
    params = {'C': 0.01, 'class_weight': 'none', 'max_iter': 2000, 'seed': 42}
    model, selected, scores, rec, metrics = fit_selection(train, val, params, 'split')
    assert len(scores) == len(rec.recording_ids) == 2
    assert set(selected['role']) == {'selection'}
    np.testing.assert_allclose(model.arrays['medians'], np.median(train['X'], axis=0))
    val['X'][2:] = 1e8
    changed, _, scores2, _, metrics2 = fit_selection(train, val, params, 'split')
    np.testing.assert_array_equal(model.arrays['coef'], changed.arrays['coef'])
    np.testing.assert_array_equal(scores, scores2)
    assert metrics == metrics2
    model.save(tmp_path/'model')
    np.testing.assert_allclose(LRDetector.load(tmp_path/'model').predict_proba_features(selected['X']), scores, atol=1e-12, rtol=0)


def test_rejects_empty_selection():
    train, val = data(); val['role'][:] = 'operating'
    with pytest.raises(ValueError, match='selection role is empty'):
        fit_selection(train, val, {'C': .01, 'class_weight': 'none', 'max_iter': 2000, 'seed': 42}, 'split')
