"""Fixed-schema LR feature-group ablation; the original 67-column model is untouched."""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from echoproof.artifacts import digest, file_digest
from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH
from echoproof.models.lr_detector import recording_weights, chunk_features, _sigmoid

REMOVED_FEATURES = ('mfcc_00_mean', 'mfcc_00_std', 'delta_00_mean', 'delta_00_std',
                    'rms_mean', 'rms_std', 'voiced_fraction')
RETAINED_FEATURES = tuple(n for n in FEATURE_NAMES if n not in REMOVED_FEATURES)
INDICES = tuple(FEATURE_NAMES.index(n) for n in RETAINED_FEATURES)
SUBSET_SCHEMA_HASH = digest({'source': FEATURE_SCHEMA_HASH, 'retained': RETAINED_FEATURES, 'removed': REMOVED_FEATURES})
ARRAYS = ('medians', 'mean', 'scale', 'coef', 'intercept')


class LRSubsetDetector:
    """Only this declared 60-feature diagnostic is supported; no feature search."""
    def __init__(self, arrays, meta):
        self.arrays = {k: np.asarray(arrays[k], dtype=np.float64) for k in ARRAYS}
        self.meta = meta
        if (meta.get('schema') != 'r3_m2_lr_subset_v1' or meta.get('feature_schema_hash') != SUBSET_SCHEMA_HASH
                or meta.get('source_feature_schema_hash') != FEATURE_SCHEMA_HASH
                or meta.get('feature_names') != list(RETAINED_FEATURES)
                or meta.get('removed_features') != list(REMOVED_FEATURES)
                or meta.get('label_map') != {'genuine': 0, 'tts': 1}):
            raise ValueError('incompatible LR subset schema')
        for name, array in self.arrays.items():
            if array.shape != ((1,) if name == 'intercept' else (60,)) or not np.isfinite(array).all():
                raise ValueError('invalid LR subset arrays')
        if (self.arrays['scale'] <= 0).any():
            raise ValueError('invalid subset scale')

    @staticmethod
    def select_columns(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or np.isinf(X).any():
            raise ValueError('expected finite-or-NaN full 67-column feature matrix')
        return X[:, INDICES]

    @classmethod
    def fit(cls, X, y, recording_ids, *, C=.01, class_weight='none', max_iter=2000, seed=42):
        selected = cls.select_columns(X)
        y = np.asarray(y)
        if y.shape != (len(selected),) or set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError('two binary classes and one label per training row are required')
        weights = recording_weights(recording_ids, y, class_weight)
        pipeline = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                             ('scaler', StandardScaler()),
                             ('classifier', LogisticRegression(C=C, max_iter=max_iter, random_state=seed))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipeline.fit(selected, y, scaler__sample_weight=weights, classifier__sample_weight=weights)
        imputer, scaler, classifier = (pipeline.named_steps[k] for k in ('imputer', 'scaler', 'classifier'))
        if classifier.classes_.tolist() != [0, 1]:
            raise ValueError('incorrect LR positive class')
        n_iter = int(classifier.n_iter_.max())
        diagnostics = {'converged': n_iter < max_iter and not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                       'n_iter': n_iter, 'max_iter': max_iter, 'chunks': len(y),
                       'recordings': len(set(recording_ids)), 'weight_sum': float(weights.sum())}
        meta = {'schema': 'r3_m2_lr_subset_v1', 'source_feature_schema_hash': FEATURE_SCHEMA_HASH,
                'feature_schema_hash': SUBSET_SCHEMA_HASH, 'feature_names': list(RETAINED_FEATURES),
                'removed_features': list(REMOVED_FEATURES), 'label_map': {'genuine': 0, 'tts': 1},
                'hyperparameters': {'C': C, 'class_weight': class_weight, 'max_iter': max_iter, 'seed': seed},
                'diagnostics': diagnostics, 'scope': 'experimental_only_energy_activity_proxy_ablation'}
        return cls({'medians': imputer.statistics_, 'mean': scaler.mean_, 'scale': scaler.scale_,
                    'coef': classifier.coef_[0], 'intercept': classifier.intercept_}, meta)

    def predict_proba_features(self, X):
        selected, a = self.select_columns(X), self.arrays
        selected = np.where(np.isnan(selected), a['medians'], selected)
        return _sigmoid(((selected-a['mean'])/a['scale']) @ a['coef'] + a['intercept'][0])

    def score_chunks(self, waveforms):
        if not len(waveforms):
            return np.empty(0, dtype=np.float64)
        return self.predict_proba_features(chunk_features(waveforms))

    def coefficient_table(self):
        return sorted(zip(RETAINED_FEATURES, self.arrays['coef'].tolist()), key=lambda x: (-abs(x[1]), x[0]))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory/'weights.npz', **self.arrays)
        meta = dict(self.meta, weights_sha256=file_digest(directory/'weights.npz'))
        (directory/'lr_subset.json').write_text(json.dumps(meta, indent=2, sort_keys=True)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        meta = json.loads((directory/'lr_subset.json').read_text(encoding='utf-8'))
        if file_digest(directory/'weights.npz') != meta['weights_sha256']:
            raise ValueError('subset weights checksum mismatch')
        with np.load(directory/'weights.npz', allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAYS):
                raise ValueError('unexpected subset arrays')
            return cls({k: archive[k] for k in ARRAYS}, meta)
