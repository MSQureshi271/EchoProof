"""Verified development LR bundle with an explicit recording-level score rule."""
from __future__ import annotations

import json
import math
from pathlib import Path

from echoproof.bundle import BundleError, load_bundle
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.evaluation.calibrate import apply_temperature
from echoproof.models.lr_detector import LRDetector


def load_calibrated_lr_bundle(bundle_path):
    """Load model and optional recording-level calibration as a LoadedBundle."""
    from echoproof.inference import LoadedBundle

    bundle = load_bundle(bundle_path)
    if bundle.spec.preprocessing_hash != PREPROCESSING_HASH or bundle.spec.model_version != 'logistic_regression:B':
        raise BundleError('incompatible LR preprocessing or model version')
    required = {'model/weights.npz', 'model/lr_model.json', 'calibration/lr_calibration.json'}
    if set(bundle.components) != required:
        raise BundleError('unexpected LR calibration bundle components')
    meta = json.loads(bundle.components['calibration/lr_calibration.json'].read_text(encoding='utf-8'))
    if (meta.get('schema') != 'r3_n1_lr_calibration_v1' or type(meta.get('temperature_accepted')) is not bool
            or meta.get('score_type') != ('calibrated_probability' if meta['temperature_accepted'] else 'model_score')
            or meta.get('method') != 'temperature_on_log_odds'):
        raise BundleError('invalid LR calibration metadata')
    t, eps = meta.get('temperature'), meta.get('epsilon')
    if (isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0
            or isinstance(eps, bool) or not isinstance(eps, (int, float)) or not 0 < eps < .5):
        raise BundleError('invalid temperature or epsilon')
    model = LRDetector.load(bundle.components['model/lr_model.json'].parent)
    calibrate = (lambda score: apply_temperature([score], t, epsilon=eps)[0]) if meta['temperature_accepted'] else None
    return LoadedBundle(spec=bundle.spec, scorer=model.score_chunks, calibrate=calibrate, score_type=meta['score_type'])
