"""Additive development-bundle packaging; no fitting, calibration or lock updates."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from echoproof.artifacts import file_digest
from echoproof.bundle import write_bundle
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.features.acoustic import SCHEMA
from echoproof.models.bigru_adapter import MODEL_SCHEMA, validate_metadata


def package_bigru(checkpoint_path, destination, *, bundle_id, architecture, protocol,
                  expected_checkpoint_sha256, split_sha256, expected_epoch=None, expected_config_hash=None):
    import torch

    source, destination = Path(checkpoint_path), Path(destination)
    if destination.exists():
        raise FileExistsError('refusing to replace an existing bundle')
    if file_digest(source) != expected_checkpoint_sha256:
        raise ValueError('source checkpoint differs from selected checkpoint hash')
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if expected_epoch is not None and checkpoint.get('epoch') != expected_epoch:
        raise ValueError('checkpoint epoch differs from selection')
    if expected_config_hash is not None and checkpoint.get('config_hash') != expected_config_hash:
        raise ValueError('checkpoint configuration differs from selection')
    state = checkpoint['model']
    from echoproof.models.bigru_waveform import WaveformBiGRU
    model = WaveformBiGRU(state['mean'].cpu().numpy(), state['std'].cpu().numpy(),
                          hidden_size=architecture['hidden_size'], dropout=architecture['dropout'])
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('non-finite inference state')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='r3_m2_package_', dir=destination.parent) as temp:
        folder = Path(temp)
        torch.save({k: v.detach().cpu().contiguous() for k, v in state.items()}, folder/'weights.pt')
        np.savez(folder/'normalization.npz', mean=state['mean'].cpu().numpy(), std=state['std'].cpu().numpy())
        meta = {'schema': MODEL_SCHEMA, 'architecture': architecture, 'frontend': SCHEMA,
                'protocol': protocol, 'label_map': {'genuine': 0, 'tts': 1}, 'score_type': 'raw_uncalibrated',
                'preprocessing_hash': PREPROCESSING_HASH, 'source_checkpoint_sha256': expected_checkpoint_sha256,
                'source_epoch': checkpoint['epoch'], 'source_config_hash': checkpoint.get('config_hash'),
                'split_sha256': split_sha256,
                'component_sha256': {name: file_digest(folder/name) for name in ('weights.pt', 'normalization.npz')}}
        validate_metadata(meta)
        (folder/'bigru_model.json').write_text(json.dumps(meta, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        spec = BundleSpecification(1, bundle_id, 'bigru:'+protocol, PREPROCESSING_HASH,
                                   DecisionRule(1, 'threshold', .5), 'genuine', 'tts')
        return write_bundle(destination, spec, {'model/'+name: folder/name for name in
                            ('weights.pt', 'normalization.npz', 'bigru_model.json')},
                            extra={'stage': 'development_uncalibrated', 'threshold_status': 'selection_reference_only',
                                   'source_checkpoint_sha256': expected_checkpoint_sha256, 'R3_N1_required': True})


def package_lr(model_dir, destination, *, bundle_id, expected_hashes):
    from echoproof.models.lr_detector import LRDetector

    source = Path(model_dir)
    for name in ('weights.npz', 'lr_model.json'):
        if file_digest(source/name) != expected_hashes[name]:
            raise ValueError('LR component differs from R2-M1 evidence')
    LRDetector.load(source)
    spec = BundleSpecification(1, bundle_id, 'logistic_regression:B', PREPROCESSING_HASH,
                               DecisionRule(1, 'threshold', .5), 'genuine', 'tts')
    return write_bundle(destination, spec, {'model/'+name: source/name for name in ('weights.npz', 'lr_model.json')},
                        extra={'stage': 'development_uncalibrated', 'threshold_status': 'selection_reference_only',
                               'R3_N1_required': True})
