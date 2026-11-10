"""Verified, waveform-facing inference adapter for the selected two-layer BiGRU.

Only tensor state dictionaries are loaded, with torch's restricted weights-only
loader. Training, recording aggregation and calibration do not belong here.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.features.acoustic import SCHEMA
from echoproof.models.base import BundlePath, validate_scores

MODEL_SCHEMA = 'echoproof_bigru_inference_v1'
COMPONENTS = ('bigru_model.json', 'weights.pt', 'normalization.npz')


def validate_metadata(meta):
    if meta.get('schema') != MODEL_SCHEMA or meta.get('frontend') != SCHEMA:
        raise ValueError('unsupported BiGRU schema/frontend')
    if meta.get('preprocessing_hash') != PREPROCESSING_HASH or meta.get('label_map') != {'genuine': 0, 'tts': 1}:
        raise ValueError('incompatible BiGRU preprocessing/label map')
    arch = meta.get('architecture', {})
    if (type(arch.get('hidden_size')) is not int or arch['hidden_size'] < 1
            or arch.get('num_layers') != 2 or arch.get('bidirectional') is not True
            or not isinstance(arch.get('dropout'), (int, float)) or not 0 <= arch['dropout'] < 1):
        raise ValueError('unsupported BiGRU architecture')
    if meta.get('protocol') not in ('A', 'B') or meta.get('score_type') != 'raw_uncalibrated':
        raise ValueError('invalid BiGRU protocol/score type')
    for name in ('weights.pt', 'normalization.npz'):
        value = meta.get('component_sha256', {}).get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'missing/invalid component hash: {name}')


def validate_waveforms(waveforms):
    result = []
    for wave in waveforms:
        x = np.asarray(wave)
        if (x.ndim != 1 or not 0 < len(x) <= 64000 or not np.issubdtype(x.dtype, np.floating)
                or not np.isfinite(x).all()):
            raise ValueError('chunks must be finite mono floating-point arrays with 1..64000 valid samples at 16 kHz')
        with np.errstate(over='ignore'):
            x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError('chunk cannot be represented in fp32')
        result.append(x)
    return result


class BiGRUBundleAdapter:
    def __init__(self, model, meta, *, device='cpu', batch_size=64, bundle_info=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self._model = model.to(device).eval()
        self._meta = meta
        self._device = device
        self._batch_size = batch_size
        self._bundle_info = dict(bundle_info or {})

    @classmethod
    def load(cls, bundle_dir: BundlePath, *, device='cpu', batch_size=64):
        import torch
        from echoproof.models.bigru_waveform import WaveformBiGRU

        path = Path(bundle_dir)
        info = {}
        if (path / 'bundle.json').is_file():
            bundle = load_bundle(path)
            if bundle.spec.preprocessing_hash != PREPROCESSING_HASH:
                raise BundleError('BiGRU bundle preprocessing hash mismatch')
            matches = {name: [p for rel, p in bundle.components.items() if Path(rel).name == name] for name in COMPONENTS}
            if any(len(v) != 1 for v in matches.values()):
                raise BundleError('BiGRU bundle must contain exactly one of each model component')
            components = {name: paths[0] for name, paths in matches.items()}
            info = {'bundle_id': bundle.spec.bundle_id, 'model_version': bundle.spec.model_version}
        else:
            if not (path / 'bigru_model.json').is_file() and (path / 'model/bigru_model.json').is_file():
                path = path / 'model'
            components = {name: path / name for name in COMPONENTS}
        meta = json.loads(components['bigru_model.json'].read_text(encoding='utf-8'))
        validate_metadata(meta)
        for name in ('weights.pt', 'normalization.npz'):
            if file_digest(components[name]) != meta['component_sha256'][name]:
                raise ValueError(f'BiGRU component hash mismatch: {name}')
        with np.load(components['normalization.npz'], allow_pickle=False) as stats:
            if set(stats.files) != {'mean', 'std'}:
                raise ValueError('unexpected normalization arrays')
            mean, std = stats['mean'], stats['std']
        arch = meta['architecture']
        model = WaveformBiGRU(mean, std, hidden_size=arch['hidden_size'], dropout=arch['dropout'])
        state = torch.load(components['weights.pt'], map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('BiGRU weights must be a finite tensor state dictionary')
        for key in ('mean', 'std'):
            if key not in state or not torch.equal(state[key], getattr(model, key)):
                raise ValueError('checkpoint normalization differs from packaged statistics')
        model.load_state_dict(state, strict=True)
        return cls(model, meta, device=device, batch_size=batch_size, bundle_info=info)

    def score_chunks(self, waveforms: list[np.ndarray]) -> np.ndarray:
        import torch

        waves = validate_waveforms(waveforms)
        if not waves:
            return np.empty(0, dtype=np.float64)
        scores = []
        self._model.eval()
        with torch.inference_mode():
            for offset in range(0, len(waves), self._batch_size):
                batch = waves[offset:offset+self._batch_size]
                padded = np.zeros((len(batch), max(map(len, batch))), dtype=np.float32)
                for i, wave in enumerate(batch):
                    padded[i, :len(wave)] = wave
                lengths = torch.tensor([len(w) for w in batch], dtype=torch.long, device=self._device)
                logits = self._model(torch.as_tensor(padded, device=self._device), lengths)
                scores.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
        return validate_scores(scores, len(waves))

    @property
    def metadata(self):
        return MappingProxyType({'model': 'bigru', 'protocol': self._meta['protocol'],
                                 'score_type': 'raw_uncalibrated', 'preprocessing_hash': PREPROCESSING_HASH,
                                 'architecture': json.dumps(self._meta['architecture'], sort_keys=True),
                                 'source_checkpoint_sha256': self._meta.get('source_checkpoint_sha256', ''),
                                 **self._bundle_info})
