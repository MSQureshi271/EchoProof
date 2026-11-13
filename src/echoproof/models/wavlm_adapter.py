"""Verified, waveform-facing inference adapter for the R3-U1 WavLM head. The encoder is inside the bundle.

`score_chunks` takes mono 16 kHz float waveforms (true valid samples only), embeds them with the frozen WavLM encoder (pooled over true frames, the same
code path that built the R2-U1 cache), applies the trained head and returns P(TTS). Only tensor state is loaded: the head with torch's weights-only loader,
the encoder from safetensors. The pinned `pytorch_model.bin` is a pickle and is converted once at packaging time (`models.r3_u1_bundle`); this module
never reads a pickle. Training, calibration and aggregation do not belong here.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.models.base import BundlePath, validate_scores

MODEL_SCHEMA = "echoproof_wavlm_head_inference_v1"
HEAD_FILES = ("wavlm_head.json", "head_weights.pt")
ENCODER_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
MAX_SAMPLES = 64000
POOLING_MULTIPLIER = {"mean": 1, "mean_std": 2}
EXPECTED_PROCESSOR = {"do_normalize": False, "feature_size": 1, "sampling_rate": 16000}


def _is_hash(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_metadata(meta: dict) -> None:
    if meta.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported WavLM head schema")
    if meta.get("preprocessing_hash") != PREPROCESSING_HASH or meta.get("label_map") != {"genuine": 0, "tts": 1}:
        raise ValueError("incompatible WavLM head preprocessing or label map")
    if meta.get("protocol") not in ("A", "B") or meta.get("score_type") != "raw_uncalibrated" or meta.get("pooling") not in POOLING_MULTIPLIER:
        raise ValueError("invalid WavLM head protocol, score type or pooling")
    arch, enc = meta.get("architecture", {}), meta.get("encoder", {})
    if (type(arch.get("in_dim")) is not int or type(arch.get("hidden")) is not int or arch["hidden"] < 1
            or not isinstance(arch.get("dropout"), (int, float)) or not 0 <= arch["dropout"] < 1):
        raise ValueError("unsupported WavLM head architecture")
    if type(enc.get("hidden_size")) is not int or arch["in_dim"] != enc["hidden_size"] * POOLING_MULTIPLIER[meta["pooling"]]:
        raise ValueError("head input width does not match the encoder width and pooling")
    if not isinstance(enc.get("repo_id"), str) or not isinstance(enc.get("revision"), str) or not _is_hash(enc.get("source_weights_sha256")):
        raise ValueError("missing encoder identity")
    if enc.get("converted_state_equals_source") is not True:
        raise ValueError("the encoder conversion was not verified equal to the pinned source")
    for name in HEAD_FILES[1:] + tuple("encoder/" + n for n in ENCODER_FILES):
        if not _is_hash(meta.get("component_sha256", {}).get(name)):
            raise ValueError(f"missing or invalid component hash: {name}")


def validate_waveforms(waveforms):
    result = []
    for wave in waveforms:
        x = np.asarray(wave)
        if x.ndim != 1 or not 0 < len(x) <= MAX_SAMPLES or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
            raise ValueError(f"chunks must be finite mono floating-point arrays with 1..{MAX_SAMPLES} valid samples at 16 kHz")
        with np.errstate(over="ignore"):
            x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("chunk cannot be represented in fp32")
        result.append(x)
    return result


class WavLMBundleAdapter:
    def __init__(self, extractor, head, meta, *, device="cpu", bundle_info=None):
        self._extractor, self._head, self._meta, self._device = extractor, head.to(device).eval(), meta, device
        self._bundle_info = dict(bundle_info or {})

    @classmethod
    def load(cls, bundle_dir: BundlePath, *, device="cpu", batch_size=16):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
        from echoproof.features.pretrained import WavLMExtractor
        from echoproof.models.wavlm_head import WavLMHead

        path, info = Path(bundle_dir), {}
        if not (path / "bundle.json").is_file():
            raise BundleError("WavLM head adapters load from a verified bundle (bundle.json required)")
        bundle = load_bundle(path)
        if bundle.spec.preprocessing_hash != PREPROCESSING_HASH:
            raise BundleError("WavLM head bundle preprocessing hash mismatch")
        info = {"bundle_id": bundle.spec.bundle_id, "model_version": bundle.spec.model_version}
        files = {}
        for name in HEAD_FILES + tuple("encoder/" + n for n in ENCODER_FILES):
            match = [p for rel, p in bundle.components.items() if rel.endswith("/" + name) or rel == name]
            if len(match) != 1:
                raise BundleError(f"WavLM head bundle must contain exactly one {name}")
            files[name] = match[0]
        meta = json.loads(files["wavlm_head.json"].read_text(encoding="utf-8"))
        validate_metadata(meta)
        for name, expected in meta["component_sha256"].items():
            if file_digest(files[name]) != expected:
                raise ValueError(f"WavLM head component hash mismatch: {name}")
        encoder_dir = files["encoder/model.safetensors"].parent
        extractor_config = json.loads(files["encoder/preprocessor_config.json"].read_text(encoding="utf-8"))
        if any(extractor_config.get(k) != v for k, v in EXPECTED_PROCESSOR.items()):
            raise ValueError("encoder preprocessing differs from the one used to build the cache")
        encoder = WavLMModel.from_pretrained(encoder_dir, local_files_only=True)
        if int(encoder.config.hidden_size) != meta["encoder"]["hidden_size"]:
            raise ValueError("encoder width differs from the packaged metadata")
        extractor = WavLMExtractor(encoder, Wav2Vec2FeatureExtractor.from_pretrained(encoder_dir), batch_size=batch_size, device=device)
        arch = meta["architecture"]
        head = WavLMHead(arch["in_dim"], hidden=arch["hidden"], dropout=arch["dropout"])
        state = torch.load(files["head_weights.pt"], map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("head weights must be a finite tensor state dictionary")
        head.load_state_dict(state, strict=True)
        return cls(extractor, head, meta, device=device, bundle_info=info)

    def embed(self, waveforms) -> np.ndarray:
        """Pooled encoder embeddings for validated waveforms (the exact vectors the head scores)."""
        return self._extractor.embed(validate_waveforms(waveforms), pooling=self._meta["pooling"])

    def score_chunks(self, waveforms: list[np.ndarray]) -> np.ndarray:
        import torch
        waves = validate_waveforms(waveforms)
        if not waves:
            return np.empty(0, dtype=np.float64)
        vectors = self._extractor.embed(waves, pooling=self._meta["pooling"])
        with torch.inference_mode():
            probabilities = self._head.scores(torch.from_numpy(vectors).to(self._device)).double().cpu().numpy()
        return validate_scores(probabilities, len(waves))

    @property
    def metadata(self):
        enc = self._meta["encoder"]
        return MappingProxyType({"model": "wavlm_head", "protocol": self._meta["protocol"], "score_type": "raw_uncalibrated",
                                 "pooling": self._meta["pooling"], "preprocessing_hash": PREPROCESSING_HASH,
                                 "architecture": json.dumps(self._meta["architecture"], sort_keys=True),
                                 "encoder_repo_id": enc["repo_id"], "encoder_revision": enc["revision"],
                                 "source_checkpoint_sha256": self._meta.get("source_checkpoint_sha256", ""), **self._bundle_info})
