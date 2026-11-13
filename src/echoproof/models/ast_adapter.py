"""Verified, waveform-facing inference adapter for the R3-U2 fine-tuned AST. The whole model is inside the bundle.

`score_chunks` takes mono 16 kHz float waveforms (true valid samples only, at most four seconds), runs the same code path that produced the R3-U2
exports (`ASTDetector`: the pinned AST feature extractor on the valid samples, then the model) and returns P(TTS). Only tensor state is loaded: the
weights come from safetensors, written once at packaging time. Training, calibration and aggregation do not belong here.

Precision: the R3-U2 exports were scored with fp16 autocast on a T4. `precision="fp16"` reproduces that numerics exactly on a CUDA device;
`precision="fp32"` (the default, and the only choice on CPU) gives slightly different probabilities. The packaging parity report states the measured
difference; thresholds fitted on the fp16 exports (R3-N1) carry that difference when the adapter runs in fp32.
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
from echoproof.models.wavlm_adapter import MAX_SAMPLES, validate_waveforms

MODEL_SCHEMA = "echoproof_ast_inference_v1"
META_FILE = "ast_detector.json"
MODEL_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
EXPECTED_PROCESSOR = {"do_normalize": True, "sampling_rate": 16000, "num_mel_bins": 128, "max_length": 1024,
                      "mean": -4.2677393, "std": 4.5689974}
PRECISIONS = ("fp32", "fp16")


def _is_hash(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_metadata(meta: dict) -> None:
    if meta.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported AST schema")
    if meta.get("preprocessing_hash") != PREPROCESSING_HASH or meta.get("label_map") != {"genuine": 0, "tts": 1}:
        raise ValueError("incompatible AST preprocessing or label map")
    if meta.get("protocol") not in ("A", "B") or meta.get("score_type") != "raw_uncalibrated" or meta.get("export_precision") not in PRECISIONS:
        raise ValueError("invalid AST protocol, score type or export precision")
    model, base, tune = meta.get("model", {}), meta.get("base", {}), meta.get("fine_tuning", {})
    if model.get("num_labels") != 2 or type(model.get("hidden_size")) is not int or type(model.get("blocks")) is not int or model["blocks"] < 1:
        raise ValueError("unsupported AST model description")
    if not isinstance(base.get("repo_id"), str) or not isinstance(base.get("revision"), str) or not _is_hash(base.get("source_weights_sha256")):
        raise ValueError("missing base model identity")
    if not _is_hash(tune.get("warmup_checkpoint_sha256")) or not _is_hash(meta.get("source_checkpoint_sha256")):
        raise ValueError("missing checkpoint identity")
    if tune.get("frozen_groups_equal_pinned") is not True or not tune.get("unfrozen_groups") or tune.get("unfrozen_groups_differ_from_pinned") is not True:
        raise ValueError("the frozen/unfrozen evidence is missing or negative")
    if meta.get("conversion", {}).get("state_equals_checkpoint") is not True:
        raise ValueError("the saved weights were not verified equal to the selected checkpoint")
    if not isinstance(meta.get("processor"), dict) or not meta["processor"]:
        raise ValueError("missing processor expectations")
    for name in MODEL_FILES[1:] + MODEL_FILES[:1]:
        if not _is_hash(meta.get("component_sha256", {}).get(name)):
            raise ValueError(f"missing or invalid component hash: {name}")


class ASTBundleAdapter:
    def __init__(self, detector, meta, *, device="cpu", batch_size=8, precision="fp32", bundle_info=None):
        self._detector, self._meta, self._device = detector, meta, device
        self._batch, self._precision = int(batch_size), precision
        self._bundle_info = dict(bundle_info or {})

    @classmethod
    def load(cls, bundle_dir: BundlePath, *, device="cpu", batch_size=8, precision="fp32"):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        from echoproof.models.ast_detector import ASTDetector

        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if precision == "fp16" and not str(device).startswith("cuda"):
            raise ValueError("fp16 inference requires a CUDA device; use fp32 on CPU")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        path = Path(bundle_dir)
        if not (path / "bundle.json").is_file():
            raise BundleError("AST adapters load from a verified bundle (bundle.json required)")
        bundle = load_bundle(path)
        if bundle.spec.preprocessing_hash != PREPROCESSING_HASH:
            raise BundleError("AST bundle preprocessing hash mismatch")
        info = {"bundle_id": bundle.spec.bundle_id, "model_version": bundle.spec.model_version}
        files = {}
        for name in (META_FILE,) + MODEL_FILES:
            match = [p for rel, p in bundle.components.items() if rel.endswith("/" + name) or rel == name]
            if len(match) != 1:
                raise BundleError(f"AST bundle must contain exactly one {name}")
            files[name] = match[0]
        meta = json.loads(files[META_FILE].read_text(encoding="utf-8"))
        validate_metadata(meta)
        for name, expected in meta["component_sha256"].items():
            if file_digest(files[name]) != expected:
                raise ValueError(f"AST component hash mismatch: {name}")
        if len({files[n].parent for n in MODEL_FILES}) != 1:
            raise BundleError("the AST model files must sit in one folder")
        model_dir = files["model.safetensors"].parent
        processor = json.loads(files["preprocessor_config.json"].read_text(encoding="utf-8"))
        if any(processor.get(k) != v for k, v in meta["processor"].items()):
            raise ValueError("feature-extractor configuration differs from the packaged expectations")
        model = ASTForAudioClassification.from_pretrained(model_dir, local_files_only=True)
        if int(model.config.num_labels) != 2 or int(model.config.hidden_size) != meta["model"]["hidden_size"]:
            raise ValueError("model shape differs from the packaged metadata")
        extractor = ASTFeatureExtractor.from_pretrained(model_dir)
        detector = ASTDetector(model.to(device).eval(), extractor)
        if detector.n_blocks != meta["model"]["blocks"]:
            raise ValueError("block count differs from the packaged metadata")
        return cls(detector, meta, device=device, batch_size=batch_size, precision=precision, bundle_info=info)

    def score_chunks(self, waveforms: list[np.ndarray]) -> np.ndarray:
        import torch
        waves = validate_waveforms(waveforms)
        if not waves:
            return np.empty(0, dtype=np.float64)
        device = torch.device(self._device)
        dtype = torch.float16 if self._precision == "fp16" else None
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(waves), self._batch):
                group = waves[start:start + self._batch]
                width = max(len(w) for w in group)
                padded = np.zeros((len(group), width), dtype=np.float32)
                for row, wave in enumerate(group):
                    padded[row, :len(wave)] = wave
                valid = torch.tensor([len(w) for w in group], dtype=torch.int64)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                    logits = self._detector(torch.from_numpy(padded).to(device), valid)
                probabilities.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
        return validate_scores(np.asarray(probabilities, dtype=np.float64), len(waves))

    @property
    def metadata(self):
        base, tune = self._meta["base"], self._meta["fine_tuning"]
        return MappingProxyType({"model": "ast_partial", "protocol": self._meta["protocol"], "score_type": "raw_uncalibrated",
                                 "preprocessing_hash": PREPROCESSING_HASH, "inference_precision": self._precision,
                                 "export_precision": self._meta["export_precision"], "base_repo_id": base["repo_id"],
                                 "base_revision": base["revision"], "warmup_checkpoint_sha256": tune["warmup_checkpoint_sha256"],
                                 "unfrozen_groups": ",".join(tune["unfrozen_groups"]),
                                 "source_checkpoint_sha256": self._meta["source_checkpoint_sha256"], **self._bundle_info})
