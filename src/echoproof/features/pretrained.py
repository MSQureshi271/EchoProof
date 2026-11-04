"""R1-U1 pinned pretrained models and the frozen WavLM-Base-Plus embedding extractor.

Design decisions (each covered by a test in tests/test_r1_u1_wavlm.py):
  * Models are fetched only at a pinned commit; the snapshot directory name must equal that commit.
  * Preprocessing is whatever the pinned feature extractor says (WavLM-Base-Plus: 16 kHz, do_normalize=false); nothing else is
    added or removed.
  * WavLM-Base-Plus uses group normalisation over time in its first conv layer, so zero-padding a clip changes its statistics:
    a padded batch is NOT equivalent to the single clip, and an input-sample mask cannot be mapped onto hidden frames after
    that. The extractor therefore batches only clips of *identical* length (bucketing), never pads, and passes no attention
    mask. Every clip then yields the same embedding it would get alone (up to float rounding).
  * Hidden frames are counted with the encoder's own conv-length formula, and pooling is masked by those lengths, so the pooling
    stays correct if a caller ever supplies padded hidden states.
  * The encoder is frozen: no trainable parameters, always in eval mode (train() cannot re-enable dropout), inference mode, fp32
    with TF32 disabled so cached and direct embeddings agree.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 16000

# Metadata read from the Hugging Face API on 2026-11-04. Revisions are the commits to use everywhere; weights_sha256 is filled
# from the first verified download (reports/r1/r1_u1_colab_*.json) and checked on later downloads when present.
PRETRAINED = {
    "wavlm": {
        "repo_id": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "last_modified": "2021-12-22T17:23:24Z",
        "license": None,
        "license_status": "not declared in the Hugging Face metadata; confirm the terms of the WavLM source (microsoft/unilm) before redistributing weights",
        "files": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
        "weights_file": "pytorch_model.bin",
        "weights_sha256": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",  # 377,617,425 bytes; reports/r1/r1_u1_colab_20261104T114311Z.json
        "expected_preprocessing": {"sampling_rate": 16000, "do_normalize": False, "feature_size": 1},
    },
    "ast": {
        "repo_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "revision": "f826b80d28226b62986cc218e5cec390b1096902",
        "last_modified": "2023-09-06T14:49:15Z",
        "license": "bsd-3-clause",
        "license_status": "declared in the Hugging Face model card metadata",
        "files": ["config.json", "preprocessor_config.json", "model.safetensors"],
        "weights_file": "model.safetensors",
        "weights_sha256": "ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d",  # 346,404,948 bytes; reports/r1/ast_probe.json
        "expected_preprocessing": {"sampling_rate": 16000},
    },
}


class PretrainedError(RuntimeError):
    """Pinned-model retrieval or verification failure."""


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pinned_snapshot(key: str, *, files=None, cache_dir=None) -> Path:
    """Download (or reuse) the pinned revision and verify the snapshot is exactly that commit."""
    from huggingface_hub import snapshot_download
    spec = PRETRAINED[key]
    path = Path(snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                  allow_patterns=list(files or spec["files"]), cache_dir=cache_dir))
    if path.name != spec["revision"]:
        raise PretrainedError(f"{key}: snapshot {path.name} is not the pinned revision {spec['revision']}")
    expected = spec["weights_sha256"]
    weights = path / spec["weights_file"]
    if expected and weights.exists() and file_sha256(weights) != expected:
        raise PretrainedError(f"{key}: {spec['weights_file']} does not match the recorded SHA-256")
    return path


def list_trainable(module: nn.Module):
    return sorted(n for n, p in module.named_parameters() if p.requires_grad)


def pool_hidden(hidden: torch.Tensor, lengths: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool [B, T, D] hidden states over each item's first `lengths` frames.

    mean -> [B, D]; mean_std -> [B, 2D] = [mean, population std]. Padded frames never contribute.
    """
    if mode not in ("mean", "mean_std"):
        raise ValueError("pooling must be 'mean' or 'mean_std'")
    if hidden.ndim != 3 or lengths.shape != (hidden.shape[0],) or (lengths < 1).any() or (lengths > hidden.shape[1]).any():
        raise ValueError("invalid hidden states or lengths")
    hidden = hidden.float()
    lengths = lengths.to(hidden.device)
    mask = (torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]).unsqueeze(-1)
    count = lengths.unsqueeze(1).float()
    mean = (hidden * mask).sum(dim=1) / count
    if mode == "mean":
        return mean
    var = (((hidden - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / count
    return torch.cat([mean, var.sqrt()], dim=1)


class WavLMExtractor(nn.Module):
    """Frozen WavLM encoder -> one pooled embedding per waveform (mono 16 kHz float, true valid samples only)."""

    def __init__(self, encoder, feature_extractor, *, batch_size: int = 16, device: str = "cpu"):
        super().__init__()
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.batch_size = int(batch_size)
        self.device_name = device
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.to(device)
        self.eval()  # wrapper and encoder both start in eval mode, so describe() and training flags are right from construction

    @classmethod
    def from_pinned(cls, *, device: str = "cpu", batch_size: int = 16, cache_dir=None):
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
        path = pinned_snapshot("wavlm", cache_dir=cache_dir)
        return cls(WavLMModel.from_pretrained(path), Wav2Vec2FeatureExtractor.from_pretrained(path), batch_size=batch_size, device=device)

    def train(self, mode: bool = True):
        super().train(False)  # frozen: a blanket model.train() must not re-enable anything in the encoder
        return self

    @property
    def dim(self) -> int:
        return int(self.encoder.config.hidden_size)

    def hidden_length(self, samples: int) -> int:
        """Number of hidden frames the encoder produces for `samples` input samples (its own conv formula)."""
        return int(self.encoder._get_feat_extract_output_lengths(torch.tensor([int(samples)])).item())

    def _check(self, wave):
        w = np.asarray(wave, dtype=np.float32)
        if w.ndim != 1 or w.size == 0 or not np.isfinite(w).all():
            raise ValueError("each waveform must be a non-empty, finite 1-D array")
        if self.hidden_length(w.size) < 1:
            raise ValueError(f"waveform of {w.size} samples is too short for the encoder")
        return w

    def embed(self, waveforms, pooling: str = "mean") -> np.ndarray:
        """Pooled embeddings [n, dim] (mean) or [n, 2*dim] (mean_std), in the order given.

        Clips of equal length are batched together; clips of different lengths are never padded into one batch.
        """
        if pooling not in ("mean", "mean_std"):
            raise ValueError("pooling must be 'mean' or 'mean_std'")
        waves = [self._check(w) for w in waveforms]
        if not waves:
            raise ValueError("no waveforms")
        out = np.empty((len(waves), self.dim * (2 if pooling == "mean_std" else 1)), dtype=np.float32)
        buckets = {}
        for i, w in enumerate(waves):
            buckets.setdefault(w.size, []).append(i)
        with torch.inference_mode(), torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=False):
            for length, indices in sorted(buckets.items()):
                expected = self.hidden_length(length)
                for start in range(0, len(indices), self.batch_size):
                    part = indices[start:start + self.batch_size]
                    features = self.feature_extractor([waves[i] for i in part], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False)
                    hidden = self.encoder(features["input_values"].to(self.device_name, torch.float32)).last_hidden_state
                    if hidden.shape[1] != expected:
                        raise PretrainedError(f"encoder produced {hidden.shape[1]} frames, expected {expected}")
                    lengths = torch.full((len(part),), expected, dtype=torch.long)
                    out[part] = pool_hidden(hidden, lengths, pooling).cpu().numpy()
        if not np.isfinite(out).all():
            raise PretrainedError("non-finite embedding")
        return out

    def describe(self) -> dict:
        return {"hidden_size": self.dim, "parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "training_mode": self.training or self.encoder.training, "batch_size": self.batch_size,
                "feature_extractor": type(self.feature_extractor).__name__,
                "do_normalize": getattr(self.feature_extractor, "do_normalize", None)}
