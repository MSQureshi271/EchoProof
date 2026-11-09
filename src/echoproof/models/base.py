"""Shared inference-facing model interface (the R0 "model adapter API").

This is the single contract that production models implement and that ``inference``,
the CLI and the app consume:

- ``load(bundle_dir)`` receives the root directory of a packaged model bundle (the
  directory holding ``bundle.json``). The adapter resolves its own components inside it,
  verifies them, and must not load executable pickle artifacts. Adapters should also
  accept a bare model-component directory so that pre-bundle runs remain loadable.
- ``score_chunks(waveforms)`` receives mono, 16 kHz floating-point waveforms containing
  valid samples only and returns one P(TTS) probability per waveform, in input order, as a
  one-dimensional array of finite values in [0, 1]. An empty list returns an empty array.
  Use :func:`validate_scores` to enforce this on any model output.
- ``metadata`` exposes read-only descriptive information as string-to-string pairs.

Relationship to ``echoproof.contracts.ModelAdapter``: that older protocol works on prepared
``FeatureBatch`` objects and returns ``ChunkScore`` rows. It describes the disposable P1
software-probe fixtures (``models/lr.py``, ``models/bigru.py``) and training-side feature
preparation only. Production inference does not use it. Callers that need ``ChunkScore``
rows pair the returned array with chunk IDs by position and call
``contracts.validate_chunk_scores``.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import numpy as np


BundlePath = str | PathLike[str] | Path


@runtime_checkable
class InferenceModelAdapter(Protocol):
    """Structural interface for models that score canonical audio chunks."""

    @classmethod
    def load(cls, bundle_dir: BundlePath) -> InferenceModelAdapter:
        """Load the model from a bundle root directory.

        Implementations resolve their own model components within
        ``bundle_dir`` and must not load executable pickle artifacts.
        """
        ...

    def score_chunks(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Return one P(TTS) probability per input waveform.

        Each waveform is a one-dimensional, mono, 16 kHz floating-point
        NumPy array containing only valid audio samples. The returned array
        must have shape ``(len(waveforms),)`` and contain finite values in
        the inclusive range [0, 1].
        """
        ...

    @property
    def metadata(self) -> Mapping[str, str]:
        """Return read-only descriptive metadata about the loaded model."""
        ...


def validate_scores(scores, expected_count: int) -> np.ndarray:
    """Return ``scores`` as a float64 vector after enforcing the adapter output contract.

    Raises ``ValueError`` if the shape is not ``(expected_count,)`` or any value is
    non-finite or outside [0, 1].
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.shape != (expected_count,):
        raise ValueError(f"expected {expected_count} scores in a 1-D array, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("scores must be finite")
    if (array < 0.0).any() or (array > 1.0).any():
        raise ValueError("scores must lie in [0, 1]")
    return array
