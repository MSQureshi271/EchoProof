"""Shared inference-facing model interface.

R0 contract:
- ``load(bundle_dir)`` receives the root directory of a packaged model bundle.
- ``score_chunks(waveforms)`` receives mono, 16 kHz floating-point audio
  waveforms containing valid samples only and returns one P(TTS) probability
  for each waveform.
- ``metadata`` exposes read-only model information.

A concrete adapter is responsible for resolving its model files inside the
bundle directory. For example, LRDetector.load(directory) currently expects
the *model component directory* containing weights.npz and lr_model.json;
callers must resolve that directory from the bundle root before invoking it.
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
