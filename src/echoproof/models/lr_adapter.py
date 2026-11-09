"""Bundle-aware adapter that makes the production LR conform to ``models.base``.

``LRDetector`` (the locked R2-M1 model) loads from its model-component directory and returns raw
arrays. This wrapper adds the R0 contract on top without changing it: ``load`` takes the bundle
root and verifies component checksums through ``echoproof.bundle``, empty input is handled, and
every output is checked to be a finite vector in [0, 1]. Scores are identical to ``LRDetector``.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from echoproof.models.base import BundlePath, validate_scores
from echoproof.models.lr_detector import LRDetector

_COMPONENTS = ("weights.npz", "lr_model.json")


def _has_components(directory: Path) -> bool:
    return all((directory / name).is_file() for name in _COMPONENTS)


class LRBundleAdapter:
    """``InferenceModelAdapter`` implementation for the production logistic-regression detector."""

    def __init__(self, detector: LRDetector, bundle_info: Mapping[str, str] | None = None):
        self._detector = detector
        self._bundle_info = dict(bundle_info or {})

    @classmethod
    def load(cls, bundle_dir: BundlePath) -> "LRBundleAdapter":
        """Load from a verified bundle root, or from a bare model-component directory.

        A directory containing ``bundle.json`` is verified (schema and SHA-256 of every component)
        before any model file is read. Otherwise the directory, or its ``model/`` subdirectory, must
        hold ``weights.npz`` and ``lr_model.json``.
        """
        path = Path(bundle_dir)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {bundle_dir}")
        info: dict[str, str] = {}
        if (path / "bundle.json").is_file():
            from echoproof.bundle import load_bundle

            validated = load_bundle(path)
            by_name = {rel: p for rel, p in validated.components.items() if Path(rel).name in _COMPONENTS}
            parents = {p.parent for p in by_name.values()}
            if len(by_name) != len(_COMPONENTS) or len(parents) != 1:
                raise FileNotFoundError(f"Verified bundle {bundle_dir} does not contain LR model components")
            model_dir = parents.pop()
            info = {"bundle_id": validated.spec.bundle_id, "model_version": validated.spec.model_version}
        elif _has_components(path):
            model_dir = path
        elif _has_components(path / "model"):
            model_dir = path / "model"
        else:
            raise FileNotFoundError(f"No valid LR model found at {bundle_dir}")
        return cls(LRDetector.load(model_dir), info)

    def score_chunks(self, waveforms: list[np.ndarray]) -> np.ndarray:
        count = len(waveforms)
        if count == 0:
            return np.empty(0, dtype=np.float64)
        return validate_scores(self._detector.score_chunks(waveforms), count)

    @property
    def metadata(self) -> Mapping[str, str]:
        return MappingProxyType({**{str(k): str(v) for k, v in self._detector.metadata.items()}, **self._bundle_info})
