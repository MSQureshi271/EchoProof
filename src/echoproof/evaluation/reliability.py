"""Reliability bins and expected calibration error for recording-level scores (numpy only).

Equal-width bins on [0, 1]; a score of exactly 1.0 falls in the last bin, so every score in [0, 1] is counted once. Empty bins are reported with
count 0 and no means, and are left out of plots and of the error. ECE is the count-weighted mean of |observed TTS rate - mean score| over the
non-empty bins. It depends on the binning and must not be used alone to choose a method.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from echoproof.contracts import ContractError


def _validate(labels: Sequence[int], scores: Sequence[float], n_bins: int):
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 1:
        raise ContractError(f"n_bins must be a positive integer, received {n_bins!r}")
    if len(labels) != len(scores):
        raise ContractError(f"labels and scores length mismatch: {len(labels)} != {len(scores)}")
    if len(labels) == 0:
        raise ContractError("empty inputs to reliability")
    y = []
    for index, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, Integral) or int(label) not in (0, 1):
            raise ContractError(f"invalid label at index {index}: {label!r}")
        y.append(int(label))
    s = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise ContractError(f"invalid score at index {index}: {score!r}")
        s.append(float(score))
    return np.asarray(y, dtype=np.int64), np.asarray(s, dtype=np.float64)


def reliability_bins(labels: Sequence[int], scores: Sequence[float], n_bins: int = 10) -> list[dict]:
    """One dict per bin: index, bounds, count, mean score and observed TTS rate (both None when the bin is empty)."""
    y, s = _validate(labels, scores, n_bins)
    index = np.minimum((s * n_bins).astype(np.int64), n_bins - 1)
    bins = []
    for i in range(n_bins):
        mask = index == i
        count = int(mask.sum())
        bins.append({"bin": i, "lower": i / n_bins, "upper": (i + 1) / n_bins, "count": count,
                     "mean_score": float(s[mask].mean()) if count else None,
                     "observed_rate": float(y[mask].mean()) if count else None})
    return bins


def expected_calibration_error(labels: Sequence[int], scores: Sequence[float], n_bins: int = 10) -> float:
    bins = reliability_bins(labels, scores, n_bins)
    total = sum(b["count"] for b in bins)
    return float(sum(b["count"] * abs(b["observed_rate"] - b["mean_score"]) for b in bins if b["count"]) / total)
