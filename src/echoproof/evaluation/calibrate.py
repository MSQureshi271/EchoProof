"""Recording-level temperature scaling calibration for EchoProof R1-N1 evaluation.

Calibrates continuous probabilities by scaling log-odds by a positive temperature T:
    calibrated = sigmoid(logit(clip(score, eps, 1 - eps)) / T)

Ranking is strictly preserved under T > 0.
Minimizes cross-entropy (log loss) on calibration validation.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from echoproof.contracts import ContractError

DEFAULT_EPSILON = 1e-7


def _validate_inputs(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(labels) != len(scores):
        raise ContractError(f"labels and scores length mismatch: {len(labels)} != {len(scores)}")
    if len(labels) == 0:
        raise ContractError("Empty inputs to calibration")

    clean_labels = []
    clean_scores = []
    for idx, (lbl, sc) in enumerate(zip(labels, scores)):
        if isinstance(lbl, bool) or not isinstance(lbl, Integral):
            raise ContractError(f"Invalid label at index {idx}: expected 0 or 1, received {lbl!r}")
        l_int = int(lbl)
        if l_int not in (0, 1):
            raise ContractError(f"Invalid label at index {idx}: expected 0 or 1, received {lbl!r}")
        clean_labels.append(l_int)

        if isinstance(sc, bool) or not isinstance(sc, Real):
            raise ContractError(f"Invalid score at index {idx}: expected numeric probability, received {sc!r}")
        s_flt = float(sc)
        if not math.isfinite(s_flt):
            raise ContractError(f"Non-finite score at index {idx}: {sc!r}")
        if not 0.0 <= s_flt <= 1.0:
            raise ContractError(f"Score outside [0, 1] at index {idx}: {sc}")
        clean_scores.append(s_flt)

    arr_labels = np.asarray(clean_labels, dtype=np.int64)
    arr_scores = np.asarray(clean_scores, dtype=np.float64)

    if len(np.unique(arr_labels)) < 2:
        raise ContractError("Both genuine (0) and TTS (1) classes must be present for calibration")

    return arr_labels, arr_scores


def apply_temperature(
    scores: Sequence[float],
    T: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> list[float]:
    """Apply temperature scaling to raw scores.

    Args:
        scores: Sequence of probabilities in [0, 1].
        T: Positive temperature parameter.
        epsilon: Boundary clipping threshold.

    Returns:
        List of calibrated float probabilities.

    Raises:
        ContractError: If T <= 0, non-finite, or scores are invalid.
    """
    if isinstance(T, bool) or not isinstance(T, Real):
        raise ContractError(f"Temperature T must be a real number, received {T!r}")
    t_flt = float(T)
    if not math.isfinite(t_flt) or t_flt <= 0.0:
        raise ContractError(f"Temperature T must be positive finite, received {T!r}")

    calibrated: list[float] = []
    for idx, sc in enumerate(scores):
        if isinstance(sc, bool) or not isinstance(sc, Real):
            raise ContractError(f"Invalid score at index {idx}: {sc!r}")
        s_flt = float(sc)
        if not math.isfinite(s_flt) or not 0.0 <= s_flt <= 1.0:
            raise ContractError(f"Score outside [0, 1] or non-finite at index {idx}: {sc}")

        s_clip = min(max(s_flt, epsilon), 1.0 - epsilon)
        logit = math.log(s_clip / (1.0 - s_clip))
        scaled_logit = logit / t_flt

        if scaled_logit >= 0:
            prob = 1.0 / (1.0 + math.exp(-scaled_logit))
        else:
            exp_z = math.exp(scaled_logit)
            prob = exp_z / (1.0 + exp_z)

        calibrated.append(float(prob))

    return calibrated


def fit_temperature(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    epsilon: float = DEFAULT_EPSILON,
    bounds: tuple[float, float] = (0.01, 50.0),
) -> float:
    """Fit temperature parameter T > 0 by minimizing log loss.

    Args:
        labels: Binary ground-truth labels (0 = genuine, 1 = TTS).
        scores: Continuous predicted TTS probabilities in [0, 1].
        epsilon: Boundary clipping threshold.
        bounds: 1-D search interval (lower, upper).

    Returns:
        Optimal temperature T as a float.

    Raises:
        ContractError: If inputs are invalid, non-finite, or lack both classes.
    """
    y, s = _validate_inputs(labels, scores)

    s_clip = np.clip(s, epsilon, 1.0 - epsilon)
    logits = np.log(s_clip / (1.0 - s_clip))

    # Cross-entropy objective: softplus((1 - 2y) * logits / T)
    margin_mult = (1.0 - 2.0 * y)

    def loss_func(t_val: float) -> float:
        if t_val <= 0:
            return float("inf")
        z = margin_mult * (logits / t_val)
        # Numerically stable softplus
        loss = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
        return float(np.mean(loss))

    # Try scipy bounded search first
    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(loss_func, bounds=bounds, method="bounded")
        if res.success and res.x > 0:
            return float(res.x)
    except ImportError:
        pass

    # Golden-section search fallback
    a, b = bounds
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.6180339887
    inv_phi_sq = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.3819660113

    h = b - a
    c = a + inv_phi_sq * h
    d = a + inv_phi * h
    yc = loss_func(c)
    yd = loss_func(d)

    for _ in range(80):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = inv_phi * h
            c = a + inv_phi_sq * h
            yc = loss_func(c)
        else:
            a = c
            c = d
            yc = yd
            h = inv_phi * h
            d = a + inv_phi * h
            yd = loss_func(d)

    return float((a + b) / 2.0)

