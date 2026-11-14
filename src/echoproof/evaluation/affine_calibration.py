"""Positive-slope affine calibration of recording log-odds (R3-N1, WavLM).

q = sigmoid(a * logit(clip(p, eps, 1 - eps)) + b), a = exp(u) > 0, so ranking is preserved but, unlike temperature scaling, the 0.5 decision boundary
moves to z = -b / a when b is nonzero. The fit minimises mean log loss plus `reg * (u^2 + b^2)` (a pull towards identity, a = 1 and b = 0) with
equal weight per recording, over bounded (u, b). `fit_affine` never raises on a failed optimisation; it reports `ok = False` and the reason so a
caller can exclude the candidate and record why.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from scipy.special import expit

from echoproof.contracts import ContractError

DEFAULT_EPSILON = 1e-7
U_BOUNDS = (-4.0, 4.0)
B_BOUNDS = (-5.0, 5.0)
REGULARISATION = 0.01
BOUND_TOLERANCE = 1e-6


def _validate(labels: Sequence[int], scores: Sequence[float]):
    if len(labels) != len(scores):
        raise ContractError(f"labels and scores length mismatch: {len(labels)} != {len(scores)}")
    if len(labels) == 0:
        raise ContractError("empty inputs to calibration")
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
    y, s = np.asarray(y, dtype=np.int64), np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ContractError("both genuine (0) and TTS (1) classes must be present for calibration")
    return y, s


def _logits(scores: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def apply_affine(scores: Sequence[float], a: float, b: float, *, epsilon: float = DEFAULT_EPSILON) -> list[float]:
    """Same clipping and the same numerically stable sigmoid as `calibrate.apply_temperature`, one score at a time."""
    for name, value in (("a", a), ("b", b)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ContractError(f"affine parameter {name} must be a finite number, received {value!r}")
    if float(a) <= 0.0:
        raise ContractError(f"affine slope a must be positive, received {a!r}")
    calibrated = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise ContractError(f"score outside [0, 1] or non-finite at index {index}: {score!r}")
        clipped = min(max(float(score), epsilon), 1.0 - epsilon)
        z = float(a) * math.log(clipped / (1.0 - clipped)) + float(b)
        if z >= 0:
            prob = 1.0 / (1.0 + math.exp(-z))
        else:
            e = math.exp(z)
            prob = e / (1.0 + e)
        calibrated.append(float(prob))
    return calibrated


def _objective(params, z, m, reg):
    u, b = params
    a = math.exp(u)
    s = a * z + b
    loss = float(np.mean(np.logaddexp(0.0, m * s)) + reg * (u * u + b * b))
    sigma = expit(m * s)                           # d/ds log(1 + exp(m s)) = m * sigmoid(m s)
    w = m * sigma
    grad_u = float(np.mean(w * z) * a + 2.0 * reg * u)
    grad_b = float(np.mean(w) + 2.0 * reg * b)
    return loss, np.array([grad_u, grad_b])


def fit_affine(labels: Sequence[int], scores: Sequence[float], *, epsilon: float = DEFAULT_EPSILON, u_bounds=U_BOUNDS, b_bounds=B_BOUNDS,
               reg: float = REGULARISATION, start=(0.0, 0.0)) -> dict:
    """Fit (u, b). Returns a dict with a, u, b, ok, failure_reason and diagnostics; it does not raise for an optimiser failure."""
    from scipy.optimize import minimize

    y, s = _validate(labels, scores)
    z = _logits(s, epsilon)
    m = (1.0 - 2.0 * y).astype(np.float64)
    identity_loss, _ = _objective((0.0, 0.0), z, m, reg)
    result = minimize(lambda p: _objective(p, z, m, reg), np.asarray(start, dtype=np.float64), jac=True, method="L-BFGS-B",
                      bounds=[tuple(u_bounds), tuple(b_bounds)], options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
    u, b = float(result.x[0]), float(result.x[1])
    at_bound = {"u": min(abs(u - u_bounds[0]), abs(u - u_bounds[1])) < BOUND_TOLERANCE, "b": min(abs(b - b_bounds[0]), abs(b - b_bounds[1])) < BOUND_TOLERANCE}
    reason = None
    if not result.success:
        reason = f"optimizer did not converge: {result.message}"
    elif not (math.isfinite(u) and math.isfinite(b) and math.isfinite(float(result.fun))):
        reason = "non-finite parameters or objective"
    elif at_bound["u"] or at_bound["b"]:
        reason = "a parameter ended within 1e-6 of its bound"
    plain_loss = float(np.mean(np.logaddexp(0.0, m * (math.exp(u) * z + b)))) if reason is None or math.isfinite(u + b) else None
    return {"a": math.exp(u) if math.isfinite(u) else None, "u": u, "b": b, "ok": reason is None, "failure_reason": reason, "at_bound": at_bound,
            "objective_at_fit": float(result.fun), "objective_at_identity": identity_loss, "log_loss_at_fit": plain_loss,
            "optimizer_message": str(result.message), "iterations": int(result.nit), "epsilon": epsilon, "regularisation": reg,
            "u_bounds": list(u_bounds), "b_bounds": list(b_bounds), "recordings": int(len(y))}
