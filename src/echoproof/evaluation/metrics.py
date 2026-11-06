"""Authoritative evaluation metrics for EchoProof R1-N1 recording-level evaluation.

Positive class: 1 (TTS).
Negative class: 0 (Genuine).
Higher scores indicate higher likelihood of TTS.
Decisions are binary: predict TTS if score >= threshold.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np

from echoproof.contracts import ContractError

DEFAULT_EPSILON = 1e-7


def _validate_inputs(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(labels) != len(scores):
        raise ContractError(f"labels and scores length mismatch: {len(labels)} != {len(scores)}")
    if len(labels) == 0:
        raise ContractError("Empty inputs to evaluation metrics")

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

    classes = np.unique(arr_labels)
    if len(classes) < 2:
        raise ContractError("Both genuine (0) and TTS (1) classes must be present")

    return arr_labels, arr_scores


def interpolated_eer(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute Equal Error Rate (EER) with linear ROC interpolation.

    EER is the operating point where False Positive Rate (FPR) equals False Negative Rate (FNR).
    Higher scores indicate TTS (class 1).

    Args:
        labels: Binary ground-truth labels (0 = genuine, 1 = TTS).
        scores: Continuous predicted TTS probabilities in [0, 1].

    Returns:
        Interpolated EER as a float in [0, 1].

    Raises:
        ContractError: If inputs are invalid, non-finite, out of range, or lack both classes.
    """
    y, s = _validate_inputs(labels, scores)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    assert pos > 0 and neg > 0

    order = np.argsort(-s, kind="mergesort")
    sorted_s = s[order]
    sorted_y = y[order]

    distinct_indices = np.r_[np.where(np.diff(sorted_s))[0], len(sorted_s) - 1]

    tpr = np.r_[0.0, np.cumsum(sorted_y == 1)[distinct_indices] / pos]
    fpr = np.r_[0.0, np.cumsum(sorted_y == 0)[distinct_indices] / neg]
    fnr = 1.0 - tpr
    d = fnr - fpr

    crossing_idx = int(np.argmax(d <= 0))
    if d[crossing_idx] == 0:
        return float(fpr[crossing_idx])

    d_prev = d[crossing_idx - 1]
    d_curr = d[crossing_idx]
    t = d_prev / (d_prev - d_curr)
    eer_val = fpr[crossing_idx - 1] + t * (fpr[crossing_idx] - fpr[crossing_idx - 1])
    return float(np.clip(eer_val, 0.0, 1.0))


def _compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        raise ContractError("Both classes required for ROC-AUC")

    greater = np.sum(pos_scores[:, None] > neg_scores[None, :])
    equal = np.sum(pos_scores[:, None] == neg_scores[None, :])
    auc = (greater + 0.5 * equal) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    """Compute comprehensive recording-level evaluation metrics.

    Args:
        labels: Binary ground-truth labels (0 = genuine, 1 = TTS).
        scores: Continuous predicted TTS probabilities in [0, 1].
        threshold: Decision threshold. Predict TTS if score >= threshold.
        epsilon: Minimum distance from 0 and 1 for log loss clipping.

    Returns:
        Dictionary containing all contract-required metrics.

    Raises:
        ContractError: If inputs are malformed, non-finite, out of range, or lack both classes.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ContractError(f"Invalid threshold: expected real number, received {threshold!r}")
    t_flt = float(threshold)
    if not math.isfinite(t_flt):
        raise ContractError(f"Non-finite threshold: {threshold!r}")
    if not 0.0 <= t_flt <= 1.0:
        raise ContractError(f"Threshold outside [0, 1]: {threshold}")

    y, s = _validate_inputs(labels, scores)
    n = len(y)

    pred_tts = s >= t_flt
    tp = int(np.sum(pred_tts & (y == 1)))
    fp = int(np.sum(pred_tts & (y == 0)))
    tn = int(np.sum((~pred_tts) & (y == 0)))
    fn = int(np.sum((~pred_tts) & (y == 1)))

    confusion = [[tn, fp], [fn, tp]]
    accuracy = (tp + tn) / n

    tts_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    tts_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    gen_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1_tts = (2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    f1_gen = (2.0 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) > 0 else 0.0
    macro_f1 = (f1_tts + f1_gen) / 2.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    eer_val = interpolated_eer(y, s)
    roc_auc_val = _compute_roc_auc(y, s)

    try:
        from sklearn.metrics import average_precision_score
        avg_precision = float(average_precision_score(y, s))
    except ImportError:
        order = np.argsort(-s, kind="mergesort")
        y_sorted = y[order]
        cum_tp = np.cumsum(y_sorted == 1)
        prec_curve = cum_tp / np.arange(1, n + 1)
        avg_precision = float(np.sum(prec_curve * (y_sorted == 1)) / (y == 1).sum())

    brier = float(np.mean((s - y) ** 2))

    s_clipped = np.clip(s, epsilon, 1.0 - epsilon)
    log_loss_val = float(-np.mean(y * np.log(s_clipped) + (1.0 - y) * np.log(1.0 - s_clipped)))

    return {
        "accuracy": accuracy,
        "tts_precision": tts_prec,
        "tts_recall": tts_rec,
        "genuine_recall": gen_rec,
        "macro_f1": macro_f1,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc_val,
        "eer": eer_val,
        "average_precision": avg_precision,
        "confusion_matrix": confusion,
        "brier_score": brier,
        "log_loss": log_loss_val,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recordings": n,
        "threshold": t_flt,
    }

