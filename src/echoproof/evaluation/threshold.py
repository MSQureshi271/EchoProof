"""Operating threshold selection for EchoProof R1-N1 evaluation.

Evaluates every distinct score-induced decision partition on operating validation,
plus the 'predict nobody TTS' (all-negative) partition.
Selects the rule that maximizes macro F1, tie-breaking according to the research contract:
lower genuine FPR, then proximity to 0.5, then smaller threshold.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from echoproof.contracts import ContractError, DecisionRule


def _validate_inputs(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(labels) != len(scores):
        raise ContractError(f"labels and scores length mismatch: {len(labels)} != {len(scores)}")
    if len(labels) == 0:
        raise ContractError("Cannot select threshold on empty inputs")

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
        raise ContractError("Both genuine (0) and TTS (1) classes must be present to select a threshold")

    return arr_labels, arr_scores


def select_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    tie_break: str = "contract",
) -> tuple[DecisionRule, float, float]:
    """Select the operating threshold maximizing macro F1 across all score-induced partitions.

    Args:
        labels: Binary ground-truth labels (0 = genuine, 1 = TTS).
        scores: Continuous predicted TTS probabilities in [0, 1].
        tie_break: Tie-breaking strategy. Defaults to 'contract' (lower FPR -> proximity to 0.5 -> smaller threshold).
            Alternative 'prompt' (lower FPR -> higher threshold).

    Returns:
        Tuple of (chosen DecisionRule, macro F1, genuine FPR).

    Raises:
        ContractError: If inputs are invalid, non-finite, out of bounds, or lack both classes.
    """
    y, s = _validate_inputs(labels, scores)

    candidates: list[dict] = []

    distinct_scores = np.unique(s)
    for u in distinct_scores:
        pred_tts = s >= u
        tp = int(np.sum(pred_tts & (y == 1)))
        fp = int(np.sum(pred_tts & (y == 0)))
        tn = int(np.sum((~pred_tts) & (y == 0)))
        fn = int(np.sum((~pred_tts) & (y == 1)))

        f1_tts = (2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        f1_gen = (2.0 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) > 0 else 0.0
        macro_f1 = (f1_tts + f1_gen) / 2.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        candidates.append({
            "kind": "threshold",
            "threshold": float(u),
            "macro_f1": macro_f1,
            "fpr": fpr,
            "dist_05": abs(float(u) - 0.5),
        })

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    all_neg_f1_gen = (2.0 * n_neg) / (2.0 * n_neg + n_pos) if (2 * n_neg + n_pos) > 0 else 0.0
    all_neg_macro_f1 = all_neg_f1_gen / 2.0
    candidates.append({
        "kind": "all_negative",
        "threshold": None,
        "macro_f1": all_neg_macro_f1,
        "fpr": 0.0,
        "dist_05": 1.0,
    })

    TOL = 1e-12

    def sort_key(c: dict):
        f1 = round(c["macro_f1"] / TOL) * TOL
        fpr = round(c["fpr"] / TOL) * TOL
        if tie_break == "prompt":
            t_val = c["threshold"] if c["threshold"] is not None else -1.0
            return (f1, -fpr, t_val)
        dist = round(c["dist_05"] / TOL) * TOL
        t_val = -c["threshold"] if c["threshold"] is not None else -2.0
        return (f1, -fpr, -dist, t_val)

    best = max(candidates, key=sort_key)

    if best["kind"] == "all_negative":
        rule = DecisionRule(schema_version=1, kind="all_negative", threshold=None)
    else:
        assert best["threshold"] is not None and best["threshold"] <= 1.0
        rule = DecisionRule(schema_version=1, kind="threshold", threshold=best["threshold"])

    return rule, float(best["macro_f1"]), float(best["fpr"])

