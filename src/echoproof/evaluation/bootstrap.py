"""Paired group-level bootstrap for EchoProof R1-N1 evaluation.

Resamples whole groups (speakers/utterance clusters) with replacement to account for
intra-group correlation. Evaluates all compared models on the exact same resample
to preserve pairing.
"""
from __future__ import annotations

import itertools
import math
from numbers import Integral, Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from echoproof.contracts import ContractError


def _check_identity(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    if value.lower() in {"unknown", "none", "null"} or ":" not in value:
        raise ContractError(f"{name} must be a namespaced identity")
    namespace, identifier = value.split(":", 1)
    if not namespace.strip() or not identifier.strip() or identifier.strip().lower() in {"unknown", "none", "null"}:
        raise ContractError(f"{name} must be a namespaced identity")


def paired_group_bootstrap(
    group_ids: Sequence[str],
    labels: Sequence[int],
    scores_by_model: Mapping[str, Sequence[float]],
    metric_fn: Callable[[Sequence[int], Sequence[float]], float],
    *,
    n_resamples: int = 1000,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Perform paired, group-aware bootstrap confidence intervals for model comparison.

    Args:
        group_ids: Group identifier (e.g. speaker) for each recording.
        labels: Binary ground-truth labels (0 = genuine, 1 = TTS) for each recording.
        scores_by_model: Mapping from model name to sequence of recording-level probabilities.
        metric_fn: Callable computing scalar evaluation metric: metric_fn(labels, scores) -> float.
        n_resamples: Number of bootstrap iterations.
        seed: Random seed for exact reproducibility.
        alpha: Significance level for 100*(1-alpha)% percentile confidence intervals.

    Returns:
        Dictionary containing per-model estimates and CIs, and pairwise difference CIs.

    Raises:
        ContractError: If inputs are invalid, mismatched, fewer than 2 groups, or lack both classes.
    """
    n = len(labels)
    if len(group_ids) != n:
        raise ContractError(f"group_ids length {len(group_ids)} != labels length {n}")
    if n == 0:
        raise ContractError("Empty dataset provided to paired group bootstrap")

    if not isinstance(scores_by_model, Mapping) or len(scores_by_model) == 0:
        raise ContractError("scores_by_model must be a non-empty mapping")

    for model_name, model_scores in scores_by_model.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ContractError(f"Invalid model name: {model_name!r}")
        if len(model_scores) != n:
            raise ContractError(
                f"Model {model_name!r} scores length {len(model_scores)} != labels length {n}"
            )

    clean_labels = []
    for idx, lbl in enumerate(labels):
        if isinstance(lbl, bool) or not isinstance(lbl, Integral):
            raise ContractError(f"Invalid label at index {idx}: expected 0 or 1, received {lbl!r}")
        l_int = int(lbl)
        if l_int not in (0, 1):
            raise ContractError(f"Invalid label at index {idx}: expected 0 or 1, received {lbl!r}")
        clean_labels.append(l_int)

    for idx, gid in enumerate(group_ids):
        _check_identity(gid, f"group_id at index {idx}")

    for model_name, model_scores in scores_by_model.items():
        for idx, sc in enumerate(model_scores):
            if isinstance(sc, bool) or not isinstance(sc, Real):
                raise ContractError(f"Invalid score for model {model_name} at index {idx}: {sc!r}")
            s_flt = float(sc)
            if not math.isfinite(s_flt) or not 0.0 <= s_flt <= 1.0:
                raise ContractError(f"Score out of bounds or non-finite for model {model_name} at index {idx}: {sc}")

    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ContractError("Need at least 2 groups for paired group bootstrap")

    if len(set(clean_labels)) < 2:
        raise ContractError("Both genuine (0) and TTS (1) classes must be present")

    if isinstance(n_resamples, bool) or not isinstance(n_resamples, Integral) or n_resamples < 1:
        raise ContractError(f"n_resamples must be integer >= 1, received {n_resamples!r}")

    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ContractError(f"seed must be non-negative integer, received {seed!r}")

    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0.0 < float(alpha) < 1.0:
        raise ContractError(f"alpha must be in (0, 1), received {alpha!r}")

    alpha_flt = float(alpha)
    n_res_int = int(n_resamples)
    seed_int = int(seed)

    group_to_indices: dict[str, list[int]] = {g: [] for g in unique_groups}
    for idx, gid in enumerate(group_ids):
        group_to_indices[gid].append(idx)

    model_names = sorted(scores_by_model.keys())
    model_pairs = list(itertools.combinations(model_names, 2))

    base_estimates: dict[str, float] = {}
    for m in model_names:
        base_estimates[m] = float(metric_fn(clean_labels, scores_by_model[m]))

    rng = np.random.RandomState(seed_int)
    num_groups = len(unique_groups)

    replicates: dict[str, list[float]] = {m: [] for m in model_names}
    pairwise_replicates: dict[tuple[str, str], list[float]] = {pair: [] for pair in model_pairs}
    skipped_single_class = 0

    for _ in range(n_res_int):
        sampled_group_indices = rng.choice(num_groups, size=num_groups, replace=True)
        sampled_rec_indices: list[int] = []
        for g_idx in sampled_group_indices:
            sampled_rec_indices.extend(group_to_indices[unique_groups[g_idx]])

        sub_labels = [clean_labels[i] for i in sampled_rec_indices]
        if len(set(sub_labels)) < 2:
            skipped_single_class += 1
            continue

        step_metrics: dict[str, float] = {}
        for m in model_names:
            sub_scores = [scores_by_model[m][i] for i in sampled_rec_indices]
            val = float(metric_fn(sub_labels, sub_scores))
            step_metrics[m] = val
            replicates[m].append(val)

        for (m1, m2) in model_pairs:
            diff = step_metrics[m1] - step_metrics[m2]
            pairwise_replicates[(m1, m2)].append(diff)

    valid_resamples = len(replicates[model_names[0]])
    if valid_resamples == 0:
        raise ContractError("No resamples contained both classes; cannot compute bootstrap confidence intervals")

    lower_pct = 100.0 * (alpha_flt / 2.0)
    upper_pct = 100.0 * (1.0 - alpha_flt / 2.0)

    models_output: dict[str, dict[str, float]] = {}
    for m in model_names:
        arr = np.asarray(replicates[m], dtype=np.float64)
        models_output[m] = {
            "estimate": base_estimates[m],
            "ci_lower": float(np.percentile(arr, lower_pct)),
            "ci_upper": float(np.percentile(arr, upper_pct)),
        }

    diffs_output: dict[str, dict[str, float]] = {}
    for (m1, m2) in model_pairs:
        arr_diff = np.asarray(pairwise_replicates[(m1, m2)], dtype=np.float64)
        diff_key = f"{m1}_vs_{m2}"
        diffs_output[diff_key] = {
            "estimate": base_estimates[m1] - base_estimates[m2],
            "ci_lower": float(np.percentile(arr_diff, lower_pct)),
            "ci_upper": float(np.percentile(arr_diff, upper_pct)),
        }

    return {
        "models": models_output,
        "differences": diffs_output,
        "valid_resamples": valid_resamples,
        "skipped_single_class": skipped_single_class,
        "n_resamples": n_res_int,
        "alpha": alpha_flt,
    }

