"""Clean R5 analysis of validated, locked, recording-level predictions only."""
from __future__ import annotations

from itertools import combinations
import math
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from echoproof.evaluation.metrics import compute_metrics
from echoproof.evaluation.reliability import reliability_bins, expected_calibration_error

MODELS = ("LR", "BiGRU", "WavLM", "AST")
PANELS = {"A_known": ("A", {"genuine", "melo", "vits", "xtts", "chatterbox"}),
          "B_known": ("B", {"genuine", "melo", "vits"}),
          "B_unseen": ("B", {"genuine", "chatterbox"})}


class AnalysisError(ValueError):
    pass


def validate_predictions(rows_by_key, entries, expected_counts):
    """Check all cohorts and score records before returning any metrics."""
    expected = {f"{p}:{m}" for p in ("A", "B") for m in MODELS}
    if set(rows_by_key) != expected or set(entries) != expected:
        raise AnalysisError("exactly eight locked models are required")
    by_protocol = {}
    for protocol in ("A", "B"):
        reference = None
        for model in MODELS:
            key = f"{protocol}:{model}"
            rows = rows_by_key[key]
            if len(rows) != expected_counts[protocol]:
                raise AnalysisError(f"{key}: unexpected denominator")
            threshold = entries[key]["decision_rule"]["threshold"]
            if not isinstance(threshold, (float, int)) or isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise AnalysisError(f"{key}: invalid threshold")
            indexed = {}
            score_types = set()
            for row in rows:
                rid, component, group = row.get("recording_id"), row.get("component"), row.get("group_id")
                if not isinstance(rid, str) or len(rid.split(":")) != 3 or rid.split(":")[1] != component or not isinstance(group, str) or not group.startswith("speaker:"):
                    raise AnalysisError(f"{key}: invalid recording metadata")
                if component not in PANELS["A_known"][1] or (protocol == "B" and component == "xtts"):
                    raise AnalysisError(f"{key}: unexpected component")
                if row.get("label") != (0 if component == "genuine" else 1) or group != "speaker:" + rid.split(":")[2].split("-")[0]:
                    raise AnalysisError(f"{key}: inconsistent label or speaker")
                if rid in indexed or row.get("status") != "success" or row.get("failure_reason"):
                    raise AnalysisError(f"{key}: duplicate or failed prediction")
                score = row.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise AnalysisError(f"{key}: invalid score")
                decision = "synthetic" if score >= threshold else "genuine"
                if row.get("assessment") != decision or row.get("model_version") != entries[key]["model_version"]:
                    raise AnalysisError(f"{key}: model or locked decision mismatch")
                if row.get("score_type") not in ("model_score", "calibrated_probability"):
                    raise AnalysisError(f"{key}: invalid score type")
                score_types.add(row["score_type"])
                indexed[rid] = row
            if len(score_types) != 1:
                raise AnalysisError(f"{key}: inconsistent score types")
            identity = {rid: (r["group_id"], r["component"], r["label"]) for rid, r in indexed.items()}
            if reference is not None and identity != reference:
                raise AnalysisError(f"{protocol}: models have different recording identities or labels")
            reference = identity
            by_protocol[key] = indexed
        present = {v[1] for v in reference.values()}
        expected_components = PANELS["A_known" if protocol == "A" else "B_known"][1] | ({"chatterbox"} if protocol == "B" else set())
        if present != expected_components:
            raise AnalysisError(f"{protocol}: incomplete component coverage")
    # B is a strict component subset of A for the same held-out speakers.
    a, b = by_protocol["A:LR"], by_protocol["B:LR"]
    if set(b) != {rid for rid, r in a.items() if r["component"] != "xtts"}:
        raise AnalysisError("A/B test recording identities differ")
    return by_protocol


def _scalar_metrics(y, s, threshold):
    positive = y == 1
    prediction = s >= threshold
    tp = np.count_nonzero(prediction & positive)
    fp = np.count_nonzero(prediction & ~positive)
    fn = np.count_nonzero(~prediction & positive)
    tn = np.count_nonzero(~prediction & ~positive)
    f1pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    ranks = rankdata(s, method="average")
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    auc = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"macro_f1": float((f1pos + f1neg) / 2), "fpr": float(fp / (fp + tn)),
            "tts_recall": float(tp / (tp + fn)), "roc_auc": float(auc)}


def paired_bootstrap(groups, labels, scores, thresholds, *, n_resamples=1000, seed=42):
    """Paired speaker bootstrap, applying each model's own fixed threshold."""
    if n_resamples < 1 or seed < 0:
        raise AnalysisError("invalid bootstrap settings")
    groups, labels = np.asarray(groups), np.asarray(labels, dtype=int)
    group_names = sorted(set(groups.tolist()))
    if len(group_names) < 2 or set(labels.tolist()) != {0, 1}:
        raise AnalysisError("bootstrap requires at least two speakers and both classes")
    arrays = {m: np.asarray(scores[m], dtype=float) for m in MODELS}
    index = [np.flatnonzero(groups == group) for group in group_names]
    point = {m: _scalar_metrics(labels, arrays[m], thresholds[m]) for m in MODELS}
    rng = np.random.RandomState(seed)
    samples = {m: {metric: [] for metric in point[m]} for m in MODELS}
    skipped = 0
    for _ in range(n_resamples):
        chosen = rng.randint(0, len(index), size=len(index))
        indices = np.concatenate([index[i] for i in chosen])
        y = labels[indices]
        if len(np.unique(y)) != 2:
            skipped += 1
            continue
        for model in MODELS:
            values = _scalar_metrics(y, arrays[model][indices], thresholds[model])
            for metric, value in values.items():
                samples[model][metric].append(value)
    if skipped == n_resamples:
        raise AnalysisError("no valid bootstrap replicates")
    def interval(values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [float(lo), float(hi)]
    models = {m: {metric: {"estimate": value, "ci95": interval(samples[m][metric])}
                  for metric, value in point[m].items()} for m in MODELS}
    diffs = {}
    for a, b in combinations(MODELS, 2):
        diffs[f"{a}_minus_{b}"] = {metric: {"estimate": point[a][metric] - point[b][metric],
                                    "ci95": interval(np.asarray(samples[a][metric]) - np.asarray(samples[b][metric]))}
                                    for metric in point[a]}
    return {"seed": seed, "requested_resamples": n_resamples, "valid_resamples": n_resamples - skipped,
            "skipped_single_class": skipped, "method": "paired speaker bootstrap, percentile 95%", "models": models, "differences": diffs}


def analyze(rows_by_key, entries, expected_counts, *, n_resamples=1000, seed=42):
    indexed = validate_predictions(rows_by_key, entries, expected_counts)
    output = {}
    for panel, (protocol, components) in PANELS.items():
        first = indexed[f"{protocol}:LR"]
        ids = sorted(rid for rid, r in first.items() if r["component"] in components)
        labels = [first[rid]["label"] for rid in ids]
        groups = [first[rid]["group_id"] for rid in ids]
        scores = {m: [indexed[f"{protocol}:{m}"][rid]["score"] for rid in ids] for m in MODELS}
        thresholds = {m: entries[f"{protocol}:{m}"]["decision_rule"]["threshold"] for m in MODELS}
        model_results = {}
        for m in MODELS:
            metrics = compute_metrics(labels, scores[m], thresholds[m])
            # Pin AP's tie-aware sklearn implementation, not the fallback in compute_metrics.
            metrics["average_precision"] = float(average_precision_score(labels, scores[m]))
            score_type = indexed[f"{protocol}:{m}"][ids[0]]["score_type"]
            model_results[m] = {"score_type": score_type, "metrics": metrics,
                                "ece_10_bins": expected_calibration_error(labels, scores[m]),
                                "reliability_bins": reliability_bins(labels, scores[m])}
        output[panel] = {"protocol": protocol, "components": sorted(components), "recordings": len(ids),
                         "speakers": len(set(groups)), "genuine": labels.count(0), "tts": labels.count(1),
                         "recordings_by_component": {c: sum(first[rid]["component"] == c for rid in ids) for c in sorted(components)},
                         "models": model_results,
                         "bootstrap": paired_bootstrap(groups, labels, scores, thresholds, n_resamples=n_resamples, seed=seed)}
    return output
