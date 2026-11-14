"""Comparison logic for the R3-N1 WavLM replay (numpy only; the replay itself needs the model and runs on Colab).

Tier 1 scores the canonical windows of a recording with the bundle's own scorer, aggregates, maps and decides. Tier 2 sends the same samples,
re-encoded as a 16 kHz mono PCM16 WAV, through the shared `analyze` path. Both are compared with the saved recording score, the saved score mapped by
the bundle's stored function, and the decision that mapped score implies. A decision flip is acceptable only when the expected score lies within the
tolerance of the threshold; every flip is listed either way.
"""
from __future__ import annotations

import numpy as np

from echoproof.contracts import DecisionRule

TOLERANCE = 1e-3
NEAR_THRESHOLD = 0.01


def compare_replay(records: list[dict], rule: DecisionRule, *, tolerance: float = TOLERANCE, near: float = NEAR_THRESHOLD) -> dict:
    """Each record: recording_id, role, expected_raw, expected_score, expected_decision, and optionally tier1_* / tier2_* (raw, score, decision)."""
    threshold = rule.threshold
    out = {"tolerance": tolerance, "recordings": len(records), "threshold": threshold, "tiers": {}, "flips": [], "near_threshold": []}
    for r in records:
        if threshold is not None and abs(r["expected_score"] - threshold) <= near:
            out["near_threshold"].append({"recording_id": r["recording_id"], "role": r["role"], "expected_score": r["expected_score"]})
    failed_flips = 0
    for tier in ("tier1", "tier2"):
        rows = [r for r in records if r.get(f"{tier}_raw") is not None]
        unscored = len(records) - len(rows)
        if not rows:
            out["tiers"][tier] = {"scored": 0, "unscored": unscored}
            continue
        raw_diff = np.array([abs(r[f"{tier}_raw"] - r["expected_raw"]) for r in rows])
        score_diff = np.array([abs(r[f"{tier}_score"] - r["expected_score"]) for r in rows])
        flips = [r for r in rows if r[f"{tier}_decision"] != r["expected_decision"]]
        explained = [r for r in flips if threshold is not None and abs(r["expected_score"] - threshold) <= tolerance]
        for r in flips:
            out["flips"].append({"tier": tier, "recording_id": r["recording_id"], "role": r["role"], "expected_score": r["expected_score"],
                                 f"{tier}_score": r[f"{tier}_score"], "within_tolerance_of_threshold": r in explained})
        failed_flips += len(flips) - len(explained)
        out["tiers"][tier] = {"scored": len(rows), "unscored": unscored, "max_raw_difference": float(raw_diff.max()), "mean_raw_difference": float(raw_diff.mean()),
                              "max_score_difference": float(score_diff.max()), "mean_score_difference": float(score_diff.mean()), "decision_flips": len(flips),
                              "unexplained_flips": len(flips) - len(explained), "within_tolerance": bool(score_diff.max() <= tolerance and raw_diff.max() <= tolerance)}
    both = [r for r in records if r.get("tier1_score") is not None and r.get("tier2_score") is not None]
    if both:
        d = np.array([abs(r["tier1_score"] - r["tier2_score"]) for r in both])
        out["tier1_vs_tier2"] = {"recordings": len(both), "max_score_difference": float(d.max()), "mean_score_difference": float(d.mean())}
    scored = [t for t in out["tiers"].values() if t["scored"]]
    out["passed"] = bool(scored and all(t["within_tolerance"] for t in scored) and failed_flips == 0)
    return out
