"""R1-N2 presentation rules for the application, kept free of Streamlit so they can be tested without it."""
from __future__ import annotations

import json

from echoproof.contracts import InferenceResult
from echoproof.inference import FAILURE_MESSAGES

NOT_PROOF = ("A model score is evidence from one model on this recording, not proof of how it was made. "
             "It is not a calibrated probability unless the label says so.")


def present(result: InferenceResult) -> dict:
    """Headline, severity ('success' | 'warning' | 'error') and detail lines for one result; never invents a score for a failure."""
    if result.status != "success":
        return {"level": "error", "headline": "Analysis failed",
                "lines": [FAILURE_MESSAGES.get(result.failure_reason, f"Analysis failed ({result.failure_reason}).")]}
    verdict = {"synthetic": "Synthetic speech indicators found", "genuine": "No synthetic speech indicators found"}[result.assessment]
    if result.score_type == "mock_score":     # a placeholder must never read like a detection
        verdict = f"MOCK result (placeholder, not a detection): would be \"{verdict.lower()}\""
    kind = {"calibrated_probability": "Calibrated probability of synthetic speech", "model_score": "Model score",
            "mock_score": "MOCK score (placeholder)"}.get(result.score_type, result.score_type)
    lines = [f"{kind}: {result.score:.3f}", f"Model: {result.model_version}",
             f"Audio analysed: {result.covered_duration_seconds:.1f} s of {result.decoded_duration_seconds:.1f} s", NOT_PROOF]
    return {"level": "warning" if result.score_type == "mock_score" else "success", "headline": verdict, "lines": lines}


def result_json(result: InferenceResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False)


def diagnostics(analysis, bundle) -> dict:
    """Score details for the details block and the downloadable JSON. The InferenceResult contract is unchanged; these are extra fields."""
    rule = bundle.spec.decision_rule
    return {"raw_recording_score": analysis.raw_score, "reported_score": analysis.result.score, "score_type": analysis.result.score_type,
            "calibrated": analysis.result.score_type == "calibrated_probability", "decision_rule": {"kind": rule.kind, "threshold": rule.threshold},
            "bundle_id": bundle.spec.bundle_id, "model_version": bundle.spec.model_version, "chunks_scored": len(analysis.chunk_scores)}


def result_json_with_diagnostics(analysis, bundle) -> str:
    return json.dumps({**analysis.result.to_dict(), "diagnostics": diagnostics(analysis, bundle)}, indent=2, sort_keys=True, allow_nan=False)
