"""The replay comparison logic: tolerances, decision flips and which flips are explainable."""
import pytest

from echoproof.contracts import DecisionRule
from echoproof.evaluation.r3_n1_wavlm_replay import compare_replay

RULE = DecisionRule(1, "threshold", 0.40)


def rec(i, expected, *, t1=None, t2=None, role="operating"):
    r = {"recording_id": f"phonemedf:genuine:{i}", "role": role, "expected_raw": expected - 0.1, "expected_score": expected, "expected_decision": int(expected >= 0.4)}
    for tier, shift in (("tier1", t1), ("tier2", t2)):
        if shift is not None:
            score = expected + shift
            r.update({f"{tier}_raw": r["expected_raw"] + shift, f"{tier}_score": score, f"{tier}_decision": int(score >= 0.4)})
    return r


def test_exact_replay_passes_with_no_flips():
    out = compare_replay([rec(1, 0.2, t1=0.0, t2=0.0), rec(2, 0.8, t1=1e-9, t2=1e-9)], RULE)
    assert out["passed"] and out["flips"] == [] and out["tiers"]["tier1"]["max_score_difference"] <= 1e-9
    assert out["tier1_vs_tier2"]["recordings"] == 2


def test_a_difference_above_the_tolerance_fails():
    out = compare_replay([rec(1, 0.2, t1=0.0, t2=0.005)], RULE)
    assert not out["passed"] and out["tiers"]["tier2"]["within_tolerance"] is False and out["tiers"]["tier1"]["within_tolerance"] is True


def test_a_flip_next_to_the_threshold_is_explained_but_listed():
    out = compare_replay([rec(1, 0.4004, t1=-0.0006, t2=0.0)], RULE)           # 0.4004 -> 0.3998: below the threshold, within 1e-3 of it
    assert out["tiers"]["tier1"]["decision_flips"] == 1 and out["tiers"]["tier1"]["unexplained_flips"] == 0
    assert out["flips"][0]["within_tolerance_of_threshold"] is True and out["passed"]


def test_a_flip_far_from_the_threshold_fails_even_if_scores_are_close():
    # an expected decision that disagrees with the tier decision while the scores agree is a logic error, not rounding
    r = rec(1, 0.9, t1=0.0, t2=0.0)
    r["tier1_decision"] = 0
    out = compare_replay([r], RULE)
    assert not out["passed"] and out["tiers"]["tier1"]["unexplained_flips"] == 1


def test_near_threshold_recordings_are_listed_and_unscored_ones_counted():
    out = compare_replay([rec(1, 0.405, t1=0.0), rec(2, 0.9), rec(3, 0.1, t1=0.0)], RULE)
    assert [n["recording_id"] for n in out["near_threshold"]] == ["phonemedf:genuine:1"]
    assert out["tiers"]["tier1"]["scored"] == 2 and out["tiers"]["tier1"]["unscored"] == 1
    assert out["tiers"]["tier2"] == {"scored": 0, "unscored": 3}


def test_nothing_scored_is_not_a_pass():
    assert compare_replay([rec(1, 0.5)], RULE)["passed"] is False


def test_an_all_negative_rule_has_no_threshold_to_flip_around():
    rule = DecisionRule(1, "all_negative", None)
    r = rec(1, 0.9, t1=0.0)
    r["expected_decision"] = r["tier1_decision"] = 0
    out = compare_replay([r], rule)
    assert out["passed"] and out["near_threshold"] == []
