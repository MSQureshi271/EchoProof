"""R3-N1 WavLM numeric building blocks: reliability bins / ECE and the affine log-odds calibration (numpy and scipy only)."""
import math

import numpy as np
import pytest

from echoproof.contracts import ContractError
from echoproof.evaluation.affine_calibration import apply_affine, fit_affine, _objective, _logits
from echoproof.evaluation.calibrate import apply_temperature
from echoproof.evaluation.reliability import expected_calibration_error, reliability_bins


# ---- reliability -----------------------------------------------------------------------------------------------------------------------------

def test_bins_cover_every_score_once_and_the_last_bin_is_closed():
    scores = [0.0, 0.05, 0.1, 0.55, 0.999, 1.0]
    bins = reliability_bins([0, 0, 0, 1, 1, 1], scores)
    assert len(bins) == 10 and sum(b["count"] for b in bins) == len(scores)
    assert bins[0]["count"] == 2 and bins[1]["count"] == 1 and bins[9]["count"] == 2     # 1.0 joins the last bin; 0.1 starts the second
    empty = [b for b in bins if b["count"] == 0]
    assert empty and all(b["mean_score"] is None and b["observed_rate"] is None for b in empty)


def test_ece_known_values_and_perfect_calibration():
    assert expected_calibration_error([0, 1], [0.1, 0.9]) == pytest.approx(0.1)
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40000)
    y = (rng.uniform(0, 1, 40000) < p).astype(int)
    assert expected_calibration_error(y.tolist(), p.tolist()) < 0.02
    assert expected_calibration_error([1, 1, 1], [0.0, 0.0, 0.0]) == pytest.approx(1.0)


@pytest.mark.parametrize("labels,scores,bins", [([0, 1], [0.2], 10), ([], [], 10), ([0, 2], [0.1, 0.2], 10), ([0, 1], [0.1, float("nan")], 10),
                                                ([0, 1], [0.1, 1.2], 10), ([0, 1], [0.1, 0.2], 0)])
def test_reliability_rejects_invalid_inputs(labels, scores, bins):
    with pytest.raises(ContractError):
        reliability_bins(labels, scores, bins)


# ---- affine apply ------------------------------------------------------------------------------------------------------------------------------

def test_identity_parameters_leave_scores_unchanged_up_to_clipping():
    scores = [1e-9, 0.01, 0.3, 0.5, 0.9, 1.0 - 1e-9, 0.0, 1.0]
    out = apply_affine(scores, 1.0, 0.0)
    expected = np.clip(scores, 1e-7, 1 - 1e-7)
    assert np.allclose(out, expected, atol=1e-12, rtol=0)


def test_affine_is_monotone_stable_and_matches_temperature_when_b_is_zero():
    scores = np.linspace(0, 1, 501).tolist()
    out = apply_affine(scores, 0.7, 0.4)
    assert all(0.0 <= v <= 1.0 and math.isfinite(v) for v in out)
    assert all(x <= y for x, y in zip(out, out[1:]))
    assert np.allclose(apply_affine(scores, 1 / 1.6, 0.0), apply_temperature(scores, 1.6), atol=1e-12, rtol=0)


def test_a_nonzero_offset_moves_the_half_boundary():
    # boundary at z = -b / a: with a = 1 and b = 1 a raw score of sigmoid(-1) ~ 0.2689 maps to exactly 0.5
    raw = 1.0 / (1.0 + math.exp(1.0))
    assert apply_affine([raw], 1.0, 1.0)[0] == pytest.approx(0.5, abs=1e-12)
    assert apply_affine([0.3], 1.0, 1.0)[0] >= 0.5 > apply_affine([0.3], 1.0, 0.0)[0]


@pytest.mark.parametrize("a,b", [(0.0, 0.0), (-1.0, 0.0), (float("nan"), 0.0), (1.0, float("inf")), (True, 0.0)])
def test_apply_affine_rejects_invalid_parameters(a, b):
    with pytest.raises(ContractError):
        apply_affine([0.5], a, b)


def test_apply_affine_rejects_invalid_scores():
    for bad in ([1.5], [float("nan")], [-0.1]):
        with pytest.raises(ContractError):
            apply_affine(bad, 1.0, 0.0)


# ---- affine fit -----------------------------------------------------------------------------------------------------------------------------------

def synthetic(n=6000, a=0.6, b=0.9, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 3, n)
    p_true = 1 / (1 + np.exp(-(a * z + b)))
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    p_raw = 1 / (1 + np.exp(-z))
    return y.tolist(), p_raw.tolist()


def test_fit_recovers_known_parameters_and_reduces_log_loss():
    labels, scores = synthetic()
    fit = fit_affine(labels, scores)
    assert fit["ok"] and fit["failure_reason"] is None
    assert fit["a"] == pytest.approx(0.6, abs=0.06) and fit["b"] == pytest.approx(0.9, abs=0.12)
    assert fit["a"] == pytest.approx(math.exp(fit["u"]))
    assert fit["objective_at_fit"] < fit["objective_at_identity"]


def test_analytic_gradient_matches_finite_differences():
    labels, scores = synthetic(n=400, seed=3)
    y, s = np.asarray(labels), np.asarray(scores)
    z, m = _logits(s, 1e-7), (1 - 2 * y).astype(float)
    for point in ((0.0, 0.0), (0.3, -0.7), (-1.0, 2.0)):
        _, grad = _objective(point, z, m, 0.01)
        h = 1e-6
        numeric = [(_objective((point[0] + h * (i == 0), point[1] + h * (i == 1)), z, m, 0.01)[0]
                    - _objective((point[0] - h * (i == 0), point[1] - h * (i == 1)), z, m, 0.01)[0]) / (2 * h) for i in range(2)]
        assert np.allclose(grad, numeric, atol=1e-6)


def test_well_calibrated_scores_keep_parameters_near_identity():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    fit = fit_affine(y.tolist(), p.tolist())
    assert fit["ok"] and abs(fit["u"]) < 0.1 and abs(fit["b"]) < 0.1


def test_a_fit_that_runs_into_a_bound_is_reported_as_failed_not_raised():
    labels = [1] * 200 + [0] * 5          # almost everything positive with moderate scores: the offset wants to leave its bounds
    scores = [0.5] * 205
    fit = fit_affine(labels, scores, b_bounds=(-0.5, 0.5))
    assert fit["ok"] is False and "bound" in fit["failure_reason"]


def test_fit_rejects_single_class_and_bad_inputs():
    with pytest.raises(ContractError):
        fit_affine([1, 1, 1], [0.2, 0.5, 0.8])
    with pytest.raises(ContractError):
        fit_affine([0, 1], [0.2])
    with pytest.raises(ContractError):
        fit_affine([0, 1], [0.2, float("nan")])


def test_fit_is_deterministic():
    labels, scores = synthetic(n=800, seed=9)
    a, b = fit_affine(labels, scores), fit_affine(labels, scores)
    assert (a["u"], a["b"]) == (b["u"], b["b"])
