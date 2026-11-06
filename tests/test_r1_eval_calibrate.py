"""Tests for R1-N1 temperature scaling calibration."""
import math
import unittest

import numpy as np

from echoproof.contracts import ContractError
from echoproof.evaluation.calibrate import apply_temperature, fit_temperature


class CalibrateTests(unittest.TestCase):
    def test_identity_when_temperature_is_one(self):
        scores = [0.05, 0.2, 0.5, 0.75, 0.95]
        calibrated = apply_temperature(scores, T=1.0)
        for original, cal in zip(scores, calibrated):
            self.assertAlmostEqual(original, cal, places=6)

    def test_ranking_is_strictly_preserved(self):
        scores = [0.1, 0.3, 0.45, 0.7, 0.85]
        for t_val in (0.2, 0.5, 1.0, 2.5, 10.0):
            with self.subTest(T=t_val):
                cal = apply_temperature(scores, T=t_val)
                # Check strict monotonic increase
                for i in range(len(cal) - 1):
                    self.assertLess(cal[i], cal[i + 1])

    def test_recovers_known_temperature_on_overconfident_data(self):
        rng = np.random.RandomState(42)
        n = 1000
        true_t = 2.5

        # Well-calibrated base logits
        base_logits = rng.normal(0.0, 1.5, size=n)
        true_probs = 1.0 / (1.0 + np.exp(-base_logits))
        labels = (rng.uniform(0.0, 1.0, size=n) < true_probs).astype(int)

        # Overconfident scores: logits scaled by true_t
        overconfident_logits = base_logits * true_t
        overconfident_scores = 1.0 / (1.0 + np.exp(-overconfident_logits))

        fitted_t = fit_temperature(labels.tolist(), overconfident_scores.tolist())
        # Should recover approx true_t
        self.assertAlmostEqual(fitted_t, true_t, delta=0.35)

    def test_single_class_rejected(self):
        with self.assertRaises(ContractError):
            fit_temperature([0, 0, 0], [0.1, 0.2, 0.3])
        with self.assertRaises(ContractError):
            fit_temperature([1, 1], [0.8, 0.9])

    def test_invalid_temperature_and_inputs(self):
        for bad_t in (0.0, -1.0, -0.5, float("nan"), float("inf"), "1.0", None, True):
            with self.subTest(T=bad_t), self.assertRaises(ContractError):
                apply_temperature([0.5], T=bad_t)

        with self.assertRaises(ContractError):
            apply_temperature([float("nan")], T=1.0)
        with self.assertRaises(ContractError):
            apply_temperature([1.5], T=1.0)
        with self.assertRaises(ContractError):
            apply_temperature([-0.1], T=1.0)


if __name__ == "__main__":
    unittest.main()

