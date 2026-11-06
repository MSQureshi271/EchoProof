"""Tests for R1-N1 operating threshold selection."""
import unittest

from echoproof.contracts import ContractError, DecisionRule
from echoproof.evaluation.threshold import select_threshold


class ThresholdTests(unittest.TestCase):
    def test_separable_data_macro_f1_one(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        rule, macro_f1, fpr = select_threshold(labels, scores)

        self.assertAlmostEqual(macro_f1, 1.0, places=9)
        self.assertAlmostEqual(fpr, 0.0, places=9)
        self.assertEqual(rule.kind, "threshold")
        self.assertIsNotNone(rule.threshold)
        self.assertLessEqual(rule.threshold, 1.0)
        # Separates at 0.8
        self.assertAlmostEqual(rule.threshold, 0.8, places=9)

    def test_agrees_with_brute_force(self):
        labels = [0, 0, 0, 1, 1, 1, 0, 1]
        scores = [0.15, 0.35, 0.45, 0.40, 0.65, 0.85, 0.70, 0.95]
        rule, macro_f1, fpr = select_threshold(labels, scores)

        best_bf_f1 = -1.0
        for u in scores:
            pred = [s >= u for s in scores]
            tp = sum(p and y == 1 for p, y in zip(pred, labels))
            fp = sum(p and y == 0 for p, y in zip(pred, labels))
            tn = sum((not p) and y == 0 for p, y in zip(pred, labels))
            fn = sum((not p) and y == 1 for p, y in zip(pred, labels))
            f1_t = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            f1_g = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0
            mf1 = (f1_t + f1_g) / 2.0
            if mf1 > best_bf_f1:
                best_bf_f1 = mf1

        self.assertAlmostEqual(macro_f1, best_bf_f1, places=7)
        self.assertLessEqual(rule.threshold, 1.0)

    def test_uninformative_constant_scores(self):
        labels = [0, 0, 1, 1]
        scores = [0.5, 0.5, 0.5, 0.5]
        rule, macro_f1, fpr = select_threshold(labels, scores)
        self.assertLessEqual(rule.threshold if rule.threshold is not None else 0.0, 1.0)
        self.assertGreater(macro_f1, 0.0)

    def test_all_negative_case(self):
        # Heavy genuine imbalance where predicting all negative gives highest macro F1
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        scores = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        rule, macro_f1, fpr = select_threshold(labels, scores)

        self.assertEqual(rule.kind, "all_negative")
        self.assertIsNone(rule.threshold)
        self.assertAlmostEqual(fpr, 0.0, places=9)
        expected_f1_gen = 2.0 * 9 / (2.0 * 9 + 1)
        self.assertAlmostEqual(macro_f1, expected_f1_gen / 2.0, places=9)

    def test_never_above_one(self):
        labels = [0, 1, 0, 1]
        scores = [0.99, 1.0, 0.0, 0.95]
        rule, _, _ = select_threshold(labels, scores)
        if rule.threshold is not None:
            self.assertLessEqual(rule.threshold, 1.0)

    def test_single_class_and_invalid_inputs(self):
        with self.assertRaises(ContractError):
            select_threshold([0, 0, 0], [0.1, 0.2, 0.3])
        with self.assertRaises(ContractError):
            select_threshold([1, 1], [0.8, 0.9])
        with self.assertRaises(ContractError):
            select_threshold([], [])
        with self.assertRaises(ContractError):
            select_threshold([0, 1], [0.5, float("nan")])


if __name__ == "__main__":
    unittest.main()

