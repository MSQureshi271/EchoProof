"""Tests for R1-N1 evaluation metrics and Equal Error Rate interpolation."""
import unittest

from echoproof.contracts import ContractError
from echoproof.evaluation.metrics import compute_metrics, interpolated_eer


class MetricsTests(unittest.TestCase):
    def test_perfect_separation(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        eer = interpolated_eer(labels, scores)
        self.assertAlmostEqual(eer, 0.0, places=9)

        m = compute_metrics(labels, scores, threshold=0.5)
        self.assertAlmostEqual(m["roc_auc"], 1.0, places=9)
        self.assertAlmostEqual(m["eer"], 0.0, places=9)
        self.assertAlmostEqual(m["macro_f1"], 1.0, places=9)
        self.assertAlmostEqual(m["accuracy"], 1.0, places=9)
        self.assertAlmostEqual(m["fpr"], 0.0, places=9)
        self.assertAlmostEqual(m["fnr"], 0.0, places=9)
        self.assertEqual(m["confusion_matrix"], [[2, 0], [0, 2]])

    def test_tied_scores(self):
        labels = [0, 1]
        scores = [0.5, 0.5]
        eer = interpolated_eer(labels, scores)
        self.assertAlmostEqual(eer, 0.5, places=9)

        m = compute_metrics(labels, scores, threshold=0.5)
        self.assertAlmostEqual(m["roc_auc"], 0.5, places=9)
        self.assertAlmostEqual(m["eer"], 0.5, places=9)

    def test_reversed_scores(self):
        labels = [0, 0, 1, 1]
        scores = [0.9, 0.8, 0.2, 0.1]
        eer = interpolated_eer(labels, scores)
        self.assertAlmostEqual(eer, 1.0, places=9)

        m = compute_metrics(labels, scores, threshold=0.5)
        self.assertAlmostEqual(m["roc_auc"], 0.0, places=9)
        self.assertAlmostEqual(m["eer"], 1.0, places=9)
        self.assertAlmostEqual(m["accuracy"], 0.0, places=9)
        self.assertEqual(m["confusion_matrix"], [[0, 2], [2, 0]])

    def test_greater_equal_threshold_boundary(self):
        labels = [0, 1]
        scores = [0.5, 0.5]

        # score >= threshold: exactly 0.5 is predicted TTS
        m_at = compute_metrics(labels, scores, threshold=0.5)
        self.assertEqual(m_at["tp"], 1)
        self.assertEqual(m_at["fp"], 1)
        self.assertEqual(m_at["tn"], 0)
        self.assertEqual(m_at["fn"], 0)

        # threshold > 0.5: predicted genuine
        m_above = compute_metrics(labels, scores, threshold=0.50001)
        self.assertEqual(m_above["tp"], 0)
        self.assertEqual(m_above["fp"], 0)
        self.assertEqual(m_above["tn"], 1)
        self.assertEqual(m_above["fn"], 1)

    def test_nothing_predicted_tts(self):
        labels = [0, 1]
        scores = [0.2, 0.4]
        m = compute_metrics(labels, scores, threshold=0.9)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["tn"], 1)
        self.assertEqual(m["fn"], 1)
        self.assertAlmostEqual(m["tts_precision"], 0.0, places=9)
        self.assertAlmostEqual(m["tts_recall"], 0.0, places=9)
        self.assertAlmostEqual(m["fpr"], 0.0, places=9)
        self.assertAlmostEqual(m["fnr"], 1.0, places=9)

    def test_single_class_rejected(self):
        with self.assertRaises(ContractError):
            interpolated_eer([0, 0, 0], [0.1, 0.2, 0.3])
        with self.assertRaises(ContractError):
            interpolated_eer([1, 1], [0.8, 0.9])
        with self.assertRaises(ContractError):
            compute_metrics([0, 0], [0.2, 0.4], threshold=0.5)
        with self.assertRaises(ContractError):
            compute_metrics([1, 1], [0.2, 0.4], threshold=0.5)

    def test_non_finite_and_out_of_bounds_rejected(self):
        for nf in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=nf), self.assertRaises(ContractError):
                interpolated_eer([0, 1], [0.5, nf])
            with self.subTest(thresh=nf), self.assertRaises(ContractError):
                compute_metrics([0, 1], [0.2, 0.8], threshold=nf)

        for oob in (-0.01, 1.01):
            with self.subTest(thresh=oob), self.assertRaises(ContractError):
                compute_metrics([0, 1], [0.2, 0.8], threshold=oob)

        # bool rejected
        with self.assertRaises(ContractError):
            compute_metrics([0, 1], [0.2, 0.8], threshold=True)
        with self.assertRaises(ContractError):
            compute_metrics([False, 1], [0.2, 0.8], threshold=0.5)


if __name__ == "__main__":
    unittest.main()

