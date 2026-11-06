"""Tests for R1-N1 paired group-level bootstrap."""
import unittest

from echoproof.contracts import ContractError
from echoproof.evaluation.bootstrap import paired_group_bootstrap


def simple_acc(labels, scores):
    correct = sum((s >= 0.5) == y for s, y in zip(scores, labels))
    return correct / len(labels)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        # 4 groups, 2 recordings each
        self.group_ids = [
            "speaker:1", "speaker:1",
            "speaker:2", "speaker:2",
            "speaker:3", "speaker:3",
            "speaker:4", "speaker:4",
        ]
        self.labels = [0, 1, 0, 1, 0, 1, 0, 1]
        self.scores_a = [0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8]
        self.scores_b = [0.4, 0.6, 0.3, 0.7, 0.4, 0.6, 0.3, 0.7]

    def test_reproducible_with_seed(self):
        models = {"model_a": self.scores_a, "model_b": self.scores_b}
        res1 = paired_group_bootstrap(self.group_ids, self.labels, models, simple_acc, n_resamples=100, seed=42)
        res2 = paired_group_bootstrap(self.group_ids, self.labels, models, simple_acc, n_resamples=100, seed=42)

        self.assertEqual(res1["valid_resamples"], res2["valid_resamples"])
        self.assertEqual(res1["models"]["model_a"], res2["models"]["model_a"])
        self.assertEqual(res1["differences"]["model_a_vs_model_b"], res2["differences"]["model_a_vs_model_b"])

    def test_ci_contains_estimate(self):
        models = {"model_a": self.scores_a}
        res = paired_group_bootstrap(self.group_ids, self.labels, models, simple_acc, n_resamples=200, seed=123)
        est = res["models"]["model_a"]["estimate"]
        ci_l = res["models"]["model_a"]["ci_lower"]
        ci_u = res["models"]["model_a"]["ci_upper"]

        self.assertLessEqual(ci_l, est)
        self.assertGreaterEqual(ci_u, est)

    def test_identical_models_give_zero_difference(self):
        models = {"model_1": self.scores_a, "model_2": self.scores_a}
        res = paired_group_bootstrap(self.group_ids, self.labels, models, simple_acc, n_resamples=100, seed=99)
        diff = res["differences"]["model_1_vs_model_2"]

        self.assertAlmostEqual(diff["estimate"], 0.0, places=9)
        self.assertAlmostEqual(diff["ci_lower"], 0.0, places=9)
        self.assertAlmostEqual(diff["ci_upper"], 0.0, places=9)

    def test_one_group_rejected(self):
        single_group_ids = ["speaker:1"] * len(self.labels)
        models = {"model_a": self.scores_a}
        with self.assertRaises(ContractError):
            paired_group_bootstrap(single_group_ids, self.labels, models, simple_acc, seed=42)

    def test_groups_kept_whole(self):
        # Tracker metric to verify all items of a group are sampled together
        seen_ratios = []

        def check_ratio(y, s):
            # Since each group has exactly 1 genuine and 1 tts,
            # any sample composed of whole groups will always have sum(y) == len(y)/2!
            seen_ratios.append(sum(y) / len(y))
            return 1.0

        models = {"m": self.scores_a}
        paired_group_bootstrap(self.group_ids, self.labels, models, check_ratio, n_resamples=50, seed=7)
        for r in seen_ratios:
            self.assertAlmostEqual(r, 0.5, places=9)

    def test_invalid_inputs(self):
        models = {"m": self.scores_a}
        with self.assertRaises(ContractError):
            paired_group_bootstrap(self.group_ids[:3], self.labels, models, simple_acc, seed=42)
        with self.assertRaises(ContractError):
            paired_group_bootstrap(self.group_ids, self.labels, {}, simple_acc, seed=42)
        with self.assertRaises(ContractError):
            paired_group_bootstrap(self.group_ids, self.labels, {"m": self.scores_a[:2]}, simple_acc, seed=42)
        with self.assertRaises(ContractError):
            paired_group_bootstrap(self.group_ids, [0] * len(self.labels), models, simple_acc, seed=42)
        with self.assertRaises(ContractError):
            paired_group_bootstrap(self.group_ids, self.labels, models, simple_acc, n_resamples=0, seed=42)
        with self.assertRaises(ContractError):
            paired_group_bootstrap(self.group_ids, self.labels, models, simple_acc, seed=-1)
        with self.assertRaises(ContractError):
            paired_group_bootstrap(self.group_ids, self.labels, models, simple_acc, seed=42, alpha=1.5)


if __name__ == "__main__":
    unittest.main()

