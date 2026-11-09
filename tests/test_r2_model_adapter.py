"""Conformance checks for the R0 inference adapter contract (``models.base``) on the production LR."""
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from echoproof.bundle import BundleError
from echoproof.models.base import InferenceModelAdapter, validate_scores
from echoproof.models.lr_adapter import LRBundleAdapter
from echoproof.models.lr_detector import LRDetector

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "bundles" / "protocol_B_lr"


def _waves():
    t = np.arange(16000, dtype=np.float32) / 16000
    rng = np.random.default_rng(7)
    return [
        (np.sin(2 * np.pi * 150 * t) * 0.2).astype(np.float32),
        (np.sin(2 * np.pi * 320 * t) * 0.1 + rng.normal(0, 0.01, t.size)).astype(np.float32),
        rng.normal(0, 0.05, 24000).astype(np.float32),
    ]


@unittest.skipUnless((BUNDLE / "bundle.json").is_file(), "Protocol B LR bundle not present")
class LRAdapterConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = LRBundleAdapter.load(BUNDLE)
        cls.waves = _waves()

    def test_declares_protocol(self):
        self.assertIsInstance(self.adapter, InferenceModelAdapter)
        self.assertTrue(callable(LRBundleAdapter.load))

    def test_scores_shape_range_and_order(self):
        scores = self.adapter.score_chunks(self.waves)
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        reversed_scores = self.adapter.score_chunks(self.waves[::-1])
        np.testing.assert_allclose(reversed_scores[::-1], scores, atol=1e-12)

    def test_scores_identical_to_locked_detector(self):
        direct = LRDetector.load(BUNDLE / "model").score_chunks(self.waves)
        np.testing.assert_array_equal(self.adapter.score_chunks(self.waves), np.asarray(direct, dtype=np.float64))

    def test_empty_input_returns_empty_vector(self):
        out = self.adapter.score_chunks([])
        self.assertEqual(out.shape, (0,))

    def test_metadata_is_read_only_strings_with_bundle_identity(self):
        meta = self.adapter.metadata
        self.assertTrue(all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()))
        self.assertIn("bundle_id", meta)
        self.assertIn("feature_schema_hash", meta)
        with self.assertRaises(TypeError):
            meta["x"] = "y"  # type: ignore[index]

    def test_legacy_directory_layouts_load_with_same_scores(self):
        expected = self.adapter.score_chunks(self.waves)
        for path in (BUNDLE / "model", BUNDLE.parent / "protocol_B_lr" / "model"):
            np.testing.assert_array_equal(LRBundleAdapter.load(path).score_chunks(self.waves), expected)
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "run"
            shutil.copytree(BUNDLE / "model", run_dir / "model")
            np.testing.assert_array_equal(LRBundleAdapter.load(run_dir).score_chunks(self.waves), expected)

    def test_tampered_component_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "bundle"
            shutil.copytree(BUNDLE, copy)
            weights = copy / "model" / "weights.npz"
            weights.write_bytes(weights.read_bytes() + b"x")
            with self.assertRaises(BundleError):
                LRBundleAdapter.load(copy)

    def test_missing_path_and_empty_directory(self):
        with self.assertRaises(FileNotFoundError):
            LRBundleAdapter.load(ROOT / "bundles" / "does_not_exist")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                LRBundleAdapter.load(folder)


class ValidateScoresTests(unittest.TestCase):
    def test_accepts_valid_vector_including_bounds(self):
        out = validate_scores([0.0, 0.5, 1.0], 3)
        self.assertEqual(out.dtype, np.float64)

    def test_rejects_wrong_count_shape_nan_and_range(self):
        for bad, n in (([0.1, 0.2], 3), ([[0.1], [0.2]], 2), ([0.1, float("nan")], 2),
                       ([0.1, float("inf")], 2), ([0.1, 1.0000001], 2), ([-0.1, 0.5], 2)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_scores(bad, n)

    def test_zero_expected_requires_empty(self):
        self.assertEqual(validate_scores([], 0).shape, (0,))
        with self.assertRaises(ValueError):
            validate_scores([0.5], 0)


if __name__ == "__main__":
    unittest.main()
