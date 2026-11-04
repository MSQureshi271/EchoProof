import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH
from echoproof.models.lr_detector import (GRID_C, LRDetector, LRModelError, aggregate_recordings, chunk_features, recording_weights,
                                          run_lr_grid)
from echoproof.training.ledger import COLUMNS

N = len(FEATURE_NAMES)


def synthetic(n_per_class=24, seed=0, signal=1.5, nan_rows=True):
    """Separable 67-column chunk table with unequal chunks per recording and unvoiced-style NaNs."""
    rng = np.random.default_rng(seed)
    X, y, ids = [], [], []
    for label in (0, 1):
        for r in range(n_per_class):
            for _ in range(int(rng.integers(1, 5))):
                row = rng.standard_normal(N)
                row[:6] += signal * (2 * label - 1)
                X.append(row)
                y.append(label)
                ids.append(f"rec:{label}:{r:03d}")
    X = np.array(X)
    X[:, 10] = 3.0                           # constant column: zero variance
    X[:, N - 2:] = np.where(rng.random((len(X), 2)) < 0.3, np.nan, X[:, N - 2:]) if nan_rows else X[:, N - 2:]
    X[:, 20] = np.nan                        # column missing in all training rows
    return {"X": X, "y": np.array(y), "recording_ids": np.array(ids)}


class Schema(unittest.TestCase):
    def test_feature_schema_is_the_67_column_vector(self):
        self.assertEqual(N, 67)
        self.assertEqual(FEATURE_SCHEMA_HASH[:12], "fce2751a069d")  # changing the features must be a conscious decision


class Weights(unittest.TestCase):
    def test_every_recording_has_equal_total_weight_and_total_is_chunk_count(self):
        ids = np.array(["a", "a", "a", "b", "c", "c"])
        y = np.array([0, 0, 0, 1, 1, 1])
        w = recording_weights(ids, y)
        for rid in "abc":
            self.assertAlmostEqual(w[ids == rid].sum(), 2.0)  # 6 chunks / 3 recordings
        self.assertAlmostEqual(w.sum(), 6.0)

    def test_balanced_counts_recordings_not_chunks(self):
        ids = np.array(["g1", "g1", "g1", "g1", "g2", "t1", "t2", "t3"])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        plain, balanced = recording_weights(ids, y), recording_weights(ids, y, "balanced")
        self.assertAlmostEqual(balanced.sum(), plain.sum())
        self.assertAlmostEqual(balanced[y == 0].sum(), balanced[y == 1].sum())
        self.assertAlmostEqual(balanced[ids == "g1"].sum(), balanced[ids == "g2"].sum())

    def test_inconsistent_labels_and_single_class_are_rejected(self):
        with self.assertRaises(LRModelError):
            recording_weights(np.array(["a", "a"]), np.array([0, 1]))
        with self.assertRaises(LRModelError):
            recording_weights(np.array(["a", "b"]), np.array([1, 1]), "balanced")
        with self.assertRaises(LRModelError):
            recording_weights(np.array(["a"]), np.array([0]), "bogus")


class Fitting(unittest.TestCase):
    def test_numpy_scoring_matches_sklearn_pipeline_exactly_including_nan_rows(self):
        data = synthetic()
        for cw in ("none", "balanced"):
            pipeline, diag = LRDetector.fit_pipeline(data["X"], data["y"], data["recording_ids"], C=1.0, class_weight=cw)
            model = LRDetector.from_pipeline(pipeline, diag, C=1.0, class_weight=cw)
            expected = pipeline.predict_proba(data["X"])[:, list(pipeline.named_steps["classifier"].classes_).index(1)]
            got = model.predict_proba_features(data["X"])
            self.assertLessEqual(float(np.max(np.abs(got - expected))), 1e-12)
            self.assertTrue(np.isfinite(got).all() and ((got >= 0) & (got <= 1)).all())

    def test_learns_separable_data_and_score_direction_is_tts(self):
        data = synthetic()
        model = LRDetector.fit(data["X"], data["y"], data["recording_ids"])
        scores = model.predict_proba_features(data["X"])
        self.assertGreater(scores[data["y"] == 1].mean(), 0.8)
        self.assertLess(scores[data["y"] == 0].mean(), 0.2)
        self.assertGreater(model.arrays["coef"][:6].mean(), 0)  # higher feature -> more TTS
        self.assertEqual(model.coefficient_table()[0][0] in FEATURE_NAMES[:6], True)

    def test_degenerate_columns_do_not_break_scoring(self):
        data = synthetic()
        model = LRDetector.fit(data["X"], data["y"], data["recording_ids"])
        self.assertEqual(model.arrays["scale"][10], 1.0)            # constant column: sklearn convention
        self.assertEqual(model.arrays["medians"][20], 0.0)           # all-missing column kept, filled with 0
        row = np.full((1, N), np.nan)                                # every feature missing -> medians
        self.assertTrue(np.isfinite(model.predict_proba_features(row)).all())

    def test_deterministic_and_convergence_is_reported(self):
        data = synthetic()
        a = LRDetector.fit(data["X"], data["y"], data["recording_ids"], seed=3)
        b = LRDetector.fit(data["X"], data["y"], data["recording_ids"], seed=3)
        for k in a.arrays:
            np.testing.assert_array_equal(a.arrays[k], b.arrays[k])
        self.assertTrue(a.meta["diagnostics"]["converged"])
        short = LRDetector.fit(data["X"], data["y"], data["recording_ids"], max_iter=1)
        self.assertFalse(short.meta["diagnostics"]["converged"])

    def test_bad_inputs_are_rejected(self):
        data = synthetic()
        with self.assertRaises(LRModelError):
            LRDetector.fit(data["X"][:, :10], data["y"], data["recording_ids"])
        with self.assertRaises(LRModelError):
            LRDetector.fit(data["X"][data["y"] == 1], data["y"][data["y"] == 1], data["recording_ids"][data["y"] == 1])
        bad = data["X"].copy()
        bad[0, 0] = np.inf
        with self.assertRaises(LRModelError):
            LRDetector.fit(bad, data["y"], data["recording_ids"])
        model = LRDetector.fit(data["X"], data["y"], data["recording_ids"])
        with self.assertRaises(LRModelError):
            model.predict_proba_features(bad)


class Persistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        data = synthetic()
        self.data = data
        self.model = LRDetector.fit(data["X"], data["y"], data["recording_ids"], C=0.1, class_weight="balanced")
        self.dir = Path(self.tmp.name) / "m"

    def tearDown(self):
        self.tmp.cleanup()

    def test_reload_is_bit_exact_and_save_refuses_overwrite(self):
        digests = self.model.save(self.dir)
        self.assertEqual(set(digests), {"weights.npz", "lr_model.json"})
        loaded = LRDetector.load(self.dir)
        np.testing.assert_array_equal(self.model.predict_proba_features(self.data["X"]), loaded.predict_proba_features(self.data["X"]))
        self.assertEqual(loaded.meta["hyperparameters"], {"C": 0.1, "class_weight": "balanced", "max_iter": 2000, "seed": 42})
        with self.assertRaises(LRModelError):
            self.model.save(self.dir)

    def test_tampered_or_incompatible_models_are_rejected(self):
        self.model.save(self.dir)
        meta = json.loads((self.dir / "lr_model.json").read_text())
        for change in ({"feature_schema_hash": "0" * 64}, {"label_map": {"genuine": 1, "tts": 0}}, {"feature_names": ["x"]}):
            (self.dir / "lr_model.json").write_text(json.dumps({**meta, **change}))
            with self.assertRaises(LRModelError):
                LRDetector.load(self.dir)
        (self.dir / "lr_model.json").write_text(json.dumps(meta))
        arrays = dict(np.load(self.dir / "weights.npz"))
        np.savez(self.dir / "weights.npz", **{**arrays, "coef": np.full(N, np.nan)})
        with self.assertRaises(LRModelError):
            LRDetector.load(self.dir)
        np.savez(self.dir / "weights.npz", **{**arrays, "extra": np.zeros(1)})
        with self.assertRaises(LRModelError):
            LRDetector.load(self.dir)
        np.savez(self.dir / "weights.npz", **{**arrays, "coef": np.array([object()] * N, dtype=object)})  # would need pickle
        with self.assertRaises(LRModelError):
            LRDetector.load(self.dir)
        with self.assertRaises(LRModelError):
            LRDetector.load(Path(self.tmp.name) / "missing")


class OnWaveformFixtures(unittest.TestCase):
    def test_waveforms_to_scores_survive_save_and_reload(self):
        rng = np.random.default_rng(1)
        t = np.arange(12000) / 16000
        waves, y, ids = [], [], []
        for r in range(12):
            label = r % 2
            for c in range(2):
                f = 150 + 20 * r if label == 0 else 2500 + 30 * r
                x = 0.3 * np.sin(2 * np.pi * f * t[: 12000 - 2000 * c]) + 0.02 * rng.standard_normal(12000 - 2000 * c)
                waves.append(x.astype(np.float32))
                y.append(label)
                ids.append(f"fix:{r:02d}")
        X = chunk_features(waves)
        self.assertEqual(X.shape, (24, N))
        model = LRDetector.fit(X, np.array(y), np.array(ids), C=1.0)
        scores = model.score_chunks(waves)
        with tempfile.TemporaryDirectory() as d:
            model.save(d + "/m")
            reloaded = LRDetector.load(d + "/m").score_chunks(waves)
        self.assertLessEqual(float(np.max(np.abs(scores - reloaded))), 1e-12)
        pipeline, diag = LRDetector.fit_pipeline(X, np.array(y), np.array(ids), C=1.0)
        self.assertLessEqual(float(np.max(np.abs(scores - pipeline.predict_proba(X)[:, 1]))), 1e-12)
        self.assertGreater(scores[np.array(y) == 1].mean(), scores[np.array(y) == 0].mean())


class Grid(unittest.TestCase):
    def setUp(self):
        self.train = synthetic(30, seed=1)
        self.selection = synthetic(15, seed=2)

    def test_eight_trials_are_ledgered_and_best_is_selected_at_recording_level(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "ledger.csv"
            out = run_lr_grid(self.train, self.selection, ledger_path=ledger, protocol="B", split_hash="s" * 64, owner="muneeb",
                              metrics_fn=None)
            rows = list(csv.DictReader(open(ledger)))
        self.assertEqual(len(out["trials"]), 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual(list(rows[0]), list(COLUMNS))
        self.assertEqual({r["model"] for r in rows}, {"lr"})
        self.assertEqual(sorted({t["C"] for t in out["trials"]}), sorted(GRID_C))
        self.assertTrue(all("converged=True" in r["notes"] for r in rows))
        best = out["trials"][out["selected"]]
        self.assertEqual(best["metrics"]["recordings"], 30)  # recordings, not chunks
        for t in out["trials"]:
            self.assertLessEqual(t["metrics"]["macro_f1"], best["metrics"]["macro_f1"])
        self.assertIsInstance(out["selected_model"], LRDetector)

    def test_ties_break_by_eer_then_fpr_then_grid_order(self):
        calls = iter([{"macro_f1": 0.9, "eer": 0.2, "fpr": 0.1}, {"macro_f1": 0.9, "eer": 0.1, "fpr": 0.3},
                      {"macro_f1": 0.9, "eer": 0.1, "fpr": 0.2}, {"macro_f1": 0.9, "eer": float("nan"), "fpr": 0.0}] + [{"macro_f1": 0.5, "eer": 0.5, "fpr": 0.5}] * 4)
        out = run_lr_grid(self.train, self.selection, metrics_fn=lambda *a: next(calls))
        self.assertEqual(out["selected"], 2)  # equal F1; lowest EER 0.1; then lower FPR 0.2
        ties = run_lr_grid(self.train, self.selection, metrics_fn=lambda *a: {"macro_f1": 0.7, "eer": 0.3, "fpr": 0.3})
        self.assertEqual(ties["selected"], 0)  # fully tied: smallest C, no class weighting

    def test_failed_trials_are_recorded_and_total_failure_raises(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "ledger.csv"
            out = run_lr_grid(self.train, self.selection, grid_c=(-1.0, 1.0), ledger_path=ledger)
            rows = list(csv.DictReader(open(ledger)))
        self.assertEqual([r["status"].split(":")[0] for r in rows], ["failed", "failed", "completed", "completed"])
        self.assertEqual(out["selected_config"]["C"], 1.0)
        with self.assertRaises(LRModelError):
            run_lr_grid(self.train, self.selection, grid_c=(-1.0,))

    def test_recording_aggregation_is_the_mean_in_first_seen_order(self):
        labels, scores = aggregate_recordings(["b", "a", "b", "a", "a"], [1, 0, 1, 0, 0], [0.2, 0.9, 0.4, 0.3, 0.6])
        self.assertEqual(labels, [1, 0])
        np.testing.assert_allclose(scores, [0.3, 0.6])


if __name__ == "__main__":
    unittest.main()
