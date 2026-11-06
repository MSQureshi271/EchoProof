import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r2_s1_features import build_mini_project  # noqa: E402

from echoproof.data import splits  # noqa: E402
from echoproof.eda import measure, selection, shortcut  # noqa: E402
from echoproof.features.r2_s1 import R2S1Error  # noqa: E402


def synthetic_frame(n_speakers=12, per=20, seed=0, duration_effect=0.0, rms_effect=0.0, components=("genuine", "melo", "vits")):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_speakers):
        for i in range(per):
            comp = components[i % len(components)]
            label = 0 if comp == "genuine" else 1
            rows.append({"recording_id": f"phonemedf:{comp}:{s}-1-{i:04d}", "group_id": f"speaker:{s}", "component": comp, "label": label,
                         "duration_s": 10 + rng.normal() * 2 - duration_effect * label * 3, "rms_dbfs": -25 + rng.normal() * 3 - rms_effect * label * 4,
                         "silence_fraction": rng.uniform(0.1, 0.5), "sample_rate": 16000.0, "leading_silence_s": rng.uniform(0, 0.5),
                         "trailing_silence_s": rng.uniform(0, 0.5), "noise_floor_dbfs": -55 + rng.normal() * 4, "active_level_dbfs": -30 + rng.normal(),
                         "peak": rng.uniform(0.3, 0.9), "clip_fraction": 0.0, "dc_offset": rng.normal() * 1e-3, "hf_ratio_4k": rng.uniform(0, 0.05),
                         "hf_ratio_7k": rng.uniform(0, 0.01)})
    return pd.DataFrame(rows)


class Matching(unittest.TestCase):
    def test_equal_counts_in_every_bin_and_deterministic(self):
        frame = synthetic_frame(duration_effect=1.0)
        ids = shortcut.duration_matched(frame)
        self.assertEqual(ids, shortcut.duration_matched(frame))
        kept = frame[frame["recording_id"].isin(ids)]
        bins = np.floor(kept["duration_s"] / shortcut.BIN_SECONDS)
        for b in set(bins):
            part = kept[bins == b]
            self.assertEqual((part["label"] == 0).sum(), (part["label"] == 1).sum())
        self.assertGreater(len(kept), 20)
        self.assertEqual(len(kept["recording_id"]), len(set(kept["recording_id"])))
        self.assertNotEqual(ids, shortcut.duration_matched(frame, seed=7))

    def test_matching_removes_the_duration_signal(self):
        frame = synthetic_frame(duration_effect=1.0, seed=3, n_speakers=30)
        before = shortcut._auc(frame["label"], frame["duration_s"])
        kept = frame[frame["recording_id"].isin(shortcut.duration_matched(frame))]
        after = shortcut._auc(kept["label"], kept["duration_s"])
        self.assertLess(before, 0.35)                                   # synthetic recordings are shorter in this fixture
        self.assertLess(abs(after - 0.5), 0.07)

    def test_bins_with_one_class_are_dropped(self):
        frame = synthetic_frame(per=6)
        frame.loc[frame["label"] == 0, "duration_s"] += 100.0              # no overlap at all
        self.assertEqual(shortcut.duration_matched(frame), [])


class Models(unittest.TestCase):
    def test_constant_columns_are_dropped_and_reported(self):
        train = synthetic_frame()
        _, info = shortcut.fit_lr(train, list(shortcut.PRIMARY))
        self.assertEqual(info["constant_columns_dropped"], ["sample_rate"])
        self.assertNotIn("sample_rate", info["features_used"])
        predict, info = shortcut.fit_lr(train, ["sample_rate"])                 # nothing varies: constant score, flagged, never a crash
        self.assertTrue(info["constant_model"])
        self.assertEqual(set(predict(train)), {0.5})
        self.assertEqual(shortcut.evaluate(train, predict(train), n_resamples=20)["roc_auc"], 0.5)

    def test_a_planted_duration_shortcut_is_found_and_noise_is_not(self):
        train, test = synthetic_frame(seed=1, duration_effect=1.0, n_speakers=20), synthetic_frame(seed=2, duration_effect=1.0, n_speakers=15)
        predict, info = shortcut.fit_lr(train, ["duration_s"])
        found = shortcut.evaluate(test, predict(test), n_resamples=200)
        self.assertGreater(found["roc_auc"], 0.7)
        self.assertEqual(found["verdict"], "separates")
        self.assertLess(info["standardised_coefficients"]["duration_s"], 0)                 # shorter -> synthetic
        clean_train, clean_test = synthetic_frame(seed=4, n_speakers=20), synthetic_frame(seed=5, n_speakers=15)
        predict, _ = shortcut.fit_lr(clean_train, ["rms_dbfs"])
        none = shortcut.evaluate(clean_test, predict(clean_test), n_resamples=200)
        self.assertLess(none["roc_auc"], 0.6)
        self.assertEqual(none["verdict"], "does not separate")

    def test_verdict_rule(self):
        self.assertEqual(shortcut.verdict(0.9, 0.7), "separates")
        self.assertEqual(shortcut.verdict(0.65, 0.55), "marginal")
        self.assertEqual(shortcut.verdict(0.55, 0.45), "does not separate")
        self.assertEqual(shortcut.verdict(0.7, 0.6), "marginal")                           # the floor itself is not enough

    def test_scores_are_independent_of_the_test_set(self):
        train, test = synthetic_frame(seed=1), synthetic_frame(seed=2)
        predict, _ = shortcut.fit_lr(train, list(shortcut.WITH_EDGES))
        np.testing.assert_array_equal(predict(test.iloc[:10]), predict(test)[:10])


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.mini = build_mini_project(cls.root, speakers=40)
        cls.train_dir = cls.root / "data" / "eda" / "r2_s2_B_train"
        cls.sel_dir = cls.root / "data" / "eda" / "r2_s3_B_selection"
        measure.build(cls.root, cls.mini["shard_dir"], cls.train_dir, example_seconds=(1.0, 6.6))
        cls.sel_manifest = selection.build(cls.root, cls.mini["shard_dir"], cls.sel_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_selection_scope_only(self):
        frame = pd.read_parquet(self.sel_dir / selection.OUTPUT_NAMES["stats"])
        expected = {a.recording_id for a in self.mini["assignments"] if a.partition == "validation" and a.validation_role == "selection"}
        self.assertEqual(set(frame["recording_id"]), expected)
        self.assertEqual(set(frame["role"]), {"selection"})
        self.assertEqual(set(frame["component"]), {"genuine", "melo", "vits", "xtts"})
        self.assertFalse(frame["recording_id"].str.contains("chatterbox").any())
        train_speakers = {a.group_id for a in self.mini["assignments"] if a.partition == "train"}
        self.assertFalse(set(frame["group_id"]) & train_speakers)
        self.assertTrue(self.sel_manifest["verification"]["passed"])
        self.assertEqual(self.sel_manifest["spot_check"]["mismatches"], [])

    def test_same_measurements_as_the_train_step(self):
        frame = pd.read_parquet(self.sel_dir / selection.OUTPUT_NAMES["stats"])
        row = frame.iloc[0]
        from test_r2_s1_features import waveform
        direct, _ = measure.measure(waveform(self.mini, row["recording_id"]))
        for column in measure.SCALAR_COLUMNS:
            self.assertEqual(direct[column], row[column], column)

    def test_finished_output_is_not_overwritten_and_other_roles_are_not_read(self):
        with self.assertRaises(R2S1Error):
            selection.build(self.root, self.mini["shard_dir"], self.sel_dir)
        eda = selection.plan_selection(*self._inputs())
        ids = {i["recording_id"] for i in eda.items}
        others = {a.recording_id for a in self.mini["assignments"] if a.partition == "validation" and a.validation_role != "selection"}
        self.assertTrue(others)
        self.assertFalse(ids & others)

    def _inputs(self):
        m = self.mini
        return m["rows"], m["assignments"], m["chunks"]

    def test_full_probe_runs_and_writes_a_memo_and_matched_ids(self):
        with tempfile.TemporaryDirectory() as d:
            report = shortcut.run(self.root, train_dir=self.train_dir, selection_dir=self.sel_dir, report_dir=Path(d) / "r2",
                                  matched_dir=Path(d) / "matched", n_resamples=30)
            memo = (Path(d) / "r2" / shortcut.OUTPUT_MEMO).read_text(encoding="utf-8")
            self.assertIn("## Answer", memo)
            self.assertIn("duration-matched", memo.lower())
            matched = json.loads((Path(d) / "matched" / shortcut.MATCHED_NAME).read_text(encoding="utf-8"))
            sel = pd.read_parquet(self.sel_dir / selection.OUTPUT_NAMES["stats"])
            self.assertTrue(set(matched["ids"]["selection"]) <= set(sel["recording_id"]))
            self.assertEqual(set(report["results"]["feature_sets"]), set(shortcut.FEATURE_SETS))
            for entry in report["results"]["feature_sets"].values():
                for key in "ABC":
                    self.assertTrue(0.0 <= entry[key]["roc_auc"] <= 1.0)
                    self.assertLessEqual(entry[key]["auc_ci95"][0], entry[key]["auc_ci95"][1])
            self.assertEqual(report["results"]["matching"]["selection"]["genuine_matched"], report["results"]["matching"]["selection"]["synthetic_matched"])
            for ext in ("png", "svg"):
                self.assertGreater((Path(d) / "r2" / f"r2_s3_shortcut_auc.{ext}").stat().st_size, 1000)
            again = shortcut.run(self.root, train_dir=self.train_dir, selection_dir=self.sel_dir, report_dir=Path(d) / "r2b",
                                 matched_dir=Path(d) / "matchedb", n_resamples=30)
            self.assertEqual(json.dumps(again["results"], sort_keys=True, default=float), json.dumps(report["results"], sort_keys=True, default=float))

    def test_tampered_or_mismatched_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            copy = Path(d) / "sel"
            shutil.copytree(self.sel_dir, copy)
            path = copy / selection.OUTPUT_NAMES["stats"]
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaises(shortcut.ProbeError):
                shortcut.load_data(self.root, self.train_dir, copy)


if __name__ == "__main__":
    unittest.main()
