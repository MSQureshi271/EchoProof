import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from echoproof.data import cohort, splits
from echoproof.eda import figures, measure
from echoproof.features.acoustic import FEATURE_NAMES, log_mel
from echoproof.features.r2_s1 import R2S1Error
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r2_s1_features import RATE, build_mini_project, waveform


def tone(seconds, amp=0.3, lead=0.0, trail=0.0):
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    x = amp * np.sin(2 * np.pi * 440 * t)
    return np.concatenate([np.zeros(int(lead * RATE)), x, np.zeros(int(trail * RATE))]).astype(np.float32)


class Measurements(unittest.TestCase):
    def test_known_signal(self):
        wave = tone(2.0, 0.5, lead=0.5, trail=1.0)
        row, mel = measure.measure(wave)
        self.assertEqual(row["n_samples"], len(wave))
        self.assertAlmostEqual(row["duration_s"], 3.5)
        self.assertAlmostEqual(row["peak"], 0.5, places=3)
        self.assertAlmostEqual(row["leading_silence_s"], 0.5, delta=0.03)
        self.assertAlmostEqual(row["trailing_silence_s"], 1.0, delta=0.03)
        self.assertAlmostEqual(row["silence_fraction"], 1.5 / 3.5, delta=0.03)
        self.assertAlmostEqual(row["rms_dbfs"], 20 * np.log10(np.sqrt(0.25 / 2 * 2.0 / 3.5)), delta=0.1)
        self.assertEqual(row["clip_fraction"], 0.0)
        self.assertLess(row["hf_ratio_4k"], 1e-3)               # a 440 Hz tone has no energy above 4 kHz
        self.assertEqual(mel.shape, (64,))

    def test_silence_and_full_scale_and_short_input(self):
        row, _ = measure.measure(np.zeros(RATE, np.float32))
        self.assertEqual(row["leading_silence_s"], 1.0)
        self.assertEqual(row["silence_fraction"], 1.0)
        self.assertEqual(row["hf_ratio_4k"], 0.0)
        loud = np.ones(RATE, np.float32)
        self.assertEqual(measure.measure(loud)[0]["clip_fraction"], 1.0)
        row, mel = measure.measure(tone(0.01))                   # shorter than one FFT window
        self.assertTrue(np.isfinite(mel).all() and np.isfinite(list(row.values())).all())

    def test_hf_ratio_orders_a_high_tone_above_a_low_tone(self):
        t = np.arange(RATE) / RATE
        high = (0.3 * np.sin(2 * np.pi * 7500 * t)).astype(np.float32)
        row_high = measure.measure(high)[0]
        self.assertGreater(row_high["hf_ratio_7k"], 0.9)
        self.assertLessEqual(row_high["hf_ratio_7k"], row_high["hf_ratio_4k"] + 1e-12)

    def test_pauses_are_found_over_a_noise_floor_and_absent_in_steady_audio(self):
        rng = np.random.default_rng(3)
        noise = lambda sec: (0.005 * rng.standard_normal(int(sec * RATE))).astype(np.float32)          # about -46 dBFS background  # noqa: E731
        speech = tone(2.0, 0.2) + noise(2.0)
        wave = np.concatenate([noise(0.4), speech, noise(0.8), speech, noise(0.3)])
        row, _ = measure.measure(wave)
        self.assertAlmostEqual(row["leading_silence_s"], 0.4, delta=0.05)
        self.assertAlmostEqual(row["trailing_silence_s"], 0.3, delta=0.05)
        self.assertAlmostEqual(row["silence_fraction"], 1.5 / 6.5, delta=0.04)
        steady, _ = measure.measure(tone(3.0, 0.2) + noise(3.0))
        self.assertEqual((steady["silence_fraction"], steady["leading_silence_s"], steady["trailing_silence_s"]), (0.0, 0.0, 0.0))
        self.assertEqual(measure.activity_threshold(np.full(50, -90.0)), float("inf"))

    def test_rejects_bad_input_and_is_deterministic(self):
        for bad in (np.zeros(0), np.array([np.nan, 0.0]), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                measure.measure(bad)
        w = tone(1.5)
        a, b = measure.measure(w), measure.measure(w)
        self.assertEqual(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class Separation(unittest.TestCase):
    def test_auc_extremes_and_ties(self):
        self.assertEqual(figures.auc([3, 4, 5], [0, 1, 2]), 1.0)
        self.assertEqual(figures.auc([0, 1, 2], [3, 4, 5]), 0.0)
        self.assertEqual(figures.auc([1, 1], [1, 1]), 0.5)
        with self.assertRaises(ValueError):
            figures.auc([], [1])
        self.assertEqual(figures.verdict(0.55), "little")
        self.assertEqual(figures.verdict(0.7), "moderate")
        self.assertEqual(figures.verdict(0.9), "strong")


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.mini = build_mini_project(cls.root)
        cls.out = cls.root / "data" / "eda" / "r2_s2_B_train"
        cls.manifest = measure.build(cls.root, cls.mini["shard_dir"], cls.out, workers=1, example_seconds=(1.0, 6.6))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_scope_is_train_only_without_chatterbox(self):
        import pandas as pd
        frame = pd.read_parquet(self.out / measure.OUTPUT_NAMES["stats"])
        train = {a.recording_id for a in self.mini["assignments"] if a.partition == "train"}
        self.assertEqual(set(frame["recording_id"]), train)
        self.assertEqual(set(frame["component"]), {"genuine", "melo", "vits"})
        self.assertFalse(frame["recording_id"].str.contains("chatterbox|xtts").any())
        self.assertTrue(self.manifest["verification"]["passed"])
        self.assertEqual(self.manifest["spot_check"]["mismatches"], [])

    def test_values_match_a_direct_measurement(self):
        import pandas as pd
        frame = pd.read_parquet(self.out / measure.OUTPUT_NAMES["stats"])
        row = frame.iloc[0]
        direct, mel = measure.measure(waveform(self.mini, row["recording_id"]))
        for column in measure.SCALAR_COLUMNS:
            self.assertEqual(direct[column], row[column], column)
        with np.load(self.out / measure.OUTPUT_NAMES["mel_mean"]) as z:
            np.testing.assert_array_equal(z["mel_mean"][0], mel)
            self.assertEqual(z["recording_id"][0], row["recording_id"])

    def test_examples_are_paired_train_stems_from_different_speakers(self):
        with np.load(self.out / measure.OUTPUT_NAMES["examples"]) as z:
            stems = z["stems"].tolist()
            self.assertEqual(len(stems), 3)
            speakers = {cohort.parse_stem(s)[0] for s in stems}
            self.assertEqual(len(speakers), 3)
            train = {a.recording_id for a in self.mini["assignments"] if a.partition == "train"}
            for s in stems:
                for c in measure.EXAMPLE_COMPONENTS:
                    self.assertIn(cohort.recording_id(c, s), train)
                    np.testing.assert_array_equal(z[f"wave__{s}__{c}"], self.mini["waves"][cohort.recording_id(c, s)])

    def test_finished_outputs_are_never_overwritten(self):
        with self.assertRaises(R2S1Error):
            measure.build(self.root, self.mini["shard_dir"], self.out, example_seconds=(1.0, 6.6))

    def test_resume_skips_current_parts_and_parity_with_two_workers(self):
        with tempfile.TemporaryDirectory() as d:
            other = Path(d) / "out"
            m2 = measure.build(self.root, self.mini["shard_dir"], other, workers=2, example_seconds=(1.0, 6.6))
            self.assertEqual({k: v["sha256"] for k, v in m2["outputs"].items() if k != "examples"},
                             {k: v["sha256"] for k, v in self.manifest["outputs"].items() if k != "examples"})
            self.assertEqual(m2["outputs"]["examples"]["sha256"], self.manifest["outputs"]["examples"]["sha256"])
            (other / measure.MANIFEST_NAME).unlink()
            for name in measure.OUTPUT_NAMES.values():
                (other / name).unlink()
            with self.assertRaises(R2S1Error):
                measure.build(self.root, self.mini["shard_dir"], other, example_seconds=(1.0, 6.6))      # parts exist, resume not requested
            m3 = measure.build(self.root, self.mini["shard_dir"], other, resume=True, example_seconds=(1.0, 6.6))
            self.assertTrue(all(s.get("skipped") for s in m3["shards"]))
            self.assertEqual(m3["outputs"]["stats"]["sha256"], self.manifest["outputs"]["stats"]["sha256"])

    def test_hash_mismatch_and_missing_shard_fail_loudly(self):
        import shutil
        rows = cohort.read_table(self.root / "data" / "manifests" / "cohort_v1.parquet")
        victim_id = next(a.recording_id for a in self.mini["assignments"] if a.partition == "train")
        next(r for r in rows if r["recording_id"] == victim_id)["pcm_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as d:
            wrong = Path(d) / "wrong"
            shutil.copytree(self.root / "data", wrong / "data")
            (wrong / "data" / "manifests" / "cohort_v1.parquet").unlink()
            cohort.write_table(rows, wrong / "data" / "manifests" / "cohort_v1.parquet")
            with self.assertRaisesRegex(R2S1Error, "shards failed"):
                measure.build(wrong, wrong / "data" / "processed" / "cohort_v1", wrong / "out", example_seconds=(1.0, 6.6))
            next(wrong.glob("data/processed/cohort_v1/shard_*.tar")).unlink()
            with self.assertRaisesRegex(R2S1Error, "shard files not found"):
                measure.build(wrong, wrong / "data" / "processed" / "cohort_v1", wrong / "out2", example_seconds=(1.0, 6.6))


class Figures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.mini = build_mini_project(cls.root)
        cls.data_dir = cls.root / "data" / "eda" / "r2_s2_B_train"
        cls.manifest = measure.build(cls.root, cls.mini["shard_dir"], cls.data_dir, example_seconds=(1.0, 6.6))
        rows = cls.mini["rows"]
        reports = cls.root / "reports" / "r1"
        reports.mkdir(parents=True)
        speakers = sorted({r["speaker_id"] for r in rows})
        support = {"train|-|genuine|genuine": {"recordings": 10, "speakers": 5}, "train|-|known_family|tts": {"recordings": 20, "speakers": 5},
                   "validation|selection|genuine|genuine": {"recordings": 4, "speakers": 2}, "validation|selection|known_family|tts": {"recordings": 8, "speakers": 2},
                   "test|-|genuine|genuine": {"recordings": 4, "speakers": 2}, "test|-|unseen_final|tts": {"recordings": 4, "speakers": 2}}
        (reports / "split_report.json").write_text(json.dumps({"protocols": {"B": {"support": support}}}), encoding="utf-8")
        (reports / "admission.json").write_text(json.dumps({
            "inputs": {"inspection_dir": "data/manifests/inspection"}, "stems_admitted": 90, "stems_excluded": {}, "speakers": len(speakers),
            "chapters": len(speakers), "per_speaker_counts": {s: 30 + int(s) % 7 for s in speakers}}), encoding="utf-8")
        (reports / "materialize.json").write_text(json.dumps({"recordings": len(rows), "excluded_ids": {}, "exclusions": {}}), encoding="utf-8")
        inspect = cls.root / "data" / "manifests" / "inspection"
        inspect.mkdir(parents=True)
        with open(inspect / "headers.jsonl", "w", encoding="utf-8") as handle:
            for r in rows:
                folder = {"genuine": "dataset/genuine/wav", "melo": "dataset/tts/MeloTTS", "vits": "dataset/tts/VITS"}.get(r["component"], "dataset/tts/Other")
                handle.write(json.dumps({"component": r["component"], "path": f"{folder}/{r['stem']}.wav", "format": "WAV", "subtype": "PCM_16",
                                         "sample_rate": RATE, "channels": 1}) + "\n")
        feats = cls.root / "data" / "features" / "r2_s1_B"
        feats.mkdir(parents=True)
        rng = np.random.default_rng(0)
        comp = np.array(["genuine", "melo", "vits"] * 50)
        np.savez(feats / "lr_train_crops_B_v1.npz", X=rng.standard_normal((150, len(FEATURE_NAMES))), component=comp,
                 y=(comp != "genuine").astype(int), feature_names=np.array(list(FEATURE_NAMES)))
        cls.out = cls.root / "reports" / "eda"
        cls.report = figures.make_all(cls.root, cls.out, cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_family_is_saved_as_png_and_svg_with_two_interpretation_lines(self):
        names = {f["figure"] for f in self.report["figures"]}
        for base in figures.FIGURE_NAMES:
            self.assertTrue(any(n == base or n.startswith(base + "_") for n in names), base)
        self.assertGreaterEqual(self.report["figure_families"], 6)
        for f in self.report["figures"]:
            self.assertEqual(len(f["interpretation"]), 2)
            self.assertTrue(all(len(line) > 40 for line in f["interpretation"]))
            for file in f["files"]:
                self.assertGreater((self.out / file).stat().st_size, 1000, file)
        self.assertEqual(len([f for f in self.report["figures"] if f["figure"].startswith("03_")]), 3)
        self.assertTrue((self.out / "exclusions_table.csv").exists())
        self.assertTrue((self.out / "interpretations.json").exists())

    def test_interpretation_numbers_come_from_the_data(self):
        import pandas as pd
        frame = pd.read_parquet(self.data_dir / measure.OUTPUT_NAMES["stats"])
        duration = next(f for f in self.report["figures"] if f["figure"].startswith("02_"))
        median = frame.loc[frame["component"] == "genuine", "duration_s"].median()
        self.assertIn(f"genuine {median:.1f} s", duration["interpretation"][0])

    def test_tampered_measurements_are_rejected(self):
        path = self.data_dir / measure.OUTPUT_NAMES["stats"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"x")
            with self.assertRaises(figures.EdaError):
                figures.load_inputs(self.root, self.data_dir)
        finally:
            path.write_bytes(original)

    def test_figures_are_reproducible(self):
        with tempfile.TemporaryDirectory() as d:
            again = figures.make_all(self.root, Path(d), self.data_dir)
            self.assertEqual({k: v for k, v in again["files"].items() if k.endswith(".svg") or k.endswith(".csv")},
                             {k: v for k, v in self.report["files"].items() if k.endswith(".svg") or k.endswith(".csv")})


if __name__ == "__main__":
    unittest.main()
