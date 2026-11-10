"""R3-S1 MP3 strength probe: metrics against signals with known answers, alignment, ordering of candidates, and the runner's scope checks."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from echoproof.data import corruption as c
from echoproof.data import corruption_sample as cs
from echoproof.data import mp3_probe as mp
from echoproof.data import mp3_probe_run as run
from echoproof.features.r2_s1 import R2S1Error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r2_s1_features import build_mini_project, waveform  # noqa: E402

SR = 16000


def _ffmpeg_ok():
    try:
        c.find_ffmpeg()
        return True
    except c.CorruptionError:
        return False


def speech(seconds=3.0):
    return cs.fixture_signal(seconds, 0.4).astype(np.float64)


class Metrics(unittest.TestCase):
    def test_identical_signals(self):
        x = speech()
        m = mp.compare(x, x)
        self.assertEqual(m["snr_raw_db"], float("inf"))
        self.assertAlmostEqual(m["gain_db"], 0.0, places=9)
        self.assertEqual(m["lsd_db"], 0.0)
        self.assertAlmostEqual(m["retained_4k_pct"], 100.0, places=6)
        self.assertAlmostEqual(m["retained_7k_pct"], 100.0, places=6)

    def test_a_pure_gain_is_separated_from_damage(self):
        x = speech()
        m = mp.compare(x, x * 0.9487)
        self.assertAlmostEqual(m["gain_db"], 20 * np.log10(0.9487), places=6)
        self.assertAlmostEqual(m["snr_raw_db"], -20 * np.log10(1 - 0.9487), places=4)      # about 25.8 dB: the ceiling a 5% level change imposes
        self.assertGreater(m["snr_gaincomp_db"], 200)                                    # nothing else changed

    def test_added_noise_is_measured_after_removing_the_gain(self):
        rng = np.random.default_rng(1)
        x = speech()
        noise = rng.normal(0, 1, x.size)
        noise *= np.sqrt(np.mean(x ** 2) / 10 ** 2.0) / np.sqrt(np.mean(noise ** 2))         # 20 dB below the signal
        m = mp.compare(x, 0.95 * (x + noise))
        self.assertAlmostEqual(m["snr_gaincomp_db"], 20.0, delta=0.3)
        self.assertLess(m["snr_raw_db"], m["snr_gaincomp_db"])

    def test_lowpass_removes_high_band_energy_and_raises_lsd(self):
        from scipy.signal import butter, sosfiltfilt
        rng = np.random.default_rng(2)
        x = rng.normal(0, 0.1, SR * 3)                                                      # broadband
        y = sosfiltfilt(butter(8, 4000, fs=SR, output="sos"), x)
        m = mp.compare(x, y)
        self.assertLess(m["retained_7k_pct"], 1.0)
        self.assertLess(m["retained_4k_pct"], 5.0)
        self.assertGreater(m["lsd_db"], 10.0)

    def test_inputs_are_validated(self):
        x = speech()
        with self.assertRaises(c.CorruptionError):
            mp.compare(x, x[:-1])
        with self.assertRaises(c.CorruptionError):
            mp.compare(np.zeros(SR), np.zeros(SR))
        with self.assertRaises(c.CorruptionError):
            mp.compare(x, -x)                                                               # sign-flipped: not aligned

    def test_active_frames_are_chosen_on_the_original_only(self):
        x = np.concatenate([speech(1.0), np.zeros(SR)])
        m = mp.compare(x, x * 0.9)
        self.assertLess(m["active_frames"], 1 + (len(x) - mp.N_FFT) // mp.HOP)               # the digital silence is excluded
        self.assertGreater(m["active_frames"], 0)


class Alignment(unittest.TestCase):
    def test_recovers_a_known_delay_and_pads_the_tail(self):
        x = speech()
        for lag in (0, 401, 1105, 2210):
            y = np.concatenate([np.zeros(lag), x, np.zeros(500)])
            aligned, found = mp.align(x, y)
            self.assertEqual(found, lag)
            np.testing.assert_allclose(aligned, x, atol=1e-12)
        aligned, found = mp.align(x, np.concatenate([np.zeros(300), x[:-200]]))
        self.assertEqual((found, len(aligned)), (300, len(x)))

    def test_unrelated_signal_is_rejected(self):
        with self.assertRaises(c.CorruptionError):
            mp.align(speech(), np.random.default_rng(3).normal(0, 0.1, SR * 4 + 1000))


@unittest.skipUnless(_ffmpeg_ok(), "ffmpeg with libmp3lame is not available")
class Candidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = speech(4.0)
        cls.result = mp.probe(cls.x)

    def test_every_candidate_is_measured_with_a_plausible_lag(self):
        self.assertEqual(set(self.result), set(mp.CANDIDATES))
        for name, r in self.result.items():
            self.assertNotIn("error", r, name)
            self.assertTrue(0 <= r["lag_samples"] <= mp.MAX_LAG)
        for name in ("mp3_16k_32", "mp3_16k_64", "mp3_16k_128"):
            self.assertEqual(self.result[name]["lag_samples"], c.MP3_DELAY_SAMPLES)             # agrees with the frozen spec's fixed delay

    def test_version_1_conditions_have_the_level_shift_and_small_damage(self):
        for name in ("mp3_16k_64", "mp3_16k_128"):
            r = self.result[name]
            self.assertAlmostEqual(r["gain_db"], -0.45, delta=0.05)
            self.assertGreater(r["snr_gaincomp_db"], 28.0)                                  # mild: well above the 20 dB noise condition

    def test_the_version_2_condition_is_much_harsher_than_64_kbps(self):
        r32, r64 = self.result["mp3_16k_32"], self.result["mp3_16k_64"]
        self.assertAlmostEqual(r32["gain_db"], -0.47, delta=0.1)
        self.assertTrue(15.0 < r32["snr_gaincomp_db"] < 26.0)                               # about the strength of the 20 dB noise condition
        self.assertGreater(r64["snr_gaincomp_db"] - r32["snr_gaincomp_db"], 6.0)

    def test_lower_bitrate_is_harsher_at_16k(self):
        order = ["mp3_16k_128", "mp3_16k_64", "mp3_16k_32", "mp3_16k_24", "mp3_16k_16", "mp3_16k_8"]
        snr = [self.result[n]["snr_gaincomp_db"] for n in order]
        self.assertEqual(snr, sorted(snr, reverse=True))

    def test_narrowband_candidates_remove_the_high_band(self):
        for name in ("mp3_8k_32", "mp3_8k_16"):
            self.assertLess(self.result[name]["retained_7k_pct"], 1.0)
            self.assertLess(self.result[name]["retained_4k_pct"], 5.0)

    def test_summary_places_candidates_relative_to_the_noise_conditions(self):
        rows = [{"candidate": n, "component": "genuine", **r} for n, r in self.result.items()]
        summary = mp.summarise(rows)
        self.assertIn("milder than noise_20", summary["mp3_16k_64"]["relative_to_noise"])
        self.assertTrue(summary["mp3_16k_8"]["relative_to_noise"].startswith(("between", "at least as harsh")))
        self.assertTrue(summary["mp3_16k_32"]["frozen_spec"] and summary["mp3_16k_128"]["frozen_spec"])
        self.assertFalse(summary["mp3_16k_64"]["frozen_spec"])
        self.assertTrue(summary["mp3_16k_64"]["superseded"] and not summary["mp3_16k_32"]["superseded"])

    def test_figure_is_written_deterministically(self):
        rows = [{"candidate": n, "component": "genuine", **r} for n, r in self.result.items()]
        summary = mp.summarise(rows)
        with tempfile.TemporaryDirectory() as d:
            a = mp.make_figure(summary, Path(d) / "a")
            b = mp.make_figure(summary, Path(d) / "b")
            self.assertEqual((Path(d) / "a.svg").read_bytes(), (Path(d) / "b.svg").read_bytes())
            self.assertEqual(len(a["interpretation"]), 2)
            self.assertTrue((Path(d) / "a.png").stat().st_size > 5000)

    def test_unknown_candidate_and_bad_waveform(self):
        with self.assertRaises(c.CorruptionError):
            mp.probe(self.x, ["mp3_nope"])
        with self.assertRaises(c.CorruptionError):
            mp.probe(np.zeros(100))


@unittest.skipUnless(_ffmpeg_ok(), "ffmpeg with libmp3lame is not available")
class Runner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.mini = build_mini_project(cls.root)
        cls.out = cls.root / "reports" / "r3"
        cls.candidates = ("mp3_16k_64", "mp3_16k_32", "mp3_8k_16")
        cls.report = run.run(cls.root, cls.out, source="shards", shard_dir=cls.mini["shard_dir"], per_component=2, candidates=cls.candidates)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_report_scope_and_counts(self):
        r = self.report
        self.assertEqual(r["role"], "selection")
        self.assertIn("selection role only", r["scope"])
        self.assertEqual((r["sample"]["drawn"], r["sample"]["measured"]), (8, 8))
        self.assertEqual(r["sample"]["measured_components"], ["genuine", "melo", "vits", "xtts"])
        self.assertEqual(r["candidates"], list(self.candidates))
        self.assertEqual(r["errors"], [])
        with open(self.out / run.report_names("shards")["rows"], newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8 * 3)
        selection_ids = {a.recording_id for a in self.mini["assignments"] if a.partition == "validation" and a.validation_role == "selection"}
        self.assertTrue({x["parent_id"] for x in rows} <= selection_ids)
        self.assertFalse(any("chatterbox" in x["parent_id"] for x in rows))

    def test_outputs_are_report_rows_and_figures_only(self):
        suffixes = sorted(p.suffix for p in self.out.iterdir())
        self.assertEqual(suffixes, [".csv", ".json", ".png", ".svg"])

    def test_not_overwritten(self):
        with self.assertRaises(R2S1Error):
            run.run(self.root, self.out, source="shards", shard_dir=self.mini["shard_dir"], per_component=2, candidates=self.candidates)

    def test_local_source_requires_matching_hashes_and_reports_unavailable(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            dataset = Path(d) / "dataset"
            (dataset / "genuine" / "wav").mkdir(parents=True)
            sample = cs.draw_sample(self._plan().items, 2, 42)
            genuine = [i for i in sample if i["component"] == "genuine"]
            for item in genuine:
                stem = item["recording_id"].rsplit(":", 1)[-1]
                sf.write(dataset / "genuine" / "wav" / f"{stem}.wav", self.mini["waves"][item["recording_id"]], SR, subtype="PCM_16")
            out = Path(d) / "out"
            report = run.run(self.root, out, source="local", dataset_dir=dataset, per_component=2, candidates=("mp3_16k_64",))
            self.assertEqual(report["sample"]["measured_components"], ["genuine"])
            self.assertEqual(len(report["sample"]["unavailable_locally"]), 6)
            self.assertTrue(report["sample"]["hash_verified_against_manifest"])
            # a file whose audio differs from the manifest is refused, not used
            bad = genuine[0]["recording_id"].rsplit(":", 1)[-1]
            sf.write(dataset / "genuine" / "wav" / f"{bad}.wav", np.zeros(SR, dtype=np.int16), SR, subtype="PCM_16")
            with self.assertRaises(R2S1Error):
                run.run(self.root, Path(d) / "out2", source="local", dataset_dir=dataset, per_component=2, candidates=("mp3_16k_64",))

    def test_bad_source_arguments(self):
        with self.assertRaises(R2S1Error):
            run.run(self.root, self.out / "x", source="cloud")
        with self.assertRaises(R2S1Error):
            run.run(self.root, self.out / "y", source="shards")
        with self.assertRaises(R2S1Error):
            run.run(self.root, self.out / "z", source="local")

    def _plan(self):
        return cs.load_inputs(self.root, 42)["eda"]


if __name__ == "__main__":
    unittest.main()
