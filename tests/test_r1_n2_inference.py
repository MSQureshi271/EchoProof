import inspect
import io
import json
import unittest

import numpy as np
import soundfile as sf

from echoproof import inference as inf
from echoproof.app_support import present, result_json
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.data import audio
from echoproof.data.chunks import PREPROCESSING_HASH, eval_windows
from echoproof.features.acoustic import FEATURE_NAMES
from echoproof.models.lr_detector import LRDetector
from echoproof import viz

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def wav_bytes(seconds, rate=16000, channels=1, seed=0, amplitude=0.3, fmt="WAV"):
    rng = np.random.default_rng(seed)
    n = int(round(seconds * rate))
    t = np.arange(n) / rate
    x = amplitude * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(n)
    x = np.tile(x[:, None], (1, channels)) if channels > 1 else x
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), rate, format=fmt, subtype="PCM_16" if fmt == "WAV" else None)
    return buf.getvalue()


def bundle(scorer, *, rule=None, calibrate=None, aggregate=inf.mean_probability, score_type="model_score", is_mock=False):
    rule = rule or DecisionRule(1, "threshold", 0.5)
    spec = BundleSpecification(1, "bundle:test", "test-model", PREPROCESSING_HASH, rule, "genuine", "tts")
    return inf.LoadedBundle(spec, scorer, aggregate, calibrate, score_type, is_mock)


def constant(value):
    return lambda chunks: [value] * len(chunks)


class Limits(unittest.TestCase):
    def test_decoder_defaults_are_the_plan_limits(self):
        sig = inspect.signature(audio.decode_audio).parameters
        self.assertEqual(sig["max_bytes"].default, 25 * 1024 * 1024)
        self.assertEqual(sig["max_seconds"].default, 60.0)
        self.assertEqual((inf.MAX_AUDIO_BYTES, inf.MAX_AUDIO_SECONDS), (25 * 1024 * 1024, 60.0))

    def test_a_45_second_clip_now_decodes_by_default(self):
        self.assertEqual(len(audio.decode_audio(wav_bytes(45)).samples), 45 * 16000)


class Success(unittest.TestCase):
    def test_recording_score_is_the_mean_of_chunk_scores_over_canonical_windows(self):
        seen = []

        def scorer(chunks):
            seen.extend(chunks)
            return [0.1 * (i + 1) for i in range(len(chunks))]

        a = inf.analyze(wav_bytes(10.5), bundle(scorer))
        windows = eval_windows(int(10.5 * 16000))
        self.assertEqual(a.windows, windows)
        self.assertEqual([len(c) for c in seen], [b - s for s, b in windows])
        self.assertAlmostEqual(a.result.score, float(np.mean([0.1 * (i + 1) for i in range(len(windows))])), places=12)
        self.assertEqual(a.result.status, "success")
        self.assertAlmostEqual(a.result.covered_duration_seconds, a.result.decoded_duration_seconds, places=9)
        self.assertEqual(len(a.chunk_scores), len(windows))

    def test_threshold_is_score_greater_or_equal(self):
        rule = DecisionRule(1, "threshold", 0.5)
        self.assertEqual(inf.predict_recording(wav_bytes(3), bundle(constant(0.5), rule=rule)).assessment, "synthetic")
        self.assertEqual(inf.predict_recording(wav_bytes(3), bundle(constant(0.4999), rule=rule)).assessment, "genuine")

    def test_all_negative_rule_never_flags(self):
        b = bundle(constant(0.99), rule=DecisionRule(1, "all_negative", None))
        self.assertEqual(inf.predict_recording(wav_bytes(3), b).assessment, "genuine")

    def test_calibration_changes_the_reported_score_and_the_decision_uses_it(self):
        b = bundle(constant(0.6), calibrate=lambda p: p * 0.5, score_type="calibrated_probability")
        r = inf.predict_recording(wav_bytes(3), b)
        self.assertAlmostEqual(r.score, 0.3)
        self.assertEqual((r.assessment, r.score_type), ("genuine", "calibrated_probability"))

    def test_same_bytes_same_result_and_predict_matches_analyze(self):
        data, b = wav_bytes(6, seed=3), inf.mock_bundle()
        r1, r2 = inf.predict_recording(data, b), inf.analyze(data, b).result
        self.assertEqual((r1.score, r1.assessment, r1.warnings), (r2.score, r2.assessment, r2.warnings))

    def test_timings_include_total(self):
        r = inf.predict_recording(wav_bytes(3), bundle(constant(0.2)))
        for key in ("decode", "chunk", "score", "total"):
            self.assertGreaterEqual(r.timings_seconds[key], 0.0)
        self.assertGreaterEqual(r.timings_seconds["total"], r.timings_seconds["decode"])

    def test_short_clip_is_scored_as_one_chunk_with_a_warning(self):
        a = inf.analyze(wav_bytes(1.0), bundle(constant(0.3)))
        self.assertEqual(a.result.status, "success")
        self.assertEqual(len(a.windows), 1)
        self.assertTrue(any("shorter than 4" in w for w in a.result.warnings))

    def test_resampling_and_stereo_are_reported_or_handled(self):
        r = inf.predict_recording(wav_bytes(3, rate=44100, channels=2), bundle(constant(0.2)))
        self.assertEqual(r.status, "success")
        self.assertAlmostEqual(r.decoded_duration_seconds, 3.0, places=3)
        self.assertTrue(any("Resampled" in w for w in r.warnings))

    def test_near_silence_warns(self):
        silent = io.BytesIO()
        sf.write(silent, np.zeros(48000, dtype=np.float32), 16000, format="WAV")
        r = inf.predict_recording(silent.getvalue(), bundle(constant(0.2)))
        self.assertTrue(any("almost silent" in w for w in r.warnings))

    def test_flac_is_accepted(self):
        self.assertEqual(inf.predict_recording(wav_bytes(3, fmt="FLAC"), bundle(constant(0.2))).status, "success")


class Mock(unittest.TestCase):
    def test_mock_output_is_labelled_everywhere(self):
        r = inf.predict_recording(wav_bytes(5), inf.mock_bundle())
        self.assertEqual(r.status, "success")
        self.assertEqual(r.score_type, "mock_score")
        self.assertIn("MOCK", r.model_version)
        self.assertTrue(any("MOCK" in w for w in r.warnings))
        view = present(r)
        self.assertTrue(any("MOCK" in line for line in view["lines"]))
        self.assertEqual(view["level"], "warning")
        self.assertIn("not a detection", view["headline"])

    def test_mock_scores_are_deterministic_and_in_range(self):
        w = [np.linspace(-1, 1, 16000, dtype=np.float32), np.ones(8000, dtype=np.float32)]
        a, b = inf.mock_scorer(w), inf.mock_scorer(w)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(((a >= 0) & (a <= 1)).all())
        self.assertNotEqual(a[0], a[1])


class Failures(unittest.TestCase):
    def check(self, data, reason, **limits):
        r = inf.predict_recording(data, bundle(constant(0.2)), **limits)
        self.assertEqual((r.status, r.failure_reason), ("failed", reason))
        self.assertIsNone(r.score)
        self.assertIsNone(r.assessment)
        self.assertIn("total", r.timings_seconds)
        self.assertIn(reason, inf.FAILURE_MESSAGES)
        return r

    def test_empty_and_wrong_types(self):
        self.check(b"", inf.EMPTY_FILE)
        self.check(None, inf.EMPTY_FILE)
        self.check("text", inf.EMPTY_FILE)

    def test_oversized_file(self):
        self.check(wav_bytes(3), inf.FILE_TOO_LARGE, max_bytes=1000)

    def test_too_long(self):
        self.check(wav_bytes(3), inf.AUDIO_TOO_LONG, max_seconds=2)

    def test_default_limit_rejects_65_seconds(self):
        self.check(wav_bytes(65, rate=8000), inf.AUDIO_TOO_LONG)

    def test_corrupt_and_truncated_audio(self):
        self.check(b"not audio at all" * 50, inf.UNDECODABLE)

    def test_truncated_wav_is_flagged(self):
        good = wav_bytes(6)
        self.assertFalse(inf.wav_looks_truncated(good))
        self.assertTrue(inf.wav_looks_truncated(good[:len(good) // 2]))
        self.assertTrue(inf.wav_looks_truncated(good[:30]))
        self.assertFalse(inf.wav_looks_truncated(wav_bytes(3, fmt="FLAC")))
        half = inf.predict_recording(good[:len(good) // 2], bundle(constant(0.2)))      # the decoder accepts the audio that is there ...
        self.assertEqual(half.status, "success")
        self.assertTrue(any("truncated" in w for w in half.warnings))                   # ... but the user is told
        self.assertEqual(inf.predict_recording(good[:60], bundle(constant(0.2))).failure_reason, inf.INSUFFICIENT_AUDIO)

    def test_more_than_two_channels_is_rejected(self):
        buf = io.BytesIO()
        sf.write(buf, np.zeros((16000, 3), dtype=np.float32), 16000, format="WAV")
        self.check(buf.getvalue(), inf.UNDECODABLE)

    def test_under_one_second_is_insufficient_audio(self):
        r = self.check(wav_bytes(0.5), inf.INSUFFICIENT_AUDIO)
        self.assertAlmostEqual(r.decoded_duration_seconds, 0.5, places=3)
        self.assertEqual(inf.predict_recording(wav_bytes(1.0), bundle(constant(0.2))).status, "success")

    def test_scorer_problems_become_scoring_failed(self):
        def boom(chunks):
            raise RuntimeError("model exploded")
        for scorer in (boom, lambda c: [0.5], constant(1.5), constant(float("nan"))):
            r = inf.predict_recording(wav_bytes(5), bundle(scorer))
            self.assertEqual((r.status, r.failure_reason), ("failed", inf.SCORING_FAILED))
        r = inf.predict_recording(wav_bytes(5), bundle(constant(0.5), calibrate=lambda p: 2.0))
        self.assertEqual(r.failure_reason, inf.SCORING_FAILED)


class RealAdapterShape(unittest.TestCase):
    def test_an_lr_detector_drops_in_as_the_scorer(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, len(FEATURE_NAMES)))
        y = np.array([0, 1] * 20)
        det = LRDetector.fit(X, y, [f"r{i}" for i in range(40)])
        r = inf.predict_recording(wav_bytes(6, seed=5), bundle(det.score_chunks))
        self.assertEqual(r.status, "success")
        self.assertTrue(0.0 <= r.score <= 1.0)
        # the same bytes give the same score twice: the CLI, evaluator and app will all call this one function
        self.assertEqual(r.score, inf.predict_recording(wav_bytes(6, seed=5), bundle(det.score_chunks)).score)


class Presentation(unittest.TestCase):
    def test_success_and_failure_views(self):
        ok = inf.predict_recording(wav_bytes(5), bundle(constant(0.9), score_type="calibrated_probability"))
        view = present(ok)
        self.assertEqual(view["level"], "success")
        self.assertIn("Synthetic", view["headline"])
        self.assertTrue(any("Calibrated probability" in l for l in view["lines"]))
        self.assertTrue(any("not proof" in l for l in view["lines"]))
        bad = present(inf.predict_recording(b"", bundle(constant(0.1))))
        self.assertEqual(bad["level"], "error")
        self.assertNotIn("score", " ".join(bad["lines"]).lower().replace("analyse", ""))

    def test_json_download_round_trips(self):
        ok = inf.predict_recording(wav_bytes(3), bundle(constant(0.2)))
        parsed = json.loads(result_json(ok))
        self.assertEqual(parsed["status"], "success")
        self.assertEqual(parsed["score"], ok.score)
        self.assertEqual(json.loads(result_json(inf.predict_recording(b"", bundle(constant(0.2)))))["failure_reason"], inf.EMPTY_FILE)


class Visuals(unittest.TestCase):
    def test_envelope_keeps_peaks_and_bounds_columns(self):
        x = np.zeros(160000, dtype=np.float32)
        x[12345], x[99999] = 0.9, -0.8
        times, lo, hi = viz.waveform_envelope(x, 16000, columns=500)
        self.assertEqual(len(times), 500)
        self.assertAlmostEqual(float(hi.max()), 0.9, places=6)
        self.assertAlmostEqual(float(lo.min()), -0.8, places=6)
        self.assertTrue(np.all(np.diff(times) > 0))
        with self.assertRaises(ValueError):
            viz.waveform_envelope(np.array([]), 16000)

    def test_mel_axes_match_the_feature_extractor(self):
        from echoproof.features.acoustic import log_mel
        x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
        mel = log_mel(x)
        times, freqs = viz.mel_axes(mel.shape[0], 16000)
        self.assertEqual((len(times), len(freqs)), mel.shape)
        self.assertTrue(np.all(np.diff(freqs) > 0))
        self.assertTrue(0 < freqs[0] and freqs[-1] < 8000)

    @unittest.skipUnless(HAVE_MPL, "matplotlib is not installed")
    def test_figures_have_labelled_axes_and_a_colour_bar(self):
        x = np.random.default_rng(1).standard_normal(48000).astype(np.float32) * 0.1
        fig = viz.waveform_figure(x, 16000, windows=eval_windows(48000))
        self.assertEqual((fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()), ("Time (s)", "Amplitude"))
        viz.close(fig)
        fig = viz.logmel_figure(x, 16000)
        self.assertEqual((fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()), ("Time (s)", "Frequency (Hz)"))
        self.assertEqual(len(fig.axes), 2)                       # image axis plus colour bar
        self.assertEqual(fig.axes[1].get_ylabel(), "log power")
        viz.close(fig)

    @unittest.skipIf(HAVE_MPL, "matplotlib is installed")
    def test_missing_matplotlib_gives_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            viz.waveform_figure(np.zeros(16000, dtype=np.float32), 16000)


if __name__ == "__main__":
    unittest.main()
