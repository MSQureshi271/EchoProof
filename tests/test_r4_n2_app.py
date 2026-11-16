"""R4-N2: app start-up verification, privacy, busy gate, accessibility text and Streamlit AppTest flows. Synthetic audio only."""
import hashlib
import io
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from echoproof import app_support as support
from echoproof import inference as inf
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH

try:
    from streamlit.testing.v1 import AppTest
    HAVE_STREAMLIT = True
except ImportError:
    HAVE_STREAMLIT = False

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "main.py")


def audio_bytes(seconds=5.0, fmt="WAV", rate=16000, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    x = 0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / rate) + 0.02 * rng.standard_normal(n)
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), rate, format=fmt, subtype="PCM_16" if fmt == "WAV" else None)
    return buf.getvalue()


def fake_bundle(score_type="calibrated_probability", threshold=0.4, scorer=None, is_mock=False):
    def default_scorer(chunks):
        return [float(min(0.99, 0.2 + 3 * float(np.mean(np.abs(c))))) for c in chunks]
    spec = BundleSpecification(1, "bundle:test-fake", "fake:test", PREPROCESSING_HASH, DecisionRule(1, "threshold", threshold), "genuine", "tts")
    return inf.LoadedBundle(spec, scorer or default_scorer, inf.mean_probability, (lambda p: p * 0.9) if score_type == "calibrated_probability" else None,
                            score_type, is_mock)


def make_lock(directory: Path, components: dict, manifest_bytes: bytes):
    """A tiny bundle folder plus a lock that describes it, in the lock's own layout."""
    bundle = directory / "bundle"
    (bundle / "model").mkdir(parents=True)
    (bundle / "bundle.json").write_bytes(manifest_bytes)
    entry = {"bundle_json_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "components": {}}
    for name, content in components.items():
        (bundle / name).write_bytes(content)
        entry["components"][name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    lock = directory / "lock.json"
    lock.write_text(json.dumps({"candidates": {"WavLM": entry}, "limitations": ["WavLM at its locked threshold flags 49 of 260 genuine operating recordings (18.8%); x."]}))
    return bundle, lock


class StartupVerification(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bundle, self.lock = make_lock(self.tmp, {"model/a.bin": b"alpha", "model/b.bin": b"beta"}, b'{"x": 1}')

    def test_matching_bundle_verifies(self):
        self.assertEqual(support.verify_bundle_against_lock(self.bundle, self.lock), (None, ""))

    def test_missing_bundle_fails_closed_with_a_distinct_message(self):
        code, message = support.verify_bundle_against_lock(self.tmp / "nope", self.lock)
        self.assertEqual(code, support.BUNDLE_MISSING)
        for text in inf.FAILURE_MESSAGES.values():
            self.assertNotEqual(message, text)

    def test_changed_manifest_or_component_or_missing_component(self):
        (self.bundle / "model/a.bin").write_bytes(b"ALPHA")
        self.assertEqual(support.verify_bundle_against_lock(self.bundle, self.lock)[0], support.BUNDLE_CHANGED)
        (self.bundle / "model/a.bin").write_bytes(b"alpha")
        (self.bundle / "model/b.bin").unlink()
        self.assertEqual(support.verify_bundle_against_lock(self.bundle, self.lock)[0], support.BUNDLE_CHANGED)
        (self.bundle / "model/b.bin").write_bytes(b"beta")
        (self.bundle / "bundle.json").write_bytes(b'{"x": 2}')
        self.assertEqual(support.verify_bundle_against_lock(self.bundle, self.lock)[0], support.BUNDLE_CHANGED)

    def test_unreadable_lock(self):
        self.assertEqual(support.verify_bundle_against_lock(self.bundle, self.tmp / "none.json")[0], support.BUNDLE_LOCK_UNAVAILABLE)

    def test_prepare_bundle_production_path_verified_loaded_and_warned(self):
        root = self.tmp / "repo"
        (root / "bundles").mkdir(parents=True)
        (root / "locks").mkdir()
        bundle, lock = make_lock(root, {"model/a.bin": b"alpha"}, b"{}")
        shutil.copytree(bundle, root / support.PRODUCTION_BUNDLE_RELATIVE)
        shutil.copy(lock, root / support.PRODUCTION_LOCK_RELATIVE)
        loaded = []
        status = support.prepare_bundle(root=root, loader=lambda p: loaded.append(p) or fake_bundle(), installed_transformers="5.22.0")
        self.assertTrue(status.ok and status.verified_against_lock)
        self.assertEqual(status.warnings, ())
        self.assertEqual(len(loaded), 1)
        status = support.prepare_bundle(root=root, loader=lambda p: fake_bundle(), installed_transformers="4.0.0")
        self.assertTrue(status.ok)
        self.assertTrue(any("5.22.0" in w for w in status.warnings))
        (root / support.PRODUCTION_BUNDLE_RELATIVE / "model/a.bin").write_bytes(b"tampered")
        loaded.clear()
        status = support.prepare_bundle(root=root, loader=lambda p: loaded.append(p) or fake_bundle())
        self.assertFalse(status.ok)
        self.assertEqual(status.code, support.BUNDLE_CHANGED)
        self.assertEqual(loaded, [])                                  # a changed bundle is never loaded

    def test_missing_production_bundle_and_override_and_mock(self):
        root = self.tmp / "repo2"
        (root / "locks").mkdir(parents=True)
        shutil.copy(self.lock, root / support.PRODUCTION_LOCK_RELATIVE)
        self.assertEqual(support.prepare_bundle(root=root).code, support.BUNDLE_MISSING)
        override = self.tmp / "other"
        override.mkdir()
        status = support.prepare_bundle(override, root=root, loader=lambda p: fake_bundle(), installed_transformers="5.22.0")
        self.assertTrue(status.ok and not status.verified_against_lock)
        self.assertTrue(any("not the locked production bundle" in w for w in status.warnings))
        status = support.prepare_bundle(override, root=root, loader=lambda p: inf.mock_bundle())
        self.assertEqual((status.ok, status.code), (False, support.BUNDLE_IS_MOCK))
        status = support.prepare_bundle(override, root=root, loader=lambda p: 1 / 0)
        self.assertEqual(status.code, support.BUNDLE_LOAD_FAILED)

    def test_default_target_is_the_locked_production_bundle(self):
        self.assertEqual(support.PRODUCTION_BUNDLE_RELATIVE, "bundles/r3_n1_wavlm_head_B_calibrated")
        lock = json.loads((ROOT / support.PRODUCTION_LOCK_RELATIVE).read_text(encoding="utf-8"))
        self.assertEqual(lock["candidates"]["WavLM"]["bundle_path"], support.PRODUCTION_BUNDLE_RELATIVE)
        self.assertIn("49 of 260", support.policy_cost_note())

    def test_transformers_version_warning(self):
        self.assertIsNone(support.transformers_version_warning(installed="5.22.0"))
        self.assertIn("5.22.0", support.transformers_version_warning(installed="5.0.0"))


class HelpersAndText(unittest.TestCase):
    def test_score_type_warning(self):
        self.assertIsNone(support.score_type_warning("calibrated_probability"))
        self.assertIn("NOT a calibrated probability", support.score_type_warning("model_score"))

    def test_scope_statement_content(self):
        for phrase in ("English", "text-to-speech", "in memory", "not saved", "hosted", "sent to that server"):
            self.assertIn(phrase, support.SCOPE_STATEMENT)

    def test_outcome_and_alt_text_are_plain_text(self):
        a = inf.analyze(audio_bytes(6), fake_bundle())
        self.assertIn("Outcome:", support.outcome_text(a.result))
        self.assertIn("failed", support.outcome_text(inf.analyze(b"", fake_bundle()).result))
        self.assertIn("Waveform", support.waveform_alt_text(a))
        self.assertIn("Log-mel", support.logmel_alt_text(a))


class BusyGate(unittest.TestCase):
    def test_second_caller_is_refused_without_blocking(self):
        gate = support.InferenceGate()
        entered, release = threading.Event(), threading.Event()
        out = {}

        def slow():
            entered.set()
            release.wait(5)
            return "done"

        worker = threading.Thread(target=lambda: out.update(first=gate.run(slow)))
        worker.start()
        self.assertTrue(entered.wait(5))
        self.assertTrue(gate.busy)
        self.assertEqual(gate.run(lambda: "never"), (False, None))
        release.set()
        worker.join(5)
        self.assertEqual(out["first"], (True, "done"))
        self.assertFalse(gate.busy)
        self.assertEqual(gate.run(lambda: 7), (True, 7))

    def test_gate_is_released_after_an_exception(self):
        gate = support.InferenceGate()
        with self.assertRaises(RuntimeError):
            gate.run(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        self.assertFalse(gate.busy)


def _mp3_supported():
    try:
        buf = io.BytesIO()
        sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="MP3")
        return True
    except Exception:
        return False


class NoTemporaryFiles(unittest.TestCase):
    """Decoding must not touch the temp directory: it is redirected to an empty folder, every tempfile creator is made to fail,
    and the folder must still be empty afterwards, on success, error and refusal."""

    def run_case(self, data):
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, empty, True)

        def forbidden(*a, **k):
            raise AssertionError("temporary file used")

        names = ("mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryFile", "TemporaryDirectory", "SpooledTemporaryFile")
        with mock.patch.multiple(tempfile, **{n: forbidden for n in names}), mock.patch.object(tempfile, "tempdir", str(empty)):
            ran, analysis = support.INFERENCE_GATE.run(inf.analyze, data, fake_bundle())
        self.assertTrue(ran)
        self.assertEqual(list(empty.iterdir()), [])
        return analysis

    def test_wav_flac_ogg_succeed_without_temp_files(self):
        for fmt in ("WAV", "FLAC", "OGG"):
            with self.subTest(fmt=fmt):
                self.assertEqual(self.run_case(audio_bytes(5, fmt=fmt)).result.status, "success")

    @unittest.skipUnless(_mp3_supported(), "this libsndfile cannot encode MP3")
    def test_mp3_without_temp_files(self):
        self.assertEqual(self.run_case(audio_bytes(5, fmt="MP3")).result.status, "success")

    def test_errors_without_temp_files(self):
        for data in (b"", b"not audio" * 100, audio_bytes(0.5), audio_bytes(3)[:30]):
            with self.subTest(size=len(data)):
                self.assertNotEqual(self.run_case(data).result.status, "success")

    def test_scoring_failure_without_temp_files(self):
        def boom(chunks):
            raise RuntimeError("x")
        self.assertEqual(self.run_case_with(boom).result.failure_reason, inf.SCORING_FAILED)

    def run_case_with(self, scorer):
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, empty, True)
        with mock.patch.object(tempfile, "tempdir", str(empty)):
            _, analysis = support.INFERENCE_GATE.run(inf.analyze, audio_bytes(5), fake_bundle(scorer=scorer))
        self.assertEqual(list(empty.iterdir()), [])
        return analysis

    def test_app_source_never_uses_tempfile_or_user_file_names(self):
        for source in (Path(APP).read_text(encoding="utf-8"), Path(support.__file__).read_text(encoding="utf-8")):
            self.assertNotIn("tempfile", source.replace("no temporary", ""))
        self.assertNotIn("upload.name", Path(APP).read_text(encoding="utf-8"))
        self.assertNotIn("cache_data", Path(APP).read_text(encoding="utf-8"))


def status_ok(**kw):
    return support.BundleStatus(True, None, "", fake_bundle(**kw), (), True)


@unittest.skipUnless(HAVE_STREAMLIT, "streamlit is not installed")
class AppFlows(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    def start(self, status=None, timeout=60):
        patcher = mock.patch.object(support, "prepare_bundle", lambda *a, **k: status or status_ok())
        patcher.start()
        self.addCleanup(patcher.stop)
        at = AppTest.from_file(APP, default_timeout=timeout)
        return at.run()

    def texts(self, at):
        parts = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.error] + [e.value for e in at.warning] + [e.value for e in at.success] + [e.value for e in at.info]
        return "\n".join(map(str, parts))

    def analyze(self, at, data, name="clip.wav"):
        at.file_uploader[0].upload(name, data)
        at.run()
        next(b for b in at.button if b.label.startswith("Analyze")).click()
        return at.run()

    def test_startup_shows_scope_and_privacy_before_upload(self):
        at = self.start()
        self.assertFalse(at.exception)
        text = self.texts(at)
        for phrase in ("English", "in memory", "hosted", "not proof"):
            self.assertIn(phrase, text)
        self.assertIn("Choose a recording", text)

    def test_startup_failure_is_a_model_error_not_an_audio_error(self):
        for code, message in ((support.BUNDLE_MISSING, "The production model bundle was not found"), (support.BUNDLE_CHANGED, "does not match the locked version")):
            with self.subTest(code=code):
                at = self.start(support.BundleStatus(False, code, message))
                self.assertFalse(at.exception)
                self.assertEqual(len(at.file_uploader), 0)             # nothing can be analysed
                self.assertIn(message, self.texts(at))
                self.assertIn("Model unavailable", self.texts(at))
                for audio_message in inf.FAILURE_MESSAGES.values():
                    self.assertNotIn(audio_message, self.texts(at))

    def test_transformers_and_override_warnings_are_shown(self):
        at = self.start(support.BundleStatus(True, None, "", fake_bundle(), ("transformers 4.0 is installed but the model was verified with 5.22.0",), False))
        self.assertIn("5.22.0", self.texts(at))

    def test_upload_then_analyze_shows_text_outcome_figures_alt_text_and_cost(self):
        at = self.start()
        at = self.analyze(at, audio_bytes(6))
        self.assertFalse(at.exception)
        text = self.texts(at)
        self.assertIn("Outcome:", text)
        self.assertIn("Calibrated probability of synthetic speech", text)
        self.assertIn("Waveform of the", text)
        self.assertIn("Log-mel spectrogram", text)
        self.assertNotIn("NOT a calibrated probability", text)
        self.assertEqual(len(at.expander), 1)
        self.assertIn("49 of 260", "\n".join(str(m.value) for m in at.expander[0].markdown))
        self.assertEqual(len(at.get("download_button")), 1)

    def test_uncalibrated_score_type_is_warned_explicitly(self):
        at = self.start(status_ok(score_type="model_score"))
        self.assertIn("NOT a calibrated probability", self.texts(at))
        at = self.analyze(at, audio_bytes(6))
        self.assertIn("Model score", self.texts(at))

    def test_each_failure_message(self):
        cases = {inf.EMPTY_FILE: b"", inf.UNDECODABLE: b"junk" * 200, inf.INSUFFICIENT_AUDIO: audio_bytes(0.5), inf.AUDIO_TOO_LONG: audio_bytes(65, rate=8000)}
        for reason, data in cases.items():
            with self.subTest(reason=reason):
                at = self.start()
                at.file_uploader[0].upload("a.wav", data)
                at.run()
                next(b for b in at.button if b.label.startswith("Analyze")).click()
                at.run()
                self.assertFalse(at.exception)
                text = self.texts(at)
                self.assertIn(inf.FAILURE_MESSAGES[reason], text)
                self.assertIn("Outcome: analysis failed", text)
                self.assertNotIn("Calibrated probability", text)

    def test_file_too_large_message(self):
        # the app calls analyze with the library limits, so the message is checked through the same function with a small limit
        self.assertEqual(inf.analyze(audio_bytes(3), fake_bundle(), max_bytes=1000).result.failure_reason, inf.FILE_TOO_LARGE)
        self.assertIn("25 MiB", inf.FAILURE_MESSAGES[inf.FILE_TOO_LARGE])

    def test_scoring_failure_message(self):
        def boom(chunks):
            raise RuntimeError("x")
        at = self.start(status_ok(scorer=boom))
        at = self.analyze(at, audio_bytes(5))
        self.assertIn(inf.FAILURE_MESSAGES[inf.SCORING_FAILED], self.texts(at))

    def test_busy_state_message_and_button_disabled_while_running(self):
        at = self.start()
        at.file_uploader[0].upload("a.wav", audio_bytes(5))
        at.run()
        self.assertFalse(next(b for b in at.button if b.label.startswith("Analyze")).disabled)
        self.assertTrue(support.INFERENCE_GATE._lock.acquire(blocking=False))     # another request holds the single inference slot
        try:
            at.run()
            self.assertTrue(next(b for b in at.button if b.label.startswith("Analyze")).disabled)
            at.session_state["busy"] = True                                      # this session was already clicked: race with the other holder
            at.run()
            self.assertIn(support.BUSY_MESSAGE, self.texts(at))
            self.assertFalse(at.session_state["busy"])
            self.assertNotIn("analysis", at.session_state)                       # no result was invented
        finally:
            support.INFERENCE_GATE._lock.release()
        at.run()
        self.assertFalse(next(b for b in at.button if b.label.startswith("Analyze")).disabled)

    def test_result_cleared_when_another_file_is_chosen_and_on_reset(self):
        at = self.start()
        at = self.analyze(at, audio_bytes(6, seed=1))
        self.assertIn("analysis", at.session_state)
        at.file_uploader[0].upload("b.wav", audio_bytes(6, seed=2))
        at.run()
        self.assertNotIn("analysis", at.session_state)
        self.assertNotIn("Outcome:", self.texts(at))
        at = self.analyze(at, audio_bytes(6, seed=3))
        self.assertIn("analysis", at.session_state)
        next(b for b in at.button if b.label.startswith("Clear")).click()
        at.run()
        self.assertNotIn("analysis", at.session_state)
        self.assertIn("Choose a recording", self.texts(at))

    def test_app_session_result_holds_no_filename(self):
        at = self.start()
        at = self.analyze(at, audio_bytes(5), name="private_name_voice.wav")
        dump = repr(at.session_state["analysis"])                 # the app's own state; the uploader widget state is Streamlit's
        self.assertNotIn("private_name_voice", dump)

    def test_no_temp_files_during_app_run(self):
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, empty, True)
        at = self.start()
        with mock.patch.object(tempfile, "tempdir", str(empty)):
            at = self.analyze(at, audio_bytes(5))
        self.assertFalse(at.exception)
        self.assertEqual(list(empty.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
