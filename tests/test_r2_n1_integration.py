"""R2-N1 integration tests: LR end-to-end integration.

Verifies:
1. Production LR bundle loading, metadata, and specification validation.
2. Full inference pipeline with real LR model on genuine and synthetic waveforms.
3. Strict parity across shared inference, CLI, and Streamlit presentation logic.
4. Robust error handling for corrupt bundles, missing files, empty/truncated audio.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest

from echoproof.app_support import NOT_PROOF, present, result_json
from echoproof.contracts import (
    Chunk,
    ChunkScore,
    ContractError,
    Recording,
    SplitAssignment,
)
from echoproof.data.audio import decode_audio
from echoproof.data.chunks import PREPROCESSING_HASH, eval_windows
from echoproof.evaluation.aggregate import aggregate_from_contracts, aggregate_recordings
from echoproof.inference import (
    EMPTY_FILE,
    INSUFFICIENT_AUDIO,
    MOCK_WARNING,
    UNDECODABLE,
    analyze,
    load_lr_bundle,
    predict_recording,
)

RUN_DIR = Path("runs/r2/r2_m1_20261106T140611Z")
MODEL_DIR = RUN_DIR / "model"


def make_wav_bytes(duration_seconds: float = 2.0, sample_rate: int = 16000, freq: float = 220.0) -> bytes:
    """Generate in-memory mono PCM_16 WAV bytes."""
    n_samples = int(round(duration_seconds * sample_rate))
    t = np.arange(n_samples) / sample_rate
    audio = 0.25 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestLRBundleLoading(unittest.TestCase):
    def test_load_from_model_dir_and_run_dir(self):
        b1 = load_lr_bundle(MODEL_DIR)
        b2 = load_lr_bundle(RUN_DIR)
        for b in (b1, b2):
            self.assertFalse(b.is_mock)
            self.assertEqual(b.score_type, "model_score")
            self.assertEqual(b.spec.bundle_id, "bundle:r2-lr-protocol-b")
            self.assertEqual(b.spec.model_version, "logistic_regression:B")
            self.assertEqual(b.spec.preprocessing_hash, PREPROCESSING_HASH)
            self.assertEqual(b.spec.decision_rule.threshold, 0.5)
            self.assertEqual(b.spec.class_0, "genuine")
            self.assertEqual(b.spec.class_1, "tts")

    def test_load_custom_threshold(self):
        b = load_lr_bundle(MODEL_DIR, threshold=0.75)
        self.assertEqual(b.spec.decision_rule.threshold, 0.75)

    def test_invalid_threshold_raises_contract_error(self):
        with self.assertRaises(ContractError):
            load_lr_bundle(MODEL_DIR, threshold=1.5)
        with self.assertRaises(ContractError):
            load_lr_bundle(MODEL_DIR, threshold=-0.1)

    def test_missing_directory_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_lr_bundle(Path("runs/r2/non_existent_run_directory"))

    def test_corrupted_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_dir = Path(tmpdir)
            (bad_dir / "weights.npz").write_bytes(b"corrupted data")
            (bad_dir / "lr_model.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(Exception):
                load_lr_bundle(bad_dir)


class TestRealInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_lr_bundle(MODEL_DIR)

    def test_valid_audio_produces_successful_result(self):
        wav = make_wav_bytes(duration_seconds=3.0)
        res = predict_recording(wav, self.bundle)
        self.assertEqual(res.status, "success")
        self.assertIsNone(res.failure_reason)
        self.assertEqual(res.model_version, "logistic_regression:B")
        self.assertEqual(res.score_type, "model_score")
        self.assertIn(res.assessment, ("genuine", "synthetic"))
        self.assertTrue(0.0 <= res.score <= 1.0)
        self.assertAlmostEqual(res.decoded_duration_seconds, 3.0, places=2)
        self.assertAlmostEqual(res.covered_duration_seconds, 3.0, places=2)
        self.assertNotIn(MOCK_WARNING, res.warnings)
        for key in ("decode", "chunk", "score", "total"):
            self.assertIn(key, res.timings_seconds)
            self.assertGreater(res.timings_seconds[key], 0.0)

    def test_short_audio_warning_present_when_under_window(self):
        wav = make_wav_bytes(duration_seconds=1.5)
        res = predict_recording(wav, self.bundle)
        self.assertEqual(res.status, "success")
        self.assertTrue(any("shorter than 4 s" in w for w in res.warnings))

    def test_empty_audio_fails_with_standard_contract(self):
        res = predict_recording(b"", self.bundle)
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.failure_reason, EMPTY_FILE)
        self.assertIsNone(res.score)
        self.assertIsNone(res.assessment)

    def test_undecodable_audio_fails(self):
        res = predict_recording(b"RIFFjunkdataNOTAWAV", self.bundle)
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.failure_reason, UNDECODABLE)
        self.assertIsNone(res.score)

    def test_insufficient_audio_fails(self):
        wav = make_wav_bytes(duration_seconds=0.5)
        res = predict_recording(wav, self.bundle)
        self.assertEqual(res.status, "failed")
        self.assertEqual(res.failure_reason, INSUFFICIENT_AUDIO)
        self.assertIsNone(res.score)


class TestParity(unittest.TestCase):
    """Parity across Evaluator / shared inference, CLI, and Streamlit app."""

    @classmethod
    def setUpClass(cls):
        cls.bundle = load_lr_bundle(MODEL_DIR)

    def test_predict_and_analyze_parity(self):
        """Verifies that the contract entry point predict_recording() faithfully preserves
        the complete analysis.result produced by analyze() without mutation.

        Note: This verifies the contract wrapper boundary; full evaluator and frontend
        parity are tested in dedicated methods below.
        """
        wav = make_wav_bytes(duration_seconds=4.5, freq=300.0)
        analysis = analyze(wav, self.bundle)
        pred = predict_recording(wav, self.bundle)

        self.assertEqual(analysis.result.score, pred.score)
        self.assertEqual(analysis.result.assessment, pred.assessment)
        self.assertEqual(analysis.result.status, pred.status)
        self.assertEqual(analysis.result.score_type, pred.score_type)
        self.assertEqual(analysis.result.model_version, pred.model_version)
        self.assertEqual(analysis.result.warnings, pred.warnings)
        self.assertAlmostEqual(analysis.raw_score, pred.score, places=12)

    def test_evaluator_scoring_boundary_parity(self):
        """Verifies parity between shared inference and the R1-N1 evaluation package.

        Architectural context:
        The R1-N1 evaluator does not evaluate standalone raw audio bytes directly; it evaluates
        chunk manifests and contract objects (Recording, Chunk, ChunkScore, SplitAssignment)
        via aggregate_recordings() and aggregate_from_contracts().

        The shared scoring boundary is:
        1. Audio is canonically decoded to 16 kHz mono samples.
        2. Chunks are extracted using the canonical eval_windows() definition.
        3. Chunks are scored using the exact same production LR-B model (LRDetector.score_chunks).
        4. Chunk scores are aggregated using R1-N1 evaluation functions (aggregate_recordings
           and aggregate_from_contracts).

        This test proves that the recording-level score produced by the R1-N1 evaluation package
        matches predict_recording() down to machine precision (<= 1e-12).
        """
        wav = make_wav_bytes(duration_seconds=5.0, freq=240.0)
        shared_result = predict_recording(wav, self.bundle)
        self.assertEqual(shared_result.status, "success")

        # R1-N1 evaluator path: canonical decode + eval_windows + LR-B scorer + R1-N1 aggregate
        wave = decode_audio(wav)
        windows = eval_windows(len(wave.samples))
        chunk_waveforms = [wave.samples[a:b] for a, b in windows]
        chunk_scores = [float(s) for s in self.bundle.scorer(chunk_waveforms)]

        # 1. R1-N1 aggregate_recordings
        eval_scores = aggregate_recordings(
            chunk_ids=[f"chunk:test:{i}" for i in range(len(chunk_waveforms))],
            recording_ids=["rec:test:001"] * len(chunk_waveforms),
            labels=[1] * len(chunk_waveforms),
            chunk_scores=chunk_scores,
        )
        self.assertEqual(len(eval_scores.scores), 1)
        self.assertAlmostEqual(eval_scores.scores[0], shared_result.score, places=12)

        # 2. R1-N1 aggregate_from_contracts with formal metadata objects
        rec = Recording(
            schema_version=1,
            recording_id="rec:test:001",
            original_path="test.wav",
            dataset="test_corpus",
            dataset_revision="v1",
            label=1,
            label_verified=True,
            method="tts",
            method_verified=True,
            generator_family="melo",
            generator_version="1",
            language="en",
            source_utterance_id=None,
            source_speaker_id=None,
            target_voice_id=None,
            reference_id=None,
            transcript_hash=None,
            duration_seconds=len(wave.samples) / wave.sample_rate,
            sample_rate=wave.sample_rate,
            channels=1,
            codec="pcm_s16le",
            sha256=hashlib.sha256(wav).hexdigest(),
            quality_flags=[],
            retained=True,
            exclusion_reason=None,
            license_ref=None,
        )
        split = SplitAssignment(
            schema_version=1,
            protocol_id="protocol:B",
            recording_id="rec:test:001",
            group_id="speaker:001",
            partition="test",
            validation_role=None,
            family_policy="unseen",
            split_version="1",
            split_hash="0" * 64,
        )
        chunks = [
            Chunk(
                schema_version=1,
                chunk_id=f"chunk:protocol:B:rec:test:001:{start}",
                recording_id="rec:test:001",
                group_id="speaker:001",
                protocol_id="protocol:B",
                partition="test",
                start_sample=start,
                end_sample=end,
                sample_rate=wave.sample_rate,
                start_seconds=start / wave.sample_rate,
                end_seconds=end / wave.sample_rate,
                valid_samples=end - start,
                input_samples=4 * wave.sample_rate,
                preprocessing_hash=PREPROCESSING_HASH,
            )
            for start, end in windows
        ]
        scores = [
            ChunkScore(chunk_id=c.chunk_id, tts_probability=s)
            for c, s in zip(chunks, chunk_scores)
        ]
        contract_rec_scores = aggregate_from_contracts(
            recordings=[rec],
            chunks=chunks,
            scores=scores,
            assignments=[split],
            protocol_id="protocol:B",
        )
        self.assertEqual(len(contract_rec_scores.scores), 1)
        self.assertAlmostEqual(contract_rec_scores.scores[0], shared_result.score, places=12)

    def test_streamlit_app_end_to_end_parity(self):
        """Verifies end-to-end parity with the real Streamlit application.

        Runs app/main.py via Streamlit's official AppTest framework:
        1. Injects real audio bytes into st.file_uploader.
        2. Clicks the 'Analyze' button to trigger st.session_state['analysis'] = (key, analyze(data, bundle)).
        3. Asserts that the app loads the real LR-B bundle (is_mock == False, model_version == 'logistic_regression:B').
        4. Asserts that the resulting score, assessment, and presentation match shared inference and CLI.
        """
        wav = make_wav_bytes(duration_seconds=2.0, freq=220.0)
        expected = predict_recording(wav, self.bundle)

        app_path = str((Path(__file__).resolve().parents[1] / "app" / "main.py").resolve())
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        self.assertEqual(len(at.error), 0, f"App errors on init: {[e.value for e in at.error]}")

        # Upload audio bytes and click Analyze
        at.file_uploader[0].upload("sample.wav", wav)
        at.run()
        self.assertEqual(len(at.error), 0, f"App errors on upload: {[e.value for e in at.error]}")

        # Find and click the 'Analyze' button
        self.assertGreater(len(at.button), 0, "No Analyze button found in Streamlit app")
        analyze_btn = [b for b in at.button if b.label == "Analyze"][0]
        analyze_btn.click().run()

        self.assertEqual(len(at.error), 0, f"App errors on analyze: {[e.value for e in at.error]}")
        self.assertIn("analysis", at.session_state, "Session state missing 'analysis'")

        stored_key, app_analysis = at.session_state["analysis"]
        self.assertEqual(stored_key, hashlib.sha256(wav).hexdigest())

        # Assert score parity with shared inference
        self.assertEqual(app_analysis.result.status, "success")
        self.assertEqual(app_analysis.result.assessment, expected.assessment)
        self.assertEqual(app_analysis.result.model_version, expected.model_version)
        self.assertEqual(app_analysis.result.score_type, expected.score_type)
        self.assertAlmostEqual(app_analysis.result.score, expected.score, places=12)

        # Assert UI presentation reflects real assessment
        view = present(expected)
        success_texts = [s.value for s in at.success]
        self.assertTrue(
            any(view["headline"] in t for t in success_texts),
            f"Expected headline {view['headline']!r} not in UI success elements: {success_texts}",
        )

    def test_cli_parity_with_shared_inference(self):
        wav = make_wav_bytes(duration_seconds=2.5, freq=180.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
            audio_path = f.name

        try:
            expected = predict_recording(wav, self.bundle)
            proc = subprocess.run(
                [sys.executable, "-m", "echoproof.cli", "predict", "--bundle", str(MODEL_DIR), "--audio", audio_path],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"CLI stderr: {proc.stderr}")
            cli_result = json.loads(proc.stdout)

            self.assertEqual(cli_result["status"], "success")
            self.assertEqual(cli_result["assessment"], expected.assessment)
            self.assertAlmostEqual(cli_result["score"], expected.score, places=12)
            self.assertEqual(cli_result["score_type"], expected.score_type)
            self.assertEqual(cli_result["model_version"], expected.model_version)
            self.assertEqual(cli_result["warnings"], expected.warnings)
            self.assertAlmostEqual(cli_result["decoded_duration_seconds"], expected.decoded_duration_seconds, places=4)
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_cli_failure_on_corrupt_audio(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"not audio")
            audio_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "echoproof.cli", "predict", "--bundle", str(MODEL_DIR), "--audio", audio_path],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 4)
            cli_result = json.loads(proc.stdout)
            self.assertEqual(cli_result["status"], "failed")
            self.assertEqual(cli_result["failure_reason"], UNDECODABLE)
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_cli_missing_files_error_code(self):
        proc = subprocess.run(
            [sys.executable, "-m", "echoproof.cli", "predict", "--bundle", str(MODEL_DIR), "--audio", "nonexistent.wav"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        err = json.loads(proc.stderr)
        self.assertEqual(err["type"], "audio_not_found")

        proc2 = subprocess.run(
            [sys.executable, "-m", "echoproof.cli", "predict", "--bundle", "nonexistent_dir", "--audio", "nonexistent.wav"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc2.returncode, 2)
        err2 = json.loads(proc2.stderr)
        self.assertEqual(err2["type"], "bundle_error")


class TestPresentationAndApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_lr_bundle(MODEL_DIR)

    def test_presentation_shows_production_labels_and_no_mock(self):
        wav = make_wav_bytes(duration_seconds=2.0)
        res = predict_recording(wav, self.bundle)
        view = present(res)

        self.assertEqual(view["level"], "success")
        self.assertNotIn("MOCK", view["headline"])
        self.assertTrue(any("Model score:" in line for line in view["lines"]))
        self.assertTrue(any("Model: logistic_regression:B" in line for line in view["lines"]))
        self.assertIn(NOT_PROOF, view["lines"])

    def test_app_main_load_bundle(self):
        import app.main as app_main
        loaded = app_main.load_bundle()
        self.assertFalse(loaded.is_mock)
        self.assertEqual(loaded.score_type, "model_score")
        self.assertEqual(loaded.spec.model_version, "logistic_regression:B")


if __name__ == "__main__":
    unittest.main()

