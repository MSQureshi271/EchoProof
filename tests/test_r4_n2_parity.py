"""R4-N2 parity: library analyze, the `echoproof predict` CLI code path and the app code path give identical results for the same bytes."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from echoproof import app_support as support
from echoproof import cli
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
PRODUCTION = ROOT / support.PRODUCTION_BUNDLE_RELATIVE


def synthetic_wav(seconds=7.0, seed=11, rate=16000):
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    x = 0.25 * np.sin(2 * np.pi * (180 + 40 * np.sin(t)) * t) + 0.03 * rng.standard_normal(n)
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def small_bundle():
    """A deterministic calibrated bundle that needs no model library."""
    def scorer(chunks):
        return [float(min(0.99, 0.1 + 2.5 * float(np.mean(np.abs(c))) + 0.05 * float(np.std(c)))) for c in chunks]
    spec = BundleSpecification(1, "bundle:parity-fake", "fake:parity", PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.404), "genuine", "tts")
    return inf.LoadedBundle(spec, scorer, inf.mean_probability, lambda p: 0.5 * p + 0.2, "calibrated_probability", False)


def library_view(data, bundle):
    r = inf.analyze(data, bundle).result
    return {"score": r.score, "score_type": r.score_type, "threshold": bundle.spec.decision_rule.threshold, "decision": r.assessment}


def cli_view(data, bundle, bundle_dir, tmp, patch_loader=True):
    audio = Path(tmp) / "clip.wav"
    audio.write_bytes(data)
    out = io.StringIO()
    patch = mock.patch.object(inf, "load_inference_bundle", lambda path, threshold=None: bundle) if patch_loader else contextlib.nullcontext()
    with patch, contextlib.redirect_stdout(out):
        code = cli.main(["predict", "--bundle", str(bundle_dir), "--audio", str(audio)])
    payload = json.loads(out.getvalue())
    assert code == 0, payload
    return {"score": payload["score"], "score_type": payload["score_type"], "threshold": bundle.spec.decision_rule.threshold,
            "decision": payload["assessment"]}


def app_code_path_view(data, bundle):
    """Exactly what app/main.py does: the single-slot gate around analyze, then the downloadable JSON."""
    ran, analysis = support.INFERENCE_GATE.run(inf.analyze, data, bundle)
    assert ran
    payload = json.loads(support.result_json_with_diagnostics(analysis, bundle))
    return {"score": payload["score"], "score_type": payload["score_type"], "threshold": payload["diagnostics"]["decision_rule"]["threshold"],
            "decision": payload["assessment"]}


def app_view(data, bundle):
    import streamlit as st
    st.cache_resource.clear()
    with mock.patch.object(support, "prepare_bundle", lambda *a, **k: support.BundleStatus(True, None, "", bundle, (), True)):
        at = AppTest.from_file(APP, default_timeout=300).run()
        at.file_uploader[0].upload("clip.wav", data)
        at.run()
        next(b for b in at.button if b.label.startswith("Analyze")).click()
        at.run()
        assert not at.exception, at.exception
        shown = json.loads(at.json[0].value)
        headline = (at.success or at.warning)[0].value
    st.cache_resource.clear()
    decision = "synthetic" if headline.startswith("Synthetic") else "genuine"
    return {"score": shown["reported_score"], "score_type": shown["score_type"], "threshold": shown["decision_rule"]["threshold"], "decision": decision}


class ParityWithSmallBundle(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bundle = small_bundle()
        self.data = synthetic_wav()

    def test_library_cli_and_app_code_path_are_identical(self):
        lib = library_view(self.data, self.bundle)
        self.assertEqual(lib["score_type"], "calibrated_probability")
        self.assertEqual(cli_view(self.data, self.bundle, self.tmp, self.tmp), lib)
        self.assertEqual(app_code_path_view(self.data, self.bundle), lib)

    def test_both_decisions_agree_across_the_threshold(self):
        for amplitude_seed, seconds in ((1, 4.5), (2, 9.0), (3, 2.0)):
            data = synthetic_wav(seconds, seed=amplitude_seed)
            lib = library_view(data, self.bundle)
            self.assertEqual(cli_view(data, self.bundle, self.tmp, self.tmp), lib)
            self.assertEqual(app_code_path_view(data, self.bundle), lib)

    @unittest.skipUnless(HAVE_STREAMLIT, "streamlit is not installed")
    def test_running_app_matches_the_library(self):
        lib = library_view(self.data, self.bundle)
        self.assertEqual(app_view(self.data, self.bundle), lib)


def _real_bundle_available():
    if not (PRODUCTION / "bundle.json").is_file():
        return False
    try:
        from importlib.metadata import version
        return version("transformers") == support.PINNED_TRANSFORMERS
    except Exception:
        return False


@unittest.skipUnless(_real_bundle_available(), "production WavLM bundle or transformers==5.22.0 is not available")
class ParityWithProductionBundle(unittest.TestCase):
    def test_production_bundle_library_cli_and_app_agree(self):
        status = support.prepare_bundle()
        self.assertTrue(status.ok, status.message)
        self.assertTrue(status.verified_against_lock)
        bundle = status.bundle
        data = synthetic_wav(6.0, seed=5)
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        lib = library_view(data, bundle)
        self.assertEqual(lib["score_type"], "calibrated_probability")
        self.assertEqual(cli_view(data, bundle, PRODUCTION, tmp, patch_loader=False), lib)       # the CLI loads the bundle itself
        self.assertEqual(app_code_path_view(data, bundle), lib)
        if HAVE_STREAMLIT:
            self.assertEqual(app_view(data, bundle), lib)


if __name__ == "__main__":
    unittest.main()
