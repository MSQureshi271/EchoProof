"""R1-N2 presentation rules for the application, kept free of Streamlit so they can be tested without it."""
from __future__ import annotations

import json

from echoproof.contracts import InferenceResult
from echoproof.inference import FAILURE_MESSAGES

NOT_PROOF = ("A model score is evidence from one model on this recording, not proof of how it was made. "
             "It is not a calibrated probability unless the label says so.")


def present(result: InferenceResult) -> dict:
    """Headline, severity ('success' | 'warning' | 'error') and detail lines for one result; never invents a score for a failure."""
    if result.status != "success":
        return {"level": "error", "headline": "Analysis failed",
                "lines": [FAILURE_MESSAGES.get(result.failure_reason, f"Analysis failed ({result.failure_reason}).")]}
    verdict = {"synthetic": "Synthetic speech indicators found", "genuine": "No synthetic speech indicators found"}[result.assessment]
    if result.score_type == "mock_score":     # a placeholder must never read like a detection
        verdict = f"MOCK result (placeholder, not a detection): would be \"{verdict.lower()}\""
    kind = {"calibrated_probability": "Calibrated probability of synthetic speech", "model_score": "Model score",
            "mock_score": "MOCK score (placeholder)"}.get(result.score_type, result.score_type)
    lines = [f"{kind}: {result.score:.3f}", f"Model: {result.model_version}",
             f"Audio analysed: {result.covered_duration_seconds:.1f} s of {result.decoded_duration_seconds:.1f} s", NOT_PROOF]
    return {"level": "warning" if result.score_type == "mock_score" else "success", "headline": verdict, "lines": lines}


def result_json(result: InferenceResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False)


def diagnostics(analysis, bundle) -> dict:
    """Score details for the details block and the downloadable JSON. The InferenceResult contract is unchanged; these are extra fields."""
    rule = bundle.spec.decision_rule
    return {"raw_recording_score": analysis.raw_score, "reported_score": analysis.result.score, "score_type": analysis.result.score_type,
            "calibrated": analysis.result.score_type == "calibrated_probability", "decision_rule": {"kind": rule.kind, "threshold": rule.threshold},
            "bundle_id": bundle.spec.bundle_id, "model_version": bundle.spec.model_version, "chunks_scored": len(analysis.chunk_scores)}


def result_json_with_diagnostics(analysis, bundle) -> str:
    return json.dumps({**analysis.result.to_dict(), "diagnostics": diagnostics(analysis, bundle)}, indent=2, sort_keys=True, allow_nan=False)


# ---------------------------------------------------------------------------------------------------------------------------
# R4-N2 additions (additive): production bundle verification, scope text, busy gate, accessibility text. No Streamlit import here.
# ---------------------------------------------------------------------------------------------------------------------------
import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

PRODUCTION_BUNDLE_RELATIVE = "bundles/r3_n1_wavlm_head_B_calibrated"
PRODUCTION_LOCK_RELATIVE = "locks/B_lock_gr3.json"
PRODUCTION_CANDIDATE = "WavLM"
PINNED_TRANSFORMERS = "5.22.0"
VALIDATED_SCOPE = "English speech; genuine human recordings versus text-to-speech (TTS) output."

# distinct start-up failure codes (never used for a user's audio; those use inference.FAILURE_MESSAGES)
BUNDLE_MISSING = "bundle_missing"
BUNDLE_CHANGED = "bundle_changed"
BUNDLE_LOCK_UNAVAILABLE = "lock_unavailable"
BUNDLE_LOAD_FAILED = "bundle_load_failed"
BUNDLE_IS_MOCK = "bundle_is_mock"

SCOPE_STATEMENT = (
    "Scope: " + VALIDATED_SCOPE + " It was not validated for other languages, music, noisy or heavily compressed audio, "
    "voice conversion, or any other kind of manipulation. "
    "Privacy: the audio you choose is processed in memory on the machine running this app and is not saved or kept; no temporary "
    "audio files are written. If this app is hosted on a server rather than run locally, your audio is sent to that server to be analysed.")

FALLBACK_POLICY_COST = ("WavLM at its locked threshold flags 49 of 260 genuine operating recordings (18.8%); "
                        "the policy assumes 75% TTS prevalence.")

BUSY_MESSAGE = "Another recording is being analysed right now. Wait a moment and select Analyze again."
UNCALIBRATED_WARNING = ("This score is NOT a calibrated probability. It is a raw model score and must not be read as a chance that the "
                        "recording is synthetic.")


def score_type_warning(score_type):
    """Explicit warning text when the reported score is anything other than a calibrated probability."""
    if score_type == "calibrated_probability":
        return None
    return UNCALIBRATED_WARNING


def outcome_text(result: InferenceResult) -> str:
    """The outcome as plain text (never colour only), for a labelled line next to the coloured banner."""
    if result.status != "success":
        return "Outcome: analysis failed, no score was produced."
    word = {"synthetic": "synthetic speech indicators found", "genuine": "no synthetic speech indicators found"}[result.assessment]
    return f"Outcome: {word} (score {result.score:.3f})."


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class BundleStatus:
    """Outcome of start-up preparation. `ok` False means the app must refuse to analyse anything (fail closed)."""
    ok: bool
    code: str | None = None
    message: str = ""
    bundle: object = None
    warnings: tuple = ()
    verified_against_lock: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verify_bundle_against_lock(bundle_path, lock_path, candidate: str = PRODUCTION_CANDIDATE) -> tuple:
    """Return (code, message); (None, "") when bundle.json and every component listed in the lock match the recorded SHA-256."""
    bundle_path, lock_path = Path(bundle_path), Path(lock_path)
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        entry = lock["candidates"][candidate]
        expected_manifest, components = entry["bundle_json_sha256"], entry["components"]
    except Exception as exc:
        return BUNDLE_LOCK_UNAVAILABLE, f"The lock file {lock_path} could not be read ({type(exc).__name__}); the model cannot be verified."
    if not bundle_path.is_dir() or not (bundle_path / "bundle.json").is_file():
        return BUNDLE_MISSING, f"The production model bundle was not found at {bundle_path}. Fetch the bundle and restart the app."
    if _sha256_file(bundle_path / "bundle.json") != expected_manifest:
        return BUNDLE_CHANGED, "The model bundle does not match the locked version (bundle.json differs). The app will not analyse audio."
    for name, meta in components.items():
        part = bundle_path / name
        if not part.is_file():
            return BUNDLE_CHANGED, f"The model bundle is incomplete: component {name} is missing. The app will not analyse audio."
        if part.stat().st_size != meta["bytes"] or _sha256_file(part) != meta["sha256"]:
            return BUNDLE_CHANGED, f"The model bundle does not match the locked version (component {name} differs). The app will not analyse audio."
    return None, ""


def transformers_version_warning(expected: str = PINNED_TRANSFORMERS, installed=None):
    """A warning (not a failure) when the transformers version differs from the one the bundles were verified with."""
    if installed is None:
        try:
            from importlib.metadata import version
            installed = version("transformers")
        except Exception:
            return f"transformers is not installed; the pinned version is {expected}."
    if installed != expected:
        return f"transformers {installed} is installed but the model was verified with {expected}; scores may differ."
    return None


def prepare_bundle(bundle_path=None, lock_path=None, *, root=None, loader=None, installed_transformers=None) -> BundleStatus:
    """Start-up: resolve the bundle (default: the locked production bundle), verify it against the lock, load it, refuse a mock.

    A bundle at a path other than the locked production one (ECHOPROOF_BUNDLE override) is loaded without lock verification and
    carries an explicit warning. Any failure returns ok=False with a message that is distinct from invalid-audio messages."""
    root = Path(root) if root else repo_root()
    lock_path = Path(lock_path) if lock_path else root / PRODUCTION_LOCK_RELATIVE
    production = root / PRODUCTION_BUNDLE_RELATIVE
    target = Path(bundle_path) if bundle_path else production
    if not target.is_absolute():
        target = root / target
    is_production = target.resolve() == production.resolve()
    warnings = []
    if is_production:
        code, message = verify_bundle_against_lock(target, lock_path)
        if code:
            return BundleStatus(False, code, message)
    else:
        warnings.append(f"Developer override: the bundle at {target} is not the locked production bundle and was not verified against the lock.")
        if not target.exists():
            return BundleStatus(False, BUNDLE_MISSING, f"The model bundle was not found at {target}.")
    version_warning = transformers_version_warning(installed=installed_transformers)
    if version_warning:
        warnings.append(version_warning)
    try:
        if loader is None:
            from echoproof.inference import load_inference_bundle as loader
        bundle = loader(target)
    except Exception as exc:
        return BundleStatus(False, BUNDLE_LOAD_FAILED, f"The model bundle could not be loaded ({type(exc).__name__}: {exc}).")
    if getattr(bundle, "is_mock", False):
        return BundleStatus(False, BUNDLE_IS_MOCK, "A placeholder (mock) model is not allowed in this app.")
    return BundleStatus(True, None, "", bundle, tuple(warnings), is_production)


def policy_cost_note(lock_path=None) -> str:
    """The lock's own wording of the known cost of the locked policy, with a fixed fallback."""
    try:
        lock = json.loads(Path(lock_path or repo_root() / PRODUCTION_LOCK_RELATIVE).read_text(encoding="utf-8"))
        for item in lock.get("limitations", []):
            if "49 of 260" in str(item):
                return str(item)
    except Exception:
        pass
    return FALLBACK_POLICY_COST


class InferenceGate:
    """At most one active inference per process. `run` never blocks: a second caller gets (False, None) at once."""

    def __init__(self):
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def run(self, fn, *args, **kwargs):
        if not self._lock.acquire(blocking=False):
            return False, None
        try:
            return True, fn(*args, **kwargs)
        finally:
            self._lock.release()


INFERENCE_GATE = InferenceGate()


def waveform_alt_text(analysis) -> str:
    """Text alternative for the waveform figure."""
    wave = analysis.waveform
    seconds = len(wave.samples) / wave.sample_rate
    peak = float(max(abs(wave.samples.max()), abs(wave.samples.min()))) if len(wave.samples) else 0.0
    return (f"Waveform of the {seconds:.1f} second recording: amplitude against time, peak amplitude {peak:.2f}. "
            f"{len(analysis.windows)} analysis window(s) of 4 seconds are used. The same windows are listed in the chunk table below.")


def logmel_alt_text(analysis) -> str:
    """Text alternative for the log-mel spectrogram figure."""
    wave = analysis.waveform
    seconds = len(wave.samples) / wave.sample_rate
    return (f"Log-mel spectrogram of the {seconds:.1f} second recording: frequency in hertz against time in seconds, "
            "brighter colours mean more energy. It summarises the signal and is not itself a result.")
