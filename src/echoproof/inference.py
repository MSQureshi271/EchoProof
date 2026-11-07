"""R1-N2 inference path: bytes -> bounded decode -> quality checks -> canonical chunks -> scorer -> mean -> calibration -> decision.

Design decisions (each covered by tests/test_r1_n2_inference.py):
  * One code path. `predict_recording` and the app both call `analyze`; the evaluator (R1-N1) and CLI (R2-N1) must call the
    same function, so a parity test can compare their scores byte for byte.
  * The model is a plain callable `scorer(list_of_chunk_waveforms) -> P(TTS) per chunk`. `LRDetector.score_chunks` already has
    this shape, so the real LR bundle drops in without changes. Aggregation and calibration are also plain callables and
    default to the mean and the identity; R1-N1's functions replace them without changing this module.
  * Chunks come from `data.chunks.eval_windows`, the same windows the evaluator scores. Short clips are never padded or repeated.
  * A mock scorer exists only so the application skeleton can run before a real bundle does. Every output built from it
    says so in `model_version`, `score_type` and `warnings`. It must be removed in R2-N1.
  * Failures return an `InferenceResult` with a reason code and no score; nothing raises for bad user input.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import io
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np
import soundfile as sf

from echoproof.contracts import BundleSpecification, DecisionRule, InferenceResult
from echoproof.data.audio import AudioError, MAX_AUDIO_BYTES, MAX_AUDIO_SECONDS, Waveform, decode_audio
from echoproof.data.chunks import PREPROCESSING_HASH, WINDOW_SECONDS, eval_windows

SCHEMA_VERSION = 1
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 1.0

# failure reason codes shown to users (stable, machine-readable)
EMPTY_FILE = "empty_file"
FILE_TOO_LARGE = "file_too_large"
AUDIO_TOO_LONG = "audio_too_long"
UNDECODABLE = "undecodable_audio"
INSUFFICIENT_AUDIO = "insufficient_audio"
SCORING_FAILED = "scoring_failed"

MOCK_MODEL_VERSION = "MOCK-not-a-trained-model"
MOCK_WARNING = "MOCK MODEL: this score is a deterministic placeholder from the audio hash, not a detection result."


def mean_probability(chunk_scores: Sequence[float]) -> float:
    """Recording score = mean P(TTS) over its chunks (the aggregation rule in the plan)."""
    return float(np.mean(np.asarray(chunk_scores, dtype=np.float64)))


def identity(score: float) -> float:
    return float(score)


def mock_scorer(waveforms) -> np.ndarray:
    """Deterministic placeholder: a pseudo-probability from the SHA-256 of each chunk's samples. Not a model."""
    out = []
    for w in waveforms:
        digest = hashlib.sha256(np.asarray(w, dtype=np.float32).tobytes()).digest()
        out.append(int.from_bytes(digest[:8], "big") / float(2 ** 64))
    return np.array(out, dtype=np.float64)


@dataclass(frozen=True)
class LoadedBundle:
    """What inference needs from a bundle. R2-N2's loader will construct this from `bundle.json`; until then tests and the mock do."""
    spec: BundleSpecification
    scorer: Callable[[Sequence[np.ndarray]], Sequence[float]]
    aggregate: Callable[[Sequence[float]], float] = mean_probability
    calibrate: Callable[[float], float] | None = None
    score_type: str = "model_score"          # "calibrated_probability" only when a calibrator was fitted and helped
    is_mock: bool = False


def mock_bundle(threshold: float = 0.5) -> LoadedBundle:
    spec = BundleSpecification(SCHEMA_VERSION, "bundle:mock-skeleton", MOCK_MODEL_VERSION, PREPROCESSING_HASH,
                               DecisionRule(SCHEMA_VERSION, "threshold", threshold), "genuine", "tts")
    return LoadedBundle(spec, mock_scorer, score_type="mock_score", is_mock=True)


def load_lr_bundle(bundle_path: str | Path, threshold: float = 0.5) -> LoadedBundle:
    """Load a provisionally fitted Logistic Regression model directory as a LoadedBundle.

    Accepts either the model directory containing weights.npz and lr_model.json or
    a run directory containing a model/ subdirectory.
    """
    from echoproof.models.lr_detector import LRDetector

    path = Path(bundle_path)
    if (path / "weights.npz").is_file() and (path / "lr_model.json").is_file():
        model_dir = path
    elif (path / "model" / "weights.npz").is_file() and (path / "model" / "lr_model.json").is_file():
        model_dir = path / "model"
    else:
        raise FileNotFoundError(
            f"No valid LR model found at {bundle_path} (expected weights.npz and lr_model.json)"
        )

    model = LRDetector.load(model_dir)

    protocol = model.meta.get("extra", {}).get("protocol", "B")
    bundle_id = f"bundle:r2-lr-protocol-{str(protocol).lower()}"
    model_version = f"logistic_regression:{protocol}"

    rule = DecisionRule(SCHEMA_VERSION, "threshold", float(threshold))
    spec = BundleSpecification(
        schema_version=SCHEMA_VERSION,
        bundle_id=bundle_id,
        model_version=model_version,
        preprocessing_hash=PREPROCESSING_HASH,
        decision_rule=rule,
        class_0="genuine",
        class_1="tts",
    )
    return LoadedBundle(
        spec=spec,
        scorer=model.score_chunks,
        aggregate=mean_probability,
        calibrate=None,
        score_type="model_score",
        is_mock=False,
    )


@dataclass
class Analysis:
    """The result plus the intermediate arrays the application draws. Only `result` is a contract object."""
    result: InferenceResult
    waveform: Waveform | None = None
    windows: list = field(default_factory=list)
    chunk_scores: list = field(default_factory=list)
    raw_score: float | None = None


def _fail(reason: str, timings, decoded=None) -> Analysis:
    return Analysis(InferenceResult(SCHEMA_VERSION, "failed", None, None, None, [], None, decoded, None, dict(timings), reason))


def wav_looks_truncated(data: bytes) -> bool:
    """True when a RIFF/WAVE file declares more sample data than it contains. libsndfile silently reads what is there, so the
    decoder alone would accept a cut-off file as a shorter recording."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    pos = 12
    while pos + 8 <= len(data):
        chunk_id, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if chunk_id == b"data":
            return size not in (0, 0xFFFFFFFF) and pos + 8 + size > len(data)
        pos += 8 + size + (size & 1)
    return True               # no data chunk reached: the header itself is cut off


def analyze(audio_bytes, bundle: LoadedBundle, *, max_bytes: int = MAX_AUDIO_BYTES, max_seconds: float = MAX_AUDIO_SECONDS) -> Analysis:
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    def done(analysis: Analysis) -> Analysis:
        timings["total"] = time.perf_counter() - t_start
        analysis.result = replace(analysis.result, timings_seconds=dict(timings))
        return analysis

    # 1. bounded decode -----------------------------------------------------------------------------------------------
    if not isinstance(audio_bytes, (bytes, bytearray)) or len(audio_bytes) == 0:
        return done(_fail(EMPTY_FILE, timings))
    audio_bytes = bytes(audio_bytes)
    if len(audio_bytes) > max_bytes:
        return done(_fail(FILE_TOO_LARGE, timings))
    try:
        info = sf.info(io.BytesIO(audio_bytes))
        if info.samplerate > 0 and info.frames / info.samplerate > max_seconds:
            return done(_fail(AUDIO_TOO_LONG, timings))
    except Exception:       # not readable as audio: the decoder below reports it with one consistent reason
        pass
    t = time.perf_counter()
    try:
        wave = decode_audio(audio_bytes, max_bytes=max_bytes, max_seconds=max_seconds)
    except AudioError:
        timings["decode"] = time.perf_counter() - t
        return done(_fail(UNDECODABLE, timings))
    timings["decode"] = time.perf_counter() - t
    duration = len(wave.samples) / wave.sample_rate

    # 2. quality checks -----------------------------------------------------------------------------------------------
    if duration < MIN_AUDIO_SECONDS:
        return done(_fail(INSUFFICIENT_AUDIO, timings, decoded=duration))
    warnings = []
    if bundle.is_mock:
        warnings.append(MOCK_WARNING)
    if "silence" in wave.quality_flags:
        warnings.append("The recording is almost silent; the score is not meaningful.")
    if duration < WINDOW_SECONDS:
        warnings.append(f"The recording is shorter than {WINDOW_SECONDS} s; it was scored as a single short chunk.")
    if wav_looks_truncated(audio_bytes):
        warnings.append("The WAV file looks truncated (it contains less audio than its header declares); only the audio present was analysed.")
    if wave.original_sample_rate != wave.sample_rate:
        warnings.append(f"Resampled from {wave.original_sample_rate} Hz to {wave.sample_rate} Hz.")

    # 3. canonical chunks -----------------------------------------------------------------------------------------------
    t = time.perf_counter()
    windows = eval_windows(len(wave.samples))
    chunks = [wave.samples[a:b] for a, b in windows]
    timings["chunk"] = time.perf_counter() - t

    # 4. adapter -> aggregation -> calibration -> decision ------------------------------------------------------------------
    t = time.perf_counter()
    try:
        scores = [float(s) for s in bundle.scorer(chunks)]
        if len(scores) != len(chunks) or not all(np.isfinite(s) and 0.0 <= s <= 1.0 for s in scores):
            raise ValueError("scorer returned the wrong number of scores or values outside [0, 1]")
        raw = float(bundle.aggregate(scores))
        final = float(bundle.calibrate(raw)) if bundle.calibrate is not None else raw
        if not (np.isfinite(raw) and np.isfinite(final) and 0.0 <= raw <= 1.0 and 0.0 <= final <= 1.0):
            raise ValueError("aggregated or calibrated score outside [0, 1]")
        label = bundle.spec.decision_rule.decide(final)
    except Exception:
        timings["score"] = time.perf_counter() - t
        return done(_fail(SCORING_FAILED, timings, decoded=duration))
    timings["score"] = time.perf_counter() - t
    assessment = "synthetic" if label == 1 else "genuine"
    covered = max(b for _, b in windows) / wave.sample_rate
    result = InferenceResult(SCHEMA_VERSION, "success", assessment, final, bundle.score_type, warnings, bundle.spec.model_version,
                             duration, min(covered, duration), dict(timings), None)
    return done(Analysis(result, wave, windows, scores, raw))


def predict_recording(audio_bytes, bundle: LoadedBundle, **limits) -> InferenceResult:
    """The contract entry point (RecordingPredictor). `timings_seconds['total']` is filled for every outcome."""
    return analyze(audio_bytes, bundle, **limits).result


FAILURE_MESSAGES = {
    EMPTY_FILE: "The file is empty. Choose an audio file.",
    FILE_TOO_LARGE: f"The file is larger than {MAX_AUDIO_BYTES // (1024 * 1024)} MiB. Choose a smaller file.",
    AUDIO_TOO_LONG: f"The recording is longer than {int(MAX_AUDIO_SECONDS)} seconds. Trim it and try again.",
    UNDECODABLE: "The file could not be read as audio (corrupt, truncated, unsupported format, or more than two channels).",
    INSUFFICIENT_AUDIO: f"The recording is shorter than {MIN_AUDIO_SECONDS:g} second, which is too little to analyse.",
    SCORING_FAILED: "The model could not score this recording.",
}
