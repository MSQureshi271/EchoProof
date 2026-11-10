"""R3-S1: the Protocol B robustness corruption suite (the "D panel" transformations).

Four conditions, frozen in `SPEC` and recorded in the decision log before any use on locked test data. Specification version 2 (10 November 2026)
replaces version 1's `mp3_64` by `mp3_32` after a measurement showed 64 kbps changes 16 kHz speech far less than the noise conditions do; `mp3_128`,
`noise_20` and `noise_10` are unchanged and give byte-identical output:

    mp3_32, mp3_128   MPEG-1/2 Layer III through ffmpeg/libmp3lame, constant bitrate, decoded back to 16 kHz mono PCM
    noise_20, noise_10   additive noise at a target SNR measured on global power, white or pink (one colour per parent recording)

Design rules:

* **Parent IDs are kept.** A variant is identified by `parent_id` + `condition`; `variant_id` is `<parent_id>@<condition>`.
* **Deterministic.** Seeds come from SHA-256 of (master seed, parent ID, purpose). No global RNG state, no wall-clock input.
* **Exactly what would be stored.** Every output is quantised to 16-bit PCM and returned as the same float32 values a decoder would read
  back, so scoring a variant in memory equals scoring it after storage. `pcm_sha256` hashes that PCM.
* **Level.** If noise or the codec pushes the peak above `PEAK_LIMIT`, the whole recording is scaled down by one gain. A single gain keeps the
  SNR unchanged. The gain is recorded in `params`.
* **ffmpeg safety.** Fixed argument arrays, no shell, audio passed over pipes, a timeout, and a round-trip check of every MP3 result.
* **No data access here.** This module transforms arrays. Choosing which recordings to transform (never test data before R5-S1) is the caller's
  job; `scripts/r3_s1_corruption_sample.py` applies it to validation recordings only.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np

from echoproof.artifacts import digest

SAMPLE_RATE = 16000
SPEC_VERSION = 2
MASTER_SEED = 42
PEAK_LIMIT = 0.999
MIN_SAMPLES = 4096                      # shorter inputs are rejected: below ~0.25 s the codec delay dominates
MIN_POWER = 1e-12                       # a recording with less mean power than this has no meaningful SNR
SNR_TOLERANCE_DB = 0.5                  # measured vs target SNR, checked only when the noise is above the 16-bit quantisation floor
MP3_DELAY_SAMPLES = 1105                # LAME encoder delay (576) + decoder delay (529); dropped from the decoded signal. ffmpeg only trims it
                                        # itself when the Xing/Info tag is written, which a pipe cannot do, so the alignment is explicit here
MP3_MAX_LENGTH_DIFF = 1152              # after alignment, the decoded length may differ from the input by at most one MPEG frame
MP3_MIN_CORRELATION = 0.9               # lag-0 correlation of the decoded MP3 with the input; a misaligned decode fails this
FFMPEG_TIMEOUT_S = 60
NOISE_COLORS = ("white", "pink")

CONDITIONS = {
    "mp3_32": {"kind": "mp3", "bitrate_kbps": 32},
    "mp3_128": {"kind": "mp3", "bitrate_kbps": 128},
    "noise_20": {"kind": "noise", "snr_db": 20.0},
    "noise_10": {"kind": "noise", "snr_db": 10.0},
}
CONDITION_NAMES = tuple(CONDITIONS)

SPEC = {
    "version": SPEC_VERSION,
    "sample_rate": SAMPLE_RATE,
    "channels": 1,
    "conditions": CONDITIONS,
    "mp3": {
        "encode_args": ["-nostdin", "-hide_banner", "-loglevel", "error", "-fflags", "+bitexact", "-flags", "+bitexact",
                        "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                        "-map_metadata", "-1", "-c:a", "libmp3lame", "-b:a", "<bitrate>k", "-ar", "16000", "-ac", "1", "-f", "mp3", "pipe:1"],
        "decode_args": ["-nostdin", "-hide_banner", "-loglevel", "error", "-f", "mp3", "-i", "pipe:0",
                        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
        "alignment": "drop the first delay_samples of the decoded signal, keep the input length, zero-pad a short tail",
        "delay_samples": MP3_DELAY_SAMPLES, "max_length_diff_samples": MP3_MAX_LENGTH_DIFF, "min_correlation": MP3_MIN_CORRELATION,
    },
    "noise": {
        "snr_definition": "10*log10(mean(signal^2) / mean(noise^2)) over the whole recording (global power)",
        "colors": list(NOISE_COLORS),
        "color_rule": "per parent recording: sha256(seed|parent_id|noise-color) first byte even = white, odd = pink; both noise conditions share it",
        "pink": "white Gaussian shaped by 1/sqrt(f) in the rFFT domain, DC removed, normalised to unit power",
        "white": "standard Gaussian, DC removed, normalised to unit power",
        "seed_rule": "sha256(seed|parent_id|condition) first 8 bytes, big-endian, numpy default_rng",
        "snr_tolerance_db": SNR_TOLERANCE_DB,
    },
    "level": {"peak_limit": PEAK_LIMIT, "rule": "if peak > limit, one gain limit/peak is applied to the whole output"},
    "quantisation": "round(x*32768) clipped to int16; returned wave is pcm/32768 as float32",
    "min_samples": MIN_SAMPLES,
    "min_power": MIN_POWER,
}
SPEC_SHA256 = digest(SPEC)


class CorruptionError(ValueError):
    """A corruption could not be applied or failed its own check."""


@dataclass(frozen=True)
class Corrupted:
    parent_id: str
    condition: str
    variant_id: str
    wave: np.ndarray = field(repr=False, compare=False)
    pcm_sha256: str
    n_samples: int
    params: dict
    checks: dict


def variant_id(parent_id: str, condition: str) -> str:
    if condition not in CONDITIONS:
        raise CorruptionError(f"unknown condition {condition!r}")
    return f"{parent_id}@{condition}"


def _hash_bytes(*parts) -> bytes:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()


def noise_color(parent_id: str, seed: int = MASTER_SEED) -> str:
    return NOISE_COLORS[_hash_bytes(seed, parent_id, "noise-color")[0] % 2]


def _noise_rng(parent_id: str, condition: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(_hash_bytes(seed, parent_id, condition)[:8], "big"))


def _prepare(wave) -> np.ndarray:
    x = np.asarray(wave)
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.floating):
        raise CorruptionError("expected a one-dimensional floating-point waveform")
    if x.size < MIN_SAMPLES:
        raise CorruptionError(f"waveform has {x.size} samples; at least {MIN_SAMPLES} are required")
    if not np.isfinite(x).all():
        raise CorruptionError("waveform contains non-finite samples")
    if np.abs(x).max() > 1.0:
        raise CorruptionError("waveform exceeds full scale")
    return x.astype(np.float64)


def _power(x: np.ndarray) -> float:
    return float(np.mean(x * x))


def measured_snr_db(clean, corrupted) -> float:
    """10*log10(P(clean) / P(corrupted - clean)); the corrupted signal must be aligned with and as long as the clean one."""
    c, k = np.asarray(clean, dtype=np.float64), np.asarray(corrupted, dtype=np.float64)
    if c.shape != k.shape:
        raise CorruptionError("signals differ in length")
    noise = _power(k - c)
    return float("inf") if noise == 0 else float(10.0 * np.log10(_power(c) / noise))


def _finish(x: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Peak guard, then 16-bit quantisation. Returns the float32 wave a decoder would read, the PCM hash and the applied gain."""
    gain = 1.0
    peak = float(np.abs(x).max())
    if peak > PEAK_LIMIT:
        gain = PEAK_LIMIT / peak
        x = x * gain
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype("<i2")
    return pcm.astype(np.float32) / 32768.0, hashlib.sha256(pcm.tobytes()).hexdigest(), gain


# ---- noise ----------------------------------------------------------------------------------------------------------------------

def make_noise(n: int, color: str, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-power noise of the requested colour."""
    if color not in NOISE_COLORS:
        raise CorruptionError(f"unknown noise colour {color!r}")
    white = rng.standard_normal(n)
    if color == "pink":
        spectrum = np.fft.rfft(white)
        scale = np.zeros(spectrum.size)
        scale[1:] = 1.0 / np.sqrt(np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)[1:])
        shaped = np.fft.irfft(spectrum * scale, n)
    else:
        shaped = white
    shaped = shaped - shaped.mean()
    power = _power(shaped)
    if not power > 0:
        raise CorruptionError("generated noise has no power")
    return shaped / np.sqrt(power)


def add_noise(x: np.ndarray, snr_db: float, color: str, rng: np.random.Generator) -> np.ndarray:
    power = _power(x)
    if power < MIN_POWER:
        raise CorruptionError("recording is silent; an SNR is not defined")
    return x + make_noise(x.size, color, rng) * np.sqrt(power / 10.0 ** (snr_db / 10.0))


# ---- mp3 ------------------------------------------------------------------------------------------------------------------------

def find_ffmpeg(explicit: str | None = None) -> str:
    """Absolute path of an ffmpeg that has libmp3lame. `ECHOPROOF_FFMPEG` or `explicit` override the PATH lookup."""
    candidate = explicit or os.environ.get("ECHOPROOF_FFMPEG") or shutil.which("ffmpeg")
    if not candidate:
        raise CorruptionError("ffmpeg was not found; install it or set ECHOPROOF_FFMPEG")
    path = os.path.abspath(candidate)
    if not os.path.isfile(path):
        raise CorruptionError(f"ffmpeg not found at {path}")
    result = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    if result.returncode != 0 or b"libmp3lame" not in result.stdout:
        raise CorruptionError(f"{path} does not provide libmp3lame")
    return path


def ffmpeg_info(ffmpeg: str | None = None) -> dict:
    path = find_ffmpeg(ffmpeg)
    out = subprocess.run([path, "-version"], capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False).stdout.decode("utf-8", "replace")
    first = out.splitlines()[0] if out else ""
    lame = next((line.strip() for line in out.splitlines() if "libmp3lame" in line), "libmp3lame (version line not printed)")
    return {"path": path, "version": first, "mp3_encoder": "libmp3lame", "build_note": lame}


def _run(path: str, args: list[str], data: bytes) -> bytes:
    try:
        result = subprocess.run([path, *args], input=data, capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise CorruptionError("ffmpeg timed out") from exc
    if result.returncode != 0 or not result.stdout:
        raise CorruptionError(f"ffmpeg failed (exit {result.returncode}): {result.stderr.decode('utf-8', 'replace')[:300]}")
    return result.stdout


def mp3_roundtrip(x: np.ndarray, bitrate_kbps: int, ffmpeg: str | None = None) -> tuple[np.ndarray, dict]:
    """Encode `x` (float, 16 kHz mono) to CBR MP3 and decode it back to the same length. Returns the wave and the round-trip checks."""
    path = find_ffmpeg(ffmpeg)
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype("<i2").tobytes()
    encode = [a.replace("<bitrate>", str(bitrate_kbps)) for a in SPEC["mp3"]["encode_args"]]
    mp3 = _run(path, encode, pcm)
    decoded = np.frombuffer(_run(path, SPEC["mp3"]["decode_args"], mp3), dtype="<i2").astype(np.float64)[MP3_DELAY_SAMPLES:] / 32768.0
    diff = int(decoded.size - x.size)
    if abs(diff) > MP3_MAX_LENGTH_DIFF:
        raise CorruptionError(f"MP3 round trip changed the aligned length by {diff} samples (limit {MP3_MAX_LENGTH_DIFF})")
    aligned = decoded[:x.size] if diff >= 0 else np.concatenate([decoded, np.zeros(-diff)])
    denom = float(np.sqrt(_power(x) * _power(aligned)))
    correlation = float(np.mean(x * aligned) / denom) if denom > 0 else 0.0
    if correlation < MP3_MIN_CORRELATION:
        raise CorruptionError(f"MP3 round trip is misaligned or distorted (correlation {correlation:.3f}, minimum {MP3_MIN_CORRELATION})")
    return aligned, {"length_diff_samples": diff, "correlation": round(correlation, 6), "mp3_bytes": len(mp3),
                     "effective_kbps": round(len(mp3) * 8 / (x.size / SAMPLE_RATE) / 1000.0, 2)}


# ---- public API -----------------------------------------------------------------------------------------------------------------

def corrupt(wave, parent_id: str, condition: str, *, seed: int = MASTER_SEED, ffmpeg: str | None = None) -> Corrupted:
    """Apply one condition to one recording. The same inputs always give the same output on the same ffmpeg build."""
    if condition not in CONDITIONS:
        raise CorruptionError(f"unknown condition {condition!r}")
    if not isinstance(parent_id, str) or ":" not in parent_id:
        raise CorruptionError("parent_id must be a namespaced recording identity")
    x = _prepare(wave)
    spec = CONDITIONS[condition]
    params: dict = {"kind": spec["kind"]}
    checks: dict = {}
    if spec["kind"] == "mp3":
        y, checks = mp3_roundtrip(x, spec["bitrate_kbps"], ffmpeg)
        params["bitrate_kbps"] = spec["bitrate_kbps"]
    else:
        color = noise_color(parent_id, seed)
        y = add_noise(x, spec["snr_db"], color, _noise_rng(parent_id, condition, seed))
        params.update({"snr_db": spec["snr_db"], "color": color})
    out, sha, gain = _finish(y)
    params["gain"] = round(gain, 9)
    params["input_peak"] = round(float(np.abs(x).max()), 6)
    snr = measured_snr_db(x, out.astype(np.float64) / gain) if gain != 1.0 else measured_snr_db(x, out)
    checks["measured_snr_db"] = round(snr, 3)
    if spec["kind"] == "noise":
        floor_ok = float(np.sqrt(_power(x) / 10.0 ** (spec["snr_db"] / 10.0))) > 8.0 / 32768.0      # noise rms above ~8 LSB
        checks["snr_checked"] = floor_ok
        if floor_ok and abs(snr - spec["snr_db"]) > SNR_TOLERANCE_DB:
            raise CorruptionError(f"measured SNR {snr:.2f} dB differs from the target {spec['snr_db']} dB by more than {SNR_TOLERANCE_DB} dB")
    return Corrupted(parent_id, condition, variant_id(parent_id, condition), out, sha, int(out.size), params, checks)


def corrupt_all(wave, parent_id: str, conditions=CONDITION_NAMES, *, seed: int = MASTER_SEED, ffmpeg: str | None = None) -> list[Corrupted]:
    return [corrupt(wave, parent_id, c, seed=seed, ffmpeg=ffmpeg) for c in conditions]


def spec_record(ffmpeg: str | None = None, *, with_ffmpeg: bool = True) -> dict:
    """The frozen specification, its hash and (optionally) the ffmpeg build used, for the decision log and run reports."""
    record = {"spec": SPEC, "spec_sha256": SPEC_SHA256, "master_seed": MASTER_SEED}
    if with_ffmpeg:
        record["ffmpeg"] = ffmpeg_info(ffmpeg)
    return record
