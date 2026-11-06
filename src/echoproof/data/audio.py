"""Bounded decoding and canonical waveform preparation for software probes."""
from dataclasses import dataclass
import io
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioError(ValueError):
    """Invalid, unreadable or oversized audio."""


@dataclass(frozen=True)
class Waveform:
    samples: np.ndarray
    sample_rate: int
    original_sample_rate: int
    original_channels: int
    original_frames: int
    quality_flags: tuple[str, ...]


MAX_AUDIO_BYTES = 25 * 1024 * 1024   # REVISED_PLAN R1-N2 input limits: 25 MiB and 60 s
MAX_AUDIO_SECONDS = 60.0


def decode_audio(data: bytes, *, target_rate: int = 16000, max_bytes: int = MAX_AUDIO_BYTES,
                 max_seconds: float = MAX_AUDIO_SECONDS, max_channels: int = 2) -> Waveform:
    if not isinstance(data, bytes) or not data or len(data) > max_bytes:
        raise AudioError("audio bytes are empty or exceed the input limit")
    if type(target_rate) is not int or not 4000 <= target_rate <= 384000:
        raise AudioError("target_rate must be an integer between 4000 and 384000")
    try:
        with sf.SoundFile(io.BytesIO(data)) as audio:
            rate, channels, frames = audio.samplerate, audio.channels, audio.frames
            if not 4000 <= rate <= 384000 or channels < 1 or channels > max_channels or frames <= 0 or frames / rate > max_seconds:
                raise AudioError("audio duration, channels, or sample rate is outside bounds")
            samples = audio.read(dtype="float32", always_2d=True)
    except AudioError:
        raise
    except (RuntimeError, ValueError, OSError) as exc:
        raise AudioError("audio cannot be decoded") from exc
    if samples.shape != (frames, channels) or not np.isfinite(samples).all():
        raise AudioError("audio is truncated or contains nonfinite samples")
    mono = samples.mean(axis=1, dtype=np.float32)
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, rate // divisor).astype(np.float32)
    if not len(mono) or not np.isfinite(mono).all():
        raise AudioError("invalid samples after resampling")
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    return Waveform(mono, target_rate, rate, channels, frames, ("silence",) if rms < 1e-4 else ())


def windows(waveform: Waveform, size: int):
    if type(size) is not int or size <= 0:
        raise AudioError("window size must be a positive integer")
    for start in range(0, len(waveform.samples), size):
        valid = waveform.samples[start:start + size]
        padded = np.zeros(size, dtype=np.float32)
        padded[:len(valid)] = valid
        yield padded, len(valid), start
