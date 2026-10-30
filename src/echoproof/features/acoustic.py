"""Fixture-stage acoustic representations; no learned normalization."""
import numpy as np
import librosa
from echoproof.artifacts import digest

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP = 160
N_MELS = 64
N_MFCC = 13
LOG_FLOOR = 1e-10
SCHEMA = {"version": 1, "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "win_length": WIN_LENGTH,
          "hop_length": HOP, "n_mels": N_MELS, "n_mfcc": N_MFCC, "center": False,
          "log_floor": LOG_FLOOR, "short_policy": "right_pad_to_fft", "voicing": "yin_rms_autocorrelation_v1"}
FEATURE_NAMES = tuple([f"{kind}_{index:02d}_{stat}" for kind in ("mfcc", "delta") for index in range(N_MFCC) for stat in ("mean", "std")]
                      + [f"{kind}_{stat}" for kind in ("centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms") for stat in ("mean", "std")]
                      + ["voiced_fraction", "voiced_f0_mean", "voiced_f0_std"])
FEATURE_SCHEMA_HASH = digest({"settings": SCHEMA, "columns": FEATURE_NAMES})


def _validate(samples, rate):
    signal = np.asarray(samples, dtype=np.float32)
    if rate != SAMPLE_RATE or signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError("expected nonempty finite mono audio at 16000 Hz")
    return np.pad(signal, (0, max(0, N_FFT - len(signal))))


def log_mel(samples, rate=SAMPLE_RATE):
    """Natural log mel power, [time, 64]; no centering, short signals pad once."""
    signal = _validate(samples, rate)
    mel = librosa.feature.melspectrogram(y=signal, sr=rate, n_fft=N_FFT, win_length=WIN_LENGTH,
                                       hop_length=HOP, n_mels=N_MELS, center=False, power=2.0)
    result = np.log(np.maximum(mel, LOG_FLOOR)).T.astype(np.float32)
    expected = 1 + (len(signal) - N_FFT) // HOP
    if result.shape != (expected, N_MELS) or not np.isfinite(result).all():
        raise ValueError("unexpected log-mel shape or nonfinite values")
    return result


def handcrafted(samples, rate=SAMPLE_RATE):
    """Ordered summary vector; unvoiced F0 statistics are NaN for imputation."""
    signal = _validate(samples, rate)
    power = librosa.feature.melspectrogram(y=signal, sr=rate, n_fft=N_FFT, win_length=WIN_LENGTH,
                                          hop_length=HOP, n_mels=N_MELS, center=False, power=2.0)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(power, ref=1.0, top_db=None), n_mfcc=N_MFCC)
    frame_count = mfcc.shape[1]
    width = min(9, frame_count if frame_count % 2 else frame_count - 1)
    delta = librosa.feature.delta(mfcc, width=width, mode="nearest") if width >= 3 else np.zeros_like(mfcc)
    output = []
    for matrix in (mfcc, delta):
        for row in matrix:
            output.extend((float(np.mean(row)), float(np.std(row))))
    magnitude = np.abs(librosa.stft(signal, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP, center=False))
    spectral = (librosa.feature.spectral_centroid(S=magnitude, sr=rate),
                librosa.feature.spectral_bandwidth(S=magnitude, sr=rate),
                librosa.feature.spectral_rolloff(S=magnitude, sr=rate),
                librosa.feature.spectral_flatness(S=magnitude ** 2),
                librosa.feature.zero_crossing_rate(signal, frame_length=N_FFT, hop_length=HOP, center=False),
                librosa.feature.rms(y=signal, frame_length=N_FFT, hop_length=HOP, center=False))
    for values in spectral:
        output.extend((float(np.mean(values)), float(np.std(values))))
    f0 = librosa.yin(signal, fmin=75, fmax=400, sr=rate, frame_length=N_FFT, hop_length=HOP, center=False)
    frames = librosa.util.frame(signal, frame_length=N_FFT, hop_length=HOP).T
    voiced = []
    for frame, pitch in zip(frames, f0):
        lag = int(round(rate / float(pitch))) if np.isfinite(pitch) and pitch > 0 else 0
        if lag < 1 or lag >= len(frame) or float(np.sqrt(np.mean(frame ** 2))) < 1e-4:
            continue
        a, b = frame[:-lag], frame[lag:]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 0 and float(np.dot(a, b) / denom) >= 0.6:
            voiced.append(float(pitch))
    output.extend((len(voiced) / frame_count, float(np.mean(voiced)) if voiced else float("nan"),
                   float(np.std(voiced)) if voiced else float("nan")))
    result = np.asarray(output, dtype=np.float64)
    if len(result) != len(FEATURE_NAMES) or np.isinf(result).any():
        raise ValueError("invalid handcrafted feature vector")
    return result
