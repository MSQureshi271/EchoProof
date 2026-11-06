"""R2-S1 building blocks: the fixed crop bank and train-fitted log-mel statistics.

Design decisions (each covered by a test in tests/test_r2_s1_features.py):
  * Mel statistics are streamed with Chan's parallel update in float64 (count, mean, sum of squared deviations per mel bin).
    Merging per-recording or per-shard partials gives the same numbers, within rounding, as one pass over every frame.
  * The statistics are frame-equal: every valid frame of every training recording counts once, so longer recordings weigh more.
    The standard deviation is the population value (ddof 0). Padding never enters, because only real frames are passed in.
  * The crop bank is a pure function of (seed, recording_id, n_samples, k). It needs no audio, never depends on iteration order
    or worker count, and each of the k crops is drawn from its own segment of the valid start range, so the crops spread over
    the recording instead of clustering. A recording not longer than one crop contributes a single crop (the whole recording).
"""
from __future__ import annotations

import numpy as np

from echoproof.features.acoustic import N_MELS
from echoproof.training.seeding import named_seed

CROP_SAMPLES = 64000          # 4 s at 16 kHz, the same crop length the trainer uses
DEFAULT_K = 5
DEFAULT_SEED = 42
BANK_VERSION = "crop-bank-v1"


def crop_bank_starts(seed: int, recording_id: str, n_samples: int, *, k: int = DEFAULT_K, crop: int = CROP_SAMPLES) -> list[int]:
    """Start samples (strictly increasing, distinct) of the fixed crops of one recording."""
    if type(n_samples) is not int or n_samples <= 0 or type(k) is not int or k < 1 or type(crop) is not int or crop < 1:
        raise ValueError("n_samples, k and crop must be positive integers")
    if n_samples <= crop:
        return [0]
    possible = n_samples - crop + 1                       # valid start positions are 0 .. n_samples - crop
    starts = []
    for j in range(k):
        low, high = (j * possible) // k, ((j + 1) * possible) // k       # segment j is [low, high)
        if high > low:
            starts.append(low + named_seed(seed, f"crop_bank:{j}:{recording_id}") % (high - low))
    return starts


class MelStats:
    """Streaming per-bin mean and population standard deviation of log-mel frames."""

    def __init__(self, n_mels: int = N_MELS):
        self.n_mels = n_mels
        self.count = 0
        self.recordings = 0
        self.mean = np.zeros(n_mels, dtype=np.float64)
        self.m2 = np.zeros(n_mels, dtype=np.float64)

    def _combine(self, count, mean, m2):
        total = self.count + count
        delta = mean - self.mean
        self.mean = self.mean + delta * (count / total)
        self.m2 = self.m2 + m2 + delta * delta * (self.count * count / total)
        self.count = total

    def add(self, frames) -> int:
        """Add one recording's valid frames [T, n_mels]; returns T."""
        x = np.asarray(frames, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.n_mels or not np.isfinite(x).all():
            raise ValueError(f"expected finite frames of shape [T>=1, {self.n_mels}]")
        mean = x.mean(axis=0)
        self._combine(x.shape[0], mean, ((x - mean) ** 2).sum(axis=0))
        self.recordings += 1
        return int(x.shape[0])

    def merge(self, other: "MelStats") -> None:
        if other.n_mels != self.n_mels:
            raise ValueError("cannot merge statistics with different mel counts")
        if other.count:
            self._combine(other.count, other.mean, other.m2)
        self.recordings += other.recordings

    def finalize(self):
        """(mean, std) as float64 arrays. Needs at least two frames."""
        if self.count < 2:
            raise ValueError("statistics need at least two frames")
        return self.mean.copy(), np.sqrt(self.m2 / self.count)

    def to_arrays(self) -> dict:
        return {"stats_count": np.int64(self.count), "stats_recordings": np.int64(self.recordings),
                "stats_mean": self.mean, "stats_m2": self.m2}

    @classmethod
    def from_arrays(cls, arrays) -> "MelStats":
        out = cls(len(arrays["stats_mean"]))
        out.count, out.recordings = int(arrays["stats_count"]), int(arrays["stats_recordings"])
        out.mean, out.m2 = np.array(arrays["stats_mean"], dtype=np.float64), np.array(arrays["stats_m2"], dtype=np.float64)
        return out


def load_mel_stats(path) -> dict:
    """Read mel_stats_B_v1.npz -> {'mean', 'std', 'frames', 'recordings'} (float32 mean/std for the models)."""
    with np.load(path, allow_pickle=False) as z:
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        if mean.shape != std.shape or not np.isfinite(mean).all() or not (std > 0).all():
            raise ValueError("invalid mel statistics file")
        return {"mean": mean, "std": std, "frames": int(z["frames"]), "recordings": int(z["recordings"])}


def normalize_log_mel(mel, mean, std):
    """(mel - mean) / std per bin for log-mel frames [T, n_mels]. Callers must apply it to valid frames only."""
    mel, mean, std = np.asarray(mel, dtype=np.float32), np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if mel.ndim != 2 or mel.shape[1] != mean.shape[0] or std.shape != mean.shape or not (std > 0).all():
        raise ValueError("mel frames and statistics do not match")
    return (mel - mean) / std
