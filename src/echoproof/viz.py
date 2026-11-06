"""R1-N2 figures for the application: a waveform and a log-mel image with labelled axes. matplotlib is imported lazily
(it is in the `analysis` extra), so importing this module never fails; drawing without it raises a clear error."""
from __future__ import annotations

import numpy as np

from echoproof.features.acoustic import HOP, N_MELS, log_mel

MAX_PLOT_COLUMNS = 2000


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to draw figures: uv pip install matplotlib") from exc
    return plt


def waveform_envelope(samples, sample_rate: int, columns: int = MAX_PLOT_COLUMNS):
    """Min/max per column so a 60 s clip plots in a few thousand points without hiding peaks. Returns (times, lo, hi)."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or sample_rate <= 0:
        raise ValueError("expected a non-empty 1-D waveform")
    columns = int(max(1, min(columns, x.size)))
    edges = np.linspace(0, x.size, columns + 1).astype(int)
    lo = np.array([x[a:b].min() for a, b in zip(edges[:-1], edges[1:])])
    hi = np.array([x[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
    times = (edges[:-1] + edges[1:]) / 2.0 / sample_rate
    return times, lo, hi


def mel_axes(n_frames: int, sample_rate: int):
    """Time (s) of each frame start and the centre frequency (Hz) of each of the 64 mel bands, matching `acoustic.log_mel`."""
    import librosa
    times = np.arange(n_frames) * HOP / sample_rate
    freqs = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0.0, fmax=sample_rate / 2.0)[1:-1]
    return times, freqs


def waveform_figure(samples, sample_rate: int, *, windows=None):
    plt = _pyplot()
    times, lo, hi = waveform_envelope(samples, sample_rate)
    fig, ax = plt.subplots(figsize=(9, 2.6), constrained_layout=True)
    ax.fill_between(times, lo, hi, color="#2b6cb0", linewidth=0)
    for start, end in (windows or []):
        ax.axvline(start / sample_rate, color="#c05621", alpha=0.25, linewidth=0.8)
    ax.set_xlim(0, len(samples) / sample_rate)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform (vertical lines mark chunk starts)" if windows else "Waveform")
    return fig


def logmel_figure(samples, sample_rate: int):
    plt = _pyplot()
    mel = log_mel(samples, sample_rate)                       # [time, 64], natural-log power
    times, freqs = mel_axes(mel.shape[0], sample_rate)
    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    mesh = ax.pcolormesh(times, freqs, mel.T, shading="nearest", cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Log-mel spectrogram (64 mel bands)")
    fig.colorbar(mesh, ax=ax, label="log power")
    return fig


def close(fig) -> None:
    _pyplot().close(fig)
