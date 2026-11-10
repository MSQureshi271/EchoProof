"""R3-S1 follow-up: how strong are candidate MP3 settings? Signal-level measurement only.

The frozen R3-S1 MP3 conditions (64 and 128 kbps at 16 kHz mono) changed validation audio very little. This module measures how much a set of
candidate MP3 settings would change the same recordings, in units that can be compared with the noise conditions. It does not decide anything
and no model scores anything: it measures audio against its own original.

Per recording and candidate it reports:

* `snr_raw_db`       10*log10(P(x) / P(y - x)) after alignment. LAME also scales the signal by a fixed factor (about 0.95), which caps this near 26 dB.
* `gain_db`          the best single scalar gain of y relative to x.
* `snr_gaincomp_db`  the same SNR after dividing y by that gain: damage other than the level change. This is the figure to compare with the
                     noise conditions (20 and 10 dB SNR).
* `lsd_db`           log-spectral distance over active frames (rms difference of the dB spectra); large when bands are removed.
* `retained_4k_pct`, `retained_7k_pct`   share of the original's energy above 4 kHz / 7 kHz that the variant keeps.

Active frames for the spectral distance are those whose mean square exceeds -55 dBFS (a fixed rule, so candidates are compared on identical frames).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from echoproof.data import corruption

SAMPLE_RATE = corruption.SAMPLE_RATE
# name -> (bitrate kbps, sample rate the MP3 is encoded at). `mp3_16k_32` and `mp3_16k_128` are the frozen R3-S1 conditions (specification version 2);
# `mp3_16k_64` was version 1's second MP3 condition and is kept here as a superseded reference.
CANDIDATES = {
    "mp3_16k_128": (128, 16000), "mp3_16k_64": (64, 16000),
    "mp3_16k_32": (32, 16000), "mp3_16k_24": (24, 16000), "mp3_16k_16": (16, 16000), "mp3_16k_8": (8, 16000),
    "mp3_44k_128": (128, 44100), "mp3_44k_64": (64, 44100), "mp3_44k_32": (32, 44100),
    "mp3_8k_32": (32, 8000), "mp3_8k_16": (16, 8000),
}
CURRENT = ("mp3_16k_32", "mp3_16k_128")
SUPERSEDED = ("mp3_16k_64",)
N_FFT, HOP = 512, 160
ACTIVE_MS_DBFS = -55.0
MAX_LAG = 6000
MIN_ALIGN_CORRELATION = 0.5
EPS = 1e-10


def _spec(name):
    if name not in CANDIDATES:
        raise corruption.CorruptionError(f"unknown candidate {name!r}")
    return CANDIDATES[name]


def roundtrip(x: np.ndarray, kbps: int, encode_rate: int, ffmpeg: str | None = None) -> tuple[np.ndarray, int]:
    """Encode `x` (16 kHz mono float) to CBR MP3 at `encode_rate`, decode back to 16 kHz, and align by cross-correlation.

    Returns the aligned signal (same length as `x`) and the estimated lag in samples. Unlike `corruption.mp3_roundtrip` the delay is estimated,
    because resampling filters add delay on top of the codec delay.
    """
    path = corruption.find_ffmpeg(ffmpeg)
    pcm = np.clip(np.rint(np.asarray(x) * 32768.0), -32768, 32767).astype("<i2").tobytes()
    base = ["-nostdin", "-hide_banner", "-loglevel", "error"]
    encode = base + ["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0", "-map_metadata", "-1", "-c:a", "libmp3lame",
                     "-b:a", f"{int(kbps)}k", "-ar", str(int(encode_rate)), "-ac", "1", "-f", "mp3", "pipe:1"]
    decode = base + ["-f", "mp3", "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"]
    mp3 = corruption._run(path, encode, pcm)
    decoded = np.frombuffer(corruption._run(path, decode, mp3), dtype="<i2").astype(np.float64) / 32768.0
    return align(np.asarray(x, dtype=np.float64), decoded)


def align(x: np.ndarray, y: np.ndarray, max_lag: int = MAX_LAG) -> tuple[np.ndarray, int]:
    """Shift `y` back by the lag (0..max_lag) that maximises its normalised cross-correlation with `x`; return it cut or zero-padded to len(x).

    The reference segment is the first four seconds of `x`, shortened (to at least one second) when `y` is too short to search the whole lag range.
    Normalising by the window energy stops loud stretches winning over the stretch that actually matches.
    """
    n = len(x)
    length = min(n, 4 * SAMPLE_RATE)
    if len(y) - length < max_lag:
        length = max(min(length, len(y) - max_lag), min(length, SAMPLE_RATE))
    if len(y) < length:
        raise corruption.CorruptionError("decoded signal is shorter than the reference segment")
    seg = x[:length]
    window = y[: min(len(y), length + max_lag)]
    lags = len(window) - length + 1
    corr = fftconvolve(window, seg[::-1], mode="valid")                       # corr[k] = sum(window[k + i] * seg[i])
    energy = np.concatenate([[0.0], np.cumsum(window ** 2)])
    local = energy[length: length + lags] - energy[:lags]                      # sum of window[k : k + length] ** 2
    score = corr / np.sqrt(np.maximum(local, 1e-30) * max(float(np.sum(seg ** 2)), 1e-30))
    lag = int(np.argmax(score))
    correlation = float(score[lag])
    if correlation < MIN_ALIGN_CORRELATION:
        raise corruption.CorruptionError(f"could not align the decoded signal (correlation {correlation:.3f} at lag {lag})")
    shifted = y[lag: lag + n]
    if len(shifted) < n:
        shifted = np.concatenate([shifted, np.zeros(n - len(shifted))])
    return shifted, lag


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    count = 1 + (len(x) - N_FFT) // HOP
    return x[np.arange(N_FFT)[None, :] + HOP * np.arange(count)[:, None]]


def _power_spectra(frames: np.ndarray) -> np.ndarray:
    window = np.hanning(N_FFT)
    return np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 / np.sum(window ** 2)


def compare(x: np.ndarray, y: np.ndarray) -> dict:
    """Signal-level comparison of aligned `y` against its original `x` (same length)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise corruption.CorruptionError("signals must be one-dimensional and equal in length")
    px = float(np.mean(x ** 2))
    if px < corruption.MIN_POWER:
        raise corruption.CorruptionError("reference is silent")
    gain = float(np.dot(x, y) / np.dot(x, x))
    if not gain > 0:
        raise corruption.CorruptionError("non-positive gain: signals are not aligned")

    def snr(ref, est):
        err = float(np.mean((est - ref) ** 2))
        return float("inf") if err == 0 else float(10.0 * np.log10(float(np.mean(ref ** 2)) / err))

    fx = _frames(x)
    sx, sy = _power_spectra(fx), _power_spectra(_frames(y))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    active = np.mean(fx ** 2, axis=1) > 10.0 ** (ACTIVE_MS_DBFS / 10.0)              # frame mean square above -55 dBFS, judged on the original
    if active.any():
        d = 10 * np.log10(sx[active] + EPS) - 10 * np.log10(sy[active] + EPS)
        lsd = float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))
    else:
        lsd = float("nan")
    out = {"snr_raw_db": snr(x, y), "gain_db": float(20 * np.log10(gain)), "snr_gaincomp_db": snr(x, y / gain), "lsd_db": lsd,
           "active_frames": int(active.sum())}
    for name, f0 in (("retained_4k_pct", 4000.0), ("retained_7k_pct", 7000.0)):
        band = freqs >= f0
        ex = float(sx[:, band].sum())
        out[name] = float(100.0 * sy[:, band].sum() / ex) if ex > 0 else float("nan")
    return out


def probe(wave: np.ndarray, candidates=tuple(CANDIDATES), ffmpeg: str | None = None) -> dict:
    """Run every candidate on one recording. Returns {candidate: metrics + lag}; a candidate that cannot be measured is recorded with an error."""
    x = np.asarray(wave, dtype=np.float64)
    if x.ndim != 1 or x.size < corruption.MIN_SAMPLES:
        raise corruption.CorruptionError("waveform is not a usable one-dimensional recording")
    result = {}
    for name in candidates:
        kbps, rate = _spec(name)
        try:
            y, lag = roundtrip(x, kbps, rate, ffmpeg)
            result[name] = {**compare(x, y), "lag_samples": lag}
        except corruption.CorruptionError as exc:
            result[name] = {"error": str(exc)}
    return result


def summarise(rows: list[dict], candidates=tuple(CANDIDATES), noise_targets=(20.0, 10.0)) -> dict:
    """Per candidate: median [min, max] of each metric over recordings, the median gain-compensated SNR per component, and where it sits
    relative to the noise conditions."""
    out = {}
    for name in candidates:
        r = [x for x in rows if x["candidate"] == name and not x.get("error")]
        errors = [x for x in rows if x["candidate"] == name and x.get("error")]
        entry = {"kbps": CANDIDATES[name][0], "encode_rate": CANDIDATES[name][1], "frozen_spec": name in CURRENT, "superseded": name in SUPERSEDED,
                 "n": len(r), "errors": len(errors)}
        for metric in ("snr_raw_db", "gain_db", "snr_gaincomp_db", "lsd_db", "retained_4k_pct", "retained_7k_pct", "lag_samples"):
            v = np.array([x[metric] for x in r], dtype=np.float64)
            v = v[np.isfinite(v)]
            entry[metric] = ({"median": round(float(np.median(v)), 3), "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}
                             if v.size else None)
        by_component = {}
        for c in sorted({x["component"] for x in r}):
            v = np.array([x["snr_gaincomp_db"] for x in r if x["component"] == c], dtype=np.float64)
            by_component[c] = round(float(np.median(v[np.isfinite(v)])), 3) if np.isfinite(v).any() else None
        entry["snr_gaincomp_median_by_component"] = by_component
        median = entry["snr_gaincomp_db"]["median"] if entry["snr_gaincomp_db"] else None
        if median is None:
            entry["relative_to_noise"] = "not measured"
        elif median > noise_targets[0]:
            entry["relative_to_noise"] = f"milder than noise_{int(noise_targets[0])} (gain-compensated SNR above {noise_targets[0]:g} dB)"
        elif median > noise_targets[1]:
            entry["relative_to_noise"] = f"between noise_{int(noise_targets[0])} and noise_{int(noise_targets[1])}"
        else:
            entry["relative_to_noise"] = f"at least as harsh as noise_{int(noise_targets[1])}"
        out[name] = entry
    return out


def make_figure(summary: dict, out_base, *, title_note: str = "") -> dict:
    """Dot-and-range chart: gain-compensated SNR (left) and energy kept above 7 kHz (right), one row per candidate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    plt.rcParams.update({"svg.hashsalt": "echoproof-r3-s1", "svg.fonttype": "none", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})
    names = [n for n in summary if summary[n]["snr_gaincomp_db"]]
    names.sort(key=lambda n: -summary[n]["snr_gaincomp_db"]["median"])             # mildest at the top
    ink, muted, accent, other = "#222222", "#6b6b6b", "#0072B2", "#8a8a8a"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, metric, label in ((axes[0], "snr_gaincomp_db", "SNR after removing the level change (dB); higher = milder"),
                              (axes[1], "retained_7k_pct", "Energy kept above 7 kHz (%); higher = milder")):
        for yi, n in zip(y, names):
            m = summary[n][metric]
            color = accent if summary[n]["frozen_spec"] else other
            ax.plot([m["min"], m["max"]], [yi, yi], color=color, lw=1.5, solid_capstyle="round", zorder=2)
            ax.plot(m["median"], yi, "o", ms=7, color=color, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.set_xlabel(label, color=ink)
        ax.tick_params(colors=muted)
    for target, name in ((20.0, "noise_20"), (10.0, "noise_10")):
        axes[0].axvline(target, color=muted, lw=1, ls="--", zorder=1)
        axes[0].text(target, y.max() + 0.7, name, ha="center", va="bottom", fontsize=8.5, color=muted)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{n}  (frozen spec)" if summary[n]["frozen_spec"] else f"{n}  (superseded v1)" if summary[n].get("superseded") else n
                             for n in names], color=ink)
    axes[1].set_xlim(left=min(0.0, axes[1].get_xlim()[0]))
    axes[0].set_ylim(-0.7, y.max() + 1.4)
    fig.suptitle("How much each MP3 setting changes validation audio (dot = median, line = range over recordings)", x=0.01, ha="left", color=ink, fontsize=11.5)
    mildest = names[0]
    harshest = names[-1]
    frozen = [n for n in names if summary[n]["frozen_spec"]]
    first = ("Blue rows are the frozen R3-S1 conditions (median " + " and ".join(f"{summary[n]['snr_gaincomp_db']['median']:.1f}" for n in frozen)
             + " dB). Dashed lines mark the two noise conditions' SNR targets.") if frozen else "Dashed lines mark the two noise conditions' SNR targets."
    lines = (first,
             f"Range: {mildest} is the mildest ({summary[mildest]['snr_gaincomp_db']['median']:.1f} dB) and {harshest} the harshest "
             f"({summary[harshest]['snr_gaincomp_db']['median']:.1f} dB). {title_note}".strip())
    text = "\n".join(textwrap.fill(line, width=int(fig.get_figwidth() * 10.5), subsequent_indent="   ") for line in lines)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    fig.text(0.01, 0.012, text, fontsize=8.5, ha="left", va="bottom", color=ink)
    paths = []
    for ext in ("png", "svg"):
        kwargs = {"dpi": 150} if ext == "png" else {}
        fig.savefig(f"{out_base}.{ext}", metadata={"Software": None} if ext == "png" else {"Date": None}, **kwargs)
        paths.append(f"{out_base}.{ext}")
    plt.close(fig)
    return {"files": paths, "interpretation": list(lines)}
