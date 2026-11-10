"""R3-S1 corruption suite: frozen specification, determinism, SNR accuracy, level guard, MP3 round trip and failure paths."""
import hashlib
from unittest import mock

import numpy as np
import pytest

from echoproof.data import corruption as c

PINNED_SPEC_SHA256 = "d68c4ed61a29ad010bfb10b8ac8995dad1c02af72a914828a050613c6fc805e0"
SR = c.SAMPLE_RATE


def _ffmpeg_ok():
    try:
        c.find_ffmpeg()
        return True
    except c.CorruptionError:
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg with libmp3lame is not available")


def speechlike(seconds=3.0, peak=0.4, seed=1):
    t = np.arange(int(SR * seconds)) / SR
    rng = np.random.default_rng(seed)
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
    x = env * sum(np.sin(2 * np.pi * f * t * (1 + 0.02 * np.sin(2 * np.pi * 4 * t))) / k for k, f in enumerate((140, 280, 420, 560, 700, 980, 1500, 2300), 1))
    x = x + rng.normal(0, 0.01, t.size)
    return (x / np.abs(x).max() * peak).astype(np.float32)


def ids_with_color(color, n=1):
    out, i = [], 0
    while len(out) < n:
        pid = f"recording:genuine:{i}-x"
        if c.noise_color(pid) == color:
            out.append(pid)
        i += 1
    return out


# ---- specification -----------------------------------------------------------------------------------------------------------------

def test_exactly_four_frozen_conditions_and_spec_hash_is_pinned():
    assert c.SPEC_VERSION == 2 and c.SPEC["version"] == 2
    assert c.CONDITION_NAMES == ("mp3_32", "mp3_128", "noise_20", "noise_10")
    assert c.CONDITIONS["mp3_32"]["bitrate_kbps"] == 32 and c.CONDITIONS["mp3_128"]["bitrate_kbps"] == 128
    assert "mp3_64" not in c.CONDITIONS                             # version 1's 64 kbps condition was replaced by 32 kbps
    assert c.CONDITIONS["noise_20"]["snr_db"] == 20.0 and c.CONDITIONS["noise_10"]["snr_db"] == 10.0
    # A deliberate specification change must update this pin and the decision-log entry together.
    assert c.SPEC_SHA256 == PINNED_SPEC_SHA256


def test_variant_id_keeps_parent_and_rejects_unknown_condition():
    assert c.variant_id("recording:xtts:7-a", "mp3_32") == "recording:xtts:7-a@mp3_32"
    with pytest.raises(c.CorruptionError):
        c.variant_id("recording:xtts:7-a", "mp3_64")                 # a version 1 name is no longer a condition


# ---- noise -------------------------------------------------------------------------------------------------------------------------

def test_noise_color_is_deterministic_balanced_and_shared_by_both_noise_conditions():
    colors = [c.noise_color(f"recording:genuine:{i}-x") for i in range(400)]
    assert colors == [c.noise_color(f"recording:genuine:{i}-x") for i in range(400)]
    assert 140 < colors.count("white") < 260
    pid = ids_with_color("pink")[0]
    x = speechlike()
    assert c.corrupt(x, pid, "noise_20").params["color"] == c.corrupt(x, pid, "noise_10").params["color"] == "pink"


@pytest.mark.parametrize("color", c.NOISE_COLORS)
@pytest.mark.parametrize("condition", ["noise_20", "noise_10"])
def test_measured_snr_matches_target(color, condition):
    x = speechlike()
    for pid in ids_with_color(color, 3):
        r = c.corrupt(x, pid, condition)
        assert r.params["color"] == color
        assert r.checks["snr_checked"] is True
        assert abs(r.checks["measured_snr_db"] - c.CONDITIONS[condition]["snr_db"]) < 0.1


def test_spectral_slopes_white_flat_pink_minus_one():
    rng = np.random.default_rng(3)
    for color, lo, hi in (("white", -0.15, 0.15), ("pink", -1.15, -0.85)):
        noise = c.make_noise(SR * 20, color, rng)
        assert abs(noise.mean()) < 1e-9 and abs(np.mean(noise ** 2) - 1.0) < 1e-9
        spec = np.abs(np.fft.rfft(noise)) ** 2
        freqs = np.fft.rfftfreq(noise.size, 1 / SR)
        band = (freqs > 100) & (freqs < 6000)
        edges = np.logspace(np.log10(100), np.log10(6000), 30)
        idx = np.digitize(freqs[band], edges)
        bf = np.array([freqs[band][idx == k].mean() for k in range(1, len(edges)) if (idx == k).any()])
        bp = np.array([spec[band][idx == k].mean() for k in range(1, len(edges)) if (idx == k).any()])
        slope = np.polyfit(np.log10(bf), np.log10(bp), 1)[0]
        assert lo < slope < hi, (color, slope)


def test_noise_differs_by_recording_and_condition_but_repeats_exactly():
    x = speechlike()
    a1, a2 = c.corrupt(x, "recording:genuine:1-a", "noise_20"), c.corrupt(x, "recording:genuine:1-a", "noise_20")
    assert a1.pcm_sha256 == a2.pcm_sha256 and np.array_equal(a1.wave, a2.wave)
    assert c.corrupt(x, "recording:genuine:2-a", "noise_20").pcm_sha256 != a1.pcm_sha256
    assert c.corrupt(x, "recording:genuine:1-a", "noise_10").pcm_sha256 != a1.pcm_sha256
    assert c.corrupt(x, "recording:genuine:1-a", "noise_20", seed=7).pcm_sha256 != a1.pcm_sha256


def test_loud_input_is_scaled_by_one_gain_without_changing_snr():
    x = speechlike(peak=0.99)
    r = c.corrupt(x, ids_with_color("white")[0], "noise_10")
    assert r.params["gain"] < 1.0
    assert np.abs(r.wave).max() <= c.PEAK_LIMIT + 1 / 32768
    assert abs(r.checks["measured_snr_db"] - 10.0) < 0.1


def test_output_is_exactly_what_16_bit_storage_would_hold():
    r = c.corrupt(speechlike(), "recording:genuine:1-a", "noise_20")
    pcm = np.rint(r.wave.astype(np.float64) * 32768).astype("<i2")
    assert np.array_equal(pcm.astype(np.float32) / 32768, r.wave)
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == r.pcm_sha256
    assert r.wave.dtype == np.float32 and r.n_samples == r.wave.size == speechlike().size


def test_input_is_not_modified():
    x = speechlike()
    before = x.copy()
    c.corrupt(x, "recording:genuine:1-a", "noise_10")
    assert np.array_equal(x, before)


# ---- rejected inputs ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    np.zeros(SR, dtype=np.float32),                              # silent: no SNR
    np.ones((SR, 2), dtype=np.float32) * 0.1,                    # not mono
    np.full(100, 0.1, dtype=np.float32),                         # too short
    np.array([0.1] * 5000 + [np.nan], dtype=np.float32),         # non-finite
    np.array([0.1] * 5000 + [1.5], dtype=np.float32),            # beyond full scale
    (np.ones(SR) * 100).astype(np.int16),                        # not floating point
])
def test_invalid_waveforms_are_rejected(bad):
    with pytest.raises(c.CorruptionError):
        c.corrupt(bad, "recording:genuine:1-a", "noise_20")


def test_bad_identity_and_unknown_condition_are_rejected():
    x = speechlike()
    with pytest.raises(c.CorruptionError):
        c.corrupt(x, "no-namespace", "noise_20")
    with pytest.raises(c.CorruptionError):
        c.corrupt(x, "recording:genuine:1-a", "reverb")


# ---- MP3 ---------------------------------------------------------------------------------------------------------------------------

@needs_ffmpeg
@pytest.mark.parametrize("kbps", [32, 128])
def test_mp3_roundtrip_is_aligned_deterministic_and_at_the_target_bitrate(kbps):
    x = speechlike()
    r = c.corrupt(x, "recording:genuine:1-a", f"mp3_{kbps}")
    assert r.wave.shape == x.shape
    assert r.checks["correlation"] > 0.99
    assert abs(r.checks["effective_kbps"] - kbps) / kbps < 0.1
    assert abs(r.checks["length_diff_samples"]) <= c.MP3_MAX_LENGTH_DIFF
    assert r.pcm_sha256 == c.corrupt(x, "recording:genuine:1-a", f"mp3_{kbps}").pcm_sha256
    assert not np.array_equal(r.wave, x)                         # it really is a lossy copy


@needs_ffmpeg
def test_higher_bitrate_is_closer_to_the_original_on_broadband_audio():
    rng = np.random.default_rng(5)
    x = (np.convolve(rng.normal(0, 1, SR * 3), np.ones(3) / 3, mode="same") * 0.1).astype(np.float32)
    s32 = c.corrupt(x, "recording:genuine:1-a", "mp3_32").checks["measured_snr_db"]
    s128 = c.corrupt(x, "recording:genuine:1-a", "mp3_128").checks["measured_snr_db"]
    assert s128 > s32 + 3                                           # 32 kbps is clearly the harsher of the two


@needs_ffmpeg
def test_corrupt_all_returns_four_variants_with_the_same_parent():
    out = c.corrupt_all(speechlike(), "recording:xtts:9-z")
    assert [v.condition for v in out] == list(c.CONDITION_NAMES)
    assert {v.parent_id for v in out} == {"recording:xtts:9-z"}
    assert [v.variant_id for v in out] == [f"recording:xtts:9-z@{n}" for n in c.CONDITION_NAMES]
    assert len({v.pcm_sha256 for v in out}) == 4


@needs_ffmpeg
def test_ffmpeg_is_called_with_fixed_argument_lists_and_no_shell():
    calls = []
    real = c.subprocess.run

    def spy(args, **kwargs):
        calls.append((args, kwargs))
        return real(args, **kwargs)

    with mock.patch.object(c.subprocess, "run", spy):
        c.corrupt(speechlike(), "recording:genuine:1-a", "mp3_32")
    assert len(calls) >= 2
    for args, kwargs in calls:
        assert isinstance(args, list) and all(isinstance(a, str) for a in args)
        assert not kwargs.get("shell", False) and kwargs.get("timeout")
    assert any("libmp3lame" in args for args, _ in calls) and any("-b:a" in args and "32k" in args for args, _ in calls)


@needs_ffmpeg
def test_roundtrip_check_rejects_a_misaligned_or_wrong_length_decode():
    rng = np.random.default_rng(9)
    x = (rng.normal(0, 0.1, SR * 2)).astype(np.float32)
    real = c._run
    pcm = lambda a: a.astype("<i2").tobytes()

    def shifted(path, args, data):                                # decoder output delayed by 600 more samples than the spec assumes
        out = real(path, args, data)
        if "pcm_s16le" in args:
            d = np.frombuffer(out, dtype="<i2")
            return pcm(np.concatenate([np.zeros(600, dtype="<i2"), d]))
        return out

    def truncated(path, args, data):
        out = real(path, args, data)
        return out[: len(out) // 2] if "pcm_s16le" in args else out

    with mock.patch.object(c, "_run", shifted), pytest.raises(c.CorruptionError, match="misaligned|aligned length"):
        c.corrupt(x, "recording:genuine:1-a", "mp3_128")
    with mock.patch.object(c, "_run", truncated), pytest.raises(c.CorruptionError, match="aligned length"):
        c.corrupt(x, "recording:genuine:1-a", "mp3_128")


def test_missing_ffmpeg_and_failed_ffmpeg_raise_clear_errors(tmp_path):
    with pytest.raises(c.CorruptionError, match="not found"):
        c.find_ffmpeg(str(tmp_path / "no-such-ffmpeg"))
    with mock.patch.object(c.subprocess, "run", return_value=mock.Mock(returncode=1, stdout=b"", stderr=b"boom")):
        with pytest.raises(c.CorruptionError, match="ffmpeg failed"):
            c._run("ffmpeg", ["-x"], b"")
    with mock.patch.object(c.subprocess, "run", side_effect=c.subprocess.TimeoutExpired("ffmpeg", 1)):
        with pytest.raises(c.CorruptionError, match="timed out"):
            c._run("ffmpeg", ["-x"], b"")


@needs_ffmpeg
def test_spec_record_carries_hash_and_ffmpeg_build():
    rec = c.spec_record()
    assert rec["spec_sha256"] == c.SPEC_SHA256 and rec["master_seed"] == c.MASTER_SEED
    assert rec["ffmpeg"]["version"].startswith("ffmpeg version") and rec["ffmpeg"]["mp3_encoder"] == "libmp3lame"
    assert "ffmpeg" not in c.spec_record(with_ffmpeg=False)
