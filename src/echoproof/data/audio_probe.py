"""Generate fixtures and exercise the actual decoding and MP3 path."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml
from echoproof.artifacts import atomic_write_new, canonical_json, create_run, digest, make_manifest, write_json
from echoproof.environment import capture_environment
from echoproof.training.seeding import seed_everything
from .audio import AudioError, decode_audio, windows
from .fixtures import create_fixtures, read_fixture


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _ffmpeg(args):
    try:
        result = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"FFmpeg unavailable or timed out: {exc}") from exc
    if result.returncode:
        raise RuntimeError("FFmpeg failed: " + result.stderr[-1000:])


def probe_audio(config, project_root):
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    run_id, folder = create_run(config.paths.run_root)
    seeds = seed_everything(config.seed)
    fixture_dir, manifest = create_fixtures(config.paths.fixtures, config.seed)
    second_dir, second = create_fixtures(config.paths.fixtures, config.seed)
    _require(manifest["files"] == second["files"], "fixture generation was not reproducible")
    specs = {"mono_1s.wav": (16000, 1, 16000), "mono_4s.wav": (16000, 1, 64000),
             "short.wav": (16000, 1, 4000), "rate_8k.wav": (8000, 1, 8000),
             "stereo.wav": (16000, 2, 16000), "silence.wav": (16000, 1, 16000)}
    records = {}
    decoded = {}
    for name, (rate, channels, frames) in specs.items():
        waveform = decode_audio(read_fixture(fixture_dir, manifest, name), target_rate=config.audio.sample_rate)
        _require((waveform.original_sample_rate, waveform.original_channels, waveform.original_frames) == (rate, channels, frames), f"original metadata mismatch: {name}")
        _require(abs(len(waveform.samples) / waveform.sample_rate - frames / rate) <= 1 / waveform.sample_rate, f"resampled duration mismatch: {name}")
        _require(waveform.samples.dtype == np.float32 and np.isfinite(waveform.samples).all(), f"invalid samples: {name}")
        parts = list(windows(waveform, config.audio.crop_samples))
        _require(sum(valid for _, valid, _ in parts) == len(waveform.samples), f"invalid coverage: {name}")
        for sample, valid, _ in parts:
            _require(len(sample) == config.audio.crop_samples and np.all(sample[valid:] == 0), f"invalid padding: {name}")
        decoded[name] = waveform
        records[name] = {"original_rate": rate, "original_channels": channels, "original_frames": frames,
                         "output_rate": waveform.sample_rate, "output_samples": len(waveform.samples),
                         "valid_lengths": [valid for _, valid, _ in parts], "quality_flags": list(waveform.quality_flags)}
    _require("silence" in decoded["silence.wav"].quality_flags, "silence flag absent")
    _require(np.allclose(decoded["stereo.wav"].samples, decoded["mono_1s.wav"].samples * 0.75, atol=5e-5), "stereo downmix mismatch")
    errors = {}
    for name in ("empty.wav", "corrupt.wav"):
        try:
            decode_audio(read_fixture(fixture_dir, manifest, name))
        except AudioError as exc:
            errors[name] = {"status": "invalid_audio", "message": str(exc)}
        else:
            raise RuntimeError(f"{name} unexpectedly decoded")

    with tempfile.TemporaryDirectory(prefix="echoproof-p1-07-") as tmp:
        source, compressed, restored = (Path(tmp) / name for name in ("source.wav", "encoded.mp3", "decoded.wav"))
        source.write_bytes(read_fixture(fixture_dir, manifest, "mono_1s.wav"))
        _ffmpeg(["-i", str(source), "-t", "1", "-c:a", "libmp3lame", "-b:a", "128k", "-y", str(compressed)])
        _ffmpeg(["-i", str(compressed), "-t", "1.1", "-c:a", "pcm_s16le", "-y", str(restored)])
        result = decode_audio(restored.read_bytes(), target_rate=config.audio.sample_rate)
        original = decoded["mono_1s.wav"].samples
        length_difference = abs(len(result.samples) - len(original)) / config.audio.sample_rate
        _require(length_difference <= 0.08, "MP3 duration differs by more than 0.08 seconds")
        length = min(len(original), len(result.samples))
        error = float(np.sqrt(np.mean((result.samples[:length] - original[:length]) ** 2)))
        _require(error <= 0.08, "MP3 waveform RMS error exceeds 0.08")
        mp3 = {"bitrate": "128k", "duration_difference_seconds": length_difference,
               "duration_tolerance_seconds": 0.08, "waveform_rms_error": error, "rms_error_tolerance": 0.08}

    metrics = {"schema_version": 1, "scope": "software_probe", "fixtures_are_speech": False,
               "fixture_directory": str(fixture_dir), "second_fixture_directory": str(second_dir),
               "fixtures_reproducible": True, "fixture_manifest": manifest, "waveforms": records, "invalid_inputs": errors, "mp3": mp3}
    env = capture_environment(project_root)
    env.update({"schema_version": 1, "run_id": run_id, "task": "P1-07", "started_utc": started,
                "config_hash": digest(asdict(config)), "seeds": seeds, "device": "cpu"})
    atomic_write_new(folder / "resolved_config.yaml", yaml.safe_dump(asdict(config), sort_keys=True).encode("utf-8"))
    write_json(folder / "environment.json", env)
    atomic_write_new(folder / "events.jsonl", (canonical_json({"event": "fixture_audio_probe_complete", "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n").encode("utf-8"))
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "artifact_manifest.json", make_manifest(folder, ["resolved_config.yaml", "environment.json", "events.jsonl", "metrics.json"]))
    check = subprocess.run([sys.executable, "-c", "import json,sys; from echoproof.artifacts import verify_manifest; print(json.dumps(verify_manifest(sys.argv[1])))", str(folder)], capture_output=True, text=True, timeout=60)
    _require(check.returncode == 0, "fresh-process artifact verification failed: " + check.stderr[-1000:])
    verification = json.loads(check.stdout)
    write_json(folder / "status.json", {"schema_version": 1, "status": "success", "run_id": run_id,
               "elapsed_seconds": time.monotonic() - start, "verification": verification})
    return {"status": "success", "run_id": run_id, "run_dir": str(folder), "verification": verification,
            "fixture_dir": str(fixture_dir), "scope": "software_probe"}
