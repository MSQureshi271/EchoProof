"""Generate and validate self-contained software-only audio fixtures."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from echoproof.artifacts import atomic_write_new, create_run, write_json
from echoproof.training.seeding import named_seed


def _wav(signal, rate):
    buffer = io.BytesIO()
    sf.write(buffer, signal, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def create_fixtures(root, seed):
    run_id, directory = create_run(root)
    rng = np.random.default_rng(named_seed(seed, "fixtures"))
    def tone(rate, seconds, frequency):
        time = np.arange(int(rate * seconds), dtype=np.float64) / rate
        return (0.2 * np.sin(2 * np.pi * frequency * time) + rng.normal(0, 0.005, len(time))).astype(np.float32)
    mono = tone(16000, 1, 440)
    signals = {
        "mono_1s.wav": _wav(mono, 16000),
        "mono_4s.wav": _wav(tone(16000, 4, 220), 16000),
        "short.wav": _wav(tone(16000, 0.25, 330), 16000),
        "rate_8k.wav": _wav(tone(8000, 1, 440), 8000),
        "stereo.wav": _wav(np.stack([mono, mono * 0.5], axis=1), 16000),
        "silence.wav": _wav(np.zeros(16000, dtype=np.float32), 16000),
        "empty.wav": b"",
        "corrupt.wav": b"not a valid audio file",
    }
    for name, data in signals.items():
        atomic_write_new(directory / name, data)
    manifest = {"schema_version": 1, "kind": "generated_software_fixtures", "algorithm": "tones-noise-pcm16-v1",
                "seed": seed, "named_seed": named_seed(seed, "fixtures"), "run_id": run_id,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(signals.items())}}
    write_json(directory / "fixture_manifest.json", manifest)
    return directory, manifest


def read_fixture(directory, manifest, name):
    if name not in manifest["files"] or Path(name).name != name:
        raise ValueError("fixture is not registered")
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ValueError("fixture path escapes fixture directory")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
        raise ValueError("fixture checksum mismatch")
    return data
