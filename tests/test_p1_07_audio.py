"""P1-07 tests use generated signals only."""
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
from echoproof.data.audio import AudioError, decode_audio, windows
from echoproof.data.fixtures import create_fixtures, read_fixture


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder, self.manifest = create_fixtures(self.tmp.name, 42)

    def test_reproducibility_and_tampering(self):
        _, other = create_fixtures(self.tmp.name, 42)
        self.assertEqual(self.manifest["files"], other["files"])
        (self.folder / "short.wav").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            read_fixture(self.folder, self.manifest, "short.wav")

    def test_resampling_stereo_padding_and_silence(self):
        def load(name):
            return decode_audio(read_fixture(self.folder, self.manifest, name))
        resampled = load("rate_8k.wav")
        self.assertEqual((resampled.original_sample_rate, resampled.sample_rate, len(resampled.samples)), (8000, 16000, 16000))
        mono, stereo = load("mono_1s.wav"), load("stereo.wav")
        self.assertEqual(stereo.original_channels, 2)
        np.testing.assert_allclose(stereo.samples, mono.samples * 0.75, atol=5e-5)
        sample, valid, start = list(windows(load("short.wav"), 64000))[0]
        self.assertEqual((valid, start, len(sample)), (4000, 0, 64000))
        self.assertTrue(np.all(sample[valid:] == 0))
        self.assertIn("silence", load("silence.wav").quality_flags)

    def test_invalid_and_nonfinite_audio(self):
        for name in ("empty.wav", "corrupt.wav"):
            with self.subTest(name=name), self.assertRaises(AudioError):
                decode_audio(read_fixture(self.folder, self.manifest, name))
        buffer = io.BytesIO()
        sf.write(buffer, np.array([0.0, float("nan")], dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
        with self.assertRaises(AudioError):
            decode_audio(buffer.getvalue())
        with self.assertRaises(AudioError):
            decode_audio(read_fixture(self.folder, self.manifest, "mono_4s.wav"), max_seconds=1)

    def test_windows_cover_long_recordings(self):
        audio = decode_audio(read_fixture(self.folder, self.manifest, "mono_4s.wav"))
        parts = list(windows(audio, 20000))
        self.assertEqual([valid for _, valid, _ in parts], [20000, 20000, 20000, 4000])
        self.assertEqual([start for _, _, start in parts], [0, 20000, 40000, 60000])


if __name__ == "__main__":
    unittest.main()
