"""Checks for artifact safety and reproducible random streams."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from echoproof.artifacts import atomic_write_new, create_run, digest, make_manifest, verify_manifest, write_json
from echoproof.training.seeding import named_seed


class ArtifactTests(unittest.TestCase):
    def test_hash_stable_and_changes_with_config(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        self.assertNotEqual(digest({"seed": 42}), digest({"seed": 43}))
        with self.assertRaises(ValueError):
            digest({"invalid": float("nan")})

    def test_atomic_write_refuses_overwrite_and_cleans_failed_write(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "artifact.json"
            atomic_write_new(path, b"first")
            with self.assertRaises(FileExistsError):
                atomic_write_new(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")
            failed = Path(temp) / "status.json"
            with mock.patch("echoproof.artifacts.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    atomic_write_new(failed, b'{"status":"success"}')
            self.assertFalse(failed.exists())
            self.assertEqual(list(Path(temp).glob(".echoproof-*")), [])

    def test_manifest_in_new_process_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            _, directory = create_run(temp)
            write_json(directory / "metrics.json", {"scope": "software_probe"})
            write_json(directory / "artifact_manifest.json", make_manifest(directory, ["metrics.json"]))
            result = subprocess.run([sys.executable, "-c", "import json,sys; from echoproof.artifacts import verify_manifest; print(json.dumps(verify_manifest(sys.argv[1])))", str(directory)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["verified_files"], 1)
            (directory / "metrics.json").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_manifest(directory)

    def test_named_seeds_are_stable_and_distinct(self):
        self.assertEqual(named_seed(42, "crops"), named_seed(42, "crops"))
        self.assertNotEqual(named_seed(42, "crops"), named_seed(42, "augmentation"))
        with self.assertRaises(ValueError):
            named_seed(-1, "crops")

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "torch or numpy unavailable")
    def test_fresh_process_seeded_samples(self):
        from echoproof.reproducibility import _seeded_sample
        first = _seeded_sample(42)
        self.assertEqual(first, _seeded_sample(42))
        self.assertNotEqual(first, _seeded_sample(43))


if __name__ == "__main__":
    unittest.main()
