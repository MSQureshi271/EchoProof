"""Bounded document capture and failure evidence without real network calls."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("p202", Path(__file__).resolve().parents[1] / "scripts/p2_02_source_evidence.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.org/document"


class SourceTests(unittest.TestCase):
    def test_capture_hash_and_oversized_document(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "document"
            with patch.object(source.urllib.request, "urlopen", return_value=Response(b"text")):
                result = source.capture("https://example.org/document", path)
            self.assertEqual(result["sha256"], hashlib.sha256(b"text").hexdigest())
            other = Path(temp) / "oversized"
            with patch.object(source.urllib.request, "urlopen", return_value=Response(b"12345")):
                with self.assertRaises(ValueError):
                    source.capture("https://example.org/document", other, limit=4)
            self.assertFalse(other.exists())

    def test_network_failures_preserve_pending_status_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "configs").mkdir()
            (root / "configs/data_sources.json").write_text("{}")
            with patch.object(source.urllib.request, "urlopen", side_effect=OSError("offline")):
                report, folder = source.run(root)
            self.assertEqual(report["status"], "needs_attention")
            self.assertFalse(report["local_publisher_revision_verified"])
            self.assertEqual(len(report["failures"]), 3)
            self.assertFalse((root / "dataset").exists())
            manifest = json.loads((folder / "artifact_manifest.json").read_text())
            self.assertEqual(manifest["files"]["source_inventory.json"], hashlib.sha256((folder / "source_inventory.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
