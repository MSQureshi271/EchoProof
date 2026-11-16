"""Offline checks for R4-N3 bundle retrieval and lock enforcement."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_bundles as fetch
import verify_install


class LockedBundleRetrieval(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.manifest = {"bundle_id": "bundle:test-r4-n3"}
        raw_manifest = json.dumps(self.manifest, separators=(",", ":")).encode()
        (self.tmp / "bundle.json").write_bytes(raw_manifest)
        payload = b"component bytes"
        component = self.tmp / "model" / "weights.bin"
        component.parent.mkdir()
        component.write_bytes(payload)
        self.entry = {
            "bundle_id": self.manifest["bundle_id"],
            "bundle_json_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "components": {"model/weights.bin": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}},
        }

    def test_local_production_bundle_matches_all_lock_records(self):
        lock = json.loads((ROOT / "locks/B_lock_gr3.json").read_text(encoding="utf-8"))
        entry = lock["candidates"]["WavLM"]
        bundle = ROOT / entry["bundle_path"]
        fetch.verify(bundle, entry)
        manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["bundle_id"], entry["bundle_id"])
        self.assertEqual(entry["bundle_id"], "bundle:r3-n1-wavlm-head-B-calibrated")

    def test_valid_bundle_passes(self):
        fetch.verify(self.tmp, self.entry)

    def test_missing_component_fails(self):
        (self.tmp / "model/weights.bin").unlink()
        with self.assertRaisesRegex(ValueError, "file set differs|Missing locked component"):
            fetch.verify(self.tmp, self.entry)

    def test_changed_component_fails(self):
        (self.tmp / "model/weights.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "byte count|mismatch"):
            fetch.verify(self.tmp, self.entry)

    def test_manifest_identity_must_match_lock(self):
        self.manifest["bundle_id"] = "bundle:wrong-identity"
        raw_manifest = json.dumps(self.manifest, separators=(",", ":")).encode()
        (self.tmp / "bundle.json").write_bytes(raw_manifest)
        self.entry["bundle_json_sha256"] = hashlib.sha256(raw_manifest).hexdigest()
        with self.assertRaisesRegex(ValueError, "Bundle identity mismatch"):
            fetch.verify(self.tmp, self.entry)

    def test_extra_file_fails(self):
        (self.tmp / "unlocked.txt").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file set differs"):
            fetch.verify(self.tmp, self.entry)

    def test_zip_path_escape_fails(self):
        archive = self.tmp / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../outside.txt", "unsafe")
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            fetch.prepare_source(archive, self.tmp / "unpack")

    def test_default_retrieval_is_protocol_b_wavlm_only(self):
        records = fetch.read_locks(False)
        self.assertEqual([(lock, name) for lock, name, _ in records], [("B_lock_gr3.json", "WavLM")])
        self.assertEqual(len(fetch.read_locks(True)), 8)

    def test_install_verifier_exits_nonzero_for_prediction_mismatch(self):
        fixture = self.tmp / "tests/fixtures/r4_n3/fixture.audio"
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b"software fixture")
        expectation = fixture.parent / "expected_prediction.json"
        expectation.write_text(json.dumps({"status": "verified", "score": 0.4, "assessment": "synthetic",
                                           "score_type": "calibrated_probability", "threshold": 0.2}), encoding="utf-8")
        status = SimpleNamespace(ok=True, verified_against_lock=True, message="",
                                 bundle=SimpleNamespace(spec=SimpleNamespace(decision_rule=SimpleNamespace(threshold=0.2))))
        fake_cli = SimpleNamespace(returncode=0, stdout=json.dumps({"status": "success", "score": 0.5,
                                                                     "assessment": "synthetic", "score_type": "calibrated_probability"}), stderr="")
        with (mock.patch.object(verify_install, "ROOT", self.tmp),
              mock.patch("echoproof.app_support.prepare_bundle", return_value=status),
              mock.patch.object(verify_install.subprocess, "run", return_value=fake_cli)):
            self.assertEqual(verify_install.main(), 1)


if __name__ == "__main__":
    unittest.main()
