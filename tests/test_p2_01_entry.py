"""Entry checks preserve missing datasets, detect tampering and retain source boundaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("p201", Path(__file__).resolve().parents[1] / "scripts/p2_01_entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class EntryTests(unittest.TestCase):
    def fixture(self, root):
        for name in entry.ROOTS.values():
            (root / name).mkdir(parents=True)
        for name in ("plans/PHASE2.md", "docs/research_contract.md", "docs/phase1_gate_checklist.md", "docs/p1_13_review.md",
                     "reports/phase1/p1_13/reproduce-20261029T192537Z-816edf2c/verification.json"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        lock = root / entry.LOCK
        lock.mkdir(parents=True)
        (lock / "requirements-lock.txt").write_text("fixture==1\n")
        (lock / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "algorithm": "sha256",
                "files": {"requirements-lock.txt": entry.sha(lock / "requirements-lock.txt")}}))

    def test_missing_root_is_reported_without_creating_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            report = entry.collect(temp)
            self.assertEqual(report["status"], "needs_attention")
            self.assertFalse((Path(temp) / "dataset").exists())
            self.assertEqual(len(report["paths"]), 4)

    def test_paths_and_lock_do_not_imply_source_or_external_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture(Path(temp))
            report, folder = entry.run(temp, "colab_drive")
            self.assertEqual(report["status"], "entry_ready_for_review")
            self.assertFalse(report["source_state"]["phonemedf"]["source_verified"])
            self.assertFalse(report["source_state"]["external_source"]["eligible"])
            self.assertIsNone(report["capacity"]["current_free_drive_gb"])
            manifest = json.loads((folder / "artifact_manifest.json").read_text())
            self.assertEqual(manifest["files"]["entry.json"], entry.sha(folder / "entry.json"))

    def test_tampered_lock_and_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.fixture(root)
            (root / entry.LOCK / "requirements-lock.txt").write_text("changed")
            report = entry.collect(root)
            self.assertEqual(report["prior_lock"]["status"], "needs_attention")
            self.assertTrue(any(i["kind"] == "lock_integrity" for i in report["issues"]))
            with self.assertRaises(ValueError):
                entry.contained(root, "../escape")


if __name__ == "__main__":
    unittest.main()
