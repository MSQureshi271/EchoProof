import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.evaluation import r4_n1_a_lock as L  # noqa: E402

HAVE_PROJECT = all((ROOT / b).is_dir() for b in L.BUNDLES.values()) and (ROOT / "locks/B_lock_gr3.json").is_file()


class BLockTests(unittest.TestCase):
    def test_accepts_the_crlf_and_lf_forms_and_rejects_changes(self):
        crlf = b'{"a": 1,\r\n "b": 2}\r\n'
        with tempfile.TemporaryDirectory() as tmp, patch.object(L, "B_LOCK_SHA256", hashlib.sha256(crlf).hexdigest()):
            root = Path(tmp)
            (root / "locks").mkdir()
            for data, expected in ((crlf, True), (crlf.replace(b"\r\n", b"\n"), True), (crlf + b" ", False), (crlf.replace(b"2", b"3"), False)):
                (root / L.B_LOCK).write_bytes(data)
                self.assertEqual(L.b_lock_matches(root), expected)


@unittest.skipUnless(HAVE_PROJECT, "the four A bundles and their runs are not all present")
class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = L.build_lock(ROOT, allow_dirty=True)

    def mutated(self):
        return copy.deepcopy(self.record)

    def test_the_record_describes_four_verified_candidates(self):
        r = self.record
        self.assertEqual(r["protocol"], "A")
        self.assertEqual(set(r["candidates"]), set(L.MODELS))
        self.assertEqual(r["split"]["split_hash"], L.SPLIT_SHA256)
        self.assertIsNone(r["production_selection"]["selected"])
        self.assertFalse(r["test_access_audit"]["test_partition_accessed"])
        for model, c in r["candidates"].items():
            self.assertEqual(c["model_version"], L.MODEL_VERSIONS[model])
            self.assertTrue(c["replay"]["passed"])
            self.assertFalse(c["bundle_extra"].get("locked", False))
            self.assertFalse(c["bundle_extra"].get("promoted", False))
        L.verify_record(ROOT, r)

    def test_thresholds_match_the_saved_evidence(self):
        c = self.record["candidates"]
        self.assertAlmostEqual(c["LR"]["decision_rule"]["threshold"], 0.5910800081638822, places=15)
        self.assertAlmostEqual(c["BiGRU"]["decision_rule"]["threshold"], 0.610876834796857, places=15)
        self.assertTrue(c["WavLM"]["calibration"]["accepted"] and c["AST"]["calibration"]["accepted"])
        self.assertFalse(c["LR"]["calibration"]["accepted"] or c["BiGRU"]["calibration"]["accepted"])

    def test_verification_refuses_changes(self):
        cases = []
        r = self.mutated(); del r["candidates"]["AST"]; cases.append(r)
        r = self.mutated(); r["split"]["split_hash"] = "0" * 64; cases.append(r)
        r = self.mutated(); r["test_access_audit"]["test_partition_accessed"] = True; cases.append(r)
        r = self.mutated(); r["b_lock"]["sha256"] = "0" * 64; cases.append(r)
        r = self.mutated(); r["candidates"]["LR"]["bundle_json_sha256"] = "0" * 64; cases.append(r)
        r = self.mutated(); k = next(iter(r["candidates"]["BiGRU"]["components"])); r["candidates"]["BiGRU"]["components"][k]["sha256"] = "0" * 64; cases.append(r)
        r = self.mutated(); r["candidates"]["WavLM"]["decision_rule"]["threshold"] += 0.01; cases.append(r)
        r = self.mutated(); r["candidates"]["LR"]["replay"]["passed"] = False; cases.append(r)
        r = self.mutated(); r["candidates"]["LR"]["replay"]["decision_flips"] = 3; r["candidates"]["LR"]["replay"]["flips_explained"] = False; cases.append(r)
        r = self.mutated(); first = next(v for v in r["candidates"]["AST"]["evidence"].values() if isinstance(v, dict)); first["sha256"] = "0" * 64; cases.append(r)
        for i, record in enumerate(cases):
            with self.subTest(case=i), self.assertRaises((L.ALockError, KeyError)):
                L.verify_record(ROOT, record)

    def test_a_dirty_tree_is_refused_unless_allowed(self):
        real_git = L.git

        def fake(root, *args):
            return " M something.py" if args[:1] == ("status",) else real_git(root, *args)
        with patch.object(L, "git", fake):
            with self.assertRaises(L.ALockError):
                L.build_lock(ROOT)
            self.assertTrue(L.build_lock(ROOT, allow_dirty=True)["code"]["working_tree_dirty"])

    def test_the_script_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "A_lock.json"
            out.write_text("existing", encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/r4_n1/a_lock.py"), "--write", "--allow-dirty", "--out", str(out)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr + result.stdout)
            self.assertEqual(out.read_text(encoding="utf-8"), "existing")


class EvidenceGuardTests(unittest.TestCase):
    def test_missing_files_and_wrong_flags_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(L.ALockError):
                L.ref(Path(tmp), "missing.json")
            with self.assertRaises(L.ALockError):
                L.bundle_record(Path(tmp), "LR")
            with self.assertRaises(L.ALockError):
                L.need(False, "x")


if __name__ == "__main__":
    unittest.main()
