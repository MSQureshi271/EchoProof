import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.artifacts import digest  # noqa: E402
from echoproof.contracts import SplitAssignment  # noqa: E402
from echoproof.data import splits  # noqa: E402
from echoproof.evaluation import r5_loaders as R  # noqa: E402

_spec = importlib.util.spec_from_file_location("r5_n1_sealed", ROOT / "scripts/r5_n1_sealed.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["r5_n1_sealed"] = S
_spec.loader.exec_module(S)

HAVE_LOCKS = (ROOT / R.B_LOCK).is_file() and (ROOT / R.A_LOCK).is_file()


def make_rows(protocol, n_test_stems=3):
    """Synthetic split rows (fake ids): train and validation rows plus a few test stems of one genuine and two TTS components."""
    raw = []
    for stem, partition, role in (("100-1-0001", "train", None), ("200-1-0001", "validation", "operating")) + tuple((f"30{i}-1-0001", "test", None) for i in range(n_test_stems)):
        for component in ("genuine", "melo", "chatterbox"):
            raw.append((f"phonemedf:{component}:{stem}", f"speaker:{stem.split('-')[0]}", partition, role, "genuine" if component == "genuine" else "known_family"))
    raw.sort()
    h = digest([[a, b, c, d, e] for a, b, c, d, e in raw])
    return [SplitAssignment(1, f"protocol:{protocol}", a, b, c, d, e, "test-v1", h) for a, b, c, d, e in raw], h


def fake_plan(protocols=("A", "B"), keys=("A:LR", "B:LR"), split_hash=None):
    entries, splits_info = [], {}
    for p in protocols:
        _, h = make_rows(p)
        splits_info[p] = {"file": f"fake_{p}.parquet", "file_sha256": "f" * 64, "expected_split_hash": split_hash or h, "locked_file_sha256": None}
    for key in keys:
        spec = R.REGISTRY[key]
        rule = {"schema_version": 1, "kind": "threshold", "threshold": 0.5}
        entries.append({"key": key, "protocol": spec.protocol, "model": spec.model, "model_version": spec.model_version, "loader": spec.loader, "bundle_path": "b",
                        "bundle_id": f"id-{key}", "bundle_json_sha256": "a" * 64, "decision_rule": rule, "device": "cpu", "precision": None, "requires_cuda": False,
                        "allow_pending_review": False, "via_shared_router": False, "split_file": f"fake_{spec.protocol}.parquet", "partition": "test",
                        "_spec": spec, "_record": {}})
    return {"verified": True, "ready": True, "cuda_available": False, "checks": [], "entries": entries, "splits": splits_info,
            "audio": {"shard_dir": "shards", "index": "shards/index.parquet", "index_sha256": None, "shards_present": [], "cohort_manifest": "m", "cohort_manifest_sha256": None},
            "code": {}, "provenance_diff_vs_locks": {}}


def good_result(rid, loaded=None):
    return SimpleNamespace(status="success", score=0.9 if "genuine" not in rid else 0.1, assessment="synthetic" if "genuine" not in rid else "genuine",
                           score_type="model_score", model_version="v", decoded_duration_seconds=5.0, failure_reason=None)


class Harness:
    """Fake loader, audio and scorer. Audio 'bytes' are the recording id, so the fake predict can tell which recording it is scoring."""

    def __init__(self, predict=None, audio=None, tmp=None):
        self.log_when_rows_opened = []
        self.sealed_calls = []
        self.tmp = Path(tmp)
        self.predict = predict or (lambda audio_bytes, loaded: good_result(audio_bytes.decode()))
        self.audio = audio or (lambda rid: rid.encode())
        self.out = self.tmp / "run"

    def load_assignments(self, path, *, sealed=False):
        self.sealed_calls.append((str(path), sealed))
        log = self.out / "test_access_log.jsonl"
        self.log_when_rows_opened.append(log.read_text(encoding="utf-8") if log.is_file() else None)
        protocol = "A" if "fake_A" in str(path) else "B"
        rows, _ = make_rows(protocol)
        return rows if sealed else [r for r in rows if r.partition != "test"]

    def run(self, plan=None, confirm=True, hook=None):
        with patch.object(splits, "load_assignments", side_effect=self.load_assignments):
            return S.execute(self.tmp, self.out, confirm=confirm, plan=plan or fake_plan(), load=lambda e: object(), read_audio=self.audio, predict=self.predict,
                             metrics_hook=hook, progress_every=0)


class CheckIntegrityTests(unittest.TestCase):
    ids = ["r1", "r2", "r3"]

    def rec(self, rid, score=0.8, status="success", assessment="synthetic", reason=None):
        return {"recording_id": rid, "status": status, "score": score, "assessment": assessment, "failure_reason": reason}

    def test_clean(self):
        report = S.check_integrity(self.ids, [self.rec(i) for i in self.ids], threshold=0.5)
        self.assertTrue(report["passed"])
        self.assertEqual(report["scored_ok"], 3)

    def test_missing_recording(self):
        report = S.check_integrity(self.ids, [self.rec("r1"), self.rec("r2")])
        self.assertFalse(report["passed"])
        self.assertEqual(report["problems"]["missing"], ["r3"])
        self.assertTrue(report["problems"]["count_mismatch"])

    def test_duplicate_recording(self):
        report = S.check_integrity(self.ids, [self.rec("r1"), self.rec("r1"), self.rec("r2"), self.rec("r3")])
        self.assertFalse(report["passed"])
        self.assertEqual(report["problems"]["duplicates"], ["r1"])

    def test_unexpected_recording(self):
        report = S.check_integrity(self.ids, [self.rec(i) for i in self.ids] + [self.rec("zzz")])
        self.assertEqual(report["problems"]["unexpected"], ["zzz"])
        self.assertFalse(report["passed"])

    def test_non_finite_and_out_of_range_scores(self):
        for bad in (float("nan"), float("inf"), -0.1, 1.5, None):
            report = S.check_integrity(self.ids, [self.rec("r1", score=bad), self.rec("r2"), self.rec("r3")])
            self.assertEqual(report["problems"]["nonfinite_scores"], ["r1"], bad)
            self.assertFalse(report["passed"])

    def test_failed_recording_is_counted_not_dropped(self):
        report = S.check_integrity(self.ids, [self.rec("r1"), self.rec("r2", score=None, status="failed", assessment=None, reason="undecodable_audio"), self.rec("r3")])
        self.assertEqual(report["problems"]["failures"], [["r2", "undecodable_audio"]])
        self.assertFalse(report["passed"])

    def test_decision_must_follow_the_locked_threshold(self):
        report = S.check_integrity(self.ids, [self.rec("r1", score=0.4, assessment="synthetic"), self.rec("r2"), self.rec("r3")], threshold=0.5)
        self.assertEqual(report["problems"]["decision_mismatch"], ["r1"])
        self.assertFalse(report["passed"])


class RefusalTests(unittest.TestCase):
    def test_refuses_without_the_confirm_flag_and_touches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp=tmp)
            with self.assertRaises(S.SealedRunError):
                h.run(confirm=False)
            self.assertFalse(h.out.exists())
            self.assertEqual(h.sealed_calls, [])

    def test_refuses_a_flag_that_is_not_exactly_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp=tmp)
            with self.assertRaises(S.SealedRunError):
                h.run(confirm="yes")
            self.assertEqual(h.sealed_calls, [])

    def test_refuses_an_existing_run_folder_and_leaves_it_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp=tmp)
            h.out.mkdir()
            (h.out / "keep.txt").write_text("x")
            with self.assertRaises(S.SealedRunError):
                h.run()
            self.assertEqual([p.name for p in h.out.iterdir()], ["keep.txt"])
            self.assertEqual(h.sealed_calls, [])

    def test_refuses_an_unverified_or_unready_plan(self):
        for flag in ("verified", "ready"):
            with tempfile.TemporaryDirectory() as tmp:
                h = Harness(tmp=tmp)
                plan = fake_plan()
                plan[flag] = False
                with self.assertRaises(S.SealedRunError):
                    h.run(plan=plan)
                self.assertFalse(h.out.exists())
                self.assertEqual(h.sealed_calls, [])

    def test_ast_without_cuda_is_refused_before_the_log_or_any_test_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp=tmp)
            plan = fake_plan(keys=("A:AST",))
            plan["entries"][0]["device"] = "cuda"
            plan["cuda_available"] = False
            with self.assertRaises(R.DevicePolicyError):
                h.run(plan=plan)
            self.assertFalse(h.out.exists())
            self.assertEqual(h.sealed_calls, [])

    def test_the_command_line_refuses_without_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(S, "build_plan") as build, patch.object(splits, "load_assignments") as load:
            code = S.main(["--root", tmp, "run", "--out", "out"])
            self.assertEqual(code, 2)
            build.assert_not_called()
            load.assert_not_called()
            self.assertFalse((Path(tmp) / "out").exists())


class RunTests(unittest.TestCase):
    def test_access_log_is_written_before_the_test_rows_are_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp=tmp)
            summary = h.run()
            self.assertEqual(sorted(sealed for _, sealed in h.sealed_calls), [True, True])
            self.assertEqual(len(h.log_when_rows_opened), 2)
            for text in h.log_when_rows_opened:
                self.assertIsNotNone(text, "the access log did not exist when the test rows were opened")
                events = [json.loads(line)["event"] for line in text.splitlines()]
                self.assertEqual(events[0], "sealed_run_start")
                self.assertIn("opening_test_rows", events)
            events = [json.loads(line)["event"] for line in (h.out / "test_access_log.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1], "sealed_run_end")
            self.assertTrue(summary["passed"])

    def test_a_clean_run_scores_every_test_recording_with_every_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(tmp=tmp)
            hook = MagicMock()
            summary = h.run(hook=hook)
            self.assertEqual(summary["status"], "complete")
            for key in ("A:LR", "B:LR"):
                b = summary["bundles"][key]
                self.assertEqual(b["integrity"]["expected"], 9)         # 3 test stems x 3 components; train and validation rows are never scored
                self.assertTrue(b["integrity"]["passed"])
                import pyarrow.parquet as pq
                rows = pq.read_table(str(h.out / b["predictions_file"])).to_pylist()
                self.assertEqual(len(rows), 9)
                self.assertTrue(all("-1-0001" in r["recording_id"] and r["recording_id"].split(":")[2].startswith("30") for r in rows))
            self.assertTrue((h.out / "integrity_report.json").is_file())
            hook.assert_called_once()

    def test_a_non_finite_score_fails_the_integrity_and_no_metrics_hook_runs(self):
        def predict(audio_bytes, loaded):
            rid = audio_bytes.decode()
            result = good_result(rid)
            if rid.endswith("genuine:301-1-0001") and "phonemedf" in rid:
                result.score = float("nan")
            return result
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(predict=predict, tmp=tmp)
            hook = MagicMock()
            summary = h.run(hook=hook)
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["status"], "integrity_failed")
            self.assertEqual(summary["bundles"]["A:LR"]["integrity"]["problems"]["nonfinite_scores"], ["phonemedf:genuine:301-1-0001"])
            hook.assert_not_called()
            saved = json.loads((h.out / "integrity_report.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["passed"])

    def test_a_recording_that_cannot_be_read_is_a_counted_failure(self):
        def audio(rid):
            if rid == "phonemedf:melo:302-1-0001":
                raise OSError("read error")
            return rid.encode()
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(audio=audio, tmp=tmp)
            summary = h.run()
            self.assertFalse(summary["passed"])
            failures = summary["bundles"]["B:LR"]["integrity"]["problems"]["failures"]
            self.assertEqual([f[0] for f in failures], ["phonemedf:melo:302-1-0001"])

    def test_a_failed_inference_result_is_a_counted_failure(self):
        def predict(audio_bytes, loaded):
            rid = audio_bytes.decode()
            if rid == "phonemedf:chatterbox:300-1-0001":
                return SimpleNamespace(status="failed", score=None, assessment=None, score_type=None, model_version=None, decoded_duration_seconds=None, failure_reason="scoring_failed")
            return good_result(rid)
        with tempfile.TemporaryDirectory() as tmp:
            summary = Harness(predict=predict, tmp=tmp).run()
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["bundles"]["A:LR"]["integrity"]["problems"]["failures"], [["phonemedf:chatterbox:300-1-0001", "scoring_failed"]])

    def test_a_split_that_does_not_match_the_locked_hash_stops_the_run_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Harness(predict=MagicMock(), tmp=tmp)
            with self.assertRaises(S.SealedRunError):
                h.run(plan=fake_plan(split_hash="0" * 64))
            h.predict.assert_not_called()
            self.assertTrue((h.out / "test_access_log.jsonl").is_file())     # the access was logged first, even though the run then stopped
            self.assertEqual(json.loads((h.out / "integrity_report.json").read_text(encoding="utf-8"))["status"], "aborted")
            self.assertEqual(list(h.out.glob("predictions_*")), [])

    def test_duplicate_test_ids_in_the_split_stop_the_run(self):
        rows, h = make_rows("A")
        dup = rows + [r for r in rows if r.partition == "test"][:1]
        with tempfile.TemporaryDirectory() as tmp:
            harness = Harness(predict=MagicMock(), tmp=tmp)
            plan = fake_plan(protocols=("A",), keys=("A:LR",))
            plan["splits"]["A"]["expected_split_hash"] = digest([[r.recording_id, r.group_id, r.partition, r.validation_role, r.family_policy] for r in sorted(dup, key=lambda r: r.recording_id)])
            with patch.object(splits, "load_assignments", return_value=dup), self.assertRaises(S.SealedRunError):
                S.execute(tmp, harness.out, confirm=True, plan=plan, load=lambda e: object(), read_audio=harness.audio, predict=harness.predict, progress_every=0)
            harness.predict.assert_not_called()


@unittest.skipUnless(HAVE_LOCKS, "lock files are not present")
class PlanTests(unittest.TestCase):
    def test_plan_reads_no_test_rows_and_writes_nothing(self):
        with patch.object(splits, "load_assignments") as load, patch.object(S.splits, "load_assignments") as load2:
            before = sorted(p.as_posix() for p in (ROOT / "reports").rglob("*")) if (ROOT / "reports").is_dir() else []
            plan = S.build_plan(ROOT, deep=False)
            text = S.format_plan(plan)
            after = sorted(p.as_posix() for p in (ROOT / "reports").rglob("*")) if (ROOT / "reports").is_dir() else []
        load.assert_not_called()
        load2.assert_not_called()
        self.assertEqual(before, after)
        self.assertFalse((ROOT / S.DEFAULT_OUT).exists())
        self.assertEqual(len(plan["entries"]), 8)
        self.assertIn("no test row was read", text)
        for key in R.REGISTRY:
            self.assertIn(key, text)

    def test_plan_does_not_open_test_rows_via_the_command_line(self):
        with patch.object(splits, "load_assignments") as load:
            S.main(["--root", str(ROOT), "plan", "--quick"])
        load.assert_not_called()

    def test_plan_fails_on_a_changed_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "locks").mkdir()
            for name in (R.B_LOCK, R.A_LOCK):
                (root / name).write_bytes((ROOT / name).read_bytes() + b" ")
            plan = S.build_plan(root, deep=False, cuda=False)
            self.assertFalse(plan["verified"])
            self.assertFalse(plan["ready"])
            self.assertEqual(plan["entries"], [])


class PlanSourceTests(unittest.TestCase):
    def test_the_plan_code_path_never_asks_for_the_sealed_rows(self):
        source = Path(S.__file__).read_text(encoding="utf-8")
        plan_part = source[source.index("def build_plan"):source.index("def public")]
        self.assertNotIn("sealed=True", plan_part)
        self.assertNotIn("load_assignments", plan_part)
        self.assertEqual(source.count("splits.load_assignments(root"), 1)   # the single call, inside `execute`


if __name__ == "__main__":
    unittest.main()
