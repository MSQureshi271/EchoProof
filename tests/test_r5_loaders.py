import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.bundle import BundleError  # noqa: E402
from echoproof.contracts import ContractError  # noqa: E402
from echoproof.evaluation import r5_loaders as R  # noqa: E402

HAVE_LOCKS = (ROOT / R.B_LOCK).is_file() and (ROOT / R.A_LOCK).is_file()
EXPECTED_KEYS = {"B:LR", "B:BiGRU", "B:WavLM", "B:AST", "A:WavLM", "A:AST", "A:LR", "A:BiGRU"}


def records():
    return {"B": R.read_lock(ROOT, "B"), "A": R.read_lock(ROOT, "A")}


def have_bundles():
    if not HAVE_LOCKS:
        return False
    return all((ROOT / c["bundle_path"]).is_dir() for rec in records().values() for c in rec["candidates"].values())


class RegistryTests(unittest.TestCase):
    def test_registry_is_exactly_the_eight_bundles(self):
        self.assertEqual(set(R.REGISTRY), EXPECTED_KEYS)
        self.assertEqual(len(R.REGISTRY), 8)

    def test_registry_policy_flags(self):
        self.assertTrue(R.REGISTRY["B:AST"].requires_cuda and R.REGISTRY["A:AST"].requires_cuda)
        self.assertEqual(R.REGISTRY["B:AST"].precision, "fp16")
        self.assertEqual([k for k, s in R.REGISTRY.items() if s.requires_cuda], ["B:AST", "A:AST"])
        self.assertEqual([k for k, s in R.REGISTRY.items() if s.allow_pending_review], ["B:BiGRU"])
        self.assertEqual({k for k, s in R.REGISTRY.items() if s.via_router}, {"B:LR", "B:WavLM"})

    @unittest.skipUnless(HAVE_LOCKS, "lock files are not present")
    def test_registry_covers_every_lock_candidate_and_nothing_else(self):
        pairs = R.candidate_bundles(records())
        self.assertEqual({s.key for s, _ in pairs}, EXPECTED_KEYS)

    @unittest.skipUnless(HAVE_LOCKS, "lock files are not present")
    def test_an_extra_or_missing_lock_candidate_is_refused(self):
        extra = records()
        extra["A"]["candidates"]["Extra"] = extra["A"]["candidates"]["LR"]
        with self.assertRaises(R.LockError):
            R.candidate_bundles(extra)
        missing = records()
        del missing["B"]["candidates"]["AST"]
        with self.assertRaises(R.LockError):
            R.candidate_bundles(missing)
        wrong = records()
        wrong["A"]["candidates"]["LR"]["model_version"] = "logistic_regression:B"
        with self.assertRaises(R.LockError):
            R.candidate_bundles(wrong)


class DevicePolicyTests(unittest.TestCase):
    def test_ast_is_cuda_only_and_never_falls_back(self):
        for key in ("B:AST", "A:AST"):
            spec = R.REGISTRY[key]
            with self.assertRaises(R.DevicePolicyError):
                R.choose_device(spec, None, cuda=False)
            with self.assertRaises(R.DevicePolicyError):
                R.choose_device(spec, "cpu", cuda=True)
            self.assertEqual(R.choose_device(spec, None, cuda=True), "cuda")

    def test_other_bundles(self):
        self.assertEqual(R.choose_device(R.REGISTRY["A:LR"], "cuda", cuda=True), "cpu")
        self.assertEqual(R.choose_device(R.REGISTRY["A:WavLM"], None, cuda=False), "cpu")
        self.assertEqual(R.choose_device(R.REGISTRY["A:WavLM"], None, cuda=True), "cuda")
        self.assertEqual(R.choose_device(R.REGISTRY["B:BiGRU"], "cpu", cuda=True), "cpu")
        with self.assertRaises(R.DevicePolicyError):
            R.choose_device(R.REGISTRY["B:WavLM"], "cuda", cuda=False)


class ThresholdOverrideTests(unittest.TestCase):
    def test_the_registry_refuses_an_override_for_every_bundle(self):
        for key, spec in R.REGISTRY.items():
            with self.subTest(key=key), self.assertRaises(ContractError):
                R.load_entry(spec, "does/not/matter", device="cpu", threshold=0.5)
            with self.subTest(key=key, via="load_verified"), self.assertRaises(ContractError):
                R.load_verified(ROOT, spec, {}, threshold=0.5)

    def test_the_raw_and_ast_a_loaders_refuse_an_override_themselves(self):
        from echoproof.models.r4_n1_ast_bundle import load_ast_a_bundle
        from echoproof.models.r4_n1_raw_bundles import load_bigru_bundle_a, load_lr_bundle_a
        with self.assertRaises(ContractError):
            load_lr_bundle_a("missing", threshold=0.4)
        with self.assertRaises(ContractError):
            load_bigru_bundle_a("missing", threshold=0.4)
        with self.assertRaises(ValueError):
            load_ast_a_bundle("missing", device="cuda", precision="fp16", threshold=0.4)

    @unittest.skipUnless(have_bundles(), "locked bundles are not present")
    def test_the_a_wavlm_loader_refuses_an_override_itself(self):
        from echoproof.models.r4_n1_wavlm_bundle import load_wavlm_bundle_a
        path = ROOT / R.read_lock(ROOT, "A")["candidates"]["WavLM"]["bundle_path"]
        with self.assertRaises(ContractError):
            load_wavlm_bundle_a(path, scorer_factory=lambda p: (lambda chunks: [0.5] * len(chunks)), threshold=0.3)


class LoaderCallTests(unittest.TestCase):
    def test_b_bigru_is_loaded_with_the_explicit_pending_review_allowance(self):
        with patch("echoproof.models.r3_n1_bigru_bundle.load_calibrated_bigru_bundle") as mocked:
            R.load_entry(R.REGISTRY["B:BiGRU"], "p", device="cpu")
        mocked.assert_called_once()
        self.assertIs(mocked.call_args.kwargs["allow_pending_review"], True)
        self.assertTrue(R.ALLOW_PENDING_REVIEW_BIGRU_B)

    def test_ast_loaders_are_called_with_cuda_fp16(self):
        with patch("echoproof.models.r3_n1_ast_platt_bundle.load_ast_platt_candidate") as b, patch("echoproof.models.r4_n1_ast_bundle.load_ast_a_bundle") as a:
            R.load_entry(R.REGISTRY["B:AST"], "p", device="cuda")
            R.load_entry(R.REGISTRY["A:AST"], "p", device="cuda")
        self.assertEqual((b.call_args.kwargs["device"], b.call_args.kwargs["precision"]), ("cuda", "fp16"))
        self.assertEqual((a.call_args.kwargs["device"], a.call_args.kwargs["precision"]), ("cuda", "fp16"))

    @unittest.skipUnless(have_bundles(), "locked bundles are not present")
    def test_real_bigru_b_loader_refuses_without_the_allowance(self):
        from echoproof.models.r3_n1_bigru_bundle import load_calibrated_bigru_bundle
        path = ROOT / R.read_lock(ROOT, "B")["candidates"]["BiGRU"]["bundle_path"]
        with self.assertRaises(BundleError):
            load_calibrated_bigru_bundle(path)


class CheckLoadedTests(unittest.TestCase):
    def loaded(self, **kw):
        from echoproof.contracts import BundleSpecification, DecisionRule
        rule = DecisionRule(1, "threshold", 0.5)
        spec = BundleSpecification(1, "bundle:x", "bigru:A", "0" * 64, rule, "genuine", "tts")
        return SimpleNamespace(spec=spec, is_mock=kw.get("is_mock", False))

    def test_accepts_the_locked_bundle_and_refuses_changes(self):
        s = R.REGISTRY["A:BiGRU"]
        record = {"bundle_id": "bundle:x", "decision_rule": {"schema_version": 1, "kind": "threshold", "threshold": 0.5}}
        R.check_loaded(s, record, self.loaded())
        with self.assertRaises(R.LockError):
            R.check_loaded(s, record, self.loaded(is_mock=True))
        with self.assertRaises(R.LockError):
            R.check_loaded(s, {**record, "bundle_id": "bundle:y"}, self.loaded())
        with self.assertRaises(R.LockError):
            R.check_loaded(s, {**record, "decision_rule": {**record["decision_rule"], "threshold": 0.6}}, self.loaded())


@unittest.skipUnless(HAVE_LOCKS, "lock files are not present")
class LockVerificationTests(unittest.TestCase):
    def test_the_real_locks_verify(self):
        R.verify_b_lock(ROOT)
        R.verify_a_lock(ROOT, deep=False)

    def test_lf_and_crlf_forms_are_accepted_and_changes_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            crlf = b'{"a": 1,\r\n "b": 2}\r\n'
            import hashlib
            digest = hashlib.sha256(crlf).hexdigest()
            for data, ok in ((crlf, True), (crlf.replace(b"\r\n", b"\n"), True), (crlf + b" ", False)):
                p.write_bytes(data)
                self.assertEqual(R.file_matches(p, digest), ok)

    def _copy_lock(self, tmp, protocol, mutate):
        root = Path(tmp)
        (root / "locks").mkdir()
        for name in (R.B_LOCK, R.A_LOCK):
            shutil.copyfile(ROOT / name, root / name)
        path = root / R.LOCKS[protocol]
        path.write_bytes(mutate(path.read_bytes()))
        return root

    def test_a_changed_lock_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_lock(tmp, "B", lambda b: b.replace(b"lock:", b"lock;", 1) if b"lock:" in b else b + b" ")
            with self.assertRaises(R.LockError):
                R.verify_b_lock(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = self._copy_lock(tmp, "A", lambda b: b + b" ")
            with self.assertRaises(R.LockError):
                R.verify_a_lock(root, deep=False)

    def test_a_missing_lock_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(R.LockError):
            R.verify_b_lock(tmp)


@unittest.skipUnless(have_bundles(), "locked bundles are not present")
class BundleVerificationTests(unittest.TestCase):
    def test_all_eight_bundles_verify_against_the_locks(self):
        verified = R.verify_all_bundles(ROOT, records())
        self.assertEqual(set(verified), EXPECTED_KEYS)

    def test_a_changed_bundle_hash_is_refused(self):
        spec, record = [(s, c) for s, c in R.candidate_bundles(records()) if s.key == "B:LR"][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / record["bundle_path"]
            shutil.copytree(ROOT / record["bundle_path"], dest)
            R.verify_bundle(root, spec, record)                      # the copy verifies
            bad = copy.deepcopy(record)
            bad["bundle_json_sha256"] = "0" * 64
            with self.assertRaises(R.LockError):
                R.verify_bundle(root, spec, bad)
            bad = copy.deepcopy(record)
            bad["decision_rule"]["threshold"] = 0.123
            with self.assertRaises(R.LockError):
                R.verify_bundle(root, spec, bad)
            bad = copy.deepcopy(record)
            bad["components"]["model/weights.npz"]["sha256"] = "0" * 64
            with self.assertRaises(R.LockError):
                R.verify_bundle(root, spec, bad)
            comp = dest / "model/weights.npz"
            comp.write_bytes(comp.read_bytes() + b"x")             # a changed component byte
            with self.assertRaises((R.LockError, BundleError)):
                R.verify_bundle(root, spec, record)

    def test_router_routes_agree_with_the_registry(self):
        from echoproof.inference import load_inference_bundle
        for key in ("B:LR", "B:WavLM"):
            spec, record = [(s, c) for s, c in R.candidate_bundles(records()) if s.key == key][0]
            path = ROOT / record["bundle_path"]
            with patch("echoproof.models.wavlm_adapter.WavLMBundleAdapter.load",
                       return_value=SimpleNamespace(score_chunks=lambda chunks: [0.5] * len(chunks))):
                via_router = load_inference_bundle(path)
                via_registry = R.load_entry(spec, path, device="cpu")
            self.assertEqual(via_router.spec, via_registry.spec)
            self.assertEqual(via_router.spec.decision_rule, via_registry.spec.decision_rule)
            self.assertEqual(via_router.score_type, via_registry.score_type)
            self.assertEqual(via_router.aggregate, via_registry.aggregate)
            self.assertEqual(via_router.calibrate is None, via_registry.calibrate is None)
            if via_router.calibrate is not None:
                for s in (0.05, 0.3, 0.5, 0.77, 0.99):
                    self.assertEqual(via_router.calibrate(s), via_registry.calibrate(s))
            R.check_loaded(spec, record, via_registry)


if __name__ == "__main__":
    unittest.main()
