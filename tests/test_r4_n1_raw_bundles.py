import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.artifacts import digest, file_digest  # noqa: E402
from echoproof.bundle import BundleError, load_bundle, write_bundle  # noqa: E402
from echoproof.contracts import BundleSpecification, ContractError, DecisionRule  # noqa: E402
from echoproof.data.chunks import PREPROCESSING_HASH  # noqa: E402
from echoproof.models import r4_n1_raw_bundles as rb  # noqa: E402

THRESHOLD = 0.6


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_project(root: Path, *, mapping="identity", accepted=False, test_read=False, threshold=THRESHOLD):
    """A tiny synthetic LR-A run: recipe entry, model files, fit and operating stage records, exported operating scores."""
    model_dir = root / "runs/r4/r4_m1_a_v1_lr/model"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.npz").write_bytes(b"weights")
    _write(model_dir / "lr_model.json", {"protocol": "A", "split_sha256": rb.SPLIT_SHA256, "label_map": {"genuine": 0, "tts": 1}})
    export = root / "runs/r4/r4_m1_a_v1_lr_export"
    export.mkdir(parents=True)
    rows = [{"recording_id": f"r{i}", "label": label, "raw_score": score} for i, (label, score) in
            enumerate([(0, 0.1), (0, 0.2), (0, 0.7), (1, 0.65), (1, 0.9), (1, 0.4)])]
    pq.write_table(pa.Table.from_pylist(rows), export / "predictions_recordings_operating.parquet")
    entry = {"verification_state": "verified_local", "split_sha256": rb.SPLIT_SHA256, "selected_epoch": None,
             "selected_model_path": "runs/r4/r4_m1_a_v1_lr/model/weights.npz", "selected_model_sha256": file_digest(model_dir / "weights.npz"),
             "export_directory": "runs/r4/r4_m1_a_v1_lr_export",
             "prediction_sha256": {"predictions_recordings_operating.parquet": file_digest(export / "predictions_recordings_operating.parquet")}}
    recipe = {"models": {"lr": entry}}
    run = root / "runs/r4/r4_n1_lr"
    fit_dir, op_dir = run / "fit", run / "operating"
    _write(fit_dir / "calibration.json", {"selected_mapping": mapping})
    _write(run / "input_audit.json", {"passed": True})
    # metrics for the rows above at threshold 0.6: decisions 0,0,1,1,1,0 -> tp 2, fp 1, tn 2, fn 1
    from echoproof.evaluation.r4_n1 import _decision_metrics
    saved = _decision_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0])
    results = {"model": "lr", "mapping": mapping, "mapping_accepted": accepted, "score_type": "model_score", "threshold_score_source": "raw",
               "test_data_read": test_read, "threshold_rule": {"decision": "score >= threshold", "kind": "threshold", "threshold": threshold},
               "operating_predictions_sha256": entry["prediction_sha256"]["predictions_recordings_operating.parquet"],
               "threshold_metrics": {"macro_f1": saved["macro_f1"], "genuine_fpr": saved["genuine_fpr"], "candidate_count": 7, "plateau": {"thresholds_tied_at_max_macro_f1": 1}}}
    _write(op_dir / "operating_results.json", results)
    _write(fit_dir / "stage_complete.json", {"recipe_sha256": digest(recipe), "files": {"calibration.json": file_digest(fit_dir / "calibration.json")}})
    _write(op_dir / "stage_complete.json", {"recipe_sha256": digest(recipe), "operating_results_sha256": file_digest(op_dir / "operating_results.json"), "test_data_read": False})
    (root / "configs").mkdir()
    _write(root / rb.RECIPE, recipe)
    return recipe


def patched(recipe):
    return patch("echoproof.evaluation.r4_n1.load_recipe", lambda path: recipe)


def make_bundle(path: Path, *, model="lr", threshold=THRESHOLD, rule_threshold=None, extra_stage=None):
    meta = rb.MODELS[model]
    rule = {"schema": rb.SCORE_RULE_SCHEMA, "model": model, "protocol": "A", "mapping": "identity", "score_type": "model_score",
            "aggregation": "mean_chunk_probability",
            "decision_rule": {"kind": "threshold", "threshold": threshold if rule_threshold is None else rule_threshold, "comparison": ">="}}
    comps = {n: f"data {n}".encode() for n in meta["components"] if n != rb.SCORE_RULE_COMPONENT}
    comps[rb.SCORE_RULE_COMPONENT] = (json.dumps(rule) + "\n").encode()
    spec = BundleSpecification(1, meta["bundle_id"], meta["model_version"], PREPROCESSING_HASH, DecisionRule(1, "threshold", threshold), "genuine", "tts")
    write_bundle(path, spec, comps, extra={"stage": extra_stage or meta["stage"], "protocol": "A"})
    return path


class VerifiedRunTests(unittest.TestCase):
    def check(self, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = make_project(root, **kw)
            with patched(recipe):
                return rb.verified_run(root, "lr")

    def test_accepts_a_rejected_identity_run(self):
        facts = self.check()
        self.assertEqual(facts["threshold"], THRESHOLD)
        self.assertEqual(facts["hashes"]["recipe_sha256"], digest({"models": {"lr": facts["entry"]}}))

    def test_refuses_an_accepted_or_non_identity_mapping(self):
        for kw in ({"mapping": "temperature"}, {"accepted": True}):
            with self.subTest(kw=kw), self.assertRaises(BundleError):
                self.check(**kw)

    def test_refuses_a_unit_range_violation_or_test_read(self):
        for kw in ({"threshold": 1.5}, {"threshold": 0}, {"test_read": True}):
            with self.subTest(kw=kw), self.assertRaises(BundleError):
                self.check(**kw)

    def test_refuses_changed_results_or_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = make_project(root)
            res = root / "runs/r4/r4_n1_lr/operating/operating_results.json"
            res.write_text(res.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with patched(recipe), self.assertRaises(BundleError):
                rb.verified_run(root, "lr")
            res.write_text(res.read_text(encoding="utf-8").rstrip(" "), encoding="utf-8")
            with patched({**recipe, "extra": 1}), self.assertRaises(BundleError):
                rb.verified_run(root, "lr")


class BuildTests(unittest.TestCase):
    def build(self, root, **kw):
        recipe = make_project(root, **kw)
        stub = SimpleNamespace(spec=SimpleNamespace(bundle_id=rb.MODELS["lr"]["bundle_id"]))
        with patched(recipe), patch.object(rb, "load_bundle_a", lambda *a, **k: stub):
            return rb.build_bundle(root, "lr")

    def test_builds_verifies_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = self.build(root)
            self.assertEqual(evidence["saved_score_parity"]["status"], "passed")
            bundle = load_bundle(root / rb.MODELS["lr"]["destination"])
            self.assertEqual(set(bundle.components), rb.MODELS["lr"]["components"])
            self.assertEqual(bundle.spec.decision_rule, DecisionRule(1, "threshold", THRESHOLD))
            self.assertFalse(evidence["test_partition_read"])
            self.assertTrue((root / "runs/r4/r4_n1_lr/handoff/bundle_packaging.json").is_file())
            with patched(json.loads((root / rb.RECIPE).read_text(encoding="utf-8"))), self.assertRaises(BundleError):
                rb.build_bundle(root, "lr")

    def test_parity_failure_is_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = make_project(root)
            res = root / "runs/r4/r4_n1_lr/operating/operating_results.json"
            data = json.loads(res.read_text(encoding="utf-8"))
            data["threshold_metrics"]["macro_f1"] += 0.01
            res.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            op = root / "runs/r4/r4_n1_lr/operating/stage_complete.json"
            done = json.loads(op.read_text(encoding="utf-8"))
            done["operating_results_sha256"] = file_digest(res)
            op.write_text(json.dumps(done), encoding="utf-8")
            stub = SimpleNamespace(spec=SimpleNamespace(bundle_id="x"))
            with patched(recipe), patch.object(rb, "load_bundle_a", lambda *a, **k: stub), self.assertRaises(BundleError):
                rb.build_bundle(root, "lr")


class LoaderTests(unittest.TestCase):
    def scorer(self, path):
        return lambda waves: [0.5 for _ in waves]

    def test_loads_and_refuses_overrides(self):
        for model in ("lr", "bigru"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                path = make_bundle(Path(tmp) / "b", model=model)
                loaded = rb.load_bundle_a(path, model, scorer_factory=self.scorer)
                self.assertEqual(loaded.score_type, "model_score")
                self.assertIsNone(loaded.calibrate)
                self.assertEqual(loaded.spec.decision_rule.threshold, THRESHOLD)
                with self.assertRaises(ContractError):
                    rb.load_bundle_a(path, model, scorer_factory=self.scorer, threshold=0.5)

    def test_refuses_a_wrong_model_a_wrong_stage_and_a_rule_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lr = make_bundle(tmp / "lr", model="lr")
            with self.assertRaises(BundleError):
                rb.load_bundle_a(lr, "bigru", scorer_factory=self.scorer)
            stage = make_bundle(tmp / "stage", extra_stage="something_else")
            with self.assertRaises(BundleError):
                rb.load_bundle_a(stage, "lr", scorer_factory=self.scorer)
            mismatch = make_bundle(tmp / "mismatch", rule_threshold=0.7)
            with self.assertRaises(BundleError):
                rb.load_bundle_a(mismatch, "lr", scorer_factory=self.scorer)

    def test_refuses_tampered_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_bundle(Path(tmp) / "b")
            (path / "model/weights.npz").write_bytes(b"tampered")
            with self.assertRaises(BundleError):
                rb.load_bundle_a(path, "lr", scorer_factory=self.scorer)


class RealBundleTests(unittest.TestCase):
    """Run only where the built A bundles and their runs are present (they are gitignored)."""

    def test_built_bundles_match_their_runs(self):
        for model in ("lr", "bigru"):
            path = ROOT / rb.MODELS[model]["destination"]
            if not path.is_dir():
                self.skipTest(f"{path.name} is not built here")
            with self.subTest(model=model):
                loaded = rb.load_bundle_a(path, model, scorer_factory=lambda p: (lambda waves: [0.5] * len(waves)))
                results = json.loads((ROOT / rb.MODELS[model]["run"] / "operating/operating_results.json").read_text(encoding="utf-8"))
                self.assertEqual(loaded.spec.decision_rule.threshold, results["threshold_rule"]["threshold"])
                evidence = json.loads((ROOT / rb.MODELS[model]["run"] / "handoff/bundle_packaging.json").read_text(encoding="utf-8"))
                self.assertEqual(evidence["bundle_json_sha256"], file_digest(path / "bundle.json"))
                self.assertEqual(evidence["saved_score_parity"]["status"], "passed")

class ParityRunnerTests(unittest.TestCase):
    def test_pick_is_deterministic_and_orders_by_hash(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import r4_n1_raw_parity as parity
        rows = [{"recording_id": f"r{i}"} for i in range(20)]
        self.assertEqual(parity.pick(rows, 5), parity.pick(list(reversed(rows)), 5))
        self.assertEqual(len(parity.pick(rows, None)), 20)
        self.assertEqual(set(parity.TOLERANCES), set(rb.MODELS))

    def test_notebook_runs_the_script_for_both_models_and_never_loosens_tolerance(self):
        nb = json.loads((ROOT / "notebooks/R4_N1_Colab_LR_BiGRU_Parity.ipynb").read_text(encoding="utf-8"))
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for needle in ("r4_n1_raw_parity.py", "parity('lr', 'cpu')", "'bigru'", "test_r4_n1_raw_bundles.py", "bundles/r4_n1_lr_A_raw_threshold", "bundles/r4_n1_bigru_A_raw_threshold"):
            self.assertIn(needle, code)
        self.assertNotIn("--tolerance", code)
        self.assertTrue(all(c["outputs"] == [] for c in nb["cells"] if c["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()
