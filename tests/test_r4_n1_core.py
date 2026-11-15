import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.evaluation import r4_n1


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _synthetic_inputs(root: Path):
    split_hash = "a" * 64
    manifest_hash = "b" * 64
    export_dir = root / "runs/r4/lr_export"
    train_dir = root / "runs/r4/lr"
    export_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)
    (train_dir / "model").mkdir()
    model_path = train_dir / "model/weights.npz"
    model_path.write_bytes(b"synthetic weights")
    metadata_path = train_dir / "model/lr_model.json"
    metadata_path.write_text(json.dumps({"protocol": "A", "split_sha256": split_hash}), encoding="utf-8")
    train_ev_path = train_dir / "evidence.json"
    train_ev_path.write_text(json.dumps({"status": "complete", "protocol": "A", "split_sha256": split_hash,
                                         "outputs": {"model/weights.npz": _sha(model_path)}}), encoding="utf-8")
    train_ev_hash = _sha(train_ev_path)

    roles = {
        "selection": {"recordings": 5, "speakers": 1, "chunks": 5, "per_component": 1},
        "calibration": {"recordings": 15, "speakers": 3, "chunks": 15, "per_component": 3},
        "operating": {"recordings": 10, "speakers": 2, "chunks": 10, "per_component": 2},
    }
    output_hashes = {}
    for role in roles:
        rec_rows, chunk_rows = [], []
        n_speakers = roles[role]["speakers"]
        for speaker_i in range(n_speakers):
            group = f"speaker:{role}_{speaker_i}"
            for i, component in enumerate(r4_n1.COMPONENTS):
                label = 0 if component == "genuine" else 1
                score = min(0.95, max(0.05, (0.22 + 0.04 * speaker_i) if label == 0 else (0.72 - 0.03 * speaker_i)))
                rec_id, chunk_id = f"rec:{role}:{speaker_i}:{component}", f"chunk:{role}:{speaker_i}:{component}"
                rec_rows.append({"recording_id": rec_id, "group_id": group, "component": component, "role": role,
                                 "label": label, "raw_score": score, "chunk_count": 1})
                chunk_rows.append({"chunk_id": chunk_id, "recording_id": rec_id, "group_id": group, "component": component,
                                   "role": role, "label": label, "raw_score": score})
        for kind, rows, columns in (
            ("recordings", rec_rows, r4_n1.RECORDING_COLUMNS),
            ("chunks", chunk_rows, r4_n1.CHUNK_COLUMNS),
        ):
            name = f"predictions_{kind}_{role}.parquet"
            path = export_dir / name
            pq.write_table(pa.Table.from_pylist(rows), path)
            output_hashes[name] = _sha(path)

    export_ev = {"status": "complete", "protocol": "A", "model": "lr", "split_sha256": split_hash,
                 "training_evidence_sha256": train_ev_hash, "selected_model_sha256": _sha(model_path),
                 "input_sha256": {"manifest": manifest_hash}, "outputs": output_hashes}
    export_ev_path = export_dir / "evidence.json"
    export_ev_path.write_text(json.dumps(export_ev, sort_keys=True), encoding="utf-8")
    selected_sha = _sha(model_path)
    recipe = json.loads((ROOT / "configs/r4_n1_calibration_a.json").read_text(encoding="utf-8"))
    recipe["protocol_contract"]["split_sha256"] = split_hash
    recipe["protocol_contract"]["r4_s1_manifest"]["sha256"] = manifest_hash
    recipe["roles"] = roles
    recipe["models"]["lr"].update({
        "selected_model_path": model_path.relative_to(root).as_posix(), "selected_model_sha256": selected_sha,
        "model_metadata_path": metadata_path.relative_to(root).as_posix(),
        "training_evidence_path": train_ev_path.relative_to(root).as_posix(), "training_evidence_sha256": train_ev_hash,
        "export_directory": export_dir.relative_to(root).as_posix(), "export_evidence_path": export_ev_path.relative_to(root).as_posix(),
        "export_evidence_sha256": _sha(export_ev_path), "prediction_sha256": output_hashes,
    })
    return recipe


def _synthetic_bigru_inputs(root: Path):
    recipe = _synthetic_inputs(root)
    lr = recipe["models"]["lr"]
    source_export = root / lr["export_directory"]
    export_dir = root / "runs/r4/bigru_export"
    export_dir.mkdir(parents=True)
    for source in source_export.glob("*.parquet"):
        (export_dir / source.name).write_bytes(source.read_bytes())
    model_dir = root / "runs/r4/bigru"
    model_dir.mkdir()
    checkpoint = model_dir / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    train_ev = {"status": "training_complete", "protocol": "A", "split_sha256": recipe["protocol_contract"]["split_sha256"],
                "best_checkpoint_sha256": _sha(checkpoint)}
    train_path = model_dir / "training_evidence.json"
    train_path.write_text(json.dumps(train_ev), encoding="utf-8")
    export_ev_path = export_dir / "evidence.json"
    export_ev = json.loads((source_export / "evidence.json").read_text(encoding="utf-8"))
    export_ev.update({"model": "bigru", "training_evidence_sha256": _sha(train_path), "selected_model_sha256": _sha(checkpoint)})
    export_ev_path.write_text(json.dumps(export_ev, sort_keys=True), encoding="utf-8")
    recipe["models"]["bigru"] = {
        "verification_state": "verified_local", "model": "bigru", "protocol": "A",
        "split_sha256": recipe["protocol_contract"]["split_sha256"],
        "selected_model_path": checkpoint.relative_to(root).as_posix(), "selected_model_sha256": _sha(checkpoint),
        "training_evidence_path": train_path.relative_to(root).as_posix(), "training_evidence_sha256": _sha(train_path),
        "export_directory": export_dir.relative_to(root).as_posix(), "export_evidence_path": export_ev_path.relative_to(root).as_posix(),
        "export_evidence_sha256": _sha(export_ev_path), "prediction_sha256": lr["prediction_sha256"],
    }
    return recipe


class R4N1CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_recipe_is_a_and_model_specific_run_directories_resolve(self):
        recipe = r4_n1.load_recipe(ROOT / "configs/r4_n1_calibration_a.json")
        self.assertEqual(recipe["protocol"], "A")
        self.assertEqual(recipe["output_root"], "runs/r4/r4_n1_lr")
        self.assertEqual(r4_n1.resolve_run_dir(ROOT, recipe), (ROOT / "runs/r4/r4_n1_lr").resolve())
        self.assertEqual(r4_n1.resolve_run_dir(ROOT, recipe, model_name="bigru"), (ROOT / "runs/r4/r4_n1_bigru").resolve())
        with self.assertRaisesRegex(r4_n1.R4N1Error, "unsupported A model"):
            r4_n1.audit_inputs(Path("."), recipe, "ast")


    def test_audit_checks_synthetic_selection_and_calibration_without_opening_operating(self):
        recipe = _synthetic_inputs(self.root)
        calls = []
        original = r4_n1._parquet_rows

        def tracked(path, columns):
            calls.append(Path(path).name)
            self.assertNotIn("operating", Path(path).name)
            return original(path, columns)

        with patch.object(r4_n1, "_parquet_rows", tracked):
            report = r4_n1.audit_inputs(self.root, recipe)
        self.assertTrue(report["passed"])
        self.assertFalse(report["operating_predictions_content_read"])
        self.assertFalse(report["operating_parquet_sha256_verified"])
        self.assertEqual(len(calls), 4)
        self.assertEqual(set(calls), {
            "predictions_recordings_selection.parquet", "predictions_chunks_selection.parquet",
            "predictions_recordings_calibration.parquet", "predictions_chunks_calibration.parquet",
        })


    def test_one_standard_error_selects_simplest_eligible_and_reports_failures(self):
        chosen = r4_n1.select_mapping(
            {"identity": [0.30, 0.31, 0.29], "temperature": [0.20, 0.35, 0.25], "affine": [0.19, 0.21, 0.20]},
            {"temperature": "failed fold"},
        )
        self.assertEqual(chosen["best"], "affine")
        self.assertEqual(chosen["selected"], "affine")
        self.assertEqual(chosen["excluded"], {"temperature": "failed fold"})


    def test_threshold_uses_smaller_threshold_for_final_tie(self):
        fake_metrics = lambda y, d: {"macro_f1": 0.8, "genuine_fpr": 0.0, "tp": 1, "fp": 0, "tn": 1, "fn": 0}
        with patch.object(r4_n1, "_decision_metrics", fake_metrics):
            rule, metrics = r4_n1.select_threshold_a([0, 1], [0.4, 0.6])
        self.assertEqual(rule.kind, "threshold")
        self.assertEqual(rule.threshold, 0.4)
        self.assertEqual(metrics["plateau"]["thresholds_tied_at_max_macro_f1"], 2)


    def test_threshold_has_explicit_all_negative_rule(self):
        rule, metrics = r4_n1.select_threshold_a([0, 1], [1.0, 1.0])
        self.assertEqual(rule.kind, "all_negative")
        self.assertIsNone(rule.threshold)
        self.assertEqual(metrics["genuine_fpr"], 0.0)


    def test_operating_stage_refuses_to_open_before_fit_is_frozen(self):
        recipe = json.loads((ROOT / "configs/r4_n1_calibration_a.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(r4_n1.R4N1Error, "fit stage is not complete"):
            r4_n1.operating_stage(self.root, recipe, self.root / "new_run")


    def test_synthetic_lr_stages_freeze_before_operating_open(self):
        recipe = _synthetic_inputs(self.root)
        recipe["bootstrap"]["resamples"] = 20
        audit = r4_n1.audit_inputs(self.root, recipe)
        self.assertTrue(audit["passed"])
        run_dir = self.root / "runs/r4/r4_n1_test_lr"
        fitted = r4_n1.fit_stage(self.root, recipe, run_dir, audit)
        self.assertFalse(fitted["operating_predictions_read"])
        self.assertTrue((run_dir / "fit/stage_complete.json").is_file())
        self.assertFalse((run_dir / "operating/opened.json").exists())
        operating = r4_n1.operating_stage(self.root, recipe, run_dir)
        self.assertFalse(operating["test_data_read"])
        self.assertTrue(operating["operating_opened_after_fit_freeze"])
        self.assertTrue((run_dir / "operating/operating_results.json").is_file())
        for view in ("raw", "mapped", "deployed"):
            self.assertIn("roc_auc", operating["probability_metrics"][view])
            self.assertIn("eer", operating["probability_metrics"][view])
            self.assertTrue(0 <= operating["probability_metrics"][view]["roc_auc"] <= 1)
            self.assertTrue(0 <= operating["probability_metrics"][view]["eer"] <= 1)
        plot = run_dir / operating["reliability_plot"]["path"]
        self.assertEqual(operating["reliability_plot"]["path"], "operating/reliability.png")
        self.assertTrue(plot.is_file())
        self.assertGreater(plot.stat().st_size, 1000)
        self.assertEqual(_sha(plot), operating["reliability_plot"]["sha256"])

    def test_synthetic_bigru_audit_and_stages_use_bigru_inputs(self):
        recipe = _synthetic_bigru_inputs(self.root)
        recipe["bootstrap"]["resamples"] = 20
        audit = r4_n1.audit_inputs(self.root, recipe, "bigru")
        self.assertTrue(audit["passed"], audit["problems"])
        self.assertFalse(audit["operating_predictions_content_read"])
        run_dir = self.root / "runs/r4/r4_n1_bigru"
        fitted = r4_n1.fit_stage(self.root, recipe, run_dir, audit, "bigru")
        self.assertEqual(fitted["model"], "bigru")
        self.assertTrue((run_dir / "fit/stage_complete.json").is_file())
        self.assertFalse((run_dir / "operating/opened.json").exists())
        operating = r4_n1.operating_stage(self.root, recipe, run_dir, "bigru")
        self.assertEqual(operating["model"], "bigru")
        self.assertTrue(operating["operating_opened_after_fit_freeze"])
        self.assertTrue((run_dir / "operating/operating_results.json").is_file())


if __name__ == "__main__":
    unittest.main()

