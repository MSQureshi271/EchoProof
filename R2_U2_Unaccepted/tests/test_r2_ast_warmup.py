"""R2-U2 acceptance tests on generated metadata/audio and a tiny untrained AST.

These tests download no weights and use no research audio. They establish software
behavior; successful real GPU training and Muneeb's review remain separate gates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from echoproof.artifacts import digest, file_digest
from echoproof.contracts import Chunk, SplitAssignment, write_parquet
from echoproof.data.chunks import PREPROCESSING_HASH, eval_windows
from echoproof.data.cohort import flac_bytes, pcm_sha256, write_table
from echoproof.models.ast_detector import ASTDetector
from echoproof.training import r2_u2
from echoproof.training.datasets import ChunkDataset, CropDataset, DictWaveformLoader
from echoproof.training.ledger import append_ledger
from echoproof.training.trainer import TrainConfig, Trainer

try:
    from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
SPLIT_HASH = "1" * 64


def metadata_fixture(root):
    """Include all B roles and unreadable excluded members to expose scope leaks."""
    rows, assignments, chunks, pcm = [], [], [], {}

    def add(speaker, component, partition, role=None, n=8000, assigned=True):
        stem = f"{speaker}-2-0001"
        rid = f"phonemedf:{component}:{stem}"
        values = np.random.default_rng(100 + len(rows)).integers(-12000, 12000, n, dtype=np.int16).astype("<i2")
        shard = "shard_00.tar" if partition == "train" else "shard_01.tar" if role == "selection" else "shard_02.tar"
        rows.append({"recording_id": rid, "stem": stem, "speaker_id": str(speaker), "chapter_id": "2",
                     "component": component, "label": int(component != "genuine"), "generator": None if component == "genuine" else component,
                     "duration_s": n / RATE, "n_samples": n, "original_sample_rate": RATE,
                     "canonical_path": f"{shard}::{rid}", "pcm_sha256": pcm_sha256(values), "retained": True,
                     "exclusion_reason": None, "quality_flags": ""})
        pcm[rid] = values
        if assigned:
            assignments.append(SplitAssignment(1, "protocol:B", rid, f"speaker:{speaker}", partition, role,
                                               "protocol_B", "r1-v1", SPLIT_HASH))
            for start, end in eval_windows(n):
                chunks.append(Chunk(1, f"chunk:B:{rid}:{start}", rid, f"speaker:{speaker}", "protocol:B", partition,
                                    start, end, RATE, start / RATE, end / RATE, end - start, 64000, PREPROCESSING_HASH))
        return rid

    for component in ("genuine", "melo", "vits"):
        add(103, component, "train", n=64040 if component == "genuine" else 8000)
    # Retained XTTS for a training speaker is intentionally outside the B filter.
    omitted_xtts = add(103, "xtts", "train", assigned=False)
    for speaker, role in ((201, "selection"), (202, "calibration"), (203, "operating")):
        for component in ("genuine", "melo", "vits", "xtts"):
            add(speaker, component, "validation", role, n=70000 if role == "selection" and component == "genuine" else 8000)
    for component in ("genuine", "melo", "vits", "chatterbox"):
        add(301, component, "test")
    shards = root / "shards"
    shards.mkdir()
    for name in ("shard_00.tar", "shard_01.tar", "shard_02.tar"):
        with tarfile.open(shards / name, "w") as archive:
            for row in rows:
                if row["canonical_path"].split("::")[0] != name:
                    continue
                # Only train and selection audio is valid. Forbidden audio must never be decoded.
                blob = flac_bytes(pcm[row["recording_id"]]) if name != "shard_02.tar" else b"FORBIDDEN AUDIO"
                info = tarfile.TarInfo(row["recording_id"].replace(":", "__") + ".flac")
                info.size = len(blob)
                archive.addfile(info, io.BytesIO(blob))
    return {"rows": rows, "assignments": assignments, "chunks": chunks, "pcm": pcm,
            "shard_dir": shards, "omitted_xtts": omitted_xtts}


class PlanningAndAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".r2-u2-test-", dir=ROOT)
        self.root = Path(self.tmp.name)
        self.data = metadata_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, **changes):
        values = {**self.data, **changes}
        return r2_u2.select_plan(values["rows"], values["assignments"], values["chunks"])

    def test_plan_uses_only_assigned_train_and_selection_recordings(self):
        plan = self.plan()
        self.assertEqual(len(plan.train_rows), 3)
        self.assertEqual(len(plan.selection_rows), 4)
        self.assertEqual(len(plan.selection_chunks), 5)
        self.assertEqual(plan.shards(), ["shard_00.tar", "shard_01.tar"])
        self.assertEqual(plan.split_sha256, SPLIT_HASH)
        self.assertNotIn(self.data["omitted_xtts"], {r["recording_id"] for r in plan.train_rows})
        self.assertEqual(plan.counts["train_by_component"], {"genuine": 1, "melo": 1, "vits": 1})

    def test_plan_order_does_not_change_with_input_order(self):
        a = self.plan()
        b = self.plan(rows=list(reversed(self.data["rows"])), assignments=list(reversed(self.data["assignments"])),
                      chunks=list(reversed(self.data["chunks"])))
        self.assertEqual(a.sha256, b.sha256)
        self.assertEqual(a.train_rows, b.train_rows)

    def test_duplicate_assignment_and_duplicate_cohort_are_rejected(self):
        with self.assertRaises(ValueError):
            self.plan(assignments=self.data["assignments"] + self.data["assignments"][:1])
        with self.assertRaises(ValueError):
            self.plan(rows=self.data["rows"] + self.data["rows"][:1])

    def test_missing_assigned_cohort_row_is_rejected(self):
        with self.assertRaises(ValueError):
            self.plan(rows=self.data["rows"][1:])

    def test_mixed_split_hashes_and_non_b_assignments_are_rejected(self):
        for changes in ({"split_hash": "2" * 64}, {"protocol_id": "protocol:A"}):
            assignments = list(self.data["assignments"])
            assignments[0] = replace(assignments[0], **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(assignments=assignments)

    def test_forbidden_training_family_and_wrong_label_are_rejected(self):
        for changes in ({"component": "xtts"}, {"label": 1}):
            rows = deepcopy(self.data["rows"])
            rows[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(rows=rows)

    def test_wrong_group_and_validation_role_are_rejected(self):
        for index, changes in ((0, {"group_id": "speaker:999"}), (3, {"validation_role": "unknown"})):
            assignments = list(self.data["assignments"])
            assignments[index] = replace(assignments[index], **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(assignments=assignments)

    def test_pcm_identity_cannot_cross_training_and_selection(self):
        rows = deepcopy(self.data["rows"])
        rows[4]["pcm_sha256"] = rows[0]["pcm_sha256"]
        with self.assertRaises(ValueError):
            self.plan(rows=rows)

    def test_missing_and_duplicate_selection_chunks_are_rejected(self):
        selected = next(c for c in self.data["chunks"] if c.partition == "validation" and c.group_id == "speaker:201")
        for chunks in ([c for c in self.data["chunks"] if c.chunk_id != selected.chunk_id], self.data["chunks"] + [selected]):
            with self.subTest(count=len(chunks)), self.assertRaises(ValueError):
                self.plan(chunks=chunks)

    def test_selection_chunk_requires_canonical_preprocessing_and_group(self):
        index = next(i for i, c in enumerate(self.data["chunks"]) if c.partition == "validation" and c.group_id == "speaker:201")
        for changes in ({"preprocessing_hash": "a" * 64}, {"group_id": "speaker:999"}):
            chunks = list(self.data["chunks"])
            chunks[index] = replace(chunks[index], **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.plan(chunks=chunks)

    def test_unsafe_canonical_shard_and_member_are_rejected(self):
        for path in ("../outside.tar::x", "shard_00.tar::../x", "repair_00.tar::x"):
            rows = deepcopy(self.data["rows"])
            rows[0]["canonical_path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.plan(rows=rows)

    def test_checked_loader_reads_only_permitted_ids_and_verifies_pcm(self):
        plan = self.plan()
        loader = r2_u2.CheckedWaveformLoader(self.data["shard_dir"], plan.train_rows + plan.selection_rows)
        try:
            for row in plan.train_rows + plan.selection_rows:
                wave = loader(row["recording_id"])
                expected = self.data["pcm"][row["recording_id"]].astype(np.float32) / 32768.0
                np.testing.assert_array_equal(wave, expected)
            self.assertEqual(loader.verified_ids, {r["recording_id"] for r in plan.train_rows + plan.selection_rows})
            forbidden = next(r["recording_id"] for r in self.data["rows"] if r["speaker_id"] == "301")
            for recording_id in (forbidden, self.data["omitted_xtts"]):
                with self.subTest(forbidden=recording_id), self.assertRaises((KeyError, ValueError)):
                    loader(recording_id)
        finally:
            loader.close()

    def test_checked_loader_rejects_pcm_hash_and_length_mismatch(self):
        for changes in ({"pcm_sha256": "a" * 64}, {"n_samples": 1}):
            rows = deepcopy(self.plan().train_rows)
            rows[0].update(changes)
            loader = r2_u2.CheckedWaveformLoader(self.data["shard_dir"], rows)
            try:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    loader(rows[0]["recording_id"])
            finally:
                loader.close()


class RecipeContract(unittest.TestCase):
    def recipe(self):
        return json.loads((ROOT / "configs/r2_u2_ast_b.json").read_text(encoding="utf-8-sig"))

    def test_delivered_recipe_is_explicit_and_within_r2_scope(self):
        recipe = self.recipe()
        r2_u2.validate_recipe(recipe)
        self.assertEqual(recipe["train"]["max_epochs"], 2)
        self.assertEqual(recipe["train"]["batch_size"] * recipe["train"]["accumulation_steps"], 16)
        self.assertEqual(recipe["train"]["selection_threshold"], 0.5)
        self.assertEqual(recipe["train"]["protocol"], "B")
        self.assertEqual(recipe["preprocessor"]["max_length"], 1024)
        self.assertEqual(recipe["preprocessor"]["num_mel_bins"], 128)

    def test_recipe_rejects_wrong_task_protocol_or_epoch_scope(self):
        for changes in ({"max_epochs": 0}, {"max_epochs": 4}, {"protocol": "A"}, {"selection_threshold": 0.6}):
            recipe = self.recipe()
            recipe["train"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                r2_u2.validate_recipe(recipe)
        recipe = self.recipe()
        recipe["task"] = "R3-U2"
        with self.assertRaises(ValueError):
            r2_u2.validate_recipe(recipe)


class PreflightEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".r2-u2-evidence-", dir=ROOT)
        self.root = Path(self.tmp.name)
        self.data = metadata_fixture(self.root)
        # Copy source bytes solely to give the temporary project genuine source fingerprints.
        shutil.copytree(ROOT / "src/echoproof", self.root / "src/echoproof", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for relative in ("scripts/r2_u2_ast_warmup.py", "tests/test_r2_ast_warmup.py", "tests/test_r1_u2_ast.py",
                         "tests/test_m1_trainer.py", "pyproject.toml"):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        self.recipe = json.loads((ROOT / "configs/r2_u2_ast_b.json").read_text(encoding="utf-8-sig"))
        tables = {"data/manifests/cohort_v1.parquet": (self.data["rows"], write_table),
                  "data/splits/protocol_B_v1.parquet": (self.data["assignments"], write_parquet),
                  "data/processed/chunks_v1.parquet": (self.data["chunks"], write_parquet)}
        for relative, (rows, writer) in tables.items():
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            writer(rows, str(destination))
        support = {}
        by_id = {r["recording_id"]: r for r in self.data["rows"]}
        groups = {}
        for assignment in self.data["assignments"]:
            row = by_id[assignment.recording_id]
            component = row["component"]
            family = ("genuine" if component == "genuine" else "unseen_validation" if component == "xtts"
                      else "unseen_final" if component == "chatterbox" else "known_family")
            key = "|".join((assignment.partition, assignment.validation_role or "-", family,
                            "genuine" if component == "genuine" else "tts"))
            support.setdefault(key, {"recordings": 0, "speakers": 0})["recordings"] += 1
            groups.setdefault(key, set()).add(assignment.group_id)
        for key, speakers in groups.items():
            support[key]["speakers"] = len(speakers)
        reports = {"reports/r1/split_report.json": {"task": "R1-S2", "protocols": {"B": {
                       "split_sha256": SPLIT_HASH, "recordings": len(self.data["assignments"]), "support": support}}},
                   "reports/r1/leakage.json": {"task": "R1-S4", "passed": True, "violation_count": 0, "violations": []},
                   "reports/r1/ast_probe.json": json.loads((ROOT / "reports/r1/ast_probe.json").read_text(encoding="utf-8-sig")),
                   "reports/r1/r1_m1_colab_20261103T182916Z.json": json.loads((ROOT / "reports/r1/r1_m1_colab_20261103T182916Z.json").read_text(encoding="utf-8-sig"))}
        for relative, payload in reports.items():
            self.write_json(relative, payload)
        self.recipe["input_sha256"] = {relative: file_digest(self.root / relative) for relative in self.recipe["input_sha256"]}
        self.write_json("configs/r2_u2_ast_b.json", self.recipe)

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, relative, payload):
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload), encoding="utf-8")

    def update_evidence(self, relative, **changes):
        path = self.root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(changes)
        self.write_json(relative, payload)
        self.recipe["input_sha256"][relative] = file_digest(path)

    def test_preflight_validates_frozen_metadata_without_opening_audio(self):
        with patch("tarfile.open", side_effect=AssertionError("preflight must not open audio")):
            plan = r2_u2.preflight(self.root, self.recipe)
        self.assertEqual(plan.counts["train_recordings"], 3)
        self.assertEqual(plan.counts["selection_recordings"], 4)
        self.assertFalse((self.root / "data/processed/cohort_v1").exists())

    def test_preflight_rejects_metadata_bytes_changed_after_recipe_freeze(self):
        path = self.root / "data/manifests/cohort_v1.parquet"
        path.write_bytes(path.read_bytes() + b"changed after recipe freeze")
        with self.assertRaises(ValueError):
            r2_u2.preflight(self.root, self.recipe)

    def test_preflight_rejects_failed_leakage_even_when_file_hash_is_updated(self):
        self.update_evidence("reports/r1/leakage.json", passed=False, violation_count=1)
        with self.assertRaises(ValueError):
            r2_u2.preflight(self.root, self.recipe)

    def test_preflight_rejects_unverified_ast_and_trainer_evidence(self):
        for relative in ("reports/r1/ast_probe.json", "reports/r1/r1_m1_colab_20261103T182916Z.json"):
            path = self.root / relative
            original = path.read_bytes()
            original_hash = self.recipe["input_sha256"][relative]
            try:
                payload = json.loads(original)
                payload["verdict"]["verified"] = False
                self.write_json(relative, payload)
                self.recipe["input_sha256"][relative] = file_digest(path)
                with self.subTest(evidence=relative), self.assertRaises(ValueError):
                    r2_u2.preflight(self.root, self.recipe)
            finally:
                path.write_bytes(original)
                self.recipe["input_sha256"][relative] = original_hash

    def test_preflight_rejects_changed_shared_trainer_source(self):
        source = self.root / "src/echoproof/training/trainer.py"
        source.write_bytes(source.read_bytes() + b"\n# injected source change\n")
        with self.assertRaises(ValueError):
            r2_u2.preflight(self.root, self.recipe)

    def test_source_identity_does_not_capture_unrelated_later_stage_modules(self):
        before = r2_u2.source_hashes(self.root)
        unrelated = self.root / "src/echoproof/training/later_stage_fixture.py"
        unrelated.write_text("# unrelated temporary source fixture\n", encoding="utf-8")
        self.assertEqual(before, r2_u2.source_hashes(self.root))

    def fixture_report(self):
        hashes = r2_u2.source_hashes(self.root)
        report = {"task": "R2-U2 fixture verification", "returncode": 0, "skipped": False, "passed_count": 23,
                  "source_sha256": hashes, "summary": "generated fixture evidence, not a real test run",
                  "command": ["fixture-python", "-B", "-m", "pytest", "tests/test_r2_ast_warmup.py",
                              "tests/test_r1_u2_ast.py", "tests/test_m1_trainer.py", "-q", "-ra", "-p", "no:cacheprovider"]}
        relative = "reports/r2/r2_u2_tests_fixture.json"
        self.write_json(relative, report)
        return relative, report, hashes

    def test_fixture_gate_requires_success_zero_skips_and_current_source_identity(self):
        relative, report, hashes = self.fixture_report()
        reference = r2_u2._test_gate(self.root, relative, hashes)
        self.assertEqual(reference["sha256"], file_digest(self.root / relative))
        for changes in ({"returncode": 1}, {"skipped": True}, {"passed_count": 0}, {"source_sha256": {}},
                        {"task": "R2-U1 fixture verification"}):
            self.write_json(relative, {**report, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                r2_u2._test_gate(self.root, relative, hashes)

    def test_fixture_gate_rejects_missing_required_suites_and_filtered_runs(self):
        relative, report, hashes = self.fixture_report()
        missing = [item for item in report["command"] if item != "tests/test_r1_u2_ast.py"]
        filtered = report["command"] + ["-k", "only_one_case"]
        for command in (missing, filtered):
            self.write_json(relative, {**report, "command": command})
            with self.subTest(command=command), self.assertRaises(ValueError):
                r2_u2._test_gate(self.root, relative, hashes)

    def published_fixture(self):
        """A temporary integrity-only fixture: tensor payloads are deliberately not loaded."""
        plan = r2_u2.preflight(self.root, self.recipe)
        relative, _, hashes = self.fixture_report()
        reference = {"file": relative, "sha256": file_digest(self.root / relative)}
        provenance = {**plan.provenance, "recipe_sha256": digest(self.recipe), "plan_sha256": plan.sha256,
                      "source_sha256": hashes}
        run_dir = self.root / "runs/r2/fixture_warmup"
        run_dir.mkdir(parents=True)
        names = ("baseline.json", "best.pt", "latest.pt", "predictions_chunks.parquet", "predictions_recordings.parquet",
                 "resolved_config.json", "history.jsonl", "status.json", "training_mode_receipt.json", "head_gradient_receipt.json")
        for name in names:
            (run_dir / name).write_bytes(b"temporary integrity fixture, not an executed checkpoint\n")
        trainability = {"stage": "head", "trainable_names": ["classifier.dense.bias", "classifier.dense.weight",
                                                               "classifier.layernorm.bias", "classifier.layernorm.weight"]}
        baseline = {"schema_version": 1, "task": "R2-U2", "recipe": self.recipe, "provenance": provenance,
                    "tests": reference, "trainability": trainability}
        (run_dir / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
        receipt_identity = {"schema_version": 1, "task": "R2-U2", "baseline_sha256": file_digest(run_dir / "baseline.json"),
                            "provenance_sha256": digest(provenance)}
        receipts = {"training_mode_receipt.json": {"backbone_all_modules_eval": True, "trainable_names": trainability["trainable_names"]},
                    "head_gradient_receipt.json": {"finite_nonzero_scaled_classifier_gradient_observed": True}}
        for name, facts in receipts.items():
            (run_dir / name).write_text(json.dumps({**receipt_identity, **facts}), encoding="utf-8")
        row = {"run_id": run_dir.name, "owner": "fixture", "model": "ast_frozen", "protocol": "B",
               "config_hash": "a" * 64, "split_hash": SPLIT_HASH, "seed": 42, "epochs_run": 2, "best_epoch": 1,
               "stop_reason": "max_epochs", "val_macro_f1": 0.5, "status": "completed"}
        ledger_path = self.root / "reports/ledger.csv"
        append_ledger(ledger_path, row)
        with ledger_path.open(newline="", encoding="utf-8") as handle:
            saved_row = list(csv.DictReader(handle))[0]
        manifest = {"schema_version": 1, "task": "R2-U2", "verified": True,
                    "status": "colab_verified_pending_muneeb_review", "provenance": provenance,
                    "label_map": {"0": "genuine", "1": "tts"},
                    "run_directory": run_dir.relative_to(self.root).as_posix(),
                    "counts": plan.counts, "tests": reference, "trainability": trainability,
                    "files": {name: {"bytes": (run_dir / name).stat().st_size,
                                     "sha256": file_digest(run_dir / name)} for name in names},
                    "ledger": {"file": "reports/ledger.csv", "run_id": run_dir.name,
                               "row": saved_row, "row_sha256": digest(saved_row)},
                    "best_freeze_proof": {"passed": True}, "latest_freeze_proof": {"passed": True},
                    "reload_parity": {"passed": True, "scope": "selection validation only"},
                    "persisted_training_proof": {"passed": True,
                        "proof": "persisted actual training observations bound to original baseline"}}
        (run_dir / r2_u2.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
        return run_dir, manifest

    def test_verifier_checks_metadata_and_hashes_without_loading_audio_or_checkpoints(self):
        run_dir, _ = self.published_fixture()
        with patch("tarfile.open", side_effect=AssertionError("integrity verification must not read audio")), \
                patch("torch.load", side_effect=AssertionError("integrity verification must not load tensors")):
            result = r2_u2.verify_run(self.root, run_dir)
        self.assertTrue(result["integrity_passed"])

    def test_verifier_rejects_reversed_output_class_mapping(self):
        run_dir, manifest = self.published_fixture()
        manifest["label_map"] = {"0": "tts", "1": "genuine"}
        (run_dir / r2_u2.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            r2_u2.verify_run(self.root, run_dir)

    def test_missing_status_recovers_once_and_conflicting_status_is_preserved(self):
        run_dir = self.root / "status_recovery"
        run_dir.mkdir()
        summary = {"epochs_run": 2, "best_epoch": 1, "stop_reason": "max_epochs"}
        self.assertTrue(r2_u2.recover_status(run_dir, summary))
        original = (run_dir / "status.json").read_bytes()
        self.assertFalse(r2_u2.recover_status(run_dir, summary))
        with self.assertRaises(ValueError):
            r2_u2.recover_status(run_dir, {**summary, "best_epoch": 2})
        self.assertEqual((run_dir / "status.json").read_bytes(), original)

    def test_verifier_rejects_corrupt_checkpoint_and_changed_baseline(self):
        run_dir, _ = self.published_fixture()
        for name in ("best.pt", "baseline.json", "predictions_recordings.parquet"):
            path = run_dir / name
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"changed")
                with self.subTest(artifact=name), self.assertRaises(ValueError):
                    r2_u2.verify_run(self.root, run_dir)
            finally:
                path.write_bytes(original)

    def test_verifier_accepts_unrelated_ledger_append_but_rejects_duplicate_completion(self):
        run_dir, manifest = self.published_fixture()
        ledger = self.root / "reports/ledger.csv"
        append_ledger(ledger, {"run_id": "another_teammate_run", "status": "completed", "model": "lr", "protocol": "B"})
        self.assertTrue(r2_u2.verify_run(self.root, run_dir)["integrity_passed"])
        append_ledger(ledger, manifest["ledger"]["row"])
        with self.assertRaises(ValueError):
            r2_u2.verify_run(self.root, run_dir)

    def test_verifier_rejects_false_or_unbound_training_receipts_even_with_updated_hashes(self):
        run_dir, manifest = self.published_fixture()
        for name, changes in (("training_mode_receipt.json", {"backbone_all_modules_eval": False}),
                              ("head_gradient_receipt.json", {"finite_nonzero_scaled_classifier_gradient_observed": False}),
                              ("training_mode_receipt.json", {"baseline_sha256": "f" * 64})):
            path = run_dir / name
            original = path.read_bytes()
            original_info = deepcopy(manifest["files"][name])
            try:
                payload = json.loads(original)
                payload.update(changes)
                path.write_text(json.dumps(payload), encoding="utf-8")
                manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": file_digest(path)}
                (run_dir / r2_u2.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(receipt=name, changes=changes), self.assertRaises(ValueError):
                    r2_u2.verify_run(self.root, run_dir)
            finally:
                path.write_bytes(original)
                manifest["files"][name] = original_info
                (run_dir / r2_u2.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    def test_finished_checkpoint_recovers_missing_ledger_once_without_retraining(self):
        cfg = TrainConfig(**self.recipe["train"])
        latest = {"finished": True, "config_hash": "b" * 64, "epoch": 2,
                  "stop_reason": "max_epochs", "train_seconds": 12.0}
        best = {"epoch": 1, "metrics": {"macro_f1": 0.5, "eer": 0.5, "fpr": 0.5}}
        ledger = self.root / "reports/ledger.csv"
        self.assertFalse(ledger.exists())
        first = r2_u2._ledger_row(ledger, "recovered_run", cfg, "b" * 64, latest, best, SPLIT_HASH)
        original = ledger.read_bytes()
        second = r2_u2._ledger_row(ledger, "recovered_run", cfg, "b" * 64, latest, best, SPLIT_HASH)
        self.assertEqual(first["row_sha256"], second["row_sha256"])
        self.assertEqual(ledger.read_bytes(), original)
        self.assertIn("recovered", first["row"]["notes"])


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class ASTTrainerIntegration(unittest.TestCase):
    """Use a tiny local random model to test the real wrapper/trainer contract."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix=".r2-u2-ast-", dir=ROOT)
        cls.root = Path(cls.tmp.name)
        cls.snapshot = cls.root / "tiny_snapshot"
        cfg = ASTConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                        patch_size=16, frequency_stride=10, time_stride=10, max_length=64, num_mel_bins=16, num_labels=7)
        torch.manual_seed(0)
        ASTForAudioClassification(cfg).save_pretrained(cls.snapshot)
        cls.extractor = ASTFeatureExtractor(feature_size=1, sampling_rate=RATE, num_mel_bins=16, max_length=64,
                                           do_normalize=True, mean=-4.27, std=4.57, return_attention_mask=False)
        try:
            cls.extractor([np.zeros(8000, dtype=np.float32)], sampling_rate=RATE, return_tensors="pt")
            cls.features_ok = True
        except Exception:
            cls.features_ok = False

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        if not self.features_ok:
            self.skipTest("AST audio feature backend is unavailable; this is not a passing gate")
        self.case_tmp = tempfile.TemporaryDirectory(dir=self.root)
        self.directory = Path(self.case_tmp.name)

    def tearDown(self):
        if hasattr(self, "case_tmp"):
            self.case_tmp.cleanup()

    def fresh(self):
        model = ASTDetector.from_dir(self.snapshot, seed=7, feature_extractor=self.extractor)
        model.set_stage("head")
        return model

    def make(self, directory, model=None):
        rng = np.random.default_rng(4)
        waves = {f"train:{i}": rng.standard_normal(8000).astype(np.float32) * 0.1 for i in range(5)}
        waves.update({f"val:{i}": rng.standard_normal(8000).astype(np.float32) * 0.1 for i in range(2)})
        loader = DictWaveformLoader(waves)
        train = CropDataset([(f"train:{i}", i % 2) for i in range(5)], loader, seed=42, crop_samples=8000)
        chunks = [Chunk(1, f"chunk:B:val:{i}:0", f"val:{i}", "speaker:201", "protocol:B", "validation",
                        0, 8000, RATE, 0.0, 0.5, 8000, 8000, PREPROCESSING_HASH) for i in range(2)]
        selection = ChunkDataset(chunks, {f"val:{i}": i % 2 for i in range(2)}, loader)
        cfg = TrainConfig(seed=42, max_epochs=2, batch_size=2, accumulation_steps=2, eval_batch_size=2,
                          lr=1e-3, weight_decay=0.01, patience=3, scheduler="constant", device="cpu", precision="fp32",
                          deterministic=True, owner="fixture", model_name="tiny_ast_fixture", protocol="B")
        return Trainer(model or self.fresh(), cfg, train, selection, directory,
                       ledger_path=directory.parent / "fixture_ledger.csv", provenance={"split_sha256": SPLIT_HASH})

    def test_head_stage_contains_classifier_norm_and_rejects_partial_stage(self):
        model = self.fresh()
        report = r2_u2.validate_head_stage(model)
        self.assertTrue(report["trainable_names"])
        self.assertTrue(all(n.startswith("classifier.") for n in report["trainable_names"]))
        self.assertTrue(any("layernorm" in n.lower() or "layer_norm" in n.lower() for n in report["trainable_names"]))
        model.set_stage("partial", n_blocks=1)
        with self.assertRaises(ValueError):
            r2_u2.validate_head_stage(model)

    def test_two_epoch_training_keeps_backbone_eval_and_unchanged_and_reloads_best(self):
        trainer = self.make(self.directory / "run")
        before = trainer.model.digests()
        observed = []

        def observe(module, _inputs):
            if trainer.model.training:
                self.assertFalse(module.training, "frozen backbone reactivated during head training")
                observed.append(True)

        handle = trainer.model.model.audio_spectrogram_transformer.register_forward_pre_hook(observe)
        original_step = trainer._optimizer_step
        try:
            with patch.object(trainer, "_optimizer_step", wraps=original_step) as step:
                result = trainer.fit()
                self.assertEqual(step.call_count, 4)  # final single sample contributes an optimizer step in each epoch
        finally:
            handle.remove()
        self.assertTrue(observed)
        self.assertEqual(result["epochs_run"], 2)
        after = trainer.model.digests()
        self.assertNotEqual(before["classifier"], after["classifier"])
        for group in before.keys() - {"classifier"}:
            self.assertEqual(before[group], after[group], group)
        self.assertTrue((trainer.run_dir / "latest.pt").is_file())
        self.assertTrue((trainer.run_dir / "best.pt").is_file())
        best = torch.load(trainer.run_dir / "best.pt", map_location="cpu", weights_only=True)
        reloaded = self.make(self.directory / "reload")
        reloaded.model.load_state_dict(best["model"])
        _, chunk_rows, recording_rows = reloaded.evaluate()
        import pyarrow.parquet as pq
        saved = pq.read_table(trainer.run_dir / "predictions_recordings.parquet").to_pydict()
        self.assertEqual(saved["recording_id"], [r[0] for r in recording_rows])
        np.testing.assert_allclose(saved["raw_score"], [r[2] for r in recording_rows], atol=1e-6, rtol=1e-6)
        self.assertEqual(len(chunk_rows), 2)

    def test_interrupted_ast_resume_matches_uninterrupted_and_completed_resume_adds_no_row(self):
        uninterrupted = self.make(self.directory / "full")
        uninterrupted.fit()
        interrupted = self.make(self.directory / "interrupted")
        original = interrupted._train_epoch

        def stop_before_second(epoch):
            if epoch == 2:
                raise KeyboardInterrupt("fixture interruption")
            return original(epoch)

        with patch.object(interrupted, "_train_epoch", side_effect=stop_before_second):
            with self.assertRaises(KeyboardInterrupt):
                interrupted.fit()
        committed, _, history = r2_u2.validate_resume_state(interrupted.run_dir, interrupted.config_hash, 2)
        self.assertEqual(committed["epoch"], 1)
        self.assertEqual([row["epoch"] for row in history], [1])
        resumed = self.make(interrupted.run_dir)
        result = resumed.fit(resume=True)
        self.assertEqual(result["epochs_run"], 2)
        for name, tensor in uninterrupted.model.state_dict().items():
            torch.testing.assert_close(tensor, resumed.model.state_dict()[name], atol=0, rtol=0, msg=name)
        ledger = resumed.run_dir.parent / "fixture_ledger.csv"
        before = ledger.read_bytes()
        complete = self.make(interrupted.run_dir)
        complete.fit(resume=True)
        self.assertEqual(ledger.read_bytes(), before)

    def test_resume_guard_rejects_history_best_and_prediction_epochs_ahead_of_commit(self):
        trainer = self.make(self.directory / "guard")
        trainer.fit()
        latest, best, _ = r2_u2.validate_resume_state(trainer.run_dir, trainer.config_hash, 2)
        self.assertTrue(latest["finished"])
        history_path = trainer.run_dir / "history.jsonl"
        original_history = history_path.read_bytes()
        history = [json.loads(line) for line in original_history.splitlines()]
        history.append({**history[-1], "epoch": 3})
        history_path.write_text("\n".join(json.dumps(row) for row in history) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            r2_u2.validate_resume_state(trainer.run_dir, trainer.config_hash, 2)
        history_path.write_bytes(original_history)
        best_path = trainer.run_dir / "best.pt"
        original_best = best_path.read_bytes()
        torch.save({**best, "epoch": 3}, best_path)
        with self.assertRaises(ValueError):
            r2_u2.validate_resume_state(trainer.run_dir, trainer.config_hash, 2)
        best_path.write_bytes(original_best)
        import pyarrow as pa
        import pyarrow.parquet as pq
        predictions = trainer.run_dir / "predictions_recordings.parquet"
        table = pq.read_table(predictions)
        modified = table.set_column(table.schema.get_field_index("epoch"), "epoch", pa.array([3] * table.num_rows))
        pq.write_table(modified, predictions)
        with self.assertRaises(ValueError):
            r2_u2.validate_resume_state(trainer.run_dir, trainer.config_hash, 2)


if __name__ == "__main__":
    unittest.main()
