"""Unit tests for R2-N2 Protocol B Lock Record (tests/test_r2_lock.py).

Verifies:
- Protocol B lock record generation and validation.
- Bundle identity consistency.
- Component SHA-256 hash consistency.
- Protocol B split hash verification.
- Selection rationale and metrics recording.
- Test-access audit structure (distinguishing verified from unverified).
- Missing required evidence handling.
- Invalid/tampered lock rejection.
- Overwrite prevention.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from echoproof.bundle import write_bundle
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.lock import (
    EXPECTED_FEATURE_SCHEMA_HASH,
    LOCK_SCHEMA_VERSION,
    PROTOCOL_B_SPLIT_HASH,
    LockError,
    generate_protocol_b_lock,
    load_lock_record,
    validate_lock_record,
)

REAL_MODEL_DIR = Path("runs/r2/r2_m1_20261106T140611Z/model")
REAL_EVIDENCE_PATH = Path("runs/r2/r2_m1_20261106T140611Z/evidence.json")


class TestProtocolBLock(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Build a fixture bundle
        self.bundle_dir = self.root / "test_bundle"
        spec = BundleSpecification(
            schema_version=1,
            bundle_id="bundle:r2-lr-protocol-b",
            model_version="logistic_regression:B",
            preprocessing_hash=PREPROCESSING_HASH,
            decision_rule=DecisionRule(schema_version=1, kind="threshold", threshold=0.5),
            class_0="genuine",
            class_1="tts",
        )
        components = {
            "model/weights.npz": REAL_MODEL_DIR / "weights.npz",
            "model/lr_model.json": REAL_MODEL_DIR / "lr_model.json",
        }
        write_bundle(self.bundle_dir, spec, components)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_and_load_lock_record(self):
        lock_file = self.root / "B_lock.json"

        # Generate lock
        generated_path = generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
            overwrite=False,
        )
        self.assertTrue(generated_path.is_file())

        # Load and validate
        lock = load_lock_record(lock_file, bundle_dir=self.bundle_dir)

        self.assertEqual(lock["schema_version"], LOCK_SCHEMA_VERSION)
        self.assertEqual(lock["protocol"], "B")
        self.assertEqual(lock["bundle"]["bundle_id"], "bundle:r2-lr-protocol-b")
        self.assertEqual(lock["split"]["split_hash"], PROTOCOL_B_SPLIT_HASH)
        self.assertEqual(lock["feature_schema"]["feature_count"], 67)
        self.assertEqual(lock["feature_schema"]["feature_schema_hash"], EXPECTED_FEATURE_SCHEMA_HASH)

        # Selection rationale and hyperparameters
        self.assertEqual(lock["selected_model"]["hyperparameters"]["C"], 0.01)
        self.assertEqual(lock["selected_model"]["hyperparameters"]["class_weight"], "none")
        self.assertEqual(lock["selected_model"]["hyperparameters"]["seed"], 42)
        self.assertIn("Trial 0", lock["selection_rationale"])

        # Limitations & Non-deployment status
        limitations = lock["limitations"]
        self.assertTrue(any("NOT_DEPLOYMENT_APPROVED" in str(l) for l in limitations))
        self.assertTrue(any("GENERALIZATION_FAILURE" in str(l) for l in limitations))

        # Test access audit
        audit = lock["test_access_audit"]
        self.assertFalse(audit["test_partition_accessed"])
        self.assertTrue(len(audit["verified_history"]) >= 3)
        self.assertTrue(len(audit["unverified_history"]) >= 1)

    def test_overwrite_prevention(self):
        lock_file = self.root / "B_lock_locked.json"
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
            overwrite=False,
        )

        # Attempt to overwrite without overwrite flag
        with self.assertRaises(LockError) as ctx:
            generate_protocol_b_lock(
                bundle_dir=self.bundle_dir,
                evidence_path=REAL_EVIDENCE_PATH,
                output_path=lock_file,
                overwrite=False,
            )
        self.assertIn("Refusing to overwrite", str(ctx.exception))

        # Succeeded when overwrite=True
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
            overwrite=True,
        )

    def test_missing_evidence_raises(self):
        with self.assertRaises(FileNotFoundError):
            generate_protocol_b_lock(
                bundle_dir=self.bundle_dir,
                evidence_path=self.root / "nonexistent_evidence.json",
                output_path=self.root / "B_lock_err.json",
            )

    def test_bundle_component_hash_mismatch_raises(self):
        lock_file = self.root / "B_lock_tamper.json"
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
        )

        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        # Tamper component hash
        lock_data["bundle"]["components"]["model/weights.npz"]["sha256"] = "0" * 64

        with self.assertRaises(LockError) as ctx:
            validate_lock_record(lock_data, bundle_dir=self.bundle_dir)
        self.assertIn("Component hash mismatch", str(ctx.exception))

    def test_split_hash_tamper_raises(self):
        lock_file = self.root / "B_lock_split_tamper.json"
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
        )

        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        lock_data["split"]["split_hash"] = "f" * 64

        with self.assertRaises(LockError) as ctx:
            validate_lock_record(lock_data)
        self.assertIn("Split hash mismatch", str(ctx.exception))

    def test_missing_limitations_or_deployment_warning_raises(self):
        lock_file = self.root / "B_lock_limitations.json"
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
        )

        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        # Remove deployment warning
        lock_data["limitations"] = ["some irrelevant note"]

        with self.assertRaises(LockError) as ctx:
            validate_lock_record(lock_data)
        self.assertIn("deployment", str(ctx.exception))

    def test_bundle_metadata_hash_mismatch_raises(self):
        lock_file = self.root / "B_lock_meta_mismatch.json"
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
        )

        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        # Tamper bundle_metadata_hash
        lock_data["bundle"]["bundle_metadata_hash"] = "0" * 64

        with self.assertRaises(LockError) as ctx:
            validate_lock_record(lock_data, bundle_dir=self.bundle_dir)
        self.assertIn("bundle_metadata_hash mismatch", str(ctx.exception))

    def test_lock_component_exact_set_mismatch_raises(self):
        lock_file = self.root / "B_lock_exact_set.json"
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
        )

        # 1. Lock has an extra component not present in bundle manifest
        lock_data_extra = json.loads(lock_file.read_text(encoding="utf-8"))
        lock_data_extra["bundle"]["components"]["model/extra.bin"] = {
            "sha256": "0" * 64,
            "bytes": 100,
        }
        with self.assertRaises(LockError) as ctx:
            validate_lock_record(lock_data_extra, bundle_dir=self.bundle_dir)
        self.assertIn("match bundle manifest exactly", str(ctx.exception))

        # 2. Lock is missing a component present in bundle manifest
        lock_data_missing = json.loads(lock_file.read_text(encoding="utf-8"))
        del lock_data_missing["bundle"]["components"]["model/weights.npz"]
        with self.assertRaises(LockError) as ctx:
            validate_lock_record(lock_data_missing, bundle_dir=self.bundle_dir)
        self.assertIn("match bundle manifest exactly", str(ctx.exception))

    def test_metric_type_and_range_validation(self):
        lock_file = self.root / "B_lock_metric_range.json"
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
        )

        # Accuracy > 1.0
        bad_acc = json.loads(lock_file.read_text(encoding="utf-8"))
        bad_acc["selection_metrics"]["accuracy"] = 1.05
        with self.assertRaises(LockError) as ctx:
            validate_lock_record(bad_acc)
        self.assertIn("[0.0, 1.0]", str(ctx.exception))

        # Negative FPR
        bad_fpr = json.loads(lock_file.read_text(encoding="utf-8"))
        bad_fpr["selection_metrics"]["fpr"] = -0.01
        with self.assertRaises(LockError) as ctx:
            validate_lock_record(bad_fpr)
        self.assertIn("[0.0, 1.0]", str(ctx.exception))

        # Non-positive integer recordings count
        bad_rec = json.loads(lock_file.read_text(encoding="utf-8"))
        bad_rec["selection_metrics"]["recordings"] = 0
        with self.assertRaises(LockError) as ctx:
            validate_lock_record(bad_rec)
        self.assertIn("integer > 0", str(ctx.exception))

        # Panel metric non-numeric
        bad_panel = json.loads(lock_file.read_text(encoding="utf-8"))
        bad_panel["evaluation_panels"]["known_generators"]["roc_auc"] = "high"
        with self.assertRaises(LockError) as ctx:
            validate_lock_record(bad_panel)
        self.assertIn("must be a numeric float", str(ctx.exception))

    def test_audit_status_validation(self):
        lock_file = self.root / "B_lock_audit.json"
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
        )

        # audit_status cannot be "fully_verified"
        tampered_audit = json.loads(lock_file.read_text(encoding="utf-8"))
        tampered_audit["test_access_audit"]["audit_status"] = "fully_verified"
        with self.assertRaises(LockError) as ctx:
            validate_lock_record(tampered_audit)
        self.assertIn("audit_status must be 'partially_verified'", str(ctx.exception))

        # unverified_history must acknowledge lack of hardware enclaves / post-hoc limitations
        tampered_hist = json.loads(lock_file.read_text(encoding="utf-8"))
        tampered_hist["test_access_audit"]["unverified_history"] = ["Everything is completely verified."]
        with self.assertRaises(LockError) as ctx:
            validate_lock_record(tampered_hist)
        self.assertIn("disclaimer", str(ctx.exception))

    def test_failure_safe_overwrite_preserves_original_on_error(self):
        lock_file = self.root / "B_lock_safe.json"
        generate_protocol_b_lock(
            bundle_dir=self.bundle_dir,
            evidence_path=REAL_EVIDENCE_PATH,
            output_path=lock_file,
        )
        original_bytes = lock_file.read_bytes()

        # Attempt to overwrite with missing evidence path
        nonexistent_evidence = self.root / "nonexistent_evidence.json"
        with self.assertRaises(FileNotFoundError):
            generate_protocol_b_lock(
                bundle_dir=self.bundle_dir,
                evidence_path=nonexistent_evidence,
                output_path=lock_file,
                overwrite=True,
            )

        # Verify that original lock file was not deleted, truncated, or modified
        self.assertTrue(lock_file.is_file())
        self.assertEqual(lock_file.read_bytes(), original_bytes)

        # Verify that load_lock_record still loads the preserved lock file without error
        loaded = load_lock_record(lock_file, bundle_dir=self.bundle_dir)
        self.assertEqual(loaded["lock_id"], "lock:protocol_b_lr_v1")


if __name__ == "__main__":
    unittest.main()

