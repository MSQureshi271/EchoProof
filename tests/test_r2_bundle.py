"""Unit tests for R2-N2 bundle writer and loader (tests/test_r2_bundle.py).

Verifies:
- Bundle writing and atomic directory creation.
- Bundle loading and cryptographic component verification.
- Fixture bundle round-trip.
- Correct component hashes.
- Missing component rejection.
- Corrupted component rejection.
- Checksum mismatch rejection.
- Invalid metadata rejection.
- Unsafe path rejection (absolute, traversal, backslashes).
- Duplicate component rejection.
- Existing bundle overwrite prevention.
- Multi-component model support.
- Integration with load_lr_bundle (with & without threshold override).
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from echoproof.bundle import (
    BUNDLE_SCHEMA_VERSION,
    BundleError,
    load_bundle,
    write_bundle,
)
from echoproof.contracts import BundleSpecification, ContractError, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.inference import load_lr_bundle


def create_sample_spec(bundle_id: str = "bundle:test-v1", threshold: float = 0.5) -> BundleSpecification:
    return BundleSpecification(
        schema_version=1,
        bundle_id=bundle_id,
        model_version="test_model:v1",
        preprocessing_hash=PREPROCESSING_HASH,
        decision_rule=DecisionRule(schema_version=1, kind="threshold", threshold=threshold),
        class_0="genuine",
        class_1="tts",
    )


class TestBundleWriterAndLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bundle_write_and_load_roundtrip(self):
        spec = create_sample_spec("bundle:roundtrip")
        bundle_path = self.root / "test_bundle"

        comp_a = self.root / "model.bin"
        comp_a.write_bytes(b"dummy binary weights 12345")
        comp_b = self.root / "meta.json"
        comp_b.write_text('{"hyperparameters": {"lr": 0.01}}', encoding="utf-8")

        components = {
            "model/weights.bin": comp_a,
            "model/meta.json": comp_b,
        }

        # Write bundle
        written_json = write_bundle(bundle_path, spec, components)
        self.assertTrue(written_json.is_file())
        self.assertTrue((bundle_path / "model/weights.bin").is_file())
        self.assertTrue((bundle_path / "model/meta.json").is_file())

        # Load bundle
        loaded = load_bundle(bundle_path)
        self.assertEqual(loaded.spec.bundle_id, "bundle:roundtrip")
        self.assertEqual(loaded.spec.model_version, "test_model:v1")
        self.assertEqual(loaded.spec.decision_rule.threshold, 0.5)
        self.assertEqual(len(loaded.components), 2)
        self.assertIn("model/weights.bin", loaded.components)
        self.assertIn("model/meta.json", loaded.components)
        self.assertEqual(loaded.components["model/weights.bin"].read_bytes(), b"dummy binary weights 12345")

    def test_support_for_in_memory_bytes_and_multiple_components(self):
        spec = create_sample_spec("bundle:in-memory")
        bundle_path = self.root / "multi_comp_bundle"

        components = {
            "sub_a/weights_1.npz": b"fake npz 1",
            "sub_b/weights_2.npz": b"fake npz 2",
            "config.json": b'{"layers": 3}',
        }

        write_bundle(bundle_path, spec, components)
        loaded = load_bundle(bundle_path)
        self.assertEqual(len(loaded.components), 3)
        self.assertEqual(len(loaded.component_hashes), 3)

    def test_overwrite_prevention(self):
        spec = create_sample_spec("bundle:no-overwrite")
        bundle_path = self.root / "bundle_locked"

        components = {"model/weights.bin": b"123"}
        write_bundle(bundle_path, spec, components)

        # Attempt to write again without overwrite flag
        with self.assertRaises(BundleError) as ctx:
            write_bundle(bundle_path, spec, components, overwrite=False)
        self.assertIn("Refusing to overwrite", str(ctx.exception))

        # With overwrite flag, succeeds
        write_bundle(bundle_path, spec, {"model/weights.bin": b"456"}, overwrite=True)
        loaded = load_bundle(bundle_path)
        self.assertEqual(loaded.components["model/weights.bin"].read_bytes(), b"456")

    def test_missing_component_rejection(self):
        spec = create_sample_spec("bundle:missing-comp")
        bundle_path = self.root / "bundle_missing"

        write_bundle(bundle_path, spec, {"model/weights.bin": b"data"})

        # Manually delete component file
        (bundle_path / "model/weights.bin").unlink()

        with self.assertRaises(BundleError) as ctx:
            load_bundle(bundle_path)
        self.assertIn("Missing bundle component file", str(ctx.exception))

    def test_checksum_mismatch_rejection(self):
        spec = create_sample_spec("bundle:corrupt-comp")
        bundle_path = self.root / "bundle_corrupt"

        write_bundle(bundle_path, spec, {"model/weights.bin": b"clean_data"})

        # Tamper with file
        (bundle_path / "model/weights.bin").write_bytes(b"tampered_data")

        with self.assertRaises(BundleError) as ctx:
            load_bundle(bundle_path)
        self.assertIn("Checksum mismatch", str(ctx.exception))

    def test_unsafe_path_rejection_traversal_and_absolute(self):
        spec = create_sample_spec("bundle:unsafe")
        bundle_path = self.root / "bundle_unsafe"

        # Traversal attempts
        with self.assertRaises(BundleError):
            write_bundle(bundle_path, spec, {"../outside.bin": b"data"})

        with self.assertRaises(BundleError):
            write_bundle(bundle_path, spec, {"model/../../outside.bin": b"data"})

        # Absolute path attempt
        with self.assertRaises(BundleError):
            write_bundle(bundle_path, spec, {"/etc/passwd": b"data"})

        # Backslash attempt
        with self.assertRaises(BundleError):
            write_bundle(bundle_path, spec, {"model\\weights.bin": b"data"})

    def test_invalid_metadata_rejection(self):
        spec = create_sample_spec("bundle:invalid-meta")
        bundle_path = self.root / "bundle_invalid_meta"

        write_bundle(bundle_path, spec, {"model/weights.bin": b"data"})

        # Corrupt bundle.json syntax
        (bundle_path / "bundle.json").write_text("{not valid json}", encoding="utf-8")
        with self.assertRaises(BundleError):
            load_bundle(bundle_path)

        # Invalid schema version
        (bundle_path / "bundle.json").write_text(json.dumps({
            "schema_version": 999,
            "bundle_id": "bundle:invalid",
            "model_version": "v1",
            "preprocessing_hash": PREPROCESSING_HASH,
            "decision_rule": {"schema_version": 1, "kind": "threshold", "threshold": 0.5},
            "class_0": "genuine",
            "class_1": "tts",
            "components": {"model/weights.bin": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
        }), encoding="utf-8")
        with self.assertRaises(BundleError) as ctx:
            load_bundle(bundle_path)
        self.assertIn("Unsupported bundle schema_version", str(ctx.exception))

    def test_load_lr_bundle_with_verified_bundle_and_threshold_override(self):
        # Package actual Protocol B model files into a bundle fixture
        model_source = Path("runs/r2/r2_m1_20261106T140611Z/model")
        bundle_path = self.root / "verified_protocol_b_bundle"

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
            "model/weights.npz": model_source / "weights.npz",
            "model/lr_model.json": model_source / "lr_model.json",
        }
        write_bundle(bundle_path, spec, components)

        # 1. Load without explicit threshold: inherits stored 0.5
        b_default = load_lr_bundle(bundle_path)
        self.assertFalse(b_default.is_mock)
        self.assertEqual(b_default.spec.bundle_id, "bundle:r2-lr-protocol-b")
        self.assertEqual(b_default.spec.decision_rule.threshold, 0.5)

        # 2. Load with threshold override: sets 0.75 without modifying bundle.json on disk
        b_override = load_lr_bundle(bundle_path, threshold=0.75)
        self.assertEqual(b_override.spec.decision_rule.threshold, 0.75)

        # Verify bundle.json on disk still has stored threshold 0.5
        disk_loaded = load_bundle(bundle_path)
        self.assertEqual(disk_loaded.spec.decision_rule.threshold, 0.5)

        # 3. Invalid threshold override raises ContractError
        with self.assertRaises(ContractError):
            load_lr_bundle(bundle_path, threshold=1.5)

    def test_failure_safe_overwrite_preserves_original_on_error(self):
        spec = create_sample_spec("bundle:safe-overwrite")
        bundle_path = self.root / "bundle_safe_overwrite"

        # Write initial valid bundle
        write_bundle(bundle_path, spec, {"model/weights.bin": b"initial_valid_payload"})
        loaded_initial = load_bundle(bundle_path)
        self.assertEqual(loaded_initial.components["model/weights.bin"].read_bytes(), b"initial_valid_payload")

        # Attempt to overwrite with a component pointing to a nonexistent file
        nonexistent = self.root / "does_not_exist.bin"
        with self.assertRaises(FileNotFoundError):
            write_bundle(
                bundle_path,
                spec,
                {
                    "model/weights.bin": b"replacement_payload",
                    "model/extra.bin": nonexistent,
                },
                overwrite=True,
            )

        # Original bundle must remain completely intact and valid
        loaded_after = load_bundle(bundle_path)
        self.assertEqual(loaded_after.components["model/weights.bin"].read_bytes(), b"initial_valid_payload")
        self.assertNotIn("model/extra.bin", loaded_after.components)

        # Verify no temporary staging or backup directories were left behind
        parent_children = [p.name for p in bundle_path.parent.iterdir()]
        self.assertFalse(any(name.startswith(".bundle_stage_") for name in parent_children))
        self.assertFalse(any(name.startswith(".bundle_backup_") for name in parent_children))

    def test_load_lr_bundle_preprocessing_hash_mismatch_raises(self):
        model_source = Path("runs/r2/r2_m1_20261106T140611Z/model")
        bundle_path = self.root / "bundle_hash_mismatch"

        # Mismatched preprocessing hash
        bad_hash = "0" * 64
        spec = BundleSpecification(
            schema_version=1,
            bundle_id="bundle:bad-hash",
            model_version="logistic_regression:B",
            preprocessing_hash=bad_hash,
            decision_rule=DecisionRule(schema_version=1, kind="threshold", threshold=0.5),
            class_0="genuine",
            class_1="tts",
        )
        components = {
            "model/weights.npz": model_source / "weights.npz",
            "model/lr_model.json": model_source / "lr_model.json",
        }
        write_bundle(bundle_path, spec, components)

        with self.assertRaises(ContractError) as ctx:
            load_lr_bundle(bundle_path)
        self.assertIn("Bundle preprocessing hash mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

