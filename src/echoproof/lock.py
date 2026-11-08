"""R2-N2: Protocol B Lock Record Generator and Validator.

Generates and validates locks/B_lock.json capturing:
- Lock schema version.
- Bundle identity and verified metadata hash.
- Component SHA-256 hashes and byte sizes.
- Protocol B split hash.
- Selected model configuration and selection rationale.
- Selection metrics with type and range verification.
- Known-generator and unseen-generator evaluation results.
- Model limitations and non-deployment status.
- Test-access audit log distinguishing verified from unverified access.
- Evidence references and source hashes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from typing import Any, Mapping

from echoproof.artifacts import canonical_json, file_digest
from echoproof.bundle import load_bundle

LOCK_SCHEMA_VERSION = 1
PROTOCOL_B_SPLIT_HASH = "0854de6b00ae059a858d60f18f95df3f1501f579ea3e2fdefbd8de076e2034d7"
EXPECTED_FEATURE_SCHEMA_HASH = "fce2751a069d5d9b972ab772b1a76adf1a86bc12b13a07502cca171c86d50649"


class LockError(ValueError):
    """Raised when lock generation, metadata, schema, or audit validation fails."""


def _validate_hex_hash(value: Any, name: str) -> str:
    """Validate that a string is a 64-character lowercase SHA-256 hex digest."""
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise LockError(f"Field '{name}' must be a 64-character lowercase SHA-256 hex digest, got: {value!r}")
    return value


def _validate_unit_interval(value: Any, name: str) -> float:
    """Validate that a numeric value is finite and in the range [0.0, 1.0]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LockError(f"Metric '{name}' must be a numeric float, got {type(value).__name__}: {value!r}")
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0 or fval > 1.0:
        raise LockError(f"Metric '{name}' must be finite and in range [0.0, 1.0], got {value}")
    return fval


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that a value is a positive integer (> 0)."""
    if type(value) is not int or value <= 0:
        raise LockError(f"Field '{name}' must be an integer > 0, got {value!r}")
    return value


def validate_lock_record(
    lock_data: Mapping[str, Any],
    *,
    bundle_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a lock record structure, fields, metrics, and exact consistency with its bundle.

    Args:
        lock_data: Parsed lock dictionary.
        bundle_dir: Optional path to local bundle directory to cross-validate metadata and component hashes.

    Returns:
        The validated lock dictionary.

    Raises:
        LockError: If any required field is missing, invalid, out of range, or inconsistent.
    """
    if not isinstance(lock_data, Mapping):
        raise LockError("Lock record must be a mapping/object")

    schema_version = lock_data.get("schema_version")
    if type(schema_version) is not int or schema_version != LOCK_SCHEMA_VERSION:
        raise LockError(f"Unsupported lock schema_version: {schema_version} (expected {LOCK_SCHEMA_VERSION})")

    # 1. Identity & Protocol
    lock_id = lock_data.get("lock_id")
    if not isinstance(lock_id, str) or not lock_id.strip() or ":" not in lock_id:
        raise LockError(f"lock_id must be a namespaced nonempty string, got: {lock_id!r}")

    created_utc = lock_data.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc.strip():
        raise LockError(f"created_utc must be a nonempty ISO-8601 string, got: {created_utc!r}")

    protocol = lock_data.get("protocol")
    if protocol != "B":
        raise LockError(f"Expected protocol 'B', got: {protocol!r}")

    # 2. Bundle section
    bundle_info = lock_data.get("bundle")
    if not isinstance(bundle_info, Mapping):
        raise LockError("Lock must define a 'bundle' mapping")
    for field in ("bundle_id", "bundle_path", "bundle_metadata_hash", "components"):
        if field not in bundle_info:
            raise LockError(f"Missing required bundle field: '{field}'")

    bundle_id = bundle_info["bundle_id"]
    if not isinstance(bundle_id, str) or not bundle_id.startswith("bundle:"):
        raise LockError(f"bundle_id must start with 'bundle:', got: {bundle_id!r}")

    _validate_hex_hash(bundle_info["bundle_metadata_hash"], "bundle.bundle_metadata_hash")

    components = bundle_info["components"]
    if not isinstance(components, Mapping) or not components:
        raise LockError("Lock bundle components must be a nonempty mapping")

    # Check component hashes formatting and structure
    for comp_path, comp_meta in components.items():
        if not isinstance(comp_path, str) or not comp_path.strip() or "\\" in comp_path or comp_path.startswith("/"):
            raise LockError(f"Invalid relative component path in lock: {comp_path!r}")
        if isinstance(comp_meta, Mapping):
            if "sha256" not in comp_meta:
                raise LockError(f"Component '{comp_path}' missing 'sha256' field")
            _validate_hex_hash(comp_meta["sha256"], f"bundle.components.{comp_path}.sha256")
            if "bytes" in comp_meta:
                if type(comp_meta["bytes"]) is not int or comp_meta["bytes"] < 0:
                    raise LockError(f"Component '{comp_path}' bytes must be non-negative int, got {comp_meta['bytes']!r}")
        elif isinstance(comp_meta, str):
            _validate_hex_hash(comp_meta, f"bundle.components.{comp_path}")
        else:
            raise LockError(f"Invalid component metadata structure for '{comp_path}': {type(comp_meta).__name__}")

    # 3. Split verification
    split_info = lock_data.get("split")
    if not isinstance(split_info, Mapping):
        raise LockError("Lock must define a 'split' mapping")
    if not isinstance(split_info.get("split_version"), str) or not split_info.get("split_version", "").strip():
        raise LockError("split.split_version must be a nonempty string")
    if split_info.get("split_hash") != PROTOCOL_B_SPLIT_HASH:
        raise LockError(
            f"Split hash mismatch: expected {PROTOCOL_B_SPLIT_HASH}, got {split_info.get('split_hash')}"
        )

    # 4. Feature schema
    feat_info = lock_data.get("feature_schema")
    if not isinstance(feat_info, Mapping):
        raise LockError("Lock must define a 'feature_schema' mapping")
    if feat_info.get("feature_count") != 67:
        raise LockError(f"Expected 67 features, got {feat_info.get('feature_count')}")
    if feat_info.get("feature_schema_hash") != EXPECTED_FEATURE_SCHEMA_HASH:
        raise LockError(
            f"Feature schema hash mismatch: expected {EXPECTED_FEATURE_SCHEMA_HASH}, got {feat_info.get('feature_schema_hash')}"
        )
    if "preprocessing_hash" in feat_info:
        _validate_hex_hash(feat_info["preprocessing_hash"], "feature_schema.preprocessing_hash")

    # 5. Selected model
    sel_model = lock_data.get("selected_model")
    if not isinstance(sel_model, Mapping):
        raise LockError("Lock must define 'selected_model'")
    if sel_model.get("architecture") != "logistic_regression":
        raise LockError(f"Expected architecture 'logistic_regression', got: {sel_model.get('architecture')!r}")
    if sel_model.get("model_version") != "logistic_regression:B":
        raise LockError(f"Expected model_version 'logistic_regression:B', got: {sel_model.get('model_version')!r}")

    hp = sel_model.get("hyperparameters", {})
    if not isinstance(hp, Mapping):
        raise LockError("selected_model.hyperparameters must be an object")
    if hp.get("C") != 0.01 or hp.get("class_weight") != "none" or hp.get("seed") != 42 or hp.get("max_iter") != 2000:
        raise LockError(f"Unexpected selected hyperparameters: {hp}")

    _validate_unit_interval(sel_model.get("selection_threshold"), "selected_model.selection_threshold")

    # 6. Evaluation metrics & limitations validation
    sel_metrics = lock_data.get("selection_metrics")
    if not isinstance(sel_metrics, Mapping):
        raise LockError("Lock must define 'selection_metrics'")
    _validate_positive_int(sel_metrics.get("recordings"), "selection_metrics.recordings")

    for metric_name in ("accuracy", "macro_f1", "roc_auc", "eer", "tts_precision", "tts_recall", "genuine_recall", "fpr", "fnr"):
        if metric_name not in sel_metrics:
            raise LockError(f"Missing required selection metric: '{metric_name}'")
        _validate_unit_interval(sel_metrics[metric_name], f"selection_metrics.{metric_name}")

    panels = lock_data.get("evaluation_panels")
    if not isinstance(panels, Mapping):
        raise LockError("Lock must define 'evaluation_panels'")
    for panel_name in ("known_generators", "unseen_generators"):
        if panel_name not in panels or not isinstance(panels[panel_name], Mapping):
            raise LockError(f"Lock evaluation_panels must define '{panel_name}'")
        panel = panels[panel_name]
        if not isinstance(panel.get("generators"), list) or not panel.get("generators"):
            raise LockError(f"Panel '{panel_name}' must define a nonempty 'generators' list")
        _validate_positive_int(panel.get("recordings"), f"evaluation_panels.{panel_name}.recordings")
        for metric_name in ("roc_auc", "macro_f1", "accuracy", "tts_recall", "genuine_recall", "fpr"):
            if metric_name in panel:
                _validate_unit_interval(panel[metric_name], f"evaluation_panels.{panel_name}.{metric_name}")

    limitations = lock_data.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise LockError("Lock must explicitly record model limitations")
    limitations_text = " ".join(str(item) for item in limitations)
    if "deployment" not in limitations_text.lower():
        raise LockError("Limitations must explicitly state that model is not approved for deployment")
    if "unseen" not in limitations_text.lower() and "xtts" not in limitations_text.lower():
        raise LockError("Limitations must explicitly document performance degradation on unseen generators (XTTS)")

    # 7. Test-access audit
    audit = lock_data.get("test_access_audit")
    if not isinstance(audit, Mapping):
        raise LockError("Lock must define 'test_access_audit'")
    if audit.get("audit_status") != "partially_verified":
        raise LockError(
            f"audit_status must be 'partially_verified', got: {audit.get('audit_status')!r}. "
            "Independent proof of zero test access cannot be claimed post-hoc without cryptographic hardware enclaves."
        )
    if audit.get("test_partition_accessed") is not False:
        raise LockError("test_partition_accessed must be false")
    if not isinstance(audit.get("verified_history"), list) or not audit.get("verified_history"):
        raise LockError("test_access_audit must define a nonempty 'verified_history' list")
    unverified_items = audit.get("unverified_history")
    if not isinstance(unverified_items, list) or not unverified_items:
        raise LockError("test_access_audit must define a nonempty 'unverified_history' list")
    unverified_text = " ".join(str(item) for item in unverified_items).lower()
    if not any(k in unverified_text for k in ("enclave", "cannot be proven", "unverified", "disclaim")):
        raise LockError("test_access_audit.unverified_history must contain an explicit disclaimer regarding unverified execution bounds")

    # 8. Evidence references
    evidence_refs = lock_data.get("evidence_references")
    if not isinstance(evidence_refs, Mapping):
        raise LockError("Lock must define 'evidence_references'")
    _validate_hex_hash(evidence_refs.get("r2_m1_evidence_sha256"), "evidence_references.r2_m1_evidence_sha256")

    # 9. Exact consistency with actual bundle directory if provided
    if bundle_dir is not None:
        bundle_path = Path(bundle_dir)
        validated = load_bundle(bundle_path)

        # 9a. Verify bundle ID
        if validated.spec.bundle_id != bundle_info["bundle_id"]:
            raise LockError(f"Bundle ID mismatch: lock has '{bundle_info['bundle_id']}', bundle has '{validated.spec.bundle_id}'")

        # 9b. Verify bundle_metadata_hash against actual SHA-256 digest of bundle.json
        actual_bundle_metadata_hash = file_digest(bundle_path / "bundle.json")
        if bundle_info["bundle_metadata_hash"] != actual_bundle_metadata_hash:
            raise LockError(
                f"bundle_metadata_hash mismatch: lock has {bundle_info['bundle_metadata_hash']}, "
                f"actual bundle.json digest is {actual_bundle_metadata_hash}"
            )

        # 9c. Exact component set matching: ensure lock and bundle component sets match exactly
        lock_comp_set = set(components.keys())
        bundle_comp_set = set(validated.components.keys())
        if lock_comp_set != bundle_comp_set:
            raise LockError(
                f"Lock component records do not match bundle manifest exactly: "
                f"lock components={sorted(lock_comp_set)}, bundle components={sorted(bundle_comp_set)}"
            )

        # 9d. Component-by-component checksum and byte verification
        for comp_path, comp_meta in components.items():
            expected_sha = comp_meta.get("sha256") if isinstance(comp_meta, Mapping) else comp_meta
            actual_sha = validated.component_hashes[comp_path]
            if actual_sha != expected_sha:
                raise LockError(
                    f"Component hash mismatch for '{comp_path}': lock has {expected_sha}, bundle has {actual_sha}"
                )
            if isinstance(comp_meta, Mapping) and "bytes" in comp_meta:
                actual_bytes = validated.components[comp_path].stat().st_size
                if actual_bytes != comp_meta["bytes"]:
                    raise LockError(
                        f"Component byte size mismatch for '{comp_path}': lock has {comp_meta['bytes']}, bundle has {actual_bytes}"
                    )

    return dict(lock_data)


def load_lock_record(lock_path: str | Path, *, bundle_dir: str | Path | None = None) -> dict[str, Any]:
    """Read and validate a lock record file.

    Args:
        lock_path: Path to B_lock.json.
        bundle_dir: Optional bundle directory path to cross-validate.

    Returns:
        Validated lock record dictionary.
    """
    path = Path(lock_path)
    if not path.is_file():
        raise FileNotFoundError(f"Lock record file does not exist: {path}")

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LockError(f"Cannot parse lock record JSON in {path}: {exc}") from exc

    return validate_lock_record(raw_data, bundle_dir=bundle_dir)


def generate_protocol_b_lock(
    bundle_dir: str | Path,
    evidence_path: str | Path,
    output_path: str | Path = "locks/B_lock.json",
    *,
    overwrite: bool = False,
) -> Path:
    """Generate and validate the Protocol B lock record (locks/B_lock.json).

    Guarantees failure-safe operation:
    - Never deletes the existing lock file before the new record has been completely prepared and validated.
    - Preserves the existing lock file if generation, verification, or write fails.

    Args:
        bundle_dir: Path to the packaged Protocol B bundle (e.g. bundles/protocol_B_lr).
        evidence_path: Path to R2-M1 evidence.json.
        output_path: Target path for B_lock.json (defaults to locks/B_lock.json).
        overwrite: If False, refuses to overwrite an existing lock file.

    Returns:
        Path to the written lock file.
    """
    out_file = Path(output_path)
    if out_file.exists() and not overwrite:
        raise LockError(f"Refusing to overwrite existing lock record: {out_file}")

    # 1. Load and verify bundle
    bundle = load_bundle(bundle_dir)

    # 2. Load and verify R2-M1 evidence
    ev_path = Path(evidence_path)
    if not ev_path.is_file():
        raise FileNotFoundError(f"R2-M1 evidence file not found: {ev_path}")
    evidence = json.loads(ev_path.read_text(encoding="utf-8"))

    if evidence.get("protocol") != "B":
        raise LockError(f"Expected evidence protocol 'B', got {evidence.get('protocol')}")

    selected_order = evidence.get("selected_order", 0)
    trials = evidence.get("trials", [])
    selected_trial = next((t for t in trials if t.get("order") == selected_order), None)
    if not selected_trial:
        raise LockError(f"Selected trial order {selected_order} not found in evidence trials")

    sel_metrics = selected_trial.get("metrics", {})
    known_panel = evidence.get("selection_panels", {}).get("known", {})
    unseen_panel = evidence.get("selection_panels", {}).get("unseen_xtts", {})

    # 3. Read bundle.json to compute actual bundle metadata hash
    bundle_json_file = Path(bundle_dir) / "bundle.json"
    bundle_metadata_hash = file_digest(bundle_json_file)

    # 4. Construct components dictionary matching bundle exactly
    components_payload = {}
    for rel_path, comp_path in sorted(bundle.components.items()):
        components_payload[rel_path] = {
            "sha256": bundle.component_hashes[rel_path],
            "bytes": comp_path.stat().st_size,
        }

    # 5. Assemble lock payload with precise test-access audit semantics
    lock_payload = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "lock_id": "lock:protocol_b_lr_v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": "B",
        "bundle": {
            "bundle_id": bundle.spec.bundle_id,
            "bundle_path": str(Path(bundle_dir).as_posix()),
            "bundle_metadata_hash": bundle_metadata_hash,
            "components": components_payload,
        },
        "split": {
            "split_version": "r1-v1",
            "split_hash": PROTOCOL_B_SPLIT_HASH,
        },
        "feature_schema": {
            "feature_count": 67,
            "feature_schema_hash": EXPECTED_FEATURE_SCHEMA_HASH,
            "preprocessing_hash": bundle.spec.preprocessing_hash,
        },
        "selected_model": {
            "architecture": "logistic_regression",
            "model_version": bundle.spec.model_version,
            "hyperparameters": {
                "C": selected_trial.get("C", 0.01),
                "class_weight": selected_trial.get("class_weight", "none"),
                "max_iter": selected_trial.get("diagnostics", {}).get("max_iter", 2000),
                "seed": evidence.get("seed", 42),
            },
            "selection_threshold": evidence.get("threshold", 0.5),
            "source_run_id": evidence.get("task", "R2-M1"),
            "trial_order": selected_order,
        },
        "selection_rationale": (
            f"Selected Trial {selected_order} of {len(trials)} in R2-M1 grid search. "
            f"Achieved highest macro F1 ({sel_metrics.get('macro_f1', 0.0):.4f}) and lowest EER "
            f"({sel_metrics.get('eer', 0.0):.4f}) on the Protocol B selection validation role "
            f"at the fixed uncalibrated selection threshold 0.5."
        ),
        "selection_metrics": {
            "recordings": sel_metrics.get("recordings", 2000),
            "accuracy": sel_metrics.get("accuracy"),
            "macro_f1": sel_metrics.get("macro_f1"),
            "roc_auc": sel_metrics.get("roc_auc"),
            "eer": sel_metrics.get("eer"),
            "tts_precision": sel_metrics.get("tts_precision"),
            "tts_recall": sel_metrics.get("tts_recall"),
            "genuine_recall": sel_metrics.get("genuine_recall"),
            "fpr": sel_metrics.get("fpr"),
            "fnr": sel_metrics.get("fnr"),
        },
        "evaluation_panels": {
            "known_generators": {
                "generators": ["melo", "vits"],
                "recordings": known_panel.get("recordings", 1500),
                "roc_auc": known_panel.get("roc_auc"),
                "macro_f1": known_panel.get("macro_f1"),
                "accuracy": known_panel.get("accuracy"),
                "tts_recall": known_panel.get("tts_recall"),
                "genuine_recall": known_panel.get("genuine_recall"),
                "fpr": known_panel.get("fpr"),
            },
            "unseen_generators": {
                "generators": ["xtts"],
                "recordings": unseen_panel.get("recordings", 1000),
                "roc_auc": unseen_panel.get("roc_auc"),
                "macro_f1": unseen_panel.get("macro_f1"),
                "accuracy": unseen_panel.get("accuracy"),
                "tts_recall": unseen_panel.get("tts_recall"),
                "genuine_recall": unseen_panel.get("genuine_recall"),
                "fpr": unseen_panel.get("fpr"),
            },
        },
        "limitations": [
            "NOT_DEPLOYMENT_APPROVED: This model is a provisional development model from Milestone GR2, not an approved production detector.",
            "GENERALIZATION_FAILURE: Severe performance degradation on unseen generator XTTS (ROC-AUC 0.5261 vs 0.9674 on known generators; TTS recall 14.2% vs 95.2%).",
            "UNCALIBRATED_SCORES: Raw model scores are class-1 probabilities from uncalibrated logistic regression; threshold 0.5 is for selection ranking only.",
            "FEATURE_SCOPE: Model relies strictly on 67 handcrafted acoustic features with a linear decision surface; vulnerable to acoustic recording channel mismatch.",
        ],
        "test_access_audit": {
            "audit_status": "partially_verified",
            "test_partition_accessed": False,
            "verified_history": [
                "Verified by codebase and training pipeline audit: R2-M1 training referenced and loaded strictly the Protocol B train partition features (1,510 recordings, 151 speakers).",
                "Verified by codebase and selection pipeline audit: R2-M1 model selection was scored strictly on the Protocol B validation selection role (2,000 recordings, 25 speakers).",
                "Verified by split manifest audit: calibration and operating validation roles were excluded from R2-M1 model fitting and selection.",
                "Verified by artifact audit: evidence.json contains no test partition recording identifiers, evaluation metrics, or predictions.",
            ],
            "unverified_history": [
                "Unverified: Host filesystem, developer shell sessions, and external processes were not monitored by cryptographic hardware enclaves or immutable access audit logs.",
                "Unverified: Absolute absence of test data access on local developer disks cannot be proven post-hoc from software logs alone.",
                "No claim of independent cryptographic proof of zero test access is made; audit status remains partially_verified.",
            ],
        },
        "evidence_references": {
            "r2_m1_evidence_path": str(ev_path.as_posix()),
            "r2_m1_evidence_sha256": file_digest(ev_path),
            "source_code_sha256": evidence.get("source_sha256", {}),
        },
    }

    # 6. Validate the complete lock record before writing
    validate_lock_record(lock_payload, bundle_dir=bundle_dir)

    # 7. Failure-safe atomic write:
    # Staged to a temporary file first; only replaces out_file atomically via os.replace when fully ready.
    # out_file is never unlinked before replacement is verified and staged.
    out_file.parent.mkdir(parents=True, exist_ok=True)
    lock_json_bytes = (canonical_json(lock_payload) + "\n").encode("utf-8")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=out_file.parent, prefix=".lock_stage_", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(lock_json_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, out_file)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return out_file
