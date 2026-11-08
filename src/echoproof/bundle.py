"""R2-N2: Reusable bundle writer and loader for EchoProof.

Provides:
- write_bundle: Packages model component files into a verified local bundle directory with bundle.json.
  Uses failure-safe atomic staging to ensure existing bundles are preserved if replacement fails.
- load_bundle: Inspects, validates schema, checks path safety, and verifies SHA-256 checksums for local bundles.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
from typing import Any, Mapping

from echoproof.artifacts import canonical_json, file_digest
from echoproof.contracts import BundleSpecification, ContractError, DecisionRule

BUNDLE_SCHEMA_VERSION = 1


class BundleError(ValueError):
    """Raised when bundle packaging, metadata, path, or checksum validation fails."""


def _validate_relative_component_path(rel_path: str) -> str:
    """Validate that a relative component path is normalized POSIX, safe, and non-traversing."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise BundleError("Component path must be a nonempty string")
    if "\\" in rel_path:
        raise BundleError(f"Unsafe component path with backslashes: {rel_path!r}")
    posix_p = PurePosixPath(rel_path)
    win_p = PureWindowsPath(rel_path)
    if posix_p.is_absolute() or win_p.is_absolute() or win_p.drive:
        raise BundleError(f"Component path must be relative: {rel_path!r}")
    parts = rel_path.split("/")
    if any(p in ("..", ".", "") for p in parts):
        raise BundleError(f"Unsafe component path containing traversal or empty parts: {rel_path!r}")
    return rel_path


@dataclass(frozen=True)
class ValidatedBundle:
    """Validated in-memory representation of a verified local bundle."""
    spec: BundleSpecification
    bundle_dir: Path
    components: dict[str, Path]
    component_hashes: dict[str, str]
    raw_metadata: dict[str, Any]

    def get_component_path(self, rel_path: str) -> Path:
        norm = _validate_relative_component_path(rel_path)
        if norm not in self.components:
            raise BundleError(f"Component '{norm}' not found in bundle '{self.spec.bundle_id}'")
        return self.components[norm]


def write_bundle(
    bundle_dir: str | Path,
    spec: BundleSpecification | Mapping[str, Any],
    components: Mapping[str, str | Path | bytes],
    *,
    extra: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Package model component files and canonical metadata into a bundle directory.

    Guarantees failure-safe operation:
    - Never deletes or modifies an existing bundle until replacement is fully prepared.
    - Preserves existing bundle if replacement staging or transfer fails.

    Args:
        bundle_dir: Target directory path for the bundle.
        spec: BundleSpecification contract instance or compatible dictionary.
        components: Mapping of relative component POSIX paths (e.g. 'model/weights.npz')
                    to source file paths or in-memory bytes.
        extra: Optional additional metadata (e.g. training references, feature schema info).
        overwrite: If False, refuses to overwrite an existing bundle directory.

    Returns:
        Path to the generated bundle.json file.
    """
    target_dir = Path(bundle_dir)

    # 1. Overwrite safety check
    if target_dir.exists() and not overwrite:
        raise BundleError(f"Refusing to overwrite existing bundle directory: {target_dir}")

    # 2. Validate and convert spec to BundleSpecification
    if isinstance(spec, BundleSpecification):
        spec_obj = spec
    elif isinstance(spec, Mapping):
        spec_data = dict(spec)
        if isinstance(spec_data.get("decision_rule"), DecisionRule):
            spec_data["decision_rule"] = spec_data["decision_rule"].to_dict()
        try:
            spec_obj = BundleSpecification.from_dict(spec_data)
        except ContractError as exc:
            raise BundleError(f"Invalid specification contract: {exc}") from exc
    else:
        raise BundleError(f"Expected BundleSpecification or mapping, got: {type(spec).__name__}")

    # 3. Validate components collection
    if not isinstance(components, Mapping) or not components:
        raise BundleError("Bundle must contain at least one component")

    normalized_components: dict[str, str | Path | bytes] = {}
    for rel_path, source in components.items():
        norm_path = _validate_relative_component_path(rel_path)
        if norm_path in normalized_components:
            raise BundleError(f"Duplicate component path: {norm_path}")
        normalized_components[norm_path] = source

    # 4. Stage bundle files completely in a temporary directory on the same parent filesystem
    parent_dir = target_dir.resolve().parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix=".bundle_stage_", dir=parent_dir))
    backup_dir: Path | None = None

    try:
        component_manifest: dict[str, dict[str, Any]] = {}
        for rel_path, source in sorted(normalized_components.items()):
            dest_file = temp_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(source, (str, Path)):
                src_path = Path(source)
                if not src_path.is_file():
                    raise FileNotFoundError(f"Source component file does not exist: {src_path}")
                shutil.copy2(src_path, dest_file)
            elif isinstance(source, (bytes, bytearray)):
                dest_file.write_bytes(source)
            else:
                raise BundleError(f"Invalid component source type for '{rel_path}': {type(source).__name__}")

            sha256 = file_digest(dest_file)
            size_bytes = dest_file.stat().st_size
            component_manifest[rel_path] = {
                "sha256": sha256,
                "bytes": size_bytes,
            }

        # 5. Generate canonical bundle.json
        metadata = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": spec_obj.bundle_id,
            "model_version": spec_obj.model_version,
            "preprocessing_hash": spec_obj.preprocessing_hash,
            "decision_rule": spec_obj.decision_rule.to_dict(),
            "class_0": spec_obj.class_0,
            "class_1": spec_obj.class_1,
            "components": component_manifest,
        }
        if extra:
            metadata["extra"] = dict(extra)

        bundle_json_bytes = (canonical_json(metadata) + "\n").encode("utf-8")
        (temp_dir / "bundle.json").write_bytes(bundle_json_bytes)

        # 6. Failure-safe commit:
        # If target_dir already exists, move it to a temporary backup first rather than deleting it.
        # If moving temp_dir into target_dir fails, restore from backup.
        if target_dir.exists():
            backup_container = Path(tempfile.mkdtemp(prefix=".bundle_backup_", dir=parent_dir))
            backup_dir = backup_container / "old_bundle"
            shutil.move(str(target_dir), str(backup_dir))

        try:
            shutil.move(str(temp_dir), str(target_dir))
            temp_dir = None  # successfully moved
            if backup_dir is not None and backup_dir.parent.exists():
                shutil.rmtree(backup_dir.parent, ignore_errors=True)
                backup_dir = None
        except Exception:
            # Restore previous target_dir if the new bundle could not be moved in
            if backup_dir is not None and backup_dir.exists() and not target_dir.exists():
                shutil.move(str(backup_dir), str(target_dir))
            raise

    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.parent.exists():
            shutil.rmtree(backup_dir.parent, ignore_errors=True)

    return target_dir / "bundle.json"


def load_bundle(bundle_dir: str | Path) -> ValidatedBundle:
    """Load, inspect, and cryptographically verify a local model bundle directory.

    Args:
        bundle_dir: Directory containing bundle.json and its associated component files.

    Returns:
        ValidatedBundle with verified specification and resolved component file paths.

    Raises:
        FileNotFoundError: If the directory or bundle.json is missing.
        BundleError: If metadata schema, path safety, or SHA-256 verification fails.
    """
    path = Path(bundle_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {path}")

    bundle_json_file = path / "bundle.json"
    if not bundle_json_file.is_file():
        raise FileNotFoundError(f"Bundle metadata file missing: {bundle_json_file}")

    try:
        raw_metadata = json.loads(bundle_json_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BundleError(f"Corrupted or invalid JSON in {bundle_json_file}: {exc}") from exc

    if not isinstance(raw_metadata, Mapping):
        raise BundleError(f"Bundle metadata in {bundle_json_file} must be an object")

    schema_version = raw_metadata.get("schema_version")
    if schema_version != BUNDLE_SCHEMA_VERSION:
        raise BundleError(f"Unsupported bundle schema_version: {schema_version} (expected {BUNDLE_SCHEMA_VERSION})")

    # Extract BundleSpecification fields
    try:
        if "specification" in raw_metadata and isinstance(raw_metadata["specification"], Mapping):
            spec_data = dict(raw_metadata["specification"])
        else:
            spec_keys = {"schema_version", "bundle_id", "model_version", "preprocessing_hash", "decision_rule", "class_0", "class_1"}
            missing = [k for k in spec_keys if k not in raw_metadata]
            if missing:
                raise BundleError(f"Missing required specification fields in bundle.json: {missing}")
            spec_data = {k: raw_metadata[k] for k in spec_keys}

        if isinstance(spec_data.get("decision_rule"), DecisionRule):
            spec_data["decision_rule"] = spec_data["decision_rule"].to_dict()
        spec = BundleSpecification.from_dict(spec_data)
    except ContractError as exc:
        raise BundleError(f"Invalid bundle specification contract: {exc}") from exc

    # Validate components mapping
    components_raw = raw_metadata.get("components")
    if not isinstance(components_raw, Mapping) or not components_raw:
        raise BundleError("Bundle metadata must define a nonempty 'components' mapping")

    resolved_components: dict[str, Path] = {}
    component_hashes: dict[str, str] = {}

    for rel_path, comp_info in components_raw.items():
        norm_path = _validate_relative_component_path(rel_path)

        if isinstance(comp_info, str):
            expected_sha = comp_info.strip().lower()
        elif isinstance(comp_info, Mapping) and "sha256" in comp_info:
            expected_sha = str(comp_info["sha256"]).strip().lower()
        else:
            raise BundleError(f"Component '{norm_path}' specification must define a SHA-256 digest")

        if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise BundleError(f"Invalid SHA-256 digest for component '{norm_path}': {expected_sha}")

        comp_file = (path / norm_path).resolve()
        # Path traversal guard
        if not comp_file.is_relative_to(path.resolve()):
            raise BundleError(f"Component path escapes bundle directory: {rel_path}")
        if not comp_file.is_file():
            raise BundleError(f"Missing bundle component file: {norm_path} (expected at {comp_file})")

        actual_sha = file_digest(comp_file)
        if actual_sha != expected_sha:
            raise BundleError(
                f"Checksum mismatch for component '{norm_path}': expected {expected_sha}, calculated {actual_sha}"
            )

        resolved_components[norm_path] = comp_file
        component_hashes[norm_path] = actual_sha

    return ValidatedBundle(
        spec=spec,
        bundle_dir=path.resolve(),
        components=resolved_components,
        component_hashes=component_hashes,
        raw_metadata=dict(raw_metadata),
    )
