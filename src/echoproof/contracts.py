"""Versioned metadata and in-memory model contracts for EchoProof.

Collection-level validation is explicit: individual objects cannot establish
uniqueness or referential integrity without their parent collection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, ClassVar, Mapping, Protocol, Sequence, runtime_checkable


class ContractError(ValueError):
    """Invalid contract data or inconsistent relationships."""


def _text(value: Any, name: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{name} must be a nonempty string")


def _identity(value: Any, name: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    _text(value, name)
    if value.lower() in {"unknown", "none", "null"} or ":" not in value:
        raise ContractError(f"{name} must be a namespaced identity or null")
    namespace, identifier = value.split(":", 1)
    if not namespace.strip() or not identifier.strip() or identifier.strip().lower() in {"unknown", "none", "null"}:
        raise ContractError(f"{name} must be a namespaced identity or null")


def _hash(value: Any, name: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{name} must be a lowercase SHA-256 hex digest")


def _number(value: Any, name: str, minimum: float = 0, maximum: float | None = None) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ContractError(f"{name} must be finite and in the allowed range")


def _int(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")


def _strings(values: Any, name: str) -> None:
    if not isinstance(values, list):
        raise ContractError(f"{name} must be a list of strings")
    for value in values:
        _text(value, name)


def _relative(value: Any) -> None:
    _text(value, "original_path")
    path = PurePosixPath(value)
    if "\\" in value or path.is_absolute() or PureWindowsPath(value).is_absolute() or PureWindowsPath(value).drive or any(p in ("..", ".") for p in value.split("/")) or any(not p for p in value.split("/")):
        raise ContractError("original_path must be a normalized relative POSIX path")


class Metadata:
    schema_version: int
    schema_name: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        if not isinstance(value, Mapping):
            raise ContractError("metadata must be an object")
        expected = {field.name for field in fields(cls)}
        if set(value) != expected:
            raise ContractError(f"{cls.__name__}: unexpected or missing fields")
        data = dict(value)
        if cls in (PredictionRow, BundleSpecification):
            data["decision_rule"] = DecisionRule.from_dict(data["decision_rule"]) if data["decision_rule"] is not None else None
        try:
            return cls(**data)
        except TypeError as exc:
            raise ContractError(str(exc)) from exc

    @classmethod
    def from_json(cls, value: str):
        try:
            return cls.from_dict(json.loads(value))
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"Invalid JSON: {exc}") from exc


def _version(value: Any) -> None:
    if type(value) is not int or value != 1:
        raise ContractError("schema_version must be 1")


@dataclass(frozen=True)
class Recording(Metadata):
    schema_name: ClassVar[str] = "recording"
    schema_version: int
    recording_id: str
    original_path: str
    dataset: str
    dataset_revision: str
    label: int | None
    label_verified: bool
    method: str | None
    method_verified: bool
    generator_family: str | None
    generator_version: str | None
    language: str | None
    source_utterance_id: str | None
    source_speaker_id: str | None
    target_voice_id: str | None
    reference_id: str | None
    transcript_hash: str | None
    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    codec: str | None
    sha256: str
    quality_flags: list[str]
    retained: bool
    exclusion_reason: str | None
    license_ref: str | None

    def __post_init__(self):
        _version(self.schema_version)
        _identity(self.recording_id, "recording_id")
        _relative(self.original_path)
        for key in ("dataset", "dataset_revision"):
            _text(getattr(self, key), key)
        for key in ("source_utterance_id", "source_speaker_id", "target_voice_id", "reference_id"):
            _identity(getattr(self, key), key, True)
        for key in ("method", "generator_family", "generator_version", "language", "codec", "exclusion_reason", "license_ref"):
            _text(getattr(self, key), key, True)
        if self.label is not None and (type(self.label) is not int or self.label not in (0, 1)):
            raise ContractError("label must be 0, 1, or null")
        if type(self.label_verified) is not bool or type(self.method_verified) is not bool:
            raise ContractError("verification flags must be boolean")
        if (self.label_verified and self.label is None) or (self.method_verified and self.method is None):
            raise ContractError("verified label and method require corresponding values")
        if self.duration_seconds is not None:
            _number(self.duration_seconds, "duration_seconds")
        for key in ("sample_rate", "channels"):
            if getattr(self, key) is not None:
                _int(getattr(self, key), key, 1)
        _hash(self.sha256, "sha256")
        _hash(self.transcript_hash, "transcript_hash", True)
        _strings(self.quality_flags, "quality_flags")
        if type(self.retained) is not bool:
            raise ContractError("retained must be boolean")
        if not self.retained and self.exclusion_reason is None:
            raise ContractError("excluded recording requires an exclusion_reason")
        if self.retained and self.exclusion_reason is not None:
            raise ContractError("retained recording cannot have an exclusion_reason")

    def validate_supervised(self) -> None:
        if not self.retained or self.label is None or self.method is None or not self.label_verified or not self.method_verified:
            raise ContractError("supervised recordings require retention and verified label and method")


@dataclass(frozen=True)
class SplitAssignment(Metadata):
    schema_name: ClassVar[str] = "split_assignment"
    schema_version: int
    protocol_id: str
    recording_id: str
    group_id: str
    partition: str
    validation_role: str | None
    family_policy: str
    split_version: str
    split_hash: str

    def __post_init__(self):
        _version(self.schema_version)
        for key in ("protocol_id", "recording_id", "group_id"):
            _identity(getattr(self, key), key)
        if self.partition not in ("train", "validation", "test"):
            raise ContractError("partition must be train, validation, or test")
        _text(self.validation_role, "validation_role", True)
        if self.partition != "validation" and self.validation_role is not None:
            raise ContractError("validation_role is only allowed for validation")
        for key in ("family_policy", "split_version"):
            _text(getattr(self, key), key)
        _hash(self.split_hash, "split_hash")


@dataclass(frozen=True)
class Chunk(Metadata):
    schema_name: ClassVar[str] = "chunk"
    schema_version: int
    chunk_id: str
    recording_id: str
    group_id: str
    protocol_id: str
    partition: str
    start_sample: int
    end_sample: int
    sample_rate: int
    start_seconds: float
    end_seconds: float
    valid_samples: int
    input_samples: int
    preprocessing_hash: str

    def __post_init__(self):
        _version(self.schema_version)
        for key in ("chunk_id", "recording_id", "group_id", "protocol_id"):
            _identity(getattr(self, key), key)
        if self.partition not in ("train", "validation", "test"):
            raise ContractError("invalid chunk partition")
        _int(self.start_sample, "start_sample")
        for key in ("end_sample", "sample_rate", "valid_samples", "input_samples"):
            _int(getattr(self, key), key, 1)
        for key in ("start_seconds", "end_seconds"):
            _number(getattr(self, key), key)
        if self.end_sample <= self.start_sample or self.end_sample - self.start_sample != self.valid_samples or self.valid_samples > self.input_samples:
            raise ContractError("chunk interval, valid length, and input length disagree")
        if not math.isclose(self.start_seconds, self.start_sample / self.sample_rate, abs_tol=1e-8, rel_tol=1e-9) or not math.isclose(self.end_seconds, self.end_sample / self.sample_rate, abs_tol=1e-8, rel_tol=1e-9):
            raise ContractError("chunk sample and second offsets disagree")
        _hash(self.preprocessing_hash, "preprocessing_hash")


@dataclass(frozen=True)
class DecisionRule(Metadata):
    schema_name: ClassVar[str] = "decision_rule"
    schema_version: int
    kind: str
    threshold: float | None

    def __post_init__(self):
        _version(self.schema_version)
        if self.kind == "threshold":
            _number(self.threshold, "threshold", maximum=1)
        elif self.kind == "all_negative":
            if self.threshold is not None:
                raise ContractError("all_negative must have a null threshold")
        else:
            raise ContractError("invalid decision rule")

    def decide(self, score: float) -> int:
        _number(score, "score", maximum=1)
        return int(self.kind == "threshold" and score >= self.threshold)


@dataclass(frozen=True)
class BundleSpecification(Metadata):
    """Minimum validated reference for a future trusted inference bundle."""
    schema_name: ClassVar[str] = "bundle_specification"
    schema_version: int
    bundle_id: str
    model_version: str
    preprocessing_hash: str
    decision_rule: DecisionRule
    class_0: str
    class_1: str

    def __post_init__(self):
        _version(self.schema_version)
        _identity(self.bundle_id, "bundle_id")
        _text(self.model_version, "model_version")
        _hash(self.preprocessing_hash, "preprocessing_hash")
        if not isinstance(self.decision_rule, DecisionRule):
            raise ContractError("bundle requires a validated decision rule")
        if (self.class_0, self.class_1) != ("genuine", "tts"):
            raise ContractError("bundle classes must map 0 to genuine and 1 to tts")


@dataclass(frozen=True)
class PredictionRow(Metadata):
    schema_name: ClassVar[str] = "prediction_row"
    schema_version: int
    run_id: str
    bundle_id: str
    protocol_id: str
    recording_id: str
    raw_score: float | None
    calibrated_score: float | None
    decision_rule: DecisionRule | None
    prediction: int | None
    status: str
    duration_seconds: float | None
    chunk_count: int

    def __post_init__(self):
        _version(self.schema_version)
        for key in ("run_id", "bundle_id", "protocol_id", "recording_id"):
            _identity(getattr(self, key), key)
        _text(self.status, "status")
        _int(self.chunk_count, "chunk_count")
        if self.duration_seconds is not None:
            _number(self.duration_seconds, "duration_seconds")
        if self.status == "success":
            _number(self.raw_score, "raw_score", maximum=1)
            if self.calibrated_score is not None:
                _number(self.calibrated_score, "calibrated_score", maximum=1)
            if not isinstance(self.decision_rule, DecisionRule) or type(self.prediction) is not int or self.prediction not in (0, 1):
                raise ContractError("success requires a decision rule and binary prediction")
            score = self.calibrated_score if self.calibrated_score is not None else self.raw_score
            if self.prediction != self.decision_rule.decide(score) or self.chunk_count < 1:
                raise ContractError("prediction disagrees with decision rule or no chunks were scored")
        elif any(v is not None for v in (self.raw_score, self.calibrated_score, self.prediction, self.decision_rule)):
            raise ContractError("failed prediction cannot have scores, rule, or prediction")


@dataclass(frozen=True)
class InferenceResult(Metadata):
    schema_name: ClassVar[str] = "inference_result"
    schema_version: int
    status: str
    assessment: str | None
    score: float | None
    score_type: str | None
    warnings: list[str]
    model_version: str | None
    decoded_duration_seconds: float | None
    covered_duration_seconds: float | None
    timings_seconds: dict[str, float]
    failure_reason: str | None

    def __post_init__(self):
        _version(self.schema_version)
        _text(self.status, "status")
        _strings(self.warnings, "warnings")
        for key in ("assessment", "score_type", "model_version", "failure_reason"):
            _text(getattr(self, key), key, True)
        for key in ("decoded_duration_seconds", "covered_duration_seconds"):
            if getattr(self, key) is not None:
                _number(getattr(self, key), key)
        if self.covered_duration_seconds is not None and (self.decoded_duration_seconds is None or self.covered_duration_seconds > self.decoded_duration_seconds + 1e-8):
            raise ContractError("coverage exceeds decoded duration")
        if not isinstance(self.timings_seconds, dict):
            raise ContractError("timings_seconds must be an object")
        for key, value in self.timings_seconds.items():
            _text(key, "timing name")
            _number(value, f"timings_seconds.{key}")
        if self.status == "success":
            _number(self.score, "score", maximum=1)
            if any(v is None for v in (self.assessment, self.score_type, self.model_version, self.decoded_duration_seconds, self.covered_duration_seconds)) or self.failure_reason is not None:
                raise ContractError("successful inference requires assessment, score type, model and coverage")
        elif self.score is not None or self.assessment is not None or self.score_type is not None or self.failure_reason is None:
            raise ContractError("failed inference requires a reason and no assessment or score")


def validate_relationships(recordings: Sequence[Recording], assignments: Sequence[SplitAssignment], chunks: Sequence[Chunk], *, supervised: bool = False) -> None:
    records = {}
    for record in recordings:
        if record.recording_id in records:
            raise ContractError("duplicate recording_id")
        records[record.recording_id] = record
    splits = {}
    groups = {}
    protocols = {}
    for split in assignments:
        key = (split.protocol_id, split.recording_id)
        if split.recording_id not in records or key in splits:
            raise ContractError("orphan or duplicate split assignment")
        if supervised:
            records[split.recording_id].validate_supervised()
        group_key = (split.protocol_id, split.group_id)
        group_partition = (split.partition, split.validation_role)
        if group_key in groups and groups[group_key] != group_partition:
            raise ContractError("a group cannot occur in multiple partitions or validation roles")
        groups[group_key] = group_partition
        version = (split.split_version, split.split_hash, split.family_policy)
        if split.protocol_id in protocols and protocols[split.protocol_id] != version:
            raise ContractError("split metadata must be consistent within a protocol")
        protocols[split.protocol_id] = version
        splits[key] = split
    chunk_ids = set()
    for chunk in chunks:
        if chunk.chunk_id in chunk_ids:
            raise ContractError("duplicate chunk_id")
        chunk_ids.add(chunk.chunk_id)
        split = splits.get((chunk.protocol_id, chunk.recording_id))
        if split is None or split.group_id != chunk.group_id or split.partition != chunk.partition:
            raise ContractError("orphan chunk or split/group mismatch")


@dataclass(frozen=True)
class FeatureBatch:
    """In-memory batch; arrays are intentionally excluded from metadata serialization."""
    values: Any
    chunk_ids: Sequence[str]
    recording_ids: Sequence[str]
    labels: Any | None
    lengths: Any | None
    mask: Any | None
    feature_schema_hash: str

    def validate(self) -> None:
        import numpy as np

        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        x = array(self.values)
        if x.ndim not in (2, 3) or x.shape[0] == 0 or any(d == 0 for d in x.shape) or x.dtype.kind not in "fiu":
            raise ContractError("values must be nonempty numeric [batch, features] or [batch, time, features]")
        n = x.shape[0]
        if len(self.chunk_ids) != n or len(self.recording_ids) != n or len(set(self.chunk_ids)) != n:
            raise ContractError("batch IDs must align and chunk IDs must be unique")
        for name, values in (("chunk_id", self.chunk_ids), ("recording_id", self.recording_ids)):
            for value in values:
                _identity(value, name)
        _hash(self.feature_schema_hash, "feature_schema_hash")
        if self.labels is not None:
            labels = array(self.labels)
            if labels.shape != (n,) or labels.dtype.kind not in "iu" or not np.isin(labels, [0, 1]).all():
                raise ContractError("labels must be a binary vector aligned to the batch")
        if x.ndim == 2:
            if self.lengths is not None or self.mask is not None or not np.isfinite(x).all():
                raise ContractError("fixed features require finite values and no temporal mask or lengths")
            return
        if self.lengths is None or self.mask is None:
            raise ContractError("temporal features require lengths and masks")
        lengths, mask = array(self.lengths), array(self.mask)
        if lengths.shape != (n,) or lengths.dtype.kind not in "iu" or (lengths < 1).any() or (lengths > x.shape[1]).any():
            raise ContractError("invalid temporal lengths")
        expected = np.arange(x.shape[1])[None, :] < lengths[:, None]
        if mask.shape != expected.shape or mask.dtype.kind != "b" or not np.array_equal(mask, expected):
            raise ContractError("mask must match right-padded lengths")
        if not np.isfinite(x[mask]).all():
            raise ContractError("valid feature values must be finite")


@dataclass(frozen=True)
class ChunkScore:
    chunk_id: str
    tts_probability: float

    def __post_init__(self):
        _identity(self.chunk_id, "chunk_id")
        _number(self.tts_probability, "tts_probability", maximum=1)


def validate_chunk_scores(expected_ids: Sequence[str], scores: Sequence[ChunkScore]) -> None:
    if len(set(expected_ids)) != len(expected_ids) or [score.chunk_id for score in scores] != list(expected_ids):
        raise ContractError("chunk scores must correspond to every requested chunk in order")


@runtime_checkable
class ModelAdapter(Protocol):
    """Feature-level protocol for the P1 software-probe fixtures (FeatureBatch in, ChunkScore rows out).

    Production inference does not use this. The inference contract is `echoproof.models.base.InferenceModelAdapter`
    (bundle root in, waveforms in, P(TTS) array out).
    """

    @property
    def metadata(self) -> Mapping[str, str]: ...
    def prepare_features(self, chunks: Any) -> FeatureBatch: ...
    def score_chunks(self, batch: FeatureBatch) -> Sequence[ChunkScore]: ...
    def save(self, destination: str) -> None: ...
    @classmethod
    def load(cls, source: str) -> ModelAdapter: ...


@runtime_checkable
class RecordingPredictor(Protocol):
    def predict_recording(self, audio_bytes: bytes, bundle: BundleSpecification) -> InferenceResult: ...


_SCHEMAS = {cls.schema_name: cls for cls in (Recording, SplitAssignment, Chunk, DecisionRule, BundleSpecification, PredictionRow, InferenceResult)}


def write_parquet(rows: Sequence[Metadata], destination: str) -> None:
    """Write a typed envelope; each JSON payload retains its schema and nulls."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not rows:
        raise ContractError("cannot infer schema from an empty collection")
    schema_name = rows[0].schema_name
    if schema_name not in _SCHEMAS or any(type(row) is not type(rows[0]) for row in rows):
        raise ContractError("Parquet rows must share a supported metadata type")
    table = pa.table({"schema_name": [schema_name] * len(rows), "schema_version": [1] * len(rows),
                      "payload_json": [row.to_json() for row in rows]})
    pq.write_table(table, destination)


def read_parquet(source: str, expected_type: type[Metadata]) -> list[Metadata]:
    import pyarrow.parquet as pq
    if expected_type not in _SCHEMAS.values():
        raise ContractError("unsupported metadata type")
    table = pq.read_table(source)
    if set(table.column_names) != {"schema_name", "schema_version", "payload_json"}:
        raise ContractError("invalid Parquet envelope")
    rows = []
    for row in table.to_pylist():
        if row["schema_name"] != expected_type.schema_name or type(row["schema_version"]) is not int or row["schema_version"] != 1:
            raise ContractError("unexpected Parquet schema or version")
        rows.append(expected_type.from_json(row["payload_json"]))
    return rows
