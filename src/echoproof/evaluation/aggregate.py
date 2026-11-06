"""Recording-level aggregation for R1-N1 evaluation.

Each recording receives exactly one score: the arithmetic mean of its chunk-level
TTS probabilities. Chunks are internal processing units; the recording is the
unit of evaluation.

Positive class: 1 (TTS).
Higher scores indicate a higher probability of TTS.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Sequence

from echoproof.contracts import (
    Chunk,
    ChunkScore,
    ContractError,
    Recording,
    SplitAssignment,
    validate_chunk_scores,
)


def _check_identity(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    if value.lower() in {"unknown", "none", "null"} or ":" not in value:
        raise ContractError(f"{name} must be a namespaced identity")
    namespace, identifier = value.split(":", 1)
    if (
        not namespace.strip()
        or namespace.strip().lower() in {"unknown", "none", "null"}
        or not identifier.strip()
        or identifier.strip().lower() in {"unknown", "none", "null"}
    ):
        raise ContractError(f"{name} must be a namespaced identity")


def _check_label(label: Any, index: int | None = None) -> int:
    ctx = f" at index {index}" if index is not None else ""
    if isinstance(label, bool) or not isinstance(label, Integral):
        raise ContractError(f"Invalid label{ctx}: expected 0 or 1, received {label!r}")
    val = int(label)
    if val not in (0, 1):
        raise ContractError(f"Invalid label{ctx}: expected 0 or 1, received {label!r}")
    return val


def _check_score(score: Any, index: int | None = None) -> float:
    ctx = f" at index {index}" if index is not None else ""
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ContractError(f"Invalid score{ctx}: expected numeric probability, received {score!r}")
    val = float(score)
    if not math.isfinite(val):
        raise ContractError(f"Non-finite score{ctx}: {score!r}")
    if not 0.0 <= val <= 1.0:
        raise ContractError(f"Score outside [0, 1]{ctx}: {score}")
    return val


@dataclass(frozen=True)
class RecordingScores:
    """Immutable collection of recording-level aggregated scores and metadata."""

    recording_ids: tuple[str, ...]
    labels: tuple[int, ...]
    scores: tuple[float, ...]
    chunk_counts: tuple[int, ...]
    group_ids: tuple[str | None, ...]
    partitions: tuple[str | None, ...]
    validation_roles: tuple[str | None, ...]

    def __post_init__(self) -> None:
        n = len(self.recording_ids)
        if not (
            len(self.labels) == n
            and len(self.scores) == n
            and len(self.chunk_counts) == n
            and len(self.group_ids) == n
            and len(self.partitions) == n
            and len(self.validation_roles) == n
        ):
            raise ContractError("All RecordingScores tuple fields must have equal lengths")
        for rid in self.recording_ids:
            _check_identity(rid, "recording_id")
        for lbl in self.labels:
            _check_label(lbl)
        for sc in self.scores:
            _check_score(sc)
        for count in self.chunk_counts:
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
                raise ContractError(f"chunk_count must be integer >= 1, received {count!r}")

    def subset(
        self,
        partition: str | None = None,
        validation_role: str | None = None,
    ) -> RecordingScores:
        """Return a new RecordingScores containing only recordings matching criteria."""
        keep_indices: list[int] = []
        for i in range(len(self.recording_ids)):
            if partition is not None and self.partitions[i] != partition:
                continue
            if validation_role is not None and self.validation_roles[i] != validation_role:
                continue
            keep_indices.append(i)

        return RecordingScores(
            recording_ids=tuple(self.recording_ids[i] for i in keep_indices),
            labels=tuple(self.labels[i] for i in keep_indices),
            scores=tuple(self.scores[i] for i in keep_indices),
            chunk_counts=tuple(self.chunk_counts[i] for i in keep_indices),
            group_ids=tuple(self.group_ids[i] for i in keep_indices),
            partitions=tuple(self.partitions[i] for i in keep_indices),
            validation_roles=tuple(self.validation_roles[i] for i in keep_indices),
        )


def aggregate_recordings(
    chunk_ids: Sequence[str],
    recording_ids: Sequence[str],
    labels: Sequence[int],
    chunk_scores: Sequence[float],
    *,
    expected_recording_ids: Sequence[str] | None = None,
    group_ids: Sequence[str | None] | None = None,
    partitions: Sequence[str | None] | None = None,
    validation_roles: Sequence[str | None] | None = None,
) -> RecordingScores:
    """Aggregate chunk-level predictions into recording-level predictions via arithmetic mean.

    Args:
        chunk_ids: Unique namespaced chunk IDs.
        recording_ids: Namespaced recording ID for each chunk.
        labels: Ground-truth binary label for each chunk (0 or 1).
        chunk_scores: Predicted TTS probability for each chunk in [0, 1].
        expected_recording_ids: Optional set/sequence of expected recording IDs.
            Any missing or unexpected recording raises ContractError.
        group_ids: Optional recording-aligned group IDs.
        partitions: Optional recording-aligned partition names.
        validation_roles: Optional recording-aligned validation roles.

    Returns:
        RecordingScores with first-seen recording order.

    Raises:
        ContractError: On any invalid, non-finite, out-of-range, duplicate or misaligned input.
    """
    n = len(chunk_ids)
    if not (len(recording_ids) == n and len(labels) == n and len(chunk_scores) == n):
        raise ContractError("chunk_ids, recording_ids, labels, and chunk_scores must have equal lengths")

    if n == 0:
        raise ContractError("Cannot aggregate an empty prediction set")

    seen_chunk_ids: set[str] = set()
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for index, (cid, rid, lbl, score) in enumerate(zip(chunk_ids, recording_ids, labels, chunk_scores)):
        _check_identity(cid, f"chunk_id at index {index}")
        if cid in seen_chunk_ids:
            raise ContractError(f"Duplicate chunk_id at index {index}: {cid!r}")
        seen_chunk_ids.add(cid)

        _check_identity(rid, f"recording_id at index {index}")
        label_val = _check_label(lbl, index)
        score_val = _check_score(score, index)

        if rid not in grouped:
            grouped[rid] = {
                "label": label_val,
                "scores": [score_val],
            }
        else:
            rec = grouped[rid]
            if rec["label"] != label_val:
                raise ContractError(f"Conflicting ground-truth labels for recording {rid!r}")
            rec["scores"].append(score_val)

    if expected_recording_ids is not None:
        expected_set = set(expected_recording_ids)
        for erid in expected_recording_ids:
            _check_identity(erid, "expected_recording_id")
        actual_set = set(grouped.keys())
        missing = expected_set - actual_set
        unexpected = actual_set - expected_set
        if missing or unexpected:
            raise ContractError(
                f"Recording ID mismatch with expected set: missing {sorted(missing)}; unexpected {sorted(unexpected)}"
            )

    rec_ids = tuple(grouped.keys())
    rec_labels = tuple(grouped[rid]["label"] for rid in rec_ids)
    rec_scores = tuple(math.fsum(grouped[rid]["scores"]) / len(grouped[rid]["scores"]) for rid in rec_ids)
    rec_chunk_counts = tuple(len(grouped[rid]["scores"]) for rid in rec_ids)

    if group_ids is not None and len(group_ids) != len(rec_ids):
        raise ContractError("group_ids length does not match recording count")
    if partitions is not None and len(partitions) != len(rec_ids):
        raise ContractError("partitions length does not match recording count")
    if validation_roles is not None and len(validation_roles) != len(rec_ids):
        raise ContractError("validation_roles length does not match recording count")

    final_group_ids = tuple(group_ids) if group_ids is not None else tuple(None for _ in rec_ids)
    final_partitions = tuple(partitions) if partitions is not None else tuple(None for _ in rec_ids)
    final_val_roles = tuple(validation_roles) if validation_roles is not None else tuple(None for _ in rec_ids)

    return RecordingScores(
        recording_ids=rec_ids,
        labels=rec_labels,
        scores=rec_scores,
        chunk_counts=rec_chunk_counts,
        group_ids=final_group_ids,
        partitions=final_partitions,
        validation_roles=final_val_roles,
    )


def aggregate_from_contracts(
    recordings: Sequence[Recording],
    chunks: Sequence[Chunk],
    scores: Sequence[ChunkScore],
    assignments: Sequence[SplitAssignment],
    *,
    protocol_id: str,
) -> RecordingScores:
    """Aggregate predictions from formal contract collections.

    Args:
        recordings: Collection of Recording metadata objects.
        chunks: Collection of Chunk metadata objects.
        scores: Sequence of ChunkScore objects matching chunks order.
        assignments: Collection of SplitAssignment metadata objects.
        protocol_id: Evaluated protocol identifier (e.g. 'protocol:A').

    Returns:
        RecordingScores with group, partition, and validation_role metadata populated.

    Raises:
        ContractError: If inputs violate contracts or consistency rules.
    """
    _check_identity(protocol_id, "protocol_id")
    expected_chunk_ids = [c.chunk_id for c in chunks]
    validate_chunk_scores(expected_chunk_ids, scores)

    rec_map: dict[str, Recording] = {}
    for r in recordings:
        if r.recording_id in rec_map:
            raise ContractError(f"Duplicate recording in recordings collection: {r.recording_id!r}")
        rec_map[r.recording_id] = r

    assignment_map: dict[tuple[str, str], SplitAssignment] = {}
    for a in assignments:
        key = (a.protocol_id, a.recording_id)
        if key in assignment_map:
            raise ContractError(f"Duplicate assignment for protocol and recording: {key!r}")
        assignment_map[key] = a

    chunk_ids: list[str] = []
    recording_ids: list[str] = []
    labels: list[int] = []
    chunk_scores: list[float] = []

    recording_meta: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for chunk, chunk_score in zip(chunks, scores):
        rid = chunk.recording_id
        if rid not in rec_map:
            raise ContractError(f"Chunk references unknown recording: {rid!r}")
        rec = rec_map[rid]
        rec.validate_supervised()
        assert rec.label is not None

        assign_key = (protocol_id, rid)
        if assign_key not in assignment_map:
            raise ContractError(f"Recording {rid!r} missing assignment for protocol {protocol_id!r}")
        assignment = assignment_map[assign_key]

        if chunk.group_id != assignment.group_id:
            raise ContractError(
                f"Chunk {chunk.chunk_id!r} group_id {chunk.group_id!r} disagrees with assignment {assignment.group_id!r}"
            )
        if chunk.partition != assignment.partition:
            raise ContractError(
                f"Chunk {chunk.chunk_id!r} partition {chunk.partition!r} disagrees with assignment {assignment.partition!r}"
            )

        if rid not in recording_meta:
            recording_meta[rid] = {
                "group_id": assignment.group_id,
                "partition": assignment.partition,
                "validation_role": assignment.validation_role,
            }
        else:
            meta = recording_meta[rid]
            if meta["group_id"] != assignment.group_id or meta["partition"] != assignment.partition:
                raise ContractError(f"Inconsistent group/partition within recording {rid!r}")

        chunk_ids.append(chunk.chunk_id)
        recording_ids.append(rid)
        labels.append(rec.label)
        chunk_scores.append(chunk_score.tts_probability)

    first_seen_rids = list(recording_meta.keys())
    group_ids = [recording_meta[rid]["group_id"] for rid in first_seen_rids]
    partitions = [recording_meta[rid]["partition"] for rid in first_seen_rids]
    val_roles = [recording_meta[rid]["validation_role"] for rid in first_seen_rids]

    return aggregate_recordings(
        chunk_ids=chunk_ids,
        recording_ids=recording_ids,
        labels=labels,
        chunk_scores=chunk_scores,
        group_ids=group_ids,
        partitions=partitions,
        validation_roles=val_roles,
    )

