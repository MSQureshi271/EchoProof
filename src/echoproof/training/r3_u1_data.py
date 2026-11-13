"""R3-U1 data: the published R2-U1 WavLM embeddings, joined to labels, recordings and validation roles.

Only development rows are read (train crops and validation chunks). The reader refuses the test partition without an explicit opt-in, and this module
never passes it. `check_alignment` proves that the cache rows are exactly the Protocol B chunks and training recordings the split, the chunk manifest and
the cohort define; it computes identities and counts only, never a model score.
"""
from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from echoproof.artifacts import file_digest

ROLES = ("selection", "calibration", "operating")
POOLINGS = ("mean", "mean_std")
MANIFEST_NAME = "r2_u1_manifest.json"
MEMO_SCHEMA = 1


class AlignmentError(ValueError):
    """The cache, the split, the chunk manifest or the cohort disagree."""


@dataclass
class Split:
    """Rows of one partition or role, aligned with `X` row by row."""
    item_ids: list
    recording_ids: list
    group_ids: list
    components: list
    labels: np.ndarray
    X: np.ndarray
    recording_weights: np.ndarray = field(default=None)

    def __len__(self):
        return len(self.item_ids)

    def recordings(self) -> dict:
        """recording_id -> (label, component, group_id), checked to be consistent across the recording's rows."""
        out = {}
        for rid, label, comp, group in zip(self.recording_ids, self.labels.tolist(), self.components, self.group_ids):
            known = out.setdefault(rid, (label, comp, group))
            if known != (label, comp, group):
                raise AlignmentError(f"inconsistent rows for {rid}")
        return out


@dataclass
class EmbeddingData:
    pooling: str
    dim: int
    train: Split
    roles: dict
    identity: dict


def _split(rows, matrix, positions):
    labels = np.array([rows[i]["label"] for i in positions], dtype=np.int64)
    split = Split([rows[i]["item_id"] for i in positions], [rows[i]["recording_id"] for i in positions],
                  [rows[i]["group_id"] for i in positions], [rows[i]["component"] for i in positions], labels,
                  np.ascontiguousarray(matrix[positions], dtype=np.float32))
    return split


def crop_weights(recording_ids) -> np.ndarray:
    """Weight each crop by 1 / (crops of its recording), scaled so the mean weight is 1: every recording counts equally."""
    counts = collections.Counter(recording_ids)
    raw = np.array([1.0 / counts[r] for r in recording_ids], dtype=np.float64)
    return (raw / raw.mean()).astype(np.float32)


def build_data(rows, matrix, pooling: str, identity: dict) -> EmbeddingData:
    if pooling not in POOLINGS:
        raise AlignmentError(f"unknown pooling {pooling!r}")
    if len(rows) != len(matrix) or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise AlignmentError("rows and embedding matrix disagree, or the matrix has non-finite values")
    if len({r["item_id"] for r in rows}) != len(rows):
        raise AlignmentError("duplicate item ids in the cache rows")
    if any(r["scope"] != "development" or r["partition"] not in ("train", "validation") for r in rows):
        raise AlignmentError("the cache rows contain a non-development scope or the test partition")
    if any(r["component"] == "chatterbox" for r in rows):
        raise AlignmentError("ChatterboxTTS rows are not allowed")
    if any(r["label"] not in (0, 1) for r in rows):
        raise AlignmentError("labels must be 0 or 1")
    train_pos = [i for i, r in enumerate(rows) if r["partition"] == "train"]
    if any(rows[i]["kind"] != "crop" for i in train_pos):
        raise AlignmentError("training rows must be crops")
    train = _split(rows, matrix, train_pos)
    train.recording_weights = crop_weights(train.recording_ids)
    roles = {}
    for role in ROLES:
        pos = [i for i, r in enumerate(rows) if r["partition"] == "validation" and r["role"] == role]
        if any(rows[i]["kind"] != "chunk" for i in pos):
            raise AlignmentError("validation rows must be chunks")
        roles[role] = _split(rows, matrix, pos)
    if sum(len(s) for s in roles.values()) != sum(1 for r in rows if r["partition"] == "validation"):
        raise AlignmentError("validation rows without a known role")
    for name, split in (("train", train), *roles.items()):
        if len(split) == 0 or set(split.labels.tolist()) != {0, 1}:
            raise AlignmentError(f"{name} must be non-empty and contain both classes")
        split.recordings()
    speakers = [set(s.group_ids) for s in (train, *roles.values())]
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            if speakers[i] & speakers[j]:
                raise AlignmentError("a speaker appears in two partitions or roles")
    return EmbeddingData(pooling, int(matrix.shape[1]), train, roles, identity)


def _memo_paths(memo_dir, manifest_sha, pooling):
    stem = f"r3_u1_embeddings_{pooling}_{manifest_sha[:12]}"
    return Path(memo_dir) / f"{stem}.npz", Path(memo_dir) / f"{stem}.json"


def load_embeddings(cache_dir, pooling: str, *, memo_dir=None) -> EmbeddingData:
    """Read development embeddings once. `read_embeddings` re-verifies every pack on each call (minutes), so a verified copy can be kept in `memo_dir`.

    The memo is reused only if the manifest hash, the index row order and the stored array hash all still match; otherwise it is rebuilt.
    """
    from echoproof.features import r2_u1
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / MANIFEST_NAME
    manifest_sha = file_digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {"manifest_sha256": manifest_sha, "plan_sha256": manifest["plan_sha256"], "pooling": pooling,
                "encoder_revision": manifest["identity"]["revision"], "encoder_weights_sha256": manifest["identity"]["weights_sha256"],
                "preprocessing_hash": manifest["identity"]["preprocessing_hash"]}
    if memo_dir is not None:
        npz_path, side_path = _memo_paths(memo_dir, manifest_sha, pooling)
        if npz_path.exists() and side_path.exists():
            side = json.loads(side_path.read_text(encoding="utf-8"))
            index_info = manifest["indices"]["development"]
            index_rows = json.loads((cache_dir / index_info["file"]).read_text(encoding="utf-8"))["rows"]
            if (side.get("schema") == MEMO_SCHEMA and side.get("manifest_sha256") == manifest_sha and side.get("pooling") == pooling
                    and file_digest(cache_dir / index_info["file"]) == index_info["sha256"] and file_digest(npz_path) == side.get("npz_sha256")):
                with np.load(npz_path, allow_pickle=False) as memo:
                    ids = memo["item_ids"].tolist()
                    if ids == [r["item_id"] for r in index_rows]:
                        return build_data(index_rows, memo["matrix"], pooling, {**identity, "source": "verified memo"})
    rows, matrix = r2_u1.read_embeddings(cache_dir, partition=None, pooling=pooling, include_test_genuine=False)
    data = build_data(rows, matrix, pooling, {**identity, "source": "read_embeddings (fully verified)"})
    if memo_dir is not None:
        memo_dir = Path(memo_dir)
        memo_dir.mkdir(parents=True, exist_ok=True)
        npz_path, side_path = _memo_paths(memo_dir, manifest_sha, pooling)
        if not npz_path.exists() and not side_path.exists():
            tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
            np.savez(tmp, matrix=matrix, item_ids=np.array([r["item_id"] for r in rows], dtype=str))
            tmp.replace(npz_path)
            side_path.write_text(json.dumps({"schema": MEMO_SCHEMA, "manifest_sha256": manifest_sha, "pooling": pooling,
                                             "npz_sha256": file_digest(npz_path), "rows": len(rows)}, indent=2), encoding="utf-8")
    return data


def check_alignment(root, data: EmbeddingData) -> dict:
    """Prove the cache rows are exactly the Protocol B rows the split, chunk manifest and cohort define. Identity and counts only."""
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data import splits
    from echoproof.data.cohort import read_table
    root = Path(root)
    cohort = {r["recording_id"]: r for r in read_table(str(root / "data/manifests/cohort_v1.parquet"))}
    assignments = {a.recording_id: a for a in splits.load_assignments(root / "data/splits/protocol_B_v1.parquet", sealed=False)}
    chunks = read_parquet(str(root / "data/processed/chunks_v1.parquet"), Chunk)
    protocol = next((c.protocol_id for c in chunks if c.chunk_id.startswith("chunk:B:")), None)
    if protocol is None:
        raise AlignmentError("no Protocol B chunks in the chunk manifest")
    manifest_chunks = {c.chunk_id: c for c in chunks if c.protocol_id == protocol and c.partition == "validation"}
    problems = collections.Counter()
    report = {"roles": {}}
    cached_chunk_ids = set()
    for role, split in data.roles.items():
        for cid, rid, gid, comp, label in zip(split.item_ids, split.recording_ids, split.group_ids, split.components, split.labels.tolist()):
            cached_chunk_ids.add(cid)
            chunk, assign, row = manifest_chunks.get(cid), assignments.get(rid), cohort.get(rid)
            if chunk is None or assign is None or row is None:
                problems["unknown chunk, recording or assignment"] += 1
                continue
            if chunk.recording_id != rid:
                problems["recording id"] += 1
            if chunk.group_id != gid or assign.group_id != gid:
                problems["group id"] += 1
            if assign.validation_role != role or assign.partition != "validation":
                problems["validation role"] += 1
            if int(row["label"]) != label or row["component"] != comp:
                problems["label or component"] += 1
        report["roles"][role] = {"chunks": len(split), "recordings": len(split.recordings())}
    if cached_chunk_ids != set(manifest_chunks):
        problems["chunk coverage differs from the Protocol B validation manifest"] += 1
    train_recordings = {}
    for rid, label, comp, gid in zip(data.train.recording_ids, data.train.labels.tolist(), data.train.components, data.train.group_ids):
        assign, row = assignments.get(rid), cohort.get(rid)
        if assign is None or row is None or assign.partition != "train":
            problems["training recording not assigned to train"] += 1
            continue
        if int(row["label"]) != label or row["component"] != comp or assign.group_id != gid:
            problems["training label, component or group"] += 1
        train_recordings[rid] = train_recordings.get(rid, 0) + 1
    expected_train = {rid for rid, a in assignments.items() if a.partition == "train"}
    if set(train_recordings) != expected_train:
        problems["training recordings differ from the split"] += 1
    for role in ROLES:
        want = {rid for rid, a in assignments.items() if a.partition == "validation" and a.validation_role == role}
        if set(data.roles[role].recordings()) != want:
            problems[f"{role} recordings differ from the split"] += 1
    if problems:
        raise AlignmentError("alignment failed: " + "; ".join(f"{k} x{v}" for k, v in sorted(problems.items())))
    report.update({"train": {"crops": len(data.train), "recordings": len(train_recordings),
                             "crops_per_recording_min": min(train_recordings.values()), "crops_per_recording_max": max(train_recordings.values()),
                             "crop_labels": {str(k): int(v) for k, v in zip(*np.unique(data.train.labels, return_counts=True))}},
                   "validation_chunks_match_manifest": len(manifest_chunks), "passed": True, "model_scores_used": False})
    return report
