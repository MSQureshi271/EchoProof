"""R4-U1a data: the Protocol A WavLM embeddings (B cache plus the R4-U0 gap cache), joined to labels, recordings and validation roles.

Same structure as `r3_u1_data` (frozen B source; it refuses Chatterbox and reads the B cache only). Only development rows are read: A train crops and A
validation chunks. `echoproof.features.r4_u0.read_embeddings_a` refuses the test partition and verifies every pack. `check_alignment_a` proves the rows are
exactly the A crops and chunks the split, the crop bank, the chunk manifest and the cohort define; it computes identities and counts only, never a score.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from echoproof.training.r3_u1_data import AlignmentError, EmbeddingData, Split, _split, crop_weights  # noqa: F401

ROLES = ("selection", "calibration", "operating")
COMPONENTS = ("genuine", "melo", "vits", "xtts", "chatterbox")
POOLING = "mean_std"
EXPECTED_COUNTS = {"train_crops": 35782, "validation_chunks": 24412, "selection": 12331, "calibration": 5777, "operating": 6304}


def build_data_a(rows, matrix, pooling: str, identity: dict) -> EmbeddingData:
    if pooling != POOLING:
        raise AlignmentError("Protocol A trains the B-selected mean_std pooling only")
    if len(rows) != len(matrix) or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise AlignmentError("rows and embedding matrix disagree, or the matrix has non-finite values")
    if len({r["item_id"] for r in rows}) != len(rows):
        raise AlignmentError("duplicate item ids in the cache rows")
    if any(r["partition"] not in ("train", "validation") for r in rows):
        raise AlignmentError("the rows contain the test partition or an unknown partition")
    if any(r["component"] not in COMPONENTS for r in rows):
        raise AlignmentError("unknown component in the cache rows")
    if any(r["label"] not in (0, 1) or r["label"] != int(r["component"] != "genuine") for r in rows):
        raise AlignmentError("labels must be 0 for genuine and 1 for every generator")
    train_pos = [i for i, r in enumerate(rows) if r["partition"] == "train"]
    if any(rows[i]["kind"] != "crop" or rows[i]["role"] is not None for i in train_pos):
        raise AlignmentError("training rows must be crops without a validation role")
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
        if set(split.components) != set(COMPONENTS):
            raise AlignmentError(f"{name} must contain all five components, found {sorted(set(split.components))}")
        split.recordings()
    speakers = [set(s.group_ids) for s in (train, *roles.values())]
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            if speakers[i] & speakers[j]:
                raise AlignmentError("a speaker appears in two partitions or roles")
    return EmbeddingData(pooling, int(matrix.shape[1]), train, roles, identity)


def load_embeddings_a(root, *, pooling: str = POOLING, a_dir=None, b_dir=None) -> EmbeddingData:
    """Read and verify the A union once. `read_embeddings_a` checks every pack and key and refuses the test partition."""
    from echoproof.features import r4_u0
    root = Path(root)
    a_dir, b_dir = Path(a_dir or root / r4_u0.A_DIR), Path(b_dir or root / r4_u0.B_DIR)
    rows, matrix = r4_u0.read_embeddings_a(root, pooling=pooling, a_dir=a_dir, b_dir=b_dir)
    from echoproof.artifacts import file_digest
    am = json.loads((a_dir / r4_u0.MANIFEST_NAME).read_text(encoding="utf-8"))
    bm = json.loads((b_dir / "r2_u1_manifest.json").read_text(encoding="utf-8"))
    identity = {"a_manifest_sha256": file_digest(a_dir / r4_u0.MANIFEST_NAME), "a_plan_sha256": am["plan_sha256"],
                "b_manifest_sha256": file_digest(b_dir / "r2_u1_manifest.json"), "b_plan_sha256": bm["plan_sha256"], "pooling": pooling,
                "encoder_revision": am["identity"]["revision"], "encoder_weights_sha256": am["identity"]["weights_sha256"],
                "preprocessing_hash": am["identity"]["preprocessing_hash"], "identity_equal_to_b": am["identity_vs_b"]["equal"],
                "source": "read_embeddings_a (fully verified)"}
    return build_data_a(rows, matrix, pooling, identity)


def check_alignment_a(root, data: EmbeddingData, *, expected_counts=EXPECTED_COUNTS) -> dict:
    """Prove the rows are exactly the Protocol A crops and chunks the manifests define. Identity and counts only; no model score."""
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data import splits
    from echoproof.data.cohort import read_table
    root = Path(root)
    cohort = {r["recording_id"]: r for r in read_table(str(root / "data/manifests/cohort_v1.parquet"))}
    assignments = {a.recording_id: a for a in splits.load_assignments(root / "data/splits/protocol_A_v1.parquet", sealed=False)}
    frozen = json.loads((root / "reports/r1/split_report.json").read_text(encoding="utf-8"))["protocols"]["A"]["split_sha256"]
    if {a.split_hash for a in assignments.values()} != {frozen}:
        raise AlignmentError("the A split hash differs from the frozen value")
    chunks = read_parquet(str(root / "data/processed/chunks_v1.parquet"), Chunk)
    manifest_chunks = {c.chunk_id: c for c in chunks if c.protocol_id == "protocol:A" and c.partition == "validation"}
    bank = {c["crop_id"]: c for c in read_table(str(root / "data/features/r4_s1_A/crop_bank_A_v1.parquet"))}
    problems = collections.Counter()
    report = {"roles": {}}
    cached_chunks = set()
    for role, split in data.roles.items():
        for cid, rid, gid, comp, label in zip(split.item_ids, split.recording_ids, split.group_ids, split.components, split.labels.tolist()):
            cached_chunks.add(cid)
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
    if cached_chunks != set(manifest_chunks):
        problems["chunk coverage differs from the Protocol A validation manifest"] += 1
    per_recording = collections.Counter()
    cached_crops = set()
    for cid, rid, gid, comp, label in zip(data.train.item_ids, data.train.recording_ids, data.train.group_ids, data.train.components,
                                          data.train.labels.tolist()):
        cached_crops.add(cid)
        crop, assign, row = bank.get(cid), assignments.get(rid), cohort.get(rid)
        if crop is None or assign is None or row is None or assign.partition != "train":
            problems["training crop or recording not in the A crop bank / train split"] += 1
            continue
        if crop["recording_id"] != rid or crop["group_id"] != gid or assign.group_id != gid:
            problems["training recording or group"] += 1
        if int(row["label"]) != label or row["component"] != comp or crop["component"] != comp or crop["label"] != label:
            problems["training label or component"] += 1
        per_recording[rid] += 1
    if cached_crops != set(bank):
        problems["crop coverage differs from the A crop bank"] += 1
    if set(per_recording) != {rid for rid, a in assignments.items() if a.partition == "train" and cohort[rid]["retained"]}:
        problems["training recordings differ from the split"] += 1
    for role in ROLES:
        want = {rid for rid, a in assignments.items() if a.partition == "validation" and a.validation_role == role and cohort[rid]["retained"]}
        if set(data.roles[role].recordings()) != want:
            problems[f"{role} recordings differ from the split"] += 1
    counts = {"train_crops": len(data.train), "validation_chunks": sum(len(s) for s in data.roles.values()),
              **{role: len(data.roles[role]) for role in ROLES}}
    if expected_counts is not None and counts != expected_counts:
        problems[f"counts {counts} differ from the R4-S1 evidence {expected_counts}"] += 1
    if problems:
        raise AlignmentError("alignment failed: " + "; ".join(f"{k} x{v}" for k, v in sorted(problems.items())))
    report.update({"train": {"crops": len(data.train), "recordings": len(per_recording), "crops_per_recording_min": min(per_recording.values()),
                             "crops_per_recording_max": max(per_recording.values()),
                             "by_component": dict(sorted(collections.Counter(data.train.components).items()))},
                   "counts": counts, "split_sha256": frozen, "passed": True, "model_scores_used": False})
    return report


def parity_rows(root, data: EmbeddingData, n: int = 50, *, seed: int = 42) -> list:
    """Deterministic stratified draw of selection-role chunks (all five components, full and short), joined to the cohort rows the shard reader needs."""
    from echoproof.artifacts import digest
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data.cohort import read_table
    root = Path(root)
    cohort = {r["recording_id"]: r for r in read_table(str(root / "data/manifests/cohort_v1.parquet"))}
    chunks = {c.chunk_id: c for c in read_parquet(str(root / "data/processed/chunks_v1.parquet"), Chunk) if c.protocol_id == "protocol:A"}
    split = data.roles["selection"]
    strata = {}
    for position, (cid, rid, comp) in enumerate(zip(split.item_ids, split.recording_ids, split.components)):
        chunk, row = chunks[cid], cohort[rid]
        shard, member = row["canonical_path"].split("::", 1)[0], row["canonical_path"].split("::", 1)[1].replace(":", "__") + ".flac"
        item = {"item_id": cid, "recording_id": rid, "component": comp, "role": "selection", "start_sample": int(chunk.start_sample),
                "end_sample": int(chunk.end_sample), "valid_samples": int(chunk.valid_samples), "n_samples": int(row["n_samples"]),
                "pcm_sha256": row["pcm_sha256"], "shard": shard, "member": member, "position": position}
        strata.setdefault((comp, item["valid_samples"] < 64000), []).append(item)
    if sum(map(len, strata.values())) < n or n < 1:
        raise AlignmentError("not enough selection chunks for the requested parity sample")
    queues = [sorted(items, key=lambda r: digest([seed, r["item_id"]])) for _, items in sorted(strata.items())]
    chosen = []
    while len(chosen) < n:
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen
