"""Guarded Protocol B inputs for R3-U2 (AST partial fine-tuning): raw 4-second crops and validation chunks, no test or Chatterbox audio.

Same frozen split, cohort, chunk manifest and shards as R3-M1. The AST wrapper extracts its own spectrograms, so no mel statistics are needed.
Audio is read only for the Protocol B training recordings and the three validation roles, and every decoded waveform is checked against the
cohort manifest's canonical PCM hash before a dataset sees it.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from echoproof.training.datasets import ChunkDataset, CropDataset, ShardWaveformLoader
from echoproof.training.r3_m1_data import VerifiedWaveformLoader, file_sha256

ROLES = ("selection", "calibration", "operating")


class R3U2DataError(ValueError):
    pass


def verify_inputs(paths: dict, manifest: dict, frozen: str) -> dict:
    """The cohort, split and chunk files must be the ones the verified R2-S1 B manifest was built from, under the frozen split hash."""
    if manifest.get("protocol") != "B" or not manifest.get("verification", {}).get("passed"):
        raise R3U2DataError("the R2-S1 manifest is not verified for Protocol B")
    inputs = manifest["inputs"]
    if inputs["split_sha256"] != frozen:
        raise R3U2DataError("frozen split hash differs from the R2-S1 manifest")
    expected = {"cohort": inputs["cohort_manifest_file_sha256"], "split": inputs["split_file_sha256"],
                "chunks": inputs["chunk_manifest_file_sha256"]}
    actual = {name: file_sha256(paths[name]) for name in expected}
    if actual != expected:
        raise R3U2DataError("input file hash mismatch")
    return actual


def build_data(root, shard_dir, seed: int = 42) -> dict:
    """B training crops plus one chunk dataset per validation role. The caller must close the returned loader."""
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data import cohort, splits
    from echoproof.features.r2_s1 import plan_b

    root = Path(root)
    paths = {"cohort": root / "data/manifests/cohort_v1.parquet", "split": root / "data/splits/protocol_B_v1.parquet",
             "chunks": root / "data/processed/chunks_v1.parquet"}
    manifest_path = root / "data/features/r2_s1_B/r2_s1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads((root / "reports/r1/split_report.json").read_text(encoding="utf-8"))["protocols"]["B"]["split_sha256"]
    hashes = verify_inputs(paths, manifest, frozen)
    assignments = splits.load_assignments(paths["split"], sealed=False)
    if not assignments or {a.split_hash for a in assignments} != {frozen}:
        raise R3U2DataError("assignment split hash mismatch")
    rows = cohort.read_table(paths["cohort"])
    if len({r["recording_id"] for r in rows}) != len(rows):
        raise R3U2DataError("duplicate cohort recording")
    chunks = read_parquet(str(paths["chunks"]), Chunk)
    plan = plan_b(rows, assignments, chunks)
    by_id = {r["recording_id"]: r for r in rows}
    selected = [by_id[x["recording_id"]] for x in plan.train + plan.validation]
    for row in selected:
        if row["component"] not in {"genuine", "melo", "vits", "xtts"} or row["label"] != (row["component"] != "genuine"):
            raise R3U2DataError("invalid component or label")
    train_ids = {x["recording_id"] for x in plan.train}
    if any(x["component"] == "xtts" for x in plan.train):
        raise R3U2DataError("XTTS must not appear in Protocol B training")
    roles = {x["recording_id"]: x["role"] for x in plan.validation}
    labels = {x["recording_id"]: x["label"] for x in plan.validation}
    groups = {x["recording_id"]: x["group_id"] for x in plan.validation}
    if train_ids & set(roles):
        raise R3U2DataError("a recording is both training and validation")
    val_chunks = [c for c in chunks if c.protocol_id == "protocol:B" and c.partition == "validation" and c.recording_id in roles]
    if len({c.chunk_id for c in val_chunks}) != len(val_chunks):
        raise R3U2DataError("duplicate validation chunk")
    for c in val_chunks:
        if c.group_id != groups[c.recording_id]:
            raise R3U2DataError("chunk group mismatch")
    base = ShardWaveformLoader(shard_dir, {r["recording_id"]: r["canonical_path"] for r in selected})
    loader = VerifiedWaveformLoader(base, selected)
    train = CropDataset([(x["recording_id"], x["label"]) for x in plan.train], loader, seed=seed)
    validation = {}
    for role in ROLES:
        role_ids = {rid for rid, value in roles.items() if value == role}
        if not role_ids or {labels[rid] for rid in role_ids} != {0, 1}:
            raise R3U2DataError(f"invalid validation role: {role}")
        validation[role] = ChunkDataset([c for c in val_chunks if c.recording_id in role_ids], labels, loader)
        validation[role].validation_role = role
    return {"train": train, "validation": validation, "loader": loader,
            "validation_components": {x["recording_id"]: x["component"] for x in plan.validation}, "validation_groups": groups,
            "split_sha256": frozen, "input_sha256": {**hashes, "features_manifest": file_sha256(manifest_path)},
            "counts": {"train": len(train), "train_by_label": dict(Counter(train.labels)),
                       "validation_recordings": dict(Counter(roles.values())),
                       "validation_chunks": {role: len(ds) for role, ds in validation.items()}}}
