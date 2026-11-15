"""Guarded Protocol A inputs for R4-U1b (AST-A warm-up and partial fine-tuning): raw 4-second crops and validation chunks, no test audio.

Same structure as `r3_u2_data` (a frozen B source, hard-coded to Protocol B and to a verified B feature manifest). The plan comes from `r4_s1.plan_a`,
which withholds the A test partition and checks components, groups and canonical windows. The cohort, split and chunk files must be the ones the verified
R4-S1 manifest was built from, under the frozen A split hash, and the counts must equal that manifest's. Every decoded waveform is checked against the
cohort manifest's canonical PCM hash before a dataset sees it.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from echoproof.artifacts import digest
from echoproof.training.datasets import ChunkDataset, CropDataset, ShardWaveformLoader
from echoproof.training.r3_m1_data import VerifiedWaveformLoader, file_sha256

ROLES = ("selection", "calibration", "operating")
COMPONENTS = ("genuine", "melo", "vits", "xtts", "chatterbox")
FULL_CHUNK = 64000


class R4U1DataError(ValueError):
    pass


def verify_inputs(paths: dict, manifest: dict, frozen: str) -> dict:
    """The cohort, split and chunk files must be the ones the verified R4-S1 manifest was built from, under the frozen A split hash."""
    if manifest.get("task") != "R4-S1" or manifest.get("protocol") != "A" or not manifest.get("verification", {}).get("passed"):
        raise R4U1DataError("the R4-S1 manifest is not verified for Protocol A")
    inputs = manifest["inputs"]
    if inputs["split_sha256"] != frozen:
        raise R4U1DataError("frozen split hash differs from the R4-S1 manifest")
    expected = {"cohort": inputs["cohort_manifest_file_sha256"], "split": inputs["split_file_sha256"], "chunks": inputs["chunk_manifest_file_sha256"]}
    actual = {name: file_sha256(paths[name]) for name in expected}
    if actual != expected:
        raise R4U1DataError("input file hash mismatch")
    return actual


def build_data(root, shard_dir, seed: int = 42) -> dict:
    """A training crops plus one chunk dataset per validation role. The caller must close the returned loader."""
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data import cohort, splits
    from echoproof.features import r4_s1

    root = Path(root)
    paths = {"cohort": root / "data/manifests/cohort_v1.parquet", "split": root / "data/splits/protocol_A_v1.parquet",
             "chunks": root / "data/processed/chunks_v1.parquet"}
    manifest_path = root / "data/features/r4_s1_A/r4_s1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads((root / "reports/r1/split_report.json").read_text(encoding="utf-8"))["protocols"]["A"]["split_sha256"]
    hashes = verify_inputs(paths, manifest, frozen)
    assignments = splits.load_assignments(paths["split"], sealed=False)               # the loader withholds test rows
    if not assignments or {a.split_hash for a in assignments} != {frozen}:
        raise R4U1DataError("assignment split hash mismatch")
    rows = cohort.read_table(paths["cohort"])
    if len({r["recording_id"] for r in rows}) != len(rows):
        raise R4U1DataError("duplicate cohort recording")
    chunks = read_parquet(str(paths["chunks"]), Chunk)
    plan = r4_s1.plan_a(rows, assignments, chunks)
    if plan.counts != manifest["counts"]:
        raise R4U1DataError("the plan counts differ from the verified R4-S1 manifest")
    if plan.counts["test_assignments_dropped"] != 0:
        raise R4U1DataError("test assignments reached the plan")
    by_id = {r["recording_id"]: r for r in rows}
    selected = [by_id[x["recording_id"]] for x in plan.train + plan.validation]
    for row in selected:
        if row["component"] not in COMPONENTS or row["label"] != (row["component"] != "genuine"):
            raise R4U1DataError("invalid component or label")
    if {x["component"] for x in plan.train} != set(COMPONENTS):
        raise R4U1DataError("Protocol A training must contain all five components")
    train_ids = {x["recording_id"] for x in plan.train}
    roles = {x["recording_id"]: x["role"] for x in plan.validation}
    labels = {x["recording_id"]: x["label"] for x in plan.validation}
    groups = {x["recording_id"]: x["group_id"] for x in plan.validation}
    if train_ids & set(roles):
        raise R4U1DataError("a recording is both training and validation")
    val_chunks = [c for c in chunks if c.protocol_id == "protocol:A" and c.partition == "validation" and c.recording_id in roles]
    if len({c.chunk_id for c in val_chunks}) != len(val_chunks):
        raise R4U1DataError("duplicate validation chunk")
    for c in val_chunks:
        if c.group_id != groups[c.recording_id]:
            raise R4U1DataError("chunk group mismatch")
    base = ShardWaveformLoader(shard_dir, {r["recording_id"]: r["canonical_path"] for r in selected})
    loader = VerifiedWaveformLoader(base, selected)
    train = CropDataset([(x["recording_id"], x["label"]) for x in plan.train], loader, seed=seed)
    validation = {}
    for role in ROLES:
        role_ids = {rid for rid, value in roles.items() if value == role}
        components = {x["component"] for x in plan.validation if x["role"] == role}
        if not role_ids or {labels[rid] for rid in role_ids} != {0, 1} or components != set(COMPONENTS):
            raise R4U1DataError(f"invalid validation role: {role}")
        validation[role] = ChunkDataset([c for c in val_chunks if c.recording_id in role_ids], labels, loader)
        validation[role].validation_role = role
    return {"train": train, "validation": validation, "loader": loader,
            "validation_components": {x["recording_id"]: x["component"] for x in plan.validation}, "validation_groups": groups,
            "split_sha256": frozen, "plan_sha256": plan.sha256, "input_sha256": {**hashes, "r4_s1_manifest": file_sha256(manifest_path)},
            "counts": {"train": len(train), "train_by_label": dict(Counter(train.labels)), "train_by_component": plan.counts["train_by_component"],
                       "validation_recordings": dict(Counter(roles.values())), "validation_chunks": {role: len(ds) for role, ds in validation.items()}}}


def select_chunks(dataset, components: dict, n: int = 50, *, seed: int = 42) -> list:
    """Indices into `dataset.chunks`: a round-robin draw over (component, short or full) strata, Chatterbox included."""
    strata = {}
    for index, chunk in enumerate(dataset.chunks):
        strata.setdefault((components[chunk.recording_id], chunk.valid_samples < FULL_CHUNK), []).append(index)
    if n < 1 or sum(map(len, strata.values())) < n:
        raise R4U1DataError("not enough validation chunks for the requested parity sample")
    queues = [sorted(items, key=lambda i: digest([seed, dataset.chunks[i].chunk_id])) for _, items in sorted(strata.items())]
    chosen = []
    while len(chosen) < n:
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen
