"""R2-U1: a resumable, content-addressed frozen WavLM cache for Protocol B.

The published five-crop bank is consumed, never regenerated. Test-genuine
extraction is an explicit, logged exception for representation storage only;
test embeddings never enter diagnostic parity or the default reader.
GPU/model execution belongs to Colab. This module imports neither torch nor
transformers at import time; tiny synthetic extractors exercise the storage path.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import tarfile
import tempfile
import time

import numpy as np

from echoproof.artifacts import canonical_json, digest, file_digest
from echoproof.data.chunks import PREPROCESSING_HASH, eval_windows
from echoproof.data.cohort import read_flac, read_table

TASK = "R2-U1"
MANIFEST_NAME = "r2_u1_manifest.json"
INDEX_NAMES = {"development": "development_index.json", "test_genuine": "test_genuine_index.json"}
REPO_ID = "microsoft/wavlm-base-plus"
REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
WEIGHTS_SHA256 = "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51"
SAMPLE_RATE = 16000
ATOL = RTOL = 1e-4


class CacheError(ValueError):
    """Invalid scope, provenance, audio, or cache contents; never silently repaired."""


def _require(condition, message):
    if not condition:
        raise CacheError(message)


def extraction_identity(*, processor_config=None, versions=None, source_sha256=None, hidden_size=768):
    """Identity of the *executable* transformation, not merely its checkpoint.

Pass the downloaded processor's configuration and observed versions in Colab.
The default processor is the recorded R1 expectation, not a download claim.
"""
    processor = processor_config or {"sampling_rate": 16000, "do_normalize": False, "feature_size": 1}
    _require(processor.get("sampling_rate") == SAMPLE_RATE and processor.get("do_normalize") is False
             and processor.get("feature_size") == 1, "processor differs from the R1 WavLM contract")
    if versions is None:
        versions = {}
        for package in ("numpy", "torch", "transformers", "huggingface-hub"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unavailable"
    if source_sha256 is None:
        source_sha256 = {name: file_digest(Path(__file__).with_name(name)) for name in ("pretrained.py", "r2_u1.py")}
    _require(type(hidden_size) is int and hidden_size > 0, "hidden_size must be positive")
    return {"schema_version": 1, "repo_id": REPO_ID, "revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
            "processor_config": json.loads(canonical_json(processor)), "versions": dict(versions),
            "source_sha256": dict(source_sha256), "hidden_size": hidden_size,
            "sample_rate": SAMPLE_RATE, "dtype": "float32", "hidden_layer": "last_hidden_state",
            "pooling": "mean and mean+population_std over true hidden frames", "padding": "none; equal-length buckets",
            "encoder": "frozen; eval; inference_mode; fp32; TF32 disabled", "preprocessing_hash": PREPROCESSING_HASH}


def cache_key(pcm_sha256, start_sample, end_sample, identity, pooling):
    _require(pooling in ("mean", "mean_std"), "unknown pooling mode")
    return digest({"pcm_sha256": pcm_sha256, "start_sample": start_sample, "end_sample": end_sample,
                   "identity": identity, "pooling": pooling})


@dataclass
class Plan:
    rows: list
    identity: dict
    include_test_genuine: bool
    provenance: dict = field(default_factory=dict)

    @cached_property
    def sha256(self):
        return digest({"rows": self.rows, "identity": self.identity, "include_test_genuine": self.include_test_genuine,
                       "provenance": self.provenance})

    @property
    def counts(self):
        return {"logical_rows": len(self.rows), "unique_mean_keys": len({r["key_mean"] for r in self.rows}),
                "unique_mean_std_keys": len({r["key_mean_std"] for r in self.rows}),
                "by_partition": dict(sorted(Counter(r["partition"] for r in self.rows).items())),
                "by_role": dict(sorted(Counter(r["role"] or r["partition"] for r in self.rows).items())),
                "by_component": dict(sorted(Counter(r["component"] for r in self.rows).items()))}

    def shards(self):
        return sorted({r["shard"] for r in self.rows})


def plan_b(cohort_rows, assignments, chunks, crop_bank, identity, *, include_test_genuine=False):
    """Pure planning; no audio read. Assignment/chunk objects use shared contracts."""
    cohort_rows, assignments, chunks, crop_bank = map(list, (cohort_rows, assignments, chunks, crop_bank))
    by_id = {r["recording_id"]: r for r in cohort_rows}
    _require(len(by_id) == len(cohort_rows), "duplicate cohort recording_id")
    selected, speakers, pcm_groups, split_hashes, seen_assignments = {}, {}, {}, set(), set()
    for a in assignments:
        _require(a.protocol_id == "protocol:B", "non-B assignment")
        _require(a.recording_id not in seen_assignments, "duplicate assignment")
        seen_assignments.add(a.recording_id)
        split_hashes.add(a.split_hash)
        row = by_id.get(a.recording_id)
        _require(row is not None, "assignment without cohort recording")
        part, component = a.partition, row["component"]
        allowed = {"train": {"genuine", "melo", "vits"}, "validation": {"genuine", "melo", "vits", "xtts"},
                   "test": {"genuine", "melo", "vits", "chatterbox"}}
        _require(part in allowed and component in allowed[part], f"forbidden component {component} in {part}")
        _require(int(row["label"]) == (0 if component == "genuine" else 1), "component/label mismatch")
        _require(a.group_id == "speaker:" + str(row["speaker_id"]), "speaker group mismatch")
        stem = row.get("stem", a.recording_id.rsplit(":", 1)[-1])
        _require(re.fullmatch(r"[0-9]+-[0-9]+-[0-9]{4}", stem) is not None
                 and stem.split("-", 1)[0] == str(row["speaker_id"]), "stem/source-speaker mismatch")
        _require(a.validation_role in {"selection", "calibration", "operating"} if part == "validation"
                 else a.validation_role is None, "invalid validation role")
        group = (part, a.validation_role)
        previous = speakers.setdefault(a.group_id, group)
        _require(previous == group, "speaker crosses partition or validation role")
        if row.get("retained"):
            _require(re.fullmatch(r"[0-9a-f]{64}", row.get("pcm_sha256", "")) is not None, "invalid PCM hash")
            prior = pcm_groups.setdefault(row["pcm_sha256"], (group, int(row["label"])))
            _require(prior == (group, int(row["label"])), "PCM duplicate crosses partition/role or has conflicting labels")
        if part == "test" and (not include_test_genuine or component != "genuine"):
            continue
        if not row.get("retained"):
            continue
        path = row.get("canonical_path", "").split("::")
        _require(len(path) == 2 and re.fullmatch(r"shard_[0-9]+\.tar", path[0]) is not None
                 and path[1] == a.recording_id and "/" not in path[1] and "\\" not in path[1], "unsafe canonical_path")
        _require(type(row.get("n_samples")) is int and row["n_samples"] > 0, "invalid sample length")
        selected[a.recording_id] = (row, a, path[0], path[1].replace(":", "__") + ".flac")
    _require(len(split_hashes) == 1, "mixed or absent split hashes")
    rows, ids, crop_coordinates = [], set(), set()

    def append(item_id, rid, start, end, valid, kind):
        _require(rid in selected, "crop/chunk outside permitted retained scope")
        row, a, shard, member = selected[rid]
        _require(item_id not in ids, "duplicate crop/chunk id")
        ids.add(item_id)
        _require(type(start) is int and type(end) is int and type(valid) is int
                 and 0 <= start < end <= row["n_samples"]
                 and valid == end - start and valid == min(64000, row["n_samples"] - start), "invalid crop/chunk interval")
        rows.append({"item_id": item_id, "kind": kind, "recording_id": rid, "group_id": a.group_id,
                     "partition": a.partition, "role": a.validation_role, "component": row["component"], "label": int(row["label"]),
                     "start_sample": start, "end_sample": end, "valid_samples": valid, "n_samples": row["n_samples"],
                     "pcm_sha256": row["pcm_sha256"], "shard": shard, "member": member,
                     "scope": "test_genuine" if a.partition == "test" else "development",
                     "key_mean": cache_key(row["pcm_sha256"], start, end, identity, "mean"),
                     "key_mean_std": cache_key(row["pcm_sha256"], start, end, identity, "mean_std")})

    bank_recordings = Counter()
    for crop in crop_bank:
        rid = crop["recording_id"]
        _require(rid in selected and selected[rid][1].partition == "train", "crop bank contains a non-training recording")
        row, a, _, _ = selected[rid]
        _require(crop["component"] == row["component"] and crop["label"] == row["label"]
                 and crop["group_id"] == a.group_id and crop["bank_version"] == "crop-bank-v1"
                 and crop["bank_seed"] == 42 and crop["bank_k"] == 5, "crop bank metadata mismatch")
        _require(crop["crop_id"] == f"crop:B:{rid}:{crop['start_sample']}", "crop id differs from its coordinates")
        coordinate = (rid, crop["start_sample"], crop["end_sample"])
        _require(coordinate not in crop_coordinates, "duplicate crop coordinates")
        crop_coordinates.add(coordinate)
        append(crop["crop_id"], rid, crop["start_sample"], crop["end_sample"], crop["valid_samples"], "crop")
        bank_recordings[rid] += 1
    train_ids = {rid for rid, (_, a, _, _) in selected.items() if a.partition == "train"}
    _require(set(bank_recordings) == train_ids, "crop bank does not cover every retained training recording")
    for rid in train_ids:
        n = selected[rid][0]["n_samples"]
        _require(bank_recordings[rid] == min(5, max(1, n - 64000 + 1)), "crop bank count differs from signed-off policy")
    windows = defaultdict(list)
    for chunk in chunks:
        if chunk.protocol_id != "protocol:B":
            continue
        _require(chunk.recording_id in seen_assignments, "orphan Protocol B chunk")
        if chunk.recording_id not in selected or selected[chunk.recording_id][1].partition == "train":
            continue
        row, a, _, _ = selected[chunk.recording_id]
        _require(chunk.partition == a.partition and chunk.group_id == a.group_id
                 and chunk.preprocessing_hash == identity["preprocessing_hash"] and chunk.sample_rate == SAMPLE_RATE,
                 "chunk inheritance/preprocessing mismatch")
        append(chunk.chunk_id, chunk.recording_id, chunk.start_sample, chunk.end_sample, chunk.valid_samples, "chunk")
        windows[chunk.recording_id].append((chunk.start_sample, chunk.end_sample))
    for rid, (row, a, _, _) in selected.items():
        if a.partition != "train":
            _require(sorted(windows[rid]) == eval_windows(row["n_samples"]), "missing, duplicate, or noncanonical evaluation chunks")
    _require(train_ids and any(r["partition"] == "validation" for r in rows), "empty training or validation scope")
    _require(not include_test_genuine or any(r["scope"] == "test_genuine" for r in rows), "genuine-test scope requested but empty")
    return Plan(sorted(rows, key=lambda r: (r["shard"], r["recording_id"], r["item_id"])),
                json.loads(canonical_json(identity)), bool(include_test_genuine))


def preflight(root, *, identity=None, include_test_genuine=False):
    """Validate frozen input hashes and return a plan; never opens audio shards."""
    from echoproof.contracts import Chunk, SplitAssignment, read_parquet
    root = Path(root)
    base = root / "data/features/r2_s1_B"
    source_manifest = json.loads((base / "r2_s1_manifest.json").read_text(encoding="utf-8"))
    _require(source_manifest.get("verification", {}).get("passed") is True, "R2-S1 verification did not pass")
    matching_evidence = []
    for report_path in sorted((root / "reports/r2").glob("r2_s1_colab_*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("verdict", {}).get("verified") is True and report.get("build_manifest") == source_manifest:
            matching_evidence.append({"file": report_path.relative_to(root).as_posix(), "sha256": file_digest(report_path)})
    _require(matching_evidence, "R2-S1 manifest is not anchored to a matching verified Colab evidence report")
    paths = {"cohort": root / "data/manifests/cohort_v1.parquet", "split": root / "data/splits/protocol_B_v1.parquet",
             "chunks": root / "data/processed/chunks_v1.parquet", "crop_bank": base / "crop_bank_B_v1.parquet"}
    expected = dict(zip(("cohort", "split", "chunks", "crop_bank"),
                        (source_manifest["inputs"]["cohort_manifest_file_sha256"], source_manifest["inputs"]["split_file_sha256"],
                         source_manifest["inputs"]["chunk_manifest_file_sha256"], source_manifest["outputs"]["crop_bank"]["sha256"])))
    actual = {name: file_digest(path) for name, path in paths.items()}
    _require(actual == expected, "input file hashes differ from the published R2-S1 manifest")
    frozen = json.loads((root / "reports/r1/split_report.json").read_text(encoding="utf-8"))["protocols"]["B"]["split_sha256"]
    leakage = json.loads((root / "reports/r1/leakage.json").read_text(encoding="utf-8"))
    _require(leakage.get("passed") is True and leakage.get("violation_count") == 0, "R1 leakage evidence does not pass")
    assignments = read_parquet(str(paths["split"]), SplitAssignment)
    _require({a.split_hash for a in assignments} == {frozen} == {source_manifest["inputs"]["split_sha256"]}, "frozen split mismatch")
    plan = plan_b(read_table(paths["cohort"]), assignments, read_parquet(str(paths["chunks"]), Chunk),
                  read_table(paths["crop_bank"]), identity or extraction_identity(), include_test_genuine=include_test_genuine)
    _require(sum(r["kind"] == "crop" for r in plan.rows) == source_manifest["counts"]["train_crops"], "training crop count mismatch")
    _require(sum(r["partition"] == "validation" for r in plan.rows) == source_manifest["counts"]["validation_chunks"], "validation count mismatch")
    plan.provenance = {"input_sha256": actual, "split_sha256": frozen,
                       "r2_s1_manifest_sha256": file_digest(base / "r2_s1_manifest.json"),
                       "r2_s1_verified_evidence": matching_evidence,
                       "leakage_report_sha256": file_digest(root / "reports/r1/leakage.json"),
                       "test_access": "explicit extraction-only genuine test scope; no scores/metrics/fitting" if include_test_genuine else "none"}
    return plan


def _atomic(path, writer):
    """Single-writer atomic publication, ignoring abandoned temporary files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite {path}")
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".r2_u1-", suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        _require(not path.exists(), f"concurrent writer detected for {path}")
        os.replace(temp, path)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _json_new(path, value):
    _atomic(path, lambda handle: handle.write((canonical_json(value) + "\n").encode("utf-8")))


def _pack_path(out_dir, shard):
    return Path(out_dir) / "parts" / ("part_" + shard.removesuffix(".tar") + ".npz")


def _pack_rows(plan, shard):
    # Deduplication is content-based within a shard; the index retains every logical row.
    unique = {}
    for row in plan.rows:
        if row["shard"] == shard:
            unique.setdefault(row["key_mean_std"], row)
    return [unique[key] for key in sorted(unique)]


def _arrays_hash(arrays):
    return {name: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() for name, array in arrays.items()}


def _read_pack(plan, path, shard):
    try:
        with np.load(path, allow_pickle=False) as stored:
            _require(set(stored.files) == {"metadata", "keys_mean", "keys_mean_std", "mean", "mean_std"}, "unknown pack fields")
            meta = json.loads(str(stored["metadata"].item()))
            arrays = {name: stored[name].copy() for name in ("keys_mean", "keys_mean_std", "mean", "mean_std")}
        rows = _pack_rows(plan, shard)
        _require(meta["plan_sha256"] == plan.sha256 and meta["identity"] == plan.identity and meta["shard"] == shard,
                 "stale/incompatible cache pack")
        _require(arrays["keys_mean"].tolist() == [r["key_mean"] for r in rows]
                 and arrays["keys_mean_std"].tolist() == [r["key_mean_std"] for r in rows], "cache keys/order mismatch")
        dim = plan.identity["hidden_size"]
        for mode, width in (("mean", dim), ("mean_std", dim * 2)):
            _require(arrays[mode].shape == (len(rows), width) and arrays[mode].dtype == np.float32
                     and np.isfinite(arrays[mode]).all(), "invalid embedding shape/dtype/values")
        _require(np.array_equal(arrays["mean"], arrays["mean_std"][:, :dim]), "pooling views disagree")
        _require(meta["array_sha256"] == _arrays_hash(arrays), "cache content checksum mismatch")
        return arrays, meta
    except CacheError:
        raise
    except Exception as exc:
        raise CacheError(f"unreadable cache pack {path}: {exc}") from exc


def _read_wave(tar, row):
    member = tar.getmember(row["member"])
    _require(member.isfile() and not member.issym() and not member.islnk(), "audio member is not a regular file")
    handle = tar.extractfile(member)
    _require(handle is not None, "audio member missing")
    with handle:
        pcm = read_flac(handle.read())
    _require(pcm.ndim == 1 and pcm.dtype == np.int16 and len(pcm) == row["n_samples"], "canonical audio shape/length mismatch")
    _require(hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest() == row["pcm_sha256"], "canonical PCM hash mismatch")
    return pcm.astype(np.float32) / 32768.0


def _extract_pack(plan, shard_dir, shard, extractor, batch_size):
    rows = _pack_rows(plan, shard)
    vectors = {}
    by_recording = defaultdict(list)
    for row in rows:
        by_recording[row["recording_id"]].append(row)
    started = time.perf_counter()
    pending_rows, pending_clips = [], []

    def flush():
        if not pending_rows:
            return
        embedding = np.asarray(extractor.embed(pending_clips, pooling="mean_std"))
        _require(embedding.shape == (len(pending_rows), plan.identity["hidden_size"] * 2)
                 and embedding.dtype == np.float32 and np.isfinite(embedding).all(), "extractor returned invalid embeddings")
        vectors.update((r["key_mean_std"], v.copy()) for r, v in zip(pending_rows, embedding))
        pending_rows.clear()
        pending_clips.clear()

    with tarfile.open(Path(shard_dir) / shard, "r") as archive:
        for rid in sorted(by_recording):
            items = by_recording[rid]
            waveform = _read_wave(archive, items[0])
            for item in items:
                pending_rows.append(item)
                # Copy slices so a bounded batch cannot retain entire recordings.
                pending_clips.append(waveform[item["start_sample"]:item["end_sample"]].copy())
                if len(pending_rows) == batch_size:
                    flush()
        flush()
    both = np.stack([vectors[r["key_mean_std"]] for r in rows])
    arrays = {"keys_mean": np.asarray([r["key_mean"] for r in rows]), "keys_mean_std": np.asarray([r["key_mean_std"] for r in rows]),
              "mean": both[:, :plan.identity["hidden_size"]].copy(), "mean_std": both}
    return arrays, time.perf_counter() - started


def verify_cache(plan, out_dir):
    """Verify expected pack membership, keys, finite vectors, and content hashes."""
    _validate_plan(plan)
    expected = {_pack_path(out_dir, shard).name for shard in plan.shards()}
    existing = {p.name for p in (Path(out_dir) / "parts").glob("*.npz")}
    _require(expected == existing, "missing or unexpected cache packs")
    packs = {}
    for shard in plan.shards():
        path = _pack_path(out_dir, shard)
        _, metadata = _read_pack(plan, path, shard)
        packs[shard] = {"file": path.relative_to(out_dir).as_posix(), "sha256": file_digest(path),
                        "bytes": path.stat().st_size, "extraction_seconds": metadata["extraction_seconds"]}
    final_path = Path(out_dir) / MANIFEST_NAME
    if final_path.exists():
        manifest = json.loads(final_path.read_text(encoding="utf-8"))
        _require(manifest.get("schema_version") == 1 and manifest.get("task") == TASK, "invalid manifest task/schema")
        _require(manifest.get("plan_sha256") == plan.sha256 and manifest.get("identity") == plan.identity
                 and manifest.get("counts") == plan.counts and manifest.get("provenance") == plan.provenance
                 and manifest.get("include_test_genuine") == plan.include_test_genuine
                 and manifest.get("full_R2_U1_scope") == plan.include_test_genuine, "manifest identity/scope/counts mismatch")
        _require(manifest.get("status") in {"fixture_verified", "extracted_pending_independent_review"}, "unpublished cache status")
        _require(manifest.get("verification", {}).get("passed") is True
                 and manifest["verification"].get("packs") == packs, "pack hashes differ from committed manifest")
        parity = manifest.get("parity", {})
        _require(parity.get("passed") is True and parity.get("scope") == "development only"
                 and parity.get("atol") == ATOL and parity.get("rtol") == RTOL, "missing/invalid parity publication gate")
        n = parity.get("items")
        _require(type(n) is int and n > 0 and (manifest["status"] == "fixture_verified" or n == 50), "invalid parity sample count")
        _require(parity.get("item_ids") == [r["item_id"] for r in _parity_rows(plan, n)], "parity sampling mismatch")
        _require(set(manifest.get("indices", {})) == set(INDEX_NAMES), "missing/unknown published indices")
        for scope, filename in INDEX_NAMES.items():
            info = manifest["indices"][scope]
            path = Path(out_dir) / filename
            payload = _index_payload(plan, scope)
            _require(info.get("file") == filename and info.get("rows") == len(payload["rows"])
                     and info.get("bytes") == path.stat().st_size and info.get("sha256") == file_digest(path), "index checksum/count mismatch")
            _require(json.loads(path.read_text(encoding="utf-8")) == payload, "index differs from expected plan rows")
    return {"passed": True, "packs": packs, "counts": plan.counts}


def _index_payload(plan, scope):
    positions = {shard: {r["key_mean_std"]: i for i, r in enumerate(_pack_rows(plan, shard))} for shard in plan.shards()}
    entries = [{**row, "pack": _pack_path(".", row["shard"]).as_posix().removeprefix("./"),
                "pack_row": positions[row["shard"]][row["key_mean_std"]]}
               for row in plan.rows if row["scope"] == scope]
    return {"schema_version": 1, "plan_sha256": plan.sha256, "identity": plan.identity, "scope": scope, "rows": entries}


def _validate_plan(plan):
    current = digest({"rows": plan.rows, "identity": plan.identity, "include_test_genuine": plan.include_test_genuine,
                      "provenance": plan.provenance})
    _require(current == plan.sha256, "plan changed after its identity was frozen")
    _require(len({r["item_id"] for r in plan.rows}) == len(plan.rows), "duplicate logical cache rows")
    allowed = {"train": {"genuine", "melo", "vits"}, "validation": {"genuine", "melo", "vits", "xtts"},
               "test": {"genuine"}}
    for row in plan.rows:
        _require(re.fullmatch(r"shard_[0-9]+\.tar", row["shard"]) is not None, "unsafe shard name in plan")
        _require(row["partition"] in allowed and row["component"] in allowed[row["partition"]]
                 and row["label"] == int(row["component"] != "genuine"), "forbidden component/label in cache plan")
        _require(row["role"] in {"selection", "calibration", "operating"} if row["partition"] == "validation"
                 else row["role"] is None, "invalid cache validation role")
        _require(type(row["start_sample"]) is int and type(row["end_sample"]) is int
                 and 0 <= row["start_sample"] < row["end_sample"] <= row["n_samples"]
                 and row["valid_samples"] == row["end_sample"] - row["start_sample"]
                 and row["valid_samples"] <= 64000, "invalid cached coordinates")
        for mode in ("mean", "mean_std"):
            _require(row["key_" + mode] == cache_key(row["pcm_sha256"], row["start_sample"], row["end_sample"],
                                                    plan.identity, mode), "cache key differs from interval identity")
        _require(row["scope"] == ("test_genuine" if row["partition"] == "test" else "development"), "plan scope mismatch")
        _require(row["component"] != "chatterbox" and (row["partition"] != "test" or
                 (plan.include_test_genuine and row["component"] == "genuine")), "forbidden cache scope")


def _parity_rows(plan, n):
    # Round-robin strata include crops, roles, short/full clips and components.
    strata = defaultdict(list)
    for row in plan.rows:
        if row["scope"] == "development":
            strata[(row["kind"], row["role"] or "train", row["component"], row["valid_samples"] < 64000)].append(row)
    _require(sum(map(len, strata.values())) >= n > 0, "not enough development items for the requested parity check")
    queues = [sorted(items, key=lambda r: digest([42, r["item_id"]])) for _, items in sorted(strata.items())]
    selected = []
    while len(selected) < n:
        for queue in queues:
            if queue and len(selected) < n:
                selected.append(queue.pop(0))
    return selected


def _probabilities(vectors, weights):
    logits = vectors.astype(np.float64) @ weights
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp[:, 1] / exp.sum(axis=1)


def parity_check(plan, shard_dir, out_dir, extractor, *, n=50):
    """Engineering consistency only; the fixed head is UNTRAINED, not a detector."""
    selected = _parity_rows(plan, n)
    rng = np.random.default_rng(42)
    weights = {mode: rng.standard_normal((plan.identity["hidden_size"] * multiplier, 2)) * 0.01
               for mode, multiplier in (("mean", 1), ("mean_std", 2))}
    max_embedding = {"mean": 0.0, "mean_std": 0.0}
    max_score = {"mean": 0.0, "mean_std": 0.0}
    for shard in sorted({r["shard"] for r in selected}):
        arrays, _ = _read_pack(plan, _pack_path(out_dir, shard), shard)
        positions = {key: i for i, key in enumerate(arrays["keys_mean_std"].tolist())}
        with tarfile.open(Path(shard_dir) / shard, "r") as archive:
            for row in [r for r in selected if r["shard"] == shard]:
                waveform = _read_wave(archive, row)
                clip = waveform[row["start_sample"]:row["end_sample"]]
                position = positions[row["key_mean_std"]]
                for mode in ("mean", "mean_std"):
                    direct = np.asarray(extractor.embed([clip], pooling=mode))
                    cached = arrays[mode][position:position + 1]
                    _require(direct.shape == cached.shape and np.isfinite(direct).all(), "invalid direct parity embedding")
                    delta = float(np.max(np.abs(direct - cached)))
                    max_embedding[mode] = max(max_embedding[mode], delta)
                    _require(np.allclose(direct, cached, atol=ATOL, rtol=RTOL), "cached/direct embedding parity failed")
                    direct_score, cached_score = _probabilities(direct, weights[mode]), _probabilities(cached, weights[mode])
                    max_score[mode] = max(max_score[mode], float(np.max(np.abs(direct_score - cached_score))))
                    _require(np.allclose(direct_score, cached_score, atol=ATOL, rtol=RTOL), "diagnostic score parity failed")
    return {"passed": True, "items": n, "item_ids": [r["item_id"] for r in selected], "scope": "development only",
            "atol": ATOL, "rtol": RTOL, "max_abs_embedding_difference": max_embedding,
            "max_abs_diagnostic_score_difference": max_score, "diagnostic_head": "UNTRAINED seeded linear two-class head; no accuracy claim",
            "diagnostic_seed": 42, "diagnostic_weights_sha256": _arrays_hash(weights),
            "trained_adapter_parity": "pending R3-U1; not established by this engineering check"}


def build(plan, shard_dir, out_dir, extractor, *, resume=False, batch_size=16, parity_n=50, fixture=False, progress=None):
    """Extract/resume, verify, parity-check, then publish immutable final metadata.

    A single writer owns an output directory; an interrupted process can leave a
    .writer.lock which must be removed only after confirming the old writer died.
    Existing completed output is verified and returned only with resume=True.
    """
    _require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
    _validate_plan(plan)
    _require(fixture or parity_n == 50, "real extraction requires exactly 50 development parity items")
    _require(int(extractor.dim) == plan.identity["hidden_size"], "extractor hidden size differs from cache identity")
    if not fixture:
        _require(plan.identity["hidden_size"] == 768 and plan.identity["repo_id"] == REPO_ID
                 and plan.identity["revision"] == REVISION and plan.identity["weights_sha256"] == WEIGHTS_SHA256,
                 "production identity differs from the pinned checkpoint")
        observed = extractor.describe()
        _require(observed["trainable_parameters"] == 0 and observed["training_mode"] is False
                 and observed["do_normalize"] is False, "extractor is not frozen/eval with pinned preprocessing")
        for package, expected in plan.identity["versions"].items():
            _require(importlib.metadata.version(package) == expected, "observed package version differs from cache identity")
        sources = plan.identity["source_sha256"]
        for filename in ("pretrained.py", "r2_u1.py"):
            values = [value for name, value in sources.items() if Path(name).name == filename]
            _require(values == [file_digest(Path(__file__).with_name(filename))], "extractor source differs from cache identity")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CacheError("cache writer lock exists; confirm the prior writer stopped before removing it") from exc
    os.close(descriptor)
    started = time.perf_counter()
    say = progress or (lambda *_: None)
    try:
        manifest_path = out_dir / MANIFEST_NAME
        if manifest_path.exists():
            _require(resume, "finished cache exists; use resume to verify it")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _require(manifest["plan_sha256"] == plan.sha256, "finished cache has a different plan")
            _require(fixture == (manifest.get("status") == "fixture_verified"), "fixture output cannot be promoted to a production run")
            verify_cache(plan, out_dir)
            for name, info in manifest["indices"].items():
                _require(file_digest(out_dir / info["file"]) == info["sha256"], f"corrupt {name} index")
            return manifest
        _require(resume or not list((out_dir / "parts").glob("*.npz")), "partial cache exists; use resume")
        reused, extracted = [], []
        for shard in plan.shards():
            path = _pack_path(out_dir, shard)
            if path.exists():
                _read_pack(plan, path, shard)
                reused.append(shard)
                say(f"{shard}: validated existing pack")
            else:
                arrays, seconds = _extract_pack(plan, shard_dir, shard, extractor, batch_size)
                metadata = {"plan_sha256": plan.sha256, "identity": plan.identity, "shard": shard,
                            "array_sha256": _arrays_hash(arrays), "extraction_seconds": seconds}
                _atomic(path, lambda handle: np.savez(handle, metadata=np.asarray(canonical_json(metadata)), **arrays))
                _read_pack(plan, path, shard)
                extracted.append(shard)
                say(f"{shard}: extracted and validated")
        verification = verify_cache(plan, out_dir)
        parity = parity_check(plan, shard_dir, out_dir, extractor, n=parity_n)
        indices = {}
        for scope, filename in INDEX_NAMES.items():
            payload = _index_payload(plan, scope)
            path = out_dir / filename
            if path.exists():
                _require(json.loads(path.read_text(encoding="utf-8")) == payload, "existing index differs from resumed plan")
            else:
                _json_new(path, payload)
            indices[scope] = {"file": filename, "sha256": file_digest(path), "rows": len(payload["rows"]), "bytes": path.stat().st_size}
        pack_bytes = sum(p["bytes"] for p in verification["packs"].values())
        index_bytes = sum(p["bytes"] for p in indices.values())
        manifest = {"schema_version": 1, "task": TASK, "plan_sha256": plan.sha256, "identity": plan.identity,
                    "provenance": plan.provenance, "counts": plan.counts, "include_test_genuine": plan.include_test_genuine,
                    "verification": verification, "parity": parity, "indices": indices,
                    "extraction_seconds_all_packs": sum(p["extraction_seconds"] for p in verification["packs"].values()),
                    "seconds_this_session": time.perf_counter() - started, "packs_extracted_this_session": extracted,
                    "packs_reused_this_session": reused, "pack_bytes": pack_bytes, "index_bytes": index_bytes,
                    "cache_bytes": pack_bytes + index_bytes, "cache_bytes_excludes_manifest_and_temp": True,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "environment": {"python": platform.python_version(), "numpy": np.__version__},
                    "status": "fixture_verified" if fixture else "extracted_pending_independent_review",
                    "full_R2_U1_scope": plan.include_test_genuine, "research_accuracy_claim": False}
        _json_new(manifest_path, manifest)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def read_embeddings(out_dir, *, partition=None, pooling="mean", include_test_genuine=False):
    """Read published embeddings; test access requires an explicit opt-in.

    Returned index rows preserve alignment. Loading only the requested partition
    avoids exposing final-test embeddings to ordinary R3 training consumers.
    """
    _require(pooling in ("mean", "mean_std"), "unknown pooling mode")
    _require(partition in (None, "train", "validation", "test"), "unknown partition")
    _require(partition != "test" or include_test_genuine, "test embeddings require explicit opt-in")
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    all_rows = []
    for scope, filename in INDEX_NAMES.items():
        _require(set(manifest.get("indices", {})) == set(INDEX_NAMES), "missing/unknown published indices")
        info = manifest["indices"][scope]
        _require(info["file"] == filename and file_digest(out_dir / filename) == info["sha256"], "index checksum mismatch")
        index = json.loads((out_dir / filename).read_text(encoding="utf-8"))
        _require(index["plan_sha256"] == manifest["plan_sha256"] and index["scope"] == scope
                 and index["identity"] == manifest["identity"], "index identity mismatch")
        _require(all(r["scope"] == scope for r in index["rows"]), "index scope mismatch")
        all_rows.extend(index["rows"])
    raw_rows = [{key: value for key, value in row.items() if key not in {"pack", "pack_row"}} for row in all_rows]
    reconstructed = Plan(sorted(raw_rows, key=lambda r: (r["shard"], r["recording_id"], r["item_id"])), manifest["identity"],
                         manifest["include_test_genuine"], manifest["provenance"])
    verify_cache(reconstructed, out_dir)
    rows = [r for r in all_rows if (r["scope"] == "development" or include_test_genuine)
            and (partition is None or r["partition"] == partition)]
    positions = defaultdict(list)
    width = manifest["identity"]["hidden_size"] * (2 if pooling == "mean_std" else 1)
    for i, row in enumerate(rows):
        positions[row["shard"]].append((i, row))
    matrix = np.empty((len(rows), width), dtype=np.float32)
    for shard, items in positions.items():
        info = manifest["verification"]["packs"][shard]
        path = out_dir / info["file"]
        _require(file_digest(path) == info["sha256"], "cache pack checksum mismatch")
        with np.load(path, allow_pickle=False) as pack:
            _require(pack[pooling].dtype == np.float32 and pack[pooling].ndim == 2
                     and pack[pooling].shape[1] == width and np.isfinite(pack[pooling]).all(), "invalid published vectors")
            for i, row in items:
                _require(str(pack["keys_" + pooling][row["pack_row"]]) == row["key_" + pooling], "cache row/key mismatch")
                matrix[i] = pack[pooling][row["pack_row"]]
    return rows, matrix
