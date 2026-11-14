"""R4-U0: frozen WavLM embeddings for the Protocol A rows the Protocol B cache (R2-U1) does not hold.

The B cache is content-addressed (key = PCM hash, interval, extraction identity, pooling) and is a frozen B artefact, so it is only read.
R4-S1 listed the A crops and validation chunks absent from it (`wavlm_cache_gap_A_v1.json`); this module extracts exactly those into a separate
A cache, proves the union of the two caches covers every A train crop and A validation chunk exactly once, proves this run reproduces
B's stored vectors, and reads the union back in the R4-S1 row order.

`r2_u1` is a frozen B source file (hashed by the GR3 lock) and refuses Chatterbox, so only its protocol-free helpers are reused
(shard packs, atomic writes, content keys, the parity check). The planner, plan validation, manifest and reader are restated here.
No A test row is planned, read or extracted. GPU/model execution belongs to Colab; nothing here imports torch at import time.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import platform
import re
import tarfile
import time

import numpy as np

from echoproof.artifacts import canonical_json, digest, file_digest
from echoproof.data import splits
from echoproof.data.chunks import PREPROCESSING_HASH, eval_windows
from echoproof.data.cohort import read_table
from echoproof.features import r2_u1
from echoproof.features.r2_u1 import ATOL, RTOL, CacheError, Plan, cache_key, extraction_identity  # noqa: F401  (re-exported for callers)

TASK = "R4-U0"
PROTOCOL_ID = "protocol:A"
MANIFEST_NAME = "r4_u0_manifest.json"
INDEX_NAME = "gap_index.json"
GAP_NAME = "wavlm_cache_gap_A_v1.json"
A_DIR = Path("data/features/r4_u0_A")
S1_DIR = Path("data/features/r4_s1_A")
B_DIR = Path("data/features/r2_u1_B")
B_PARITY_N = 50
CROP_SAMPLES = 64000
ALLOWED = {"train": {"genuine", "melo", "vits", "xtts", "chatterbox"}, "validation": {"genuine", "melo", "vits", "xtts", "chatterbox"}}


def _require(condition, message):
    if not condition:
        raise CacheError(message)


def _coord(kind, rid, start, end):
    return (kind, rid, int(start), int(end))


# ---- planning ----------------------------------------------------------------------------------------------------------------

def a_universe(assignments, chunks, a_crop_bank):
    """Every A train crop and A validation chunk, as {coordinate: row info}. Test assignments are refused, not skipped."""
    seen, info = {}, {}
    for a in assignments:
        _require(a.protocol_id == PROTOCOL_ID, "non-A assignment")
        _require(a.partition in ALLOWED, f"{a.partition} assignment in the R4-U0 scope (A test is sealed)")
        _require(a.recording_id not in seen, "duplicate assignment")
        _require((a.partition == "validation") == (a.validation_role is not None), "invalid validation role")
        seen[a.recording_id] = a
    universe = {}
    per_recording = Counter()
    for c in a_crop_bank:
        rid = c["recording_id"]
        _require(rid in seen and seen[rid].partition == "train", "A crop bank contains a non-training recording")
        _require(c["crop_id"] == f"crop:A:{rid}:{c['start_sample']}" and c["bank_version"] == "crop-bank-v1" and c["bank_seed"] == 42
                 and c["bank_k"] == 5 and c["group_id"] == seen[rid].group_id, "A crop bank metadata mismatch")
        key = _coord("crop", rid, c["start_sample"], c["end_sample"])
        _require(key not in universe, "duplicate A crop coordinates")
        universe[key] = {"item_id": c["crop_id"], "partition": "train", "role": None, "group_id": c["group_id"],
                         "valid_samples": int(c["valid_samples"])}
        per_recording[rid] += 1
    train_ids = {rid for rid, a in seen.items() if a.partition == "train"}
    _require(set(per_recording) <= train_ids, "A crop bank outside train")
    windows = defaultdict(list)
    for c in chunks:
        if c.protocol_id != PROTOCOL_ID or c.partition != "validation":      # train chunks are not used (crops are); test chunks are sealed
            continue
        _require(c.recording_id in seen and seen[c.recording_id].partition == "validation", "A validation chunk without a validation assignment")
        a = seen[c.recording_id]
        _require(c.group_id == a.group_id and c.preprocessing_hash == PREPROCESSING_HASH and c.sample_rate == r2_u1.SAMPLE_RATE,
                 "chunk inheritance/preprocessing mismatch")
        key = _coord("chunk", c.recording_id, c.start_sample, c.end_sample)
        _require(key not in universe, "duplicate A chunk coordinates")
        universe[key] = {"item_id": c.chunk_id, "partition": "validation", "role": a.validation_role, "group_id": a.group_id,
                         "valid_samples": int(c.valid_samples)}
        windows[c.recording_id].append((int(c.start_sample), int(c.end_sample)))
    return universe, seen, windows, per_recording


def plan_gap(cohort_rows, assignments, chunks, a_crop_bank, b_index_rows, gap_payload, identity, provenance=None):
    """Pure planning, no audio. The plan is the gap file, re-derived independently and required to equal it exactly."""
    cohort_rows, a_crop_bank, b_index_rows = list(cohort_rows), list(a_crop_bank), list(b_index_rows)
    by_id = {r["recording_id"]: r for r in cohort_rows}
    _require(len(by_id) == len(cohort_rows), "duplicate cohort recording_id")
    universe, seen, windows, per_recording = a_universe(list(assignments), list(chunks), a_crop_bank)
    for rid, a in seen.items():
        row = by_id.get(rid)
        _require(row is not None, "assignment without cohort recording")
        _require(row["component"] in ALLOWED[a.partition], f"forbidden component {row['component']} in {a.partition}")
        _require(a.group_id == "speaker:" + str(row["speaker_id"]), "speaker group mismatch")
        _require(int(row["label"]) == (0 if row["component"] == "genuine" else 1), "component/label mismatch")
        if not row.get("retained"):
            continue
        n = row.get("n_samples")
        _require(type(n) is int and n > 0, "invalid sample length")
        if a.partition == "train":
            _require(per_recording[rid] == min(5, max(1, n - CROP_SAMPLES + 1)), "crop bank count differs from signed-off policy")
        else:
            _require(sorted(windows[rid]) == eval_windows(n), "missing, duplicate or noncanonical evaluation chunks")
    have = set()
    for r in b_index_rows:
        _require(r["partition"] != "test" and r["scope"] == "development", "B development index contains a test row")
        have.add(_coord(r["kind"], r["recording_id"], r["start_sample"], r["end_sample"]))
    _require(have <= set(universe), "the B cache holds development rows that are not A rows; the A split would differ from B")
    expected = {k for k in universe if k not in have}
    listed = {}
    for g in gap_payload["rows"]:
        key = _coord(g["kind"], g["recording_id"], g["start_sample"], g["end_sample"])
        _require(key not in listed, "duplicate row in the gap file")
        listed[key] = g
    _require(set(listed) == expected, f"gap file differs from the independently derived gap ({len(listed)} listed, {len(expected)} expected)")
    summary = Counter((r["kind"], r["partition"], r["component"]) for r in gap_payload["rows"])
    _require(gap_payload["summary"] == {"|".join(map(str, k)): v for k, v in sorted(summary.items())}, "gap summary does not match its rows")
    rows = []
    for key in sorted(expected, key=lambda k: (universe[k]["item_id"])):
        kind, rid, start, end = key
        u, g, a = universe[key], listed[key], seen[key[1]]
        row = by_id[rid]
        _require(row.get("retained"), f"gap row of an unretained recording {rid}")
        _require(g["item_id"] == u["item_id"] and g["partition"] == u["partition"] and g["role"] == u["role"]
                 and g["group_id"] == u["group_id"] and g["component"] == row["component"] and g["label"] == int(row["label"])
                 and g["valid_samples"] == u["valid_samples"], "gap row disagrees with the manifests")
        path = row.get("canonical_path", "").split("::")
        _require(len(path) == 2 and re.fullmatch(r"shard_[0-9]+\.tar", path[0]) is not None and path[1] == rid
                 and "/" not in path[1] and "\\" not in path[1], "unsafe canonical_path")
        _require(g["shard"] == path[0], "gap row shard differs from the cohort manifest")
        _require(re.fullmatch(r"[0-9a-f]{64}", row.get("pcm_sha256", "")) is not None, "invalid PCM hash")
        _require(0 <= start < end <= row["n_samples"] and u["valid_samples"] == end - start <= CROP_SAMPLES
                 and (kind == "chunk" or u["valid_samples"] == min(CROP_SAMPLES, row["n_samples"] - start)), "invalid crop/chunk interval")
        rows.append({"item_id": u["item_id"], "kind": kind, "recording_id": rid, "group_id": u["group_id"], "partition": u["partition"],
                     "role": u["role"], "component": row["component"], "label": int(row["label"]), "start_sample": start, "end_sample": end,
                     "valid_samples": u["valid_samples"], "n_samples": row["n_samples"], "pcm_sha256": row["pcm_sha256"], "shard": path[0],
                     "member": path[1].replace(":", "__") + ".flac", "scope": "development",
                     "key_mean": cache_key(row["pcm_sha256"], start, end, identity, "mean"),
                     "key_mean_std": cache_key(row["pcm_sha256"], start, end, identity, "mean_std")})
    _require(rows, "empty gap")
    coverage = {"a_rows": len(universe), "b_cache_rows_in_a": len(have), "gap_rows": len(rows)}
    _require(coverage["b_cache_rows_in_a"] + coverage["gap_rows"] == coverage["a_rows"], "B cache and gap do not partition the A rows")
    prov = dict(provenance or {})
    prov["coverage"] = coverage
    return Plan(sorted(rows, key=lambda r: (r["shard"], r["recording_id"], r["item_id"])), json.loads(canonical_json(identity)), False, prov)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight(root, *, identity=None):
    """Validate the frozen evidence chain (R4-S1 verified on Colab, B cache published) and return the plan; never opens audio shards."""
    from echoproof.contracts import Chunk, read_parquet
    root = Path(root)
    s1 = root / S1_DIR
    manifest = _load_json(s1 / "r4_s1_manifest.json")
    _require(manifest.get("task") == "R4-S1" and manifest.get("verification", {}).get("passed") is True
             and manifest.get("crosscheck_vs_b_passed") is True, "R4-S1 manifest is not verified")
    evidence = []
    expected_gap_hash = None
    for report_path in sorted((root / "reports/r4").glob("r4_s1_colab_*.json")):
        report = _load_json(report_path)
        built = dict(report.get("build_manifest", {}))
        side = built.pop("side_outputs", {})
        if report.get("verdict", {}).get("verified") is True and built == manifest:
            expected_gap_hash = side.get(GAP_NAME, {}).get("sha256")
            evidence.append({"file": report_path.relative_to(root).as_posix(), "sha256": file_digest(report_path)})
    _require(evidence and expected_gap_hash, "R4-S1 manifest is not anchored to a matching verified Colab evidence report")
    paths = {"cohort": root / "data/manifests/cohort_v1.parquet", "split": root / "data/splits/protocol_A_v1.parquet",
             "chunks": root / "data/processed/chunks_v1.parquet", "crop_bank": s1 / manifest["outputs"]["crop_bank"]["file"], "gap": s1 / GAP_NAME}
    actual = {name: file_digest(path) for name, path in paths.items()}
    expected = {"cohort": manifest["inputs"]["cohort_manifest_file_sha256"], "split": manifest["inputs"]["split_file_sha256"],
                "chunks": manifest["inputs"]["chunk_manifest_file_sha256"], "crop_bank": manifest["outputs"]["crop_bank"]["sha256"],
                "gap": expected_gap_hash}
    _require(actual == expected, "input file hashes differ from the published R4-S1 evidence")
    frozen = _load_json(root / "reports/r1/split_report.json")["protocols"]["A"]["split_sha256"]
    leakage = _load_json(root / "reports/r1/leakage.json")
    _require(leakage.get("passed") is True and leakage.get("violation_count") == 0, "R1 leakage evidence does not pass")
    assignments = splits.load_assignments(paths["split"], sealed=False)       # the loader withholds test rows
    _require({a.split_hash for a in assignments} == {frozen} == {manifest["inputs"]["split_sha256"]}, "frozen A split mismatch")
    b = root / B_DIR
    b_manifest = _load_json(b / "r2_u1_manifest.json")
    _require(b_manifest.get("verification", {}).get("passed") is True, "B cache manifest did not verify")
    b_index_path = b / "development_index.json"
    _require(file_digest(b_index_path) == b_manifest["indices"]["development"]["sha256"], "B development index differs from its manifest")
    b_index = _load_json(b_index_path)
    gap = _load_json(paths["gap"])
    _require(gap.get("task") == "R4-U0-input" and gap.get("plan_sha256") == manifest["plan_sha256"]
             and gap.get("b_cache_index_plan_sha256") == b_index["plan_sha256"], "gap file does not belong to this R4-S1 plan and B cache")
    plan = plan_gap(_read_table(paths["cohort"]), assignments, read_parquet(str(paths["chunks"]), Chunk), _read_table(paths["crop_bank"]),
                    b_index["rows"], gap, identity or extraction_identity(),
                    {"input_sha256": actual, "split_sha256": frozen, "r4_s1_manifest_sha256": file_digest(s1 / "r4_s1_manifest.json"),
                     "r4_s1_verified_evidence": evidence, "leakage_report_sha256": file_digest(root / "reports/r1/leakage.json"),
                     "b_cache_manifest_sha256": file_digest(b / "r2_u1_manifest.json"), "b_cache_identity": b_manifest["identity"],
                     "test_access": "none; A test is sealed and no A test row is planned or extracted"})
    return plan


def _read_table(path):
    return read_table(Path(path))


# ---- verification ------------------------------------------------------------------------------------------------------------

def _validate_plan(plan):
    _require(digest({"rows": plan.rows, "identity": plan.identity, "include_test_genuine": plan.include_test_genuine,
                     "provenance": plan.provenance}) == plan.sha256, "plan changed after its identity was frozen")
    _require(plan.include_test_genuine is False, "R4-U0 never includes test rows")
    _require(len({r["item_id"] for r in plan.rows}) == len(plan.rows), "duplicate logical cache rows")
    for row in plan.rows:
        _require(re.fullmatch(r"shard_[0-9]+\.tar", row["shard"]) is not None, "unsafe shard name in plan")
        _require(row["partition"] in ALLOWED and row["component"] in ALLOWED[row["partition"]]
                 and row["label"] == int(row["component"] != "genuine") and row["scope"] == "development", "forbidden row in the plan")
        _require((row["role"] in {"selection", "calibration", "operating"}) == (row["partition"] == "validation")
                 and (row["role"] is None) == (row["partition"] == "train"), "invalid validation role")
        _require(0 <= row["start_sample"] < row["end_sample"] <= row["n_samples"]
                 and row["valid_samples"] == row["end_sample"] - row["start_sample"] <= CROP_SAMPLES, "invalid cached coordinates")
        for mode in ("mean", "mean_std"):
            _require(row["key_" + mode] == cache_key(row["pcm_sha256"], row["start_sample"], row["end_sample"], plan.identity, mode),
                     "cache key differs from interval identity")


def b_parity_rows(b_index_rows, shards, n=B_PARITY_N):
    """Stratified deterministic sample of B-cached development rows that live in shards this run stages."""
    strata = defaultdict(list)
    for r in b_index_rows:
        if r["scope"] == "development" and r["shard"] in shards:
            strata[(r["kind"], r["role"] or "train", r["component"], r["valid_samples"] < CROP_SAMPLES)].append(r)
    _require(sum(map(len, strata.values())) >= n, "not enough B-cached rows in the staged shards for the B parity check")
    queues = [sorted(items, key=lambda r: digest([42, r["item_id"]])) for _, items in sorted(strata.items())]
    out = []
    while len(out) < n:
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def b_parity_check(b_dir, b_index_rows, plan, shard_dir, extractor, *, n=B_PARITY_N):
    """Re-extract n rows the B cache already holds and compare with B's stored vectors: shows this run reproduces B's numbers."""
    b_dir = Path(b_dir)
    selected = b_parity_rows(b_index_rows, set(plan.shards()), n)
    worst = {"mean": 0.0, "mean_std": 0.0}
    for shard in sorted({r["shard"] for r in selected}):
        with tarfile.open(Path(shard_dir) / shard, "r") as archive:
            for row in [r for r in selected if r["shard"] == shard]:
                wave = r2_u1._read_wave(archive, row)[row["start_sample"]:row["end_sample"]]
                with np.load(b_dir / row["pack"], allow_pickle=False) as pack:
                    _require(str(pack["keys_mean_std"][row["pack_row"]]) == row["key_mean_std"], "B index row does not match its B pack")
                    for mode in ("mean", "mean_std"):
                        stored = pack[mode][row["pack_row"]:row["pack_row"] + 1]
                        direct = np.asarray(extractor.embed([wave], pooling=mode))
                        _require(direct.shape == stored.shape and np.isfinite(direct).all(), "invalid direct embedding in the B parity check")
                        worst[mode] = max(worst[mode], float(np.max(np.abs(direct - stored))))
                        _require(np.allclose(direct, stored, atol=ATOL, rtol=RTOL),
                                 f"B parity failed for {row['item_id']} ({mode}): this run does not reproduce the stored B vector")
    return {"passed": True, "items": n, "item_ids": [r["item_id"] for r in selected], "atol": ATOL, "rtol": RTOL,
            "max_abs_embedding_difference": worst, "scope": "development rows already held by the B cache"}


def identity_vs_b(identity, b_identity):
    keys = sorted(set(identity) | set(b_identity))
    diff = {k: {"a": identity.get(k), "b": b_identity.get(k)} for k in keys if identity.get(k) != b_identity.get(k)}
    return {"equal": not diff, "differences": diff}


def verify_cache(plan, out_dir):
    """Packs, keys, finite vectors, hashes and (when published) the manifest and index."""
    _validate_plan(plan)
    out_dir = Path(out_dir)
    expected = {r2_u1._pack_path(out_dir, shard).name for shard in plan.shards()}
    existing = {p.name for p in (out_dir / "parts").glob("*.npz")}
    _require(expected == existing, "missing or unexpected cache packs")
    packs = {}
    for shard in plan.shards():
        path = r2_u1._pack_path(out_dir, shard)
        _, meta = r2_u1._read_pack(plan, path, shard)
        packs[shard] = {"file": path.relative_to(out_dir).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size,
                        "extraction_seconds": meta["extraction_seconds"]}
    final = out_dir / MANIFEST_NAME
    if final.exists():
        m = _load_json(final)
        _require(m.get("task") == TASK and m.get("schema_version") == 1, "invalid manifest task/schema")
        _require(m.get("plan_sha256") == plan.sha256 and m.get("identity") == plan.identity and m.get("counts") == plan.counts
                 and m.get("provenance") == plan.provenance, "manifest identity/scope/counts mismatch")
        _require(m.get("status") in {"fixture_verified", "extracted_pending_independent_review"}, "unpublished cache status")
        _require(m.get("verification", {}).get("passed") is True and m["verification"].get("packs") == packs, "pack hashes differ from manifest")
        parity, bpar = m.get("parity", {}), m.get("b_parity", {})
        n = parity.get("items")
        _require(parity.get("passed") is True and parity.get("scope") == "development only" and type(n) is int and n > 0
                 and parity.get("item_ids") == [r["item_id"] for r in r2_u1._parity_rows(plan, n)], "missing/invalid parity gate")
        _require(bpar.get("passed") is True and type(bpar.get("items")) is int and bpar["items"] > 0, "missing/invalid B parity gate")
        index_path = out_dir / INDEX_NAME
        payload = r2_u1._index_payload(plan, "development")
        info = m.get("index", {})
        _require(info.get("file") == INDEX_NAME and info.get("rows") == len(payload["rows"]) and info.get("sha256") == file_digest(index_path)
                 and _load_json(index_path) == payload, "index checksum/count mismatch")
    return {"passed": True, "packs": packs, "counts": plan.counts}


# ---- build -------------------------------------------------------------------------------------------------------------------

def build(plan, shard_dir, out_dir, extractor, *, b_dir, b_index_rows, resume=False, batch_size=16, parity_n=50, b_parity_n=B_PARITY_N,
          fixture=False, progress=None):
    """Extract/resume the gap packs, verify, run both parity checks, then publish the manifest. Never overwrites a finished cache."""
    _require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
    _validate_plan(plan)
    _require(fixture or (parity_n == 50 and b_parity_n == B_PARITY_N), "real extraction requires 50 + 50 parity items")
    _require(int(extractor.dim) == plan.identity["hidden_size"], "extractor hidden size differs from cache identity")
    if not fixture:
        _require(plan.identity["hidden_size"] == 768 and plan.identity["repo_id"] == r2_u1.REPO_ID
                 and plan.identity["revision"] == r2_u1.REVISION and plan.identity["weights_sha256"] == r2_u1.WEIGHTS_SHA256,
                 "production identity differs from the pinned checkpoint")
        seen = extractor.describe()
        _require(seen["trainable_parameters"] == 0 and seen["training_mode"] is False and seen["do_normalize"] is False,
                 "extractor is not frozen/eval with pinned preprocessing")
        import importlib.metadata
        for package, want in plan.identity["versions"].items():
            _require(importlib.metadata.version(package) == want, "observed package version differs from cache identity")
        for filename in ("pretrained.py", "r2_u1.py"):
            values = [v for n, v in plan.identity["source_sha256"].items() if Path(n).name == filename]
            _require(values == [file_digest(Path(r2_u1.__file__).with_name(filename))], "extractor source differs from cache identity")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / ".writer.lock"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError as exc:
        raise CacheError("cache writer lock exists; confirm the prior writer stopped before removing it") from exc
    say = progress or (lambda *_: None)
    started = time.perf_counter()
    try:
        manifest_path = out_dir / MANIFEST_NAME
        if manifest_path.exists():
            _require(resume, "finished cache exists; use resume to verify it")
            manifest = _load_json(manifest_path)
            _require(manifest["plan_sha256"] == plan.sha256, "finished cache has a different plan")
            _require(fixture == (manifest.get("status") == "fixture_verified"), "fixture output cannot be promoted to a production run")
            verify_cache(plan, out_dir)
            return manifest
        _require(resume or not list((out_dir / "parts").glob("*.npz")), "partial cache exists; use resume")
        reused, extracted = [], []
        for shard in plan.shards():
            path = r2_u1._pack_path(out_dir, shard)
            if path.exists():
                r2_u1._read_pack(plan, path, shard)
                reused.append(shard)
                say(f"{shard}: validated existing pack")
            else:
                arrays, seconds = r2_u1._extract_pack(plan, shard_dir, shard, extractor, batch_size)
                meta = {"plan_sha256": plan.sha256, "identity": plan.identity, "shard": shard,
                        "array_sha256": r2_u1._arrays_hash(arrays), "extraction_seconds": seconds}
                r2_u1._atomic(path, lambda handle: np.savez(handle, metadata=np.asarray(canonical_json(meta)), **arrays))
                r2_u1._read_pack(plan, path, shard)
                extracted.append(shard)
                say(f"{shard}: extracted and validated")
        verification = verify_cache(plan, out_dir)
        parity = r2_u1.parity_check(plan, shard_dir, out_dir, extractor, n=parity_n)
        say("parity of gap rows (cached vs direct): passed")
        b_parity = b_parity_check(b_dir, b_index_rows, plan, shard_dir, extractor, n=b_parity_n)
        say("parity against the stored B vectors: passed")
        payload = r2_u1._index_payload(plan, "development")
        index_path = out_dir / INDEX_NAME
        if index_path.exists():
            _require(_load_json(index_path) == payload, "existing index differs from the resumed plan")
        else:
            r2_u1._json_new(index_path, payload)
        index = {"file": INDEX_NAME, "sha256": file_digest(index_path), "rows": len(payload["rows"]), "bytes": index_path.stat().st_size}
        pack_bytes = sum(p["bytes"] for p in verification["packs"].values())
        b_identity = plan.provenance["b_cache_identity"]
        manifest = {"schema_version": 1, "task": TASK, "plan_sha256": plan.sha256, "identity": plan.identity,
                    "identity_vs_b": identity_vs_b(plan.identity, b_identity), "provenance": plan.provenance, "counts": plan.counts,
                    "verification": verification, "parity": parity, "b_parity": b_parity, "index": index,
                    "extraction_seconds_all_packs": sum(p["extraction_seconds"] for p in verification["packs"].values()),
                    "seconds_this_session": time.perf_counter() - started, "packs_extracted_this_session": extracted,
                    "packs_reused_this_session": reused, "pack_bytes": pack_bytes, "index_bytes": index["bytes"],
                    "cache_bytes": pack_bytes + index["bytes"], "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "environment": {"python": platform.python_version(), "numpy": np.__version__},
                    "source_sha256": {"r4_u0.py": file_digest(Path(__file__))},
                    "status": "fixture_verified" if fixture else "extracted_pending_independent_review", "research_accuracy_claim": False,
                    "test_rows_extracted": 0}
        r2_u1._json_new(manifest_path, manifest)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


# ---- reading the union -------------------------------------------------------------------------------------------------------

def read_embeddings_a(root, *, partition=None, pooling="mean", a_dir=None, b_dir=None):
    """Embeddings of every A train crop and A validation chunk, from the B cache or the A gap cache, in the A crop-bank / chunk order.

    Returns (rows, matrix). rows carry the A item ids, labels, groups and roles. Refuses test, a missing or unverified manifest,
    and an A cache whose identity differs from B's unless its B parity gate passed.
    """
    from echoproof.contracts import Chunk, read_parquet
    _require(pooling in ("mean", "mean_std"), "unknown pooling mode")
    _require(partition in (None, "train", "validation"), "partition must be train or validation (A test is sealed)")
    root = Path(root)
    a_dir, b_dir = Path(a_dir or root / A_DIR), Path(b_dir or root / B_DIR)
    am, bm = _load_json(a_dir / MANIFEST_NAME), _load_json(b_dir / "r2_u1_manifest.json")
    _require(am.get("task") == TASK and am["verification"]["passed"] is True and am["parity"]["passed"] is True
             and am["b_parity"]["passed"] is True and am.get("test_rows_extracted") == 0, "A cache manifest is not verified")
    _require(am["identity_vs_b"]["equal"] or am["b_parity"]["passed"], "A cache identity differs from B and was not shown to reproduce B")
    _require(file_digest(a_dir / INDEX_NAME) == am["index"]["sha256"], "A index checksum mismatch")
    _require(file_digest(b_dir / "development_index.json") == bm["indices"]["development"]["sha256"], "B index checksum mismatch")
    a_index, b_index = _load_json(a_dir / INDEX_NAME), _load_json(b_dir / "development_index.json")
    _require(a_index["plan_sha256"] == am["plan_sha256"], "A index belongs to a different plan")
    s1 = root / S1_DIR
    s1m = _load_json(s1 / "r4_s1_manifest.json")
    split_path, chunk_path = root / "data/splits/protocol_A_v1.parquet", root / "data/processed/chunks_v1.parquet"
    crop_path = s1 / s1m["outputs"]["crop_bank"]["file"]
    _require(file_digest(crop_path) == s1m["outputs"]["crop_bank"]["sha256"], "A crop bank differs from the R4-S1 manifest")
    assignments = splits.load_assignments(split_path, sealed=False)
    chunks = read_parquet(str(chunk_path), Chunk)
    universe, _, _, _ = a_universe(assignments, chunks, list(read_table(crop_path)))
    where = {}
    for tag, directory, index in (("b", b_dir, b_index), ("a", a_dir, a_index)):
        for r in index["rows"]:
            where[_coord(r["kind"], r["recording_id"], r["start_sample"], r["end_sample"])] = (tag, directory, r)
    ordered = [k for k in universe if partition is None or universe[k]["partition"] == partition]
    missing = [k for k in ordered if k not in where]
    _require(not missing, f"{len(missing)} A rows are held by neither cache")
    width = am["identity"]["hidden_size"] * (2 if pooling == "mean_std" else 1)
    matrix = np.empty((len(ordered), width), dtype=np.float32)
    directories = {"a": a_dir, "b": b_dir}
    rows, groups = [], defaultdict(list)
    for i, key in enumerate(ordered):
        tag, _, r = where[key]
        u = universe[key]
        rows.append({"item_id": u["item_id"], "kind": key[0], "recording_id": key[1], "group_id": u["group_id"], "partition": u["partition"],
                     "role": u["role"], "component": r["component"], "label": r["label"], "start_sample": key[2], "end_sample": key[3],
                     "source": tag})
        groups[(tag, r["pack"], r["shard"])].append((i, r))
    packs = {"b": bm["verification"]["packs"], "a": am["verification"]["packs"]}
    for (tag, pack_rel, shard), items in groups.items():
        _require(file_digest(directories[tag] / pack_rel) == packs[tag][shard]["sha256"], f"{tag} cache pack checksum mismatch ({shard})")
        with np.load(directories[tag] / pack_rel, allow_pickle=False) as pack:
            vectors, keys = pack[pooling], pack["keys_" + pooling]      # NpzFile re-reads an array on every access: load each once
        _require(vectors.dtype == np.float32 and vectors.shape[1] == width and np.isfinite(vectors).all(), "invalid vectors")
        for i, r in items:
            _require(str(keys[r["pack_row"]]) == r["key_" + pooling], "cache row/key mismatch")
            matrix[i] = vectors[r["pack_row"]]
    return rows, matrix
