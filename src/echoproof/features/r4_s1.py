"""R4-S1 (Protocol A): train-fitted log-mel statistics, the fixed crop bank and the handcrafted LR feature tables for Protocol A.

Same computation as R2-S1 (`echoproof.features.r2_s1`), different scope:
  * mel statistics   - every valid frame of every retained A-train recording (all five components, including xtts and ChatterboxTTS);
  * train features   - the K fixed crops of every retained A-train recording (`features.stats.crop_bank_starts`, same seed and K as B,
                       so recordings shared with B get the same crops);
  * validation       - every manifest chunk of every retained A-validation recording (all three roles).
A test assignments are counted and dropped by the loader (`sealed=False`); no A test row is read, listed or written. The part
reading, merging and spot check are the R2-S1 functions (protocol-independent); shard processing is re-stated with the A crop
prefix because r2_s1 is a frozen B source file. This module adds the Protocol A plan, its verification, its outputs, the cross-check against the saved B features and the WavLM cache gap list.
Nothing here writes into a Protocol B directory; B outputs are only read, for the cross-check and the gap list.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tarfile
import time

import numpy as np

from echoproof.data import splits
from echoproof.data.chunks import eval_windows
from echoproof.data.cohort import read_table, write_table
from echoproof.features import r2_s1
from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH, SCHEMA as MEL_SCHEMA, handcrafted, log_mel
from echoproof.features.stats import BANK_VERSION, CROP_SAMPLES, DEFAULT_K, DEFAULT_SEED, MelStats, crop_bank_starts

PROTOCOL_ID = "protocol:A"
TASK = "R4-S1"
CROP_PREFIX = "crop:A"
MANIFEST_NAME = "r4_s1_manifest.json"
OUTPUT_NAMES = {"crop_bank": "crop_bank_A_v1.parquet", "mel_stats": "mel_stats_A_v1.npz",
                "train_features": "lr_train_crops_A_v1.npz", "validation_features": "lr_validation_A_v1.npz"}
GAP_NAME = "wavlm_cache_gap_A_v1.json"
CROSSCHECK_NAME = "r4_s1_vs_b_crosscheck.json"
R2_S1_B_DIR = Path("data") / "features" / "r2_s1_B"
R2_U1_B_INDEX = Path("data") / "features" / "r2_u1_B" / "development_index.json"
# Tolerance for the A-versus-B feature cross-check: exact when the environment matches, reported (not hidden) when it does not.
CROSSCHECK_ATOL = 1e-9


class R4S1Error(r2_s1.R2S1Error):
    """A rule of the R4-S1 data scope was violated, or an output failed verification."""


def _allowed_components():
    return {"train": set(splits.PROTOCOL_A[("train", None)]), "validation": set(splits.PROTOCOL_A[("validation", "*")])}


class PlanA(r2_s1.Plan):
    """The R2-S1 plan object (jobs, digest, shard list); only the crop-id prefix of the shard processing differs."""


def plan_a(cohort_rows, assignments, chunks, *, k: int = DEFAULT_K, seed: int = DEFAULT_SEED, crop: int = CROP_SAMPLES) -> PlanA:
    """Select exactly the rows R4-S1 may touch. Test assignments are counted and dropped; component rules are checked, not assumed."""
    by_id = {r["recording_id"]: r for r in cohort_rows}
    allowed = _allowed_components()
    train, validation, unretained, dropped_test, seen = [], {}, [], 0, set()
    validation_ids = set()
    for a in assignments:
        if a.protocol_id != PROTOCOL_ID:
            raise R4S1Error(f"assignment {a.recording_id} belongs to {a.protocol_id}, not {PROTOCOL_ID}")
        if a.recording_id in seen:
            raise R4S1Error(f"duplicate assignment {a.recording_id}")
        seen.add(a.recording_id)
        if a.partition == "test":
            dropped_test += 1
            continue
        if a.partition not in allowed:
            raise R4S1Error(f"unknown partition {a.partition!r}")
        row = by_id.get(a.recording_id)
        if row is None:
            raise R4S1Error(f"assignment without a cohort row: {a.recording_id}")
        if row["component"] not in allowed[a.partition]:
            raise R4S1Error(f"{row['component']} recordings may not appear in A {a.partition} ({a.recording_id})")
        if a.group_id != f"speaker:{row['speaker_id']}":
            raise R4S1Error(f"group {a.group_id} does not match speaker of {a.recording_id}")
        if a.partition == "validation":
            validation_ids.add(a.recording_id)
            if a.validation_role not in splits.ROLES:
                raise R4S1Error(f"validation recording {a.recording_id} has no role")
        if not row["retained"]:
            unretained.append(a.recording_id)
            continue
        shard = row["canonical_path"].split("::", 1)[0]
        item = {"recording_id": a.recording_id, "group_id": a.group_id, "component": row["component"], "label": int(row["label"]),
                "n_samples": int(row["n_samples"]), "pcm_sha256": row["pcm_sha256"], "shard": shard}
        if a.partition == "train":
            item["starts"] = crop_bank_starts(seed, a.recording_id, item["n_samples"], k=k, crop=crop)
            train.append(item)
        else:
            validation[a.recording_id] = {**item, "role": a.validation_role, "chunks": []}
    for c in chunks:
        if c.protocol_id != PROTOCOL_ID or c.partition != "validation":
            continue
        if c.recording_id not in validation_ids:
            raise R4S1Error(f"validation chunk {c.chunk_id} has no validation assignment")
        item = validation.get(c.recording_id)
        if item is not None:
            item["chunks"].append([c.chunk_id, int(c.start_sample), int(c.end_sample)])
    for item in validation.values():
        item["chunks"].sort(key=lambda ch: (ch[1], ch[0]))
        if [(s, e) for _, s, e in item["chunks"]] != eval_windows(item["n_samples"]):
            raise R4S1Error(f"chunks of {item['recording_id']} do not equal the canonical windows of its audio")
    train.sort(key=lambda t: t["recording_id"])
    validation = sorted(validation.values(), key=lambda v: v["recording_id"])
    crop_bank = []
    for t in train:
        for start in t["starts"]:
            valid = min(crop, t["n_samples"] - start)
            crop_bank.append({"crop_id": f"{CROP_PREFIX}:{t['recording_id']}:{start}", "recording_id": t["recording_id"], "group_id": t["group_id"],
                              "component": t["component"], "label": t["label"], "start_sample": start, "end_sample": start + valid,
                              "valid_samples": valid, "bank_version": BANK_VERSION, "bank_seed": seed, "bank_k": k})
    counts = {"train_recordings": len(train), "train_by_component": dict(sorted(Counter(t["component"] for t in train).items())),
              "train_crops": len(crop_bank), "validation_recordings": len(validation),
              "validation_by_component": dict(sorted(Counter(v["component"] for v in validation).items())),
              "validation_chunks": sum(len(v["chunks"]) for v in validation),
              "unretained_skipped": sorted(unretained), "test_assignments_dropped": dropped_test}
    by_role = Counter()
    for v in validation:
        by_role[v["role"]] += len(v["chunks"])
    counts["validation_chunks_by_role"] = dict(sorted(by_role.items()))
    if not train or not validation:
        raise R4S1Error("plan has no train or no validation recordings")
    settings = {"k": k, "seed": seed, "crop_samples": crop, "bank_version": BANK_VERSION, "protocol": PROTOCOL_ID}
    return PlanA(train, validation, crop_bank, counts, settings)


# ---- one shard (A) -----------------------------------------------------------------------------------------------------------
# Same computation as r2_s1.process_shard / run_jobs. They are re-stated here, not imported, only because r2_s1 is a frozen Protocol B
# source file (hashed by the GR3 lock) and its crop ids are written as "crop:B:". Everything else (reading, part format, merging) is shared.

def process_shard(job: dict) -> dict:
    """Read each needed recording once; accumulate mel statistics, crop features and validation chunk features; write the part atomically."""
    started = time.perf_counter()
    stats = MelStats()
    rec_ids, rec_frames = [], []
    crop_ids, crop_rec, crop_start, crop_valid, crop_x = [], [], [], [], []
    val_ids, val_rec, val_start, val_end, val_x = [], [], [], [], []
    with tarfile.open(job["shard_path"], "r") as tar:
        for item in sorted(job["train"], key=lambda i: i["recording_id"]):
            wave = r2_s1._read_member(tar, item)  # noqa: SLF001
            rec_ids.append(item["recording_id"])
            rec_frames.append(stats.add(log_mel(wave)))
            for start in item["starts"]:
                piece = wave[start:start + job["crop"]]
                crop_ids.append(f"{CROP_PREFIX}:{item['recording_id']}:{start}")
                crop_rec.append(item["recording_id"])
                crop_start.append(start)
                crop_valid.append(len(piece))
                crop_x.append(handcrafted(piece))
        for item in sorted(job["validation"], key=lambda i: i["recording_id"]):
            wave = r2_s1._read_member(tar, item)  # noqa: SLF001
            for chunk_id, start, end in item["chunks"]:
                val_ids.append(chunk_id)
                val_rec.append(item["recording_id"])
                val_start.append(start)
                val_end.append(end)
                val_x.append(handcrafted(wave[start:end]))
    width, text = len(FEATURE_NAMES), r2_s1._text  # noqa: SLF001
    arrays = {"meta_plan_sha256": np.array(job["plan_sha256"]), "meta_feature_schema_hash": np.array(FEATURE_SCHEMA_HASH),
              "meta_shard": np.array(job["shard"]), **stats.to_arrays(),
              "rec_ids": text(rec_ids), "rec_frames": np.array(rec_frames, dtype=np.int64),
              "crop_ids": text(crop_ids), "crop_recording": text(crop_rec), "crop_start": np.array(crop_start, dtype=np.int64),
              "crop_valid": np.array(crop_valid, dtype=np.int64), "crop_X": np.array(crop_x, dtype=np.float64).reshape(len(crop_x), width),
              "val_ids": text(val_ids), "val_recording": text(val_rec), "val_start": np.array(val_start, dtype=np.int64),
              "val_end": np.array(val_end, dtype=np.int64), "val_X": np.array(val_x, dtype=np.float64).reshape(len(val_x), width)}
    out = Path(job["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, out)
    return {"shard": job["shard"], "seconds": round(time.perf_counter() - started, 2), "train_recordings": len(rec_ids),
            "crops": len(crop_ids), "validation_chunks": len(val_ids), "mel_frames": int(sum(rec_frames))}


def run_jobs(jobs, *, workers: int = 1, resume: bool = False, progress=None):
    """Run every shard job; returns (summaries, failures). A failed shard does not stop the others, so a rerun only redoes it."""
    pending, summaries, failures = [], [], {}
    for job in jobs:
        if resume and r2_s1.part_is_current(job):
            summaries.append({"shard": job["shard"], "skipped": True})
        else:
            pending.append(job)
    if workers <= 1 or len(pending) <= 1:
        for job in pending:
            try:
                summaries.append(process_shard(job))
            except Exception as exc:     # noqa: BLE001 - reported with the shard name
                failures[job["shard"]] = f"{type(exc).__name__}: {exc}"
            if progress:
                progress(job["shard"], failures.get(job["shard"]))
    else:
        import concurrent.futures as cf
        import multiprocessing
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(process_shard, job): job for job in sorted(pending, key=lambda j: -(len(j["train"]) + len(j["validation"])))}
            for future in cf.as_completed(futures):
                job = futures[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures[job["shard"]] = f"{type(exc).__name__}: {exc}"
                if progress:
                    progress(job["shard"], failures.get(job["shard"]))
    return sorted(summaries, key=lambda s: s["shard"]), failures


def verify_a(plan: PlanA, tables, stats: MelStats, frames) -> dict:
    """Independent checks on the merged results; returns {'passed', 'problems', details}. Nothing here trusts how the inputs were filtered."""
    problems = []
    train, val = tables["train"], tables["validation"]
    width = len(FEATURE_NAMES)
    for name, X in (("train", train["X"]), ("validation", val["X"])):
        if X.ndim != 2 or X.shape[1] != width:
            problems.append(f"{name} feature matrix has shape {X.shape}")
        if np.isinf(X).any():
            problems.append(f"{name} features contain infinities")
    nan_columns = {}
    for name, X in (("train", train["X"]), ("validation", val["X"])):
        for j in np.flatnonzero(np.isnan(X).any(axis=0)):
            nan_columns.setdefault(FEATURE_NAMES[j], {})[name] = int(np.isnan(X[:, j]).sum())
    unexpected = sorted(set(nan_columns) - r2_s1.EXPECTED_NAN_COLUMNS)
    if unexpected:
        problems.append(f"NaN in unexpected feature columns: {unexpected}")
    allowed = _allowed_components()
    train_components, val_components = set(train["component"].tolist()), set(val["component"].tolist())
    if not train_components <= allowed["train"] or not val_components <= allowed["validation"]:
        problems.append("components outside the Protocol A train or validation set")
    for name, got in (("train", train_components), ("validation", val_components)):
        if got != allowed["train" if name == "train" else "validation"]:
            problems.append(f"{name} is missing components: {sorted(allowed['train' if name == 'train' else 'validation'] - got)}")
    if any(not str(c).startswith(f"{CROP_PREFIX}:") for c in train["crop_id"].tolist()):
        problems.append("a crop id does not carry the Protocol A prefix")
    if any(not str(c).startswith("chunk:A:") for c in val["chunk_id"].tolist()):
        problems.append("a validation chunk id is not a Protocol A chunk")
    if set(train["group_id"].tolist()) & set(val["group_id"].tolist()):
        problems.append("a speaker appears in both train and validation outputs")
    if set(train["y"].tolist()) != {0, 1} or set(val["y"].tolist()) != {0, 1}:
        problems.append("train or validation is missing a class")
    if any(not (0 < v <= plan.settings["crop_samples"]) for v in train["valid_samples"].tolist()):
        problems.append("a crop has an invalid length")
    expected_frames = sum(r2_s1.expected_mel_frames(t["n_samples"]) for t in plan.train)
    if stats.count != expected_frames or sum(frames.values()) != expected_frames:
        problems.append(f"mel frames {stats.count} (parts {sum(frames.values())}) differ from the manifest-implied {expected_frames}")
    if stats.recordings != len(plan.train) or set(frames) != {t["recording_id"] for t in plan.train}:
        problems.append("mel statistics do not cover exactly the train recordings")
    mean, std = stats.finalize()
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        problems.append("mel statistics are not finite and positive")
    by_role = Counter(val["role"].tolist())
    if dict(sorted(by_role.items())) != plan.counts["validation_chunks_by_role"]:
        problems.append("validation chunk counts per role differ from the plan")
    planned = {r["recording_id"] for r in plan.train} | {v["recording_id"] for v in plan.validation}
    if not (set(train["recording_id"].tolist()) | set(val["recording_id"].tolist())) <= planned:
        problems.append("an output row belongs to a recording outside the planned train and validation sets (test leakage)")
    return {"passed": not problems, "problems": problems, "nan_columns": nan_columns, "mel_frames": stats.count,
            "mel_std_min": float(std.min()), "mel_std_max": float(std.max()),
            "rows": {"train_crops": int(len(train["y"])), "validation_chunks": int(len(val["y"]))}}


def write_crop_bank(out_dir, plan: PlanA) -> Path:
    """The crop bank depends only on the manifests and the seed, so it is written before any audio is touched."""
    path = Path(out_dir) / OUTPUT_NAMES["crop_bank"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read_table(path)
        if [dict(r) for r in existing] != plan.crop_bank:
            raise R4S1Error(f"{path} exists and differs from the crop bank of this plan")
        return path
    write_table(plan.crop_bank, path)
    return path


def write_outputs(out_dir, plan: PlanA, tables, stats: MelStats, provenance: dict) -> dict:
    """Write the feature/statistics files (never overwriting) and finally the manifest that hashes them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = stats.finalize()
    save = r2_s1._save_npz_new  # noqa: SLF001 - refuses to overwrite; shared on purpose
    save(out_dir / OUTPUT_NAMES["mel_stats"], {"mean": mean, "std": std, "frames": np.int64(stats.count), "recordings": np.int64(stats.recordings),
                                               "n_mels": np.int64(len(mean)), "plan_sha256": np.array(plan.sha256)})
    meta = {"feature_names": r2_s1._text(list(FEATURE_NAMES)), "plan_sha256": np.array(plan.sha256), "feature_schema_hash": np.array(FEATURE_SCHEMA_HASH)}  # noqa: SLF001
    save(out_dir / OUTPUT_NAMES["train_features"], {**tables["train"], **meta})
    save(out_dir / OUTPUT_NAMES["validation_features"], {**tables["validation"], **meta})
    outputs = {}
    for key, name in OUTPUT_NAMES.items():
        path = out_dir / name
        if path.exists():
            outputs[key] = {"file": name, "sha256": r2_s1.file_sha256(path), "bytes": path.stat().st_size}
    manifest = {"schema_version": 1, "task": TASK, "protocol": "A", "plan_sha256": plan.sha256, "outputs": outputs, **provenance}
    path = out_dir / MANIFEST_NAME
    if path.exists():
        raise R4S1Error(f"refusing to overwrite {path}")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


# ---- cross-check against the saved Protocol B features -----------------------------------------------------------------------

def crosscheck_against_b(tables, b_dir, *, atol: float = CROSSCHECK_ATOL) -> dict:
    """Read-only comparison of A rows with the saved B rows for the same (recording, window). B files are opened for reading only."""
    b_dir = Path(b_dir)
    report = {"b_dir": str(b_dir), "atol": atol}
    for kind, a, name, key in (("train_crops", tables["train"], "lr_train_crops_B_v1.npz", ("recording_id", "start")),
                               ("validation_chunks", tables["validation"], "lr_validation_B_v1.npz", ("recording_id", "start", "end"))):
        with np.load(b_dir / name, allow_pickle=False) as z:
            b = {k: z[k] for k in z.files}
        index_b = {tuple(str(b[f][i]) if f == "recording_id" else int(b[f][i]) for f in key): i for i in range(len(b["y"]))}
        worst = 0.0
        matched = 0
        differing = []
        nan_pattern_differs = 0
        missing_in_a = 0
        index_a = {tuple(str(a[f][i]) if f == "recording_id" else int(a[f][i]) for f in key): i for i in range(len(a["y"]))}
        for k, ib in index_b.items():
            ia = index_a.get(k)
            if ia is None:
                missing_in_a += 1
                continue
            matched += 1
            xa, xb = a["X"][ia], b["X"][ib]
            if not np.array_equal(np.isnan(xa), np.isnan(xb)):
                nan_pattern_differs += 1
                continue
            diff = float(np.nanmax(np.abs(xa - xb))) if np.isfinite(xa).any() else 0.0
            worst = max(worst, diff)
            if diff > atol or int(a["y"][ia]) != int(b["y"][ib]):
                differing.append(list(map(str, k)))
        report[kind] = {"b_rows": len(index_b), "a_rows": len(index_a), "matched": matched, "b_rows_missing_in_a": missing_in_a,
                        "nan_pattern_differs": nan_pattern_differs, "rows_over_tolerance": len(differing), "max_abs_difference": worst,
                        "examples_over_tolerance": differing[:10]}
    report["passed"] = all(report[k]["b_rows_missing_in_a"] == 0 and report[k]["nan_pattern_differs"] == 0 and report[k]["rows_over_tolerance"] == 0
                           for k in ("train_crops", "validation_chunks"))
    return report


def mel_stats_shift_vs_b(a_mean, a_std, b_stats_path) -> dict:
    with np.load(b_stats_path, allow_pickle=False) as z:
        b_mean, b_std, b_frames = z["mean"], z["std"], int(z["frames"])
    return {"b_frames": b_frames, "max_abs_mean_shift": float(np.abs(a_mean - b_mean).max()), "max_abs_std_shift": float(np.abs(a_std - b_std).max()),
            "max_rel_std_shift": float((np.abs(a_std - b_std) / b_std).max())}


# ---- WavLM cache gap list for R4-U0 ------------------------------------------------------------------------------------------

def wavlm_gap(plan: PlanA, b_index_path) -> dict:
    """A crops and validation chunks that the Protocol B WavLM cache (R2-U1 development index) does not hold, by (kind, recording, window)."""
    index = json.loads(Path(b_index_path).read_text(encoding="utf-8"))
    have = {(r["kind"], r["recording_id"], int(r["start_sample"]), int(r["end_sample"])) for r in index["rows"]}
    rows = []
    for c in plan.crop_bank:
        if ("crop", c["recording_id"], c["start_sample"], c["end_sample"]) not in have:
            rows.append({"kind": "crop", "item_id": c["crop_id"], "recording_id": c["recording_id"], "group_id": c["group_id"], "component": c["component"],
                         "label": c["label"], "partition": "train", "role": None, "start_sample": c["start_sample"], "end_sample": c["end_sample"],
                         "valid_samples": c["valid_samples"], "shard": next(t["shard"] for t in plan.train if t["recording_id"] == c["recording_id"])})
    for v in plan.validation:
        for chunk_id, start, end in v["chunks"]:
            if ("chunk", v["recording_id"], start, end) not in have:
                rows.append({"kind": "chunk", "item_id": chunk_id, "recording_id": v["recording_id"], "group_id": v["group_id"], "component": v["component"],
                             "label": v["label"], "partition": "validation", "role": v["role"], "start_sample": start, "end_sample": end,
                             "valid_samples": min(end, v["n_samples"]) - start, "shard": v["shard"]})
    rows.sort(key=lambda r: (r["kind"], r["item_id"]))
    summary = Counter((r["kind"], r["partition"], r["component"]) for r in rows)
    return {"schema_version": 1, "task": "R4-U0-input", "scope": "protocol A train and validation only; no test chunks",
            "b_cache_index_plan_sha256": index["plan_sha256"], "plan_sha256": plan.sha256,
            "summary": {"|".join(map(str, k)): v for k, v in sorted(summary.items())}, "rows": rows}


# ---- orchestration -----------------------------------------------------------------------------------------------------------

def build(root, shard_dir, out_dir=None, *, k: int = DEFAULT_K, seed: int = DEFAULT_SEED, workers: int = 1, resume: bool = False,
          spot_n: int = 25, progress=None) -> dict:
    """Run R4-S1 end to end against the locked manifests under `root`; returns the manifest dict it also writes to out_dir."""
    import platform
    import librosa
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data.chunks import PREPROCESSING_HASH
    from echoproof.training.datasets import ShardWaveformLoader
    say = progress or (lambda *_: None)
    t0 = time.perf_counter()
    root, shard_dir = Path(root), Path(shard_dir)
    out_dir = Path(out_dir) if out_dir else root / "data" / "features" / "r4_s1_A"
    parts_dir = out_dir / "parts"
    if (out_dir / MANIFEST_NAME).exists():
        raise R4S1Error(f"{out_dir / MANIFEST_NAME} exists: R4-S1 is already complete for this directory")
    if any((out_dir / name).exists() for key, name in OUTPUT_NAMES.items() if key != "crop_bank"):
        raise R4S1Error("final outputs exist without a manifest; inspect them before rerunning")
    if parts_dir.exists() and any(parts_dir.glob("part_*.npz")) and not resume:
        raise R4S1Error("part files exist from an earlier run; pass resume=True to continue it")

    paths = {"cohort": root / "data" / "manifests" / "cohort_v1.parquet", "split": root / "data" / "splits" / "protocol_A_v1.parquet",
             "chunks": root / "data" / "processed" / "chunks_v1.parquet"}
    for name, path in paths.items():
        if not path.exists():
            raise R4S1Error(f"missing {name} input: {path}")
    cohort_rows = read_table(paths["cohort"])
    assignments = splits.load_assignments(paths["split"], sealed=False)        # test rows are withheld by the loader
    split_hashes = {a.split_hash for a in assignments}
    if len(split_hashes) != 1:
        raise R4S1Error("protocol A assignments carry more than one split hash")
    split_hash = split_hashes.pop()
    report_path = root / "reports" / "r1" / "split_report.json"
    if not report_path.exists():
        raise R4S1Error(f"missing frozen split report {report_path}; the frozen A split hash cannot be checked")
    frozen = json.loads(report_path.read_text(encoding="utf-8"))["protocols"]["A"]["split_sha256"]
    if frozen != split_hash:
        raise R4S1Error(f"protocol A split hash {split_hash[:12]} differs from the frozen value {frozen[:12]} in split_report.json")
    chunks = read_parquet(str(paths["chunks"]), Chunk)
    if {c.preprocessing_hash for c in chunks} != {PREPROCESSING_HASH}:
        raise R4S1Error("chunk manifest preprocessing hash differs from the code's PREPROCESSING_HASH")

    plan = plan_a(cohort_rows, assignments, chunks, k=k, seed=seed)
    say(f"plan {plan.sha256[:12]}: {plan.counts['train_recordings']} train recordings -> {plan.counts['train_crops']} crops, "
        f"{plan.counts['validation_recordings']} validation recordings -> {plan.counts['validation_chunks']} chunks")
    missing = [s for s in plan.shards() if not (shard_dir / s).is_file()]
    if missing:
        raise R4S1Error(f"shard files not found in {shard_dir}: {missing}")
    write_crop_bank(out_dir, plan)

    summaries, failures = run_jobs(plan.jobs(shard_dir, parts_dir), workers=workers, resume=resume,
                                         progress=lambda shard, error: say(f"{shard}: {'FAILED ' + error if error else 'done'}"))
    if failures:
        raise R4S1Error(f"shards failed (rerun with resume=True after fixing): {failures}")
    tables, stats, frames = r2_s1.merge_parts(plan, parts_dir)
    verification = verify_a(plan, tables, stats, frames)
    if not verification["passed"]:
        raise R4S1Error(f"verification failed: {verification['problems']}")
    loader = ShardWaveformLoader(shard_dir, {r["recording_id"]: r["canonical_path"] for r in cohort_rows if r["retained"]})
    spot = r2_s1.spot_check(tables, loader, n=spot_n, seed=seed)
    loader.close()
    if spot["mismatches"] or spot["streaming_vs_two_pass_max_abs"] > 1e-9:
        raise R4S1Error(f"spot check failed: {spot}")
    mean, std = stats.finalize()
    crosscheck = None
    if (root / R2_S1_B_DIR / "lr_train_crops_B_v1.npz").exists():
        crosscheck = crosscheck_against_b(tables, root / R2_S1_B_DIR)
        crosscheck["mel_stats_shift_vs_b"] = mel_stats_shift_vs_b(mean, std, root / R2_S1_B_DIR / "mel_stats_B_v1.npz")
    gap = wavlm_gap(plan, root / R2_U1_B_INDEX) if (root / R2_U1_B_INDEX).exists() else None
    provenance = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"cohort_manifest_file_sha256": r2_s1.file_sha256(paths["cohort"]), "cohort_manifest_rows_digest": r2_s1.rows_digest(cohort_rows),
                   "split_file_sha256": r2_s1.file_sha256(paths["split"]), "split_sha256": split_hash, "split_sha256_frozen_in_report": frozen,
                   "chunk_manifest_file_sha256": r2_s1.file_sha256(paths["chunks"]), "preprocessing_hash": PREPROCESSING_HASH,
                   "feature_schema_hash": FEATURE_SCHEMA_HASH},
        "settings": {**plan.settings, "mel": MEL_SCHEMA, "feature_columns": len(FEATURE_NAMES),
                     "mel_scope": "all valid frames of every retained A-train recording, frame-equal weights, population std (ddof 0)",
                     "feature_scope": "A train crop bank and A validation chunks only; no test rows"},
        "counts": plan.counts, "verification": verification, "spot_check": spot,
        "crosscheck_vs_b_passed": None if crosscheck is None else crosscheck["passed"],
        "mel_stats": {"frames": stats.count, "recordings": stats.recordings, "mean_min": float(mean.min()), "mean_max": float(mean.max()),
                      "std_min": float(std.min()), "std_max": float(std.max())},
        "shards": summaries, "workers": workers, "resumed": bool(resume),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "librosa": librosa.__version__},
        "seconds_total": round(time.perf_counter() - t0, 1)}
    manifest = write_outputs(out_dir, plan, tables, stats, provenance)
    for name, payload in ((CROSSCHECK_NAME, crosscheck), (GAP_NAME, gap)):
        if payload is not None:
            target = out_dir / name
            if target.exists():
                raise R4S1Error(f"refusing to overwrite {target}")
            target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            manifest.setdefault("side_outputs", {})[name] = {"sha256": r2_s1.file_sha256(target), "bytes": target.stat().st_size}
    # The side outputs are hashed in a second, additive manifest so the first manifest stays exactly what write_outputs wrote.
    if "side_outputs" in manifest:
        side = out_dir / "r4_s1_side_outputs.json"
        side.write_text(json.dumps({"manifest_sha256": r2_s1.file_sha256(out_dir / MANIFEST_NAME), "side_outputs": manifest["side_outputs"]},
                                   indent=2, sort_keys=True), encoding="utf-8")
    return manifest
