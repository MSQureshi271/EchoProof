"""R2-S1 (Protocol B): train-fitted log-mel statistics, the fixed crop bank and the handcrafted LR feature tables.

What is computed, from which rows only:
  * mel statistics   - every valid frame of every retained B-train recording (full length, no cropping);
  * train features   - the K fixed crops of every retained B-train recording (the crop bank, `features.stats.crop_bank_starts`);
  * validation       - every manifest chunk of every retained B-validation recording (all three roles).
Nothing from the B test partition is read or listed, and nothing from ChatterboxTTS exists in B train or validation. Both rules
are enforced in `plan_b` and again in `verify`, not just by the way the inputs happen to be filtered.

Execution model: one pass per shard (each tar opened once). A shard job writes its own part file atomically, so an
interrupted Colab session resumes by skipping parts whose plan hash still matches. `merge_parts` reassembles the parts in plan
order, so the final tables do not depend on worker count or completion order.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

import numpy as np

from echoproof.artifacts import digest
from echoproof.data import splits
from echoproof.data.chunks import eval_windows
from echoproof.data.cohort import CohortError, SAMPLE_RATE, read_flac, read_table, write_table
from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH, SCHEMA as MEL_SCHEMA, handcrafted, log_mel
from echoproof.features.stats import BANK_VERSION, CROP_SAMPLES, DEFAULT_K, DEFAULT_SEED, MelStats, crop_bank_starts

PROTOCOL_ID = "protocol:B"
TASK = "R2-S1"
MANIFEST_NAME = "r2_s1_manifest.json"
OUTPUT_NAMES = {"crop_bank": "crop_bank_B_v1.parquet", "mel_stats": "mel_stats_B_v1.npz",
                "train_features": "lr_train_crops_B_v1.npz", "validation_features": "lr_validation_B_v1.npz"}
EXPECTED_NAN_COLUMNS = {"voiced_f0_mean", "voiced_f0_std"}     # unvoiced clips have no pitch; the LR imputer fills these


class R2S1Error(CohortError):
    """A rule of the R2-S1 data scope was violated, or an output failed verification."""


# ---- planning --------------------------------------------------------------------------------------------------------------

def _allowed_components():
    return {"train": set(splits.PROTOCOL_B[("train", None)]), "validation": set(splits.PROTOCOL_B[("validation", "*")])}


class Plan:
    def __init__(self, train, validation, crop_bank, counts, settings):
        self.train, self.validation, self.crop_bank, self.counts, self.settings = train, validation, crop_bank, counts, settings
        self.sha256 = digest({"settings": settings,
                              "train": [[t["recording_id"], t["n_samples"], t["pcm_sha256"], t["starts"]] for t in train],
                              "validation": [[v["recording_id"], v["role"], v["n_samples"], v["pcm_sha256"], v["chunks"]] for v in validation]})

    def shards(self):
        return sorted({i["shard"] for i in self.train} | {i["shard"] for i in self.validation})

    def jobs(self, shard_dir, parts_dir):
        by_shard = defaultdict(lambda: {"train": [], "validation": []})
        for item in self.train:
            by_shard[item["shard"]]["train"].append(item)
        for item in self.validation:
            by_shard[item["shard"]]["validation"].append(item)
        return [{"shard": s, "shard_path": str(Path(shard_dir) / s), "out_path": str(Path(parts_dir) / f"part_{s.removesuffix('.tar')}.npz"),
                 "plan_sha256": self.sha256, "crop": self.settings["crop_samples"], **by_shard[s]} for s in self.shards()]


def plan_b(cohort_rows, assignments, chunks, *, k: int = DEFAULT_K, seed: int = DEFAULT_SEED, crop: int = CROP_SAMPLES) -> Plan:
    """Select exactly the rows R2-S1 may touch. Test assignments are counted and dropped; component rules are checked, not assumed."""
    by_id = {r["recording_id"]: r for r in cohort_rows}
    allowed = _allowed_components()
    train, validation, unretained, dropped_test, seen = [], {}, [], 0, set()
    validation_ids = set()
    for a in assignments:
        if a.protocol_id != PROTOCOL_ID:
            raise R2S1Error(f"assignment {a.recording_id} belongs to {a.protocol_id}, not {PROTOCOL_ID}")
        if a.recording_id in seen:
            raise R2S1Error(f"duplicate assignment {a.recording_id}")
        seen.add(a.recording_id)
        if a.partition == "test":
            dropped_test += 1
            continue
        if a.partition not in allowed:
            raise R2S1Error(f"unknown partition {a.partition!r}")
        row = by_id.get(a.recording_id)
        if row is None:
            raise R2S1Error(f"assignment without a cohort row: {a.recording_id}")
        if row["component"] not in allowed[a.partition]:
            raise R2S1Error(f"{row['component']} recordings may not appear in B {a.partition} ({a.recording_id})")
        if a.group_id != f"speaker:{row['speaker_id']}":
            raise R2S1Error(f"group {a.group_id} does not match speaker of {a.recording_id}")
        if a.partition == "validation":
            validation_ids.add(a.recording_id)
            if a.validation_role not in splits.ROLES:
                raise R2S1Error(f"validation recording {a.recording_id} has no role")
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
            raise R2S1Error(f"validation chunk {c.chunk_id} has no validation assignment")
        item = validation.get(c.recording_id)
        if item is not None:
            item["chunks"].append([c.chunk_id, int(c.start_sample), int(c.end_sample)])
    for item in validation.values():
        item["chunks"].sort(key=lambda ch: (ch[1], ch[0]))
        if [(s, e) for _, s, e in item["chunks"]] != eval_windows(item["n_samples"]):
            raise R2S1Error(f"chunks of {item['recording_id']} do not equal the canonical windows of its audio")
    train.sort(key=lambda t: t["recording_id"])
    validation = sorted(validation.values(), key=lambda v: v["recording_id"])
    crop_bank = []
    for t in train:
        for start in t["starts"]:
            valid = min(crop, t["n_samples"] - start)
            crop_bank.append({"crop_id": f"crop:B:{t['recording_id']}:{start}", "recording_id": t["recording_id"], "group_id": t["group_id"],
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
        raise R2S1Error("plan has no train or no validation recordings")
    settings = {"k": k, "seed": seed, "crop_samples": crop, "bank_version": BANK_VERSION, "protocol": PROTOCOL_ID}
    return Plan(train, validation, crop_bank, counts, settings)


# ---- one shard -------------------------------------------------------------------------------------------------------------

def _empty_text():
    return np.array([], dtype="<U1")


def _text(values):
    return np.array(values, dtype=str) if values else _empty_text()


def _read_member(tar, item):
    handle = tar.extractfile(item["recording_id"].replace(":", "__") + ".flac")
    if handle is None:
        raise R2S1Error(f"{item['recording_id']} is missing from its shard")
    pcm = read_flac(handle.read())
    if hashlib.sha256(pcm.tobytes()).hexdigest() != item["pcm_sha256"]:
        raise R2S1Error(f"{item['recording_id']}: decoded audio does not match the manifest hash")
    if len(pcm) != item["n_samples"]:
        raise R2S1Error(f"{item['recording_id']}: {len(pcm)} samples, manifest says {item['n_samples']}")
    return pcm.astype(np.float32) / 32768.0


def process_shard(job: dict) -> dict:
    """Read each needed recording once; accumulate mel statistics, crop features and validation chunk features; write the part atomically."""
    started = time.perf_counter()
    stats = MelStats()
    rec_ids, rec_frames = [], []
    crop_ids, crop_rec, crop_start, crop_valid, crop_x = [], [], [], [], []
    val_ids, val_rec, val_start, val_end, val_x = [], [], [], [], []
    with tarfile.open(job["shard_path"], "r") as tar:
        for item in sorted(job["train"], key=lambda i: i["recording_id"]):
            wave = _read_member(tar, item)
            rec_ids.append(item["recording_id"])
            rec_frames.append(stats.add(log_mel(wave)))
            for start in item["starts"]:
                piece = wave[start:start + job["crop"]]
                crop_ids.append(f"crop:B:{item['recording_id']}:{start}")
                crop_rec.append(item["recording_id"])
                crop_start.append(start)
                crop_valid.append(len(piece))
                crop_x.append(handcrafted(piece))
        for item in sorted(job["validation"], key=lambda i: i["recording_id"]):
            wave = _read_member(tar, item)
            for chunk_id, start, end in item["chunks"]:
                val_ids.append(chunk_id)
                val_rec.append(item["recording_id"])
                val_start.append(start)
                val_end.append(end)
                val_x.append(handcrafted(wave[start:end]))
    width = len(FEATURE_NAMES)
    arrays = {"meta_plan_sha256": np.array(job["plan_sha256"]), "meta_feature_schema_hash": np.array(FEATURE_SCHEMA_HASH),
              "meta_shard": np.array(job["shard"]), **stats.to_arrays(),
              "rec_ids": _text(rec_ids), "rec_frames": np.array(rec_frames, dtype=np.int64),
              "crop_ids": _text(crop_ids), "crop_recording": _text(crop_rec), "crop_start": np.array(crop_start, dtype=np.int64),
              "crop_valid": np.array(crop_valid, dtype=np.int64),
              "crop_X": np.array(crop_x, dtype=np.float64).reshape(len(crop_x), width),
              "val_ids": _text(val_ids), "val_recording": _text(val_rec), "val_start": np.array(val_start, dtype=np.int64),
              "val_end": np.array(val_end, dtype=np.int64), "val_X": np.array(val_x, dtype=np.float64).reshape(len(val_x), width)}
    out = Path(job["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, out)
    return {"shard": job["shard"], "seconds": round(time.perf_counter() - started, 2), "train_recordings": len(rec_ids),
            "crops": len(crop_ids), "validation_chunks": len(val_ids), "mel_frames": int(sum(rec_frames))}


def part_is_current(job: dict) -> bool:
    out = Path(job["out_path"])
    if not out.exists():
        return False
    try:
        with np.load(out, allow_pickle=False) as z:
            return (str(z["meta_plan_sha256"]) == job["plan_sha256"] and str(z["meta_feature_schema_hash"]) == FEATURE_SCHEMA_HASH
                    and str(z["meta_shard"]) == job["shard"] and len(z["crop_ids"]) == sum(len(i["starts"]) for i in job["train"])
                    and len(z["val_ids"]) == sum(len(i["chunks"]) for i in job["validation"]))
    except Exception:                    # noqa: BLE001 - an unreadable part is simply recomputed
        return False


def run_jobs(jobs, *, workers: int = 1, resume: bool = False, progress=None):
    """Run every shard job; returns (summaries, failures). A failed shard does not stop the others, so a rerun only redoes it."""
    pending, summaries, failures = [], [], {}
    for job in jobs:
        if resume and part_is_current(job):
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


# ---- merge and verify ------------------------------------------------------------------------------------------------------

def merge_parts(plan: Plan, parts_dir):
    """Reassemble the parts in plan order -> (tables, MelStats, per-recording frame counts). Every row must appear exactly once."""
    stats, frames = MelStats(), {}
    crops, chunks = {}, {}
    for shard in plan.shards():
        path = Path(parts_dir) / f"part_{shard.removesuffix('.tar')}.npz"
        if not path.exists():
            raise R2S1Error(f"missing part file for {shard}")
        with np.load(path, allow_pickle=False) as z:
            if str(z["meta_plan_sha256"]) != plan.sha256 or str(z["meta_feature_schema_hash"]) != FEATURE_SCHEMA_HASH:
                raise R2S1Error(f"{path.name} was produced under a different plan or feature schema")
            stats.merge(MelStats.from_arrays(z))
            frames.update(zip(z["rec_ids"].tolist(), z["rec_frames"].tolist()))
            for i, cid in enumerate(z["crop_ids"].tolist()):
                if cid in crops:
                    raise R2S1Error(f"crop {cid} appears twice")
                crops[cid] = (int(z["crop_start"][i]), int(z["crop_valid"][i]), z["crop_X"][i].copy())
            for i, cid in enumerate(z["val_ids"].tolist()):
                if cid in chunks:
                    raise R2S1Error(f"chunk {cid} appears twice")
                chunks[cid] = (int(z["val_start"][i]), int(z["val_end"][i]), z["val_X"][i].copy())
    bank = plan.crop_bank
    if set(crops) != {r["crop_id"] for r in bank}:
        raise R2S1Error("crop rows in the parts differ from the crop bank")
    train = {"crop_id": _text([r["crop_id"] for r in bank]), "recording_id": _text([r["recording_id"] for r in bank]),
             "group_id": _text([r["group_id"] for r in bank]), "component": _text([r["component"] for r in bank]),
             "y": np.array([r["label"] for r in bank], dtype=np.int64), "start": np.array([r["start_sample"] for r in bank], dtype=np.int64),
             "valid_samples": np.array([r["valid_samples"] for r in bank], dtype=np.int64),
             "X": np.stack([crops[r["crop_id"]][2] for r in bank])}
    rows = [(v, ch) for v in plan.validation for ch in v["chunks"]]
    if set(chunks) != {ch[0] for _, ch in rows}:
        raise R2S1Error("validation rows in the parts differ from the chunk manifest")
    validation = {"chunk_id": _text([ch[0] for _, ch in rows]), "recording_id": _text([v["recording_id"] for v, _ in rows]),
                  "group_id": _text([v["group_id"] for v, _ in rows]), "component": _text([v["component"] for v, _ in rows]),
                  "role": _text([v["role"] for v, _ in rows]), "y": np.array([v["label"] for v, _ in rows], dtype=np.int64),
                  "start": np.array([ch[1] for _, ch in rows], dtype=np.int64), "end": np.array([ch[2] for _, ch in rows], dtype=np.int64),
                  "X": np.stack([chunks[ch[0]][2] for _, ch in rows])}
    for (v, ch) in rows:
        if chunks[ch[0]][:2] != (ch[1], ch[2]):
            raise R2S1Error(f"chunk {ch[0]} window differs between part and plan")
    return {"train": train, "validation": validation}, stats, frames


def expected_mel_frames(n_samples: int) -> int:
    return 1 + (max(n_samples, MEL_SCHEMA["n_fft"]) - MEL_SCHEMA["n_fft"]) // MEL_SCHEMA["hop_length"]


def verify(plan: Plan, tables, stats: MelStats, frames) -> dict:
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
    unexpected = sorted(set(nan_columns) - EXPECTED_NAN_COLUMNS)
    if unexpected:
        problems.append(f"NaN in unexpected feature columns: {unexpected}")
    allowed = _allowed_components()
    if not set(train["component"].tolist()) <= allowed["train"] or "chatterbox" in set(train["component"].tolist()) | set(val["component"].tolist()):
        problems.append("train components outside the Protocol B train set, or Chatterbox present")
    if not set(val["component"].tolist()) <= allowed["validation"]:
        problems.append("validation components outside the Protocol B validation set")
    if set(train["group_id"].tolist()) & set(val["group_id"].tolist()):
        problems.append("a speaker appears in both train and validation outputs")
    if set(train["y"].tolist()) != {0, 1} or set(val["y"].tolist()) != {0, 1}:
        problems.append("train or validation is missing a class")
    if any(not (0 < v <= plan.settings["crop_samples"]) for v in train["valid_samples"].tolist()):
        problems.append("a crop has an invalid length")
    expected_frames = sum(expected_mel_frames(t["n_samples"]) for t in plan.train)
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
    return {"passed": not problems, "problems": problems, "nan_columns": nan_columns, "mel_frames": stats.count,
            "mel_std_min": float(std.min()), "mel_std_max": float(std.max()),
            "rows": {"train_crops": int(len(train["y"])), "validation_chunks": int(len(val["y"]))}}


# ---- outputs ---------------------------------------------------------------------------------------------------------------

def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _save_npz_new(path: Path, arrays: dict):
    if path.exists():
        raise R2S1Error(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def write_outputs(out_dir, plan: Plan, tables, stats: MelStats, provenance: dict) -> dict:
    """Write the three feature/statistics files (never overwriting) and finally the manifest that hashes them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean, std = stats.finalize()
    _save_npz_new(out_dir / OUTPUT_NAMES["mel_stats"], {"mean": mean, "std": std, "frames": np.int64(stats.count),
                                                         "recordings": np.int64(stats.recordings), "n_mels": np.int64(len(mean)),
                                                         "plan_sha256": np.array(plan.sha256)})
    train, val = tables["train"], tables["validation"]
    _save_npz_new(out_dir / OUTPUT_NAMES["train_features"], {**train, "feature_names": _text(list(FEATURE_NAMES)),
                                                            "plan_sha256": np.array(plan.sha256), "feature_schema_hash": np.array(FEATURE_SCHEMA_HASH)})
    _save_npz_new(out_dir / OUTPUT_NAMES["validation_features"], {**val, "feature_names": _text(list(FEATURE_NAMES)),
                                                                 "plan_sha256": np.array(plan.sha256), "feature_schema_hash": np.array(FEATURE_SCHEMA_HASH)})
    outputs = {}
    for key, name in OUTPUT_NAMES.items():
        path = out_dir / name
        if path.exists():
            outputs[key] = {"file": name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest = {"schema_version": 1, "task": TASK, "protocol": "B", "plan_sha256": plan.sha256, "outputs": outputs, **provenance}
    path = out_dir / MANIFEST_NAME
    if path.exists():
        raise R2S1Error(f"refusing to overwrite {path}")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def write_crop_bank(out_dir, plan: Plan) -> Path:
    """The crop bank depends only on the manifest and the seed, so it is written before any audio is touched (R2-U1 uses the same file)."""
    path = Path(out_dir) / OUTPUT_NAMES["crop_bank"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = read_table(path)
        if [dict(r) for r in existing] != plan.crop_bank:
            raise R2S1Error(f"{path} exists and differs from the crop bank of this plan")
        return path
    write_table(plan.crop_bank, path)
    return path


def rows_digest(rows) -> str:
    """Same logical manifest digest scripts/r1_data_lock.py reports as cohort_manifest_sha256."""
    return digest([[r["recording_id"], r["pcm_sha256"], r["retained"]] for r in rows])


def spot_check(tables, loader, *, n: int = 25, seed: int = 42, crop: int = CROP_SAMPLES, mel_recordings: int = 10, check_stats: bool = True):
    """Recompute rows from the audio with an independent reader. Features must match exactly; streamed statistics must match a two-pass numpy result."""
    from echoproof.training.seeding import named_seed
    rng = np.random.default_rng(named_seed(seed, "r2_s1_spot_check"))
    train, val = tables["train"], tables["validation"]
    worst_train = worst_val = 0.0
    mismatches = []
    for i in rng.choice(len(train["y"]), size=min(n, len(train["y"])), replace=False):
        wave = loader(str(train["recording_id"][i]))
        direct = handcrafted(wave[int(train["start"][i]):int(train["start"][i]) + crop])
        if not np.array_equal(direct, train["X"][i], equal_nan=True):
            mismatches.append(str(train["crop_id"][i]))
    for i in rng.choice(len(val["y"]), size=min(n, len(val["y"])), replace=False):
        wave = loader(str(val["recording_id"][i]))
        direct = handcrafted(wave[int(val["start"][i]):int(val["end"][i])])
        if not np.array_equal(direct, val["X"][i], equal_nan=True):
            mismatches.append(str(val["chunk_id"][i]))
    result = {"rows_checked": min(n, len(train["y"])) + min(n, len(val["y"])), "mismatches": mismatches}
    if check_stats:
        recs = sorted(set(train["recording_id"].tolist()))
        pick = [recs[j] for j in rng.choice(len(recs), size=min(mel_recordings, len(recs)), replace=False)]
        frames = np.concatenate([log_mel(loader(r)).astype(np.float64) for r in pick])
        sample = MelStats()
        for r in pick:
            sample.add(log_mel(loader(r)))
        mean, std = sample.finalize()
        result["streaming_vs_two_pass_max_abs"] = float(max(np.abs(mean - frames.mean(axis=0)).max(), np.abs(std - frames.std(axis=0)).max()))
    return result


# ---- orchestration ---------------------------------------------------------------------------------------------------------

def build(root, shard_dir, out_dir=None, *, k: int = DEFAULT_K, seed: int = DEFAULT_SEED, workers: int = 1, resume: bool = False,
          spot_n: int = 25, progress=None) -> dict:
    """Run R2-S1 end to end against the locked manifests under `root`; returns the manifest dict it also writes to out_dir."""
    import platform
    import librosa
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data.chunks import PREPROCESSING_HASH
    from echoproof.training.datasets import ShardWaveformLoader
    say = progress or (lambda *_: None)
    t0 = time.perf_counter()
    root, shard_dir = Path(root), Path(shard_dir)
    out_dir = Path(out_dir) if out_dir else root / "data" / "features" / "r2_s1_B"
    parts_dir = out_dir / "parts"
    if (out_dir / MANIFEST_NAME).exists():
        raise R2S1Error(f"{out_dir / MANIFEST_NAME} exists: R2-S1 is already complete for this directory")
    if any((out_dir / name).exists() for name in OUTPUT_NAMES.values() if name != OUTPUT_NAMES["crop_bank"]):
        raise R2S1Error("final outputs exist without a manifest; inspect them before rerunning")
    if parts_dir.exists() and any(parts_dir.glob("part_*.npz")) and not resume:
        raise R2S1Error("part files exist from an earlier run; pass resume=True to continue it")

    paths = {"cohort": root / "data" / "manifests" / "cohort_v1.parquet", "split": root / "data" / "splits" / "protocol_B_v1.parquet",
             "chunks": root / "data" / "processed" / "chunks_v1.parquet"}
    for name, path in paths.items():
        if not path.exists():
            raise R2S1Error(f"missing {name} input: {path}")
    cohort_rows = read_table(paths["cohort"])
    assignments = splits.load_assignments(paths["split"], sealed=False)        # test rows are withheld by the loader
    split_hashes = {a.split_hash for a in assignments}
    if len(split_hashes) != 1:
        raise R2S1Error("protocol B assignments carry more than one split hash")
    split_hash = split_hashes.pop()
    report_path = root / "reports" / "r1" / "split_report.json"
    frozen = json.loads(report_path.read_text(encoding="utf-8"))["protocols"]["B"]["split_sha256"] if report_path.exists() else None
    if frozen is not None and frozen != split_hash:
        raise R2S1Error(f"protocol B split hash {split_hash[:12]} differs from the frozen value {frozen[:12]} in split_report.json")
    chunks = read_parquet(str(paths["chunks"]), Chunk)
    if {c.preprocessing_hash for c in chunks} != {PREPROCESSING_HASH}:
        raise R2S1Error("chunk manifest preprocessing hash differs from the code's PREPROCESSING_HASH")

    plan = plan_b(cohort_rows, assignments, chunks, k=k, seed=seed)
    say(f"plan {plan.sha256[:12]}: {plan.counts['train_recordings']} train recordings -> {plan.counts['train_crops']} crops, "
        f"{plan.counts['validation_recordings']} validation recordings -> {plan.counts['validation_chunks']} chunks")
    missing = [s for s in plan.shards() if not (shard_dir / s).is_file()]
    if missing:
        raise R2S1Error(f"shard files not found in {shard_dir}: {missing}")
    write_crop_bank(out_dir, plan)

    summaries, failures = run_jobs(plan.jobs(shard_dir, parts_dir), workers=workers, resume=resume,
                                   progress=lambda shard, error: say(f"{shard}: {'FAILED ' + error if error else 'done'}"))
    if failures:
        raise R2S1Error(f"shards failed (rerun with resume=True after fixing): {failures}")
    tables, stats, frames = merge_parts(plan, parts_dir)
    verification = verify(plan, tables, stats, frames)
    if not verification["passed"]:
        raise R2S1Error(f"verification failed: {verification['problems']}")
    loader = ShardWaveformLoader(shard_dir, {r["recording_id"]: r["canonical_path"] for r in cohort_rows if r["retained"]})
    spot = spot_check(tables, loader, n=spot_n, seed=seed)
    loader.close()
    if spot["mismatches"] or spot["streaming_vs_two_pass_max_abs"] > 1e-9:
        raise R2S1Error(f"spot check failed: {spot}")
    mean, std = stats.finalize()
    provenance = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {"cohort_manifest_file_sha256": file_sha256(paths["cohort"]), "cohort_manifest_rows_digest": rows_digest(cohort_rows),
                   "split_file_sha256": file_sha256(paths["split"]), "split_sha256": split_hash, "split_sha256_frozen_in_report": frozen,
                   "chunk_manifest_file_sha256": file_sha256(paths["chunks"]), "preprocessing_hash": PREPROCESSING_HASH,
                   "feature_schema_hash": FEATURE_SCHEMA_HASH},
        "settings": {**plan.settings, "mel": MEL_SCHEMA, "feature_columns": len(FEATURE_NAMES),
                     "mel_scope": "all valid frames of every retained B-train recording, frame-equal weights, population std (ddof 0)",
                     "feature_scope": "B train crop bank and B validation chunks only; no test rows, no Chatterbox"},
        "counts": plan.counts, "verification": verification, "spot_check": spot,
        "mel_stats": {"frames": stats.count, "recordings": stats.recordings, "mean_min": float(mean.min()), "mean_max": float(mean.max()),
                      "std_min": float(std.min()), "std_max": float(std.max())},
        "shards": summaries, "workers": workers, "resumed": bool(resume),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "librosa": librosa.__version__},
        "seconds_total": round(time.perf_counter() - t0, 1)}
    return write_outputs(out_dir, plan, tables, stats, provenance)
