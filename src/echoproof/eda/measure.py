"""R2-S2 step 1 (needs audio, runs on Colab next to the shards): per-recording measurements for the Protocol B train partition.

Scope: exactly the retained B-train recordings that `r2_s1.plan_b` selects (genuine, MeloTTS, VITS; 151 speakers). Validation, test and
ChatterboxTTS recordings are never listed or read. The plan is the R2-S1 plan, so it has the same 4,530 recordings and the same plan hash.

Definitions (frozen here; the figures and the R2-S3 shortcut probe read the columns, they do not redefine them):
  * frame level  - RMS of 20 ms frames (320 samples) every 10 ms (160 samples), in dBFS, floored at -160;
  * active frame - level above a per-recording two-class energy threshold (Otsu's method on the frame levels, 0.5 dB bins, levels clipped to the
    loudest 70 dB). If the two classes are less than 10 dB apart the recording has no pauses and every frame is active; if even the loudest frame
    is at or below -70 dBFS no frame is active. A fixed offset from the peak was tried first and found no pauses in noisy real LibriSpeech files;
  * leading/trailing silence - time before the first / after the last active frame;
  * silence_fraction - share of frames that are not active (low-energy frames, pauses inside the recording included);
  * noise_floor_dbfs / active_level_dbfs - 10th percentile of all frame levels / mean level of active frames;
  * hf_ratio_4k / hf_ratio_7k - share of total STFT power (512-point Hann, hop 160) at or above 4 kHz / 7 kHz.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

import numpy as np

from echoproof.data import splits
from echoproof.data.cohort import SAMPLE_RATE, read_table
from echoproof.features.acoustic import log_mel
from echoproof.features.r2_s1 import R2S1Error, _read_member, file_sha256, plan_b, rows_digest

TASK = "R2-S2"
MANIFEST_NAME = "r2_s2_manifest.json"
OUTPUT_NAMES = {"stats": "recording_stats_B_train_v1.parquet", "mel_mean": "mean_logmel_B_train_v1.npz",
                "examples": "examples_B_train_v1.npz"}
MEASURE_VERSION = 1
FRAME, HOP, N_FFT = 320, 160, 512
ACTIVE_FLOOR_DBFS, OTSU_SPAN_DB, OTSU_STEP_DB, MIN_CLASS_GAP_DB = -70.0, 70.0, 0.5, 10.0
EXAMPLE_COMPONENTS = ("genuine", "melo", "vits")
SCALAR_COLUMNS = ("n_samples", "duration_s", "rms_dbfs", "peak", "clip_fraction", "dc_offset", "leading_silence_s", "trailing_silence_s",
                  "silence_fraction", "noise_floor_dbfs", "active_level_dbfs", "hf_ratio_4k", "hf_ratio_7k")
SETTINGS = {"measure_version": MEASURE_VERSION, "frame": FRAME, "hop": HOP, "n_fft": N_FFT, "activity_rule": "otsu_per_recording", "otsu_span_db": OTSU_SPAN_DB,
            "otsu_step_db": OTSU_STEP_DB, "min_class_gap_db": MIN_CLASS_GAP_DB, "active_floor_dbfs": ACTIVE_FLOOR_DBFS, "clip_threshold": 0.999, "sample_rate": SAMPLE_RATE}


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    n = 1 + (len(x) - size) // hop
    return np.lib.stride_tricks.as_strided(x, shape=(n, size), strides=(x.strides[0] * hop, x.strides[0]), writeable=False)


def frame_levels_db(wave: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(wave, dtype=np.float64)
    rms = np.sqrt((_frames(x, FRAME, HOP) ** 2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-8))


def activity_threshold(levels: np.ndarray) -> float:
    """Level (dBFS) above which a frame counts as active; -inf means every frame is active, +inf means none is."""
    top = float(levels.max())
    if top <= ACTIVE_FLOOR_DBFS:
        return float("inf")
    clipped = np.clip(levels, top - OTSU_SPAN_DB, top)
    edges = np.arange(top - OTSU_SPAN_DB, top + 2 * OTSU_STEP_DB, OTSU_STEP_DB)       # last bin must contain the loudest frame
    counts, _ = np.histogram(clipped, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = counts / counts.sum()
    cum_w, cum_m = np.cumsum(weight), np.cumsum(weight * centers)
    total_mean = cum_m[-1]
    best_var, best = -1.0, None
    for i in range(len(centers) - 1):
        w0, w1 = cum_w[i], 1.0 - cum_w[i]
        if w0 <= 0 or w1 <= 0:
            continue
        m0, m1 = cum_m[i] / w0, (total_mean - cum_m[i]) / w1
        var = w0 * w1 * (m1 - m0) ** 2
        if var > best_var:
            best_var, best = var, (float(edges[i + 1]), m1 - m0)
    if best is None or best[1] < MIN_CLASS_GAP_DB:
        return float("-inf")
    return best[0]


def active_mask(levels: np.ndarray) -> np.ndarray:
    return levels > activity_threshold(levels)


def measure(wave) -> tuple[dict, np.ndarray]:
    """One recording (float waveform in [-1, 1]) -> (scalar measurements, mean log-mel [64] over all frames)."""
    x = np.ascontiguousarray(wave, dtype=np.float64)
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("expected a finite mono waveform")
    n = len(x)
    levels = frame_levels_db(x)
    active = active_mask(levels)
    idx = np.flatnonzero(active)
    if len(idx):
        leading, trailing = idx[0] * HOP / SAMPLE_RATE, (len(levels) - 1 - idx[-1]) * HOP / SAMPLE_RATE
        active_level = float(levels[active].mean())
    else:
        leading = trailing = n / SAMPLE_RATE
        active_level = float(levels.min())
    padded = x if n >= N_FFT else np.pad(x, (0, N_FFT - n))
    power = np.abs(np.fft.rfft(_frames(padded, N_FFT, HOP) * np.hanning(N_FFT), axis=1)) ** 2
    total = float(power.sum())
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)

    def hf(cutoff):
        return float(power[:, freqs >= cutoff].sum() / total) if total > 0 else 0.0

    rms = float(np.sqrt((x ** 2).mean()))
    row = {"n_samples": n, "duration_s": n / SAMPLE_RATE, "rms_dbfs": 20.0 * np.log10(max(rms, 1e-8)), "peak": float(np.abs(x).max()),
           "clip_fraction": float((np.abs(x) >= SETTINGS["clip_threshold"]).mean()), "dc_offset": float(x.mean()),
           "leading_silence_s": float(leading), "trailing_silence_s": float(trailing), "silence_fraction": float(1.0 - active.mean()),
           "noise_floor_dbfs": float(np.percentile(levels, 10)), "active_level_dbfs": active_level,
           "hf_ratio_4k": hf(4000.0), "hf_ratio_7k": hf(7000.0)}
    return row, log_mel(x.astype(np.float32)).astype(np.float64).mean(axis=0).astype(np.float32)


# ---- plan: train items only --------------------------------------------------------------------------------------------------

def choose_examples(train_items, *, n: int = 3, seed: int = 42, low_s: float = 5.0, high_s: float = 9.0):
    """n stems that exist as genuine, Melo and VITS in the train partition, all three lasting low_s..high_s, from different speakers."""
    by_stem = {}
    for t in train_items:
        by_stem.setdefault(t["recording_id"].rsplit(":", 1)[1], {})[t["component"]] = t
    pool = []
    for stem, comps in by_stem.items():
        if set(comps) >= set(EXAMPLE_COMPONENTS) and all(low_s <= comps[c]["n_samples"] / SAMPLE_RATE <= high_s for c in EXAMPLE_COMPONENTS):
            pool.append((hashlib.sha256(f"{seed}|r2_s2_example|{stem}".encode()).hexdigest(), stem, comps))
    chosen, speakers = [], set()
    for _, stem, comps in sorted(pool, key=lambda p: (p[0], p[1])):
        group = comps["genuine"]["group_id"]
        if group not in speakers:
            speakers.add(group)
            chosen.append({"stem": stem, **{c: comps[c] for c in EXAMPLE_COMPONENTS}})
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise R2S1Error(f"only {len(chosen)} suitable paired stems for the waveform figure")
    return chosen


class Eda:
    def __init__(self, plan, examples):
        self.plan, self.examples = plan, examples
        self.items = plan.train
        self.sha256 = plan.sha256

    def shards(self):
        return sorted({i["shard"] for i in self.items})

    def jobs(self, shard_dir, parts_dir):
        by = {}
        for item in self.items:
            by.setdefault(item["shard"], []).append(item)
        return [{"shard": s, "shard_path": str(Path(shard_dir) / s), "plan_sha256": self.sha256, "items": by[s],
                 "out_path": str(Path(parts_dir) / f"eda_part_{s.removesuffix('.tar')}.npz")} for s in self.shards()]


def plan_eda(cohort_rows, assignments, chunks, *, seed: int = 42, example_seconds=(5.0, 9.0)) -> Eda:
    plan = plan_b(cohort_rows, assignments, chunks, seed=seed)          # enforces test dropped, no Chatterbox, groups, shards
    comps = {t["component"] for t in plan.train}
    if comps != set(EXAMPLE_COMPONENTS):
        raise R2S1Error(f"unexpected train components {sorted(comps)}")
    return Eda(plan, choose_examples(plan.train, seed=seed, low_s=example_seconds[0], high_s=example_seconds[1]))


# ---- one shard ---------------------------------------------------------------------------------------------------------------

def process_shard(job: dict) -> dict:
    started = time.perf_counter()
    ids, rows, mels = [], [], []
    with tarfile.open(job["shard_path"], "r") as tar:
        for item in sorted(job["items"], key=lambda i: i["recording_id"]):
            row, mel = measure(_read_member(tar, item))          # hash- and length-checked against the manifest
            if row["n_samples"] != item["n_samples"]:
                raise R2S1Error(f"{item['recording_id']}: measured length differs from the manifest")
            ids.append(item["recording_id"])
            rows.append([row[c] for c in SCALAR_COLUMNS])
            mels.append(mel)
    arrays = {"meta_plan_sha256": np.array(job["plan_sha256"]), "meta_measure_version": np.array(MEASURE_VERSION),
              "meta_shard": np.array(job["shard"]), "ids": np.array(ids, dtype=str),
              "scalars": np.array(rows, dtype=np.float64).reshape(len(rows), len(SCALAR_COLUMNS)),
              "mel_mean": np.array(mels, dtype=np.float32).reshape(len(mels), -1)}
    out = Path(job["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, out)
    return {"shard": job["shard"], "recordings": len(ids), "seconds": round(time.perf_counter() - started, 2)}


def part_is_current(job: dict) -> bool:
    out = Path(job["out_path"])
    if not out.exists():
        return False
    try:
        with np.load(out, allow_pickle=False) as z:
            return (str(z["meta_plan_sha256"]) == job["plan_sha256"] and int(z["meta_measure_version"]) == MEASURE_VERSION
                    and str(z["meta_shard"]) == job["shard"] and len(z["ids"]) == len(job["items"]))
    except Exception:                    # noqa: BLE001 - an unreadable part is recomputed
        return False


def run_jobs(jobs, *, workers: int = 1, resume: bool = False, progress=None):
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
            except Exception as exc:     # noqa: BLE001
                failures[job["shard"]] = f"{type(exc).__name__}: {exc}"
            if progress:
                progress(job["shard"], failures.get(job["shard"]))
    else:
        import concurrent.futures as cf
        import multiprocessing
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(process_shard, job): job for job in sorted(pending, key=lambda j: -len(j["items"]))}
            for future in cf.as_completed(futures):
                job = futures[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures[job["shard"]] = f"{type(exc).__name__}: {exc}"
                if progress:
                    progress(job["shard"], failures.get(job["shard"]))
    return sorted(summaries, key=lambda s: s["shard"]), failures


# ---- merge, examples, verify -------------------------------------------------------------------------------------------------

def merge_parts(eda: Eda, parts_dir):
    """-> (pandas DataFrame of scalars in recording_id order, mean log-mel [n, 64] aligned with it)."""
    import pandas as pd
    scalars, mels = {}, {}
    for shard in eda.shards():
        path = Path(parts_dir) / f"eda_part_{shard.removesuffix('.tar')}.npz"
        if not path.exists():
            raise R2S1Error(f"missing part file for {shard}")
        with np.load(path, allow_pickle=False) as z:
            if str(z["meta_plan_sha256"]) != eda.sha256 or int(z["meta_measure_version"]) != MEASURE_VERSION:
                raise R2S1Error(f"{path.name} was produced under a different plan or measurement version")
            for i, rid in enumerate(z["ids"].tolist()):
                if rid in scalars:
                    raise R2S1Error(f"{rid} appears twice")
                scalars[rid], mels[rid] = z["scalars"][i].copy(), z["mel_mean"][i].copy()
    if set(scalars) != {i["recording_id"] for i in eda.items}:
        raise R2S1Error("measured recordings differ from the plan")
    order = sorted(eda.items, key=lambda i: i["recording_id"])
    frame = pd.DataFrame({"recording_id": [i["recording_id"] for i in order], "group_id": [i["group_id"] for i in order],
                          "component": [i["component"] for i in order], "label": [i["label"] for i in order]})
    matrix = np.stack([scalars[i["recording_id"]] for i in order])
    for j, name in enumerate(SCALAR_COLUMNS):
        frame[name] = matrix[:, j].astype(np.int64) if name == "n_samples" else matrix[:, j]
    frame["stem"] = frame["recording_id"].str.rsplit(":", n=1).str[1]
    frame["speaker_id"] = frame["group_id"].str.split(":").str[1]
    return frame, np.stack([mels[i["recording_id"]] for i in order])


def read_examples(eda: Eda, shard_dir) -> dict:
    """Waveforms (int16) of the paired example stems, each hash-checked against the manifest."""
    arrays = {"components": np.array(EXAMPLE_COMPONENTS, dtype=str), "stems": np.array([e["stem"] for e in eda.examples], dtype=str)}
    for e in eda.examples:
        for component in EXAMPLE_COMPONENTS:
            item = e[component]
            with tarfile.open(Path(shard_dir) / item["shard"], "r") as tar:
                wave = _read_member(tar, item)
            arrays[f"wave__{e['stem']}__{component}"] = np.round(wave * 32768.0).astype(np.int16)
    return arrays


def verify(eda: Eda, frame, mel_mean, examples) -> dict:
    problems = []
    n = len(eda.items)
    if len(frame) != n or mel_mean.shape != (n, 64):
        problems.append(f"expected {n} rows, found {len(frame)} / mel {mel_mean.shape}")
    if set(frame["component"]) != set(EXAMPLE_COMPONENTS):
        problems.append(f"components {sorted(set(frame['component']))}")
    counts = {k: int(v) for k, v in frame["component"].value_counts().sort_index().items()}
    expected = {}
    for i in eda.items:
        expected[i["component"]] = expected.get(i["component"], 0) + 1
    if counts != dict(sorted(expected.items())):
        problems.append(f"component counts {counts} differ from the plan {expected}")
    num = frame[list(SCALAR_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(num).all() or not np.isfinite(mel_mean).all():
        problems.append("non-finite measurements")
    by_id = {i["recording_id"]: i for i in eda.items}
    if any(by_id[r]["n_samples"] != s for r, s in zip(frame["recording_id"], frame["n_samples"])):
        problems.append("n_samples differs from the manifest")
    if ((frame["silence_fraction"] < 0) | (frame["silence_fraction"] > 1)).any() or ((frame["clip_fraction"] < 0) | (frame["clip_fraction"] > 1)).any():
        problems.append("a fraction is outside 0..1")
    if ((frame["leading_silence_s"] + frame["trailing_silence_s"]) > frame["duration_s"] + 1e-9).any():
        problems.append("leading + trailing silence exceeds the duration")
    if (frame["hf_ratio_7k"] > frame["hf_ratio_4k"] + 1e-12).any():
        problems.append("hf_ratio_7k exceeds hf_ratio_4k")
    for e in eda.examples:
        for c in EXAMPLE_COMPONENTS:
            if f"wave__{e['stem']}__{c}" not in examples:
                problems.append(f"example waveform missing: {e['stem']} {c}")
    return {"passed": not problems, "problems": problems, "rows": int(len(frame)), "counts": counts}


def spot_check(eda: Eda, frame, mel_mean, shard_dir, *, n: int = 10, seed: int = 42) -> dict:
    """Re-measure a random subset through a fresh tar handle; values must be identical (the code is deterministic)."""
    from echoproof.training.seeding import named_seed
    rng = np.random.default_rng(named_seed(seed, "r2_s2_spot_check"))
    by_id = {i["recording_id"]: i for i in eda.items}
    mismatches = []
    for i in rng.choice(len(frame), size=min(n, len(frame)), replace=False):
        item = by_id[frame["recording_id"].iloc[i]]
        with tarfile.open(Path(shard_dir) / item["shard"], "r") as tar:
            row, mel = measure(_read_member(tar, item))
        if any(row[c] != frame[c].iloc[i] for c in SCALAR_COLUMNS) or not np.array_equal(mel, mel_mean[i]):
            mismatches.append(item["recording_id"])
    return {"rows_checked": min(n, len(frame)), "mismatches": mismatches}


# ---- orchestration -----------------------------------------------------------------------------------------------------------

def build(root, shard_dir, out_dir=None, *, seed: int = 42, workers: int = 1, resume: bool = False, progress=None,
          example_seconds=(5.0, 9.0)) -> dict:
    import platform
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data.chunks import PREPROCESSING_HASH
    say = progress or (lambda *_: None)
    t0 = time.perf_counter()
    root, shard_dir = Path(root), Path(shard_dir)
    out_dir = Path(out_dir) if out_dir else root / "data" / "eda" / "r2_s2_B_train"
    parts_dir = out_dir / "parts"
    if (out_dir / MANIFEST_NAME).exists():
        raise R2S1Error(f"{out_dir / MANIFEST_NAME} exists: R2-S2 measurements are already complete for this directory")
    if any((out_dir / n).exists() for n in OUTPUT_NAMES.values()):
        raise R2S1Error("outputs exist without a manifest; inspect them before rerunning")
    if parts_dir.exists() and any(parts_dir.glob("eda_part_*.npz")) and not resume:
        raise R2S1Error("part files exist from an earlier run; pass resume=True to continue it")
    paths = {"cohort": root / "data" / "manifests" / "cohort_v1.parquet", "split": root / "data" / "splits" / "protocol_B_v1.parquet",
             "chunks": root / "data" / "processed" / "chunks_v1.parquet"}
    for name, path in paths.items():
        if not path.exists():
            raise R2S1Error(f"missing {name} input: {path}")
    cohort_rows = read_table(paths["cohort"])
    assignments = splits.load_assignments(paths["split"], sealed=False)
    hashes = {a.split_hash for a in assignments}
    if len(hashes) != 1:
        raise R2S1Error("protocol B assignments carry more than one split hash")
    split_hash = hashes.pop()
    report = root / "reports" / "r1" / "split_report.json"
    frozen = json.loads(report.read_text(encoding="utf-8"))["protocols"]["B"]["split_sha256"] if report.exists() else None
    if frozen is not None and frozen != split_hash:
        raise R2S1Error("protocol B split hash differs from the frozen value in split_report.json")
    chunks = read_parquet(str(paths["chunks"]), Chunk)
    if {c.preprocessing_hash for c in chunks} != {PREPROCESSING_HASH}:
        raise R2S1Error("chunk manifest preprocessing hash differs from the code's PREPROCESSING_HASH")
    eda = plan_eda(cohort_rows, assignments, chunks, seed=seed, example_seconds=example_seconds)
    say(f"plan {eda.sha256[:12]}: {len(eda.items)} B-train recordings, {len(eda.shards())} shards, examples {[e['stem'] for e in eda.examples]}")
    missing = [s for s in eda.shards() if not (shard_dir / s).is_file()]
    if missing:
        raise R2S1Error(f"shard files not found in {shard_dir}: {missing}")
    summaries, failures = run_jobs(eda.jobs(shard_dir, parts_dir), workers=workers, resume=resume,
                                   progress=lambda shard, error: say(f"{shard}: {'FAILED ' + error if error else 'done'}"))
    if failures:
        raise R2S1Error(f"shards failed (rerun with resume=True): {failures}")
    frame, mel_mean = merge_parts(eda, parts_dir)
    examples = read_examples(eda, shard_dir)
    verification = verify(eda, frame, mel_mean, examples)
    if not verification["passed"]:
        raise R2S1Error(f"verification failed: {verification['problems']}")
    spot = spot_check(eda, frame, mel_mean, shard_dir, seed=seed)
    if spot["mismatches"]:
        raise R2S1Error(f"spot check failed: {spot}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / OUTPUT_NAMES["stats"], index=False)
    np.savez(out_dir / OUTPUT_NAMES["mel_mean"], recording_id=np.array(frame["recording_id"], dtype=str), mel_mean=mel_mean,
             plan_sha256=np.array(eda.sha256))
    np.savez(out_dir / OUTPUT_NAMES["examples"], **examples, plan_sha256=np.array(eda.sha256))
    outputs = {k: {"file": n, "sha256": file_sha256(out_dir / n), "bytes": (out_dir / n).stat().st_size} for k, n in OUTPUT_NAMES.items()}
    manifest = {"schema_version": 1, "task": TASK, "protocol": "B", "partition": "train", "plan_sha256": eda.sha256,
                "settings": SETTINGS, "scope": "B train only: genuine, melo, vits; no validation, test or Chatterbox audio is read",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "inputs": {"cohort_manifest_file_sha256": file_sha256(paths["cohort"]), "cohort_manifest_rows_digest": rows_digest(cohort_rows),
                           "split_sha256": split_hash, "split_sha256_frozen_in_report": frozen, "split_file_sha256": file_sha256(paths["split"])},
                "counts": verification["counts"], "examples": [e["stem"] for e in eda.examples], "verification": verification,
                "spot_check": spot, "shards": summaries, "workers": workers, "resumed": bool(resume), "outputs": outputs,
                "environment": {"python": platform.python_version(), "numpy": np.__version__},
                "seconds_total": round(time.perf_counter() - t0, 1)}
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
