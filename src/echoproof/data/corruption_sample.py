"""R3-S1 validation-sample run (needs the audio shards, so it runs on Colab): apply the four corruptions to a small, seeded sample of B
*selection*-role recordings and check the builder on real audio.

Scope and firewall. The sample is drawn only from the selection validation role (genuine, MeloTTS, VITS, XTTS). It reuses the R2-S3 plan, which
drops every test assignment, rejects ChatterboxTTS and never lists calibration or operating recordings. Nothing is scored by any model and no
audio is written: the output is a report and a per-variant table with PCM hashes. Generating the stored corrupted copies of the locked test
recordings is R5-S1, after GR4.

What the report establishes: every variant was produced, the same inputs reproduce the same PCM hashes in a second pass, noise SNR is on target,
MP3 round trips are aligned and at the requested bitrate, level adjustments are counted, and the effect on the R2-S2 measurement columns is
the expected direction (noise raises the noise floor, MP3 removes high-band energy). It does not claim anything about detector robustness.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from echoproof.artifacts import digest
from echoproof.data import corruption, splits
from echoproof.data.cohort import read_table
from echoproof.eda import measure, selection
from echoproof.features.r2_s1 import R2S1Error, _read_member, file_sha256, rows_digest

TASK = "R3-S1"
REPORT_NAME = "r3_s1_corruption_sample.json"
TABLE_NAME = "r3_s1_corruption_sample_variants.csv"
DELTA_COLUMNS = ("rms_dbfs", "noise_floor_dbfs", "hf_ratio_4k", "hf_ratio_7k")
TABLE_COLUMNS = ("parent_id", "component", "group_id", "label", "condition", "variant_id", "n_samples", "pcm_sha256", "noise_color", "gain",
                 "measured_snr_db", "snr_checked", "mp3_correlation", "mp3_length_diff", "mp3_effective_kbps",
                 "delta_rms_dbfs", "delta_noise_floor_dbfs", "delta_hf_ratio_4k", "delta_hf_ratio_7k", "pass_pcm_sha256_repeat")


def draw_sample(items, per_component: int, seed: int) -> list[dict]:
    """Deterministic draw of `per_component` recordings from each component by hashed order. Items must already be selection-role only."""
    if per_component < 1:
        raise R2S1Error("per_component must be at least 1")
    by_component = defaultdict(list)
    for item in items:
        if item["role"] != selection.ROLE:
            raise R2S1Error(f"{item['recording_id']} is not a {selection.ROLE}-role recording")
        by_component[item["component"]].append(item)
    chosen = []
    for component in sorted(by_component):
        pool = sorted(by_component[component], key=lambda i: hashlib.sha256(f"{seed}|{i['recording_id']}".encode()).hexdigest())
        chosen.extend(pool[:per_component])
    return sorted(chosen, key=lambda i: i["recording_id"])


def fixture_signal(seconds: float = 4.0, peak: float = 0.4, seed: int = 1) -> np.ndarray:
    """A fixed synthetic speech-like signal; its corrupted-hash digest is compared across machines as an informational reproducibility check."""
    sr = corruption.SAMPLE_RATE
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(seed)
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) ** 2
    x = env * sum(np.sin(2 * np.pi * f * t * (1 + 0.02 * np.sin(2 * np.pi * 4 * t))) / k for k, f in enumerate((140, 280, 420, 560, 700, 980, 1500, 2300), 1))
    x = x + rng.normal(0, 0.01, t.size)
    return (x / np.abs(x).max() * peak).astype(np.float32)


def fixture_digest(ffmpeg: str | None = None) -> dict:
    wave = fixture_signal()
    hashes = {v.condition: v.pcm_sha256 for v in corruption.corrupt_all(wave, "recording:fixture:r3s1-a", ffmpeg=ffmpeg)}
    return {"pcm_sha256": hashes, "digest": digest(hashes)}


def _delta(before: dict, after: dict) -> dict:
    return {f"delta_{c}": float(after[c] - before[c]) for c in DELTA_COLUMNS}


def _summary(values) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "min": round(float(v.min()), 4), "mean": round(float(v.mean()), 4), "max": round(float(v.max()), 4)}


def summarise(rows: list[dict]) -> dict:
    out = {}
    for condition in corruption.CONDITION_NAMES:
        r = [x for x in rows if x["condition"] == condition]
        entry = {"variants": len(r), "gain_applied": sum(1 for x in r if x["gain"] < 1.0),
                 "mean_delta": {c: round(float(np.mean([x[f"delta_{c}"] for x in r])), 4) for c in DELTA_COLUMNS} if r else {}}
        if corruption.CONDITIONS[condition]["kind"] == "noise":
            entry.update({"target_snr_db": corruption.CONDITIONS[condition]["snr_db"], "measured_snr_db": _summary(x["measured_snr_db"] for x in r),
                          "snr_checked": sum(1 for x in r if x["snr_checked"]), "white": sum(1 for x in r if x["noise_color"] == "white"),
                          "pink": sum(1 for x in r if x["noise_color"] == "pink")})
        else:
            entry.update({"target_kbps": corruption.CONDITIONS[condition]["bitrate_kbps"], "effective_kbps": _summary(x["mp3_effective_kbps"] for x in r),
                          "correlation": _summary(x["mp3_correlation"] for x in r), "length_diff_samples": _summary(x["mp3_length_diff"] for x in r),
                          "measured_snr_db": _summary(x["measured_snr_db"] for x in r)})
        out[condition] = entry
    return out


def load_inputs(root, seed: int = corruption.MASTER_SEED) -> dict:
    """Read and check the frozen B inputs, then build the selection-role plan. No audio is read."""
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data.chunks import PREPROCESSING_HASH
    root = Path(root)
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
    report_path = root / "reports" / "r1" / "split_report.json"
    frozen = json.loads(report_path.read_text(encoding="utf-8"))["protocols"]["B"]["split_sha256"] if report_path.exists() else None
    if frozen is not None and frozen != split_hash:
        raise R2S1Error("protocol B split hash differs from the frozen value in split_report.json")
    chunks = read_parquet(str(paths["chunks"]), Chunk)
    if {c.preprocessing_hash for c in chunks} != {PREPROCESSING_HASH}:
        raise R2S1Error("chunk manifest preprocessing hash differs from the code's PREPROCESSING_HASH")
    return {"paths": paths, "cohort_rows": cohort_rows, "split_hash": split_hash, "frozen": frozen,
            "eda": selection.plan_selection(cohort_rows, assignments, chunks, seed=seed)}


def run(root, shard_dir, out_dir=None, *, per_component: int = 8, seed: int = corruption.MASTER_SEED, ffmpeg: str | None = None,
        progress=None) -> dict:
    say = progress or (lambda *_: None)
    t0 = time.perf_counter()
    root, shard_dir = Path(root), Path(shard_dir)
    out_dir = Path(out_dir) if out_dir else root / "reports" / "r3"
    if (out_dir / REPORT_NAME).exists() or (out_dir / TABLE_NAME).exists():
        raise R2S1Error(f"{out_dir / REPORT_NAME} or its table exists: the sample run is not overwritten")
    ffmpeg_path = corruption.find_ffmpeg(ffmpeg)                                          # fail before any data is read
    loaded = load_inputs(root, seed)
    paths, cohort_rows, split_hash, frozen, eda = (loaded[k] for k in ("paths", "cohort_rows", "split_hash", "frozen", "eda"))
    sample = draw_sample(eda.items, per_component, seed)
    missing = sorted({i["shard"] for i in sample if not (shard_dir / i["shard"]).is_file()})
    if missing:
        raise R2S1Error(f"shard files not found in {shard_dir}: {missing}")
    say(f"plan {eda.plan.sha256[:12]}: sampling {len(sample)} of {len(eda.items)} selection recordings, {len(corruption.CONDITION_NAMES)} conditions each")

    rows, failures = [], []
    by_shard = defaultdict(list)
    for item in sample:
        by_shard[item["shard"]].append(item)
    for shard in sorted(by_shard):
        with tarfile.open(shard_dir / shard, "r") as tar:
            for item in by_shard[shard]:
                wave = _read_member(tar, item)                                            # hash- and length-checked against the manifest
                base, _ = measure.measure(wave)
                for condition in corruption.CONDITION_NAMES:
                    try:
                        first = corruption.corrupt(wave, item["recording_id"], condition, seed=seed, ffmpeg=ffmpeg_path)
                        again = corruption.corrupt(wave, item["recording_id"], condition, seed=seed, ffmpeg=ffmpeg_path)
                        after, _ = measure.measure(first.wave)
                    except corruption.CorruptionError as exc:
                        failures.append({"recording_id": item["recording_id"], "condition": condition, "error": str(exc)})
                        continue
                    rows.append({"parent_id": item["recording_id"], "component": item["component"], "group_id": item["group_id"],
                                 "label": item["label"], "condition": condition, "variant_id": first.variant_id, "n_samples": first.n_samples,
                                 "pcm_sha256": first.pcm_sha256, "noise_color": first.params.get("color"), "gain": first.params["gain"],
                                 "measured_snr_db": first.checks["measured_snr_db"], "snr_checked": first.checks.get("snr_checked"),
                                 "mp3_correlation": first.checks.get("correlation"), "mp3_length_diff": first.checks.get("length_diff_samples"),
                                 "mp3_effective_kbps": first.checks.get("effective_kbps"), **_delta(base, after),
                                 "pass_pcm_sha256_repeat": first.pcm_sha256 == again.pcm_sha256})
                say(f"{item['recording_id']}: done")

    expected = len(sample) * len(corruption.CONDITION_NAMES)
    checks = {"all_variants_built": len(rows) == expected and not failures,
              "repeat_hashes_identical": all(r["pass_pcm_sha256_repeat"] for r in rows),
              "parent_ids_kept": all(r["variant_id"] == f"{r['parent_id']}@{r['condition']}" for r in rows),
              "only_selection_role": {i["role"] for i in sample} == {selection.ROLE},
              "no_chatterbox": not any("chatterbox" in r["parent_id"] for r in rows),
              "lengths_preserved": all(r["n_samples"] == next(i["n_samples"] for i in sample if i["recording_id"] == r["parent_id"]) for r in rows),
              "unique_variant_ids": len({r["variant_id"] for r in rows}) == len(rows)}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / TABLE_NAME, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x["variant_id"]):
            writer.writerow({k: r[k] for k in TABLE_COLUMNS})
    source = Path(corruption.__file__)
    report = {"schema_version": 1, "task": TASK, "step": "validation_sample", "protocol": "B", "partition": "validation", "role": selection.ROLE,
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passed": all(checks.values()), "checks": checks,
              "failures": failures, "sample": {"per_component": per_component, "seed": seed, "recordings": len(sample), "variants_expected": expected,
                                               "variants_built": len(rows), "by_component": {c: sum(1 for i in sample if i["component"] == c)
                                                                                           for c in sorted({i["component"] for i in sample})},
                                               "speakers": len({i["group_id"] for i in sample})},
              "conditions": summarise(rows), "spec": corruption.spec_record(ffmpeg_path), "fixture": fixture_digest(ffmpeg_path),
              "scope": "B validation, selection role only; no calibration, operating, test or Chatterbox audio is read; no model scores anything; no audio is written",
              "inputs": {"plan_sha256": eda.plan.sha256, "split_sha256": split_hash, "split_sha256_frozen_in_report": frozen,
                         "cohort_manifest_file_sha256": file_sha256(paths["cohort"]), "cohort_manifest_rows_digest": rows_digest(cohort_rows),
                         "corruption_module_sha256": hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
              "outputs": {TABLE_NAME: {"sha256": file_sha256(out_dir / TABLE_NAME), "rows": len(rows)}},
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
              "seconds_total": round(time.perf_counter() - t0, 1)}
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
