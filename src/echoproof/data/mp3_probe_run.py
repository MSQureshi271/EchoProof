"""R3-S1 follow-up run: measure candidate MP3 settings on the same seeded sample of B selection-role recordings.

Two sources for the audio, same sample, same checks:

* `shards`  the cohort shards (Colab; all four components).
* `local`   the original WAV files in a local `dataset/` folder (genuine and XTTS only). Every file is decoded with the project's canonical decoder
            and used only if its PCM SHA-256 equals the cohort manifest's, so a local file is the very audio the shards hold.

Scope and firewall are those of R3-S1: selection-role recordings only (the sample comes from the R2-S3 plan, which drops every test assignment,
rejects ChatterboxTTS and never lists calibration or operating recordings). Nothing is scored by any model and no audio is written. The output is
signal-level measurements of each candidate against its own original.
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

from echoproof.data import corruption, corruption_sample as cs, mp3_probe
from echoproof.data.cohort import canonical_pcm, pcm_sha256
from echoproof.eda import selection
from echoproof.features.r2_s1 import R2S1Error, _read_member, file_sha256

TASK = "R3-S1"
LOCAL_DIRS = {"genuine": ("genuine", "wav"), "xtts": ("tts", "XTTSv2")}
ROW_COLUMNS = ("parent_id", "component", "group_id", "candidate", "kbps", "encode_rate", "snr_raw_db", "gain_db", "snr_gaincomp_db", "lsd_db",
               "retained_4k_pct", "retained_7k_pct", "lag_samples", "error")


def report_names(source: str) -> dict:
    return {"report": f"r3_s1_mp3_probe_{source}.json", "rows": f"r3_s1_mp3_probe_{source}_rows.csv", "figure": f"r3_s1_mp3_probe_{source}"}


def local_path(dataset_dir: Path, item: dict) -> Path | None:
    parts = LOCAL_DIRS.get(item["component"])
    if parts is None:
        return None
    stem = item["recording_id"].rsplit(":", 1)[-1]
    path = Path(dataset_dir).joinpath(*parts, f"{stem}.wav")
    return path if path.is_file() else None


def run(root, out_dir=None, *, source: str, shard_dir=None, dataset_dir=None, per_component: int = 8, seed: int = corruption.MASTER_SEED,
        candidates=tuple(mp3_probe.CANDIDATES), ffmpeg: str | None = None, progress=None, loaded=None) -> dict:
    if source not in ("shards", "local"):
        raise R2S1Error("source must be 'shards' or 'local'")
    say = progress or (lambda *_: None)
    t0 = time.perf_counter()
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / "reports" / "r3"
    names = report_names(source)
    if any((out_dir / n).exists() for n in (names["report"], names["rows"])):
        raise R2S1Error(f"{names['report']} or its rows file exists in {out_dir}: the probe is not overwritten")
    ffmpeg_path = corruption.find_ffmpeg(ffmpeg)
    if source == "shards" and shard_dir is None or source == "local" and dataset_dir is None:
        raise R2S1Error(f"source '{source}' needs {'shard_dir' if source == 'shards' else 'dataset_dir'}")
    loaded = loaded or cs.load_inputs(root, seed)
    eda = loaded["eda"]
    sample = cs.draw_sample(eda.items, per_component, seed)
    say(f"plan {eda.plan.sha256[:12]}: sample of {len(sample)} selection recordings, {len(candidates)} candidates")

    waves, unavailable, mismatched = {}, [], []
    if source == "local":
        for item in sample:
            path = local_path(Path(dataset_dir), item)
            if path is None:
                unavailable.append(item["recording_id"])
                continue
            pcm, _ = canonical_pcm(str(path))
            if pcm_sha256(pcm) != item["pcm_sha256"] or len(pcm) != item["n_samples"]:
                mismatched.append(item["recording_id"])
                continue
            waves[item["recording_id"]] = pcm.astype(np.float32) / 32768.0
    else:
        by_shard = defaultdict(list)
        for item in sample:
            by_shard[item["shard"]].append(item)
        missing = sorted(s for s in by_shard if not (Path(shard_dir) / s).is_file())
        if missing:
            raise R2S1Error(f"shard files not found in {shard_dir}: {missing}")
        for shard in sorted(by_shard):
            with tarfile.open(Path(shard_dir) / shard, "r") as tar:
                for item in by_shard[shard]:
                    waves[item["recording_id"]] = _read_member(tar, item)                 # hash- and length-checked against the manifest
    if mismatched:
        raise R2S1Error(f"local files do not match the cohort manifest hash: {mismatched}")
    if not waves:
        raise R2S1Error("no recordings could be loaded for the sample")

    rows = []
    for item in sample:
        wave = waves.get(item["recording_id"])
        if wave is None:
            continue
        result = mp3_probe.probe(wave, candidates, ffmpeg_path)
        for name, metrics in result.items():
            kbps, rate = mp3_probe.CANDIDATES[name]
            rows.append({"parent_id": item["recording_id"], "component": item["component"], "group_id": item["group_id"], "candidate": name,
                         "kbps": kbps, "encode_rate": rate, **{k: metrics.get(k) for k in ROW_COLUMNS[6:]}})
        say(f"{item['recording_id']}: done")

    summary = mp3_probe.summarise(rows, candidates)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / names["rows"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (x["candidate"], x["parent_id"])):
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ROW_COLUMNS})
    figure = mp3_probe.make_figure(summary, out_dir / names["figure"],
                                   title_note=f"{len(waves)} selection recordings ({', '.join(sorted({i['component'] for i in sample if i['recording_id'] in waves}))}).")
    measured = sorted({i["component"] for i in sample if i["recording_id"] in waves})
    report = {"schema_version": 1, "task": TASK, "step": "mp3_probe", "source": source, "protocol": "B", "partition": "validation", "role": selection.ROLE,
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "scope": "B validation, selection role only; no calibration, operating, test or Chatterbox audio is read; no model scores anything; no audio is written",
              "sample": {"per_component": per_component, "seed": seed, "drawn": len(sample), "measured": len(waves), "measured_components": measured,
                         "unavailable_locally": unavailable, "hash_verified_against_manifest": True},
              "candidates": list(candidates), "frozen_conditions": list(mp3_probe.CURRENT), "summary": summary,
              "errors": [r for r in rows if r.get("error")] and [{k: r[k] for k in ("parent_id", "candidate", "error")} for r in rows if r.get("error")],
              "method": {"alignment": "cross-correlation lag within 0..6000 samples", "active_frame_rule_dbfs": mp3_probe.ACTIVE_MS_DBFS,
                         "n_fft": mp3_probe.N_FFT, "hop": mp3_probe.HOP, "snr_gaincomp": "SNR after dividing the variant by its best scalar gain"},
              "spec_reference": {"frozen_spec_sha256": corruption.SPEC_SHA256, "ffmpeg": corruption.ffmpeg_info(ffmpeg_path)},
              "inputs": {"plan_sha256": eda.plan.sha256, "split_sha256": loaded["split_hash"], "split_sha256_frozen_in_report": loaded["frozen"],
                         "cohort_manifest_file_sha256": file_sha256(loaded["paths"]["cohort"]),
                         "mp3_probe_module_sha256": hashlib.sha256(Path(mp3_probe.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
              "outputs": {names["rows"]: {"sha256": file_sha256(out_dir / names["rows"]), "rows": len(rows)},
                          **{Path(p).name: {"sha256": file_sha256(p)} for p in figure["files"]}},
              "figure_interpretation": figure["interpretation"],
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
              "seconds_total": round(time.perf_counter() - t0, 1)}
    (out_dir / names["report"]).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
