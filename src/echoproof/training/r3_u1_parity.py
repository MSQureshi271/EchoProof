"""R3-U1 parity: scoring raw waveforms directly must agree with scoring the cached embeddings, for the trained head.

R2-U1 checked the cache with an untrained head. This checks the trained one, on validation chunks (selection role by default; audio of other roles is
not needed). The direct path is the one the inference adapter uses: encoder -> pooled embedding -> head. Chunks are taken from the cache index, the audio is
hash-verified against the cohort manifest values stored in the index, and the sample is a deterministic stratified draw (role, component, full or short).
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from echoproof.artifacts import digest, file_digest

ATOL = RTOL = 1e-4
FULL_CHUNK = 64000
LOCAL_DIRS = {"genuine": ("genuine", "wav"), "xtts": ("tts", "XTTSv2")}


class ParityError(ValueError):
    pass


def development_rows(cache_dir) -> list:
    """Index rows of the development scope, after checking the index hash recorded in the manifest."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "r2_u1_manifest.json").read_text(encoding="utf-8"))
    info = manifest["indices"]["development"]
    if file_digest(cache_dir / info["file"]) != info["sha256"]:
        raise ParityError("development index checksum mismatch")
    return json.loads((cache_dir / info["file"]).read_text(encoding="utf-8"))["rows"]


def select_items(rows, n: int = 50, *, roles=("selection",), available=None, seed: int = 42) -> list:
    """Deterministic round-robin draw over (role, component, short) strata; `available(row)` limits it to chunks whose audio can be read."""
    strata = {}
    for row in rows:
        if row["partition"] != "validation" or row["kind"] != "chunk" or row["role"] not in roles or row["component"] == "chatterbox":
            continue
        if available is not None and not available(row):
            continue
        strata.setdefault((row["role"], row["component"], row["valid_samples"] < FULL_CHUNK), []).append(row)
    if sum(map(len, strata.values())) < n or n < 1:
        raise ParityError("not enough readable validation chunks for the requested parity sample")
    queues = [sorted(items, key=lambda r: digest([seed, r["item_id"]])) for _, items in sorted(strata.items())]
    chosen = []
    while len(chosen) < n:
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen


def _chunk(waveform: np.ndarray, row) -> np.ndarray:
    clip = waveform[row["start_sample"]:row["end_sample"]]
    if len(clip) != row["valid_samples"]:
        raise ParityError(f"chunk {row['item_id']} has {len(clip)} samples, expected {row['valid_samples']}")
    return clip


def local_wav_path(dataset_dir, row):
    parts = LOCAL_DIRS.get(row["component"])
    if parts is None:
        return None
    path = Path(dataset_dir).joinpath(*parts, row["recording_id"].rsplit(":", 1)[-1] + ".wav")
    return path if path.is_file() else None


def local_available(dataset_dir):
    return lambda row: local_wav_path(dataset_dir, row) is not None


def waves_from_dataset(rows, dataset_dir) -> list:
    """Chunks read from the original WAV files; every file must decode to exactly the canonical PCM the manifest recorded."""
    from echoproof.data.cohort import canonical_pcm, pcm_sha256
    waves = []
    for row in rows:
        path = local_wav_path(dataset_dir, row)
        if path is None:
            raise ParityError(f"no local audio for {row['recording_id']}")
        pcm, _ = canonical_pcm(str(path))
        if pcm_sha256(pcm) != row["pcm_sha256"] or len(pcm) != row["n_samples"]:
            raise ParityError(f"local file for {row['recording_id']} does not match the cohort manifest")
        waves.append(_chunk(pcm.astype(np.float32) / 32768.0, row))
    return waves


def waves_from_shards(rows, shard_dir) -> list:
    from echoproof.features.r2_u1 import _read_wave
    waves, by_shard = {}, {}
    for row in rows:
        by_shard.setdefault(row["shard"], []).append(row)
    for shard, items in sorted(by_shard.items()):
        with tarfile.open(Path(shard_dir) / shard, "r") as archive:
            for row in items:
                waves[row["item_id"]] = _chunk(_read_wave(archive, row), row)         # hash- and length-checked against the manifest
    return [waves[row["item_id"]] for row in rows]


def parity_check(rows, waves, cached: np.ndarray, head, *, embed_fn, pooling, score_fn=None, atol=ATOL, rtol=RTOL) -> dict:
    """Compare direct and cached embeddings, and the head's probabilities on each. `score_fn` (the adapter) is compared too when given."""
    import torch
    if not (len(rows) == len(waves) == len(cached)) or len(rows) == 0:
        raise ParityError("rows, waveforms and cached vectors must align")
    direct = np.asarray(embed_fn(waves), dtype=np.float32)
    if direct.shape != cached.shape or not np.isfinite(direct).all():
        raise ParityError("invalid direct embeddings")
    p_direct = head.scores(torch.from_numpy(direct)).double().numpy()
    p_cached = head.scores(torch.from_numpy(np.ascontiguousarray(cached, dtype=np.float32))).double().numpy()
    report = {"items": len(rows), "item_ids": [r["item_id"] for r in rows], "pooling": pooling, "atol": atol, "rtol": rtol,
              "components": sorted({r["component"] for r in rows}), "roles": sorted({r["role"] for r in rows}),
              "short_chunks": int(sum(r["valid_samples"] < FULL_CHUNK for r in rows)),
              "max_abs_embedding_difference": float(np.max(np.abs(direct - cached))),
              "max_abs_probability_difference": float(np.max(np.abs(p_direct - p_cached))),
              "max_decision_flips_at_0.5": int(np.sum((p_direct >= .5) != (p_cached >= .5)))}
    ok = bool(np.allclose(direct, cached, atol=atol, rtol=rtol) and np.allclose(p_direct, p_cached, atol=atol, rtol=rtol))
    if score_fn is not None:
        p_adapter = np.asarray(score_fn(waves), dtype=np.float64)
        report["max_abs_adapter_difference"] = float(np.max(np.abs(p_adapter - p_cached)))
        ok = ok and bool(np.allclose(p_adapter, p_cached, atol=atol, rtol=rtol))
    report["passed"] = ok
    return report
