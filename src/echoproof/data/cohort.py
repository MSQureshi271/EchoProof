"""R1-S1 admission and R1-S3 materialisation helpers for the PhonemeDF research cohort.

Pure-Python parts (stem parsing, admission) need only the standard library. Audio
materialisation imports numpy/soundfile lazily so the admission step can run anywhere.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time

COMPONENTS = ("genuine", "xtts", "chatterbox", "melo", "vits")
GENERATOR = {"genuine": None, "xtts": "xtts", "chatterbox": "chatterbox", "melo": "melo", "vits": "vits"}
LABEL = {"genuine": 0, "xtts": 1, "chatterbox": 1, "melo": 1, "vits": 1}
SAMPLE_RATE = 16000
_STEM = re.compile(r"^(\d+)-(\d+)-(\d{4})$")


class CohortError(ValueError):
    """Invalid stem, inconsistent inventory or unsafe materialisation request."""


def parse_stem(stem: str) -> tuple[str, str, str]:
    """Return (speaker, chapter, utterance) from a LibriSpeech-style stem."""
    match = _STEM.match(stem) if isinstance(stem, str) else None
    if not match:
        raise CohortError(f"stem does not match speaker-chapter-utterance: {stem!r}")
    return match.group(1), match.group(2), match.group(3)


def recording_id(component: str, stem: str) -> str:
    if component not in COMPONENTS:
        raise CohortError(f"unknown component {component!r}")
    parse_stem(stem)
    return f"phonemedf:{component}:{stem}"


def speaker_group(stem: str) -> str:
    return f"speaker:{parse_stem(stem)[0]}"


def stem_from_path(path: str) -> str:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if not name.lower().endswith(".wav"):
        raise CohortError(f"not a wav path: {path!r}")
    return name[:-4]


def admission(coverage_rows, header_rows, *, expected_components=COMPONENTS):
    """R1-S1: verify stem format, chapter->speaker consistency, component presence and header health.

    coverage_rows: dicts with candidate_stem and counts (Phase 2 P2-06 coverage.jsonl).
    header_rows: dicts with component, path, state (Phase 2 P2-07 headers.jsonl).
    A stem with any non-ok component header, or any component missing, is excluded
    from *every* component so that all five recordings of a stem stay together.
    """
    stems, format_errors = [], []
    seen = set()
    for row in coverage_rows:
        stem = row["candidate_stem"]
        if stem in seen:
            raise CohortError(f"duplicate stem in coverage: {stem}")
        seen.add(stem)
        try:
            parse_stem(stem)
        except CohortError:
            format_errors.append(stem)
            continue
        stems.append(stem)
    missing = {}
    counts = {row["candidate_stem"]: row["counts"] for row in coverage_rows}
    for stem in stems:
        gaps = [c for c in expected_components if counts[stem].get(c, 0) != 1]
        if counts[stem].get("transcripts", 0) != 1:
            gaps.append("transcripts")
        if gaps:
            missing[stem] = gaps
    unhealthy = defaultdict(list)
    for row in header_rows:
        if row.get("state") != "ok":
            unhealthy[stem_from_path(row["path"])].append({"component": row["component"], "state": row["state"], "failure": row.get("failure")})
    excluded = {}
    for stem, gaps in missing.items():
        excluded[stem] = {"reason": "component_missing", "detail": gaps}
    for stem, items in unhealthy.items():
        if stem in seen:
            excluded.setdefault(stem, {"reason": "header_not_ok", "detail": items})
    chapter_speaker = defaultdict(set)
    for stem in stems:
        speaker, chapter, _ = parse_stem(stem)
        chapter_speaker[chapter].add(speaker)
    shared_chapters = sorted(c for c, s in chapter_speaker.items() if len(s) > 1)
    admitted = [s for s in stems if s not in excluded]
    per_speaker = Counter(parse_stem(s)[0] for s in admitted)
    sizes = sorted(per_speaker.values())
    return {
        "schema_version": 1,
        "task": "R1-S1",
        "stems_in_coverage": len(coverage_rows),
        "format_errors": format_errors,
        "format_consistent": not format_errors,
        "chapters": len(chapter_speaker),
        "chapters_shared_by_multiple_speakers": shared_chapters,
        "stems_admitted": len(admitted),
        "stems_excluded": excluded,
        "speakers": len(per_speaker),
        "utterances_per_speaker": {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]} if sizes else None,
        "per_speaker_counts": dict(sorted(per_speaker.items(), key=lambda kv: int(kv[0]))),
        "components": list(expected_components),
        "grouping_unit": "speaker" if not format_errors and not shared_chapters else "stem",
        "admitted_stems": admitted,
    }


# ---------------------------------------------------------------- R1-S3 materialisation

def _pcm16(samples):
    import numpy as np
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 32767 / 32768)
    return np.round(clipped * 32768.0).astype("<i2")


def _read_bytes(path):
    return Path(path).read_bytes()


# Errors that cannot succeed on retry. Any other OSError (e.g. Errno 5 / 107 from a Google Drive mount) is retried.
_PERMANENT_READ_ERRORS = (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)


def read_bytes_with_retries(path, *, retries: int = 3, backoff: float = 1.0, sleep=None):
    """Read a file, retrying transient I/O errors with exponential backoff; the last error is re-raised."""
    attempt = 0
    while True:
        try:
            return _read_bytes(path)
        except _PERMANENT_READ_ERRORS:
            raise
        except OSError:
            if attempt >= retries:
                raise
            (sleep or time.sleep)(backoff * 2 ** attempt)
            attempt += 1


def canonical_pcm(path_or_bytes, *, max_seconds: float = 60.0):
    """Decode to mono 16 kHz int16 using the project decoder (single resample)."""
    from echoproof.data.audio import decode_audio
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else read_bytes_with_retries(path_or_bytes)
    wave = decode_audio(data, target_rate=SAMPLE_RATE, max_bytes=max(len(data), 1), max_seconds=max_seconds)
    pcm = _pcm16(wave.samples)
    return pcm, wave


def pcm_sha256(pcm) -> str:
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def flac_bytes(pcm) -> bytes:
    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(buffer, pcm, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def read_flac(data: bytes):
    import soundfile as sf
    pcm, rate = sf.read(io.BytesIO(data), dtype="int16", always_2d=False)
    if rate != SAMPLE_RATE:
        raise CohortError("unexpected shard sample rate")
    return pcm


def resolve_duplicates(rows):
    """Exact-duplicate disposition on canonical PCM hashes.

    rows: dicts with recording_id, label, pcm_sha256 (None means undecodable and is ignored).
    Same hash + same label -> keep the lexicographically first, exclude the rest ('exact_duplicate').
    Same hash + different labels -> exclude every member ('label_conflict').
    Returns {recording_id: reason} for excluded rows.
    """
    by_hash = defaultdict(list)
    for row in rows:
        if row.get("pcm_sha256"):
            by_hash[row["pcm_sha256"]].append(row)
    excluded = {}
    for members in by_hash.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda r: r["recording_id"])
        if len({m["label"] for m in members}) > 1:
            for m in members:
                excluded[m["recording_id"]] = "label_conflict"
        else:
            for m in members[1:]:
                excluded[m["recording_id"]] = "exact_duplicate"
    return excluded


def write_shards(items, out_dir, *, shard_bytes: int = 256 * 1024 * 1024, prefix: str = "shard"):
    """Write (key, flac_bytes) items into tar shards; return {key: (shard_name, size)}.

    Refuses to overwrite an existing shard. Each member is named '<key>.flac' with ':' replaced by '__'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index, tar, size, number = {}, None, 0, 0
    try:
        for key, blob in items:
            if tar is None or size >= shard_bytes:
                if tar is not None:
                    tar.close()
                name = f"{prefix}_{number:02d}.tar"
                number += 1
                path = out_dir / name
                if path.exists():
                    raise CohortError(f"refusing to overwrite {path}")
                tar, size = tarfile.open(path, "w"), 0
            member = tarfile.TarInfo(key.replace(":", "__") + ".flac")
            member.size = len(blob)
            tar.addfile(member, io.BytesIO(blob))
            size += len(blob)
            index[key] = (name, len(blob))
    finally:
        if tar is not None:
            tar.close()
    return index


def read_from_shard(shard_path, key):
    member_name = key.replace(":", "__") + ".flac"
    with tarfile.open(shard_path, "r") as tar:
        handle = tar.extractfile(member_name)
        if handle is None:
            raise CohortError(f"{key} not in shard")
        return handle.read()


def stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_table(rows, path) -> None:
    """Plain (non-envelope) Parquet table for the cohort manifest; refuses to overwrite."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    if path.exists():
        raise CohortError(f"refusing to overwrite {path}")
    if not rows:
        raise CohortError("cannot write an empty table")
    pq.write_table(pa.Table.from_pylist(list(rows)), str(path))


def read_table(path):
    import pyarrow.parquet as pq
    return pq.read_table(str(path)).to_pylist()


SOURCE_DIRS = {"genuine": "dataset/genuine/wav", "xtts": "dataset/tts/XTTSv2", "chatterbox": "dataset/tts/ChatterboxTTS",
               "melo": "dataset/tts/MeloTTS", "vits": "dataset/tts/VITS"}


def _failed_row(base, reason):
    return {**base, "duration_s": 0.0, "n_samples": 0, "original_sample_rate": 0, "canonical_path": None,
            "pcm_sha256": None, "retained": False, "exclusion_reason": reason, "quality_flags": ""}


def load_recording(base, path):
    """Read, decode and canonicalise one recording -> (cohort row, FLAC bytes or None).

    Failures become retained=False rows with a reason that keeps the error class and message:
    'read_error:<Class>:<message>' (the file could not be read, e.g. a Drive I/O error) or
    'decode_error:<Class>:<message>' (it was read but is not valid audio). Nothing is dropped silently.
    """
    try:
        data = read_bytes_with_retries(path)
    except OSError as exc:
        return _failed_row(base, f"read_error:{type(exc).__name__}:{str(exc)[:120]}"), None
    try:
        pcm, wave = canonical_pcm(data)
        if len(pcm) == 0:
            raise CohortError("empty audio")
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest
        return _failed_row(base, f"decode_error:{type(exc).__name__}:{str(exc)[:120]}"), None
    row = {**base, "duration_s": len(pcm) / SAMPLE_RATE, "n_samples": int(len(pcm)), "original_sample_rate": int(wave.original_sample_rate),
           "canonical_path": None, "pcm_sha256": pcm_sha256(pcm), "retained": True, "exclusion_reason": None,
           "quality_flags": ",".join(wave.quality_flags)}
    return row, flac_bytes(pcm)


REPAIRABLE_PREFIXES = ("read_error:", "decode_error:")


def repair(rows, project_root, out_dir, *, shard_bytes: int = 256 * 1024 * 1024):
    """Retry only the recordings that failed to read/decode; return (updated rows, report).

    New audio goes to a separate 'repair_NN.tar' shard (existing shards and index.parquet are never modified; an existing
    repair shard blocks a second run). Exact-duplicate disposition is recomputed over all retained rows and any change is
    reported. Rows excluded for other reasons (duplicates, label conflicts) are left alone.
    """
    project_root, out_dir = Path(project_root), Path(out_dir)
    rows = [dict(r) for r in rows]
    by_id = {r["recording_id"]: r for r in rows}
    todo = sorted((r for r in rows if not r["retained"] and (r["exclusion_reason"] or "").startswith(REPAIRABLE_PREFIXES)),
                  key=lambda r: r["recording_id"])
    repaired, blobs, still_failing = [], [], []
    for old in todo:
        base = {k: old[k] for k in ("recording_id", "stem", "speaker_id", "chapter_id", "component", "label", "generator")}
        row, blob = load_recording(base, project_root / SOURCE_DIRS[old["component"]] / f"{old['stem']}.wav")
        if blob is None:
            still_failing.append({"recording_id": old["recording_id"], "reason": row["exclusion_reason"]})
        else:
            blobs.append((old["recording_id"], blob))
            by_id[old["recording_id"]].update(row)
            repaired.append(old["recording_id"])
    index = write_shards(iter(blobs), out_dir, shard_bytes=shard_bytes, prefix="repair") if blobs else {}
    for rid, (shard, _size) in index.items():
        by_id[rid]["canonical_path"] = f"{shard}::{rid}"
    changed = {}
    for rid, reason in resolve_duplicates([r for r in rows if r["retained"]]).items():
        by_id[rid]["retained"], by_id[rid]["exclusion_reason"] = False, reason
        changed[rid] = reason
    return rows, {"attempted": [r["recording_id"] for r in todo], "repaired": repaired, "still_failing": still_failing,
                  "duplicate_dispositions_changed": changed, "repair_shards": sorted({v[0] for v in index.values()}),
                  "repair_bytes": sum(v[1] for v in index.values())}


def materialize(plan, project_root, out_dir, *, shard_bytes: int = 256 * 1024 * 1024, progress=None):
    """Decode every planned recording once, write FLAC shards, return (cohort rows, shard index) (R1-S3).

    plan: sorted list of stems. Reads <project_root>/<SOURCE_DIRS[component]>/<stem>.wav.
    Undecodable files become retained=False rows; exact duplicates are resolved afterwards.
    Excluded duplicates stay in the shards (flagged in the manifest) so that decoding happens only once.
    """
    project_root, out_dir = Path(project_root), Path(out_dir)
    rows = []

    def stream():
        for stem in plan:
            speaker, chapter, _ = parse_stem(stem)
            for component in COMPONENTS:
                rid = recording_id(component, stem)
                base = {"recording_id": rid, "stem": stem, "speaker_id": speaker, "chapter_id": chapter,
                        "component": component, "label": LABEL[component], "generator": GENERATOR[component]}
                row, blob = load_recording(base, project_root / SOURCE_DIRS[component] / f"{stem}.wav")
                rows.append(row)
                if progress and len(rows) % 500 == 0:
                    progress(len(rows))
                if blob is not None:
                    yield rid, blob

    index = write_shards(stream(), out_dir, shard_bytes=shard_bytes)
    by_id = {row["recording_id"]: row for row in rows}
    for rid, (shard, _size) in index.items():
        by_id[rid]["canonical_path"] = f"{shard}::{rid}"
    for rid, reason in resolve_duplicates(rows).items():
        by_id[rid]["retained"], by_id[rid]["exclusion_reason"] = False, reason
    return rows, index
