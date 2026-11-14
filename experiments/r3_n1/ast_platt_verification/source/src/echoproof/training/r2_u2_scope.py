"""Scope/freeze/parity helpers reused from the historical R2-U2 implementation.
No historical launch, old source-hash waiver, or publication code is included.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
import hashlib
import re
from pathlib import Path
from echoproof.artifacts import canonical_json, digest
from echoproof.data.chunks import PREPROCESSING_HASH, eval_windows
ATOL = RTOL = 1e-4


class WarmupError(ValueError):
    """Invalid recipe, scope, evidence or warm-up state; never silently repaired."""


def _require(condition, message):
    if not condition:
        raise WarmupError(message)


@dataclass
class Plan:
    train_rows: list
    selection_rows: list
    selection_chunks: list
    split_sha256: str
    provenance: dict = field(default_factory=dict)

    @property
    def counts(self):
        return {"train_recordings": len(self.train_rows),
                "train_by_component": dict(sorted(Counter(r["component"] for r in self.train_rows).items())),
                "selection_recordings": len(self.selection_rows),
                "selection_by_component": dict(sorted(Counter(r["component"] for r in self.selection_rows).items())),
                "selection_chunks": len(self.selection_chunks)}

    @property
    def sha256(self):
        return digest({"train": self.train_rows, "selection": self.selection_rows,
                       "chunks": [asdict(c) for c in self.selection_chunks], "split_sha256": self.split_sha256,
                       "provenance": self.provenance})

    def shards(self):
        return sorted({r["canonical_path"].split("::", 1)[0] for r in self.train_rows + self.selection_rows})


def select_plan(cohort_rows, assignments, chunks):
    """Validate metadata and select B train plus selection validation, never audio.

    Retention in the overall cohort does not imply membership of Protocol B.
    In particular, XTTS recordings of training speakers are correctly unassigned.
    """
    cohort_rows, assignments, chunks = list(cohort_rows), list(assignments), list(chunks)
    by_id = {r["recording_id"]: dict(r) for r in cohort_rows}
    _require(len(by_id) == len(cohort_rows), "duplicate cohort recording_id")
    allowed = {"train": {"genuine", "melo", "vits"}, "validation": {"genuine", "melo", "vits", "xtts"},
               "test": {"genuine", "melo", "vits", "chatterbox"}}
    seen, groups, pcm, split_hashes, selected = {}, {}, {}, set(), {"train": [], "selection": []}
    for a in assignments:
        _require(a.protocol_id == "protocol:B", "non-B assignment")
        _require(a.recording_id not in seen, "duplicate assignment")
        seen[a.recording_id] = a
        split_hashes.add(a.split_hash)
        row = by_id.get(a.recording_id)
        _require(row is not None, "assignment without cohort recording")
        component, part = row["component"], a.partition
        _require(part in allowed and component in allowed[part], "forbidden component in Protocol B partition")
        _require(row["label"] == int(component != "genuine"), "component/label mismatch")
        stem = row.get("stem", "")
        _require(re.fullmatch(r"[0-9]+-[0-9]+-[0-9]{4}", stem) is not None
                 and stem.split("-", 1)[0] == str(row["speaker_id"])
                 and a.recording_id == f"phonemedf:{component}:{stem}", "recording identity/stem mismatch")
        _require(a.group_id == "speaker:" + str(row["speaker_id"]), "speaker group mismatch")
        _require(a.validation_role in {"selection", "calibration", "operating"} if part == "validation"
                 else a.validation_role is None, "invalid validation role")
        slot = (part, a.validation_role)
        _require(groups.setdefault(a.group_id, slot) == slot, "speaker crosses partition or validation role")
        if not row["retained"]:
            continue
        _require(re.fullmatch(r"[0-9a-f]{64}", row.get("pcm_sha256") or "") is not None, "invalid canonical PCM hash")
        _require(pcm.setdefault(row["pcm_sha256"], (slot, row["label"])) == (slot, row["label"]),
                 "PCM duplicate crosses partition/role or conflicts in label")
        if part != "train" and not (part == "validation" and a.validation_role == "selection"):
            continue
        path = row.get("canonical_path", "").split("::")
        _require(len(path) == 2 and re.fullmatch(r"shard_[0-9]+\.tar", path[0]) is not None
                 and path[1] == a.recording_id and "/" not in path[1] and "\\" not in path[1], "unsafe canonical path")
        _require(type(row.get("n_samples")) is int and row["n_samples"] > 0, "invalid canonical sample count")
        selected["train" if part == "train" else "selection"].append(row)
    _require(len(split_hashes) == 1, "mixed or absent split hashes")
    selected_rows = {r["recording_id"]: r for r in selected["selection"]}
    windows, selection_chunks, chunk_ids = defaultdict(list), [], set()
    for c in chunks:
        if c.protocol_id != "protocol:B":
            continue
        _require(c.chunk_id not in chunk_ids, "duplicate Protocol B chunk_id")
        chunk_ids.add(c.chunk_id)
        _require(c.recording_id in seen, "orphan Protocol B chunk")
        a = seen[c.recording_id]
        _require(c.partition == a.partition and c.group_id == a.group_id, "chunk inheritance mismatch")
        if c.recording_id not in selected_rows:
            continue
        _require(c.sample_rate == 16000 and c.input_samples == 64000 and c.preprocessing_hash == PREPROCESSING_HASH,
                 "selection chunk preprocessing mismatch")
        _require(c.chunk_id == f"chunk:B:{c.recording_id}:{c.start_sample}", "selection chunk identity mismatch")
        windows[c.recording_id].append((c.start_sample, c.end_sample))
        selection_chunks.append(c)
    for rid, row in selected_rows.items():
        _require(sorted(windows[rid]) == eval_windows(row["n_samples"]), "missing, duplicate or noncanonical selection windows")
    _require(selected["train"] and selected["selection"], "empty training or selection scope")
    for scope in selected:
        selected[scope].sort(key=lambda r: r["recording_id"])
        _require({r["label"] for r in selected[scope]} == {0, 1}, f"{scope} lacks a class")
    return Plan(selected["train"], selected["selection"],
                sorted(selection_chunks, key=lambda c: (c.recording_id, c.start_sample)), next(iter(split_hashes)))


class CheckedWaveformLoader:
    """Scope guard plus canonical PCM identity check on the first decode per ID."""
    def __init__(self, shard_dir, rows):
        from echoproof.training.datasets import ShardWaveformLoader
        rows = list(rows)
        self.rows = {r["recording_id"]: r for r in rows}
        _require(len(self.rows) == len(rows), "duplicate loader recording_id")
        self.loader = ShardWaveformLoader(shard_dir, {rid: r["canonical_path"] for rid, r in self.rows.items()})
        self.verified_ids = set()

    def __call__(self, recording_id):
        import numpy as np
        _require(recording_id in self.rows, "audio outside declared training/selection scope")
        wave = self.loader(recording_id)
        if recording_id not in self.verified_ids:
            row = self.rows[recording_id]
            _require(wave.ndim == 1 and len(wave) == row["n_samples"] and np.isfinite(wave).all(), "canonical PCM length/values differ")
            pcm = np.rint(wave.astype(np.float64) * 32768).astype("<i2")
            _require(hashlib.sha256(pcm.tobytes()).hexdigest() == row["pcm_sha256"], "canonical PCM checksum mismatch")
            self.verified_ids.add(recording_id)
        return wave

    def close(self):
        self.loader.close()


def validate_head_stage(model):
    """The complete classifier (including its LayerNorm) is the trainable head."""
    _require(model.stage == "head", "AST is not in head warm-up stage")
    named = list(model.model.named_parameters())
    trainable = {name for name, p in named if p.requires_grad}
    expected = {name for name, _ in named if name.startswith("classifier.")}
    _require(trainable == expected and expected, "trainable parameters differ from complete classifier.* head")
    return model.parameter_report()


def frozen_state_digest(model):
    """Include buffers as well as parameters when proving backbone immutability."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.model.state_dict().items()):
        if not name.startswith("classifier."):
            value = tensor.detach().cpu().contiguous()
            hasher.update(canonical_json([name, str(value.dtype), list(value.shape)]).encode("utf-8"))
            hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def _finite_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        _require(torch.isfinite(value).all().item(), "checkpoint contains a non-finite tensor")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tensors(item)


def _state_proof(model, baseline):
    observed = model.digests()
    original = baseline["initial_group_sha256"]
    _require(set(observed) == set(original), "AST parameter group layout changed")
    unchanged = sorted(k for k in observed if observed[k] == original[k])
    changed = sorted(k for k in observed if observed[k] != original[k])
    _require(changed == ["classifier"], "warm-up must change the classifier and no backbone parameter group")
    state_digest = frozen_state_digest(model)
    _require(state_digest == baseline["initial_frozen_state_sha256"], "a frozen backbone parameter or buffer changed")
    return {"passed": True, "changed_groups": changed, "unchanged_groups": unchanged,
            "group_sha256": observed, "frozen_state_sha256": state_digest}


def _prediction_parity(run_dir, plan, chunk_rows, recordings, epoch):
    import numpy as np
    import pyarrow.parquet as pq
    saved_chunks = pq.read_table(Path(run_dir) / "predictions_chunks.parquet").to_pylist()
    saved_recordings = pq.read_table(Path(run_dir) / "predictions_recordings.parquet").to_pylist()
    current = {r[0]: r for r in chunk_rows}
    expected_chunks = {c.chunk_id: c for c in plan.selection_chunks}
    labels = {r["recording_id"]: r["label"] for r in plan.selection_rows}
    expected_ids = {c.chunk_id for c in plan.selection_chunks}
    _require(len(current) == len(chunk_rows) == len(saved_chunks) == len(expected_ids)
             and set(current) == expected_ids and {r["chunk_id"] for r in saved_chunks} == expected_ids,
             "selection prediction chunk coverage mismatch")
    max_chunk = 0.0
    for row in saved_chunks:
        now = current[row["chunk_id"]]
        expected = expected_chunks[row["chunk_id"]]
        _require(now[1] == expected.recording_id and now[2] == labels[expected.recording_id],
                 "reloaded selection chunk differs from frozen ID/label")
        _require(row["recording_id"] == now[1] and row["label"] == now[2] and row["epoch"] == epoch,
                 "saved selection chunk metadata mismatch")
        _require(np.isfinite(row["tts_probability"]) and 0 <= row["tts_probability"] <= 1 and 0 <= now[3] <= 1
                 and np.isclose(row["tts_probability"], now[3], atol=ATOL, rtol=RTOL),
                 "best-checkpoint chunk reload parity failed")
        max_chunk = max(max_chunk, abs(row["tts_probability"] - now[3]))
    current = {r[0]: r for r in recordings}
    expected_ids = {r["recording_id"] for r in plan.selection_rows}
    _require(len(current) == len(recordings) == len(saved_recordings) == len(expected_ids)
             and set(current) == expected_ids and {r["recording_id"] for r in saved_recordings} == expected_ids,
             "selection prediction recording coverage mismatch")
    max_recording = 0.0
    chunk_counts = Counter(c.recording_id for c in plan.selection_chunks)
    for row in saved_recordings:
        now = current[row["recording_id"]]
        _require(now[1] == labels[now[0]] and now[3] == chunk_counts[now[0]], "selection recording label/chunk-count differs from plan")
        _require(row["label"] == now[1] and row["chunks"] == now[3] and row["epoch"] == epoch,
                 "saved selection recording metadata mismatch")
        _require(np.isfinite(row["raw_score"]) and 0 <= row["raw_score"] <= 1 and 0 <= now[2] <= 1
                 and np.isclose(row["raw_score"], now[2], atol=ATOL, rtol=RTOL),
                 "best-checkpoint recording reload parity failed")
        max_recording = max(max_recording, abs(row["raw_score"] - now[2]))
    return {"passed": True, "scope": "selection validation only", "atol": ATOL, "rtol": RTOL,
            "chunks": len(chunk_rows), "recordings": len(recordings),
            "max_abs_chunk_difference": max_chunk, "max_abs_recording_difference": max_recording}
