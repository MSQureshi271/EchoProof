"""R2-U2: audited Protocol B AST head warm-up using the unchanged R1 trainer.

Planning reads metadata only. Production fitting is CUDA-only and consumes just
B training recordings and selection-validation chunks. No final-test, calibration
or operating audio is made available to the loader. A published manifest denotes
an executed engineering gate, with independent teammate review still pending.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass, field, fields
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time
import uuid

from echoproof.artifacts import canonical_json, digest, file_digest, write_json
from echoproof.data.chunks import PREPROCESSING_HASH, eval_windows

TASK = "R2-U2"
MANIFEST_NAME = "r2_u2_manifest.json"
RECIPE_PATH = "configs/r2_u2_ast_b.json"
CHECKPOINT_POLICY = "best selection recording macro_f1 at threshold 0.5; first epoch wins ties"
ATOL = RTOL = 1e-4
CORE_INPUTS = ("data/manifests/cohort_v1.parquet", "data/splits/protocol_B_v1.parquet",
               "data/processed/chunks_v1.parquet", "reports/r1/split_report.json", "reports/r1/leakage.json",
               "reports/r1/ast_probe.json", "reports/r1/r1_m1_colab_20261103T182916Z.json")


class WarmupError(ValueError):
    """Invalid recipe, scope, evidence or warm-up state; never silently repaired."""


def _require(condition, message):
    if not condition:
        raise WarmupError(message)


def _inside(root, path):
    root = Path(root).resolve()
    path = Path(path)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    _require(resolved.is_relative_to(root), f"path outside project: {path}")
    return resolved


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_recipe(root, recipe_path=None):
    """Load the explicitly recorded recipe; defaults are never filled silently."""
    recipe = _read(_inside(root, recipe_path or RECIPE_PATH))
    validate_recipe(recipe)
    return recipe


def validate_recipe(recipe):
    from echoproof.training.trainer import TrainConfig
    _require(recipe.get("schema_version") == 1 and recipe.get("task") == TASK, "invalid R2-U2 recipe schema/task")
    train = recipe.get("train", {})
    _require(set(train) == {f.name for f in fields(TrainConfig)}, "recipe must explicitly specify every TrainConfig field")
    cfg = TrainConfig(**train)
    cfg.validate()
    _require(1 <= cfg.max_epochs <= 3 and cfg.protocol == "B" and cfg.model_name == "ast_frozen",
             "warm-up must be 1-3 epochs of the Protocol B frozen AST")
    _require(cfg.owner and cfg.precision == "fp16" and cfg.device == "cuda" and cfg.deterministic is True,
             "production recipe requires named owner, CUDA FP16 and explicit determinism")
    _require(cfg.num_workers == 0 and cfg.scheduler == "constant" and cfg.warmup_steps == 0
             and cfg.selection_threshold == 0.5 and cfg.class_strategy == "none" and cfg.min_delta == 0,
             "recipe differs from the declared constant, unweighted, selection-only baseline")
    _require(cfg.batch_size == 4 and cfg.accumulation_steps == 4 and cfg.eval_batch_size == 16,
             "head microbatch/accumulation/evaluation must be 4/4/16")
    for value in (cfg.lr, cfg.weight_decay, cfg.grad_clip, cfg.min_delta):
        _require(math.isfinite(value), "non-finite recipe setting")
    _require(recipe.get("checkpoint_policy") == CHECKPOINT_POLICY, "unknown warm-up checkpoint policy")
    _require(set(recipe.get("input_sha256", {})) >= set(CORE_INPUTS), "recipe omits required frozen input/evidence hashes")
    for name, checksum in recipe["input_sha256"].items():
        _require(not Path(name).is_absolute() and ".." not in Path(name).parts
                 and re.fullmatch(r"[0-9a-f]{64}", checksum or "") is not None, "invalid input path/hash")
    expected_processor = {"sampling_rate": 16000, "do_normalize": True, "num_mel_bins": 128,
                          "max_length": 1024, "mean": -4.2677393, "std": 4.5689974}
    _require(recipe.get("preprocessor") == expected_processor, "recipe changes the pinned AST processor/grid")
    _require(isinstance(recipe.get("decisions"), dict) and recipe["decisions"], "recipe decisions must be recorded")
    return recipe


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


def preflight(root, recipe=None):
    """Validate frozen metadata/evidence; do not download models or open audio."""
    from echoproof.contracts import Chunk, SplitAssignment, read_parquet
    from echoproof.data.cohort import read_table
    from echoproof.features.pretrained import PRETRAINED
    root = Path(root).resolve()
    recipe = load_recipe(root) if recipe is None else validate_recipe(recipe)
    actual = {}
    for name, expected in recipe["input_sha256"].items():
        path = _inside(root, name)
        _require(path.is_file(), f"missing frozen input: {name}")
        actual[name] = file_digest(path)
        _require(actual[name] == expected, f"frozen input/evidence hash mismatch: {name}")
    split_report = _read(root / "reports/r1/split_report.json")
    leakage = _read(root / "reports/r1/leakage.json")
    ast = _read(root / "reports/r1/ast_probe.json")
    trainer = _read(root / "reports/r1/r1_m1_colab_20261103T182916Z.json")
    _require(leakage.get("passed") is True and leakage.get("violation_count") == 0, "R1 leakage evidence did not pass")
    _require(ast.get("verdict", {}).get("verified") is True and trainer.get("verdict", {}).get("verified") is True,
             "R1 AST/trainer evidence did not pass")
    for name, relative in (("ast_detector.py", "src/echoproof/models/ast_detector.py"),
                           ("pretrained.py", "src/echoproof/features/pretrained.py")):
        _require(file_digest(root / relative) == ast["source_sha256"][name], f"R1 AST source changed: {relative}")
    _require(file_digest(root / "src/echoproof/training/trainer.py") == trainer["trainer_sha256"], "shared trainer differs from R1 evidence")
    _require(ast["weights"]["revision"] == PRETRAINED["ast"]["revision"]
             and ast["weights"]["sha256"] == PRETRAINED["ast"]["weights_sha256"]
             and ast["weights"].get("recorded_sha256_matches") is True, "AST pin differs from R1 evidence")
    _require(all(ast["preprocessor"].get(k) == v for k, v in recipe["preprocessor"].items()),
             "AST processor differs from R1 evidence")
    plan = select_plan(read_table(root / CORE_INPUTS[0]), read_parquet(str(root / CORE_INPUTS[1]), SplitAssignment),
                       read_parquet(str(root / CORE_INPUTS[2]), Chunk))
    expected = split_report["protocols"]["B"]
    _require(plan.split_sha256 == expected["split_sha256"], "Protocol B split differs from frozen report")
    support = expected["support"]
    train_expected = sum(v["recordings"] for k, v in support.items() if k.startswith("train|"))
    selection_expected = sum(v["recordings"] for k, v in support.items() if k.startswith("validation|selection|"))
    _require(plan.counts["train_recordings"] == train_expected and plan.counts["selection_recordings"] == selection_expected,
             "retained training/selection coverage differs from frozen split report")
    plan.provenance = {"input_sha256": actual, "split_sha256": plan.split_sha256,
                       "scope": "B train and selection validation only; no calibration, operating or test audio"}
    return plan


def source_hashes(root):
    """Bind the runtime and test dependencies without capturing unrelated stages.

    Adding a separate R3 module must not invalidate this unchanged warm-up.
    Changes to the shared code actually used here still invalidate its identity.
    """
    root = Path(root).resolve()
    names = ["src/echoproof/" + name for name in (
        "__init__.py", "artifacts.py", "contracts.py", "gpu_budget.py",
        "data/__init__.py", "data/audio.py", "data/cohort.py", "data/chunks.py", "data/splits.py",
        "features/__init__.py", "features/pretrained.py", "models/__init__.py", "models/ast_detector.py",
        "training/__init__.py", "training/r2_u2.py", "training/trainer.py", "training/datasets.py",
        "training/seeding.py", "training/ledger.py")]
    names += ["scripts/r2_u2_ast_warmup.py", "tests/test_r2_ast_warmup.py", "tests/test_r1_u2_ast.py",
              "tests/test_m1_trainer.py", RECIPE_PATH]
    return {name: file_digest(root / name) for name in sorted(set(names))}


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


def _test_gate(root, path, hashes):
    path = _inside(root, path)
    value = _read(path)
    _require(value.get("task") == "R2-U2 fixture verification" and value.get("returncode") == 0
             and value.get("skipped") is False and value.get("passed_count", 0) > 0,
             "fixture gate requires successful tests with zero skips")
    _require(value.get("source_sha256") == hashes, "fixture report is not bound to the current sources/config")
    command = value.get("command", [])
    expected = ["-B", "-m", "pytest", "tests/test_r2_ast_warmup.py", "tests/test_r1_u2_ast.py",
                "tests/test_m1_trainer.py", "-q", "-ra", "-p", "no:cacheprovider"]
    _require(isinstance(command, list) and command[1:] == expected and isinstance(command[0], str),
             "fixture report must execute all three complete required suites, without deselection or collection-only flags")
    return {"file": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def _runtime():
    import torch
    import transformers
    _require(torch.cuda.is_available(), "R2-U2 requires a prepared Colab CUDA runtime")
    _require(transformers.__version__ == "5.22.0", "AST checkpoint compatibility requires transformers==5.22.0")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "numpy", "pyarrow", "soundfile", "huggingface-hub", "safetensors", "torchaudio")}
    return {"python": platform.python_version(), "versions": versions, "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


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


def validate_training_receipts(run_dir, baseline):
    """Read persisted observations without claiming new training on resume."""
    run_dir = Path(run_dir)
    identity = {"baseline_sha256": file_digest(run_dir / "baseline.json"),
                "provenance_sha256": digest(baseline["provenance"])}
    for name in ("training_mode_receipt.json", "head_gradient_receipt.json"):
        receipt = _read(run_dir / name)
        _require(receipt.get("schema_version") == 1 and receipt.get("task") == TASK
                 and all(receipt.get(k) == v for k, v in identity.items()), "persisted training proof identity differs")
        if name == "training_mode_receipt.json":
            _require(receipt.get("backbone_all_modules_eval") is True
                     and receipt.get("trainable_names") == baseline["trainability"]["trainable_names"],
                     "missing actual frozen-backbone evaluation-mode proof")
        else:
            _require(receipt.get("finite_nonzero_scaled_classifier_gradient_observed") is True,
                     "missing actual finite nonzero classifier-gradient proof")
    return {"passed": True, "proof": "persisted actual training observations bound to original baseline"}


def validate_resume_state(run_dir, config_hash, max_epochs):
    """Fail closed if a hard stop left epoch artifacts ahead of latest.pt.

    The shared trainer commits best/predictions/history before latest.pt. Such an
    interruption needs inspection; it must not append a duplicate epoch on retry.
    This helper only reads generated checkpoint and prediction artifacts.
    """
    import torch
    import pyarrow.parquet as pq
    run_dir = Path(run_dir)
    latest = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    _finite_tensors(latest)
    _finite_tensors(best)
    _require(latest["config_hash"] == best["config_hash"] == config_hash, "committed checkpoint configuration mismatch")
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    _require(type(latest["epoch"]) is int and 1 <= latest["epoch"] <= max_epochs
             and [row["epoch"] for row in history] == list(range(1, latest["epoch"] + 1)),
             "history is ahead of or inconsistent with the committed latest checkpoint; inspect interrupted epoch artifacts")
    scores = [row["val"]["macro_f1"] for row in history]
    _require(all(math.isfinite(s) for s in scores), "non-finite checkpoint selection score")
    best_epoch = max(range(len(scores)), key=lambda i: scores[i]) + 1
    _require(latest["best_epoch"] == best["epoch"] == best_epoch
             and latest["best_metric"] == scores[best_epoch - 1]
             and best["metrics"] == history[best_epoch - 1]["val"],
             "best checkpoint/metrics do not match committed selection history")
    for name in ("predictions_chunks.parquet", "predictions_recordings.parquet"):
        table = pq.read_table(run_dir / name, columns=["epoch"]).to_pydict()
        _require(table["epoch"] and set(table["epoch"]) == {best_epoch}, "saved predictions belong to an uncommitted or wrong best epoch")
    return latest, best, history


def recover_status(run_dir, summary):
    """Recover only a missing completion summary from an actual finished fit."""
    path = Path(run_dir) / "status.json"
    if path.exists():
        _require(_read(path) == summary, "status summary differs from finished checkpoint")
        return False
    write_json(path, summary)
    return True


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


def _ledger_row(path, run_id, cfg, config_hash, latest, best, split_sha256):
    from echoproof.training.ledger import append_ledger
    existing = []
    if Path(path).exists():
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            existing = [r for r in csv.DictReader(handle) if r["run_id"] == run_id and r["status"] == "completed"]
    _require(len(existing) <= 1, "duplicate completion rows for warm-up run")
    if not existing:
        # A hard stop between latest.pt and the trainer's completion append can
        # leave a fully finished checkpoint. Recover its actual saved facts only.
        _require(latest["finished"] and latest["config_hash"] == config_hash, "missing completed ledger row")
        row = {"run_id": run_id, "owner": cfg.owner, "model": cfg.model_name, "protocol": cfg.protocol,
               "config_hash": config_hash, "split_hash": split_sha256, "seed": cfg.seed,
               "epochs_run": latest["epoch"], "best_epoch": best["epoch"], "stop_reason": latest["stop_reason"],
               "val_macro_f1": best["metrics"]["macro_f1"], "val_eer": best["metrics"].get("eer"),
               "val_fpr": best["metrics"].get("fpr"), "train_minutes": round(latest["train_seconds"] / 60, 3),
               "device": "cuda", "status": "completed",
               "notes": "completion row recovered from finished checkpoint after interrupted finalization"}
        append_ledger(path, row)
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            existing = [r for r in csv.DictReader(handle) if r["run_id"] == run_id and r["status"] == "completed"]
    row = existing[0]
    _require(row["config_hash"] == config_hash and row["split_hash"] == split_sha256 and row["model"] == cfg.model_name
             and row["owner"] == cfg.owner and int(row["seed"]) == cfg.seed
             and row["protocol"] == "B" and int(row["epochs_run"]) == latest["epoch"]
             and int(row["best_epoch"]) == best["epoch"] and row["stop_reason"] == latest["stop_reason"],
             "ledger does not describe the saved warm-up")
    for column, metric in (("val_macro_f1", "macro_f1"), ("val_eer", "eer"), ("val_fpr", "fpr")):
        _require(math.isclose(float(row[column]), float(best["metrics"][metric]), abs_tol=1e-12, rel_tol=1e-12),
                 "ledger metrics differ from selected checkpoint")
    return {"file": "reports/ledger.csv", "run_id": run_id, "row": row, "row_sha256": digest(row)}


def run_warmup(root, shard_dir, run_dir, test_report, *, resume=False, recipe_path=None, model_cache=None):
    """Execute the fixed recipe on CUDA; preserve evidence and the original baseline.

    This API is for the Colab notebook. Preparation/imports/tests and metadata
    checks finish before output creation or research audio is read.
    """
    import torch
    import pyarrow.parquet  # noqa: F401 -- fail before opening an output directory
    from echoproof.features.pretrained import PRETRAINED, pinned_snapshot
    from echoproof.models.ast_detector import ASTDetector, BACKBONE
    from echoproof.training.datasets import CropDataset, ChunkDataset
    from echoproof.training.trainer import TrainConfig, Trainer
    root = Path(root).resolve()
    _require((root / "pyproject.toml").is_file(), "root is not the selected EchoProof project")
    _require(Path(__file__).resolve() == root / "src/echoproof/training/r2_u2.py",
             "execute the R2-U2 module from the same project selected by root")
    run_dir = _inside(root, run_dir)
    _require(run_dir.is_relative_to(root / "runs/r2") and run_dir != root / "runs/r2", "warm-up run must be under runs/r2/<run_id>")
    shard_dir = Path(shard_dir).resolve()
    _require(not shard_dir.is_relative_to(root), "stage canonical shards on Colab scratch before fitting")
    recipe = load_recipe(root, recipe_path)
    plan = preflight(root, recipe)
    hashes = source_hashes(root)
    fixture_reference = _test_gate(root, test_report, hashes)
    environment = _runtime()
    cfg = TrainConfig(**recipe["train"])
    for shard in plan.shards():
        _require((shard_dir / shard).is_file(), f"missing staged canonical shard: {shard}")
    if run_dir.exists():
        _require(resume or not any(run_dir.iterdir()), "run directory exists; resume explicitly or choose a new run_id")
        if (run_dir / MANIFEST_NAME).exists():
            verify_run(root, run_dir)
            return _read(run_dir / MANIFEST_NAME)
    else:
        _require(not resume, "resume requested for a missing run directory")
        run_dir.mkdir(parents=True)
    lock = run_dir / ".writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise WarmupError("writer lock exists; confirm the previous writer stopped before removing that lock") from exc
    os.close(descriptor)
    loader, hooks = None, []
    started = time.perf_counter()
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        snapshot = pinned_snapshot("ast", cache_dir=model_cache)
        processor_file = snapshot / "preprocessor_config.json"
        processor = _read(processor_file)
        _require(all(processor.get(k) == v for k, v in recipe["preprocessor"].items()), "downloaded processor differs from frozen AST preprocessing")
        model = ASTDetector.from_dir(snapshot, device="cuda", seed=cfg.seed)
        model.model.config.id2label = {0: "genuine", 1: "tts"}
        model.model.config.label2id = {"genuine": 0, "tts": 1}
        _require(model.loading["ok"] is True and model.loading["shapes"]["ok"] is True, "AST loading audit did not pass")
        _require(all(getattr(model.feature_extractor, k, None) == v for k, v in recipe["preprocessor"].items()),
                 "loaded AST feature extractor differs from recipe")
        _require(all(p.dtype == torch.float32 for p in model.parameters()), "AST parameters must remain FP32 under FP16 autocast")
        model.set_stage("head")
        trainability = validate_head_stage(model)
        model_identity = {"repo_id": PRETRAINED["ast"]["repo_id"], "revision": snapshot.name,
                          "weights_sha256": file_digest(snapshot / PRETRAINED["ast"]["weights_file"]),
                          "processor_file_sha256": file_digest(processor_file), "processor": recipe["preprocessor"]}
        provenance = {**plan.provenance, "plan_sha256": plan.sha256, "recipe_sha256": digest(recipe),
                      "source_sha256": hashes, "environment": environment, "model": model_identity,
                      "checkpoint_policy": recipe["checkpoint_policy"]}
        baseline_path = run_dir / "baseline.json"
        if resume:
            baseline = _read(baseline_path)
            _require(baseline.get("task") == TASK and baseline.get("provenance") == provenance,
                     "warm-up resume provenance differs from the original baseline")
            # Reconstructing the pinned seed must reproduce the original reference.
            _require(model.digests() == baseline["initial_group_sha256"]
                     and frozen_state_digest(model) == baseline["initial_frozen_state_sha256"], "initial AST reference cannot be reproduced")
            original_fixture = baseline["tests"]
            _require(file_digest(_inside(root, original_fixture["file"])) == original_fixture["sha256"], "original fixture evidence changed")
        else:
            baseline = {"schema_version": 1, "task": TASK, "provenance": provenance, "tests": fixture_reference,
                        "recipe": recipe, "initial_group_sha256": model.digests(),
                        "initial_frozen_state_sha256": frozen_state_digest(model), "trainability": trainability,
                        "loading_audit": model.loading}
            write_json(baseline_path, baseline)
        loader = CheckedWaveformLoader(shard_dir, plan.train_rows + plan.selection_rows)
        train_ds = CropDataset([(r["recording_id"], r["label"]) for r in plan.train_rows], loader, seed=cfg.seed)
        val_ds = ChunkDataset(plan.selection_chunks, {r["recording_id"]: r["label"] for r in plan.selection_rows}, loader)
        trainer = Trainer(model, cfg, train_ds, val_ds, run_dir,
                          param_groups=model.param_groups(head_lr=cfg.lr, backbone_lr=cfg.lr, weight_decay=cfg.weight_decay),
                          ledger_path=root / "reports/ledger.csv", provenance=provenance)
        if resume:
            validate_resume_state(run_dir, trainer.config_hash, cfg.max_epochs)
        observations = {"training_forwards": 0, "evaluation_forwards": 0, "head_gradient_calls": 0,
                        "head_nonzero_gradient_calls": 0}

        receipt_identity = {"baseline_sha256": file_digest(baseline_path), "provenance_sha256": digest(provenance)}
        validated_receipts = set()

        def retain_observation(name, facts):
            if name in validated_receipts:
                return
            path = run_dir / name
            if not path.exists():
                write_json(path, {"schema_version": 1, "task": TASK, **receipt_identity, **facts})
            receipt = _read(path)
            _require(receipt.get("schema_version") == 1 and receipt.get("task") == TASK
                     and all(receipt.get(k) == v for k, v in receipt_identity.items())
                     and all(receipt.get(k) == v for k, v in facts.items()), "persisted training proof differs from original baseline")
            validated_receipts.add(name)

        def guard(module, _inputs):
            validate_head_stage(module)
            backbone = getattr(module.model, BACKBONE)
            _require(not any(m.training for m in backbone.modules()), "frozen AST backbone reentered training mode")
            _require(all(p.grad is None for p in backbone.parameters()), "frozen backbone received gradients")
            key = "training_forwards" if module.training else "evaluation_forwards"
            observations[key] += 1
            if module.training:
                retain_observation("training_mode_receipt.json", {"backbone_all_modules_eval": True,
                                   "trainable_names": trainability["trainable_names"]})

        def observe_gradient(value):
            observations["head_gradient_calls"] += 1
            if torch.isfinite(value).all() and torch.count_nonzero(value).item() > 0:
                observations["head_nonzero_gradient_calls"] += 1
                retain_observation("head_gradient_receipt.json", {"finite_nonzero_scaled_classifier_gradient_observed": True})
            return value

        hooks.append(model.register_forward_pre_hook(guard))
        hooks.extend(p.register_hook(observe_gradient) for p in model.parameters() if p.requires_grad)
        summary = trainer.fit(resume=resume)
        latest, best, history = validate_resume_state(run_dir, trainer.config_hash, cfg.max_epochs)
        _require(latest["finished"] is True and latest["epoch"] == cfg.max_epochs and summary["epochs_run"] == cfg.max_epochs,
                 "warm-up did not finish the declared epoch count")
        _require(latest["config_hash"] == trainer.config_hash == best["config_hash"], "checkpoint configuration mismatch")
        recovery = []
        if recover_status(run_dir, summary):
            recovery.append("status.json restored from finished checkpoint summary")
        model.load_state_dict(latest["model"], strict=True)
        latest_proof = _state_proof(model, baseline)
        model.load_state_dict(best["model"], strict=True)
        best_proof = _state_proof(model, baseline)
        metrics, chunk_rows, recordings = trainer.evaluate()
        _require(all(math.isclose(metrics[k], v, abs_tol=ATOL, rel_tol=RTOL) for k, v in best["metrics"].items()),
                 "reloaded selection metrics differ from selected checkpoint")
        parity = _prediction_parity(run_dir, plan, chunk_rows, recordings, best["epoch"])
        # A finalization-only resume must still check every training PCM identity.
        for rid in sorted(set(loader.rows) - loader.verified_ids):
            loader(rid)
        _require(loader.verified_ids == set(loader.rows), "audio verification coverage is incomplete")
        _require([r["epoch"] for r in history] == list(range(1, cfg.max_epochs + 1)), "history does not cover each declared epoch exactly once")
        _require(all(math.isfinite(row[key]) for row in history for key in ("train_loss", "grad_norm", "lr")),
                 "non-finite training metrics; do not publish this warm-up")
        _require(all(math.isfinite(row["val"][key]) for row in history for key in ("loss", "macro_f1", "eer", "fpr")),
                 "non-finite selection validation metrics")
        if observations["training_forwards"]:
            _require(observations["head_nonzero_gradient_calls"] > 0, "head gradients were never nonzero")
        for name in ("training_mode_receipt.json", "head_gradient_receipt.json"):
            _require((run_dir / name).is_file(), "missing persisted actual training-mode/gradient proof")
        persisted_training_proof = validate_training_receipts(run_dir, baseline)
        _require(source_hashes(root) == hashes, "source/config changed during warm-up")
        preflight(root, recipe)
        ledger = _ledger_row(root / "reports/ledger.csv", run_dir.name, cfg, trainer.config_hash, latest, best, plan.split_sha256)
        names = ("baseline.json", "best.pt", "latest.pt", "predictions_chunks.parquet", "predictions_recordings.parquet",
                 "resolved_config.json", "history.jsonl", "status.json", "training_mode_receipt.json", "head_gradient_receipt.json")
        files = {name: {"sha256": file_digest(run_dir / name), "bytes": (run_dir / name).stat().st_size} for name in names}
        report = {"schema_version": 1, "task": TASK, "verified": True, "status": "colab_verified_pending_muneeb_review",
                  "run_directory": run_dir.relative_to(root).as_posix(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "provenance": provenance, "counts": plan.counts, "trainability": trainability, "summary": summary,
                  "tests": baseline["tests"], "latest_freeze_proof": latest_proof, "best_freeze_proof": best_proof,
                  "reload_parity": parity, "selection_metrics": metrics, "ledger": ledger, "files": files,
                  "observations_this_session": observations, "skipped_fp16_steps": latest["skipped_steps"],
                  "persisted_training_proof": persisted_training_proof,
                  "finalization_recovery": recovery,
                  "audio_integrity": {"verified_recordings": len(loader.verified_ids), "recording_ids_sha256": digest(sorted(loader.verified_ids))},
                  "seconds_this_session": time.perf_counter() - started,
                  "frozen_comparison_checkpoint": "best.pt", "R3_U2_initial_checkpoint": "best.pt",
                  "resume_checkpoint": "latest.pt", "label_map": {"0": "genuine", "1": "tts"},
                  "independent_reviewer": None,
                  "partial_adaptation": "not performed; R3-U2 must rebuild optimizer after unfreezing"}
        write_json(run_dir / MANIFEST_NAME, report)
        return report
    except BaseException as exc:
        failure = {"schema_version": 1, "task": TASK, "verified": False, "status": "failed_or_interrupted",
                   "run_directory": run_dir.relative_to(root).as_posix(), "exception": type(exc).__name__, "message": str(exc),
                   "source_sha256": hashes, "recipe_sha256": digest(recipe), "seconds_this_session": time.perf_counter() - started,
                   "generated_checkpoint_is_not_accepted": True}
        report_dir = root / "reports/r2"
        report_dir.mkdir(parents=True, exist_ok=True)
        write_json(report_dir / ("r2_u2_failure_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8] + ".json"), failure)
        raise
    finally:
        for handle in hooks:
            handle.remove()
        if loader is not None:
            loader.close()
        lock.unlink(missing_ok=True)


def verify_run(root, run_dir):
    """Independent artifact/evidence integrity check; never load audio or models.

    The original GPU training and reload parity remain assertions in the committed
    run evidence. This routine rechecks bytes/provenance, not research performance.
    """
    root = Path(root).resolve()
    run_dir = _inside(root, run_dir)
    report = _read(run_dir / MANIFEST_NAME)
    _require(report.get("schema_version") == 1 and report.get("task") == TASK and report.get("verified") is True,
             "no accepted R2-U2 run manifest")
    _require(report.get("status") == "colab_verified_pending_muneeb_review", "invalid warm-up status")
    _require(report.get("run_directory") == run_dir.relative_to(root).as_posix(), "manifest belongs to a different run directory")
    recipe = load_recipe(root)
    plan = preflight(root, recipe)
    provenance = report["provenance"]
    _require(provenance["recipe_sha256"] == digest(recipe) and provenance["plan_sha256"] == plan.sha256
             and provenance["source_sha256"] == source_hashes(root) and report["counts"] == plan.counts,
             "current recipe/input/source differs from warm-up provenance")
    required_files = {"baseline.json", "best.pt", "latest.pt", "predictions_chunks.parquet", "predictions_recordings.parquet",
                      "resolved_config.json", "history.jsonl", "status.json", "training_mode_receipt.json", "head_gradient_receipt.json"}
    _require(set(report["files"]) == required_files, "missing or unknown manifest deliverables")
    for name, info in report["files"].items():
        path = run_dir / name
        _require(Path(name).name == name and path.is_file() and path.stat().st_size == info["bytes"]
                 and file_digest(path) == info["sha256"], f"missing or corrupt warm-up artifact: {name}")
    baseline = _read(run_dir / "baseline.json")
    _require(baseline["provenance"] == provenance and baseline["recipe"] == recipe and baseline["tests"] == report["tests"],
             "baseline differs from manifest provenance")
    validate_training_receipts(run_dir, baseline)
    _require(report.get("label_map") == {"0": "genuine", "1": "tts"}, "warm-up class mapping differs")
    reference = report["tests"]
    _require(file_digest(_inside(root, reference["file"])) == reference["sha256"], "fixture report checksum differs")
    _test_gate(root, reference["file"], source_hashes(root))
    _require(report["ledger"]["file"] == "reports/ledger.csv", "unexpected ledger reference")
    with _inside(root, report["ledger"]["file"]).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["run_id"] == report["ledger"]["run_id"] and row["status"] == "completed"]
    _require(len(matches) == 1 and matches[0] == report["ledger"]["row"]
             and digest(matches[0]) == report["ledger"]["row_sha256"], "ledger completion row differs")
    _require(report["best_freeze_proof"]["passed"] is True and report["latest_freeze_proof"]["passed"] is True
             and report["reload_parity"]["passed"] is True and report["reload_parity"]["scope"] == "selection validation only",
             "warm-up publication gates are missing")
    return {"task": TASK, "integrity_passed": True, "verified": False, "verified_files": len(report["files"]),
            "manifest_sha256": file_digest(run_dir / MANIFEST_NAME), "counts": report["counts"],
            "status": "artifact_integrity_checked", "review_status": report["status"],
            "note": "integrity only; original Colab evidence contains the GPU, freeze and reload checks"}
