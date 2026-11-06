"""R2-S3 step 1 (needs audio, runs on Colab): the same per-recording measurements as R2-S2, for the Protocol B *selection* validation role.

The shortcut probe is trained on B train (measured in R2-S2) and scored on B selection validation, so the selection recordings need the same
columns. Scope: the 2,000 retained recordings whose validation role is `selection` (genuine, MeloTTS, VITS, XTTS; 25 speakers). Calibration and
operating recordings, test recordings and ChatterboxTTS are never listed or read. The measurement code is `eda.measure` (unchanged, same
definitions and version), so train and selection values are directly comparable.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from echoproof.artifacts import digest
from echoproof.data import splits
from echoproof.data.cohort import read_table
from echoproof.eda import measure
from echoproof.features.r2_s1 import R2S1Error, file_sha256, plan_b, rows_digest

TASK = "R2-S3"
ROLE = "selection"
MANIFEST_NAME = "r2_s3_measure_manifest.json"
OUTPUT_NAMES = {"stats": "recording_stats_B_selection_v1.parquet"}
SELECTION_COMPONENTS = ("genuine", "melo", "vits", "xtts")


class SelectionEda(measure.Eda):
    """Same job/part machinery as the train EDA, over the selection-role validation recordings."""

    def __init__(self, plan):
        self.plan, self.examples = plan, []
        self.items = sorted((v for v in plan.validation if v["role"] == ROLE), key=lambda v: v["recording_id"])
        self.train_groups = {t["group_id"] for t in plan.train}
        self.sha256 = digest({"plan": plan.sha256, "scope": f"validation:{ROLE}", "ids": [[i["recording_id"], i["pcm_sha256"]] for i in self.items]})


def plan_selection(cohort_rows, assignments, chunks, *, seed: int = 42) -> SelectionEda:
    plan = plan_b(cohort_rows, assignments, chunks, seed=seed)         # test dropped, Chatterbox rejected, groups and shards checked
    eda = SelectionEda(plan)
    if not eda.items:
        raise R2S1Error("no selection recordings in the plan")
    components = {i["component"] for i in eda.items}
    if components != set(SELECTION_COMPONENTS):
        raise R2S1Error(f"unexpected selection components {sorted(components)}")
    if {i["group_id"] for i in eda.items} & eda.train_groups:
        raise R2S1Error("a selection speaker also appears in train")
    return eda


def verify(eda: SelectionEda, frame) -> dict:
    problems = []
    expected = {}
    for i in eda.items:
        expected[i["component"]] = expected.get(i["component"], 0) + 1
    counts = {k: int(v) for k, v in frame["component"].value_counts().sort_index().items()}
    if len(frame) != len(eda.items) or counts != dict(sorted(expected.items())):
        problems.append(f"rows {len(frame)} / counts {counts} differ from the plan {expected}")
    if set(frame["recording_id"]) != {i["recording_id"] for i in eda.items}:
        problems.append("measured recordings differ from the plan")
    if not np.isfinite(frame[list(measure.SCALAR_COLUMNS)].to_numpy(float)).all():
        problems.append("non-finite measurements")
    if set(frame["group_id"]) & eda.train_groups:
        problems.append("a speaker appears in train and selection")
    by_id = {i["recording_id"]: i for i in eda.items}
    if any(by_id[r]["n_samples"] != n for r, n in zip(frame["recording_id"], frame["n_samples"])):
        problems.append("n_samples differs from the manifest")
    if set(frame["label"]) != {0, 1}:
        problems.append("a class is missing")
    return {"passed": not problems, "problems": problems, "rows": int(len(frame)), "counts": counts}


def build(root, shard_dir, out_dir=None, *, seed: int = 42, workers: int = 1, resume: bool = False, progress=None) -> dict:
    import platform
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data.chunks import PREPROCESSING_HASH
    say = progress or (lambda *_: None)
    t0 = time.perf_counter()
    root, shard_dir = Path(root), Path(shard_dir)
    out_dir = Path(out_dir) if out_dir else root / "data" / "eda" / "r2_s3_B_selection"
    parts_dir = out_dir / "parts"
    if (out_dir / MANIFEST_NAME).exists():
        raise R2S1Error(f"{out_dir / MANIFEST_NAME} exists: the selection measurement is already complete for this directory")
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
    eda = plan_selection(cohort_rows, assignments, chunks, seed=seed)
    say(f"plan {eda.sha256[:12]}: {len(eda.items)} selection recordings, {len(eda.shards())} shards")
    missing = [s for s in eda.shards() if not (shard_dir / s).is_file()]
    if missing:
        raise R2S1Error(f"shard files not found in {shard_dir}: {missing}")
    summaries, failures = measure.run_jobs(eda.jobs(shard_dir, parts_dir), workers=workers, resume=resume,
                                           progress=lambda shard, error: say(f"{shard}: {'FAILED ' + error if error else 'done'}"))
    if failures:
        raise R2S1Error(f"shards failed (rerun with resume=True): {failures}")
    frame, mel_mean = measure.merge_parts(eda, parts_dir)
    roles = {i["recording_id"]: i["role"] for i in eda.items}
    frame["role"] = frame["recording_id"].map(roles)
    verification = verify(eda, frame)
    if not verification["passed"]:
        raise R2S1Error(f"verification failed: {verification['problems']}")
    spot = measure.spot_check(eda, frame, mel_mean, shard_dir, seed=seed)
    if spot["mismatches"]:
        raise R2S1Error(f"spot check failed: {spot}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / OUTPUT_NAMES["stats"], index=False)
    outputs = {k: {"file": n, "sha256": file_sha256(out_dir / n), "bytes": (out_dir / n).stat().st_size} for k, n in OUTPUT_NAMES.items()}
    manifest = {"schema_version": 1, "task": TASK, "step": "measure", "protocol": "B", "partition": "validation", "role": ROLE,
                "plan_sha256": eda.plan.sha256, "scope_sha256": eda.sha256, "settings": measure.SETTINGS,
                "scope": "B validation, selection role only: genuine, melo, vits, xtts; no calibration, operating, test or Chatterbox audio is read",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "inputs": {"cohort_manifest_file_sha256": file_sha256(paths["cohort"]), "cohort_manifest_rows_digest": rows_digest(cohort_rows),
                           "split_sha256": split_hash, "split_sha256_frozen_in_report": frozen, "split_file_sha256": file_sha256(paths["split"])},
                "counts": verification["counts"], "verification": verification, "spot_check": spot, "shards": summaries, "workers": workers,
                "resumed": bool(resume), "outputs": outputs, "environment": {"python": platform.python_version(), "numpy": np.__version__},
                "seconds_total": round(time.perf_counter() - t0, 1)}
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
