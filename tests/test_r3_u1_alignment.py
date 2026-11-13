"""R3-U1 alignment check: cache rows must be exactly the Protocol B chunks and training recordings that the split, chunk manifest and cohort define."""
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from echoproof.training import r3_u1_data as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r2_s1_features import build_mini_project  # noqa: E402


@pytest.fixture(scope="module")
def mini():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        project = build_mini_project(root, speakers=24)
        yield root, project


def cache_rows(project):
    """Cache-shaped rows derived from the mini project's real manifests."""
    cohort = {r["recording_id"]: r for r in project["rows"]}
    assign = {a.recording_id: a for a in project["assignments"]}
    rows = []
    for c in project["chunks"]:
        a = assign[c.recording_id]
        if a.partition != "validation":
            continue
        r = cohort[c.recording_id]
        rows.append({"item_id": c.chunk_id, "recording_id": c.recording_id, "group_id": c.group_id, "component": r["component"],
                     "label": int(r["label"]), "scope": "development", "partition": "validation", "role": a.validation_role, "kind": "chunk"})
    for rid, a in assign.items():
        if a.partition == "train":
            r = cohort[rid]
            for k in range(2):
                rows.append({"item_id": f"crop:B:{rid}:{k}", "recording_id": rid, "group_id": a.group_id, "component": r["component"],
                             "label": int(r["label"]), "scope": "development", "partition": "train", "role": None, "kind": "crop"})
    return rows


def build(rows):
    rng = np.random.default_rng(0)
    return D.build_data(rows, rng.normal(size=(len(rows), 768)).astype(np.float32), "mean", {})


def test_matching_rows_pass_and_report_counts(mini):
    root, project = mini
    report = D.check_alignment(root, build(cache_rows(project)))
    assert report["passed"] is True and report["model_scores_used"] is False
    assert set(report["roles"]) == set(D.ROLES)
    assert report["train"]["crops_per_recording_min"] == report["train"]["crops_per_recording_max"] == 2
    assert report["validation_chunks_match_manifest"] == sum(r["chunks"] for r in report["roles"].values())


def mutated(project, edit):
    rows = copy.deepcopy(cache_rows(project))
    edit(rows)
    return rows


def first(rows, partition, role=None):
    return next(r for r in rows if r["partition"] == partition and (role is None or r["role"] == role))


def set_for(rows, key, value, **changes):
    """Change every row of one recording or speaker, so the rows stay internally consistent and only the cross-check can notice."""
    for r in rows:
        if r[key] == value:
            r.update(changes)


@pytest.mark.parametrize("name,edit,expected", [
    ("flipped validation label", lambda rows: set_for(rows, "recording_id", first(rows, "validation")["recording_id"],
                                                      label=1 - first(rows, "validation")["label"]), "label or component"),
    ("wrong validation component", lambda rows: set_for(rows, "recording_id", first(rows, "validation")["recording_id"], component="vits"
                                                        if first(rows, "validation")["component"] != "vits" else "melo"), "label or component"),
    ("wrong group", lambda rows: set_for(rows, "group_id", first(rows, "validation")["group_id"], group_id="speaker:999999"), "group id"),
    ("missing chunk", lambda rows: rows.pop(next(i for i, r in enumerate(rows) if r["partition"] == "validation"
                                                 and sum(1 for x in rows if x["recording_id"] == r["recording_id"]) > 1)), "coverage"),
    ("unknown chunk id", lambda rows: first(rows, "validation").update(item_id="chunk:B:phonemedf:genuine:0-0-0:0"), "unknown chunk|coverage"),
])
def test_validation_disagreements_are_caught_by_the_cross_check(mini, name, edit, expected):
    root, project = mini
    data = build(mutated(project, edit))                      # internally consistent: the builder accepts it
    with pytest.raises(D.AlignmentError, match=expected):
        D.check_alignment(root, data)


def test_a_speaker_moved_to_another_role_is_caught(mini):
    root, project = mini
    rows = mutated(project, lambda rows: None)
    mover = first(rows, "validation", "selection")["group_id"]
    set_for(rows, "group_id", mover, role="operating")        # the whole speaker moves, so no speaker is shared between roles
    data = build(rows)
    with pytest.raises(D.AlignmentError, match="validation role|differ from the split"):
        D.check_alignment(root, data)


def test_training_label_and_membership_disagreements_are_caught(mini):
    root, project = mini
    rid = first(cache_rows(project), "train")["recording_id"]
    flipped = mutated(project, lambda rows: set_for(rows, "recording_id", rid, label=1 - first(rows, "train")["label"]))
    with pytest.raises(D.AlignmentError, match="training label"):
        D.check_alignment(root, build(flipped))
    dropped = mutated(project, lambda rows: [rows.remove(r) for r in [x for x in rows if x["recording_id"] == rid]])
    with pytest.raises(D.AlignmentError, match="training recordings differ"):
        D.check_alignment(root, build(dropped))


def test_a_validation_recording_presented_as_training_is_caught(mini):
    root, project = mini
    def edit(rows):
        v = first(rows, "validation")
        # the whole speaker is relabelled as a training speaker, so the partitions stay speaker-disjoint
        set_for(rows, "group_id", v["group_id"], partition="train", role=None, kind="crop")
    with pytest.raises(D.AlignmentError, match="not assigned to train"):
        D.check_alignment(root, build(mutated(project, edit)))
