"""R3-U1: head, data builder, trainer, selection freeze and export on synthetic embeddings (no model weights, no audio)."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from echoproof.models.wavlm_head import POOLING_DIM, WavLMHead
from echoproof.training import r3_u1_data as D
from echoproof.training import r3_u1_train as T

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs" / "r3_u1_wavlm_head_b.json"


def recipe(epochs=6):
    """The shipped recipe, with a small batch and a higher learning rate so the tiny synthetic training set takes enough steps to learn."""
    r = copy.deepcopy(T.load_recipe(RECIPE))
    r["train"].update(max_epochs=epochs, batch_size=16, lr=0.02)
    return r


def synthetic_rows(seed=0, dim=768):
    """Separable embeddings: class 1 shifted along the first axes. Speakers are disjoint across train and the three roles."""
    rng = np.random.default_rng(seed)
    rows, vectors = [], []

    def add(kind, partition, role, recording, group, component, label, n, item_prefix):
        for k in range(n):
            vec = rng.normal(0, 1, dim)
            vec[:8] += 1.5 if label else -1.5
            vectors.append(vec)
            rows.append({"item_id": f"{item_prefix}:B:{recording}:{k}", "recording_id": recording, "group_id": group, "component": component,
                         "label": label, "scope": "development", "partition": partition, "role": role, "kind": kind})

    comps = [("genuine", 0), ("melo", 1), ("vits", 1)]
    for s in range(12):                                             # training: 12 speakers, 1..3 crops per recording
        for j, (comp, label) in enumerate(comps):
            add("crop", "train", None, f"phonemedf:{comp}:{100 + s}-1-{j}", f"speaker:{100 + s}", comp, label, 1 + (s + j) % 3, "crop")
    for r_i, role in enumerate(D.ROLES):
        for s in range(4):
            for comp, label in comps + [("xtts", 1)]:
                add("chunk", "validation", role, f"phonemedf:{comp}:{200 + 10 * r_i + s}-1-0", f"speaker:{200 + 10 * r_i + s}", comp, label, 2, "chunk")
    return rows, np.array(vectors, dtype=np.float32)


def synthetic_data(pooling="mean", seed=0):
    dim = POOLING_DIM[pooling]
    rows, matrix = synthetic_rows(seed, dim)
    return D.build_data(rows, matrix, pooling, {"source": "synthetic"})


# ---- head ----------------------------------------------------------------------------------------------------------------------------------

def test_head_shapes_scores_and_validation():
    for pooling, dim in POOLING_DIM.items():
        head = WavLMHead(dim)
        x = torch.randn(5, dim)
        assert head(x).shape == (5, 2)
        p = head.scores(x)
        assert p.shape == (5,) and ((p >= 0) & (p <= 1)).all()
        assert head.architecture() == {"in_dim": dim, "hidden": 128, "dropout": 0.2}
    with pytest.raises(ValueError):
        WavLMHead(768)(torch.randn(2, 1536))
    with pytest.raises(ValueError):
        WavLMHead(768, dropout=1.0)
    with pytest.raises(ValueError):
        WavLMHead(0)


def test_scores_use_eval_semantics_without_changing_the_mode():
    head = WavLMHead(768, dropout=0.5).train()
    x = torch.randn(64, 768)
    assert torch.equal(head.scores(x), head.scores(x)) and head.training


def test_layer_layout_matches_the_plan():
    head = WavLMHead(768)
    kinds = [type(m).__name__ for m in head.net]
    assert kinds == ["Linear", "GELU", "Dropout", "Linear"]
    assert (head.net[0].in_features, head.net[0].out_features, head.net[3].out_features) == (768, 128, 2)


# ---- recipe ---------------------------------------------------------------------------------------------------------------------------------

def test_shipped_recipe_loads_and_matches_the_plan():
    r = T.load_recipe(RECIPE)
    assert (r["train"]["optimizer"], r["train"]["lr"], r["train"]["max_epochs"]) == ("AdamW", 0.001, 20)
    assert r["variants"] == ["mean", "mean_std"] and r["status"] == "team_agreed_user_reported_2026-11-11"


@pytest.mark.parametrize("path,value", [(("train", "lr"), 0.01), (("train", "max_epochs"), 10), (("train", "optimizer"), "SGD"),
                                        (("train", "early_stopping"), True), (("variants",), ["mean"]), (("task",), "R3-U2")])
def test_recipe_departing_from_the_plan_is_rejected(tmp_path, path, value):
    r = json.loads(RECIPE.read_text(encoding="utf-8"))
    target = r
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    p = tmp_path / "r.json"
    p.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(T.R3U1Error):
        T.load_recipe(p)


# ---- data builder -----------------------------------------------------------------------------------------------------------------------------

def test_build_data_shapes_roles_and_equal_recording_weights():
    data = synthetic_data()
    assert set(data.roles) == set(D.ROLES) and data.dim == 768 and len(data.train) == len(data.train.X)
    w = data.train.recording_weights
    assert abs(float(w.mean()) - 1.0) < 1e-6
    per_recording = {}
    for rid, weight in zip(data.train.recording_ids, w.tolist()):
        per_recording[rid] = per_recording.get(rid, 0.0) + weight
    assert max(per_recording.values()) - min(per_recording.values()) < 1e-5            # every recording carries the same total weight


@pytest.mark.parametrize("mutate,message", [
    (lambda rows, m: rows[0].update(partition="test"), "non-development scope or the test partition"),
    (lambda rows, m: rows[0].update(scope="test"), "non-development scope or the test partition"),
    (lambda rows, m: rows[0].update(component="chatterbox"), "Chatterbox"),
    (lambda rows, m: rows[1].update(item_id=rows[0]["item_id"]), "duplicate item ids"),
    (lambda rows, m: rows[0].update(label=2), "labels must be 0 or 1"),
    (lambda rows, m: rows[0].update(kind="chunk"), "training rows must be crops"),
])
def test_build_data_rejections(mutate, message):
    rows, matrix = synthetic_rows()
    mutate(rows, matrix)
    with pytest.raises(D.AlignmentError, match=message):
        D.build_data(rows, matrix, "mean", {})


def test_build_data_rejects_non_finite_vectors_a_missing_class_and_speaker_overlap():
    rows, matrix = synthetic_rows()
    bad = matrix.copy()
    bad[3, 3] = np.nan
    with pytest.raises(D.AlignmentError, match="non-finite"):
        D.build_data(rows, bad, "mean", {})
    one_class = [dict(r, label=0) if r["role"] == "operating" else r for r in rows]
    with pytest.raises(D.AlignmentError, match="both classes"):
        D.build_data(one_class, matrix, "mean", {})
    shared = [dict(r, group_id="speaker:100") if r["role"] == "selection" and r["component"] == "genuine" else r for r in rows]
    with pytest.raises(D.AlignmentError, match="speaker appears in two"):
        D.build_data(shared, matrix, "mean", {})
    with pytest.raises(D.AlignmentError):
        D.build_data(rows, matrix, "median", {})


# ---- training -----------------------------------------------------------------------------------------------------------------------------------

def run(tmp_path, name="run", pooling="mean", epochs=6, ledger=None, rec=None):
    data = synthetic_data(pooling)
    return data, T.train_variant(data, rec or recipe(epochs), tmp_path / name, owner="tester", split_sha256="0" * 64, run_id=f"r3_u1_{pooling}",
                                 ledger_path=ledger)


def test_training_writes_artifacts_and_improves(tmp_path):
    ledger = tmp_path / "ledger.csv"
    data, summary = run(tmp_path, ledger=ledger)
    d = tmp_path / "run"
    assert {"best.pt", "history.jsonl", "resolved_config.json", "status.json"} <= {p.name for p in d.iterdir()}
    history = [json.loads(line) for line in (d / "history.jsonl").read_text().splitlines()]
    assert [h["epoch"] for h in history] == [1, 2, 3, 4, 5, 6]
    assert history[0]["improved"] is True and "informational" in history[0]["panels_note"]
    assert summary["best_macro_f1"] >= 0.9                                       # the synthetic classes are separable
    row = list(csv.DictReader(open(ledger, newline="")))[0]
    assert row["model"] == "wavlm_head_mean" and row["protocol"] == "B" and row["status"] == "completed" and row["owner"] == "tester"
    ckpt = torch.load(d / "best.pt", map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == summary["best_epoch"] and ckpt["pooling"] == "mean" and ckpt["config_hash"] == summary["config_hash"]


def test_training_is_deterministic(tmp_path):
    _, a = run(tmp_path, "a")
    _, b = run(tmp_path, "b")
    ha = (tmp_path / "a" / "history.jsonl").read_text()
    hb = (tmp_path / "b" / "history.jsonl").read_text()
    strip = lambda text: [{k: v for k, v in json.loads(line).items() if k != "seconds"} for line in text.splitlines()]
    assert strip(ha) == strip(hb)
    ca = torch.load(tmp_path / "a" / "best.pt", weights_only=True)["model"]
    cb = torch.load(tmp_path / "b" / "best.pt", weights_only=True)["model"]
    assert all(torch.equal(ca[k], cb[k]) for k in ca)


def test_a_different_seed_changes_the_run(tmp_path):
    other = recipe()
    other["train"]["seed"] = 7
    _, a = run(tmp_path, "a")
    _, b = run(tmp_path, "b", rec=other)
    assert a["config_hash"] != b["config_hash"]


def test_training_refuses_reuse_missing_owner_and_wrong_width(tmp_path):
    run(tmp_path)
    data = synthetic_data()
    with pytest.raises(T.R3U1Error, match="non-empty run directory"):
        T.train_variant(data, recipe(), tmp_path / "run", owner="x", split_sha256="0" * 64)
    with pytest.raises(T.R3U1Error, match="owner"):
        T.train_variant(data, recipe(), tmp_path / "fresh", owner="", split_sha256="0" * 64)
    wide = synthetic_data("mean_std")
    wide.pooling = "mean"
    with pytest.raises(T.R3U1Error, match="width"):
        T.train_variant(wide, recipe(), tmp_path / "fresh2", owner="x", split_sha256="0" * 64)


def test_epoch_ties_keep_the_earliest_epoch(tmp_path, monkeypatch):
    real = T.evaluate_selection

    def constant(model, data, *, device="cpu"):
        metrics, recordings = real(model, data, device=device)
        return dict(metrics, macro_f1=0.8), recordings

    monkeypatch.setattr(T, "evaluate_selection", constant)
    _, summary = run(tmp_path)
    assert summary["best_epoch"] == 1


def test_non_finite_loss_stops_the_run(tmp_path):
    data = synthetic_data()
    data.train.X[0, 0] = np.inf
    with pytest.raises(T.R3U1Error, match="non-finite"):
        T.train_variant(data, recipe(), tmp_path / "bad", owner="x", split_sha256="0" * 64)


# ---- selection, freeze and export ---------------------------------------------------------------------------------------------------------------

def test_choose_winner_rules():
    assert T.choose_winner({"mean": 0.80, "mean_std": 0.81}) == "mean_std"
    assert T.choose_winner({"mean": 0.81, "mean_std": 0.80}) == "mean"
    assert T.choose_winner({"mean": 0.8, "mean_std": 0.8}) == "mean"
    for bad in ({"mean": 0.8}, {"mean": 0.8, "mean_std": float("nan")}, {"mean": 0.8, "mean_std": 1.2}):
        with pytest.raises(T.R3U1Error):
            T.choose_winner(bad)


@pytest.fixture()
def finished(tmp_path):
    summaries, datas = {}, {}
    for pooling in D.POOLINGS:
        datas[pooling], summaries[pooling] = run(tmp_path, pooling, pooling)
    evidence = T.finish(tmp_path, datas, summaries, recipe(), split_sha256="0" * 64)
    return tmp_path, datas, summaries, evidence


def test_finish_freezes_selection_and_exports_all_roles_in_the_r3_n1_layout(finished):
    out, datas, summaries, evidence = finished
    selection = json.loads((out / "selection.json").read_text())
    assert selection["selected"] in D.POOLINGS and selection["selection_threshold"] == 0.5 and selection["tie_rule"] == "mean pooling"
    assert selection["recipe_status"] == "team_agreed_user_reported_2026-11-11"
    import pyarrow.parquet as pq
    for role in D.ROLES:
        chunks = pq.read_table(out / f"predictions_chunks_{role}.parquet").to_pandas()
        recs = pq.read_table(out / f"predictions_recordings_{role}.parquet").to_pandas()
        assert list(chunks.columns) == ["chunk_id", "recording_id", "group_id", "label", "component", "validation_role", "tts_probability"]
        assert list(recs.columns) == ["recording_id", "group_id", "label", "component", "validation_role", "raw_score", "chunk_count"]
        assert (chunks.validation_role == role).all() and (recs.validation_role == role).all()
        assert len(chunks) == len(datas["mean"].roles[role]) and len(recs) == len(datas["mean"].roles[role].recordings())
        assert chunks.tts_probability.between(0, 1).all() and recs.raw_score.between(0, 1).all()
        assert recs.chunk_count.sum() == len(chunks)
    assert evidence["score_type"] == "raw uncalibrated TTS score" and set(evidence["counts"]) == set(D.ROLES)


def test_exported_recording_scores_are_the_mean_of_chunk_probabilities(finished):
    out, *_ = finished
    import pyarrow.parquet as pq
    chunks = pq.read_table(out / "predictions_chunks_selection.parquet").to_pandas()
    recs = pq.read_table(out / "predictions_recordings_selection.parquet").to_pandas().set_index("recording_id")
    means = chunks.groupby("recording_id").tts_probability.mean()
    assert np.allclose(means.loc[recs.index].values, recs.raw_score.values, atol=1e-12)


def test_selection_is_idempotent_and_refuses_a_changed_decision(finished, tmp_path):
    out, datas, summaries, _ = finished
    T.finish(out, datas, summaries, recipe(), split_sha256="0" * 64)               # same inputs: accepted
    other = recipe()
    other["decisions"]["batch_size"] = "changed after the fact"
    with pytest.raises(T.R3U1Error, match="refusing to overwrite"):
        T.finish(out, datas, summaries, other, split_sha256="0" * 64)


def test_finish_detects_a_changed_checkpoint_and_incomplete_inputs(finished):
    out, datas, summaries, _ = finished
    path = Path(summaries["mean"]["best_checkpoint"])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt["model"]["net.0.bias"] = ckpt["model"]["net.0.bias"] + 1.0
    torch.save(ckpt, path)
    with pytest.raises(T.R3U1Error, match="changed since training"):
        T.finish(out, datas, summaries, recipe(), split_sha256="0" * 64)
    with pytest.raises(T.R3U1Error, match="both"):
        T.finish(out, datas, {"mean": summaries["mean"]}, recipe(), split_sha256="0" * 64)


def test_scoring_before_selection_is_impossible_without_both_runs(tmp_path):
    data, summary = run(tmp_path, "only", "mean")
    with pytest.raises(T.R3U1Error):
        T.finish(tmp_path, {"mean": data}, {"mean": summary}, recipe(), split_sha256="0" * 64)
    assert not (tmp_path / "selection.json").exists() and not list(tmp_path.glob("predictions_*"))


def test_aggregation_requires_exact_recording_coverage():
    data = synthetic_data()
    split = data.roles["selection"]
    probs = np.full(len(split), 0.5)
    assert len(T.aggregate(split, probs).recording_ids) == len(split.recordings())
    from echoproof.contracts import ContractError
    from echoproof.evaluation.aggregate import aggregate_recordings
    # an expected recording with no chunks must be rejected, never silently dropped
    with pytest.raises(ContractError):
        aggregate_recordings(split.item_ids, split.recording_ids, split.labels.tolist(), probs.tolist(),
                             expected_recording_ids=set(split.recordings()) | {"phonemedf:genuine:missing"})
    # and a score outside [0, 1] must be rejected
    bad = probs.copy()
    bad[0] = 1.5
    with pytest.raises(ContractError):
        T.aggregate(split, bad)


def test_memo_roundtrip_and_invalidation(tmp_path, monkeypatch):
    """A verified memo is reused only while the manifest, index order and stored array hash still match."""
    rows, matrix = synthetic_rows()
    cache = tmp_path / "cache"
    cache.mkdir()
    index = {"rows": rows}
    (cache / "development_index.json").write_text(json.dumps(index), encoding="utf-8")
    from echoproof.artifacts import file_digest
    manifest = {"plan_sha256": "p" * 64, "identity": {"revision": "r", "weights_sha256": "w", "preprocessing_hash": "h"},
                "indices": {"development": {"file": "development_index.json", "sha256": file_digest(cache / "development_index.json")}}}
    (cache / D.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    from echoproof.features import r2_u1
    monkeypatch.setattr(r2_u1, "read_embeddings", lambda *a, **k: (calls.append(1), (rows, matrix))[1])
    first = D.load_embeddings(cache, "mean", memo_dir=tmp_path / "memo")
    second = D.load_embeddings(cache, "mean", memo_dir=tmp_path / "memo")
    assert len(calls) == 1 and second.identity["source"] == "verified memo" and np.array_equal(first.train.X, second.train.X)
    npz = next((tmp_path / "memo").glob("*.npz"))
    npz.write_bytes(npz.read_bytes() + b"x")                                         # tampered memo: must be rebuilt, not trusted
    D.load_embeddings(cache, "mean", memo_dir=tmp_path / "memo")
    assert len(calls) == 2
