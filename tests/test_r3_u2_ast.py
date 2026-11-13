"""R3-U2 fixture tests: recipe, warm-up loading, partial stage, gradient proof, a tiny end-to-end run, selection freeze, export and paired comparison.

They use a tiny random AST (see tests/test_r1_u2_ast.py) and synthetic audio, run on CPU in fp32, and need torch, transformers and pyarrow.
They are meant to run on Colab; where a dependency is missing they skip, and the Colab notebook refuses to continue past a skip.
"""
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pyarrow")

from echoproof.artifacts import file_digest
from echoproof.models.ast_detector import ASTDetector
from echoproof.training import r3_u2_data as D
from echoproof.training import r3_u2_train as T
from echoproof.training.datasets import ChunkDataset, CropDataset, DictWaveformLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r1_u2_ast import tiny_feature_extractor, tiny_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
FAST = {"device": "cpu", "precision": "fp32", "deterministic": False, "max_epochs": 2, "patience": 2, "warmup_steps": 1, "batch_size": 4,
        "accumulation_steps": 2, "eval_batch_size": 8, "num_workers": 0}


def shipped():
    return T.load_recipe(ROOT / "configs/r3_u2_ast_ft_b.json")


# ---- recipe -----------------------------------------------------------------------------------------------------------------------------------

def test_shipped_recipe_matches_the_plan():
    r = shipped()
    t = r["train"]
    assert t["batch_size"] * t["accumulation_steps"] == 16 and t["scheduler"] == "cosine" and t["weight_decay"] == 0.01
    assert r["experiments"]["baseline"] == {"backbone_lr": 1e-5, "head_lr": 1e-4}
    assert r["model"]["unfrozen_blocks"] == 2 and 6 <= t["max_epochs"] <= 10 and t["patience"] >= t["max_epochs"]
    assert r["init"]["checkpoint_sha256"] == "6a3883e59fc48ec18697fcdfdb59641adbd7dee6e965c1f80ad6f5287ea0bd57"
    assert len(r["init"]["pretrained_group_sha256"]) == 14 and "classifier" not in r["init"]["pretrained_group_sha256"]
    steps = math.ceil(math.ceil(4530 / t["batch_size"]) / t["accumulation_steps"]) * t["max_epochs"]
    assert steps == 2272 and 0 < t["warmup_steps"] < steps // 10                 # the recorded step arithmetic


@pytest.mark.parametrize("edit,message", [
    (lambda r: r["train"].update(batch_size=4), "effective batch"),
    (lambda r: r["train"].update(max_epochs=12), "epochs must be"),
    (lambda r: r["train"].update(patience=3), "epochs must be"),
    (lambda r: r["train"].update(warmup_steps=0), "epochs must be"),
    (lambda r: r["train"].update(scheduler="constant"), "departs from the plan"),
    (lambda r: r["train"].update(weight_decay=0.1), "departs from the plan"),
    (lambda r: r["model"].update(unfrozen_blocks=4), "last two blocks"),
    (lambda r: r["experiments"]["baseline"].update(backbone_lr=1e-4), "learning rates"),
    (lambda r: r["experiments"].pop("lower_backbone_lr"), "baseline and one alternative"),
    (lambda r: r["selection"].update(metric="recording AUC"), "fixed 0.5"),
    (lambda r: r.update(protocol="A"), "Protocol B"),
])
def test_recipe_departures_from_the_plan_are_rejected(tmp_path, edit, message):
    r = copy.deepcopy(shipped())
    edit(r)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(T.R3U2Error, match=message):
        T.load_recipe(path)


def test_train_config_follows_the_recipe_and_requires_an_owner():
    r = shipped()
    cfg = T.train_config(r, "lower_backbone_lr", owner="tester")
    assert (cfg.max_epochs, cfg.batch_size, cfg.accumulation_steps, cfg.scheduler, cfg.warmup_steps, cfg.precision) == (8, 8, 2, "cosine", 114, "fp16")
    assert cfg.lr == 1e-4 and cfg.owner == "tester" and cfg.model_name == "ast_partial" and cfg.protocol == "B"
    with pytest.raises(T.R3U2Error):
        T.train_config(r, "baseline", owner="")


def test_winner_rule_and_ties():
    assert T.choose_winner({"baseline": .8, "lower_backbone_lr": .8}) == "baseline"
    assert T.choose_winner({"baseline": .8, "lower_backbone_lr": .81}) == "lower_backbone_lr"
    with pytest.raises(T.R3U2Error):
        T.choose_winner({"baseline": .8})
    with pytest.raises(T.R3U2Error):
        T.choose_winner({"baseline": float("nan"), "lower_backbone_lr": .8})


def test_input_verification_rejects_a_changed_file(tmp_path):
    for name, content in (("cohort", b"a"), ("split", b"b"), ("chunks", b"c")):
        (tmp_path / name).write_bytes(content)
    paths = {k: tmp_path / k for k in ("cohort", "split", "chunks")}
    sha = {k: file_digest(v) for k, v in paths.items()}
    manifest = {"protocol": "B", "verification": {"passed": True},
                "inputs": {"split_sha256": "f" * 64, "cohort_manifest_file_sha256": sha["cohort"], "split_file_sha256": sha["split"],
                           "chunk_manifest_file_sha256": sha["chunks"]}}
    assert D.verify_inputs(paths, manifest, "f" * 64) == sha
    with pytest.raises(D.R3U2DataError, match="frozen split"):
        D.verify_inputs(paths, manifest, "e" * 64)
    (tmp_path / "chunks").write_bytes(b"changed")
    with pytest.raises(D.R3U2DataError, match="hash mismatch"):
        D.verify_inputs(paths, manifest, "f" * 64)
    with pytest.raises(D.R3U2DataError, match="not verified"):
        D.verify_inputs(paths, {**manifest, "verification": {"passed": False}}, "f" * 64)


# ---- tiny AST fixtures ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("r3u2")
    snapshot = root / "snapshot"
    tiny_snapshot(snapshot)
    extractor = tiny_feature_extractor()
    try:
        extractor([np.zeros(RATE, dtype=np.float32)], sampling_rate=RATE, return_tensors="pt")
    except Exception:
        pytest.skip("AST feature extractor backend unavailable")
    fresh = ASTDetector.from_dir(snapshot, seed=1, feature_extractor=extractor)
    pins = {g: h for g, h in fresh.digests().items() if g != "classifier"}
    warm = ASTDetector.from_dir(snapshot, seed=7, feature_extractor=extractor)           # a different head stands in for the warm-up head
    ckpt = root / "best.pt"
    torch.save({"epoch": 2, "model": warm.state_dict(), "metrics": {"macro_f1": .5}, "config": {}, "config_hash": "c" * 64,
                "optimizer": {"must": "be ignored"}, "scaler": {"x": 1}}, ckpt)
    recipe = copy.deepcopy(shipped())
    recipe["init"].update(checkpoint_sha256=file_digest(ckpt), warmup_selection_macro_f1=None)
    return SimpleNamespace(root=root, snapshot=snapshot, extractor=extractor, pins=pins, ckpt=ckpt, recipe=recipe)


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    waves, train_items = {}, []
    for i in range(24):
        rid = f"phonemedf:train:{i}"                         # the aggregator requires namespaced recording ids
        waves[rid] = rng.normal(0, .1, 12000 + 500 * i).astype(np.float32)
        train_items.append((rid, i % 2))
    comps = {"selection": ["genuine"] * 4 + ["melo", "vits"] * 1 + ["xtts"] * 4, "calibration": ["genuine"] * 3 + ["vits"] * 3,
             "operating": ["genuine"] * 3 + ["melo"] * 3}
    chunks, labels, components, groups, k = {}, {}, {}, {}, 0
    for role, names in comps.items():
        role_chunks = []
        for j, comp in enumerate(names):
            rid = f"phonemedf:{role}:{j}"
            waves[rid] = rng.normal(0, .1, 16000 + 400 * j).astype(np.float32)
            labels[rid], components[rid], groups[rid] = int(comp != "genuine"), comp, f"speaker:{k}"
            k += 1
            n = len(waves[rid])
            for s, e in ((0, 12000), (n - 12000, n)):
                role_chunks.append(SimpleNamespace(chunk_id=f"chunk:B:{rid}:{s}", recording_id=rid, start_sample=s, end_sample=e,
                                                   valid_samples=e - s, input_samples=12000))
        chunks[role] = role_chunks
    loader = DictWaveformLoader(waves)
    train = CropDataset(train_items, loader, seed=3)
    validation = {}
    for role in chunks:
        validation[role] = ChunkDataset(chunks[role], labels, loader)
        validation[role].validation_role = role
    return {"train": train, "validation": validation, "validation_components": components, "validation_groups": groups,
            "split_sha256": "0" * 64, "input_sha256": {"x": "1" * 64}, "loader": loader}


# ---- warm-up loading and the partial stage ------------------------------------------------------------------------------------------------

def test_warmup_loads_weights_only_and_sets_the_partial_stage(world):
    detector, info = T.build_detector(world.recipe, world.ckpt, snapshot=world.snapshot, feature_extractor=world.extractor, pins=world.pins)
    warm = torch.load(world.ckpt, map_location="cpu", weights_only=True)["model"]
    assert all(torch.equal(detector.state_dict()[k], v) for k, v in warm.items())
    assert info["loaded"].startswith("model weights only") and info["epoch"] == 2
    assert info["stage"]["stage"] == "partial"
    trainable = {g for g, v in info["stage"]["groups"].items() if v["trainable"]}
    assert trainable == {"block.02", "block.03", "final_norm", "classifier"} == set(T.expected_groups(detector, 2))
    assert T.state_group_digests(detector, detector.state_dict()) == detector.digests() == info["group_sha256_after_load"]


def test_warmup_loading_refusals(world, tmp_path):
    kwargs = dict(snapshot=world.snapshot, feature_extractor=world.extractor, pins=world.pins)
    r = copy.deepcopy(world.recipe)
    r["init"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(T.R3U2Error, match="SHA-256"):
        T.build_detector(r, world.ckpt, **kwargs)
    r = copy.deepcopy(world.recipe)
    r["init"]["epoch"] = 3
    with pytest.raises(T.R3U2Error, match="epoch"):
        T.build_detector(r, world.ckpt, **kwargs)
    with pytest.raises(T.R3U2Error, match="pretrained groups"):
        T.build_detector(world.recipe, world.ckpt, snapshot=world.snapshot, feature_extractor=world.extractor,
                         pins={**world.pins, "block.00": "0" * 64})
    bad = torch.load(world.ckpt, map_location="cpu", weights_only=True)
    key = next(k for k in bad["model"] if k.endswith("classifier.dense.weight"))
    bad["model"][key] = bad["model"][key].clone()
    bad["model"][key][0, 0] = float("nan")
    path = tmp_path / "nan.pt"
    torch.save(bad, path)
    r = copy.deepcopy(world.recipe)
    r["init"]["checkpoint_sha256"] = file_digest(path)
    with pytest.raises(T.R3U2Error, match="non-finite"):
        T.build_detector(r, path, **kwargs)
    missing = torch.load(world.ckpt, map_location="cpu", weights_only=True)
    missing["model"].pop(next(iter(missing["model"])))
    path2 = tmp_path / "short.pt"
    torch.save(missing, path2)
    r["init"]["checkpoint_sha256"] = file_digest(path2)
    with pytest.raises(RuntimeError):
        T.build_detector(r, path2, **kwargs)                                         # strict load: no silent partial loads


def test_gradient_proof_passes_and_detects_a_leaky_freeze(world):
    data = make_data()
    detector, _ = T.build_detector(world.recipe, world.ckpt, snapshot=world.snapshot, feature_extractor=world.extractor, pins=world.pins)
    batch = T.proof_batch(data["train"])
    assert sorted(batch["labels"].tolist()) == [0] * 4 + [1] * 4
    exp = world.recipe["experiments"]["baseline"]
    start = detector.digests()
    proof = T.gradient_proof(detector, batch, exp, weight_decay=0.01, n_blocks=2, device="cpu")
    assert proof["ok"] and proof["groups_with_gradients"] == proof["groups_changed_by_one_step"] == proof["expected_updated_groups"]
    assert all(v > 0 for v in proof["gradient_norms"].values()) and len(proof["frozen_groups_bit_identical"]) == len(proof["frozen_groups"])
    assert detector.digests() == start                                              # the proof ran on a copy
    for name, p in detector.model.named_parameters():
        if detector._group(name) == "block.00":
            p.requires_grad_(True)                                                   # a leaky freeze must not pass
    with pytest.raises(T.R3U2Error, match="gradient proof failed"):
        T.gradient_proof(detector, batch, exp, weight_decay=0.01, n_blocks=2, device="cpu")
    with pytest.raises(T.R3U2Error, match="trainable flag"):
        T.check_partial_stage(detector, 2)


# ---- end to end -------------------------------------------------------------------------------------------------------------------------------

def run(world, output, name="baseline", data=None, recipe=None, **kw):
    return T.run_experiment(name, recipe or world.recipe, data or make_data(), checkpoint=world.ckpt, snapshot=world.snapshot,
                            feature_extractor=world.extractor, pins=world.pins, output=output, ledger=output.parent / "ledger.csv",
                            owner="tester", overrides=FAST, **kw)


def test_one_experiment_writes_the_adaptation_evidence_and_a_ledger_row(world, tmp_path):
    output = tmp_path / "out"
    summary = run(world, output)
    run_dir = output / "out-baseline"
    assert summary["epochs_run"] == 2 and summary["stop_reason"] == "max_epochs"
    for name in ("gradient_proof.json", "epoch0_metrics.json", "epoch0_predictions_chunks.parquet", "adaptation_evidence.json", "best.pt",
                 "latest.pt", "history.jsonl", "status.json", "resolved_config.json"):
        assert (run_dir / name).is_file(), name
    ev = json.loads((run_dir / "adaptation_evidence.json").read_text(encoding="utf-8"))
    assert ev["trainable_groups"] == ["block.02", "block.03", "classifier", "final_norm"]
    assert ev["frozen_groups_unchanged_final_and_best"] is True and ev["problems"] == []
    assert ev["unfrozen_groups_changed_final"] == ev["unfrozen_groups_changed_best"] == ev["trainable_groups"]
    assert all(ev["group_sha256_final_epoch"][g] == ev["group_sha256_start"][g] for g in ev["frozen_groups"])
    assert ev["optimizer_steps_per_epoch"] == 3 and ev["optimizer_steps_total"] == 6                # 24 crops, microbatch 4, accumulation 2
    assert len(ev["selection_macro_f1_by_epoch"]) == 2 and ev["initial_checkpoint"]["epoch"] == 2
    assert json.loads((run_dir / "gradient_proof.json").read_text(encoding="utf-8"))["ok"] is True
    rows = (output.parent / "ledger.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and "ast_partial" in rows[1] and ",completed," in rows[1] and "tester" in rows[1]


def test_the_cosine_schedule_warms_up_then_decays(world, tmp_path):
    output = tmp_path / "sched"
    run(world, output)
    hist = [json.loads(line) for line in (output / "sched-baseline" / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    lrs = [h["lr"] for h in hist]
    # 3 optimiser steps per epoch, 1 warm-up step, 6 in total: cosine factor 0.5 * (1 + cos(pi * 2 / 5)) after epoch 1 and exactly 0 at the end
    assert lrs[0] == pytest.approx(1e-4 * 0.5 * (1 + math.cos(math.pi * 2 / 5)), rel=1e-6) and lrs[1] == 0


def test_a_run_directory_is_never_reused_and_the_loaded_warmup_must_reproduce_the_recorded_score(world, tmp_path):
    output = tmp_path / "out"
    run(world, output)
    with pytest.raises(T.R3U2Error, match="non-empty"):
        run(world, output)
    r = copy.deepcopy(world.recipe)
    r["init"].update(warmup_selection_macro_f1=0.123456, warmup_epoch0_tolerance=1e-6)
    with pytest.raises(T.R3U2Error, match="wrong weights or preprocessing"):
        run(world, tmp_path / "other", recipe=r)
    check = json.loads((tmp_path / "other" / "other-baseline" / "epoch0_metrics.json").read_text(encoding="utf-8"))
    assert check["within_tolerance"] is False                                        # the evidence is written before the refusal


def test_a_foreign_directory_without_a_checkpoint_blocks_a_resume(world, tmp_path):
    output = tmp_path / "x"
    (output / "x-baseline").mkdir(parents=True)
    (output / "x-baseline" / "stray.txt").write_text("stray", encoding="utf-8")
    with pytest.raises(T.R3U2Error, match="recoverable checkpoint"):
        run(world, output, resume=True)


def two_runs(world, tmp_path):
    output = tmp_path / "pair"
    data = make_data()
    summaries = {name: run(world, output, name=name, data=data) for name in T.EXPERIMENTS}
    return output, data, summaries


def test_finish_freezes_the_selection_exports_every_role_and_compares_with_the_warmup(world, tmp_path):
    import pyarrow.parquet as pq
    output, data, summaries = two_runs(world, tmp_path)
    evidence = T.finish(output, data, summaries, world.recipe, snapshot=world.snapshot, feature_extractor=world.extractor, device="cpu",
                        overrides=FAST)
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert selection["selected"] in T.EXPERIMENTS and selection["tie_rule"] == "baseline" and selection["selection_threshold"] == 0.5
    assert selection["init_checkpoint_sha256"] == world.recipe["init"]["checkpoint_sha256"]
    for role in T.ROLES:
        chunks = pq.read_table(output / f"predictions_chunks_{role}.parquet")
        recs = pq.read_table(output / f"predictions_recordings_{role}.parquet")
        assert chunks.column_names == ["chunk_id", "recording_id", "group_id", "label", "component", "validation_role", "tts_probability"]
        assert recs.column_names == ["recording_id", "group_id", "label", "component", "validation_role", "raw_score", "chunk_count"]
        assert set(chunks.column("validation_role").to_pylist()) == {role} and recs.num_rows == evidence["counts"][role]["recordings"]
        scores = recs.column("raw_score").to_pylist()
        assert all(0 <= s <= 1 for s in scores)
        assert "chatterbox" not in set(recs.column("component").to_pylist())
    assert evidence["reload_parity"]["max_abs_probability_difference_vs_saved_best_epoch"] <= T.PARITY_ATOL
    paired = evidence["paired_warmup_vs_adapted"]
    assert paired["recordings"] == evidence["counts"]["selection"]["recordings"] == 10
    changes = paired["decision_changes_at_0.5"]
    assert sum(changes.values()) == paired["recordings"]
    assert set(paired["warmup"]) == set(paired["adapted"]) == {"all", "known", "unseen_xtts"}
    assert abs(paired["macro_f1_change"] - (paired["adapted"]["all"]["macro_f1"] - paired["warmup"]["all"]["macro_f1"])) < 1e-12
    assert evidence["selection_sha256"] == file_digest(output / "selection.json")


def test_the_selection_is_idempotent_and_cannot_be_silently_changed(world, tmp_path):
    output, data, summaries = two_runs(world, tmp_path)
    kwargs = dict(snapshot=world.snapshot, feature_extractor=world.extractor, device="cpu", overrides=FAST)
    first = T.finish(output, data, summaries, world.recipe, **kwargs)
    again = T.finish(output, data, summaries, world.recipe, **kwargs)
    assert first["selection"] == again["selection"]
    changed = copy.deepcopy(world.recipe)
    changed["status"] = "something_else"
    with pytest.raises(T.R3U2Error, match="refusing to overwrite"):
        T.finish(output, data, summaries, changed, **kwargs)


def test_finish_refuses_incomplete_or_tampered_runs(world, tmp_path):
    output, data, summaries = two_runs(world, tmp_path)
    kwargs = dict(snapshot=world.snapshot, feature_extractor=world.extractor, device="cpu", overrides=FAST)
    with pytest.raises(T.R3U2Error, match="both completed runs"):
        T.finish(output, data, {"baseline": summaries["baseline"]}, world.recipe, **kwargs)
    bad = copy.deepcopy(summaries)
    bad["baseline"]["best_macro_f1"] = 0.999
    with pytest.raises(T.R3U2Error, match="inconsistent"):
        T.finish(output, data, bad, world.recipe, **kwargs)
    unfinished = copy.deepcopy(summaries)
    unfinished["baseline"]["stop_reason"] = "early_stopping"
    with pytest.raises(T.R3U2Error, match="inconsistent"):
        T.finish(output, data, unfinished, world.recipe, **kwargs)
    assert not (output / "selection.json").exists()                                  # nothing was frozen from bad inputs
