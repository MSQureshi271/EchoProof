"""R4-U1b fixture tests: A recipes against their B counterparts, guarded A data, the A warm-up, the fine-tune proofs, selection freeze and export.

They use a tiny random AST (see tests/test_r1_u2_ast.py) and synthetic audio, run on CPU in fp32, and need torch, transformers and pyarrow.
No research audio, model download or GPU is used. The real AST-A runs happen on Colab.
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

from echoproof.artifacts import file_digest  # noqa: E402
from echoproof.models.ast_detector import ASTDetector  # noqa: E402
from echoproof.training import r4_u1_ast_data as D  # noqa: E402
from echoproof.training import r4_u1_ast_train as T  # noqa: E402
from echoproof.training.datasets import ChunkDataset, CropDataset, DictWaveformLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r1_u2_ast import tiny_feature_extractor, tiny_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def digest_in_lock(path, lock_text):
    """True if the file's SHA-256 is in the GR3 lock. The lock hashed the Windows (CRLF) checkout, so the same text with LF line endings (a Linux
    or Colab checkout) is accepted too: only the line endings may differ, never the content."""
    import hashlib
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return any(hashlib.sha256(variant).hexdigest() in lock_text for variant in (raw, lf, lf.replace(b"\n", b"\r\n")))

RATE = 16000
COMPONENTS = ("genuine", "melo", "vits", "xtts", "chatterbox")
FAST_WARM = {"device": "cpu", "precision": "fp32", "deterministic": False, "max_epochs": 2, "patience": 2, "batch_size": 4, "accumulation_steps": 2,
             "eval_batch_size": 8, "num_workers": 0}
FAST_FT = {"device": "cpu", "precision": "fp32", "deterministic": False, "max_epochs": 2, "patience": 2, "warmup_steps": 1, "batch_size": 4,
           "accumulation_steps": 2, "eval_batch_size": 8, "num_workers": 0}


def warm_recipe():
    return T.load_warmup_recipe(ROOT / "configs/r4_u1_ast_warmup_a.json", root=ROOT)


def ft_recipe():
    return T.load_finetune_recipe(ROOT / "configs/r4_u1_ast_ft_a.json", root=ROOT)


# ---- recipes -------------------------------------------------------------------------------------------------------------------------------

def test_shipped_recipes_equal_the_b_recipes_except_the_warmup_steps_and_carry_the_team_agreement():
    w, f = warm_recipe(), ft_recipe()
    b_w = json.loads((ROOT / T.B_WARMUP).read_text(encoding="utf-8"))
    b_f = json.loads((ROOT / T.B_FINETUNE).read_text(encoding="utf-8"))
    assert w["train"] == {k: v for k, v in b_w["train"].items() if k not in ("owner", "model_name", "protocol")}
    assert {k for k in b_f["train"] if b_f["train"][k] != f["train"][k]} == {"warmup_steps"}
    assert (b_f["train"]["warmup_steps"], f["train"]["warmup_steps"]) == (114, 189) == (T.warmup_steps_for(4530, b_f["train"]),
                                                                                       T.warmup_steps_for(7550, f["train"]))
    assert f["experiments"] == {"baseline": b_f["experiments"]["baseline"]} and f["model"] == b_f["model"]
    assert w["protocol"] == f["protocol"] == "A" and len(f["init"]["pretrained_group_sha256"]) == 14
    for recipe in (w, f):
        assert recipe["status"] == "team_agreed_user_reported_2026-11-15"


def test_training_refuses_a_recipe_without_the_recorded_team_agreement(tmp_path):
    root = tmp_path
    (root / "configs").mkdir()
    for rel in (T.B_WARMUP, T.B_FINETUNE):
        (root / rel).write_bytes((ROOT / rel).read_bytes())
    for recipe, name, loader in ((warm_recipe(), "w.json", T.load_warmup_recipe), (ft_recipe(), "f.json", T.load_finetune_recipe)):
        draft = copy.deepcopy(recipe)
        draft["status"] = "draft_pending_team_agreement"
        (root / "configs" / name).write_text(json.dumps(draft), encoding="utf-8")
        with pytest.raises(T.R4U1Error, match="reported agreement"):
            loader(root / "configs" / name, root=root, require_agreed=True)
        agreed = copy.deepcopy(recipe)
        agreed["status"] = "team_agreed_user_reported_2026-11-15"
        (root / "configs" / name).write_text(json.dumps(agreed), encoding="utf-8")
        assert loader(root / "configs" / name, root=root, require_agreed=True)["status"].startswith("team_agreed")


@pytest.mark.parametrize("edit,message", [
    (lambda r: r["train"].update(lr=1e-3), "departs from the B R2-U2"),
    (lambda r: r["train"].update(max_epochs=3), "departs from the B R2-U2"),
    (lambda r: r["model"].update(stage="partial"), "head only"),
    (lambda r: r.update(protocol="B"), "not an R4-U1b warm-up"),
    (lambda r: r["derived_from"].update(sha256="0" * 64), "differs from the one"),
])
def test_warmup_recipe_departures_are_rejected(tmp_path, edit, message):
    root = tmp_path
    (root / "configs").mkdir()
    (root / T.B_WARMUP).write_bytes((ROOT / T.B_WARMUP).read_bytes())
    r = copy.deepcopy(warm_recipe())
    edit(r)
    path = root / "configs/w.json"
    path.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(T.R4U1Error, match=message):
        T.load_warmup_recipe(path, root=root)


@pytest.mark.parametrize("edit,message", [
    (lambda r: r["train"].update(batch_size=4), "departs from the B recipe"),
    (lambda r: r["train"].update(max_epochs=6), "departs from the B recipe"),
    (lambda r: r["train"].update(warmup_steps=114), "5 percent of the A optimiser steps"),
    (lambda r: r["train"].update(scheduler="constant"), "departs from the B recipe"),
    (lambda r: r["model"].update(unfrozen_blocks=4), "model block departs"),
    (lambda r: r["experiments"]["baseline"].update(backbone_lr=1e-4), "baseline experiment only"),
    (lambda r: r["experiments"].update(lower_backbone_lr={"backbone_lr": 3e-6, "head_lr": 1e-4}), "baseline experiment only"),
    (lambda r: r["init"]["pretrained_group_sha256"].update({"block.00": "0" * 64}), "pinned B values"),
    (lambda r: r["selection"].update(metric="recording AUC"), "fixed 0.5"),
    (lambda r: r.update(protocol="B"), "not an R4-U1b"),
    (lambda r: r["derived_from"].update(sha256="0" * 64), "differs from the one"),
])
def test_finetune_recipe_departures_are_rejected(tmp_path, edit, message):
    root = tmp_path
    (root / "configs").mkdir()
    (root / T.B_FINETUNE).write_bytes((ROOT / T.B_FINETUNE).read_bytes())
    r = copy.deepcopy(ft_recipe())
    edit(r)
    path = root / "configs/f.json"
    path.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(T.R4U1Error, match=message):
        T.load_finetune_recipe(path, root=root)


def test_train_configs_follow_the_recipes_use_protocol_a_and_require_an_owner():
    c = T.finetune_train_config(ft_recipe(), owner="tester")
    assert (c.max_epochs, c.batch_size, c.accumulation_steps, c.scheduler, c.warmup_steps, c.precision) == (8, 8, 2, "cosine", 189, "fp16")
    assert c.lr == 1e-4 and c.model_name == "ast_partial" and c.protocol == "A" and c.owner == "tester"
    w = T.warmup_train_config(warm_recipe(), owner="tester")
    assert (w.max_epochs, w.batch_size, w.accumulation_steps, w.scheduler, w.lr, w.model_name, w.protocol) == (2, 4, 4, "constant", 1e-4, "ast_frozen", "A")
    for fn, r in ((T.finetune_train_config, ft_recipe()), (T.warmup_train_config, warm_recipe())):
        with pytest.raises(T.R4U1Error):
            fn(r, owner="")


# ---- guarded A data --------------------------------------------------------------------------------------------------------------------------

def test_input_verification_rejects_changed_files_wrong_split_and_unverified_manifest(tmp_path):
    for name, content in (("cohort", b"a"), ("split", b"b"), ("chunks", b"c")):
        (tmp_path / name).write_bytes(content)
    paths = {k: tmp_path / k for k in ("cohort", "split", "chunks")}
    sha = {k: file_digest(v) for k, v in paths.items()}
    manifest = {"task": "R4-S1", "protocol": "A", "verification": {"passed": True},
                "inputs": {"split_sha256": "f" * 64, "cohort_manifest_file_sha256": sha["cohort"], "split_file_sha256": sha["split"],
                           "chunk_manifest_file_sha256": sha["chunks"]}}
    assert D.verify_inputs(paths, manifest, "f" * 64) == sha
    with pytest.raises(D.R4U1DataError, match="frozen split"):
        D.verify_inputs(paths, manifest, "e" * 64)
    (tmp_path / "chunks").write_bytes(b"changed")
    with pytest.raises(D.R4U1DataError, match="hash mismatch"):
        D.verify_inputs(paths, manifest, "f" * 64)
    for bad in ({"verification": {"passed": False}}, {"protocol": "B"}, {"task": "R2-S1"}):
        with pytest.raises(D.R4U1DataError, match="not verified"):
            D.verify_inputs(paths, {**manifest, **bad}, "f" * 64)


def test_build_data_reads_only_a_train_and_validation_and_never_a_test_member(tmp_path_factory):
    from test_r4_u0_wavlm_cache import World
    root = tmp_path_factory.mktemp("r4u1data")
    world = World(root)
    manifest = json.loads(world.s1_path.read_text(encoding="utf-8"))
    manifest.update(protocol="A", counts={**world.plan_a.counts, "test_assignments_dropped": 0})   # the real build used the sealed=False loader
    world.s1_path.write_text(json.dumps(manifest), encoding="utf-8")
    data = D.build_data(root, world.shard_dir, seed=42)
    try:
        assert set(data["validation"]) == set(D.ROLES)
        assert set(data["validation_components"].values()) == set(COMPONENTS)
        assert data["counts"]["train"] == len(world.plan_a.train) == len(data["train"])
        assert data["counts"]["validation_chunks"] == {r: len(data["validation"][r]) for r in D.ROLES}
        data["train"].set_epoch(1)
        for i in range(len(data["train"])):
            assert len(data["train"][i]["waveform"]) > 0                                   # decoded and PCM-hash verified
        for role in D.ROLES:
            for i in range(len(data["validation"][role])):
                data["validation"][role][i]
        assert data["plan_sha256"] == world.plan_a.sha256
        # a test recording is unreachable: its shard member is undecodable and it is not in the loader scope
        test_id = next(r["recording_id"] for r in world.rows if r["recording_id"].endswith("301-2-0001") and r["component"] == "chatterbox")
        with pytest.raises(KeyError):
            data["loader"](test_id)
        indices = D.select_chunks(data["validation"]["selection"], data["validation_components"], 10)
        assert len(indices) == 10 and {data["validation_components"][data["validation"]["selection"].chunks[i].recording_id] for i in indices} == set(COMPONENTS)
        # tampering: counts that differ from the R4-S1 manifest are refused
        bad = dict(manifest, counts={**manifest["counts"], "train_recordings": 1})
        world.s1_path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(D.R4U1DataError, match="counts differ"):
            D.build_data(root, world.shard_dir)
    finally:
        data["loader"].close()


# ---- tiny AST fixtures ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("r4u1ast")
    snapshot = root / "snapshot"
    tiny_snapshot(snapshot)
    extractor = tiny_feature_extractor()
    try:
        extractor([np.zeros(RATE, dtype=np.float32)], sampling_rate=RATE, return_tensors="pt")
    except Exception:
        pytest.skip("AST feature extractor backend unavailable")
    fresh = ASTDetector.from_dir(snapshot, seed=1, feature_extractor=extractor)
    pins = {g: h for g, h in fresh.digests().items() if g != "classifier"}
    return SimpleNamespace(root=root, snapshot=snapshot, extractor=extractor, pins=pins)


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    waves, train_items = {}, []
    for i in range(30):
        rid = f"phonemedf:{COMPONENTS[i % 5]}:train{i}"
        waves[rid] = rng.normal(0, .1, 12000 + 500 * i).astype(np.float32)
        train_items.append((rid, int(i % 5 != 0)))
    comps = {"selection": ["genuine"] * 4 + ["melo", "vits", "xtts", "chatterbox"] * 2, "calibration": ["genuine"] * 3 + ["vits", "xtts", "chatterbox"],
             "operating": ["genuine"] * 3 + ["melo", "xtts", "chatterbox"]}
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
                role_chunks.append(SimpleNamespace(chunk_id=f"chunk:A:{rid}:{s}", recording_id=rid, start_sample=s, end_sample=e, valid_samples=e - s,
                                                   input_samples=12000))
        chunks[role] = role_chunks
    loader = DictWaveformLoader(waves)
    train = CropDataset(train_items, loader, seed=3)
    validation = {}
    for role in chunks:
        validation[role] = ChunkDataset(chunks[role], labels, loader)
        validation[role].validation_role = role
    return {"train": train, "validation": validation, "validation_components": components, "validation_groups": groups, "split_sha256": "0" * 64,
            "input_sha256": {"x": "1" * 64}, "loader": loader}


def warmup(world, tmp_path, name="warm", data=None, recipe=None, **kw):
    return T.run_warmup(recipe or warm_recipe(), data or make_data(), snapshot=world.snapshot, run_dir=tmp_path / name, ledger=tmp_path / "ledger.csv",
                        owner="tester", pins=world.pins, feature_extractor=world.extractor, overrides=FAST_WARM, **kw)


def finetune(world, tmp_path, warm, name="ft", data=None, recipe=None, **kw):
    return T.run_finetune(recipe or ft_recipe(), warm, data or make_data(), snapshot=world.snapshot, run_dir=tmp_path / name,
                          ledger=tmp_path / "ledger.csv", owner="tester", feature_extractor=world.extractor, overrides=FAST_FT, **kw)


# ---- warm-up -----------------------------------------------------------------------------------------------------------------------------------

def test_warmup_trains_the_head_only_and_records_a_result(world, tmp_path):
    result = warmup(world, tmp_path)
    run = tmp_path / "warm"
    assert result["epochs_run"] == 2 and result["problems"] == [] and result["backbone_unchanged"] is True
    assert all(result["group_sha256_final"][g] == result["group_sha256_start"][g] == result["group_sha256_best"][g] for g in world.pins)
    assert result["group_sha256_final"]["classifier"] != result["group_sha256_start"]["classifier"]
    assert result["best_checkpoint_sha256"] == file_digest(run / "best.pt") and result["protocol"] == "A"
    loaded = T.load_warmup_result(run, warm_recipe())
    assert loaded["best_checkpoint_path"] == str(run / "best.pt")
    rows = (tmp_path / "ledger.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and "ast_frozen" in rows[1] and ",A," in rows[1] and ",completed," in rows[1]
    with pytest.raises(T.R4U1Error, match="non-empty"):
        warmup(world, tmp_path)


def test_warmup_refuses_a_backbone_that_differs_from_the_pins(world, tmp_path):
    with pytest.raises(T.R4U1Error, match="pinned pretrained groups"):
        T.run_warmup(warm_recipe(), make_data(), snapshot=world.snapshot, run_dir=tmp_path / "w", ledger=None, owner="tester",
                     pins={**world.pins, "block.00": "0" * 64}, feature_extractor=world.extractor, overrides=FAST_WARM)


def test_the_warmup_result_is_checked_before_it_is_used(world, tmp_path):
    warmup(world, tmp_path)
    run = tmp_path / "warm"
    (run / "best.pt").write_bytes((run / "best.pt").read_bytes() + b"x")
    with pytest.raises(T.R4U1Error, match="changed since"):
        T.load_warmup_result(run)
    other = copy.deepcopy(warm_recipe())
    other["status"] = "something_else"
    (run / "best.pt").write_bytes((run / "best.pt").read_bytes()[:-1])
    with pytest.raises(T.R4U1Error, match="recipe changed"):
        T.load_warmup_result(run, other)
    result = json.loads((run / "warmup_result.json").read_text(encoding="utf-8"))
    result["problems"] = ["a backbone group changed during the head-only warm-up"]
    (run / "warmup_result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(T.R4U1Error, match="clean A warm-up"):
        T.load_warmup_result(run)


# ---- fine-tune ---------------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(world, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("r4u1run")
    data = make_data()
    result = T.run_warmup(warm_recipe(), data, snapshot=world.snapshot, run_dir=tmp / "warm", ledger=tmp / "ledger.csv", owner="tester", pins=world.pins,
                          feature_extractor=world.extractor, overrides=FAST_WARM)
    warm = T.load_warmup_result(tmp / "warm", warm_recipe())
    recipe = copy.deepcopy(ft_recipe())
    recipe["init"]["pretrained_group_sha256"] = world.pins                      # the tiny model's pretrained groups stand in for the real pins
    summary = T.run_finetune(recipe, warm, data, snapshot=world.snapshot, run_dir=tmp / "ft", ledger=tmp / "ledger.csv", owner="tester",
                             feature_extractor=world.extractor, overrides=FAST_FT)
    return SimpleNamespace(tmp=tmp, data=data, warm=warm, recipe=recipe, summary=summary, result=result)


def test_finetune_evidence_and_ledger(world, trained):
    run = trained.tmp / "ft"
    for name in ("gradient_proof.json", "epoch0_metrics.json", "epoch0_predictions_chunks.parquet", "adaptation_evidence.json", "best.pt", "latest.pt",
                 "history.jsonl", "status.json", "resolved_config.json"):
        assert (run / name).is_file(), name
    ev = json.loads((run / "adaptation_evidence.json").read_text(encoding="utf-8"))
    assert ev["task"] == "R4-U1b" and ev["protocol"] == "A" and ev["problems"] == []
    assert ev["trainable_groups"] == ["block.02", "block.03", "classifier", "final_norm"]
    assert ev["frozen_groups_unchanged_final_and_best"] is True
    assert ev["unfrozen_groups_changed_final"] == ev["unfrozen_groups_changed_best"] == ev["trainable_groups"]
    assert ev["initial_checkpoint"]["checkpoint_sha256"] == trained.warm["best_checkpoint_sha256"]
    assert ev["warmup_best_macro_f1"] == trained.warm["best_macro_f1"]
    assert ev["optimizer_steps_per_epoch"] == 4 and ev["optimizer_steps_total"] == 8                    # 30 crops, microbatch 4, accumulation 2, 2 epochs
    epoch0 = json.loads((run / "epoch0_metrics.json").read_text(encoding="utf-8"))
    assert epoch0["within_tolerance"] is True and epoch0["recorded_warmup_macro_f1"] == trained.warm["best_macro_f1"]
    assert set(epoch0["panels"]) == {"genuine_vs_" + c for c in COMPONENTS if c != "genuine"}
    rows = (trained.tmp / "ledger.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 and "ast_partial" in rows[2] and ",A," in rows[2] and ",completed," in rows[2]


def test_finetune_directory_is_never_reused_and_a_wrong_warmup_score_is_refused(world, trained, tmp_path):
    with pytest.raises(T.R4U1Error, match="non-empty"):
        T.run_finetune(trained.recipe, trained.warm, make_data(), snapshot=world.snapshot, run_dir=trained.tmp / "ft", ledger=None, owner="tester",
                       feature_extractor=world.extractor, overrides=FAST_FT)
    wrong = dict(trained.warm, best_macro_f1=0.123456)
    with pytest.raises(T.R4U1Error, match="wrong weights or preprocessing"):
        T.run_finetune(trained.recipe, wrong, make_data(), snapshot=world.snapshot, run_dir=tmp_path / "other", ledger=None, owner="tester",
                       feature_extractor=world.extractor, overrides=FAST_FT)
    check = json.loads((tmp_path / "other/epoch0_metrics.json").read_text(encoding="utf-8"))
    assert check["within_tolerance"] is False                                    # the evidence is written before the refusal


def test_finetune_refuses_a_warmup_checkpoint_that_does_not_match_its_recorded_hash(world, trained, tmp_path):
    forged = dict(trained.warm, best_checkpoint_sha256="0" * 64)
    with pytest.raises(Exception, match="SHA-256"):
        T.run_finetune(trained.recipe, forged, make_data(), snapshot=world.snapshot, run_dir=tmp_path / "forged", ledger=None, owner="tester",
                       feature_extractor=world.extractor, overrides=FAST_FT)


def test_finish_freezes_the_selection_exports_every_role_and_compares_per_component(world, trained):
    import pyarrow.parquet as pq
    run = trained.tmp / "ft"
    evidence = T.finish(run, trained.data, trained.summary, trained.recipe, trained.warm, snapshot=world.snapshot, feature_extractor=world.extractor,
                        device="cpu", overrides=FAST_FT)
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    assert selection["protocol"] == "A" and selection["selected"] == "baseline" and selection["selection_threshold"] == 0.5
    assert selection["init_checkpoint_sha256"] == trained.warm["best_checkpoint_sha256"]
    assert selection["selected_epoch"] == trained.summary["best_epoch"]
    for role in T.ROLES:
        chunks = pq.read_table(run / f"predictions_chunks_{role}.parquet")
        recs = pq.read_table(run / f"predictions_recordings_{role}.parquet")
        assert chunks.column_names == ["chunk_id", "recording_id", "group_id", "label", "component", "validation_role", "tts_probability"]
        assert recs.column_names == ["recording_id", "group_id", "label", "component", "validation_role", "raw_score", "chunk_count"]
        assert set(chunks.column("validation_role").to_pylist()) == {role}
        assert "chatterbox" in set(recs.column("component").to_pylist())                                  # A includes Chatterbox
        assert all(0 <= s <= 1 for s in recs.column("raw_score").to_pylist())
    assert evidence["reload_parity"]["max_abs_probability_difference_vs_saved_best_epoch"] <= T.PARITY_ATOL
    paired = evidence["paired_warmup_vs_adapted"]
    assert paired["recordings"] == evidence["counts"]["selection"]["recordings"] == 12
    assert sum(paired["decision_changes_at_0.5"].values()) == paired["recordings"]
    assert set(paired["warmup"]) == set(paired["adapted"]) == {"all", *("genuine_vs_" + c for c in COMPONENTS if c != "genuine")}
    assert set(evidence["selection_panels"]) == {"genuine_vs_" + c for c in COMPONENTS if c != "genuine"}
    assert evidence["selection_sha256"] == file_digest(run / "selection.json")


def test_the_selection_is_idempotent_and_cannot_be_silently_changed(world, trained):
    run = trained.tmp / "ft"
    kwargs = dict(snapshot=world.snapshot, feature_extractor=world.extractor, device="cpu", overrides=FAST_FT)
    first = T.finish(run, trained.data, trained.summary, trained.recipe, trained.warm, **kwargs)
    again = T.finish(run, trained.data, trained.summary, trained.recipe, trained.warm, **kwargs)
    assert first["selection"] == again["selection"]
    changed = copy.deepcopy(trained.recipe)
    changed["status"] = "something_else"
    with pytest.raises(T.R4U1Error, match="refusing to overwrite"):
        T.finish(run, trained.data, trained.summary, changed, trained.warm, **kwargs)
    with pytest.raises(T.R4U1Error, match="inconsistent or unfinished"):
        T.finish(run, trained.data, dict(trained.summary, stop_reason="early_stopping"), trained.recipe, trained.warm, **kwargs)


def test_frozen_b_sources_are_untouched():
    lock = json.dumps(json.loads((ROOT / "locks/B_lock_gr3.json").read_text(encoding="utf-8")))
    for rel in ("src/echoproof/training/r3_u2_train.py", "src/echoproof/training/r3_u2_data.py", "src/echoproof/training/r2_u2_scope.py",
                "src/echoproof/models/ast_detector.py", "src/echoproof/models/ast_adapter.py", "src/echoproof/models/r3_u2_bundle.py",
                "src/echoproof/training/trainer.py"):
        assert digest_in_lock(ROOT / rel, lock), f"{rel} differs from the hash recorded in the GR3 lock (beyond line endings)"


# ---- resuming a run on a new VM -----------------------------------------------------------------------------------------------------------

def _script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r4_u1_ast_script", ROOT / "scripts/r4_u1_ast.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_tolerates_only_the_platform_string_and_names_every_other_difference(tmp_path):
    script = _script()
    run = tmp_path / "run"
    contract = {"task": "R4-U1b", "recipe_sha256": "a" * 64, "sources": {"x.py": "1" * 64, "scripts/r4_u1_ast.py": "1" * 64},
                "environment": {"python": "3.13", "torch": "2.14.1", "gpu": "Tesla T4", "platform": "Linux-6.6.159+"}}
    path = script.freeze_contract(run, contract, resume=False)
    assert path.is_file()
    with pytest.raises(FileExistsError):
        script.freeze_contract(run, contract, resume=False)
    assert script.freeze_contract(run, contract, resume=True) == path                       # identical contract resumes
    moved = copy.deepcopy(contract)
    moved["environment"]["platform"] = "Linux-6.6.200+"
    moved["sources"]["scripts/r4_u1_ast.py"] = "3" * 64
    script.freeze_contract(run, moved, resume=True)                                         # the kernel string and the orchestration script differ
    log = (tmp_path / "run_resume_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1 and "environment.platform" in log[0]
    for edit, field in ((lambda c: c["environment"].update(gpu="L4"), "environment.gpu"),
                        (lambda c: c["environment"].update(torch="2.15.0"), "environment.torch"),
                        (lambda c: c["sources"].update({"x.py": "2" * 64}), "sources.x.py"),
                        (lambda c: c.update(recipe_sha256="b" * 64), "recipe_sha256")):
        changed = copy.deepcopy(contract)
        edit(changed)
        with pytest.raises(SystemExit, match=field):
            script.freeze_contract(run, changed, resume=True)
    with pytest.raises(SystemExit, match="no frozen contract"):
        script.freeze_contract(tmp_path / "other", contract, resume=True)
