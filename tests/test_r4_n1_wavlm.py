"""R4-N1 WavLM-A calibration on a synthetic Protocol A project: recipe guard, audit, frozen fit, single operating pass, plateau report, per-component panels,
bundle packaging and loader. Numpy, scipy, pyarrow and matplotlib only; no model, no audio, no transformers."""
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle, write_bundle
from echoproof.contracts import BundleSpecification, ContractError, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.evaluation import r3_n1_wavlm as W3
from echoproof.evaluation import r4_n1_wavlm as W
from echoproof.evaluation.metrics import compute_metrics
from echoproof.models import r3_n1_wavlm_bundle as B3
from echoproof.models import r4_n1_wavlm_bundle as B

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = W.COMPONENTS
SPEAKERS = {"selection": 4, "calibration": 6, "operating": 5}
PER_SPEAKER = 3                       # recordings per component per speaker
CHECKPOINT = "c" * 64
SPLIT = "a" * 64


def digest_in_lock(path, lock_text):
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return any(hashlib.sha256(variant).hexdigest() in lock_text for variant in (raw, lf, lf.replace(b"\n", b"\r\n")))


def make_exports(directory: Path, scale=3.0, seed=0):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    for role, speakers in SPEAKERS.items():
        recs, chunks = [], []
        for s in range(speakers):
            group = f"speaker:{role[0]}{s}"
            for comp in COMPONENTS:
                label = int(comp != "genuine")
                for r in range(PER_SPEAKER):
                    rid = f"phonemedf:{comp}:{role[0]}{s}-{r}"
                    base = (1.2 if comp == "xtts" else 2.5) if label else -2.5
                    probs = []
                    for c in range(2):
                        p = float(np.clip(1 / (1 + np.exp(-scale * (base + rng.normal(0, 1.2)))), 0.0, 1.0))
                        probs.append(p)
                        chunks.append({"chunk_id": f"chunk:A:{rid}:{c * 32000}", "recording_id": rid, "group_id": group, "label": label, "component": comp,
                                       "validation_role": role, "tts_probability": p})
                    recs.append({"recording_id": rid, "group_id": group, "label": label, "component": comp, "validation_role": role,
                                 "raw_score": float(np.mean(probs)), "chunk_count": 2})
        schema = pa.schema([("recording_id", pa.string()), ("group_id", pa.string()), ("label", pa.int64()), ("component", pa.string()),
                            ("validation_role", pa.string()), ("raw_score", pa.float64()), ("chunk_count", pa.int64())])
        pq.write_table(pa.Table.from_pylist(recs, schema=schema), directory / f"predictions_recordings_{role}.parquet")
        pq.write_table(pa.Table.from_pylist(chunks), directory / f"predictions_chunks_{role}.parquet")


def make_project(root: Path, **kw):
    inputs = root / "reports/r4/r4_u1_wavlm_A"
    make_exports(inputs, **kw)
    predictions = {f"predictions_{kind}_{role}.parquet": file_digest(inputs / f"predictions_{kind}_{role}.parquet") for kind in ("chunks", "recordings") for role in W.ROLES}
    (inputs / "evidence.json").write_text(json.dumps({"output_sha256": predictions}), encoding="utf-8")
    y, s, *_ = W3.arrays(W3.read_recordings(inputs, "selection"))
    selection = {"selected": "mean_std", "selected_epoch": 13, "protocol": "A", "checkpoint_sha256": {"mean_std": CHECKPOINT}, "split_sha256": SPLIT,
                 "selection_macro_f1": {"mean_std": compute_metrics(y, s, 0.5)["macro_f1"]}}
    (inputs / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    bundle = make_source(root / "bundles", name="src")
    recipe = json.loads((ROOT / "configs/r4_n1_wavlm_calibration_a.json").read_text(encoding="utf-8"))
    recipe["inputs"].update(directory="reports/r4/r4_u1_wavlm_A", selection_json_sha256=file_digest(inputs / "selection.json"),
                            evidence_json_sha256=file_digest(inputs / "evidence.json"), predictions_sha256=predictions)
    recipe["model"].update(checkpoint_sha256=CHECKPOINT, split_sha256=SPLIT, source_bundle="bundles/src", source_bundle_json_sha256=file_digest(bundle / "bundle.json"))
    recipe["bundle_destination"] = "bundles/out"
    for role, speakers in SPEAKERS.items():
        recipe["roles"][role] = {"recordings": speakers * 5 * PER_SPEAKER, "speakers": speakers, "chunks": speakers * 5 * PER_SPEAKER * 2, "per_component": speakers * PER_SPEAKER}
    for role in ("calibration", "operating"):
        y, s, *_ = W3.arrays(W3.read_recordings(inputs, role))
        m = compute_metrics(y, s, 0.5)
        recipe["expected_raw_summary_at_0_5"][role] = {"macro_f1": round(m["macro_f1"], 4), "genuine_fpr": round(m["fpr"], 4), "eer": round(m["eer"], 4),
                                                       "roc_auc": round(m["roc_auc"], 4)}
    for role in W.ROLES:
        _, s, *_ = W3.arrays(W3.read_recordings(inputs, role))
        recipe["expected_saturation"][role] = W3.saturation(s, 1e-7)
    return recipe


def run_audit(root, recipe, run_dir):
    run_dir.mkdir(exist_ok=True)
    return W.audit(root, recipe, run_dir, join_manifests=False)


# ---- the recipe ------------------------------------------------------------------------------------------------------------------------------

def test_the_shipped_recipe_loads_and_equals_the_b_procedure():
    recipe = W.load_recipe(ROOT / "configs/r4_n1_wavlm_calibration_a.json", root=ROOT)
    b = json.loads((ROOT / "configs/r3_n1_wavlm_calibration_b.json").read_text(encoding="utf-8"))
    for key in ("epsilon", "seed", "candidates", "temperature", "affine", "selection_rule", "acceptance_rule", "bootstrap", "reliability", "transformers_pin"):
        assert recipe[key] == b[key]
    assert {k: v for k, v in recipe["threshold_rule"].items() if k != "plateau_report"} == b["threshold_rule"]
    assert recipe["protocol"] == "A" and recipe["components"] == list(COMPONENTS) and recipe["split"]["file"].endswith("protocol_A_v1.parquet")
    assert sum(r["recordings"] for r in recipe["roles"].values()) == 5000


@pytest.mark.parametrize("edit,message", [
    (lambda r: r.update(protocol="B"), "Protocol A"),
    (lambda r: r.update(status="owner_confirmed_2026-11-14"), "agreed"),
    (lambda r: r.update(candidates=["identity", "temperature"]), "fixed"),
    (lambda r: r.update(components=["genuine", "melo", "vits", "xtts"]), "five"),
    (lambda r: r["acceptance_rule"].update(tie="accepted"), "differs from the B recipe"),
    (lambda r: r["selection_rule"].update(choice="best candidate"), "differs from the B recipe"),
    (lambda r: r["affine"].update(regularisation=0.5), "differs from the B recipe"),
    (lambda r: r.update(seed=7), "differs from the B recipe"),
])
def test_recipe_guard_refuses_each_departure(tmp_path, edit, message):
    recipe = json.loads((ROOT / "configs/r4_n1_wavlm_calibration_a.json").read_text(encoding="utf-8"))
    edit(recipe)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(W.A_CalibrationError, match=message):
        W.load_recipe(path, root=ROOT)


# ---- role checks and A-only pieces -----------------------------------------------------------------------------------------------------------------

def test_role_rows_need_all_five_components(tmp_path):
    recipe = make_project(tmp_path)
    rows = W3.read_recordings(tmp_path / "reports/r4/r4_u1_wavlm_A", "calibration")
    assert W.check_role_rows(rows, "calibration", recipe) == []
    no_chatterbox = [r for r in rows if r["component"] != "chatterbox"]
    assert any("component counts" in p for p in W.check_role_rows(no_chatterbox, "calibration", recipe))
    bad = copy.deepcopy(rows)
    bad[0]["label"] = 1 - bad[0]["label"]
    assert any("label/component" in p for p in W.check_role_rows(bad, "calibration", recipe))


def test_plateau_counts_the_tied_partitions():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.01, 0.02, 0.03, 0.9, 0.95, 0.99])                   # separable: every threshold between 0.03 and 0.9 is perfect, i.e. the thresholds 0.9 only
    p = W.plateau(y, s)
    assert p["best_macro_f1"] == 1.0 and p["partitions_at_best"] == 1 and p["lowest_threshold_at_best"] == 0.9
    s2 = np.array([0.01, 0.02, 0.03, 0.5, 0.95, 0.99])
    assert W.plateau(y, s2)["partitions_at_best"] == 1
    wide = np.array([0.10, 0.30, 0.50, 0.55, 0.60, 0.99])              # a tie that spans scores
    y2 = np.array([0, 0, 0, 1, 1, 1])
    assert W.plateau(y2, wide)["highest_threshold_at_best"] == 0.55


def test_panel_metrics_cover_each_generator():
    y = np.array([0, 0, 1, 1, 1, 1, 1])
    comp = np.array(["genuine", "genuine", "melo", "vits", "xtts", "chatterbox", "chatterbox"])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.4])
    out = W.panel_metrics(y, comp, s, DecisionRule(1, "threshold", 0.5))
    assert set(out) == {"full"} | {f"genuine_vs_{g}" for g in W.GENERATORS}
    assert out["genuine_vs_chatterbox"]["fn"] == 1 and out["genuine_vs_melo"]["fn"] == 0


# ---- stages on the synthetic project ----------------------------------------------------------------------------------------------------------

def test_audit_passes_and_fails_on_a_changed_file(tmp_path):
    recipe = make_project(tmp_path)
    result = run_audit(tmp_path, recipe, tmp_path / "run1")
    assert result["passed"] and result["test_partition_read"] is False and all(result["checks"].values())
    target = tmp_path / "reports/r4/r4_u1_wavlm_A/predictions_chunks_operating.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(W.A_CalibrationError):
        run_audit(tmp_path, recipe, tmp_path / "run2")
    assert json.loads((tmp_path / "run2/input_audit.json").read_text(encoding="utf-8"))["passed"] is False


def test_audit_rejects_a_selection_that_is_not_protocol_a(tmp_path):
    recipe = make_project(tmp_path)
    path = tmp_path / "reports/r4/r4_u1_wavlm_A/selection.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    d["protocol"] = "B"
    path.write_text(json.dumps(d), encoding="utf-8")
    recipe["inputs"]["selection_json_sha256"] = file_digest(path)
    with pytest.raises(W.A_CalibrationError, match="selection_protocol_is_A"):
        run_audit(tmp_path, recipe, tmp_path / "run")


def test_audit_join_refuses_unknown_recordings_and_withheld_test_rows(tmp_path, monkeypatch):
    from echoproof.data import cohort, splits
    recipe = make_project(tmp_path)
    inputs = tmp_path / "reports/r4/r4_u1_wavlm_A"
    rows = [r for role in W.ROLES for r in W3.read_recordings(inputs, role)]
    cohort_rows = [{"recording_id": r["recording_id"], "label": r["label"], "component": r["component"]} for r in rows]
    assignments = [SimpleNamespace(recording_id=r["recording_id"], partition="validation", validation_role=r["validation_role"], group_id=r["group_id"]) for r in rows]
    asked = {}
    monkeypatch.setattr(cohort, "read_table", lambda path: cohort_rows)

    def fake_load(path, *, sealed=False):
        asked["sealed"], asked["path"] = sealed, str(path)
        return list(assignments)
    monkeypatch.setattr(splits, "load_assignments", fake_load)
    ok = W.audit(tmp_path, recipe, (tmp_path / "ok_dir").mkdir() or tmp_path / "ok_dir")
    assert ok["passed"] and asked["sealed"] is False and asked["path"].replace("\\", "/").endswith("data/splits/protocol_A_v1.parquet")
    assignments.pop()                                                     # a recording the test-withholding loader no longer returns
    bad_dir = tmp_path / "bad_dir"
    bad_dir.mkdir()
    with pytest.raises(W.A_CalibrationError, match="locked A cohort"):
        W.audit(tmp_path, recipe, bad_dir)
    assignments[0] = SimpleNamespace(recording_id=assignments[0].recording_id, partition="validation", validation_role="operating", group_id="x")
    other = tmp_path / "other_dir"
    other.mkdir()
    with pytest.raises(W.A_CalibrationError, match="locked A cohort"):
        W.audit(tmp_path, recipe, other)


def test_fit_never_reads_the_operating_role_and_freezes_the_mapping(tmp_path, monkeypatch):
    recipe = make_project(tmp_path)
    run_dir = tmp_path / "run"
    run_audit(tmp_path, recipe, run_dir)
    asked = []
    original = W.read_recordings
    monkeypatch.setattr(W, "read_recordings", lambda d, role: (asked.append(role), original(d, role))[1])
    fit = W.fit_stage(tmp_path, recipe, run_dir)
    assert asked == ["calibration"] and fit["operating_predictions_read"] is False and fit["task"] == "R4-N1-WavLM mapping fit"
    assert fit["selected_mapping"] in W.ORDER
    done = json.loads((run_dir / "fit/stage_complete.json").read_text(encoding="utf-8"))
    assert all(file_digest(run_dir / "fit" / n) == h for n, h in done["files"].items())
    comparison = json.loads((run_dir / "fit/loso_comparison.json").read_text(encoding="utf-8"))
    assert len(comparison["folds"]) == SPEAKERS["calibration"]
    with pytest.raises(FileExistsError):
        W.fit_stage(tmp_path, recipe, run_dir)


def test_fit_refuses_without_a_passing_audit_or_with_a_different_recipe(tmp_path):
    recipe = make_project(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        W.fit_stage(tmp_path, recipe, run_dir)
    run_audit(tmp_path, recipe, run_dir)
    other = copy.deepcopy(recipe)
    other["epsilon"] = 1e-6
    with pytest.raises(W.A_CalibrationError):
        W.fit_stage(tmp_path, other, run_dir)


@pytest.mark.parametrize("scale", [3.0, 1.0])
def test_operating_stage_end_to_end_is_consistent_and_opens_the_role_once(tmp_path, scale):
    recipe = make_project(tmp_path, scale=scale, seed=3)
    run_dir = tmp_path / "run"
    run_audit(tmp_path, recipe, run_dir)
    fit = W.fit_stage(tmp_path, recipe, run_dir)
    result = W.operating_stage(tmp_path, recipe, run_dir, bootstrap_resamples=30, plots=False)
    thr, ev = result["threshold"], result["evidence"]
    assert thr["task"] == "R4-N1-WavLM operating policy" and thr["calibration_accepted"] == thr["acceptance"]["accepted"]
    assert thr["selected_mapping"] == fit["selected_mapping"]
    assert thr["score_type"] == ("calibrated_probability" if thr["calibration_accepted"] else "model_score")
    out = run_dir / "operating"
    preds = pq.read_table(out / "operating_predictions.parquet").to_pylist()
    rule = DecisionRule(1, thr["decision_rule"]["kind"], thr["decision_rule"]["threshold"])
    assert all(rule.decide(r["deployed_score"]) == r["decision"] for r in preds) and len(preds) == recipe["roles"]["operating"]["recordings"]
    if not thr["calibration_accepted"]:
        assert all(r["deployed_score"] == r["raw_score"] for r in preds)
    rows = result["metrics"]["rows"]
    assert rows["A_raw_at_0_5"]["rule"]["threshold"] == 0.5
    assert rows["C_raw_at_raw_optimum"]["full"]["macro_f1"] >= rows["A_raw_at_0_5"]["full"]["macro_f1"] - 1e-12
    assert {f"genuine_vs_{g}" for g in W.GENERATORS} <= set(rows["A_raw_at_0_5"])
    plateau = thr["plateau"]
    assert plateau["partitions_at_best"] >= 1 and plateau["lowest_threshold_at_best"] <= plateau["highest_threshold_at_best"]
    assert all(file_digest(out / n) == h for n, h in ev["outputs"].items())
    assert ev["fit_on_operating"] is False and ev["test_data_read"] is False
    assert not (out / "duration_matched.json").exists() and "Threshold plateau" in (out / "comparison.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        W.operating_stage(tmp_path, recipe, run_dir, bootstrap_resamples=30, plots=False)


def test_operating_stage_refuses_a_fit_that_changed_after_freezing(tmp_path):
    recipe = make_project(tmp_path)
    run_dir = tmp_path / "run"
    run_audit(tmp_path, recipe, run_dir)
    W.fit_stage(tmp_path, recipe, run_dir)
    path = run_dir / "fit/calibration.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"fit_role"', '"fit_role_"'), encoding="utf-8")
    with pytest.raises(W.A_CalibrationError):
        W.operating_stage(tmp_path, recipe, run_dir, bootstrap_resamples=10, plots=False)
    assert not (run_dir / "operating").exists()


# ---- bundle ---------------------------------------------------------------------------------------------------------------------------------

def make_source(tmp_path: Path, name: str = "source") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    contents = {name_: f"dummy {name_}".encode() for name_ in B.MODEL_COMPONENTS}
    contents["model/wavlm_head.json"] = json.dumps({"source_checkpoint_sha256": CHECKPOINT, "pooling": "mean_std", "split_sha256": SPLIT}).encode()
    spec = BundleSpecification(1, "bundle:test-source-A", B.MODEL_VERSION, PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.5), "genuine", "tts")
    write_bundle(tmp_path / name, spec, contents, extra={"stage": "development_uncalibrated"})
    return tmp_path / name


def calibration(source: Path, *, method="affine_on_log_odds", params=None, accepted=True, threshold=0.4):
    params = params if params is not None else ({"a": math.exp(0.1), "b": 0.9, "u": 0.1} if method == "affine_on_log_odds" else {"temperature": 1.5})
    return {"schema": B3.CALIBRATION_SCHEMA, "model": "wavlm_head", "method": method, "parameters": params, "epsilon": 1e-7, "calibration_accepted": accepted,
            "score_type": "calibrated_probability" if accepted else "model_score", "operating_decision_rule": {"kind": "threshold", "threshold": threshold},
            "raw_equivalent_threshold": None, "source": {"source_bundle_json_sha256": file_digest(source / "bundle.json"), "source_checkpoint_sha256": CHECKPOINT,
                                                         "split_sha256": SPLIT, "recipe_sha256": "f" * 64, "fit_calibration_json_sha256": "1" * 64,
                                                         "operating_evidence_sha256": "2" * 64}}


@pytest.fixture()
def built(tmp_path):
    source = make_source(tmp_path)
    meta = calibration(source)
    return tmp_path, source, meta, B.package_calibrated_wavlm_a(source, tmp_path / "calibrated", meta)


def test_packaging_copies_the_model_bytes_and_labels_the_bundle_as_a(built):
    tmp, source, meta, dest = built
    original, written = load_bundle(source), load_bundle(dest)
    for name in B.MODEL_COMPONENTS:
        assert file_digest(written.components[name]) == file_digest(original.components[name])
    assert set(written.components) == set(B.MODEL_COMPONENTS) | {B.CALIBRATION_COMPONENT}
    assert written.spec.decision_rule == DecisionRule(1, "threshold", 0.4) and written.spec.model_version == "wavlm_head:A"
    extra = written.raw_metadata["extra"]
    assert extra["stage"] == "r4_n1_wavlm_development" and extra["protocol"] == "A" and extra["promoted"] is False and extra["locked"] is False
    assert written.spec.bundle_id == B.BUNDLE_ID


def test_packaging_refuses_overwrite_b_sources_and_mismatches(built):
    tmp, source, meta, dest = built
    with pytest.raises(BundleError, match="overwrite"):
        B.package_calibrated_wavlm_a(source, dest, meta)
    for field, value, message in (("source_bundle_json_sha256", "9" * 64, "source bundle differs"), ("source_checkpoint_sha256", "9" * 64, "different checkpoint"),
                                  ("split_sha256", "9" * 64, "different split")):
        other = copy.deepcopy(meta)
        other["source"][field] = value
        with pytest.raises(BundleError, match=message):
            B.package_calibrated_wavlm_a(source, tmp / f"o_{field}", other)
        assert not (tmp / f"o_{field}").exists()
    b_spec = BundleSpecification(1, "bundle:b", "wavlm_head:B", PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.5), "genuine", "tts")
    contents = {name: b"x" for name in B.MODEL_COMPONENTS}
    write_bundle(tmp / "b_source", b_spec, contents, extra={"stage": "development_uncalibrated"})
    other = calibration(tmp / "b_source")
    with pytest.raises(BundleError, match="Protocol A"):
        B.package_calibrated_wavlm_a(tmp / "b_source", tmp / "o_b", other)


def constant_scorer(value):
    return lambda _path: (lambda chunks: [value] * len(chunks))


def test_loader_applies_the_stored_mapping_and_refuses_a_threshold_override(built):
    tmp, source, meta, dest = built
    loaded = B.load_wavlm_bundle_a(dest, scorer_factory=constant_scorer(0.8))
    assert loaded.score_type == "calibrated_probability" and loaded.spec.decision_rule.threshold == 0.4 and loaded.calibrate is not None
    expected = B3.calibration_function(meta)(0.8)
    assert loaded.calibrate(0.8) == pytest.approx(expected)
    with pytest.raises(ContractError, match="override"):
        B.load_wavlm_bundle_a(dest, threshold=0.3, scorer_factory=constant_scorer(0.8))
    rejected = B.package_calibrated_wavlm_a(source, tmp / "rejected", calibration(source, accepted=False, threshold=0.5))
    plain = B.load_wavlm_bundle_a(rejected, scorer_factory=constant_scorer(0.8))
    assert plain.calibrate is None and plain.score_type == "model_score"
    uncal = B.load_wavlm_bundle_a(source, scorer_factory=constant_scorer(0.8), threshold=0.3)
    assert uncal.spec.decision_rule.threshold == 0.3 and uncal.score_type == "model_score"


def test_loader_refuses_a_b_bundle_and_a_corrupted_component(built, tmp_path):
    tmp, source, meta, dest = built
    (dest / "model/head_weights.pt").write_bytes(b"corrupted")
    with pytest.raises(BundleError):
        B.load_wavlm_bundle_a(dest, scorer_factory=constant_scorer(0.5))
    b_spec = BundleSpecification(1, "bundle:b", "wavlm_head:B", PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.5), "genuine", "tts")
    write_bundle(tmp / "b_source", b_spec, {name: b"x" for name in B.MODEL_COMPONENTS}, extra={"stage": "development_uncalibrated"})
    with pytest.raises(BundleError, match="incompatible"):
        B.load_wavlm_bundle_a(tmp / "b_source", scorer_factory=constant_scorer(0.5))


def test_package_stage_builds_a_loadable_bundle_from_a_finished_run(tmp_path):
    recipe = make_project(tmp_path, scale=4.0, seed=11)
    run_dir = tmp_path / "run"
    run_audit(tmp_path, recipe, run_dir)
    W.fit_stage(tmp_path, recipe, run_dir)
    result = W.operating_stage(tmp_path, recipe, run_dir, bootstrap_resamples=10, plots=False)
    destination = Path(B.package_stage(tmp_path, recipe, run_dir))
    loaded = B.load_wavlm_bundle_a(destination, scorer_factory=constant_scorer(0.7))
    thr = result["threshold"]
    assert loaded.score_type == thr["score_type"] and loaded.spec.decision_rule.threshold == thr["decision_rule"]["threshold"]
    assert (loaded.calibrate is not None) == thr["calibration_accepted"]
    cal = json.loads((destination / B.CALIBRATION_COMPONENT).read_text(encoding="utf-8"))
    assert cal["protocol"] == "A" and cal["source"]["run_id"] == run_dir.name and cal["source"]["split_sha256"] == SPLIT
    handoff = json.loads((run_dir / "handoff/bundle_packaging.json").read_text(encoding="utf-8"))
    assert handoff["model_components_equal_source_bytes"] is True and handoff["task"] == "R4-N1-WavLM bundle packaging"
    with pytest.raises(BundleError, match="overwrite"):
        B.package_stage(tmp_path, recipe, run_dir)
    # a fit changed after freezing is refused before anything is written
    run2 = tmp_path / "run2"
    run_audit(tmp_path, recipe, run2)
    W.fit_stage(tmp_path, recipe, run2)
    W.operating_stage(tmp_path, recipe, run2, bootstrap_resamples=10, plots=False)
    path = run2 / "fit/calibration.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"fit_role"', '"fit_role_"'), encoding="utf-8")
    recipe2 = copy.deepcopy(recipe)
    recipe2["bundle_destination"] = "bundles/out2"
    with pytest.raises(BundleError, match="frozen fit changed"):
        B.package_stage(tmp_path, recipe2, run2)
    assert not (tmp_path / "bundles/out2").exists()


# ---- B is untouched ---------------------------------------------------------------------------------------------------------------------------

def test_the_b_sources_this_work_imports_are_unchanged_in_the_gr3_lock():
    lock = (ROOT / "locks/B_lock_gr3.json").read_text(encoding="utf-8")
    for name in ("src/echoproof/evaluation/r3_n1_wavlm.py", "src/echoproof/evaluation/r3_n1_wavlm_operating.py", "src/echoproof/models/r3_n1_wavlm_bundle.py",
                 "src/echoproof/evaluation/threshold.py", "src/echoproof/evaluation/calibrate.py", "src/echoproof/evaluation/affine_calibration.py",
                 "src/echoproof/inference.py"):
        assert digest_in_lock(ROOT / name, lock), f"{name} differs from the GR3 lock"


# ---- notebook and scripts ----------------------------------------------------------------------------------------------------------------------

def test_notebook_parses_and_runs_every_stage_without_b_paths():
    import ast
    notebook = json.loads((ROOT / "notebooks/R4_N1_Colab_WavLM_Calibration_A.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    for stage in ("'audit'", "'fit'", "'operating'", "'package'", "r4_n1_wavlm_replay.py"):
        assert stage in code
    assert "r4_n1_wavlm_calibration_a.json" in code and "transformers==5.22.0" in code
    assert "protocol_B" not in code and "r3_n1_wavlm_head_B_calibrated" not in code and "zip" not in code.lower()


def test_scripts_import_the_a_modules_only_and_never_the_b_loader():
    for name in ("scripts/r4_n1_wavlm_calibrate.py", "scripts/r4_n1_wavlm_replay.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "r4_n1_wavlm" in text and "r3_n1_wavlm_bundle" not in text and "calibration_b.json" not in text
