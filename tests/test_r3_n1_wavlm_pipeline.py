"""R3-N1 WavLM pipeline on a synthetic project: audit, mapping selection, the frozen fit, the acceptance rule and the single operating pass.

Numpy, scipy, pyarrow and matplotlib only; no model, no audio, no transformers.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from echoproof.artifacts import file_digest
from echoproof.contracts import DecisionRule
from echoproof.evaluation import r3_n1_wavlm as W
from echoproof.evaluation import r3_n1_wavlm_operating as O

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

COMPONENTS = W.COMPONENTS
SPEAKERS = {"selection": 4, "calibration": 6, "operating": 5}
PER_SPEAKER = 3                       # recordings per component per speaker


def logit(p):
    return np.log(p / (1 - p))


def make_exports(directory: Path, scale=3.0, shift=0.0, seed=0):
    """Chunk-level exports whose recording scores are exactly the mean of their chunk probabilities."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    k = 0
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
                        z = scale * (base + rng.normal(0, 1.2)) + shift
                        p = float(np.clip(1 / (1 + np.exp(-z)), 0.0, 1.0))
                        probs.append(p)
                        chunks.append({"chunk_id": f"chunk:B:{rid}:{c * 32000}", "recording_id": rid, "group_id": group, "label": label, "component": comp,
                                       "validation_role": role, "tts_probability": p})
                    recs.append({"recording_id": rid, "group_id": group, "label": label, "component": comp, "validation_role": role,
                                 "raw_score": float(np.mean(probs)), "chunk_count": 2})
                    k += 1
        pq.write_table(pa.Table.from_pylist(recs, schema=pa.schema([("recording_id", pa.string()), ("group_id", pa.string()), ("label", pa.int64()),
                                                                       ("component", pa.string()), ("validation_role", pa.string()), ("raw_score", pa.float64()),
                                                                       ("chunk_count", pa.int64())])), directory / f"predictions_recordings_{role}.parquet")
        pq.write_table(pa.Table.from_pylist(chunks), directory / f"predictions_chunks_{role}.parquet")


def make_project(root: Path, **kw):
    inputs = root / "reports/r3/r3_u1_B"
    make_exports(inputs, **kw)
    predictions = {f"predictions_{kind}_{role}.parquet": file_digest(inputs / f"predictions_{kind}_{role}.parquet") for kind in ("chunks", "recordings") for role in W.ROLES}
    (inputs / "evidence.json").write_text(json.dumps({"output_sha256": predictions}), encoding="utf-8")
    selection = {"selected": "mean_std", "selected_epoch": 14, "checkpoint_sha256": {"mean_std": "a" * 64}, "split_sha256": "b" * 64,
                 "selection_macro_f1": {"mean_std": 0.0}}
    sel_rows = W.read_recordings(inputs, "selection")
    y, s, *_ = W.arrays(sel_rows)
    from echoproof.evaluation.metrics import compute_metrics
    selection["selection_macro_f1"]["mean_std"] = compute_metrics(y, s, 0.5)["macro_f1"]
    (inputs / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    bundle = root / "bundles/src"
    (bundle / "model").mkdir(parents=True)
    (bundle / "model/wavlm_head.json").write_text(json.dumps({"source_checkpoint_sha256": "a" * 64, "pooling": "mean_std", "split_sha256": "b" * 64}), encoding="utf-8")
    components = {"model/wavlm_head.json": {"sha256": file_digest(bundle / "model/wavlm_head.json"), "bytes": 1}}
    (bundle / "bundle.json").write_text(json.dumps({"components": components, "model_version": "wavlm_head:B", "extra": {"stage": "development_uncalibrated"}}), encoding="utf-8")
    recipe = json.loads(Path(__file__).resolve().parents[1].joinpath("configs/r3_n1_wavlm_calibration_b.json").read_text(encoding="utf-8"))
    recipe["inputs"].update(directory="reports/r3/r3_u1_B", selection_json_sha256=file_digest(inputs / "selection.json"),
                            evidence_json_sha256=file_digest(inputs / "evidence.json"), predictions_sha256=predictions)
    recipe["model"].update(checkpoint_sha256="a" * 64, split_sha256="b" * 64, source_bundle="bundles/src", source_bundle_json_sha256=file_digest(bundle / "bundle.json"))
    for role, speakers in SPEAKERS.items():
        recipe["roles"][role] = {"recordings": speakers * 4 * PER_SPEAKER, "speakers": speakers, "chunks": speakers * 4 * PER_SPEAKER * 2, "per_component": speakers * PER_SPEAKER}
    for role in ("calibration", "operating"):
        rows = W.read_recordings(inputs, role)
        y, s, *_ = W.arrays(rows)
        m = compute_metrics(y, s, 0.5)
        recipe["expected_raw_summary_at_0_5"][role] = {"macro_f1": round(m["macro_f1"], 4), "genuine_fpr": round(m["fpr"], 4), "eer": round(m["eer"], 4),
                                                       "roc_auc": round(m["roc_auc"], 4)}
    for role in W.ROLES:
        _, s, *_ = W.arrays(W.read_recordings(inputs, role))
        recipe["expected_saturation"][role] = W.saturation(s, 1e-7)
    recipe["duration_matched"]["file"] = "data/none.json"
    return recipe


@pytest.fixture()
def project(tmp_path):
    return tmp_path, make_project(tmp_path)


# ---- pure pieces ------------------------------------------------------------------------------------------------------------------------------

def test_select_mapping_is_simplest_first_within_one_standard_error():
    flat = [0.30] * 12
    noisy = [0.20, 0.40, 0.25, 0.35, 0.22, 0.38, 0.26, 0.34, 0.24, 0.36, 0.23, 0.37]      # mean 0.30, large spread
    assert W.select_mapping({"identity": flat, "temperature": [0.30] * 12, "affine": [0.30] * 12}, {})["selected"] == "identity"
    better_temp = [0.25] * 12
    affine_spread = [0.1999 + 0.1 * (i % 2) for i in range(12)]                                # mean 0.2499, standard error about 0.0144
    r = W.select_mapping({"identity": [0.40] * 12, "temperature": better_temp, "affine": affine_spread}, {})
    assert r["selected"] == "temperature" and r["best"] == "affine"                         # affine is best, but temperature is within one standard error
    r = W.select_mapping({"identity": [0.40] * 12, "temperature": better_temp, "affine": [0.2499] * 12}, {})
    assert r["selected"] == "affine"                                                          # a zero standard error leaves no tolerance
    r = W.select_mapping({"identity": flat, "temperature": [0.29] * 12, "affine": [0.10] * 12}, {})
    assert r["selected"] == "affine"
    spread = np.array(noisy) - 0.10
    r = W.select_mapping({"identity": [0.5] * 12, "temperature": [0.30] * 12, "affine": list(spread + 0.0)}, {})
    assert r["standard_error_of_best"] > 0 and r["selected"] in ("temperature", "affine")


def test_failed_candidates_are_excluded_and_identity_is_the_fallback():
    losses = {"identity": [0.5] * 12, "temperature": [0.1] * 12, "affine": [0.05] * 12}
    r = W.select_mapping(losses, {"affine": "fit failed"})
    assert r["selected"] == "temperature" and "affine" in r["excluded"]
    r = W.select_mapping({"identity": [0.5] * 12}, {"temperature": "x", "affine": "y"})
    assert r["selected"] == "identity"
    with pytest.raises(W.WavLMCalibrationError):
        W.select_mapping({"temperature": [0.1] * 12}, {})


def test_acceptance_rule_cases():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2.0, 4000)
    y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-z))).astype(int)                # the truth is sigmoid(z)
    raw = 1 / (1 + np.exp(-3 * z))                                                 # overconfident by a factor of three
    cal = 1 / (1 + np.exp(-z))
    assert W.acceptance("temperature", True, y, raw, cal)["accepted"] is True
    worse = W.acceptance("temperature", True, y, cal, raw)
    assert worse["accepted"] is False and "log loss not strictly lower" in worse["reason"]
    assert W.acceptance("temperature", True, y, raw, raw)["accepted"] is False            # a tie is rejected
    assert W.acceptance("temperature", False, y, raw, cal)["accepted"] is False            # an invalid fit is rejected
    assert W.acceptance("identity", True, y, raw, raw)["accepted"] is False
    brier_worse = np.clip(raw * 0.0 + np.where(y == 1, 0.55, 0.45), 0, 1)
    assert W.acceptance("temperature", True, y, raw, brier_worse)["accepted"] is False


def test_decision_metrics_and_all_negative_rule():
    y = np.array([0, 0, 1, 1, 1])
    m = W.decision_metrics(y, [0, 1, 1, 0, 1])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 1)
    none = W.decision_metrics(y, [0] * 5)
    assert none["tts_recall"] == 0.0 and none["genuine_fpr"] == 0.0
    rule = DecisionRule(1, "all_negative", None)
    assert list(W.decide(rule, [0.9, 0.1])) == [0, 0]


def test_raw_equivalent_thresholds_reproduce_the_mapped_decision():
    scores = np.linspace(0.001, 0.999, 999)
    for method, params in (("temperature", {"temperature": 1.7}), ("affine", {"a": 0.6, "b": 0.8})):
        mapped = np.array(W.apply_mapping(method, params, scores, 1e-7))
        tau = 0.42
        raw_t = W.raw_equivalent_threshold(method, params, tau, 1e-7)
        assert list(mapped >= tau) == list(scores >= raw_t)
    assert W.raw_equivalent_threshold("identity", {}, 0.4, 1e-7) is None


def test_temperature_checks_agree_with_the_shared_fit():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=400) < 0.7).astype(int)
    s = 1 / (1 + np.exp(-(3 * (2 * y - 1) + rng.normal(0, 2.5, 400)) * 2))
    t = W.fit_candidate("temperature", y, s, {"epsilon": 1e-7, "temperature": {"bounds": [0.01, 50.0], "bound_tolerance": 1e-6}})
    assert t["ok"]
    check = W.temperature_checks(y, s, t["params"]["temperature"], [0.01, 50.0], 1e-7)
    assert check["agrees"] and check["inside_bounds"]


# ---- stages on the synthetic project ----------------------------------------------------------------------------------------------------------

def run_audit(root, recipe, run_dir):
    return W.audit(root, recipe, run_dir, join_manifests=False)


def test_audit_passes_on_clean_exports_and_fails_on_a_changed_file(tmp_path):
    recipe = make_project(tmp_path)
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    result = run_audit(tmp_path, recipe, run_dir)
    assert result["passed"] and result["operating_role_used_for_fitting"] is False and all(result["checks"].values())
    # tamper with one export: the hash check must fail and the failure must still be written
    target = tmp_path / "reports/r3/r3_u1_B/predictions_chunks_operating.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    run2 = tmp_path / "run2"
    run2.mkdir()
    with pytest.raises(W.WavLMCalibrationError):
        run_audit(tmp_path, recipe, run2)
    assert json.loads((run2 / "input_audit.json").read_text(encoding="utf-8"))["passed"] is False


def test_audit_rejects_a_recording_score_that_is_not_the_chunk_mean(tmp_path):
    recipe = make_project(tmp_path)
    path = tmp_path / "reports/r3/r3_u1_B/predictions_recordings_calibration.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["raw_score"] = min(1.0, rows[0]["raw_score"] + 0.01)
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_table(path).schema), path)
    recipe["inputs"]["predictions_sha256"]["predictions_recordings_calibration.parquet"] = file_digest(path)
    ev = json.loads((tmp_path / "reports/r3/r3_u1_B/evidence.json").read_text(encoding="utf-8"))
    ev["output_sha256"] = recipe["inputs"]["predictions_sha256"]
    (tmp_path / "reports/r3/r3_u1_B/evidence.json").write_text(json.dumps(ev), encoding="utf-8")
    recipe["inputs"]["evidence_json_sha256"] = file_digest(tmp_path / "reports/r3/r3_u1_B/evidence.json")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(W.WavLMCalibrationError):
        run_audit(tmp_path, recipe, run_dir)


def test_fit_never_reads_the_operating_role_and_freezes_the_mapping(tmp_path, monkeypatch):
    recipe = make_project(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_audit(tmp_path, recipe, run_dir)
    asked = []
    original = W.read_recordings
    monkeypatch.setattr(W, "read_recordings", lambda d, role: (asked.append(role), original(d, role))[1])
    fit = W.fit_stage(tmp_path, recipe, run_dir)
    assert asked == ["calibration"] and fit["operating_predictions_read"] is False
    assert fit["selected_mapping"] in W.ORDER
    done = json.loads((run_dir / "fit/stage_complete.json").read_text(encoding="utf-8"))
    assert all(file_digest(run_dir / "fit" / n) == h for n, h in done["files"].items())
    comparison = json.loads((run_dir / "fit/loso_comparison.json").read_text(encoding="utf-8"))
    assert len(comparison["folds"]) == SPEAKERS["calibration"] and comparison["selection"]["selected"] == fit["selected_mapping"]
    assert set(comparison["per_speaker_log_loss"]) == set(W.ORDER)
    with pytest.raises(FileExistsError):
        W.fit_stage(tmp_path, recipe, run_dir)                                     # the fit directory is never reused


def test_fit_refuses_without_a_passing_audit_or_with_a_different_recipe(tmp_path):
    recipe = make_project(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        W.fit_stage(tmp_path, recipe, run_dir)
    run_audit(tmp_path, recipe, run_dir)
    other = copy.deepcopy(recipe)
    other["seed"] = 7
    with pytest.raises(W.WavLMCalibrationError):
        W.fit_stage(tmp_path, other, run_dir)


@pytest.mark.parametrize("scale", [3.0, 1.0])
def test_operating_stage_end_to_end_is_consistent_and_opens_the_role_once(tmp_path, scale):
    recipe = make_project(tmp_path, scale=scale, seed=3)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_audit(tmp_path, recipe, run_dir)
    fit = W.fit_stage(tmp_path, recipe, run_dir)
    result = O.operating_stage(tmp_path, recipe, run_dir, bootstrap_resamples=30, plots=False)
    thr, ev = result["threshold"], result["evidence"]
    assert thr["calibration_accepted"] == thr["acceptance"]["accepted"] and thr["selected_mapping"] == fit["selected_mapping"]
    assert thr["score_type"] == ("calibrated_probability" if thr["calibration_accepted"] else "model_score")
    assert thr["deployed_scores"] == ("calibrated" if thr["calibration_accepted"] else "raw")
    assert fit["selected_mapping"] != "identity" or not thr["calibration_accepted"]            # identity can never be accepted
    out = run_dir / "operating"
    preds = pq.read_table(out / "operating_predictions.parquet").to_pylist()
    rule = DecisionRule(1, thr["decision_rule"]["kind"], thr["decision_rule"]["threshold"])
    assert all(rule.decide(r["deployed_score"]) == r["decision"] for r in preds) and len(preds) == recipe["roles"]["operating"]["recordings"]
    if not thr["calibration_accepted"]:
        assert all(r["deployed_score"] == r["raw_score"] for r in preds)
    metrics = result["metrics"]
    assert "A_raw_at_0_5" in metrics["rows"] and "C_raw_at_raw_optimum" in metrics["rows"]
    assert metrics["rows"]["A_raw_at_0_5"]["rule"]["threshold"] == 0.5
    # the chosen raw optimum is never worse than 0.5 on the same scores (it maximises macro F1 over all partitions)
    assert metrics["rows"]["C_raw_at_raw_optimum"]["full"]["macro_f1"] >= metrics["rows"]["A_raw_at_0_5"]["full"]["macro_f1"] - 1e-12
    assert all(file_digest(out / n) == h for n, h in ev["outputs"].items())
    assert ev["fit_on_operating"] is False and ev["test_data_read"] is False
    with pytest.raises(FileExistsError):
        O.operating_stage(tmp_path, recipe, run_dir, bootstrap_resamples=30, plots=False)       # the operating role is opened once


def test_operating_stage_refuses_a_fit_that_changed_after_freezing(tmp_path):
    recipe = make_project(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_audit(tmp_path, recipe, run_dir)
    W.fit_stage(tmp_path, recipe, run_dir)
    path = run_dir / "fit/calibration.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"fit_role"', '"fit_role_"'), encoding="utf-8")
    with pytest.raises(W.WavLMCalibrationError):
        O.operating_stage(tmp_path, recipe, run_dir, bootstrap_resamples=10, plots=False)
    assert not (run_dir / "operating").exists()                                                   # nothing was opened


def test_an_overconfident_model_is_calibrated_and_accepted(tmp_path):
    recipe = make_project(tmp_path, scale=4.0, seed=11)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_audit(tmp_path, recipe, run_dir)
    fit = W.fit_stage(tmp_path, recipe, run_dir)
    assert fit["selected_mapping"] in ("temperature", "affine")
    result = O.operating_stage(tmp_path, recipe, run_dir, bootstrap_resamples=20, plots=False)
    assert result["threshold"]["acceptance"]["valid_fit"] is True
    assert result["metrics"]["probability"]["selected"]["log_loss"] < result["metrics"]["probability"]["raw"]["log_loss"] or not result["threshold"]["calibration_accepted"]
