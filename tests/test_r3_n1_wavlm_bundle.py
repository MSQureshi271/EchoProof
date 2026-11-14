"""R3-N1 WavLM calibrated bundle: packaging, validation, the loader, the single application of the mapping, the dispatcher, the CLI and the app helpers.

Synthetic bundles with dummy model files and an injected fake scorer: no torch, no transformers, no audio model.
"""
import copy
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from echoproof import app_support, cli
from echoproof import inference as inf
from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle, write_bundle
from echoproof.contracts import BundleSpecification, ContractError, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.evaluation.affine_calibration import apply_affine
from echoproof.evaluation.calibrate import apply_temperature
from echoproof.models import r3_n1_wavlm_bundle as B

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r1_n2_inference import wav_bytes  # noqa: E402

CHECKPOINT = "c" * 64
SOURCE_JSON_SHA = "d" * 64


def make_source(tmp_path: Path) -> Path:
    """A stand-in for the uncalibrated development bundle: same component names, tiny dummy contents."""
    contents = {name: (f"dummy {name}").encode() for name in B.MODEL_COMPONENTS}
    contents["model/wavlm_head.json"] = json.dumps({"source_checkpoint_sha256": CHECKPOINT, "pooling": "mean_std"}).encode()
    spec = BundleSpecification(1, "bundle:test-source", B.MODEL_VERSION, PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.5), "genuine", "tts")
    write_bundle(tmp_path / "source", spec, contents, extra={"stage": "development_uncalibrated"})
    return tmp_path / "source"


def calibration(source: Path, *, method="affine_on_log_odds", params=None, accepted=True, threshold=0.4):
    params = params if params is not None else ({"a": math.exp(0.1), "b": 0.9, "u": 0.1} if method == "affine_on_log_odds" else {"temperature": 1.5})
    return {"schema": B.CALIBRATION_SCHEMA, "model": "wavlm_head", "method": method, "parameters": params, "epsilon": 1e-7, "calibration_accepted": accepted,
            "score_type": "calibrated_probability" if accepted else "model_score", "operating_decision_rule": {"kind": "threshold", "threshold": threshold},
            "raw_equivalent_threshold": None, "source": {"source_bundle_json_sha256": file_digest(source / "bundle.json"), "source_checkpoint_sha256": CHECKPOINT,
                                                         "split_sha256": "e" * 64, "recipe_sha256": "f" * 64, "fit_calibration_json_sha256": "1" * 64,
                                                         "operating_evidence_sha256": "2" * 64}}


def constant_scorer(value):
    return lambda _path: (lambda chunks: [value] * len(chunks))


def varying_scorer():
    return lambda _path: (lambda chunks: [float(min(0.99, max(0.01, 0.5 + 0.4 * np.mean(c)))) for c in chunks])


@pytest.fixture()
def built(tmp_path):
    source = make_source(tmp_path)
    meta = calibration(source)
    dest = B.package_calibrated_wavlm(source, tmp_path / "calibrated", meta, bundle_id="bundle:test-calibrated")
    return tmp_path, source, meta, dest


# ---- packaging -------------------------------------------------------------------------------------------------------------------------------

def test_packaging_copies_the_model_bytes_and_serialises_the_rule(built):
    tmp, source, meta, dest = built
    original, written = load_bundle(source), load_bundle(dest)
    for name in B.MODEL_COMPONENTS:
        assert file_digest(written.components[name]) == file_digest(original.components[name])
    assert set(written.components) == set(B.MODEL_COMPONENTS) | {B.CALIBRATION_COMPONENT}
    assert written.spec.decision_rule == DecisionRule(1, "threshold", 0.4) and written.spec.model_version == B.MODEL_VERSION
    assert written.raw_metadata["extra"]["stage"] == "r3_n1_wavlm_development" and written.raw_metadata["extra"]["promoted"] is False
    assert (source / "bundle.json").exists() and load_bundle(source).spec.decision_rule.threshold == 0.5            # the source is untouched


def test_packaging_refuses_overwrite_and_mismatched_sources(built, tmp_path):
    tmp, source, meta, dest = built
    with pytest.raises(BundleError, match="overwrite"):
        B.package_calibrated_wavlm(source, dest, meta, bundle_id="bundle:x")
    other = copy.deepcopy(meta)
    other["source"]["source_bundle_json_sha256"] = "9" * 64
    with pytest.raises(BundleError, match="source bundle differs"):
        B.package_calibrated_wavlm(source, tmp / "o1", other, bundle_id="bundle:x")
    other = copy.deepcopy(meta)
    other["source"]["source_checkpoint_sha256"] = "9" * 64
    with pytest.raises(BundleError, match="different checkpoint"):
        B.package_calibrated_wavlm(source, tmp / "o2", other, bundle_id="bundle:x")
    assert not (tmp / "o1").exists() and not (tmp / "o2").exists()


@pytest.mark.parametrize("edit,message", [
    (lambda m: m.update(schema="other"), "schema"),
    (lambda m: m.update(method="isotonic"), "unknown calibration method"),
    (lambda m: m.update(epsilon=0.7), "epsilon"),
    (lambda m: m.update(calibration_accepted="yes"), "boolean"),
    (lambda m: m.update(score_type="model_score"), "inconsistent"),
    (lambda m: m["parameters"].update(b=7.0), "outside"),
    (lambda m: m["parameters"].update(a=2.0), "invalid affine"),
    (lambda m: m["parameters"].update(u=float("nan")), "invalid affine"),
    (lambda m: m.update(operating_decision_rule={"kind": "threshold", "threshold": 1.5}), "decision rule"),
    (lambda m: m.update(operating_decision_rule={"kind": "all_negative", "threshold": 0.3}), "decision rule"),
    (lambda m: m["source"].update(recipe_sha256="short"), "source hash"),
])
def test_calibration_validation_rejects_each_inconsistency(tmp_path, edit, message):
    meta = calibration(make_source(tmp_path))
    edit(meta)
    with pytest.raises(BundleError, match=message):
        B.validate_calibration(meta)


def test_temperature_identity_and_acceptance_validation(tmp_path):
    source = make_source(tmp_path)
    B.validate_calibration(calibration(source, method="temperature_on_log_odds"))
    for t in (0.005, 60.0, float("inf")):
        with pytest.raises(BundleError):
            B.validate_calibration(calibration(source, method="temperature_on_log_odds", params={"temperature": t}))
    B.validate_calibration(calibration(source, method="identity", params={}, accepted=False))
    with pytest.raises(BundleError, match="identity cannot be accepted"):
        B.validate_calibration(calibration(source, method="identity", params={}, accepted=True))
    with pytest.raises(BundleError, match="identity takes no parameters"):
        B.validate_calibration(calibration(source, method="identity", params={"temperature": 1.0}, accepted=False))


# ---- loader ----------------------------------------------------------------------------------------------------------------------------------

def test_loader_installs_the_calibration_only_when_accepted(tmp_path):
    source = make_source(tmp_path)
    accepted = B.package_calibrated_wavlm(source, tmp_path / "acc", calibration(source), bundle_id="bundle:acc")
    rejected = B.package_calibrated_wavlm(source, tmp_path / "rej", calibration(source, accepted=False), bundle_id="bundle:rej")
    a = B.load_wavlm_bundle(accepted, scorer_factory=constant_scorer(0.5))
    r = B.load_wavlm_bundle(rejected, scorer_factory=constant_scorer(0.5))
    assert a.calibrate is not None and a.score_type == "calibrated_probability"
    assert r.calibrate is None and r.score_type == "model_score"
    assert a.calibrate(0.3) == pytest.approx(apply_affine([0.3], math.exp(0.1), 0.9)[0], abs=1e-15)
    t = B.package_calibrated_wavlm(source, tmp_path / "tmp", calibration(source, method="temperature_on_log_odds"), bundle_id="bundle:t")
    assert B.load_wavlm_bundle(t, scorer_factory=constant_scorer(0.5)).calibrate(0.3) == pytest.approx(apply_temperature([0.3], 1.5)[0], abs=1e-15)


def test_loader_refuses_overrides_tampering_and_inconsistent_components(built, tmp_path):
    tmp, source, meta, dest = built
    with pytest.raises(ContractError, match="threshold override"):
        B.load_wavlm_bundle(dest, scorer_factory=constant_scorer(0.5), threshold=0.6)
    # tamper with the calibration component: the bundle hash check refuses it before anything is used
    copy_dir = tmp_path / "tampered"
    import shutil
    shutil.copytree(dest, copy_dir)
    path = copy_dir / "calibration/calibration.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"b": 0.9', '"b": 0.1'), encoding="utf-8")
    with pytest.raises(BundleError):
        B.load_wavlm_bundle(copy_dir, scorer_factory=constant_scorer(0.5))
    # a bundle whose stored rule disagrees with its calibration component
    odd = calibration(source)
    odd["operating_decision_rule"] = {"kind": "threshold", "threshold": 0.45}
    spec = BundleSpecification(1, "bundle:odd", B.MODEL_VERSION, PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.4), "genuine", "tts")
    comps = {n: load_bundle(source).components[n] for n in B.MODEL_COMPONENTS}
    comps[B.CALIBRATION_COMPONENT] = json.dumps(odd).encode()
    write_bundle(tmp_path / "odd", spec, comps, extra={"stage": "r3_n1_wavlm_development"})
    with pytest.raises(BundleError, match="decision rule differs"):
        B.load_wavlm_bundle(tmp_path / "odd", scorer_factory=constant_scorer(0.5))
    # an extra component, or a wrong checkpoint in the model metadata
    comps2 = dict(comps)
    comps2[B.CALIBRATION_COMPONENT] = json.dumps(calibration(source)).encode()
    comps2["model/extra.bin"] = b"x"
    spec2 = BundleSpecification(1, "bundle:extra", B.MODEL_VERSION, PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.4), "genuine", "tts")
    write_bundle(tmp_path / "extra", spec2, comps2, extra={"stage": "r3_n1_wavlm_development"})
    with pytest.raises(BundleError, match="unexpected components"):
        B.load_wavlm_bundle(tmp_path / "extra", scorer_factory=constant_scorer(0.5))
    wrong = calibration(source)
    wrong["source"]["source_checkpoint_sha256"] = "7" * 64
    comps3 = {n: load_bundle(source).components[n] for n in B.MODEL_COMPONENTS}
    comps3[B.CALIBRATION_COMPONENT] = json.dumps(wrong).encode()
    write_bundle(tmp_path / "wrong", spec2, comps3, extra={"stage": "r3_n1_wavlm_development"})
    with pytest.raises(BundleError, match="different checkpoint"):
        B.load_wavlm_bundle(tmp_path / "wrong", scorer_factory=constant_scorer(0.5))


def test_the_uncalibrated_development_bundle_still_loads_as_a_model_score_and_allows_an_override(tmp_path):
    source = make_source(tmp_path)
    plain = B.load_wavlm_bundle(source, scorer_factory=constant_scorer(0.5))
    assert plain.calibrate is None and plain.score_type == "model_score" and plain.spec.decision_rule.threshold == 0.5
    assert B.load_wavlm_bundle(source, scorer_factory=constant_scorer(0.5), threshold=0.7).spec.decision_rule.threshold == 0.7
    spec = BundleSpecification(1, "bundle:no-stage", B.MODEL_VERSION, PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.5), "genuine", "tts")
    comps = {n: load_bundle(source).components[n] for n in B.MODEL_COMPONENTS}
    write_bundle(tmp_path / "nostage", spec, comps, extra={"stage": "something_else"})
    with pytest.raises(BundleError, match="uncalibrated development bundle"):
        B.load_wavlm_bundle(tmp_path / "nostage", scorer_factory=constant_scorer(0.5))


# ---- the mapping is applied exactly once, in the shared inference path -----------------------------------------------------------------------------

def test_analyze_applies_the_stored_mapping_once_and_the_rule_decides(built):
    tmp, source, meta, dest = built
    bundle = B.load_wavlm_bundle(dest, scorer_factory=varying_scorer())
    analysis = inf.analyze(wav_bytes(9), bundle)
    assert analysis.result.status == "success"
    raw = float(np.mean(analysis.chunk_scores))
    assert analysis.raw_score == pytest.approx(raw, abs=1e-15)
    expected = apply_affine([analysis.raw_score], math.exp(0.1), 0.9)[0]
    assert analysis.result.score == pytest.approx(expected, abs=1e-15) and analysis.result.score != pytest.approx(analysis.raw_score, abs=1e-6)
    assert analysis.result.score_type == "calibrated_probability"
    assert analysis.result.assessment == ("synthetic" if expected >= 0.4 else "genuine")


def test_a_score_equal_to_the_threshold_is_synthetic(tmp_path):
    source = make_source(tmp_path)
    first = B.load_wavlm_bundle(B.package_calibrated_wavlm(source, tmp_path / "p1", calibration(source), bundle_id="bundle:p1"), scorer_factory=varying_scorer())
    score = inf.analyze(wav_bytes(9), first).result.score
    exact = B.package_calibrated_wavlm(source, tmp_path / "p2", calibration(source, threshold=score), bundle_id="bundle:p2")
    bundle = B.load_wavlm_bundle(exact, scorer_factory=varying_scorer())
    assert inf.analyze(wav_bytes(9), bundle).result.assessment == "synthetic"                       # score >= threshold
    above = B.package_calibrated_wavlm(source, tmp_path / "p3", calibration(source, threshold=min(1.0, score + 1e-9)), bundle_id="bundle:p3")
    assert inf.analyze(wav_bytes(9), B.load_wavlm_bundle(above, scorer_factory=varying_scorer())).result.assessment == "genuine"


def test_a_rejected_calibration_reports_the_raw_model_score(tmp_path):
    source = make_source(tmp_path)
    dest = B.package_calibrated_wavlm(source, tmp_path / "rej", calibration(source, accepted=False, threshold=0.3), bundle_id="bundle:rej")
    analysis = inf.analyze(wav_bytes(9), B.load_wavlm_bundle(dest, scorer_factory=varying_scorer()))
    assert analysis.result.score_type == "model_score" and analysis.result.score == analysis.raw_score
    assert "Model score" in app_support.present(analysis.result)["lines"][0]
    assert "Calibrated probability" in app_support.present(inf.analyze(wav_bytes(9), B.load_wavlm_bundle(
        B.package_calibrated_wavlm(source, tmp_path / "acc", calibration(source), bundle_id="bundle:acc"), scorer_factory=varying_scorer()))
    .result)["lines"][0]


def test_an_all_negative_rule_never_flags(tmp_path):
    source = make_source(tmp_path)
    meta = calibration(source)
    meta["operating_decision_rule"] = {"kind": "all_negative", "threshold": None}
    dest = B.package_calibrated_wavlm(source, tmp_path / "neg", meta, bundle_id="bundle:neg")
    assert inf.analyze(wav_bytes(9), B.load_wavlm_bundle(dest, scorer_factory=constant_scorer(0.99))).result.assessment == "genuine"


# ---- dispatcher, CLI and application helpers -------------------------------------------------------------------------------------------------------

def test_dispatcher_routes_by_model_family(built, tmp_path, monkeypatch):
    tmp, source, meta, dest = built
    seen = {}
    monkeypatch.setattr(B, "load_wavlm_bundle", lambda path, **kw: seen.update(wavlm=(Path(path), kw)) or "wavlm-loaded")
    monkeypatch.setattr(inf, "load_lr_bundle", lambda path, threshold=None: seen.update(lr=(Path(path), threshold)) or "lr-loaded")
    assert inf.load_inference_bundle(dest) == "wavlm-loaded" and seen["wavlm"] == (dest, {"threshold": None})
    assert inf.load_inference_bundle(dest, threshold=0.7) == "wavlm-loaded" and seen["wavlm"][1] == {"threshold": 0.7}
    legacy = tmp_path / "legacy_model"
    legacy.mkdir()
    assert inf.load_inference_bundle(legacy, threshold=0.2) == "lr-loaded" and seen["lr"] == (legacy, 0.2)
    lr_spec = BundleSpecification(1, "bundle:lr", "logistic_regression:B", PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.5), "genuine", "tts")
    write_bundle(tmp_path / "lrb", lr_spec, {"model/weights.npz": b"x", "model/lr_model.json": b"{}"})
    assert inf.load_inference_bundle(tmp_path / "lrb") == "lr-loaded"
    unknown = BundleSpecification(1, "bundle:u", "mystery:B", PREPROCESSING_HASH, DecisionRule(1, "threshold", 0.5), "genuine", "tts")
    write_bundle(tmp_path / "unk", unknown, {"model/x": b"x"})
    with pytest.raises(ContractError, match="no inference loader"):
        inf.load_inference_bundle(tmp_path / "unk")
    with pytest.raises(FileNotFoundError):
        inf.load_inference_bundle(tmp_path / "missing")


def test_cli_passes_no_threshold_by_default_and_matches_the_library(built, tmp_path, monkeypatch, capsys):
    tmp, source, meta, dest = built
    calls = []
    loaded = B.load_wavlm_bundle(dest, scorer_factory=varying_scorer())
    monkeypatch.setattr(inf, "load_inference_bundle", lambda path, threshold=None: (calls.append(threshold), loaded)[1])
    audio = tmp_path / "a.wav"
    audio.write_bytes(wav_bytes(9))
    assert cli.main(["predict", "--bundle", str(dest), "--audio", str(audio)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert calls == [None]
    assert out["score"] == inf.predict_recording(wav_bytes(9), loaded).score and out["score_type"] == "calibrated_probability"
    assert cli.main(["predict", "--bundle", str(dest), "--audio", str(audio), "--threshold", "0.9"]) == 0
    capsys.readouterr()
    assert calls == [None, 0.9]
    assert cli.main(["predict", "--bundle", str(dest), "--audio", str(audio), "--threshold", "1.5"]) == 2


def test_app_diagnostics_do_not_change_the_result_contract(built):
    tmp, source, meta, dest = built
    bundle = B.load_wavlm_bundle(dest, scorer_factory=varying_scorer())
    analysis = inf.analyze(wav_bytes(9), bundle)
    d = app_support.diagnostics(analysis, bundle)
    assert d["raw_recording_score"] == analysis.raw_score and d["reported_score"] == analysis.result.score and d["calibrated"] is True
    assert d["decision_rule"] == {"kind": "threshold", "threshold": 0.4} and d["chunks_scored"] == len(analysis.chunk_scores)
    full = json.loads(app_support.result_json_with_diagnostics(analysis, bundle))
    plain = json.loads(app_support.result_json(analysis.result))
    assert {k: v for k, v in full.items() if k != "diagnostics"} == plain and full["diagnostics"] == json.loads(json.dumps(d))
