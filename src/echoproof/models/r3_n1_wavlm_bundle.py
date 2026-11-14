"""Calibrated development bundle for the selected R3-U1 WavLM head: packaging, schema validation and the inference loader.

The calibrated bundle holds the original `model/` components byte for byte plus one hashed `calibration/calibration.json`. `bundle.json` carries the
serialized decision rule; inference uses only that rule. The loader installs a calibration function only when the stored `calibration_accepted` flag is
true, and builds it from the same functions the fit used (`apply_temperature`, `apply_affine`), so evaluation and application cannot drift.
A bundle without a calibration component is accepted only if it is the uncalibrated development bundle (stage `development_uncalibrated`).
"""
from __future__ import annotations

import json
import math
import shutil
import warnings
from pathlib import Path

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle, write_bundle
from echoproof.contracts import BundleSpecification, ContractError, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.evaluation.affine_calibration import apply_affine
from echoproof.evaluation.calibrate import apply_temperature

CALIBRATION_SCHEMA = "echoproof_recording_calibration_v1"
CALIBRATION_COMPONENT = "calibration/calibration.json"
MODEL_COMPONENTS = ("model/encoder/config.json", "model/encoder/model.safetensors", "model/encoder/preprocessor_config.json", "model/head_weights.pt",
                    "model/wavlm_head.json")
METHODS = {"identity": "identity", "temperature": "temperature_on_log_odds", "affine": "affine_on_log_odds"}
MODEL_VERSION = "wavlm_head:B"
TEMPERATURE_BOUNDS = (0.01, 50.0)
AFFINE_U_BOUNDS, AFFINE_B_BOUNDS = (-4.0, 4.0), (-5.0, 5.0)
TRANSFORMERS_PIN = "5.22.0"


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_calibration(meta: dict) -> None:
    """Reject a calibration component that is malformed, out of bounds or inconsistent with itself."""
    if not isinstance(meta, dict) or meta.get("schema") != CALIBRATION_SCHEMA or meta.get("model") != "wavlm_head":
        raise BundleError("unsupported calibration schema")
    method, params = meta.get("method"), meta.get("parameters")
    if method not in METHODS.values() or not isinstance(params, dict):
        raise BundleError("unknown calibration method")
    eps = meta.get("epsilon")
    if not _finite(eps) or not 0 < eps < 0.5:
        raise BundleError("invalid epsilon")
    if method == "temperature_on_log_odds":
        t = params.get("temperature")
        if not _finite(t) or not TEMPERATURE_BOUNDS[0] < t < TEMPERATURE_BOUNDS[1]:
            raise BundleError("temperature outside its bounds")
    elif method == "affine_on_log_odds":
        a, b, u = params.get("a"), params.get("b"), params.get("u")
        if not (_finite(a) and _finite(b) and _finite(u)) or a <= 0 or not abs(a - math.exp(u)) < 1e-9 * max(1.0, a):
            raise BundleError("invalid affine parameters")
        if not AFFINE_U_BOUNDS[0] < u < AFFINE_U_BOUNDS[1] or not AFFINE_B_BOUNDS[0] < b < AFFINE_B_BOUNDS[1]:
            raise BundleError("affine parameters outside their bounds")
    elif params:
        raise BundleError("identity takes no parameters")
    accepted = meta.get("calibration_accepted")
    if not isinstance(accepted, bool):
        raise BundleError("calibration_accepted must be a boolean")
    if accepted and method == "identity":
        raise BundleError("identity cannot be accepted as a calibration")
    if meta.get("score_type") != ("calibrated_probability" if accepted else "model_score"):
        raise BundleError("score type is inconsistent with the acceptance flag")
    rule = meta.get("operating_decision_rule")
    if not isinstance(rule, dict) or rule.get("kind") not in ("threshold", "all_negative"):
        raise BundleError("missing operating decision rule")
    try:
        DecisionRule(1, rule["kind"], rule.get("threshold"))
    except ContractError as exc:
        raise BundleError(f"invalid operating decision rule: {exc}") from exc
    source = meta.get("source", {})
    for key in ("source_bundle_json_sha256", "source_checkpoint_sha256", "split_sha256", "recipe_sha256", "fit_calibration_json_sha256", "operating_evidence_sha256"):
        value = source.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise BundleError(f"missing or invalid source hash: {key}")


def calibration_function(meta: dict):
    """The recording-level mapping to install in the inference path, or None when calibration was not accepted."""
    if not meta["calibration_accepted"]:
        return None
    eps, params = meta["epsilon"], meta["parameters"]
    if meta["method"] == "temperature_on_log_odds":
        return lambda score: apply_temperature([score], params["temperature"], epsilon=eps)[0]
    return lambda score: apply_affine([score], params["a"], params["b"], epsilon=eps)[0]


def decision_rule_of(meta: dict) -> DecisionRule:
    rule = meta["operating_decision_rule"]
    return DecisionRule(1, rule["kind"], rule.get("threshold"))


def package_calibrated_wavlm(source_bundle, destination, calibration: dict, *, bundle_id: str, extra: dict | None = None) -> Path:
    """Copy the verified model components of the source bundle and add the calibration component. Refuses to overwrite and verifies the result."""
    validate_calibration(calibration)
    source, destination = Path(source_bundle), Path(destination)
    if destination.exists():
        raise BundleError(f"refusing to overwrite {destination}")
    loaded = load_bundle(source)
    if loaded.spec.model_version != MODEL_VERSION or loaded.spec.preprocessing_hash != PREPROCESSING_HASH:
        raise BundleError("the source bundle is not the WavLM development bundle")
    if set(loaded.components) != set(MODEL_COMPONENTS):
        raise BundleError("unexpected components in the source bundle")
    if file_digest(source / "bundle.json") != calibration["source"]["source_bundle_json_sha256"]:
        raise BundleError("the source bundle differs from the one the calibration was fitted for")
    model_meta = json.loads(loaded.components["model/wavlm_head.json"].read_text(encoding="utf-8"))
    if model_meta["source_checkpoint_sha256"] != calibration["source"]["source_checkpoint_sha256"]:
        raise BundleError("the calibration was fitted for a different checkpoint")
    rule = decision_rule_of(calibration)
    spec = BundleSpecification(1, bundle_id, MODEL_VERSION, PREPROCESSING_HASH, rule, "genuine", "tts")
    components = {name: loaded.components[name] for name in MODEL_COMPONENTS}
    components[CALIBRATION_COMPONENT] = (json.dumps(calibration, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    meta_extra = {"stage": "r3_n1_wavlm_development", "calibration_accepted": calibration["calibration_accepted"], "score_type": calibration["score_type"],
                  "R3_N1_required": False, "promoted": False, "locked": False, "source_bundle_id": loaded.spec.bundle_id,
                  "source_bundle_json_sha256": calibration["source"]["source_bundle_json_sha256"], "transformers_pin": TRANSFORMERS_PIN, **(extra or {})}
    write_bundle(destination, spec, components, extra=meta_extra)
    # verification of what was written: it loads, every model component equals the source byte for byte, and the stored rule is the serialized one
    written = load_bundle(destination)
    for name in MODEL_COMPONENTS:
        if file_digest(written.components[name]) != file_digest(loaded.components[name]):
            shutil.rmtree(destination, ignore_errors=True)
            raise BundleError(f"component {name} differs from the source after packaging")
    if written.spec.decision_rule != rule or set(written.components) != set(MODEL_COMPONENTS) | {CALIBRATION_COMPONENT}:
        shutil.rmtree(destination, ignore_errors=True)
        raise BundleError("the written bundle does not match its specification")
    return destination


def load_wavlm_bundle(bundle_path, *, device: str = "cpu", scorer_factory=None, threshold: float | None = None):
    """Verified loader returning a `LoadedBundle`. `scorer_factory(path) -> callable` lets tests inject a scorer without any model library."""
    from echoproof.inference import LoadedBundle, mean_probability

    path = Path(bundle_path)
    bundle = load_bundle(path)
    if bundle.spec.preprocessing_hash != PREPROCESSING_HASH or bundle.spec.model_version != MODEL_VERSION:
        raise BundleError("incompatible WavLM bundle")
    names = set(bundle.components)
    if names == set(MODEL_COMPONENTS):                                    # the uncalibrated development bundle: a model score, the stored 0.5 rule
        if bundle.raw_metadata.get("extra", {}).get("stage") != "development_uncalibrated":
            raise BundleError("a bundle without a calibration component must be the uncalibrated development bundle")
        calibrate, score_type, spec = None, "model_score", bundle.spec
        if threshold is not None:
            spec = BundleSpecification(1, bundle.spec.bundle_id, bundle.spec.model_version, bundle.spec.preprocessing_hash,
                                       DecisionRule(1, "threshold", float(threshold)), bundle.spec.class_0, bundle.spec.class_1)
    elif names == set(MODEL_COMPONENTS) | {CALIBRATION_COMPONENT}:
        if threshold is not None:
            raise ContractError("a threshold override is unavailable for a calibrated WavLM bundle; its serialized decision rule is the official one")
        meta = json.loads(bundle.components[CALIBRATION_COMPONENT].read_text(encoding="utf-8"))
        validate_calibration(meta)
        model_meta = json.loads(bundle.components["model/wavlm_head.json"].read_text(encoding="utf-8"))
        if meta["source"]["source_checkpoint_sha256"] != model_meta["source_checkpoint_sha256"]:
            raise BundleError("the calibration belongs to a different checkpoint")
        if bundle.spec.decision_rule != decision_rule_of(meta):
            raise BundleError("the bundle's decision rule differs from the calibration component's")
        calibrate, score_type, spec = calibration_function(meta), meta["score_type"], bundle.spec
    else:
        raise BundleError("unexpected components in the WavLM bundle")
    if scorer_factory is not None:
        scorer = scorer_factory(path)
    else:
        import importlib.metadata as metadata
        try:
            version = metadata.version("transformers")
        except metadata.PackageNotFoundError:
            version = None
        if version != TRANSFORMERS_PIN:
            warnings.warn(f"transformers {version} is installed; the WavLM bundle is verified with {TRANSFORMERS_PIN}", RuntimeWarning, stacklevel=2)
        from echoproof.models.wavlm_adapter import WavLMBundleAdapter
        scorer = WavLMBundleAdapter.load(path, device=device).score_chunks
    return LoadedBundle(spec=spec, scorer=scorer, aggregate=mean_probability, calibrate=calibrate, score_type=score_type)


def package_stage(root, recipe: dict, run_dir) -> str:
    """Build `bundles/r3_n1_wavlm_head_B_calibrated` from a finished run (fit and operating complete). Needs no model library."""
    from echoproof.evaluation.r3_n1_wavlm import environment, load_json, now, recipe_hash, write

    root, run_dir = Path(root), Path(run_dir)
    fit_dir, op_dir = run_dir / "fit", run_dir / "operating"
    complete = load_json(fit_dir / "stage_complete.json")
    if complete["recipe_sha256"] != recipe_hash(recipe) or any(file_digest(fit_dir / n) != h for n, h in complete["files"].items()):
        raise BundleError("the frozen fit changed")
    evidence = load_json(op_dir / "evidence.json")
    if evidence.get("status") != "complete" or any(file_digest(op_dir / n) != h for n, h in evidence["outputs"].items()):
        raise BundleError("the operating evaluation is incomplete or changed")
    fit, threshold = load_json(fit_dir / "calibration.json"), load_json(op_dir / "threshold.json")
    method = fit["selected_mapping"]
    meta = {"schema": CALIBRATION_SCHEMA, "model": "wavlm_head", "method": METHODS[method], "parameters": fit["selected_parameters"], "epsilon": recipe["epsilon"],
            "calibration_accepted": threshold["calibration_accepted"], "score_type": threshold["score_type"],
            "operating_decision_rule": threshold["decision_rule"], "raw_equivalent_threshold": threshold["raw_equivalent_threshold"],
            "fitting_population": {"role": "calibration", "recordings": fit["fit_recordings"], "speakers": fit["fit_speakers"]},
            "temperature_reference": fit["temperature_reference"]["fit"], "transformers_pin": recipe["transformers_pin"],
            "source": {"source_bundle_json_sha256": recipe["model"]["source_bundle_json_sha256"], "source_checkpoint_sha256": recipe["model"]["checkpoint_sha256"],
                       "split_sha256": recipe["model"]["split_sha256"], "recipe_sha256": recipe_hash(recipe), "run_id": run_dir.name,
                       "calibration_predictions_sha256": recipe["inputs"]["predictions_sha256"]["predictions_recordings_calibration.parquet"],
                       "operating_predictions_sha256": recipe["inputs"]["predictions_sha256"]["predictions_recordings_operating.parquet"],
                       "fit_calibration_json_sha256": file_digest(fit_dir / "calibration.json"), "operating_evidence_sha256": file_digest(op_dir / "evidence.json")},
            "created_utc": now()}
    destination = root / recipe["bundle_destination"]
    package_calibrated_wavlm(root / recipe["model"]["source_bundle"], destination, meta, bundle_id="bundle:r3-n1-wavlm-head-B-calibrated",
                             extra={"run_id": run_dir.name})
    handoff = run_dir / "handoff"
    handoff.mkdir(exist_ok=True)
    verification = {"task": "R3-N1-WavLM bundle packaging", "created_utc": now(), "bundle": str(destination),
                    "bundle_json_sha256": file_digest(destination / "bundle.json"), "calibration_component_sha256": file_digest(destination / CALIBRATION_COMPONENT),
                    "model_components_equal_source_bytes": True, "loads_with_verified_bundle_loader": True, "calibration_accepted": meta["calibration_accepted"],
                    "score_type": meta["score_type"], "decision_rule": meta["operating_decision_rule"], "environment": environment(recipe),
                    "runtime_parity": "pending: runs on Colab with the pinned transformers (notebooks/R3_N1_WavLM_Colab.ipynb)"}
    write(handoff / "bundle_packaging.json", verification)
    return str(destination)
