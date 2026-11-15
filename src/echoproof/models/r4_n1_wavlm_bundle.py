"""Calibrated development bundle for the Protocol A WavLM head (R4-N1): packaging and the inference loader.

Same design as the B bundle (`r3_n1_wavlm_bundle`, frozen by the GR3 lock): the calibrated bundle holds the original `model/` components of
`bundles/r4_u1_wavlm_head_A` byte for byte plus one hashed `calibration/calibration.json`; `bundle.json` carries the serialized decision rule and
inference uses only that rule. The calibration component's schema, bounds checks and mapping functions are imported from the B module unchanged, so
A and B cannot drift apart. What differs is the model version (`wavlm_head:A`), the bundle id, the stage label and the recipe.

`echoproof.inference.load_inference_bundle` routes the family `wavlm_head` to the B loader, which refuses `wavlm_head:A`; that file is hashed in the
GR3 lock and is not edited here. Load an A bundle with `load_wavlm_bundle_a` below (the app routing for A is a separate decision, R4-N2).
"""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle, write_bundle
from echoproof.contracts import BundleSpecification, ContractError, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.models.r3_n1_wavlm_bundle import (CALIBRATION_COMPONENT, CALIBRATION_SCHEMA, METHODS, MODEL_COMPONENTS, TRANSFORMERS_PIN, calibration_function,
                                                 decision_rule_of, validate_calibration)

MODEL_VERSION = "wavlm_head:A"
BUNDLE_ID = "bundle:r4-n1-wavlm-head-A-calibrated"
STAGE = "r4_n1_wavlm_development"


def package_calibrated_wavlm_a(source_bundle, destination, calibration: dict, *, bundle_id: str = BUNDLE_ID, extra: dict | None = None) -> Path:
    """Copy the verified model components of the A source bundle and add the calibration component. Refuses to overwrite and verifies the result."""
    validate_calibration(calibration)
    source, destination = Path(source_bundle), Path(destination)
    if destination.exists():
        raise BundleError(f"refusing to overwrite {destination}")
    loaded = load_bundle(source)
    if loaded.spec.model_version != MODEL_VERSION or loaded.spec.preprocessing_hash != PREPROCESSING_HASH:
        raise BundleError("the source bundle is not the Protocol A WavLM development bundle")
    if set(loaded.components) != set(MODEL_COMPONENTS):
        raise BundleError("unexpected components in the source bundle")
    if file_digest(source / "bundle.json") != calibration["source"]["source_bundle_json_sha256"]:
        raise BundleError("the source bundle differs from the one the calibration was fitted for")
    model_meta = json.loads(loaded.components["model/wavlm_head.json"].read_text(encoding="utf-8"))
    if model_meta["source_checkpoint_sha256"] != calibration["source"]["source_checkpoint_sha256"]:
        raise BundleError("the calibration was fitted for a different checkpoint")
    if model_meta.get("split_sha256") != calibration["source"]["split_sha256"]:
        raise BundleError("the calibration was fitted for a different split")
    rule = decision_rule_of(calibration)
    spec = BundleSpecification(1, bundle_id, MODEL_VERSION, PREPROCESSING_HASH, rule, "genuine", "tts")
    components = {name: loaded.components[name] for name in MODEL_COMPONENTS}
    components[CALIBRATION_COMPONENT] = (json.dumps(calibration, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    meta_extra = {"stage": STAGE, "protocol": "A", "calibration_accepted": calibration["calibration_accepted"], "score_type": calibration["score_type"],
                  "R4_N1_required": False, "promoted": False, "locked": False, "source_bundle_id": loaded.spec.bundle_id,
                  "source_bundle_json_sha256": calibration["source"]["source_bundle_json_sha256"], "transformers_pin": TRANSFORMERS_PIN, **(extra or {})}
    write_bundle(destination, spec, components, extra=meta_extra)
    written = load_bundle(destination)
    for name in MODEL_COMPONENTS:
        if file_digest(written.components[name]) != file_digest(loaded.components[name]):
            shutil.rmtree(destination, ignore_errors=True)
            raise BundleError(f"component {name} differs from the source after packaging")
    if written.spec.decision_rule != rule or set(written.components) != set(MODEL_COMPONENTS) | {CALIBRATION_COMPONENT}:
        shutil.rmtree(destination, ignore_errors=True)
        raise BundleError("the written bundle does not match its specification")
    return destination


def load_wavlm_bundle_a(bundle_path, *, device: str = "cpu", scorer_factory=None, threshold: float | None = None):
    """Verified loader returning a `LoadedBundle`. `scorer_factory(path) -> callable` lets tests inject a scorer without any model library."""
    from echoproof.inference import LoadedBundle, mean_probability

    path = Path(bundle_path)
    bundle = load_bundle(path)
    if bundle.spec.preprocessing_hash != PREPROCESSING_HASH or bundle.spec.model_version != MODEL_VERSION:
        raise BundleError("incompatible Protocol A WavLM bundle")
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
    """Build `recipe["bundle_destination"]` from a finished run (fit and operating complete). Needs no model library."""
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
            "temperature_reference": fit["temperature_reference"]["fit"], "transformers_pin": recipe["transformers_pin"], "protocol": "A",
            "source": {"source_bundle_json_sha256": recipe["model"]["source_bundle_json_sha256"], "source_checkpoint_sha256": recipe["model"]["checkpoint_sha256"],
                       "split_sha256": recipe["model"]["split_sha256"], "recipe_sha256": recipe_hash(recipe), "run_id": run_dir.name,
                       "calibration_predictions_sha256": recipe["inputs"]["predictions_sha256"]["predictions_recordings_calibration.parquet"],
                       "operating_predictions_sha256": recipe["inputs"]["predictions_sha256"]["predictions_recordings_operating.parquet"],
                       "fit_calibration_json_sha256": file_digest(fit_dir / "calibration.json"), "operating_evidence_sha256": file_digest(op_dir / "evidence.json")},
            "created_utc": now()}
    destination = root / recipe["bundle_destination"]
    package_calibrated_wavlm_a(root / recipe["model"]["source_bundle"], destination, meta, extra={"run_id": run_dir.name})
    handoff = run_dir / "handoff"
    handoff.mkdir(exist_ok=True)
    verification = {"task": "R4-N1-WavLM bundle packaging", "created_utc": now(), "bundle": str(destination),
                    "bundle_json_sha256": file_digest(destination / "bundle.json"), "calibration_component_sha256": file_digest(destination / CALIBRATION_COMPONENT),
                    "model_components_equal_source_bytes": True, "loads_with_verified_bundle_loader": True, "calibration_accepted": meta["calibration_accepted"],
                    "score_type": meta["score_type"], "decision_rule": meta["operating_decision_rule"], "environment": environment(recipe),
                    "runtime_parity": "pending: runs on Colab with the pinned transformers (notebooks/R4_N1_Colab_WavLM_Calibration_A.ipynb)"}
    write(handoff / "bundle_packaging.json", verification)
    return str(destination)
