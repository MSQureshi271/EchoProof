"""Development bundles for the Protocol A LR and BiGRU models with their serialized R4-N1 decision rule (raw scores).

R4-N1 rejected calibration for LR-A and BiGRU-A (identity mapping), so both keep their raw chunk probabilities as `model_score`s; the operating role only
fixed a threshold. Each bundle holds the model components of the selected A checkpoint plus one hashed `decision/score_rule.json` and carries the threshold
in `bundle.json`'s decision rule. Nothing here edits a B file: the bundle writer, the verified loader and the BiGRU adapter are used as they are.
`load_inference_bundle` routes `logistic_regression:A` to the legacy LR loader, which accepts a threshold override and has no BiGRU route, so A bundles
load with `load_lr_bundle_a` and `load_bigru_bundle_a` below, which refuse an override. App routing for A is an R4-N2 decision.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from echoproof.artifacts import digest, file_digest
from echoproof.bundle import BundleError, load_bundle, write_bundle
from echoproof.contracts import BundleSpecification, ContractError, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH

SCORE_RULE_COMPONENT = "decision/score_rule.json"
SCORE_RULE_SCHEMA = "r4_n1_score_rule_v1"
RECIPE = "configs/r4_n1_calibration_a.json"
SPLIT_SHA256 = "a9075367c89295a953eab1de7dc9f2380b14b3f628c2fabca42aaff94a6201a5"

MODELS = {
    "lr": {"model_version": "logistic_regression:A", "bundle_id": "bundle:r4-n1-lr-A-raw-threshold", "stage": "r4_n1_lr_development",
           "destination": "bundles/r4_n1_lr_A_raw_threshold", "run": "runs/r4/r4_n1_lr",
           "components": {"model/weights.npz", "model/lr_model.json", SCORE_RULE_COMPONENT}},
    "bigru": {"model_version": "bigru:A", "bundle_id": "bundle:r4-n1-bigru-A-raw-threshold", "stage": "r4_n1_bigru_development",
              "destination": "bundles/r4_n1_bigru_A_raw_threshold", "run": "runs/r4/r4_n1_bigru",
              "components": {"model/weights.pt", "model/bigru_model.json", "model/normalization.npz", SCORE_RULE_COMPONENT}},
}


def _json(path) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return obj


def _encode(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def verified_run(root, model: str) -> dict:
    """Check the R4-N1 recipe, run folder and saved results for `model`; returns the facts the bundle needs. Reads no operating predictions."""
    from echoproof.evaluation.r4_n1 import load_recipe

    spec, root = MODELS[model], Path(root)
    recipe = load_recipe(root / RECIPE)
    entry = recipe["models"][model]
    if entry.get("verification_state") != "verified_local" or entry.get("split_sha256") != SPLIT_SHA256:
        raise BundleError(f"{model}: recipe inputs are not verified for split {SPLIT_SHA256[:8]}")
    run = root / spec["run"]
    fit_dir, op_dir = run / "fit", run / "operating"
    fit_done, op_done = _json(fit_dir / "stage_complete.json"), _json(op_dir / "stage_complete.json")
    if fit_done.get("recipe_sha256") != digest(recipe) or op_done.get("recipe_sha256") != digest(recipe):
        raise BundleError(f"{model}: the run was made with a different recipe")
    if any(file_digest(fit_dir / n) != h for n, h in fit_done["files"].items()):
        raise BundleError(f"{model}: the frozen fit changed")
    if file_digest(op_dir / "operating_results.json") != op_done["operating_results_sha256"] or op_done.get("test_data_read") is not False:
        raise BundleError(f"{model}: the operating results changed or the test partition was read")
    results, fit = _json(op_dir / "operating_results.json"), _json(fit_dir / "calibration.json")
    if (results.get("model") != model or results.get("mapping") != "identity" or results.get("mapping_accepted") is not False
            or results.get("score_type") != "model_score" or results.get("threshold_score_source") != "raw"
            or fit.get("selected_mapping") != "identity" or results.get("test_data_read") is not False):
        raise BundleError(f"{model}: expected an identity mapping on raw model scores")
    rule = results["threshold_rule"]
    threshold = rule.get("threshold")
    if rule.get("kind") != "threshold" or isinstance(threshold, bool) or not isinstance(threshold, float) or not 0.0 < threshold < 1.0:
        raise BundleError(f"{model}: the saved decision rule is not a usable threshold")
    if file_digest(root / entry["selected_model_path"]) != entry["selected_model_sha256"]:
        raise BundleError(f"{model}: the selected model file differs from the recipe")
    return {"recipe": recipe, "entry": entry, "results": results, "fit": fit, "threshold": threshold, "run": run,
            "hashes": {"recipe_sha256": digest(recipe), "recipe_file_sha256": file_digest(root / RECIPE),
                       "fit_calibration_sha256": file_digest(fit_dir / "calibration.json"),
                       "operating_results_sha256": op_done["operating_results_sha256"], "input_audit_sha256": file_digest(run / "input_audit.json"),
                       "operating_predictions_sha256": results["operating_predictions_sha256"], "selected_model_sha256": entry["selected_model_sha256"]}}


def score_rule(model: str, facts: dict) -> dict:
    return {"schema": SCORE_RULE_SCHEMA, "model": model, "protocol": "A", "mapping": "identity", "calibration_accepted": False,
            "score_type": "model_score", "aggregation": "mean_chunk_probability",
            "decision_rule": {"kind": "threshold", "threshold": facts["threshold"], "comparison": ">="},
            "threshold_selection": {"rule": "macro F1 over score-induced partitions on the operating role", "development_result": True,
                                    "plateau": facts["results"]["threshold_metrics"]["plateau"]},
            "split_sha256": SPLIT_SHA256, "source": facts["hashes"]}


def _spec(model: str, threshold: float) -> BundleSpecification:
    m = MODELS[model]
    return BundleSpecification(1, m["bundle_id"], m["model_version"], PREPROCESSING_HASH, DecisionRule(1, "threshold", threshold), "genuine", "tts")


def _extra(model: str, facts: dict, more: dict) -> dict:
    return {"stage": MODELS[model]["stage"], "protocol": "A", "calibration_accepted": False, "score_type": "model_score", "promoted": False, "locked": False,
            "run_id": Path(facts["run"]).name, "score_rule_sha256_source": facts["hashes"]["operating_results_sha256"], **more}


def _saved_score_parity(root, model: str, facts: dict) -> dict:
    """Replay the saved operating raw recording scores through the serialized rule; compare with the saved threshold metrics."""
    import pyarrow.parquet as pq
    from echoproof.evaluation.r4_n1 import _decision_metrics

    export = Path(root) / facts["entry"]["export_directory"] / "predictions_recordings_operating.parquet"
    if file_digest(export) != facts["entry"]["prediction_sha256"]["predictions_recordings_operating.parquet"]:
        raise BundleError(f"{model}: operating predictions differ from the recipe")
    rows = pq.read_table(export).to_pylist()
    y = [int(r["label"]) for r in rows]
    decisions = [int(float(r["raw_score"]) >= facts["threshold"]) for r in rows]
    got, saved = _decision_metrics(y, decisions), facts["results"]["threshold_metrics"]
    ok = abs(got["macro_f1"] - saved["macro_f1"]) <= 1e-12 and abs(got["genuine_fpr"] - saved["genuine_fpr"]) <= 1e-12
    return {"status": "passed" if ok else "failed", "recordings": len(rows), "decisions_positive": int(sum(decisions)), "tp": got["tp"], "fp": got["fp"],
            "tn": got["tn"], "fn": got["fn"], "macro_f1": got["macro_f1"], "genuine_fpr": got["genuine_fpr"],
            "scope": "serialized threshold replayed on the exported raw recording scores; no waveform scoring"}


def _lr_components(root, facts: dict):
    model_dir = Path(root) / Path(facts["entry"]["selected_model_path"]).parent
    meta = _json(model_dir / "lr_model.json")
    if meta.get("protocol") != "A" or meta.get("split_sha256") != SPLIT_SHA256 or meta.get("label_map") != {"genuine": 0, "tts": 1}:
        raise BundleError("lr: the model file is not the Protocol A LR on the A split")
    return {"model/weights.npz": model_dir / "weights.npz", "model/lr_model.json": model_dir / "lr_model.json"}, {}, None


def _bigru_components(root, facts: dict, stage: Path):
    import torch
    from echoproof.features.acoustic import SCHEMA
    from echoproof.models.bigru_adapter import MODEL_SCHEMA, validate_metadata

    root = Path(root)
    checkpoint = torch.load(root / facts["entry"]["selected_model_path"], map_location="cpu", weights_only=True)
    state, config = checkpoint.get("model"), checkpoint.get("config", {})
    contract = _json(root / "runs/r4/r4_m1_a_v1_bigru/experiment_contract.json")
    arch = contract["architecture"]
    if (checkpoint.get("epoch") != facts["entry"]["selected_epoch"] or config.get("protocol") != "A" or not isinstance(state, dict)
            or arch != {"input_size": 64, "hidden_size": 128, "num_layers": 2, "bidirectional": True, "dropout": 0.3, "pooling": "masked_mean"}):
        raise BundleError("bigru: the checkpoint is not the selected Protocol A BiGRU")
    if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values()):
        raise BundleError("bigru: the checkpoint holds non-finite or non-tensor state")
    if tuple(state["bigru.gru.weight_hh_l0"].shape) != (3 * arch["hidden_size"], arch["hidden_size"]) or tuple(state["bigru.gru.weight_ih_l0"].shape)[1] != 64:
        raise BundleError("bigru: tensor shapes disagree with the recorded architecture")
    mean, std = state["mean"].numpy(), state["std"].numpy()
    (stage / "model").mkdir(parents=True)
    weights, norm = stage / "model/weights.pt", stage / "model/normalization.npz"
    torch.save({k: v.detach().cpu().contiguous() for k, v in state.items()}, weights)
    np.savez(norm, mean=mean, std=std)
    meta = {"schema": MODEL_SCHEMA, "architecture": arch, "frontend": SCHEMA, "protocol": "A", "label_map": {"genuine": 0, "tts": 1},
            "score_type": "raw_uncalibrated", "preprocessing_hash": PREPROCESSING_HASH, "source_checkpoint_sha256": facts["entry"]["selected_model_sha256"],
            "source_epoch": int(checkpoint["epoch"]), "source_config_hash": checkpoint["config_hash"], "split_sha256": SPLIT_SHA256,
            "normalization_source": "selected checkpoint model.mean/model.std",
            "component_sha256": {"weights.pt": file_digest(weights), "normalization.npz": file_digest(norm)}}
    validate_metadata(meta)
    meta_path = stage / "model/bigru_model.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ({"model/weights.pt": weights, "model/bigru_model.json": meta_path, "model/normalization.npz": norm},
            {"source_checkpoint": {"sha256": facts["entry"]["selected_model_sha256"], "epoch": int(checkpoint["epoch"])}}, stage)


def build_bundle(root, model: str, destination=None) -> dict:
    """Package the A model with its serialized R4-N1 rule. Never overwrites; verifies the written bundle before returning."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    root = Path(root)
    destination = Path(destination) if destination is not None else root / MODELS[model]["destination"]
    if destination.exists():
        raise BundleError(f"refusing to overwrite {destination}")
    facts = verified_run(root, model)
    rule_bytes = _encode(score_rule(model, facts))
    with tempfile.TemporaryDirectory(prefix=f"r4_n1_{model}_") as tmp:
        if model == "lr":
            components, more, _ = _lr_components(root, facts)
        else:
            components, more, _ = _bigru_components(root, facts, Path(tmp))
        components = {**components, SCORE_RULE_COMPONENT: rule_bytes}
        write_bundle(destination, _spec(model, facts["threshold"]), components, extra=_extra(model, facts, more))
    loaded = load_bundle(destination)
    if set(loaded.components) != MODELS[model]["components"] or loaded.spec.decision_rule != DecisionRule(1, "threshold", facts["threshold"]):
        raise BundleError("the written bundle does not match its specification")
    parity = _saved_score_parity(root, model, facts)
    verify = load_bundle_a(destination, model)           # the A loader's own checks (CPU, no audio)
    handoff = facts["run"] / "handoff"
    handoff.mkdir(exist_ok=True)
    evidence = {"task": f"R4-N1-{model.upper()} bundle packaging", "bundle": str(destination), "bundle_json_sha256": file_digest(destination / "bundle.json"),
                "components": {n: file_digest(p) for n, p in loaded.components.items()}, "decision_rule": {"kind": "threshold", "threshold": facts["threshold"]},
                "score_type": "model_score", "calibration_accepted": False, "saved_score_parity": parity, "loads_with_a_loader": verify.spec.bundle_id,
                "source": facts["hashes"], "waveform_parity": "pending: Colab (needs the canonical shards)", "test_partition_read": False}
    out = handoff / "bundle_packaging.json"
    if out.exists():
        raise BundleError(f"refusing to overwrite {out}")
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if parity["status"] != "passed":
        raise BundleError(f"{model}: saved-score parity failed; see {out}")
    return evidence


def _check_loaded(path, model: str, threshold):
    if threshold is not None:
        raise ContractError("a threshold override is unavailable for an A bundle; its serialized decision rule is the official one")
    bundle, meta = load_bundle(path), MODELS[model]
    if bundle.spec.preprocessing_hash != PREPROCESSING_HASH or bundle.spec.model_version != meta["model_version"]:
        raise BundleError(f"not a {meta['model_version']} bundle")
    if set(bundle.components) != meta["components"]:
        raise BundleError("unexpected components in the bundle")
    extra = bundle.raw_metadata.get("extra", {})
    rule = json.loads(bundle.components[SCORE_RULE_COMPONENT].read_text(encoding="utf-8"))
    if (extra.get("stage") != meta["stage"] or extra.get("protocol") != "A" or rule.get("schema") != SCORE_RULE_SCHEMA or rule.get("model") != model
            or rule.get("mapping") != "identity" or rule.get("score_type") != "model_score" or rule.get("aggregation") != "mean_chunk_probability"):
        raise BundleError("the score rule does not describe this bundle")
    r = rule["decision_rule"]
    if bundle.spec.decision_rule != DecisionRule(1, r["kind"], r["threshold"]) or r.get("comparison") != ">=":
        raise BundleError("the bundle's decision rule differs from its score rule")
    return bundle


def load_bundle_a(path, model: str, *, device: str = "cpu", scorer_factory=None, threshold: float | None = None):
    """Verified loader returning a `LoadedBundle` (raw model scores, mean chunk probability, the serialized threshold)."""
    from echoproof.inference import LoadedBundle, mean_probability

    bundle = _check_loaded(path, model, threshold)
    if scorer_factory is not None:
        scorer = scorer_factory(Path(path))
    elif model == "lr":
        from echoproof.models.lr_detector import LRDetector
        scorer = LRDetector.load(bundle.components["model/lr_model.json"].parent).score_chunks
    else:
        from echoproof.models.bigru_adapter import BiGRUBundleAdapter
        adapter = BiGRUBundleAdapter.load(path, device=device)
        if adapter.metadata["protocol"] != "A":
            raise BundleError("the BiGRU components are not Protocol A")
        scorer = adapter.score_chunks
    return LoadedBundle(spec=bundle.spec, scorer=scorer, aggregate=mean_probability, calibrate=None, score_type="model_score")


def load_lr_bundle_a(path, **kw):
    return load_bundle_a(path, "lr", **kw)


def load_bigru_bundle_a(path, **kw):
    return load_bundle_a(path, "bigru", **kw)
