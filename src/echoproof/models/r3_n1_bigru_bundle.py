"""Selected R3-M1 BiGRU bundle builder and calibrated loader.

The builder preserves pending N1 review and creates a candidate only. Loading for
normal inference fails closed until that review is approved.
"""
from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle, write_bundle
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.features.acoustic import SCHEMA
from echoproof.inference import LoadedBundle, mean_probability
from echoproof.models.bigru_adapter import BiGRUBundleAdapter, MODEL_SCHEMA, validate_metadata

SELECTED_ID = "b01_lower_lr_full20"
CHECKPOINT_REL = Path("experiments/r3_m1/runs/batch01/b01_lower_lr_full20/best.pt")
SELECTION_REL = Path("experiments/r3_m1/reports/final_selection.json")
HANDOFF_REL = Path("experiments/r3_m1/reports/final_handoff.json")
N1_REL = Path("runs/r3/r3_n1_operational_calibration_selected/20261113T132213Z-0b91eafbbec9")
BUNDLE_COMPONENTS = {
    "model/weights.pt", "model/bigru_model.json", "model/normalization.npz",
    "calibration/calibrators.json", "calibration/thresholds.json",
}


def _json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _selected_sources(root: Path, checkpoint_path: Path | None = None):
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else root / CHECKPOINT_REL
    selection = _json(root / SELECTION_REL)
    handoff = _json(root / HANDOFF_REL)
    checkpoint_sha = file_digest(checkpoint_path)
    selected = handoff.get("selected", {})
    if (selection.get("id") != SELECTED_ID or selection.get("checkpoint") != CHECKPOINT_REL.as_posix()
            or selection.get("checkpoint_sha256") != checkpoint_sha
            or selected.get("id") != SELECTED_ID or selected.get("best_epoch") != selection.get("best_epoch")
            or selected.get("checkpoint") != CHECKPOINT_REL.as_posix()
            or selected.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError("selected checkpoint identity differs from final selection or handoff")
    return checkpoint_path, selection, handoff, checkpoint_sha


def _architecture(checkpoint: dict, root: Path):
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model state dictionary")
    required = {"bigru.gru.weight_ih_l0", "bigru.gru.weight_hh_l0", "bigru.gru.weight_ih_l0_reverse",
                "bigru.gru.weight_hh_l0_reverse", "bigru.gru.weight_ih_l1", "bigru.gru.weight_hh_l1",
                "bigru.gru.weight_ih_l1_reverse", "bigru.gru.weight_hh_l1_reverse", "bigru.head.weight",
                "bigru.head.bias", "mean", "std", "mel_filter", "window"}
    if not required.issubset(state):
        raise ValueError(f"checkpoint is missing model tensors: {sorted(required-set(state))}")
    ih, hh = state["bigru.gru.weight_ih_l0"], state["bigru.gru.weight_hh_l0"]
    hidden, input_size = int(hh.shape[1]), int(ih.shape[1])
    if tuple(hh.shape) != (3*hidden, hidden) or tuple(ih.shape) != (3*hidden, input_size) or input_size != 64:
        raise ValueError("checkpoint GRU tensor dimensions are incompatible")
    layers = sorted({int(k.split("_l",1)[1].split("_",1)[0]) for k in state
                     if k.startswith("bigru.gru.weight_ih_l") and not k.endswith("_reverse")})
    bidir = all(f"bigru.gru.weight_ih_l{i}_reverse" in state for i in layers)
    if layers != [0, 1] or not bidir:
        raise ValueError("checkpoint is not a two-layer bidirectional GRU")
    # Dropout has no learned tensor; take it from the recorded construction call.
    train_file = root / "scripts/r3_m1_train.py"
    tree = ast.parse(train_file.read_text(encoding="utf-8"))
    dropout = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "WaveformBiGRU":
            dropout.extend(ast.literal_eval(k.value) for k in node.keywords if k.arg == "dropout")
    if not dropout or any(float(x) != float(dropout[0]) for x in dropout):
        raise ValueError("training source has no unique BiGRU dropout value")
    resolved = _json(root / "experiments/r3_m1/configs/batch01_resolved.json")
    expected = resolved.get("reference_sources", {}).get("scripts/r3_m1_train.py")
    actual = file_digest(train_file)
    arch = {"input_size": input_size, "hidden_size": hidden, "num_layers": len(layers),
            "bidirectional": bidir, "dropout": float(dropout[0]), "pooling": "masked_mean"}
    source = {"architecture_from_checkpoint_tensor_shapes": True,
              "dropout_source": "scripts/r3_m1_train.py WaveformBiGRU construction",
              "dropout_source_sha256": actual, "frozen_training_source_sha256": expected,
              "dropout_source_matches_frozen_training_hash": actual == expected}
    return arch, source


def apply_platt(score: float, calibrators: dict) -> float:
    """Apply serialized Platt scaling to one recording score."""
    eps = float(calibrators["score_transform"]["epsilon"])
    p = float(score)
    if not math.isfinite(p) or not 0 <= p <= 1 or not 0 < eps < 0.5:
        raise ValueError("invalid score or Platt epsilon")
    p = min(max(p, eps), 1-eps)
    params = calibrators["platt"]
    z = float(params["a"]) * math.log(p/(1-p)) + float(params["b"])
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def _n1_inputs(root: Path, checkpoint_sha: str, selection: dict):
    n1 = root / N1_REL
    names = ("artifact_manifest.json", "evidence.json", "calibrators.json", "thresholds.json", "operating_predictions.csv")
    for name in names:
        if not (n1/name).is_file():
            raise FileNotFoundError(f"missing N1 input: {n1/name}")
    manifest, evidence = _json(n1/"artifact_manifest.json"), _json(n1/"evidence.json")
    calibrators, thresholds = _json(n1/"calibrators.json"), _json(n1/"thresholds.json")
    for name in ("evidence.json", "calibrators.json", "thresholds.json", "operating_predictions.csv"):
        actual = file_digest(n1/name)
        if actual != manifest.get("files", {}).get(name):
            raise ValueError(f"N1 manifest checksum mismatch: {name}")
        if name != "evidence.json" and actual != evidence.get("output_sha256", {}).get(name):
            raise ValueError(f"N1 evidence checksum mismatch: {name}")
    selected = evidence.get("selection", {})
    if (evidence.get("status") != "complete" or selected.get("selected") != SELECTED_ID
            or selected.get("best_epoch") != selection.get("best_epoch")
            or selected.get("checkpoint_sha256") != checkpoint_sha
            or evidence.get("source_hashes", {}).get("selected_checkpoint") != checkpoint_sha):
        raise ValueError("N1 evidence does not identify the selected checkpoint")
    if not evidence.get("boundaries", {}).get("checkpoint_hash_rechecked_after"):
        raise ValueError("N1 did not attest to rechecking the selected checkpoint hash")
    if not evidence.get("boundaries", {}).get("operating_used_for_threshold_and_evaluation"):
        raise ValueError("N1 did not attest to operating threshold selection")
    if "platt" not in calibrators or "platt" not in thresholds:
        raise ValueError("N1 Platt artifacts are missing")
    return n1, manifest, evidence, calibrators, thresholds


def _saved_score_parity(n1: Path, calibrators: dict, threshold: float) -> dict:
    max_diff, decisions, count = 0.0, 0, 0
    with (n1/"operating_predictions.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            calibrated = apply_platt(float(row["raw_score"]), calibrators)
            max_diff = max(max_diff, abs(calibrated-float(row["calibrated_platt"])))
            decisions += int(calibrated >= threshold)
            count += 1
    if count == 0:
        raise ValueError("N1 operating predictions are empty")
    return {"status": "passed" if max_diff <= 1e-12 else "failed", "recordings": count,
            "max_abs_difference": max_diff, "decisions_at_threshold": decisions,
            "scope": "serialized calibration replay against saved selected-model raw recording scores; no waveform replay"}


def _selected_waveform_parity(root: Path, adapter: BiGRUBundleAdapter) -> dict:
    """Re-score all three frozen B validation roles when the original waveform inputs exist."""
    shard_dir = root / "data/processed/cohort_v1"
    required = [
        root / "data/manifests/cohort_v1.parquet",
        root / "data/splits/protocol_B_v1.parquet",
        root / "data/processed/chunks_v1.parquet",
        root / "data/features/r2_s1_B/mel_stats_B_v1.npz",
        root / "data/features/r2_s1_B/r2_s1_manifest.json",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.is_file()]
    if not shard_dir.is_dir():
        missing.append(str(shard_dir.relative_to(root)) + "/")
    if missing:
        return {"status": "not_run", "reason": "required original data inputs are absent",
                "missing_inputs": missing, "roles": {}}

    import pyarrow.parquet as pq
    from echoproof.training.r3_m1_data import build_data

    data = build_data(root, shard_dir, seed=42)
    role_results = {}
    try:
        for role in ("selection", "calibration", "operating"):
            dataset = data["validation"][role]
            expected_path = root / "experiments/r3_m1/runs/batch01/selected" / f"predictions_chunks_{role}.parquet"
            expected_rows = pq.read_table(expected_path).to_pylist()
            expected = {row["chunk_id"]: row for row in expected_rows}
            if len(expected) != len(dataset.chunks) or set(expected) != {c.chunk_id for c in dataset.chunks}:
                raise ValueError(f"{role} prediction export coverage differs from verified data plan")
            scores = {}
            for start in range(0, len(dataset), 64):
                items = [dataset[i] for i in range(start, min(start + 64, len(dataset)))]
                waves = [item["waveform"][:item["valid_samples"]] for item in items]
                values = adapter.score_chunks(waves)
                for item, value in zip(items, values):
                    ref = expected[item["chunk_id"]]
                    if ref["recording_id"] != item["recording_id"] or int(ref["label"]) != item["label"]:
                        raise ValueError(f"{role} chunk identity differs from selected export")
                    scores[item["chunk_id"]] = float(value)
            max_abs = max(abs(scores[cid] - float(row["tts_probability"])) for cid, row in expected.items())
            role_results[role] = {"chunks": len(scores), "max_abs_score_difference": max_abs,
                                  "status": "passed" if max_abs <= 1e-5 else "failed"}
        return {"status": "passed" if all(v["status"] == "passed" for v in role_results.values()) else "failed",
                "tolerance": 1e-5, "roles": role_results,
                "scope": "fresh raw chunk-score replay against authoritative selected-checkpoint exports"}
    finally:
        data["loader"].close()


def build_selected_bigru_bundle(root: str | Path, destination: str | Path | None = None) -> dict:
    """Extract the fixed selected checkpoint and package the exact N1 Platt artifacts."""
    import torch
    root = Path(root).resolve()
    destination = Path(destination) if destination is not None else root/"runs/r3/r3_n1_bigru/selected_bigru_v1/handoff/bundle"
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    checkpoint_path, selection, handoff, checkpoint_sha = _selected_sources(root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != selection.get("best_epoch"):
        raise ValueError("checkpoint epoch differs from model selection")
    resolved = _json(root/"experiments/r3_m1/configs/batch01_resolved.json")
    if checkpoint.get("config") != resolved.get("configs", {}).get(SELECTED_ID):
        raise ValueError("checkpoint config differs from frozen selected config")
    n1, _, n1_evidence, calibrators, thresholds = _n1_inputs(root, checkpoint_sha, selection)
    arch, arch_source = _architecture(checkpoint, root)
    state = checkpoint["model"]
    if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("checkpoint contains non-finite or non-tensor model state")
    mean, std = state["mean"].cpu().numpy(), state["std"].cpu().numpy()
    if mean.shape != (64,) or std.shape != (64,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("checkpoint normalization is invalid")
    threshold = float(thresholds["platt"]["threshold"])
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("serialized Platt threshold is invalid")
    calibration_parity = _saved_score_parity(n1, calibrators, threshold)

    from echoproof.models.bigru_waveform import WaveformBiGRU
    model = WaveformBiGRU(mean, std, hidden_size=arch["hidden_size"], dropout=arch["dropout"],
                          mel_filter=state["mel_filter"], window=state["window"])
    model.load_state_dict(state, strict=True)
    model.eval()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r3_n2_bigru_build_", dir=destination.parent) as td:
        stage = Path(td)
        (stage/"model").mkdir()
        (stage/"calibration").mkdir()
        weights = stage/"model/weights.pt"
        torch.save({k:v.detach().cpu().contiguous() for k,v in state.items()}, weights)
        norm = stage/"model/normalization.npz"
        np.savez(norm, mean=mean, std=std)
        model_meta = {
            "schema": MODEL_SCHEMA, "architecture": arch, "frontend": SCHEMA,
            "protocol": checkpoint["config"]["protocol"], "label_map": {"genuine": 0, "tts": 1},
            "score_type": "raw_uncalibrated", "preprocessing_hash": PREPROCESSING_HASH,
            "source_checkpoint_sha256": checkpoint_sha, "source_epoch": int(checkpoint["epoch"]),
            "source_config_hash": checkpoint["config_hash"],
            "split_artifact_sha256": resolved.get("inputs", {}).get("split"),
            "normalization_source": "selected checkpoint model.mean/model.std",
            "component_sha256": {"weights.pt":file_digest(weights), "normalization.npz":file_digest(norm)},
            "architecture_source": arch_source,
        }
        validate_metadata(model_meta)
        meta_path = stage/"model/bigru_model.json"
        meta_path.write_text(json.dumps(model_meta, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        shutil.copyfile(n1/"calibrators.json", stage/"calibration/calibrators.json")
        shutil.copyfile(n1/"thresholds.json", stage/"calibration/thresholds.json")
        staged_adapter = BiGRUBundleAdapter.load(stage, device="cpu")
        waveform_parity = _selected_waveform_parity(root, staged_adapter)

        handoff_sha = file_digest(root/HANDOFF_REL)
        selection_sha = file_digest(root/SELECTION_REL)
        n1_selection = n1_evidence.get("selection", {})
        handoff_match = n1_selection.get("handoff_sha256") == handoff_sha
        selection_match = n1_selection.get("selection_record_sha256") == selection_sha
        review = n1_selection.get("review_status", "unrecorded")
        source_hashes = {
            "checkpoint":checkpoint_sha, "selection_record":selection_sha, "handoff_record":handoff_sha,
            "n1_evidence":file_digest(n1/"evidence.json"), "n1_calibrators":file_digest(n1/"calibrators.json"),
            "n1_thresholds":file_digest(n1/"thresholds.json"),
            "n1_operating_predictions":file_digest(n1/"operating_predictions.csv"),
            "n1_artifact_manifest":file_digest(n1/"artifact_manifest.json"),
        }
        frozen_sources = resolved.get("reference_sources", {})
        source_code_hashes = {}
        for rel in ("scripts/r3_m1_train.py", "src/echoproof/models/bigru.py",
                    "src/echoproof/models/bigru_waveform.py", "src/echoproof/features/acoustic.py"):
            actual = file_digest(root/rel)
            source_code_hashes[rel] = {"sha256":actual, "frozen_training_sha256":frozen_sources.get(rel),
                                       "matches_frozen_training_source":actual == frozen_sources.get(rel)}
        code_match = all(x["matches_frozen_training_source"] for x in source_code_hashes.values())
        ready = bool(review == "approved" and code_match and handoff_match and waveform_parity["status"] == "passed"
                     and calibration_parity["status"] == "passed")
        extra = {
            "stage":"r3_n2_bigru_calibrated_candidate", "deployment_ready":ready, "review_status":review,
            "model_configuration":arch,
            "source_checkpoint":{"path":CHECKPOINT_REL.as_posix(),"sha256":checkpoint_sha,"model_id":selection["id"],"epoch":selection["best_epoch"]},
            "selection_record":{"path":SELECTION_REL.as_posix(),"sha256":selection_sha},
            "handoff_record":{"path":HANDOFF_REL.as_posix(),"sha256":handoff_sha,
                              "n1_recorded_sha256":n1_selection.get("handoff_sha256"),"matches_n1_record":handoff_match},
            "calibration":{"method":"platt","order":"after_mean_chunk_probability_aggregation",
                           "threshold":threshold,"threshold_comparison":">=","review_status":review,
                           "n1_run_id":n1_evidence["run_id"],"calibrators_sha256":source_hashes["n1_calibrators"],
                           "thresholds_sha256":source_hashes["n1_thresholds"]},
            "provenance_hashes":source_hashes,
            "validation":{"checkpoint_state_strict_load":"passed","saved_operating_calibration_parity":calibration_parity,
                          "selected_checkpoint_waveform_prediction_parity":waveform_parity,
                          "training_source_hashes_match":code_match,"n1_handoff_record_hash_match":handoff_match,
                          "n1_selection_record_hash_match":selection_match},
            "source_code_hashes":source_code_hashes,
        }
        spec = BundleSpecification(1,"bundle:r3-n1-bigru-selected-b01-lower-lr-full20","bigru:B",
                                  PREPROCESSING_HASH,DecisionRule(1,"threshold",threshold),"genuine","tts")
        write_bundle(destination,spec,{
            "model/weights.pt":weights, "model/bigru_model.json":meta_path, "model/normalization.npz":norm,
            "calibration/calibrators.json":stage/"calibration/calibrators.json",
            "calibration/thresholds.json":stage/"calibration/thresholds.json",
        },extra=extra)
    verified = load_bundle(destination)
    adapter = BiGRUBundleAdapter.load(destination,device="cpu")
    actual = {name:file_digest(path) for name,path in verified.components.items()}
    if actual != verified.component_hashes:
        raise BundleError("post-write component hashes do not match bundle manifest")
    return {"bundle_path":str(destination),"bundle_json_sha256":file_digest(destination/"bundle.json"),
            "components":actual,"review_status":review,"deployment_ready":ready,
            "validation":extra["validation"],"adapter_loaded":adapter.metadata["model"]=="bigru"}


def load_calibrated_bigru_bundle(bundle_dir: str | Path, *, device: str = "cpu", batch_size: int = 64,
                                 allow_pending_review: bool = False) -> LoadedBundle:
    """Load into shared inference; pending review fails closed unless explicitly allowed for validation."""
    bundle = load_bundle(bundle_dir)
    extra = bundle.raw_metadata.get("extra", {})
    if extra.get("calibration", {}).get("method") != "platt":
        raise BundleError("bundle does not declare R3-N1 Platt calibration")
    if extra.get("review_status") != "approved" and not allow_pending_review:
        raise BundleError(f"calibration review is {extra.get('review_status', 'unrecorded')}; loading is blocked")
    if set(bundle.components) != BUNDLE_COMPONENTS:
        raise BundleError("bundle has missing or unexpected components")
    calibrators = _json(bundle.get_component_path("calibration/calibrators.json"))
    thresholds = _json(bundle.get_component_path("calibration/thresholds.json"))
    threshold = float(thresholds["platt"]["threshold"])
    if threshold != bundle.spec.decision_rule.threshold:
        raise BundleError("bundle decision threshold differs from serialized Platt threshold")
    adapter = BiGRUBundleAdapter.load(bundle_dir,device=device,batch_size=batch_size)
    calibrate = lambda raw: apply_platt(raw,calibrators)
    return LoadedBundle(spec=bundle.spec,scorer=adapter.score_chunks,aggregate=mean_probability,
                        calibrate=calibrate,score_type="calibrated_probability",is_mock=False)

