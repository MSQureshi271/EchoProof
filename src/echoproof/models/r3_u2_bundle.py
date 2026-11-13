"""Package the selected R3-U2 fine-tuned AST into a verified development bundle (uncalibrated; R3-N1 still required).

The selected `best.pt` is read once (weights-only loader), loaded strictly into the pinned AST architecture, and re-saved as safetensors together with a
two-class config and the pinned feature-extractor config. The conversion is verified tensor by tensor against the checkpoint, and the group checksums
prove the bundle is the pinned backbone with only the unfrozen groups changed. Inference never reads a pickle. Nothing here fits, calibrates or updates a lock.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from echoproof.artifacts import file_digest
from echoproof.bundle import write_bundle
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.features.pretrained import PRETRAINED
from echoproof.models.ast_adapter import EXPECTED_PROCESSOR, META_FILE, MODEL_FILES, MODEL_SCHEMA, validate_metadata

LABELS = {0: "genuine", 1: "tts"}


def unfrozen_group_names(n_blocks: int, unfrozen_blocks: int) -> list:
    return sorted(["classifier", "final_norm"] + [f"block.{i:02d}" for i in range(n_blocks - unfrozen_blocks, n_blocks)])


def package_ast_detector(checkpoint_path, snapshot, destination, *, bundle_id, protocol, expected_checkpoint_sha256, split_sha256, expected_epoch,
                         expected_config_hash, recipe_sha256, warmup_checkpoint_sha256, pretrained_group_sha256, unfrozen_blocks=2,
                         experiment=None, source_spec=None, processor_expectations=None, feature_extractor=None):
    """`source_spec`, `processor_expectations` and `feature_extractor` exist so tests can package a tiny random AST under its own identity."""
    import torch
    from transformers import ASTForAudioClassification
    from echoproof.models.ast_detector import ASTDetector

    spec = source_spec or PRETRAINED["ast"]
    expected_processor = dict(processor_expectations or EXPECTED_PROCESSOR)
    source, snapshot, destination = Path(checkpoint_path), Path(snapshot), Path(destination)
    if destination.exists():
        raise FileExistsError("refusing to replace an existing bundle")
    if file_digest(source) != expected_checkpoint_sha256:
        raise ValueError("source checkpoint differs from the selected checkpoint hash")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != expected_epoch:
        raise ValueError("checkpoint epoch differs from the selection")
    if checkpoint.get("config_hash") != expected_config_hash:
        raise ValueError("checkpoint configuration differs from the selection")
    state = checkpoint["model"]
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError("non-finite model weights")
    if snapshot.name != spec["revision"]:
        raise ValueError(f"AST snapshot {snapshot.name} is not the pinned revision {spec['revision']}")
    base_sha = file_digest(snapshot / spec["weights_file"])
    if base_sha != spec["weights_sha256"]:
        raise ValueError("AST base weights differ from the pinned SHA-256")
    processor = json.loads((snapshot / "preprocessor_config.json").read_text(encoding="utf-8"))
    if any(processor.get(k) != v for k, v in expected_processor.items()):
        raise ValueError("AST preprocessing differs from the one used in training")

    detector = ASTDetector.from_dir(snapshot, device="cpu", seed=0, feature_extractor=feature_extractor)
    detector.load_state_dict(state, strict=True)
    digests = detector.digests()
    unfrozen = unfrozen_group_names(detector.n_blocks, unfrozen_blocks)
    for group, value in digests.items():
        if group in unfrozen:
            if pretrained_group_sha256.get(group) == value:
                raise ValueError(f"unfrozen group {group} still equals the pretrained weights")
        elif pretrained_group_sha256.get(group) != value:
            raise ValueError(f"frozen group {group} differs from the pinned pretrained weights")
    config = detector.model.config
    config.id2label, config.label2id = dict(LABELS), {v: k for k, v in LABELS.items()}
    if int(config.num_labels) != 2:
        raise ValueError("the saved config must describe two classes")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r3_u2_package_", dir=destination.parent) as temp:
        folder = Path(temp)
        model_dir = folder / "model"
        detector.model.save_pretrained(model_dir, safe_serialization=True)
        shutil.copyfile(snapshot / "preprocessor_config.json", model_dir / "preprocessor_config.json")
        reloaded = ASTForAudioClassification.from_pretrained(model_dir, local_files_only=True)
        original, converted = detector.model.state_dict(), reloaded.state_dict()
        ignorable = lambda keys: {k for k in keys if k.endswith("position_ids")}
        if set(original) - set(converted) - ignorable(original) or set(converted) - set(original) - ignorable(converted):
            raise ValueError("the saved weights have a different tensor set")
        common = sorted(set(original) & set(converted))
        if not common or not all(torch.equal(original[k], converted[k]) for k in common):
            raise ValueError("the safetensors conversion is not identical to the selected checkpoint")
        stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        if not all(torch.equal(stripped[k], converted[k]) for k in common if k in stripped) or not set(common) <= set(stripped):
            raise ValueError("the saved weights differ from the checkpoint's own tensors")
        names = {n: model_dir / n for n in MODEL_FILES}
        meta = {"schema": MODEL_SCHEMA, "protocol": protocol, "label_map": {"genuine": 0, "tts": 1}, "score_type": "raw_uncalibrated",
                "preprocessing_hash": PREPROCESSING_HASH, "export_precision": checkpoint.get("config", {}).get("precision", "fp16"),
                "model": {"num_labels": 2, "hidden_size": int(config.hidden_size), "blocks": detector.n_blocks,
                          "parameters": sum(p.numel() for p in detector.model.parameters())},
                "base": {"repo_id": spec["repo_id"], "revision": spec["revision"], "source_weights_sha256": base_sha},
                "fine_tuning": {"warmup_checkpoint_sha256": warmup_checkpoint_sha256, "experiment": experiment,
                                "unfrozen_groups": unfrozen, "frozen_groups_equal_pinned": True, "unfrozen_groups_differ_from_pinned": True,
                                "group_sha256": digests},
                "conversion": {"state_equals_checkpoint": True, "tensors_compared": len(common)},
                "processor": expected_processor, "source_checkpoint_sha256": expected_checkpoint_sha256, "source_epoch": checkpoint["epoch"],
                "source_config_hash": checkpoint.get("config_hash"), "recipe_sha256": recipe_sha256, "split_sha256": split_sha256,
                "component_sha256": {name: file_digest(path) for name, path in names.items()}}
        validate_metadata(meta)
        (folder / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bundle_spec = BundleSpecification(1, bundle_id, "ast_partial:" + protocol, PREPROCESSING_HASH, DecisionRule(1, "threshold", .5), "genuine", "tts")
        components = {"model/" + META_FILE: folder / META_FILE, **{"model/" + n: p for n, p in names.items()}}
        return write_bundle(destination, bundle_spec, components,
                            extra={"stage": "development_uncalibrated", "threshold_status": "selection_reference_only",
                                   "source_checkpoint_sha256": expected_checkpoint_sha256, "R3_N1_required": True})
