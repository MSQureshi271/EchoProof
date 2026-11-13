"""Package the selected R3-U1 WavLM head with its frozen encoder into a verified development bundle (uncalibrated; R3-N1 still required).

The pinned encoder weights are a pickle (`pytorch_model.bin`). They are read once here, after their SHA-256 equals the pinned value, and re-saved as
safetensors; the conversion is verified tensor by tensor against the source before it is accepted. Inference never reads a pickle.
Nothing here fits, calibrates or updates a lock.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from echoproof.artifacts import file_digest
from echoproof.bundle import write_bundle
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.features.pretrained import PRETRAINED
from echoproof.models.wavlm_adapter import EXPECTED_PROCESSOR, MODEL_SCHEMA, POOLING_MULTIPLIER, validate_metadata


def package_wavlm_head(checkpoint_path, encoder_snapshot, destination, *, bundle_id, protocol, expected_checkpoint_sha256, split_sha256,
                       expected_epoch=None, expected_config_hash=None, source_spec=None):
    """`source_spec` defaults to the pinned WavLM entry; it exists so tests can package a tiny random encoder under its own identity."""
    import torch
    from transformers import WavLMModel

    spec = source_spec or PRETRAINED["wavlm"]
    source, snapshot, destination = Path(checkpoint_path), Path(encoder_snapshot), Path(destination)
    if destination.exists():
        raise FileExistsError("refusing to replace an existing bundle")
    if file_digest(source) != expected_checkpoint_sha256:
        raise ValueError("source checkpoint differs from the selected checkpoint hash")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if expected_epoch is not None and checkpoint.get("epoch") != expected_epoch:
        raise ValueError("checkpoint epoch differs from the selection")
    if expected_config_hash is not None and checkpoint.get("config_hash") != expected_config_hash:
        raise ValueError("checkpoint configuration differs from the selection")
    pooling, arch, state = checkpoint["pooling"], checkpoint["architecture"], checkpoint["model"]
    if pooling not in POOLING_MULTIPLIER or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("invalid pooling or non-finite head weights")
    if snapshot.name != spec["revision"]:
        raise ValueError(f"encoder snapshot {snapshot.name} is not the pinned revision {spec['revision']}")
    weights = snapshot / spec["weights_file"]
    source_sha = file_digest(weights)
    if source_sha != spec["weights_sha256"]:
        raise ValueError("encoder weights differ from the pinned SHA-256")
    processor = json.loads((snapshot / "preprocessor_config.json").read_text(encoding="utf-8"))
    if any(processor.get(k) != v for k, v in EXPECTED_PROCESSOR.items()):
        raise ValueError("encoder preprocessing differs from the one used to build the cache")
    encoder = WavLMModel.from_pretrained(snapshot)
    hidden = int(encoder.config.hidden_size)
    if arch["in_dim"] != hidden * POOLING_MULTIPLIER[pooling]:
        raise ValueError("head input width does not match the encoder width and pooling")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r3_u1_package_", dir=destination.parent) as temp:
        folder = Path(temp)
        encoder_dir = folder / "encoder"
        encoder.save_pretrained(encoder_dir, safe_serialization=True)
        (encoder_dir / "preprocessor_config.json").write_bytes((snapshot / "preprocessor_config.json").read_bytes())
        reloaded = WavLMModel.from_pretrained(encoder_dir, local_files_only=True)
        original, converted = encoder.state_dict(), reloaded.state_dict()
        equal = set(original) == set(converted) and all(torch.equal(original[k], converted[k]) for k in original)
        if not equal:
            raise ValueError("the safetensors conversion is not identical to the pinned source weights")
        torch.save({k: v.detach().cpu().contiguous() for k, v in state.items()}, folder / "head_weights.pt")
        names = {"head_weights.pt": folder / "head_weights.pt", **{f"encoder/{n}": encoder_dir / n for n in
                                                                  ("config.json", "preprocessor_config.json", "model.safetensors")}}
        meta = {"schema": MODEL_SCHEMA, "pooling": pooling, "architecture": arch, "protocol": protocol, "label_map": {"genuine": 0, "tts": 1},
                "score_type": "raw_uncalibrated", "preprocessing_hash": PREPROCESSING_HASH,
                "encoder": {"repo_id": spec["repo_id"], "revision": spec["revision"], "hidden_size": hidden, "source_weights_sha256": source_sha,
                            "converted_state_equals_source": True, "tensors_compared": len(original)},
                "source_checkpoint_sha256": expected_checkpoint_sha256, "source_epoch": checkpoint["epoch"],
                "source_config_hash": checkpoint.get("config_hash"), "split_sha256": split_sha256,
                "component_sha256": {name: file_digest(path) for name, path in names.items()}}
        validate_metadata(meta)
        (folder / "wavlm_head.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bundle_spec = BundleSpecification(1, bundle_id, "wavlm_head:" + protocol, PREPROCESSING_HASH, DecisionRule(1, "threshold", .5), "genuine", "tts")
        components = {"model/wavlm_head.json": folder / "wavlm_head.json", **{"model/" + n: p for n, p in names.items()}}
        return write_bundle(destination, bundle_spec, components,
                            extra={"stage": "development_uncalibrated", "threshold_status": "selection_reference_only",
                                   "source_checkpoint_sha256": expected_checkpoint_sha256, "R3_N1_required": True})
