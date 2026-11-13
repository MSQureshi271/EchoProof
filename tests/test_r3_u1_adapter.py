"""R3-U1 adapter, bundle and parity on a tiny random WavLM (real transformers code paths, no large download).

These tests need `transformers`; they are skipped where it is not installed. The tiny encoder carries its own identity (`source_spec`), so nothing here
claims anything about the pinned WavLM weights: that identity is checked in the real parity run.
"""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle
from echoproof.models.base import InferenceModelAdapter
from echoproof.models.r3_u1_bundle import package_wavlm_head
from echoproof.models.wavlm_adapter import WavLMBundleAdapter
from echoproof.models.wavlm_head import WavLMHead
from echoproof.training import r3_u1_parity as P

HIDDEN = 16
PROCESSOR = {"do_normalize": False, "feature_extractor_type": "Wav2Vec2FeatureExtractor", "feature_size": 1, "padding_side": "right",
             "padding_value": 0.0, "return_attention_mask": True, "sampling_rate": 16000}
REVISION = "0123456789abcdef0123456789abcdef01234567"


def tiny_encoder():
    config = transformers.WavLMConfig(hidden_size=HIDDEN, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, conv_dim=(8, 8),
                                      conv_stride=(5, 2), conv_kernel=(10, 3), num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                                      num_buckets=8, max_bucket_distance=16, mask_time_prob=0.0, mask_feature_prob=0.0)
    torch.manual_seed(0)
    return transformers.WavLMModel(config).eval()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("r3u1")
    snapshot = root / "snapshots" / REVISION
    encoder = tiny_encoder()
    encoder.save_pretrained(snapshot, safe_serialization=True)
    (snapshot / "preprocessor_config.json").write_text(json.dumps(PROCESSOR), encoding="utf-8")
    spec = {"repo_id": "tiny/wavlm", "revision": REVISION, "weights_file": "model.safetensors",
            "weights_sha256": file_digest(snapshot / "model.safetensors")}
    torch.manual_seed(1)
    head = WavLMHead(HIDDEN, hidden=8, dropout=0.2)
    ckpt = root / "best.pt"
    torch.save({"epoch": 3, "model": head.state_dict(), "metrics": {"macro_f1": 0.8}, "config": {}, "config_hash": "c" * 64, "pooling": "mean",
                "architecture": head.architecture()}, ckpt)
    bundle = root / "bundle"
    package_wavlm_head(ckpt, snapshot, bundle, bundle_id="bundle:test-wavlm", protocol="B", expected_checkpoint_sha256=file_digest(ckpt),
                       split_sha256="0" * 64, expected_epoch=3, expected_config_hash="c" * 64, source_spec=spec)
    return {"root": root, "snapshot": snapshot, "spec": spec, "head": head, "ckpt": ckpt, "bundle": bundle, "encoder": encoder}


def waves(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.1, size).astype(np.float32) for size in (4000, 16000, 16000, 6400)[:n]]


def reference_scores(world, clips):
    """What the head gives on embeddings from the original (unconverted) tiny encoder: the number the adapter must reproduce."""
    from echoproof.features.pretrained import WavLMExtractor
    extractor = WavLMExtractor(world["encoder"], transformers.Wav2Vec2FeatureExtractor(**{k: v for k, v in PROCESSOR.items()
                                                                                          if k != "feature_extractor_type"}))
    return world["head"].scores(torch.from_numpy(extractor.embed(clips, pooling="mean"))).double().numpy(), extractor


# ---- adapter -----------------------------------------------------------------------------------------------------------------------------------

def test_bundle_is_verified_and_the_adapter_satisfies_the_contract(world):
    bundle = load_bundle(world["bundle"])
    assert bundle.spec.bundle_id == "bundle:test-wavlm" and bundle.spec.model_version == "wavlm_head:B"
    adapter = WavLMBundleAdapter.load(world["bundle"])
    assert isinstance(adapter, InferenceModelAdapter)
    meta = adapter.metadata
    assert meta["model"] == "wavlm_head" and meta["score_type"] == "raw_uncalibrated" and meta["pooling"] == "mean" and meta["protocol"] == "B"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())
    with pytest.raises(TypeError):
        meta["x"] = "y"                                                            # read-only


def test_adapter_scores_equal_the_direct_pipeline_exactly(world):
    clips = waves()
    expected, _ = reference_scores(world, clips)
    got = WavLMBundleAdapter.load(world["bundle"]).score_chunks(clips)
    assert got.shape == (4,) and got.dtype == np.float64 and np.all((got >= 0) & (got <= 1))
    assert np.allclose(got, expected, atol=1e-6, rtol=0)                          # the safetensors conversion is verified tensor-identical


def test_adapter_is_order_preserving_and_handles_empty_input(world):
    adapter = WavLMBundleAdapter.load(world["bundle"])
    clips = waves()
    forward, backward = adapter.score_chunks(clips), adapter.score_chunks(clips[::-1])
    assert np.allclose(forward, backward[::-1], atol=1e-6)
    assert adapter.score_chunks([]).shape == (0,)


@pytest.mark.parametrize("bad", [np.full(4000, np.nan, np.float32), np.zeros((2, 4000), np.float32), np.zeros(0, np.float32),
                                 np.zeros(64001, np.float32), np.ones(4000, np.int16), np.full(4000, np.inf, np.float32)])
def test_adapter_rejects_invalid_waveforms(world, bad):
    with pytest.raises(ValueError):
        WavLMBundleAdapter.load(world["bundle"]).score_chunks([bad])


def test_a_waveform_too_short_for_the_encoder_is_rejected(world):
    with pytest.raises(ValueError, match="too short"):
        WavLMBundleAdapter.load(world["bundle"]).score_chunks([np.zeros(5, np.float32)])


def test_loading_requires_a_verified_bundle(world, tmp_path):
    with pytest.raises(BundleError):
        WavLMBundleAdapter.load(tmp_path)


def test_a_tampered_component_is_rejected_before_use(world, tmp_path):
    copy_dir = tmp_path / "bundle"
    shutil.copytree(world["bundle"], copy_dir)
    weights = next(copy_dir.rglob("head_weights.pt"))
    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(BundleError):
        WavLMBundleAdapter.load(copy_dir)


# ---- packaging refusals --------------------------------------------------------------------------------------------------------------------------

def package(world, tmp_path, **overrides):
    args = dict(bundle_id="bundle:t", protocol="B", expected_checkpoint_sha256=file_digest(world["ckpt"]), split_sha256="0" * 64,
                expected_epoch=3, expected_config_hash="c" * 64, source_spec=world["spec"])
    args.update(overrides)
    return package_wavlm_head(args.pop("checkpoint", world["ckpt"]), args.pop("snapshot", world["snapshot"]), args.pop("dest", tmp_path / "out"), **args)


def test_packaging_refuses_each_inconsistency(world, tmp_path):
    with pytest.raises(FileExistsError):
        package(world, tmp_path, dest=world["bundle"])
    with pytest.raises(ValueError, match="selected checkpoint hash"):
        package(world, tmp_path, expected_checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="epoch"):
        package(world, tmp_path, expected_epoch=4)
    with pytest.raises(ValueError, match="configuration"):
        package(world, tmp_path, expected_config_hash="d" * 64)
    with pytest.raises(ValueError, match="pinned SHA-256"):
        package(world, tmp_path, source_spec=dict(world["spec"], weights_sha256="0" * 64))
    wrong_name = tmp_path / "snapshots" / "not-the-revision"
    shutil.copytree(world["snapshot"], wrong_name)
    with pytest.raises(ValueError, match="pinned revision"):
        package(world, tmp_path, snapshot=wrong_name)
    assert not (tmp_path / "out").exists()                                          # a refused package leaves nothing behind


def test_packaging_refuses_a_head_that_does_not_fit_the_encoder(world, tmp_path):
    other = WavLMHead(HIDDEN * 2, hidden=8)                                         # mean+std width, but the checkpoint says mean pooling
    ckpt = tmp_path / "wide.pt"
    torch.save({"epoch": 3, "model": other.state_dict(), "metrics": {}, "config": {}, "config_hash": "c" * 64, "pooling": "mean",
                "architecture": other.architecture()}, ckpt)
    with pytest.raises(ValueError, match="input width"):
        package(world, tmp_path, checkpoint=ckpt, expected_checkpoint_sha256=file_digest(ckpt))


def test_packaging_refuses_non_finite_head_weights_and_wrong_preprocessing(world, tmp_path):
    bad_state = {k: v.clone() for k, v in world["head"].state_dict().items()}
    bad_state["net.0.bias"][0] = float("nan")
    ckpt = tmp_path / "nan.pt"
    torch.save({"epoch": 3, "model": bad_state, "metrics": {}, "config": {}, "config_hash": "c" * 64, "pooling": "mean",
                "architecture": world["head"].architecture()}, ckpt)
    with pytest.raises(ValueError, match="non-finite"):
        package(world, tmp_path, checkpoint=ckpt, expected_checkpoint_sha256=file_digest(ckpt))
    snap = tmp_path / "snapshots" / REVISION
    shutil.copytree(world["snapshot"], snap)
    (snap / "preprocessor_config.json").write_text(json.dumps(dict(PROCESSOR, do_normalize=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessing"):
        package(world, tmp_path, snapshot=snap, dest=tmp_path / "out2")


# ---- parity -----------------------------------------------------------------------------------------------------------------------------------------

def rows_for(n):
    comps = ["genuine", "melo", "vits", "xtts"]
    return [{"item_id": f"chunk:B:phonemedf:{comps[i % 4]}:{i}-1-0:0", "component": comps[i % 4], "role": "selection",
             "valid_samples": 16000 if i % 2 else 4000, "kind": "chunk", "partition": "validation", "recording_id": f"phonemedf:{comps[i % 4]}:{i}-1-0"}
            for i in range(n)]


def test_parity_passes_when_the_cache_is_the_direct_embedding(world):
    clips = waves()
    _, extractor = reference_scores(world, clips)
    cached = extractor.embed(clips, pooling="mean")
    adapter = WavLMBundleAdapter.load(world["bundle"])
    report = P.parity_check(rows_for(4), clips, cached, world["head"], embed_fn=lambda w: extractor.embed(w, pooling="mean"), pooling="mean",
                            score_fn=adapter.score_chunks)
    assert report["passed"] is True and report["max_abs_embedding_difference"] == 0.0
    assert report["max_abs_adapter_difference"] < 1e-6 and report["max_decision_flips_at_0.5"] == 0 and report["items"] == 4


def test_parity_detects_a_drifted_cache_and_a_drifted_adapter(world):
    clips = waves()
    _, extractor = reference_scores(world, clips)
    cached = extractor.embed(clips, pooling="mean")
    embed = lambda w: extractor.embed(w, pooling="mean")
    drifted = cached.copy()
    drifted[1] += 0.01
    assert P.parity_check(rows_for(4), clips, drifted, world["head"], embed_fn=embed, pooling="mean")["passed"] is False
    assert P.parity_check(rows_for(4), clips, cached, world["head"], embed_fn=embed, pooling="mean",
                          score_fn=lambda w: np.full(len(w), 0.123))["passed"] is False
    with pytest.raises(P.ParityError):
        P.parity_check(rows_for(3), clips, cached, world["head"], embed_fn=embed, pooling="mean")


# ---- sample selection ---------------------------------------------------------------------------------------------------------------------------

def index_rows():
    out = []
    for role in ("selection", "calibration", "operating"):
        for comp in ("genuine", "melo", "vits", "xtts", "chatterbox"):
            for k in range(6):
                out.append({"item_id": f"chunk:B:{role}:{comp}:{k}", "recording_id": f"phonemedf:{comp}:{role}{k}", "component": comp, "role": role,
                            "partition": "validation", "kind": "chunk", "valid_samples": 64000 if k % 3 else 20000})
    out.append({"item_id": "crop:B:x:0", "recording_id": "phonemedf:genuine:x", "component": "genuine", "role": None, "partition": "train",
                "kind": "crop", "valid_samples": 64000})
    return out


def test_selection_is_deterministic_stratified_and_scoped():
    rows = index_rows()
    a, b = P.select_items(rows, 20), P.select_items(rows, 20)
    assert [r["item_id"] for r in a] == [r["item_id"] for r in b]
    assert {r["role"] for r in a} == {"selection"} and "chatterbox" not in {r["component"] for r in a}
    assert {r["component"] for r in a} == {"genuine", "melo", "vits", "xtts"} and any(r["valid_samples"] < 64000 for r in a)
    assert len({r["item_id"] for r in a}) == 20
    assert {r["role"] for r in P.select_items(rows, 20, roles=("selection", "operating"))} == {"selection", "operating"}


def test_selection_respects_availability_and_refuses_too_few():
    rows = index_rows()
    only_genuine = P.select_items(rows, 5, available=lambda r: r["component"] == "genuine")
    assert {r["component"] for r in only_genuine} == {"genuine"}
    with pytest.raises(P.ParityError):
        P.select_items(rows, 500)
    with pytest.raises(P.ParityError):
        P.select_items(rows, 0)


def test_chunk_slicing_checks_the_expected_length():
    row = {"item_id": "x", "start_sample": 10, "end_sample": 30, "valid_samples": 20}
    assert len(P._chunk(np.zeros(100, np.float32), row)) == 20
    with pytest.raises(P.ParityError):
        P._chunk(np.zeros(25, np.float32), row)                                     # window runs past the audio


def test_local_wav_mapping_only_covers_genuine_and_xtts(tmp_path):
    (tmp_path / "genuine" / "wav").mkdir(parents=True)
    (tmp_path / "genuine" / "wav" / "1-2-3.wav").write_bytes(b"x")
    assert P.local_wav_path(tmp_path, {"component": "genuine", "recording_id": "phonemedf:genuine:1-2-3"}) is not None
    assert P.local_wav_path(tmp_path, {"component": "melo", "recording_id": "phonemedf:melo:1-2-3"}) is None
    assert P.local_wav_path(tmp_path, {"component": "xtts", "recording_id": "phonemedf:xtts:1-2-3"}) is None
