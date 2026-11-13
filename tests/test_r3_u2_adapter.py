"""R3-U2 AST adapter, bundle and parity on a tiny random AST (real transformers code paths, no large download).

They need torch and transformers and are meant to run on Colab; where a dependency is missing they skip, and the Colab notebook refuses to continue past
a skip. The tiny model carries its own identity (`source_spec`, `processor_expectations`), so nothing here says anything about the pinned AST weights:
that identity is checked in the real packaging run.
"""
import copy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError, load_bundle
from echoproof.models.ast_adapter import ASTBundleAdapter
from echoproof.models.ast_detector import ASTDetector
from echoproof.models.base import InferenceModelAdapter
from echoproof.models.r3_u2_bundle import package_ast_detector
from echoproof.training import r3_u2_parity as P

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r1_u2_ast import tiny_feature_extractor, tiny_snapshot  # noqa: E402

RATE = 16000
REVISION = "0123456789abcdef0123456789abcdef01234567"
PROCESSOR = {"do_normalize": True, "sampling_rate": 16000, "num_mel_bins": 16, "max_length": 64, "mean": -4.27, "std": 4.57}
CONFIG_HASH, WARMUP, RECIPE, SPLIT = "c" * 64, "e" * 64, "d" * 64, "0" * 64


def perturb(detector, scale=0.05, seed=0):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in detector.model.parameters():
            if p.requires_grad:
                p.add_(scale * torch.randn(p.shape, generator=generator))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("r3u2adapter")
    snapshot = root / "snapshots" / REVISION
    tiny_snapshot(snapshot)
    extractor = tiny_feature_extractor()
    try:
        extractor([np.zeros(RATE, dtype=np.float32)], sampling_rate=RATE, return_tensors="pt")
    except Exception:
        pytest.skip("AST feature extractor backend unavailable")
    extractor.save_pretrained(snapshot)
    spec = {"repo_id": "tiny/ast", "revision": REVISION, "weights_file": "model.safetensors", "weights_sha256": file_digest(snapshot / "model.safetensors")}
    fresh = ASTDetector.from_dir(snapshot, seed=1, feature_extractor=extractor)
    pins = {g: h for g, h in fresh.digests().items() if g != "classifier"}
    tuned = ASTDetector.from_dir(snapshot, seed=1, feature_extractor=extractor)
    tuned.set_stage("partial", 2)
    perturb(tuned)
    ckpt = root / "best.pt"
    torch.save({"epoch": 1, "model": tuned.state_dict(), "metrics": {"macro_f1": .8}, "config": {"precision": "fp16"}, "config_hash": CONFIG_HASH}, ckpt)
    unchanged = root / "unchanged.pt"
    torch.save({"epoch": 1, "model": fresh.state_dict(), "metrics": {}, "config": {"precision": "fp16"}, "config_hash": CONFIG_HASH}, unchanged)
    world = SimpleNamespace(root=root, snapshot=snapshot, spec=spec, extractor=extractor, pins=pins, ckpt=ckpt, unchanged=unchanged, tuned=tuned)
    world.bundle = root / "bundle"
    package(world, root, dest=world.bundle)
    return world


def package(world, tmp_path, **overrides):
    args = dict(bundle_id="bundle:test-ast", protocol="B", expected_checkpoint_sha256=file_digest(world.ckpt), split_sha256=SPLIT, expected_epoch=1,
                expected_config_hash=CONFIG_HASH, recipe_sha256=RECIPE, warmup_checkpoint_sha256=WARMUP, pretrained_group_sha256=world.pins,
                unfrozen_blocks=2, experiment="baseline", source_spec=world.spec, processor_expectations=PROCESSOR, feature_extractor=world.extractor)
    args.update(overrides)
    return package_ast_detector(args.pop("checkpoint", world.ckpt), args.pop("snapshot", world.snapshot), args.pop("dest", tmp_path / "out"), **args)


def waves(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.1, size).astype(np.float32) for size in (4000, 16000, 16000, 6400, 9000)[:n]]


def reference_scores(world, clips):
    """What the tuned detector gives on the same clips through the training-time code path: the number the adapter must reproduce."""
    detector = world.tuned.eval()
    out = []
    with torch.no_grad():
        for clip in clips:
            logits = detector(torch.from_numpy(clip)[None, :], torch.tensor([len(clip)]))
            out.append(float(torch.softmax(logits.float(), dim=1)[0, 1]))
    return np.array(out)


# ---- adapter -----------------------------------------------------------------------------------------------------------------------------------

def test_bundle_is_verified_and_the_adapter_satisfies_the_contract(world):
    bundle = load_bundle(world.bundle)
    assert bundle.spec.bundle_id == "bundle:test-ast" and bundle.spec.model_version == "ast_partial:B"
    adapter = ASTBundleAdapter.load(world.bundle)
    assert isinstance(adapter, InferenceModelAdapter)
    meta = adapter.metadata
    assert meta["model"] == "ast_partial" and meta["score_type"] == "raw_uncalibrated" and meta["inference_precision"] == "fp32"
    assert meta["export_precision"] == "fp16" and meta["warmup_checkpoint_sha256"] == WARMUP
    assert meta["unfrozen_groups"] == "block.02,block.03,classifier,final_norm"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())
    with pytest.raises(TypeError):
        meta["x"] = "y"                                                            # read-only


def test_adapter_scores_equal_the_training_time_code_path(world):
    clips = waves(5)
    got = ASTBundleAdapter.load(world.bundle).score_chunks(clips)
    assert got.shape == (5,) and got.dtype == np.float64 and np.all((got >= 0) & (got <= 1))
    assert np.allclose(got, reference_scores(world, clips), atol=1e-6, rtol=0)      # serialisation changed no tensor


def test_adapter_is_batch_size_independent_order_preserving_and_handles_empty_input(world):
    clips = waves(5)
    a, b = ASTBundleAdapter.load(world.bundle, batch_size=1), ASTBundleAdapter.load(world.bundle, batch_size=4)
    forward, batched = a.score_chunks(clips), b.score_chunks(clips)
    assert np.allclose(forward, batched, atol=1e-5)
    assert np.allclose(b.score_chunks(clips[::-1])[::-1], batched, atol=1e-5)
    assert b.score_chunks([]).shape == (0,)


@pytest.mark.parametrize("bad", [np.full(4000, np.nan, np.float32), np.zeros((2, 4000), np.float32), np.zeros(0, np.float32),
                                 np.zeros(64001, np.float32), np.ones(4000, np.int16), np.full(4000, np.inf, np.float32)])
def test_adapter_rejects_invalid_waveforms(world, bad):
    with pytest.raises(ValueError):
        ASTBundleAdapter.load(world.bundle).score_chunks([bad])


def test_loading_requires_a_verified_bundle_and_a_valid_precision(world, tmp_path):
    with pytest.raises(BundleError):
        ASTBundleAdapter.load(tmp_path)
    with pytest.raises(ValueError, match="CUDA"):
        ASTBundleAdapter.load(world.bundle, device="cpu", precision="fp16")
    with pytest.raises(ValueError, match="precision"):
        ASTBundleAdapter.load(world.bundle, precision="bf16")
    with pytest.raises(ValueError, match="batch_size"):
        ASTBundleAdapter.load(world.bundle, batch_size=0)


def test_a_tampered_component_is_rejected_before_use(world, tmp_path):
    copy_dir = tmp_path / "bundle"
    shutil.copytree(world.bundle, copy_dir)
    weights = next(copy_dir.rglob("model.safetensors"))
    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(BundleError):
        ASTBundleAdapter.load(copy_dir)


# ---- packaging refusals -----------------------------------------------------------------------------------------------------------------------

def test_packaging_refuses_each_inconsistency_and_leaves_nothing_behind(world, tmp_path):
    with pytest.raises(FileExistsError):
        package(world, tmp_path, dest=world.bundle)
    with pytest.raises(ValueError, match="selected checkpoint hash"):
        package(world, tmp_path, expected_checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="epoch"):
        package(world, tmp_path, expected_epoch=2)
    with pytest.raises(ValueError, match="configuration"):
        package(world, tmp_path, expected_config_hash="d" * 64)
    with pytest.raises(ValueError, match="pinned SHA-256"):
        package(world, tmp_path, source_spec=dict(world.spec, weights_sha256="0" * 64))
    wrong = tmp_path / "snapshots" / "not-the-revision"
    shutil.copytree(world.snapshot, wrong)
    with pytest.raises(ValueError, match="pinned revision"):
        package(world, tmp_path, snapshot=wrong)
    with pytest.raises(ValueError, match="preprocessing"):
        package(world, tmp_path, processor_expectations=dict(PROCESSOR, num_mel_bins=128))
    assert not (tmp_path / "out").exists()


def test_packaging_checks_the_frozen_and_unfrozen_groups(world, tmp_path):
    with pytest.raises(ValueError, match="frozen group block.00"):
        package(world, tmp_path, pretrained_group_sha256={**world.pins, "block.00": "0" * 64})
    with pytest.raises(ValueError, match="unfrozen group"):
        package(world, tmp_path, checkpoint=world.unchanged, expected_checkpoint_sha256=file_digest(world.unchanged))
    bad = torch.load(world.ckpt, map_location="cpu", weights_only=True)
    key = next(k for k in bad["model"] if k.endswith("classifier.dense.weight"))
    bad["model"][key] = bad["model"][key].clone()
    bad["model"][key][0, 0] = float("nan")
    path = tmp_path / "nan.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="non-finite"):
        package(world, tmp_path, checkpoint=path, expected_checkpoint_sha256=file_digest(path))
    short = torch.load(world.ckpt, map_location="cpu", weights_only=True)
    short["model"].pop(next(iter(short["model"])))
    path2 = tmp_path / "short.pt"
    torch.save(short, path2)
    with pytest.raises(RuntimeError):
        package(world, tmp_path, checkpoint=path2, expected_checkpoint_sha256=file_digest(path2))    # strict load: no silent partial loads


def test_the_bundle_records_the_evidence_it_was_built_on(world):
    import json
    meta = json.loads(next(Path(world.bundle).rglob("ast_detector.json")).read_text(encoding="utf-8"))
    assert meta["conversion"]["state_equals_checkpoint"] is True and meta["conversion"]["tensors_compared"] > 0
    assert meta["fine_tuning"]["frozen_groups_equal_pinned"] and meta["fine_tuning"]["unfrozen_groups_differ_from_pinned"]
    assert meta["source_checkpoint_sha256"] == file_digest(world.ckpt) and meta["source_epoch"] == 1 and meta["recipe_sha256"] == RECIPE
    assert meta["model"]["num_labels"] == 2 and meta["base"]["source_weights_sha256"] == world.spec["weights_sha256"]


# ---- parity helpers -----------------------------------------------------------------------------------------------------------------------------

def fake_dataset():
    comps = {}
    chunks = []
    for comp in ("genuine", "melo", "vits", "xtts", "chatterbox"):
        for k in range(6):
            rid = f"phonemedf:{comp}:{k}"
            comps[rid] = comp
            chunks.append(SimpleNamespace(chunk_id=f"chunk:B:{rid}:0", recording_id=rid, valid_samples=64000 if k % 3 else 20000))
    return SimpleNamespace(chunks=chunks), comps


def test_chunk_selection_is_deterministic_stratified_and_scoped():
    dataset, comps = fake_dataset()
    a, b = P.select_chunks(dataset, comps, 20), P.select_chunks(dataset, comps, 20)
    assert a == b and len(set(a)) == 20
    picked = {comps[dataset.chunks[i].recording_id] for i in a}
    assert picked == {"genuine", "melo", "vits", "xtts"} and any(dataset.chunks[i].valid_samples < 64000 for i in a)
    with pytest.raises(P.ParityError):
        P.select_chunks(dataset, comps, 500)
    with pytest.raises(P.ParityError):
        P.select_chunks(dataset, comps, 0)


def test_parity_gate_and_reported_measurements():
    expected = np.array([.1, .9, .5001, .7])
    report = P.parity_check(["a", "b", "c", "d"], expected, scores_fp16=expected + 0.004, others={"fp32_cpu": expected + 0.03})
    assert report["passed"] is True and report["fp16"]["decision_flips_at_0.5"] == 0
    assert report["reported_not_gated"]["fp32_cpu"]["max_abs_difference"] == pytest.approx(0.03)
    assert P.parity_check(["a", "b", "c", "d"], expected, scores_fp16=expected + 0.05)["passed"] is False
    with pytest.raises(P.ParityError):
        P.parity_check(["a"], expected, scores_fp16=expected)
    with pytest.raises(P.ParityError):
        P.parity_check(["a", "b", "c", "d"], expected, scores_fp16=np.array([np.nan, .1, .1, .1]))
