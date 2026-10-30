"""CPU model compatibility checks using generated signals and dummy labels."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
import numpy as np
import torch
import yaml
from echoproof.artifacts import atomic_write_new, canonical_json, create_run, digest, make_manifest, write_json
from echoproof.data.audio import decode_audio
from echoproof.data.fixtures import create_fixtures, read_fixture
from echoproof.environment import capture_environment
from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH, SCHEMA
from echoproof.models.lr import FixtureLRAdapter
from echoproof.models.bigru import FixtureBiGRUAdapter
from echoproof.training.seeding import seed_everything


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_probe(kind, config, project_root):
    if kind not in ("lr", "bigru"):
        raise ValueError("unknown model probe")
    if config.audio.sample_rate != 16000:
        raise ValueError("P1-08 features require 16000 Hz")
    torch.set_num_threads(1)
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    seeds = seed_everything(config.seed)
    run_id, folder = create_run(config.paths.run_root)
    fixture_dir, manifest = create_fixtures(config.paths.fixtures, config.seed)
    names = ("short.wav", "mono_1s.wav", "mono_4s.wav", "silence.wav")
    waves = [decode_audio(read_fixture(fixture_dir, manifest, name), target_rate=16000).samples for name in names]
    chunks = [(f"fixture:chunk-{i}", f"fixture:recording-{i}", signal) for i, signal in enumerate(waves)]
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    metrics = {"schema_version": 1, "scope": "software_probe", "fixture_only": True,
               "labels_are_dummy": True, "feature_schema_hash": FEATURE_SCHEMA_HASH,
               "fixture_manifest": manifest, "fixture_directory": str(fixture_dir),
               "input_samples": [len(w) for w in waves]}
    if kind == "lr":
        model = FixtureLRAdapter.fit(waves, labels, config.seed)
        batch = model.prepare_features(chunks)
        scores = model.score_chunks(batch)
        artifact = "disposable_lr.joblib"
        model.save(folder / artifact)
        reloaded = FixtureLRAdapter.load(folder / artifact)
        new_scores = reloaded.score_chunks(batch)
        difference = float(np.max(np.abs(np.asarray([s.tts_probability for s in scores]) - np.asarray([s.tts_probability for s in new_scores]))))
        _require(difference <= 1e-12, "LR reload scores differ")
        metrics.update({"feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
                        "feature_values_finite_after_imputation": bool(np.isfinite(batch.values).all()),
                        "class_order": [int(x) for x in model.pipeline.named_steps["classifier"].classes_],
                        "scores": [s.tts_probability for s in scores], "reload_max_abs_difference": difference,
                        "reload_tolerance": 1e-12})
    else:
        model = FixtureBiGRUAdapter.from_fixtures(waves)
        batch = model.prepare_features(chunks)
        x = torch.as_tensor(batch.values, dtype=torch.float32)
        lengths = torch.as_tensor(batch.lengths, dtype=torch.int64)
        model.model.eval()
        with torch.no_grad():
            logits = model.model(x, lengths)
            singles = torch.cat([model.model(x[i:i+1, :int(lengths[i])], lengths[i:i+1]) for i in range(len(chunks))])
            altered = x.clone()
            altered[~torch.as_tensor(batch.mask)] = 123.0
            padding_logits = model.model(altered, lengths)
        _require(tuple(logits.shape) == (len(chunks), 2) and torch.isfinite(logits).all().item(), "invalid BiGRU logits")
        difference = float(torch.max(torch.abs(logits - singles)).item())
        padding_difference = float(torch.max(torch.abs(logits - padding_logits)).item())
        _require(difference <= 1e-5 and padding_difference <= 1e-5, "BiGRU padding or batch mismatch")
        scores = model.score_chunks(batch)
        model.model.train()
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model.model(x, lengths), torch.as_tensor(labels))
        _require(torch.isfinite(loss).item(), "nonfinite loss")
        loss.backward()
        grads = [p.grad for p in model.model.parameters() if p.requires_grad]
        _require(all(g is not None and torch.isfinite(g).all().item() for g in grads) and any(torch.count_nonzero(g).item() for g in grads), "missing or invalid gradients")
        before = model.model.head.weight.detach().clone()
        optimizer.step()
        _require(not torch.equal(before, model.model.head.weight.detach()), "optimizer did not update head")
        artifact = "disposable_bigru.pt"
        model.save(folder / artifact)
        reloaded = FixtureBiGRUAdapter.load(folder / artifact)
        trained_scores = model.score_chunks(batch)
        reload_scores = reloaded.score_chunks(batch)
        reload_diff = max(abs(a.tts_probability - b.tts_probability) for a, b in zip(trained_scores, reload_scores))
        _require(reload_diff <= 1e-6, "BiGRU reload mismatch")
        metrics.update({"frame_lengths": batch.lengths.tolist(), "feature_shape": list(batch.values.shape),
                        "logit_shape": list(logits.shape), "scores_before_disposable_step": [s.tts_probability for s in scores],
                        "single_batch_max_abs_difference": difference, "padding_max_abs_difference": padding_difference,
                        "padding_tolerance": 1e-5, "loss": float(loss.detach().item()), "gradients_finite": True,
                        "head_updated": True, "reload_max_abs_difference": reload_diff, "reload_tolerance": 1e-6,
                        "normalization_scope": "fixture_only"})
    env = capture_environment(project_root)
    env.update({"schema_version": 1, "run_id": run_id, "task": "P1-08", "started_utc": started,
                "config_hash": digest(asdict(config)), "seeds": seeds, "device": "cpu", "model": kind})
    atomic_write_new(folder / "resolved_config.yaml", yaml.safe_dump(asdict(config), sort_keys=True).encode("utf-8"))
    write_json(folder / "environment.json", env)
    atomic_write_new(folder / "events.jsonl", (canonical_json({"event": "fixture_model_probe_complete", "model": kind}) + "\n").encode("utf-8"))
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "feature_schema.json", {"schema_version": 1, "hash": FEATURE_SCHEMA_HASH, "settings": SCHEMA, "names": FEATURE_NAMES, "scope": "software_probe"})
    names_to_hash = ["resolved_config.yaml", "environment.json", "events.jsonl", "metrics.json", "feature_schema.json", artifact]
    write_json(folder / "artifact_manifest.json", make_manifest(folder, names_to_hash))
    result = subprocess.run([sys.executable, "-c", "import json,sys; from echoproof.artifacts import verify_manifest; print(json.dumps(verify_manifest(sys.argv[1])))", str(folder)], capture_output=True, text=True, timeout=60)
    _require(result.returncode == 0, "artifact readback failed: " + result.stderr[-1000:])
    verification = json.loads(result.stdout)
    write_json(folder / "status.json", {"schema_version": 1, "status": "success", "run_id": run_id,
               "elapsed_seconds": time.monotonic() - start, "verification": verification})
    return {"status": "success", "run_id": run_id, "run_dir": str(folder), "model": kind,
            "scope": "software_probe", "verification": verification}
