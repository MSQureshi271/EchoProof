"""Isolated baseline fixture timing for P1-12; never detector evaluation."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import sys
import subprocess

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import psutil
import torch
import joblib
from sklearn.base import clone
from echoproof.artifacts import atomic_write_new, create_run, digest, make_manifest, write_json
from echoproof.data.audio import decode_audio
from echoproof.data.fixtures import create_fixtures, read_fixture
from echoproof.environment import capture_environment
from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH, handcrafted, log_mel
from echoproof.models.lr import FixtureLRAdapter
from echoproof.models.bigru import BiGRU
from echoproof.training.seeding import seed_everything


def _measure(function, *, warmup, count, limit, device, samples_per_iteration, unit="batch"):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    started = time.monotonic()
    completed_warmup = 0
    for _ in range(warmup):
        if time.monotonic() - started >= limit:
            break
        function()
        sync()
        completed_warmup += 1
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    timings = []
    try:
        for _ in range(count):
            if time.monotonic() - started >= limit:
                break
            sync()
            before = time.perf_counter()
            function()
            sync()
            timings.append(time.perf_counter() - before)
            peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    return {"unit": unit, "samples_per_iteration": samples_per_iteration,
            "warmup_requested": warmup, "warmup_completed": completed_warmup,
            "requested_count": count, "measured_count": len(timings), "seconds": timings,
            "p50_seconds": float(np.percentile(timings, 50)) if timings else None,
            "p95_seconds": float(np.percentile(timings, 95)) if timings else None,
            "samples_per_second": samples_per_iteration / float(np.mean(timings)) if timings else None,
            "rss_baseline_bytes": baseline, "sampled_peak_rss_bytes": peak[0],
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "truncated": len(timings) != count or completed_warmup != warmup}


def _cold_process_load(kind, path, device):
    code = """import json,sys,time
start=time.perf_counter()
if sys.argv[1]=='lr':
    from echoproof.models.lr import FixtureLRAdapter
    model=FixtureLRAdapter.load(sys.argv[2])
else:
    import torch
    from echoproof.models.bigru import BiGRU
    checkpoint=torch.load(sys.argv[2],map_location='cpu',weights_only=True)
    model=BiGRU()
    model.load_state_dict(checkpoint['model'])
    model.to(sys.argv[3])
    if sys.argv[3]=='cuda': torch.cuda.synchronize()
print(json.dumps({'imports_and_load_seconds':time.perf_counter()-start}))
"""
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-c", code, kind, str(path), str(device)], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError("cold-process loading failed: " + result.stderr[-1000:])
    return {**json.loads(result.stdout), "wall_seconds": time.perf_counter() - started,
            "repetitions": 1, "process_cold": True, "filesystem_cache_controlled": False}


def run_benchmark(config, project_root, *, models=("lr", "bigru"), batch_size=4, accumulation_steps=1, lr_fit_rows=256):
    if set(models) != {"lr", "bigru"} or len(models) != 2:
        raise ValueError("This baseline pass requires --models lr bigru; pretrained paths are deferred")
    if config.audio.sample_rate != 16000 or config.audio.crop_samples != 64000 or config.probe.precision != "fp32":
        raise ValueError("Baseline profiling currently requires 16 kHz, four-second crops and fp32")
    for value in (batch_size, accumulation_steps, lr_fit_rows):
        if type(value) is not int or value < 1:
            raise ValueError("benchmark dimensions must be positive integers")
    if lr_fit_rows < 8:
        raise ValueError("lr_fit_rows must be at least 8")
    torch.set_num_threads(1)
    seeds = seed_everything(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose CPU explicitly or reconnect")
    device = torch.device("cuda" if config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available()) else "cpu")
    cpu = torch.device("cpu")
    run_id, directory = create_run(config.paths.run_root)
    started = time.monotonic()
    fixtures, manifest = create_fixtures(config.paths.fixtures, config.seed)
    raw = read_fixture(fixtures, manifest, "mono_4s.wav")
    base = decode_audio(raw).samples
    waves = [np.roll(base, i * 17) * (0.5 + (i % 4) * 0.1) for i in range(batch_size)]
    fixture_waves = [base, base * 0.5, -base, np.zeros_like(base)]
    adapter = FixtureLRAdapter.fit(fixture_waves, [0, 1, 0, 1], config.seed)
    raw_features = np.stack([handcrafted(w) for w in waves])
    training = np.tile(np.stack([handcrafted(w) for w in fixture_waves]), (int(np.ceil(lr_fit_rows / 4)), 1))[:lr_fit_rows]
    labels = np.arange(lr_fit_rows) % 2
    sequence = np.stack([log_mel(w) for w in waves])
    normalized = (sequence - sequence.mean(axis=(0, 1), keepdims=True)) / np.maximum(sequence.std(axis=(0, 1), keepdims=True), 1e-5)
    inputs = torch.tensor(normalized, dtype=torch.float32, device=device)
    lengths = torch.full((batch_size,), inputs.shape[1], dtype=torch.int64)
    target = torch.arange(batch_size, device=device) % 2
    network = BiGRU().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    path_settings = {"warmup": config.probe.warmup_batches, "count": config.probe.measured_batches,
                     "limit": config.probe.timing_limit_seconds}
    paths = {}
    def record(name, function, accelerator=False, samples=None, unit="batch"):
        print(f"Measuring {name} ...", flush=True, file=sys.stderr)
        try:
            paths[name] = _measure(function, **path_settings, device=device if accelerator else cpu,
                                   samples_per_iteration=samples or batch_size, unit=unit)
        except Exception as exc:
            write_json(directory / "benchmark_failure.json", {"path": name, "error": str(exc), "completed_paths": paths, "device": str(device)})
            write_json(directory / "status.json", {"status": "failed", "run_id": run_id})
            raise
    def fit_grid():
        for c in (0.01, 0.1, 1.0, 10.0):
            for weight in (None, "balanced"):
                model = clone(adapter.pipeline).set_params(classifier__C=c, classifier__class_weight=weight)
                model.fit(training, labels)
    def forward():
        network.eval()
        with torch.no_grad():
            logits = network(inputs, lengths)
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("nonfinite benchmark logits")
    def backward():
        network.train()
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(network(inputs, lengths), target).backward()
    def train_step():
        network.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation_steps):
            loss = torch.nn.functional.cross_entropy(network(inputs, lengths), target) / accumulation_steps
            if not torch.isfinite(loss).item():
                raise RuntimeError("nonfinite benchmark loss")
            loss.backward()
        optimizer.step()
    record("decode", lambda: [decode_audio(raw) for _ in waves])
    record("drive_read_decode", lambda: [decode_audio(read_fixture(fixtures, manifest, "mono_4s.wav")) for _ in waves])
    record("lr_features", lambda: [handcrafted(w) for w in waves])
    record("lr_fit_grid", fit_grid, samples=lr_fit_rows, unit="eight_candidate_grid")
    record("lr_scoring", lambda: adapter.pipeline.predict_proba(raw_features))
    record("lr_end_to_end", lambda: adapter.pipeline.predict_proba(np.stack([handcrafted(decode_audio(read_fixture(fixtures, manifest, "mono_4s.wav")).samples) for _ in waves])))
    record("bigru_features", lambda: [log_mel(w) for w in waves])
    record("bigru_forward", forward, accelerator=True)
    def neural_end_to_end():
        features = np.stack([log_mel(decode_audio(read_fixture(fixtures, manifest, "mono_4s.wav")).samples) for _ in waves])
        features = (features - sequence.mean(axis=(0, 1), keepdims=True)) / np.maximum(sequence.std(axis=(0, 1), keepdims=True), 1e-5)
        network.eval()
        with torch.no_grad():
            network(torch.tensor(features, dtype=torch.float32, device=device), lengths)
    record("bigru_end_to_end", neural_end_to_end, accelerator=True)
    record("bigru_forward_backward", backward, accelerator=True)
    record("bigru_train_step", train_step, accelerator=True, samples=batch_size * accumulation_steps, unit="optimizer_step")
    with tempfile.TemporaryDirectory(prefix="echoproof-profile-", dir=directory) as tmp:
        lr_file, neural_file = Path(tmp) / "lr.joblib", Path(tmp) / "bigru.pt"
        adapter.save(lr_file)
        torch.save({"model": network.state_dict(), "optimizer": optimizer.state_dict()}, neural_file)
        lr_bytes, neural_bytes = lr_file.stat().st_size, neural_file.stat().st_size
        def load_neural():
            checkpoint = torch.load(neural_file, map_location="cpu", weights_only=True)
            loaded = BiGRU()
            loaded.load_state_dict(checkpoint["model"])
            loaded.to(device)
        def save_neural():
            with open(neural_file, "wb") as output:
                torch.save({"model": network.state_dict(), "optimizer": optimizer.state_dict()}, output)
                output.flush()
                os.fsync(output.fileno())
        # Disk/page caches may already be warm: these are reload, not cache-flush measurements.
        loads = {"lr_reload": _measure(lambda: FixtureLRAdapter.load(lr_file), warmup=0, count=5, limit=60, device=cpu, samples_per_iteration=1, unit="load"),
                 "bigru_reload": _measure(load_neural, warmup=0, count=5, limit=60, device=device, samples_per_iteration=1, unit="load"),
                 "bigru_checkpoint_write": _measure(save_neural, warmup=0, count=5, limit=60, device=device, samples_per_iteration=1, unit="checkpoint")}
        cold_loads = {"lr": _cold_process_load("lr", lr_file, cpu), "bigru": _cold_process_load("bigru", neural_file, device)}
    benchmark = {"schema_version": 1, "scope": "baseline_fixture_benchmark", "run_id": run_id,
                 "models": ["lr", "bigru"], "deferred_models": ["wavlm", "ast"],
                 "provisional": True, "truncated": any(p["truncated"] for p in paths.values()),
                 "batch_size": batch_size, "accumulation_steps": accumulation_steps, "lr_fit_rows": lr_fit_rows,
                 "input_seconds_per_crop": 4, "precision": "fp32", "torch_threads": torch.get_num_threads(), "loader_workers": 0,
                 "bigru_device": str(device), "bigru_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                 "lr_device": "cpu", "lr_feature_count": len(FEATURE_NAMES), "mel_frames": sequence.shape[1],
                 "bigru_parameters": sum(p.numel() for p in network.parameters()),
                 "bigru_trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
                 "lr_pipeline_bytes": lr_bytes, "bigru_checkpoint_optimizer_bytes": neural_bytes,
                 "lr_fitted_coefficient_count": int(adapter.pipeline.named_steps["classifier"].coef_.size + adapter.pipeline.named_steps["classifier"].intercept_.size),
                 "paths": paths, "loads": loads, "cold_process_loads": cold_loads, "nonmeasured_wall_seconds_including_setup_warmup_reload": time.monotonic() - started - sum(sum(p["seconds"]) for p in paths.values()),
                 "profiling_wall_seconds": time.monotonic() - started,
                 "feature_schema_hash": FEATURE_SCHEMA_HASH, "fixture_manifest": manifest,
                 "limitations": ["Generated repeated signals and synthetic LR rows, not research throughput.",
                                 "Cached tensor timing excludes data-loader overhead; separate warm read/decode and end-to-end paths include fixture file access.",
                                 "Reload timing includes process-local disk/page caching and is not cold application startup.",
                                 "RSS is sampled whole-process memory, not an exact allocation peak."]}
    write_json(directory / "benchmark.json", benchmark)
    write_json(directory / "resolved_config.json", asdict(config))
    write_json(directory / "environment.json", {**capture_environment(project_root), "seeds": seeds, "task": "P1-12"})
    write_json(directory / "artifact_manifest.json", make_manifest(directory, ["benchmark.json", "resolved_config.json", "environment.json"]))
    write_json(directory / "status.json", {"status": "provisional", "run_id": run_id, "elapsed_seconds": time.monotonic() - started})
    summaries = Path(project_root) / "reports/phase1"
    summaries.mkdir(parents=True, exist_ok=True)
    if not (summaries / "benchmark.json").exists():
        write_json(summaries / "benchmark.json", benchmark)
    return {"status": "provisional", "run_dir": str(directory), "benchmark_file": str(directory / "benchmark.json"),
            "truncated": benchmark["truncated"], "bigru_device_name": benchmark["bigru_device_name"]}
