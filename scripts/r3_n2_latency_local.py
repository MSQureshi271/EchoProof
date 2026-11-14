"""R3-N2 latency harness for the local development machine (batch 1, CPU, production `analyze()` path).

It follows the pattern of the Colab notebooks `notebooks/R3_N2_{LR,BiGRU}_Latency.ipynb`: stage timings are taken by wrapping the exact production
callables for the duration of one `analyze` call and restoring them afterwards; the first `analyze` of a recording is the parity reference; fresh
subprocesses measure bundle load and the first prediction separately (a cold interpreter, not a cold machine or cold disk cache).

Differences from the notebooks, on purpose:
  * durations are the plan's 1, 4, 10, 30 and 60 s;
  * each clip is the first N seconds of ONE in-memory concatenation of explicit genuine fixtures (nothing is written to disk);
  * the number of timed calls can differ by duration, because a 60 s WavLM call takes tens of seconds;
  * a derived `other` stage (total minus the listed stages) is reported so the stages account for the whole call.

Models: lr (calibrated LR bundle), wavlm (calibrated WavLM bundle), bigru (needs Nouman's calibrated BiGRU bundle and loader).
Never reads data other than the explicit fixtures; never changes a bundle. Results go to new files only.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

FIXTURES = [  # explicit genuine fixtures, concatenated in this order (about 64 s) and cut to the requested length
    "dataset/genuine/wav/19-198-0002.wav", "dataset/genuine/wav/19-198-0001.wav", "dataset/genuine/wav/19-198-0003.wav",
    "dataset/genuine/wav/911-128684-0002.wav", "dataset/genuine/wav/89-218-0004.wav", "dataset/genuine/wav/19-198-0037.wav",
]
DURATIONS = (1, 4, 10, 30, 60)
STAGES = ("decode", "chunk_construction", "feature_extraction", "model_prediction", "aggregation", "calibration", "threshold_decision")
TOLERANCE = 1e-10
SAMPLE_RATE = 16000


def source_samples(root: Path) -> np.ndarray:
    parts = []
    for rel in FIXTURES:
        data, rate = sf.read(root / rel, dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            raise ValueError(f"fixture {rel} is {rate} Hz")
        parts.append(data[:, 0])
    return np.concatenate(parts)


def clip_bytes(root: Path, seconds: int) -> bytes:
    samples = source_samples(root)
    if len(samples) < seconds * SAMPLE_RATE:
        raise ValueError(f"fixtures are shorter than {seconds} s")
    buffer = io.BytesIO()
    sf.write(buffer, samples[: seconds * SAMPLE_RATE], SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def load_bundle(model: str, bundle: Path):
    if model == "lr":
        from echoproof.inference import load_lr_bundle
        return load_lr_bundle(bundle)
    if model == "wavlm":
        from echoproof.models.r3_n1_wavlm_bundle import load_wavlm_bundle
        return load_wavlm_bundle(bundle, device="cpu")
    if model == "bigru":
        from echoproof.models.r3_n1_bigru_bundle import load_calibrated_bigru_bundle
        return load_calibrated_bigru_bundle(bundle, device="cpu", batch_size=1, allow_pending_review=True)
    raise SystemExit(f"unknown model {model}")


def model_patches(model: str, loaded):
    """(owner, attribute, stage) for the model-specific stages: feature extraction and model prediction."""
    if model == "lr":
        import echoproof.models.lr_detector as lr_detector
        from echoproof.models.lr_detector import LRDetector
        return [(lr_detector, "chunk_features", "feature_extraction"), (LRDetector, "predict_proba_features", "model_prediction")]
    adapter = loaded.scorer.__self__
    if model == "wavlm":
        from echoproof.models.wavlm_adapter import WavLMBundleAdapter
        if not isinstance(adapter, WavLMBundleAdapter):
            raise TypeError("production scorer is not a WavLMBundleAdapter")
        return [(adapter._extractor, "embed", "feature_extraction"), (adapter._head, "scores", "model_prediction")]
    if model == "bigru":
        from echoproof.models.bigru_adapter import BiGRUBundleAdapter
        if not isinstance(adapter, BiGRUBundleAdapter):
            raise TypeError("production scorer is not a BiGRUBundleAdapter")
        return [(adapter._model, "features", "feature_extraction"), (adapter._model.bigru, "forward", "model_prediction")]
    raise SystemExit(f"unknown model {model}")


def put(owner, name, value):
    """Classes and modules take a plain setattr; instances (some frozen) need object.__setattr__."""
    if isinstance(owner, type) or hasattr(owner, "__spec__"):
        setattr(owner, name, value)
    else:
        object.__setattr__(owner, name, value)


def instrumented_analyze(model: str, audio: bytes, loaded):
    import echoproof.inference as inference
    from echoproof.inference import analyze

    stages = {k: 0 for k in STAGES + ("total",)}
    calls = {k: 0 for k in stages}
    spec = loaded.spec
    originals = {"decode_audio": inference.decode_audio, "eval_windows": inference.eval_windows, "aggregate": loaded.aggregate,
                 "calibrate": loaded.calibrate, "decision_rule": spec.decision_rule}
    patches = [(inference, "decode_audio", "decode"), (inference, "eval_windows", "chunk_construction")] + model_patches(model, loaded)
    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in patches]

    def timed(key, fn):
        def call(*args, **kwargs):
            start = time.perf_counter_ns()
            calls[key] += 1
            try:
                return fn(*args, **kwargs)
            finally:
                stages[key] += time.perf_counter_ns() - start
        return call

    try:
        for owner, name, key in patches:
            put(owner, name, timed(key, getattr(owner, name)))
        object.__setattr__(loaded, "aggregate", timed("aggregation", originals["aggregate"]))
        if originals["calibrate"] is not None:
            object.__setattr__(loaded, "calibrate", timed("calibration", originals["calibrate"]))

        class DecisionProxy:
            def decide(self, score):
                return timed("threshold_decision", originals["decision_rule"].decide)(score)

        object.__setattr__(spec, "decision_rule", DecisionProxy())
        start = time.perf_counter_ns()
        analysis = analyze(audio, loaded)
        stages["total"] = time.perf_counter_ns() - start
        calls["total"] += 1
        if analysis.result.status != "success":
            raise RuntimeError(f"analysis failed: {analysis.result.error_code if hasattr(analysis.result, 'error_code') else analysis.result.status}")
        if calls["feature_extraction"] == 0 or calls["model_prediction"] == 0:
            raise RuntimeError("the production model did not invoke the instrumented feature or prediction callable")
        stages["other"] = stages["total"] - sum(stages[k] for k in STAGES)
        return analysis, stages, calls
    finally:
        for owner, name, original in saved:
            put(owner, name, original)
        object.__setattr__(loaded, "aggregate", originals["aggregate"])
        object.__setattr__(loaded, "calibrate", originals["calibrate"])
        object.__setattr__(spec, "decision_rule", originals["decision_rule"])


def pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if len(values) else None


def stats(values):
    return {"count": len(values), "p50": pct(values, 50), "p95": pct(values, 95), "mean": float(np.mean(values)) if len(values) else None,
            "min": float(np.min(values)) if len(values) else None, "max": float(np.max(values)) if len(values) else None}


def fresh_process(model: str, bundle: Path, seconds: int):
    start = time.perf_counter_ns()
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", "--model", model, "--bundle", str(bundle), "--seconds", str(seconds)],
                          capture_output=True, text=True, timeout=900, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    elapsed = time.perf_counter_ns() - start
    if proc.returncode:
        raise RuntimeError(proc.stderr[-1500:])
    return {"process_to_first_prediction_ns": elapsed, **json.loads(proc.stdout.strip().splitlines()[-1])}


def child(args):
    root = ROOT
    raw = clip_bytes(root, args.seconds)
    t = time.perf_counter_ns()
    loaded = load_bundle(args.model, Path(args.bundle))
    load_ns = time.perf_counter_ns() - t
    from echoproof.inference import analyze
    t = time.perf_counter_ns()
    result = analyze(raw, loaded)
    first_ns = time.perf_counter_ns() - t
    print(json.dumps({"bundle_load_ns": load_ns, "first_prediction_ns": first_ns, "status": result.result.status, "score": result.result.score,
                      "assessment": result.result.assessment, "model_version": result.result.model_version}))


def device_details(model: str):
    details = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "cpu_count_logical": os.cpu_count(),
               "python": sys.version, "model_runtime": "NumPy" if model == "lr" else "PyTorch CPU fp32", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress"],
                             capture_output=True, text=True, timeout=30)
        details["cpu"] = json.loads(out.stdout)
    except Exception as exc:   # noqa: BLE001  (device description is best effort)
        details["cpu"] = f"unavailable: {exc}"
    try:
        import ctypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                        ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
        mem = Memory(); mem.length = ctypes.sizeof(Memory)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        details["memory_gb_total"] = round(mem.total / 2**30, 1)
        details["memory_gb_available_at_start"] = round(mem.avail / 2**30, 1)
    except Exception:   # noqa: BLE001
        pass
    import importlib.metadata as metadata
    pkgs = {}
    for name in ("numpy", "scipy", "soundfile", "scikit-learn", "torch", "transformers", "librosa", "echoproof"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    details["packages"] = pkgs
    if model != "lr":
        import torch
        details["torch_num_threads"] = torch.get_num_threads()
        details["cuda_available"] = torch.cuda.is_available()
    return details


def machine_state():
    """CPU clock and load at this instant (Windows CIM); lets a reader see whether background load or a low clock coincided with a group."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Processor | Select-Object -First 1 CurrentClockSpeed,LoadPercentage | ConvertTo-Json -Compress"],
                             capture_output=True, text=True, timeout=30)
        state = json.loads(out.stdout)
    except Exception as exc:   # noqa: BLE001
        state = {"unavailable": str(exc)}
    state["utc"] = datetime.now(timezone.utc).isoformat()
    return state


def bundle_identity(bundle: Path):
    from echoproof.artifacts import file_digest
    files = sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "desktop.ini")
    return {"bundle_path": str(bundle), "bundle_json_sha256": file_digest(bundle / "bundle.json"),
            "files": {p.relative_to(bundle).as_posix(): file_digest(p) for p in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["lr", "wavlm", "bigru"])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--owner", default="")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seconds", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", default="100", help="timed calls per duration: one number, or 'N' for all or e.g. 1:30,4:30,10:30,30:10,60:10")
    parser.add_argument("--fresh", type=int, default=10, help="fresh-process repetitions per duration")
    parser.add_argument("--durations", default=",".join(map(str, DURATIONS)))
    parser.add_argument("--out-dir", default="reports/r3")
    args = parser.parse_args()
    if args.child:
        return child(args)
    if not args.owner:
        raise SystemExit("--owner is required")

    bundle = Path(args.bundle)
    bundle = bundle if bundle.is_absolute() else ROOT / bundle
    durations = [int(x) for x in args.durations.split(",")]
    measured = {d: int(args.measured) for d in durations} if ":" not in args.measured else {int(k): int(v) for k, v in (p.split(":") for p in args.measured.split(","))}
    loaded = load_bundle(args.model, bundle)
    from echoproof.inference import analyze
    identity = bundle_identity(bundle)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"r3_n2_local_{args.model}_latency_{stamp}.json"
    report = {"schema": "r3-n2-local-latency-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "owner": args.owner, "model": args.model,
              "benchmark_kind": "local development machine; batch 1; fresh subprocesses are cold interpreters, not a cold machine or cold disk cache",
              "device": device_details(args.model), "bundle": identity, "model_version": loaded.spec.model_version, "score_type": loaded.score_type,
              "configuration": {"batch_size": 1, "recordings_per_call": 1, "warmup_calls": args.warmup, "measured_calls": measured, "fresh_process_repetitions": args.fresh,
                                "durations_seconds": durations, "clock": "time.perf_counter_ns", "tolerance": TOLERANCE,
                                "fixtures": FIXTURES, "fixture_note": "first N seconds of one in-memory concatenation of these genuine fixtures"},
              "groups": {}, "errors": []}
    for seconds in durations:
        group = {"duration_seconds": seconds, "warmup_raw_ns": [], "raw_samples_ns": [], "stage_samples_ns": {}, "fresh_process_repetitions": [],
                 "fresh_process_errors": [], "errors": []}
        report["groups"][str(seconds)] = group
        try:
            raw = clip_bytes(ROOT, seconds)
            reference = analyze(raw, loaded)
            group["production_reference"] = {"status": reference.result.status, "score": reference.result.score, "assessment": reference.result.assessment,
                                             "chunks": len(reference.windows), "model_version": reference.result.model_version}
            for rep in range(args.fresh):
                try:
                    group["fresh_process_repetitions"].append(fresh_process(args.model, bundle, seconds))
                except Exception as exc:   # noqa: BLE001
                    group["fresh_process_errors"].append({"repetition": rep + 1, "type": type(exc).__name__, "message": str(exc)})
                print(f"{args.model} {seconds}s fresh {rep + 1}/{args.fresh}", flush=True)
            fresh = {}
            for metric in ("process_to_first_prediction_ns", "bundle_load_ns", "first_prediction_ns"):
                vals = [int(x[metric]) for x in group["fresh_process_repetitions"]]
                fresh[metric] = {**stats(vals), "raw_samples_ns": vals}
            group["fresh_process_summary_ns"] = fresh
            group["machine_state_before_warmup"] = machine_state()
            for i in range(args.warmup):
                _, stages, _ = instrumented_analyze(args.model, raw, loaded)
                group["warmup_raw_ns"].append(int(stages["total"]))
                print(f"{args.model} {seconds}s warmup {i + 1}/{args.warmup}", flush=True)
            deltas, flips = [], 0
            for i in range(measured[seconds]):
                got, stages, _ = instrumented_analyze(args.model, raw, loaded)
                deltas.append(abs(float(got.result.score) - float(reference.result.score)))
                flips += got.result.assessment != reference.result.assessment
                group["raw_samples_ns"].append(int(stages["total"]))
                for name, value in stages.items():
                    group["stage_samples_ns"].setdefault(name, []).append(int(value))
                if (i + 1) % 10 == 0 or i + 1 == measured[seconds]:
                    print(f"{args.model} {seconds}s measured {i + 1}/{measured[seconds]}", flush=True)
            group["machine_state_after_measurement"] = machine_state()
            group["parity"] = {"reference_match": bool(max(deltas) <= TOLERANCE), "max_abs_score_delta": max(deltas), "decision_flips": int(flips), "checks": len(deltas)}
            group["summary_ns"] = {name: stats(vals) for name, vals in group["stage_samples_ns"].items()}
            group["status"] = "completed" if len(group["fresh_process_repetitions"]) == args.fresh else "completed_with_fresh_process_errors"
        except Exception as exc:   # noqa: BLE001
            group["status"] = "error"
            group["errors"].append(f"{type(exc).__name__}: {exc}")
            print(f"ERROR {seconds}s: {exc}", flush=True)
        with report_path.open("w", encoding="utf-8") as handle:   # rewritten after each duration so an interruption keeps what finished
            json.dump(report, handle, indent=1, default=str)
    print("Saved report:", report_path)
    print("Status by duration:", {k: v["status"] for k, v in report["groups"].items()})


if __name__ == "__main__":
    main()
