"""Time the R3-U2 AST bundle adapter (and optionally the R3-U1 WavLM bundle adapter) on the machine it runs on. Reads no dataset audio.

Run it on the machine the app will use. Synthetic noise stands in for audio: the cost of the AST depends on the chunk length, not on its content. It reports,
per adapter and per precision: chunk latency at batch 1 and batch 8 (median, 95th percentile, and the share spent in feature extraction), and the time a
whole recording of 10, 30 and 60 seconds would take with the project's chunking (4-second windows, 2-second stride). Warm-up runs are discarded.

  python scripts/r3_u2_latency.py --owner <name> [--device cpu|cuda] [--precision fp32|fp16] [--threads N] [--wavlm-bundle bundles/r3_u1_wavlm_head_B]
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.artifacts import write_json  # noqa: E402

CHUNK = 64000
RATE = 16000


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def timed(fn, repeats, warmup, sync=None):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        if sync:
            sync()
        start = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - start)
    return times


def summarise(times, per_chunk):
    return {"median_s": statistics.median(times), "p95_s": percentile(times, .95), "per_chunk_median_s": statistics.median(times) / per_chunk,
            "runs": len(times)}


def recording_chunks(seconds: int) -> int:
    from echoproof.data.chunks import eval_windows
    return len(eval_windows(seconds * RATE))


def benchmark(name, adapter, *, repeats, warmup, sync, split=None):
    import numpy as np
    rng = np.random.default_rng(0)
    waves = [rng.normal(0, .1, CHUNK).astype(np.float32) for _ in range(8)]
    result = {"adapter": name}
    for batch in (1, 8):
        group = waves[:batch]
        result[f"batch_{batch}"] = summarise(timed(lambda: adapter.score_chunks(group), repeats, warmup, sync), batch)
    if split is not None:
        result["feature_extraction_share_batch_8"] = split(waves)
    per_chunk = result["batch_8"]["per_chunk_median_s"]
    result["estimated_recording_seconds"] = {f"{s}s_audio": {"chunks": recording_chunks(s), "seconds": recording_chunks(s) * per_chunk} for s in (10, 30, 60)}
    return result


def ast_feature_share(adapter, sync):
    import torch
    detector = adapter._detector
    def run(waves):
        valid = torch.tensor([len(w) for w in waves], dtype=torch.int64)
        batch = torch.from_numpy(__import__("numpy").stack(waves)).to(adapter._device)
        start = time.perf_counter()
        detector.extract_features(batch, valid)
        feature = time.perf_counter() - start
        total = statistics.median(timed(lambda: adapter.score_chunks(list(waves)), 3, 1, sync))
        return feature / total
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--owner", required=True)
    parser.add_argument("--bundle", default="bundles/r3_u2_ast_partial_B")
    parser.add_argument("--wavlm-bundle", default=None, help="also time the R3-U1 WavLM bundle (fp32)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's own)")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device available")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    sync = torch.cuda.synchronize if args.device == "cuda" else None
    root = Path(args.root).resolve()
    from echoproof.models.ast_adapter import ASTBundleAdapter
    results = []
    start = time.perf_counter()
    ast = ASTBundleAdapter.load(root / args.bundle, device=args.device, precision=args.precision, batch_size=8)
    load_seconds = time.perf_counter() - start
    entry = benchmark(f"ast_partial ({args.precision}, {args.device})", ast, repeats=args.repeats, warmup=args.warmup, sync=sync)
    entry["load_seconds"] = load_seconds
    entry["feature_extraction_share_batch_8"] = ast_feature_share(ast, sync)([__import__("numpy").random.default_rng(1).normal(0, .1, CHUNK).astype("float32") for _ in range(8)])
    results.append(entry)
    del ast
    if args.wavlm_bundle:
        from echoproof.models.wavlm_adapter import WavLMBundleAdapter
        start = time.perf_counter()
        wavlm = WavLMBundleAdapter.load(root / args.wavlm_bundle, device=args.device, batch_size=8)
        load_seconds = time.perf_counter() - start
        entry = benchmark(f"wavlm_head (fp32, {args.device})", wavlm, repeats=args.repeats, warmup=args.warmup, sync=sync)
        entry["load_seconds"] = load_seconds
        results.append(entry)
    report = {"task": "R3-U2", "kind": "latency", "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "owner": args.owner,
              "device": args.device, "precision": args.precision, "synthetic_input": "gaussian noise, 4-second chunks",
              "machine": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
                          "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None},
              "results": results,
              "note": "Timings on this machine only. Recording estimates multiply the batch-8 per-chunk median by the chunk count and exclude audio decoding."}
    path = root / "reports/r3" / f"r3_u2_latency_{args.device}_{args.precision}_{report['created_utc']}.json"
    write_json(path, report)
    for r in results:
        print(r["adapter"])
        for key in ("batch_1", "batch_8"):
            print(f"  {key}: median {r[key]['median_s']:.3f}s  p95 {r[key]['p95_s']:.3f}s  per chunk {r[key]['per_chunk_median_s']:.3f}s")
        for k, v in r["estimated_recording_seconds"].items():
            print(f"  {k}: {v['chunks']} chunks, about {v['seconds']:.1f}s")
        if "feature_extraction_share_batch_8" in r:
            print(f"  feature-extraction share of a batch of 8: {r['feature_extraction_share_batch_8']:.0%}")
    print("report:", path)


if __name__ == "__main__":
    main()
