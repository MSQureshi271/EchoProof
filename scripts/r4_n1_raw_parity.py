"""Waveform-to-decision parity for the LR-A and BiGRU-A development bundles (run on Colab, where the canonical shards are).

For each selected recording the canonical audio is read from the hash-verified cohort shards and scored two ways:
  tier 1: canonical 4 s windows -> the bundle's scorer -> mean chunk probability -> the serialized threshold;
  tier 2: the same samples re-encoded as a 16 kHz mono PCM16 WAV -> the shared `analyze` path (decode, quality checks, chunking, scoring, decision).
Expected values are the saved raw recording scores of the R4-M1 export (hash-checked against the recipe) and the decision the serialized rule implies.
Replays all operating recordings and a hash-ordered sample of the selection and calibration roles. Reads Protocol A validation audio only: no test recording.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from echoproof.artifacts import digest, file_digest  # noqa: E402
from echoproof.evaluation.r3_n1_wavlm_replay import compare_replay  # noqa: E402
from echoproof.models import r4_n1_raw_bundles as rb  # noqa: E402

TOLERANCES = {"lr": 1e-6, "bigru": 1e-5}          # the exports' own reload tolerances were 1e-12 (LR) and 1e-6 (BiGRU, CUDA); no relaxation to fit a result
ROLES = ("operating", "selection", "calibration")


def pick(rows, n, seed=42):
    ordered = sorted(rows, key=lambda r: digest([seed, r["recording_id"]]))
    return ordered if n is None else ordered[:n]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(rb.MODELS))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--operating", type=int, default=None, help="operating recordings to replay (default: all)")
    parser.add_argument("--others", type=int, default=100, help="recordings per selection and calibration role")
    parser.add_argument("--tolerance", type=float, default=None)
    args = parser.parse_args(argv)

    import soundfile as sf
    import pyarrow.parquet as pq
    from echoproof import inference as inf
    from echoproof.data import cohort
    from echoproof.data.audio import MAX_AUDIO_SECONDS
    from echoproof.data.chunks import eval_windows
    from echoproof.evaluation.r4_n1 import load_recipe
    from echoproof.training.datasets import ShardWaveformLoader
    from echoproof.training.r3_m1_data import VerifiedWaveformLoader

    root, model = Path(args.root).resolve(), args.model
    tolerance = TOLERANCES[model] if args.tolerance is None else args.tolerance
    meta = rb.MODELS[model]
    facts = rb.verified_run(root, model)
    recipe = load_recipe(root / rb.RECIPE)
    entry = recipe["models"][model]
    bundle_dir = root / meta["destination"]
    run_dir = root / meta["run"]
    out = run_dir / "handoff"
    out.mkdir(exist_ok=True)
    if (out / "waveform_parity.json").exists():
        raise SystemExit("a parity report already exists for this bundle; keep it or move it aside deliberately")
    packaging = json.loads((out / "bundle_packaging.json").read_text(encoding="utf-8"))
    if packaging["bundle_json_sha256"] != file_digest(bundle_dir / "bundle.json") or packaging["saved_score_parity"]["status"] != "passed":
        raise SystemExit("the bundle on this machine is not the one packaged (hash differs) or its saved-score parity did not pass")
    bundle = rb.load_bundle_a(bundle_dir, model, device=args.device)
    rule = bundle.spec.decision_rule

    export = root / entry["export_directory"]
    selected = []
    for role, limit in (("operating", args.operating), ("selection", args.others), ("calibration", args.others)):
        path = export / f"predictions_recordings_{role}.parquet"
        if file_digest(path) != entry["prediction_sha256"][f"predictions_recordings_{role}.parquet"]:
            raise SystemExit(f"{path.name} differs from the recipe")
        rows = pq.read_table(path).to_pylist()
        if any(r["role"] != role for r in rows):
            raise SystemExit(f"{path.name} holds rows of another role")
        selected += [(role, r) for r in pick(rows, limit)]
    cohort_rows = {r["recording_id"]: r for r in cohort.read_table(str(root / "data/manifests/cohort_v1.parquet"))}
    wanted = [r["recording_id"] for _, r in selected]
    base = ShardWaveformLoader(args.shard_dir, {i: cohort_rows[i]["canonical_path"] for i in wanted})
    loader = VerifiedWaveformLoader(base, [cohort_rows[i] for i in wanted])

    records, chunk_count_mismatch = [], []
    try:
        for n, (role, r) in enumerate(selected, 1):
            wave = loader(r["recording_id"])                                  # hash-verified canonical samples
            expected = r["raw_score"]
            rec = {"recording_id": r["recording_id"], "role": role, "expected_raw": expected, "expected_score": expected,
                   "expected_decision": rule.decide(expected)}
            windows = [wave[a:b] for a, b in eval_windows(len(wave))]
            if len(windows) != r["chunk_count"]:
                chunk_count_mismatch.append(r["recording_id"])
            raw1 = inf.mean_probability([float(s) for s in bundle.scorer(windows)])
            rec.update(tier1_raw=raw1, tier1_score=raw1, tier1_decision=rule.decide(raw1))
            if len(wave) / 16000 <= MAX_AUDIO_SECONDS:
                pcm = np.rint(wave.astype(np.float64) * 32768).astype("<i2")
                buffer = io.BytesIO()
                sf.write(buffer, pcm, 16000, format="WAV", subtype="PCM_16")
                analysis = inf.analyze(buffer.getvalue(), bundle)
                if analysis.result.status == "success":
                    rec.update(tier2_raw=analysis.raw_score, tier2_score=analysis.result.score, tier2_decision=int(analysis.result.assessment == "synthetic"))
            records.append(rec)
            if n % 100 == 0:
                print(f"replayed {n}/{len(selected)}", flush=True)
    finally:
        loader.close()

    report = compare_replay(records, rule, tolerance=tolerance)
    report["passed"] = bool(report["passed"] and not chunk_count_mismatch)
    import importlib.metadata as metadata
    import torch
    report.update({"task": f"R4-N1-{model.upper()} waveform parity", "owner": args.owner, "device": args.device, "bundle": str(bundle_dir),
                   "bundle_json_sha256": file_digest(bundle_dir / "bundle.json"), "run": run_dir.name, "recipe_sha256": facts["hashes"]["recipe_sha256"],
                   "chunk_count_mismatches": chunk_count_mismatch, "score_type": bundle.score_type,
                   "roles": {role: sum(1 for x, _ in selected if x == role) for role in ROLES},
                   "environment": {"torch": torch.__version__, "numpy": np.__version__, "librosa": metadata.version("librosa"),
                                   "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None},
                   "tier2_note": "canonical PCM re-encoded as 16 kHz mono PCM16 WAV; this validates decode, chunking, scoring and decision, not resampling of original files",
                   "test_partition_read": False})
    report["records"] = records
    path = out / "waveform_parity.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("parity", "PASSED" if report["passed"] else "FAILED", path)
    print(json.dumps({k: report[k] for k in ("recordings", "tolerance", "tiers", "tier1_vs_tier2") if k in report}, indent=1))
    print("decision flips:", len(report["flips"]), "| near-threshold recordings:", len(report["near_threshold"]), "| chunk-count mismatches:", len(chunk_count_mismatch))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
