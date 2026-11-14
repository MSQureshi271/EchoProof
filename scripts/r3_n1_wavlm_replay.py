"""Replay development recordings through the calibrated WavLM bundle and compare with the saved scores (run on Colab with transformers 5.22.0).

Tier 1: canonical 4 s windows -> the bundle's scorer -> mean -> stored mapping -> stored decision rule.
Tier 2: the same samples re-encoded as a 16 kHz mono PCM16 WAV -> the shared `analyze` path (decode, quality checks, chunking, scoring, mapping, decision).
Expected values: the saved raw recording score, that score mapped by the bundle's own function, and the decision it implies. Operating-role expectations
are also checked against the saved deployed scores of the run. Reads Protocol B validation audio from the cohort shards (hash-verified); nothing else.
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

from echoproof.artifacts import digest, file_digest, write_json  # noqa: E402
from echoproof.evaluation import r3_n1_wavlm as W  # noqa: E402
from echoproof.evaluation.r3_n1_wavlm_replay import compare_replay  # noqa: E402

RECIPE = "configs/r3_n1_wavlm_calibration_b.json"


def pick(rows, n, seed=42):
    ordered = sorted(rows, key=lambda r: digest([seed, r["recording_id"]]))
    return ordered if n is None else ordered[:n]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--operating", type=int, default=None, help="operating recordings to replay (default: all)")
    parser.add_argument("--others", type=int, default=100, help="recordings per selection and calibration role")
    args = parser.parse_args(argv)

    import importlib.metadata as metadata
    import soundfile as sf
    import torch
    import pyarrow.parquet as pq
    from echoproof import inference as inf
    from echoproof.data import cohort
    from echoproof.data.audio import MAX_AUDIO_SECONDS
    from echoproof.data.chunks import eval_windows
    from echoproof.models.r3_n1_wavlm_bundle import load_wavlm_bundle
    from echoproof.training.datasets import ShardWaveformLoader
    from echoproof.training.r3_m1_data import VerifiedWaveformLoader

    root = Path(args.root).resolve()
    recipe = W.load_recipe(root / RECIPE)
    version = metadata.version("transformers")
    if version != recipe["transformers_pin"]:
        raise SystemExit(f"transformers {version} is installed; the replay requires the pinned {recipe['transformers_pin']}")
    run_dir = Path(args.run_dir)
    run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    out = run_dir / "handoff"
    out.mkdir(exist_ok=True)
    if (out / "replay.json").exists():
        raise SystemExit("a replay report already exists for this run; use a new run or keep the existing one")
    bundle_dir = root / recipe["bundle_destination"]
    bundle = load_wavlm_bundle(bundle_dir, device=args.device)
    rule = bundle.spec.decision_rule
    inputs = root / recipe["inputs"]["directory"]

    operating = {r["recording_id"]: r for r in pq.read_table(run_dir / "operating/operating_predictions.parquet").to_pylist()}
    selected = []
    for role, limit in (("operating", args.operating), ("selection", args.others), ("calibration", args.others)):
        rows = W.read_recordings(inputs, role)
        selected += [(role, r) for r in pick(rows, limit)]
    cohort_rows = {r["recording_id"]: r for r in cohort.read_table(str(root / "data/manifests/cohort_v1.parquet"))}
    wanted = [r["recording_id"] for _, r in selected]
    base = ShardWaveformLoader(args.shard_dir, {i: cohort_rows[i]["canonical_path"] for i in wanted})
    loader = VerifiedWaveformLoader(base, [cohort_rows[i] for i in wanted])

    records, saved_vs_run = [], []
    try:
        for n, (role, r) in enumerate(selected, 1):
            wave = loader(r["recording_id"])                                  # hash-verified canonical samples
            raw_saved = r["raw_score"]
            expected_score = bundle.calibrate(raw_saved) if bundle.calibrate else raw_saved
            rec = {"recording_id": r["recording_id"], "role": role, "expected_raw": raw_saved, "expected_score": expected_score,
                   "expected_decision": rule.decide(expected_score)}
            if role == "operating":
                saved_vs_run.append(abs(operating[r["recording_id"]]["deployed_score"] - expected_score))
            chunks = [wave[a:b] for a, b in eval_windows(len(wave))]
            raw1 = inf.mean_probability([float(s) for s in bundle.scorer(chunks)])
            score1 = bundle.calibrate(raw1) if bundle.calibrate else raw1
            rec.update(tier1_raw=raw1, tier1_score=score1, tier1_decision=rule.decide(score1))
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

    report = compare_replay(records, rule)
    report.update({"task": "R3-N1-WavLM replay", "owner": args.owner, "device": args.device, "bundle": str(bundle_dir),
                   "bundle_json_sha256": file_digest(bundle_dir / "bundle.json"), "run_dir": run_dir.name, "recipe_sha256": W.recipe_hash(recipe),
                   "saved_operating_scores_equal_the_mapped_raw_scores": {"max_abs_difference": float(max(saved_vs_run)) if saved_vs_run else None},
                   "roles": {role: sum(1 for x, _ in selected if x == role) for role in ("operating", "selection", "calibration")},
                   "environment": {"transformers": version, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                                   "transformers_pin": recipe["transformers_pin"]},
                   "tier2_note": "canonical PCM re-encoded as 16 kHz mono PCM16 WAV; this validates decode, chunking, scoring, mapping and decision, not resampling of original files"})
    report["records"] = records
    write_json(out / "replay.json", report)
    print("replay", "PASSED" if report["passed"] else "FAILED", out / "replay.json")
    print(json.dumps({k: report[k] for k in ("recordings", "tiers", "tier1_vs_tier2") if k in report}, indent=1))
    print("decision flips:", len(report["flips"]), "| near-threshold recordings:", len(report["near_threshold"]))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
