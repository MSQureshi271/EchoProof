"""R3-S1 specification version 2: does the MP3 round trip hold up on every locally available selection-role genuine and XTTS recording?

Version 2 replaces `mp3_64` with `mp3_32`. The MP3 round-trip check rejects a decode whose lag-0 correlation with the input is below
`corruption.MP3_MIN_CORRELATION`; a harsher codec setting could, in principle, make correct decodes fall below it and stop R5-S1. This script
measures that on real validation audio, using the same encode and decode arguments as `corruption.mp3_roundtrip`, without applying the check, so the
margin is visible.

Scope: the B selection role only (the R2-S3 plan, which drops every test assignment and rejects Chatterbox), genuine and XTTS recordings found in the
local `dataset/` folder, each verified against the cohort manifest's PCM hash before use. Melo and VITS audio is not available locally. No model scores
anything and no audio is written.

    python scripts/r3_s1_v2_acceptance.py --root . --dataset-dir dataset
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

KBPS = (32, 128)
WINDOW = 16
NAME = "r3_s1_v2_acceptance_local"


def work(job):
    from echoproof.data import corruption as c
    from echoproof.data.cohort import canonical_pcm, pcm_sha256
    path, sha, n, rid, comp = job
    pcm, _ = canonical_pcm(path)
    if pcm_sha256(pcm) != sha or len(pcm) != n:
        return {"id": rid, "error": "hash mismatch"}
    x = pcm.astype(np.float64) / 32768.0
    ff = c.find_ffmpeg()
    out = {"id": rid, "component": comp, "n": int(n)}
    for k in KBPS:
        mp3 = c._run(ff, [a.replace("<bitrate>", str(k)) for a in c.SPEC["mp3"]["encode_args"]], np.clip(np.rint(x * 32768.0), -32768, 32767).astype("<i2").tobytes())
        dec = np.frombuffer(c._run(ff, c.SPEC["mp3"]["decode_args"], mp3), dtype="<i2").astype(np.float64)[c.MP3_DELAY_SAMPLES:] / 32768.0
        diff = int(dec.size - x.size)
        al = dec[:x.size] if diff >= 0 else np.concatenate([dec, np.zeros(-diff)])
        corr = []
        for s in range(-WINDOW, WINDOW + 1):
            a, b = (x[:len(x) - s], al[s:]) if s >= 0 else (x[-s:], al[:len(al) + s])
            corr.append(float(np.dot(a, b) / np.sqrt(max(np.dot(a, a) * np.dot(b, b), 1e-30))))
        corr = np.array(corr)
        gain = float(np.dot(x, al) / max(float(np.dot(x, x)), 1e-30))
        err = float(np.mean((al / gain - x) ** 2))
        out[f"k{k}"] = {"corr0": float(corr[WINDOW]), "peak_offset": int(np.argmax(corr) - WINDOW), "length_diff": diff,
                        "gcsnr": float(10 * np.log10(max(np.mean(x * x), 1e-30) / max(err, 1e-30))), "kbps": len(mp3) * 8 / (x.size / 16000) / 1000.0}
    return out


def summarise(results, k):
    key = f"k{k}"
    c0 = np.array([r[key]["corr0"] for r in results])
    off = np.array([r[key]["peak_offset"] for r in results])
    ld = np.array([r[key]["length_diff"] for r in results])
    g = np.array([r[key]["gcsnr"] for r in results])
    kb = np.array([r[key]["kbps"] for r in results])
    from echoproof.data import corruption as c
    def pct(v, q): return round(float(np.percentile(v, q)), 4)
    return {"kbps": k, "recordings": len(results),
            "lag0_correlation": {"min": round(float(c0.min()), 4), "p1": pct(c0, 1), "p5": pct(c0, 5), "median": pct(c0, 50)},
            "below_floor": int((c0 < c.MP3_MIN_CORRELATION).sum()), "floor": c.MP3_MIN_CORRELATION,
            "peak_offsets": {str(o): n for o, n in sorted(Counter(off.tolist()).items())},
            "length_diff_samples": {"min": int(ld.min()), "max": int(ld.max()), "limit": c.MP3_MAX_LENGTH_DIFF,
                                    "over_limit": int((np.abs(ld) > c.MP3_MAX_LENGTH_DIFF).sum())},
            "snr_after_removing_level_change_db": {"min": round(float(g.min()), 2), "p5": pct(g, 5), "median": pct(g, 50), "max": round(float(g.max()), 2)},
            "effective_kbps": {"min": round(float(kb.min()), 2), "max": round(float(kb.max()), 2)},
            "by_component": {comp: {"corr_min": round(float(c0[m].min()), 4), "snr_median_db": round(float(np.median(g[m])), 2), "snr_min_db": round(float(g[m].min()), 2)}
                             for comp in sorted({r["component"] for r in results})
                             for m in [np.array([r["component"] == comp for r in results])]}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    from echoproof.data import corruption as c, corruption_sample as cs, mp3_probe_run as pr
    from echoproof.features.r2_s1 import R2S1Error, file_sha256
    t0 = time.time()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "reports" / "r3"
    if (out_dir / f"{NAME}.json").exists() or (out_dir / f"{NAME}_rows.csv").exists():
        raise R2S1Error(f"{NAME} outputs exist in {out_dir}: not overwritten")
    loaded = cs.load_inputs(root, 42)
    items = [i for i in loaded["eda"].items if i["component"] in ("genuine", "xtts")]
    jobs = []
    for i in items:
        path = pr.local_path(Path(args.dataset_dir), i)
        if path is not None:
            jobs.append((str(path), i["pcm_sha256"], i["n_samples"], i["recording_id"], i["component"]))
    print(f"{len(items)} selection genuine/xtts recordings, {len(jobs)} found locally", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, r in enumerate(pool.map(work, jobs, chunksize=4), 1):
            if "error" in r:
                raise R2S1Error(f"{r['id']}: {r['error']}")
            results.append(r)
            if n % 100 == 0:
                print(n, f"{time.time() - t0:.0f}s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{NAME}_rows.csv", "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["recording_id", "component", "n_samples"] + [f"{m}_{k}" for k in KBPS for m in ("corr0", "peak_offset", "length_diff", "snr_after_level_db", "kbps")])
        for r in sorted(results, key=lambda r: r["id"]):
            w.writerow([r["id"], r["component"], r["n"]] + [r[f"k{k}"][m] for k in KBPS for m in ("corr0", "peak_offset", "length_diff", "gcsnr", "kbps")])
    report = {"schema_version": 1, "task": "R3-S1", "step": "v2_acceptance", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "scope": "B validation, selection role only; genuine and XTTS recordings found locally, each verified against the cohort manifest hash; "
                       "no calibration, operating, test or Chatterbox audio is read; no model scores anything; no audio is written",
              "specification": {"version": c.SPEC_VERSION, "sha256": c.SPEC_SHA256, "ffmpeg": c.ffmpeg_info()},
              "selection_recordings_in_plan": len(items), "measured": len(results), "unavailable_locally": len(items) - len(jobs),
              "summary": {f"mp3_{k}": summarise(results, k) for k in KBPS},
              "inputs": {"plan_sha256": loaded["eda"].plan.sha256, "split_sha256": loaded["split_hash"],
                         "cohort_manifest_file_sha256": file_sha256(loaded["paths"]["cohort"]),
                         "corruption_module_sha256": hashlib.sha256(Path(c.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
              "outputs": {f"{NAME}_rows.csv": {"sha256": file_sha256(out_dir / f"{NAME}_rows.csv"), "rows": len(results)}},
              "seconds_total": round(time.time() - t0, 1)}
    (out_dir / f"{NAME}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("measured", "unavailable_locally", "summary")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
