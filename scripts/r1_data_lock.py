"""R1 data-lock commands (Subhan): admission, split, materialize, repair, leakage.

  python scripts/r1_data_lock.py admission --pairs-dir <phase2_pairs/run> --inspection-dir <phase2_inspection/run>
  python scripts/r1_data_lock.py split [--inspection-dir <phase2_inspection/run>]
  python scripts/r1_data_lock.py materialize --project-root <folder containing dataset/>   # needs audio; run on Colab
  python scripts/r1_data_lock.py repair --project-root <folder containing dataset/>        # retry only failed recordings
  python scripts/r1_data_lock.py leakage

Every command refuses to overwrite an existing output so a locked artifact cannot change silently.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from echoproof.artifacts import digest  # noqa: E402
from echoproof.contracts import Chunk, SplitAssignment, read_parquet, write_parquet  # noqa: E402
from echoproof.data import chunks as chunk_mod, cohort, leakage, splits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}; delete it deliberately or use a new version")
    path.write_text(text, encoding="utf-8")


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def cmd_admission(a):
    report = cohort.admission(_jsonl(Path(a.pairs_dir) / "coverage.jsonl"), _jsonl(Path(a.inspection_dir) / "headers.jsonl"))
    stems = report.pop("admitted_stems")
    report["inputs"] = {"pairs_dir": str(a.pairs_dir), "inspection_dir": str(a.inspection_dir)}
    report["admitted_stems_sha256"] = digest(stems)
    out = ROOT / a.reports
    _write_new(out / "admitted_stems.txt", "\n".join(stems) + "\n")
    _write_new(out / "admission.json", _json(report))
    keys = ("stems_in_coverage", "stems_admitted", "speakers", "chapters", "format_consistent", "grouping_unit", "stems_excluded")
    print(json.dumps({k: report[k] for k in keys}, indent=2))


def _storage(plan, headers, flac_ratio=0.55):
    wanted = set(plan)
    seconds, files = 0.0, 0
    for h in headers:
        if h.get("state") == "ok" and cohort.stem_from_path(h["path"]) in wanted:
            seconds += float(h["duration_seconds"])
            files += 1
    pcm_gb = seconds * cohort.SAMPLE_RATE * 2 / 1e9
    return {"basis": "Phase 2 header durations; FLAC ratio (0.55 of 16-bit PCM) is an assumption verified at materialisation",
            "files": files, "hours": round(seconds / 3600, 2), "pcm16_gb": round(pcm_gb, 2), "estimated_flac_gb": round(pcm_gb * flac_ratio, 2)}


def cmd_split(a):
    out = ROOT / a.reports
    stems = (out / "admitted_stems.txt").read_text().split()
    speakers = sorted({cohort.parse_stem(s)[0] for s in stems})
    sp = splits.assign_speakers(speakers, seed=a.seed)
    per = {"train": a.train_per_speaker, "validation": a.eval_per_speaker, "test": a.eval_per_speaker}
    plan = splits.sample_stems(stems, sp, per_speaker=per, seed=a.seed)
    data = ROOT / "data" / "splits"
    data.mkdir(parents=True, exist_ok=True)
    assignments_json = {s: list(v) for s, v in sorted(sp.items())}
    report = {"schema_version": 1, "task": "R1-S2", "seed": a.seed, "split_version": splits.SPLIT_VERSION,
              "admitted_stems_sha256": digest(stems), "per_speaker": per, "speakers": len(speakers),
              "speaker_counts": splits.speaker_counts(sp), "planned_stems": len(plan), "cohort_plan_sha256": digest(plan),
              "speaker_assignments_sha256": digest(assignments_json), "protocols": {}}
    for protocol in ("A", "B"):
        assignments, h = splits.build_assignments(protocol, plan, sp)
        target = data / f"protocol_{protocol}_v1.parquet"
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        write_parquet(assignments, str(target))
        rows = splits.protocol_rows(protocol, plan, sp)
        report["protocols"][protocol] = {"split_sha256": h, "recordings": len(rows), "support": splits.panel_support(rows)}
    _write_new(data / "speaker_assignments_v1.json", _json(assignments_json))
    _write_new(data / "cohort_plan_v1.txt", "\n".join(plan) + "\n")
    if a.inspection_dir:
        report["storage_estimate"] = _storage(plan, _jsonl(Path(a.inspection_dir) / "headers.jsonl"))
    _write_new(out / "split_report.json", _json(report))
    print(json.dumps({k: report[k] for k in ("speakers", "speaker_counts", "planned_stems")}, indent=2))
    for p, v in report["protocols"].items():
        print(p, v["split_sha256"][:16], v["recordings"], "recordings")
    print(json.dumps(report.get("storage_estimate"), indent=2))


def cmd_materialize(a):
    plan = (ROOT / "data" / "splits" / "cohort_plan_v1.txt").read_text().split()
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "data" / "processed" / "cohort_v1"
    rows, index = cohort.materialize(plan, a.project_root, out_dir, progress=lambda n: print(f"{n} recordings", flush=True))
    manifests = ROOT / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    cohort.write_table(rows, manifests / "cohort_v1.parquet")
    cohort.write_table([{"recording_id": k, "shard": v[0], "bytes": v[1]} for k, v in sorted(index.items())], out_dir / "index.parquet")
    by_id = {r["recording_id"]: r for r in rows}
    chunks = []
    for protocol in ("A", "B"):
        assignments = read_parquet(str(ROOT / "data" / "splits" / f"protocol_{protocol}_v1.parquet"), SplitAssignment)
        chunks.extend(chunk_mod.build_chunks(protocol, assignments, by_id))
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    write_parquet(chunks, str(processed / "chunks_v1.parquet"))
    excluded = {}
    for r in rows:
        if not r["retained"]:
            excluded.setdefault(r["exclusion_reason"], []).append(r["recording_id"])
    report = {"schema_version": 1, "task": "R1-S3", "recordings": len(rows), "retained": sum(r["retained"] for r in rows),
              "exclusions": {k: len(v) for k, v in excluded.items()}, "excluded_ids": excluded, "chunks": len(chunks),
              "shard_gb": round(sum(v[1] for v in index.values()) / 1e9, 3), "preprocessing_hash": chunk_mod.PREPROCESSING_HASH,
              "cohort_manifest_sha256": digest([[r["recording_id"], r["pcm_sha256"], r["retained"]] for r in rows])}
    _write_new(ROOT / a.reports / "materialize.json", _json(report))
    print(json.dumps({k: v for k, v in report.items() if k != "excluded_ids"}, indent=2))


def _manifest_hash(rows):
    return digest([[r["recording_id"], r["pcm_sha256"], r["retained"]] for r in rows])


def _supersede(path, tag):
    """Move a locked artifact to a 'superseded/' folder beside it (kept as evidence, never deleted)."""
    path = Path(path)
    if not path.exists():
        return None
    target = path.parent / "superseded" / f"{path.stem}.{tag}{path.suffix}"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    path.replace(target)
    return str(target)


def cmd_repair(a):
    manifests, processed = ROOT / "data" / "manifests", ROOT / "data" / "processed"
    cohort_path, chunks_path = manifests / "cohort_v1.parquet", processed / "chunks_v1.parquet"
    out_dir = Path(a.out_dir) if a.out_dir else processed / "cohort_v1"
    old_rows = cohort.read_table(cohort_path)
    new_rows, result = cohort.repair(old_rows, a.project_root, out_dir)
    if not result["attempted"]:
        print("nothing to repair: no recording failed to read/decode")
        return 0
    print(json.dumps({k: result[k] for k in ("attempted", "repaired", "still_failing", "duplicate_dispositions_changed")}, indent=2))
    if not result["repaired"]:
        print("no recording could be repaired; no manifest was changed")
        return 1
    # Everything is computed in memory before any locked file is moved.
    by_id = {r["recording_id"]: r for r in new_rows}
    old_chunks = read_parquet(str(chunks_path), Chunk)
    kept = [c for c in old_chunks if by_id[c.recording_id]["retained"]]
    fresh = []
    for protocol in ("A", "B"):
        assignments = read_parquet(str(ROOT / "data" / "splits" / f"protocol_{protocol}_v1.parquet"), SplitAssignment)
        wanted = {r for r in result["repaired"] if by_id[r]["retained"]}
        fresh.extend(chunk_mod.build_chunks(protocol, [x for x in assignments if x.recording_id in wanted], by_id))
    tag = "pre_repair-" + __import__("time").strftime("%Y%m%dT%H%M%SZ", __import__("time").gmtime())
    report = {"schema_version": 1, "task": "R1-S3-repair", **result, "chunks_before": len(old_chunks), "chunks_after": len(kept) + len(fresh),
              "retained_before": sum(r["retained"] for r in old_rows), "retained_after": sum(r["retained"] for r in new_rows),
              "cohort_manifest_sha256_before": _manifest_hash(old_rows), "cohort_manifest_sha256_after": _manifest_hash(new_rows),
              "superseded": {}}
    report["superseded"]["cohort"] = _supersede(cohort_path, tag)
    report["superseded"]["chunks"] = _supersede(chunks_path, tag)
    report["superseded"]["leakage"] = _supersede(ROOT / a.reports / "leakage.json", tag)  # no longer describes the manifests
    cohort.write_table(new_rows, cohort_path)
    write_parquet(kept + fresh, str(chunks_path))
    _write_new(ROOT / a.reports / f"repair_{tag.split('-')[1]}.json", _json(report))
    print(json.dumps({k: report[k] for k in ("retained_before", "retained_after", "chunks_before", "chunks_after")}, indent=2))
    print("now rerun:  python scripts/r1_data_lock.py leakage")
    return 0


def cmd_leakage(a):
    rows = cohort.read_table(ROOT / "data" / "manifests" / "cohort_v1.parquet")
    by_protocol = {p: splits.load_assignments(ROOT / "data" / "splits" / f"protocol_{p}_v1.parquet", sealed=True) for p in ("A", "B")}
    chunks = read_parquet(str(ROOT / "data" / "processed" / "chunks_v1.parquet"), Chunk)
    per_protocol = {p: [c for c in chunks if c.protocol_id == f"protocol:{p}"] for p in ("A", "B")}
    report = leakage.check_leakage(rows, by_protocol, per_protocol)
    report["note"] = "integrity check over manifests only; no model scores are read"
    _write_new(ROOT / a.reports / "leakage.json", _json(report))
    print(json.dumps({k: report[k] for k in ("passed", "violation_count", "summary")}, indent=2))
    return 0 if report["passed"] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", default="reports/r1")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("admission")
    s.add_argument("--pairs-dir", required=True)
    s.add_argument("--inspection-dir", required=True)
    s = sub.add_parser("split")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--train-per-speaker", type=int, default=10)
    s.add_argument("--eval-per-speaker", type=int, default=20)
    s.add_argument("--inspection-dir")
    s = sub.add_parser("materialize")
    s.add_argument("--project-root", required=True)
    s.add_argument("--out-dir")
    s = sub.add_parser("repair")
    s.add_argument("--project-root", required=True)
    s.add_argument("--out-dir")
    sub.add_parser("leakage")
    a = p.parse_args(argv)
    return {"admission": cmd_admission, "split": cmd_split, "materialize": cmd_materialize, "repair": cmd_repair, "leakage": cmd_leakage}[a.command](a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
