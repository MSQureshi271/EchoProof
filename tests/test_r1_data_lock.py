import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from echoproof.contracts import Chunk, SplitAssignment, read_parquet
from echoproof.data import chunks as chunk_mod, cohort, leakage, splits
from echoproof.data.cohort import CohortError

RATE = 16000


def make_stems(speakers=30, per=30):
    return [f"{1000 + s}-{200 + s}-{u:04d}" for s in range(speakers) for u in range(per)]


def cohort_rows(plan, *, hashes=True, n_samples=RATE * 5):
    rows = []
    for stem in plan:
        speaker, chapter, _ = cohort.parse_stem(stem)
        for component in cohort.COMPONENTS:
            rid = cohort.recording_id(component, stem)
            rows.append({"recording_id": rid, "stem": stem, "speaker_id": speaker, "chapter_id": chapter, "component": component,
                         "label": cohort.LABEL[component], "generator": cohort.GENERATOR[component], "n_samples": n_samples,
                         "pcm_sha256": cohort.hashlib.sha256(rid.encode()).hexdigest() if hashes else None,
                         "retained": True, "exclusion_reason": None})
    return rows


def build(speakers=30):
    stems = make_stems(speakers)
    sp = splits.assign_speakers({cohort.parse_stem(s)[0] for s in stems}, seed=42)
    plan = splits.sample_stems(stems, sp, per_speaker={"train": 4, "validation": 6, "test": 6}, seed=42)
    return stems, sp, plan


class StemAndAdmission(unittest.TestCase):
    def test_parse_stem(self):
        self.assertEqual(cohort.parse_stem("103-1240-0000"), ("103", "1240", "0000"))
        for bad in ("103-1240", "a-b-c", "103-1240-000", "103-1240-0000.wav", None):
            with self.assertRaises(CohortError):
                cohort.parse_stem(bad)

    def coverage(self, stems):
        counts = {c: 1 for c in cohort.COMPONENTS}
        counts["transcripts"] = 1
        return [{"candidate_stem": s, "counts": dict(counts)} for s in stems]

    def test_header_failure_excludes_stem_across_components(self):
        stems = make_stems(3, 3)
        headers = [{"component": "vits", "path": f"dataset/tts/VITS/{stems[0]}.wav", "state": "header_error", "failure": "FileNotFoundError"},
                   {"component": "melo", "path": f"dataset/tts/MeloTTS/{stems[1]}.wav", "state": "ok"}]
        report = cohort.admission(self.coverage(stems), headers)
        self.assertEqual(list(report["stems_excluded"]), [stems[0]])
        self.assertNotIn(stems[0], report["admitted_stems"])
        self.assertEqual(report["stems_admitted"], 8)
        self.assertEqual(report["speakers"], 3)
        self.assertEqual(report["grouping_unit"], "speaker")

    def test_format_error_missing_component_and_shared_chapter(self):
        stems = ["1-2-0000", "1-2-0001", "9-2-0000", "bad"]
        rows = self.coverage(stems)
        del rows[1]["counts"]["xtts"]
        report = cohort.admission(rows, [])
        self.assertFalse(report["format_consistent"])
        self.assertEqual(report["chapters_shared_by_multiple_speakers"], ["2"])
        self.assertEqual(report["stems_excluded"]["1-2-0001"]["reason"], "component_missing")
        self.assertEqual(report["grouping_unit"], "stem")

    def test_duplicate_stem_rejected(self):
        with self.assertRaises(CohortError):
            cohort.admission(self.coverage(["1-2-0000", "1-2-0000"]), [])


class SpeakerSplits(unittest.TestCase):
    def test_deterministic_disjoint_and_proportioned(self):
        speakers = [str(i) for i in range(251)]
        a = splits.assign_speakers(speakers, seed=42)
        self.assertEqual(a, splits.assign_speakers(reversed(speakers), seed=42))
        self.assertNotEqual(a, splits.assign_speakers(speakers, seed=7))
        counts = splits.speaker_counts(a)
        self.assertEqual(sum(counts.values()), 251)
        self.assertEqual({k: v for k, v in counts.items() if k.startswith("train")}, {"train/-": 151})
        self.assertEqual(counts["test/-"], 50)
        self.assertEqual(sum(v for k, v in counts.items() if k.startswith("validation")), 50)
        self.assertEqual([counts[f"validation/{r}"] for r in splits.ROLES], [25, 12, 13])

    def test_sampling_same_stems_for_every_component(self):
        stems, sp, plan = build()
        per = {}
        for stem in plan:
            speaker = cohort.parse_stem(stem)[0]
            per[speaker] = per.get(speaker, 0) + 1
        for speaker, n in per.items():
            self.assertEqual(n, {"train": 4, "validation": 6, "test": 6}[sp[speaker][0]])
        self.assertEqual(plan, sorted(plan))
        with self.assertRaises(CohortError):
            splits.sample_stems(stems, sp, per_speaker={"train": 99, "validation": 99, "test": 99})

    def test_protocol_filters(self):
        _, sp, plan = build()
        b = splits.protocol_rows("B", plan, sp)
        comps = {}
        for r in b:
            comps.setdefault(r["partition"], set()).add(r["component"])
        self.assertEqual(comps["train"], {"genuine", "melo", "vits"})
        self.assertEqual(comps["validation"], {"genuine", "melo", "vits", "xtts"})
        self.assertEqual(comps["test"], {"genuine", "melo", "vits", "chatterbox"})
        a = splits.protocol_rows("A", plan, sp)
        self.assertTrue(all(c == set(cohort.COMPONENTS) for c in [{r["component"] for r in a if r["partition"] == p} for p in splits.PARTITIONS]))

    def test_assignments_validate_and_hash_stable(self):
        _, sp, plan = build()
        one, h1 = splits.build_assignments("B", plan, sp)
        two, h2 = splits.build_assignments("B", plan, sp)
        self.assertEqual(h1, h2)
        self.assertEqual(len({a.recording_id for a in one}), len(one))
        self.assertTrue(all(a.validation_role in splits.ROLES for a in one if a.partition == "validation"))

    def test_test_rows_hidden_unless_sealed(self):
        _, sp, plan = build()
        assignments, _ = splits.build_assignments("B", plan, sp)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "b.parquet"
            splits.write_assignments(assignments, path)
            self.assertNotIn("test", {a.partition for a in splits.load_assignments(path)})
            self.assertIn("test", {a.partition for a in splits.load_assignments(path, sealed=True)})


class Chunks(unittest.TestCase):
    def test_windows(self):
        w, s = 4 * RATE, 2 * RATE
        self.assertEqual(chunk_mod.eval_windows(RATE // 2), [(0, RATE // 2)])
        self.assertEqual(chunk_mod.eval_windows(w), [(0, w)])
        self.assertEqual(chunk_mod.eval_windows(8 * RATE), [(0, w), (s, 6 * RATE), (4 * RATE, 8 * RATE)])
        self.assertEqual(chunk_mod.eval_windows(9 * RATE)[-1], (5 * RATE, 9 * RATE))
        for n in (1, 63999, 64001, 100000, 33 * RATE + 17):
            windows = chunk_mod.eval_windows(n)
            starts = [a for a, _ in windows]
            self.assertEqual(len(starts), len(set(starts)))
            self.assertEqual(windows[0][0], 0)
            self.assertEqual(windows[-1][1], n)
            self.assertTrue(all(b - a <= w for a, b in windows))
            covered = np.zeros(n, bool)
            for a, b in windows:
                covered[a:b] = True
            self.assertTrue(covered.all())
        with self.assertRaises(CohortError):
            chunk_mod.eval_windows(0)

    def test_build_chunks_skips_excluded_and_validates(self):
        _, sp, plan = build()
        assignments, _ = splits.build_assignments("B", plan, sp)
        rows = cohort_rows(plan)
        rows[0]["retained"], rows[0]["exclusion_reason"] = False, "exact_duplicate"
        built = chunk_mod.build_chunks("B", assignments, {r["recording_id"]: r for r in rows})
        self.assertNotIn(rows[0]["recording_id"], {c.recording_id for c in built})
        self.assertEqual(len({c.chunk_id for c in built}), len(built))


class Leakage(unittest.TestCase):
    def setUp(self):
        _, sp, plan = build()
        self.plan, self.sp = plan, sp
        self.rows = cohort_rows(plan)
        self.assign = {p: splits.build_assignments(p, plan, sp)[0] for p in ("A", "B")}
        by_id = {r["recording_id"]: r for r in self.rows}
        self.chunks = {p: chunk_mod.build_chunks(p, self.assign[p], by_id) for p in ("A", "B")}

    def kinds(self, report):
        return {v["check"] for v in report["violations"]}

    def test_clean_cohort_passes(self):
        report = leakage.check_leakage(self.rows, self.assign, self.chunks)
        self.assertTrue(report["passed"], report["violations"][:3])

    def test_detects_speaker_crossing(self):
        import dataclasses
        victim = next(a for a in self.assign["B"] if a.partition == "train")
        moved = [dataclasses.replace(a, partition="test") if a is victim else a for a in self.assign["B"]]
        report = leakage.check_leakage(self.rows, {"B": moved})
        self.assertTrue({"speaker_crosses_partitions", "stem_crosses_partitions"} & self.kinds(report))

    def test_detects_chatterbox_in_b_training(self):
        import dataclasses
        a = next(x for x in self.assign["B"] if x.partition == "train")
        rid = a.recording_id.replace(a.recording_id.split(":")[1], "chatterbox")
        extra = dataclasses.replace(a, recording_id=rid)
        rows = self.rows + [dict(self.rows[0], recording_id=rid, component="chatterbox", label=1, pcm_sha256="a" * 64)]
        report = leakage.check_leakage(rows, {"B": self.assign["B"] + [extra]})
        self.assertIn("chatterbox_outside_final_test", self.kinds(report))
        self.assertIn("b_family_not_allowed", self.kinds(report))

    def test_detects_pcm_hash_crossing_and_label_conflict(self):
        tr = next(r for r in self.rows if r["component"] == "genuine" and r["speaker_id"] in {c.split(":")[1] for c in {a.group_id for a in self.assign["B"] if a.partition == "train"}})
        te = next(r for r in self.rows if r["component"] == "melo" and r["speaker_id"] in {c.split(":")[1] for c in {a.group_id for a in self.assign["B"] if a.partition == "test"}})
        te["pcm_sha256"] = tr["pcm_sha256"]
        report = leakage.check_leakage(self.rows, self.assign)
        self.assertIn("pcm_hash_crosses_partitions", self.kinds(report))
        self.assertIn("label_conflict_retained", self.kinds(report))

    def test_detects_missing_class_and_missing_chunks(self):
        rows = [dict(r, retained=False, exclusion_reason="x") if r["component"] == "genuine" and True else r for r in self.rows]
        report = leakage.check_leakage(rows, self.assign, self.chunks)
        self.assertIn("panel_missing_class", self.kinds(report))
        self.assertIn("chunk_for_excluded_recording", self.kinds(report))
        report = leakage.check_leakage(self.rows, self.assign, {"B": [c for c in self.chunks["B"] if c.recording_id != self.chunks["B"][0].recording_id]})
        self.assertIn("recording_without_chunks", self.kinds(report))

    def test_detects_double_assignment_and_orphan_chunk(self):
        report = leakage.check_leakage(self.rows, {"B": self.assign["B"] + [self.assign["B"][0]]})
        self.assertIn("recording_assigned_twice", self.kinds(report))
        report = leakage.check_leakage(self.rows, {"B": self.assign["B"][1:]}, {"B": self.chunks["B"]})
        self.assertIn("orphan_chunk", self.kinds(report))


class Materialise(unittest.TestCase):
    def test_roundtrip_dedupe_and_decode_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "root"
            stems = ["10-20-0000", "10-20-0001"]
            rng = np.random.default_rng(0)
            for component, folder in cohort.SOURCE_DIRS.items():
                (root / folder).mkdir(parents=True)
                for i, stem in enumerate(stems):
                    if component == "vits" and i == 1:
                        continue  # missing file -> decode failure
                    wave = (rng.uniform(-0.5, 0.5, RATE * (1 + i)) if not (component == "melo" and i == 1) else np.zeros(RATE))
                    sf.write(root / folder / f"{stem}.wav", wave.astype("float32"), RATE, subtype="PCM_16")
            # exact duplicate: xtts copy of genuine for stem 0 would be a label conflict
            (root / "dataset/tts/XTTSv2/10-20-0000.wav").write_bytes((root / "dataset/genuine/wav/10-20-0000.wav").read_bytes())
            rows, index = cohort.materialize(stems, root, Path(d) / "shards", shard_bytes=10_000)
            by = {r["recording_id"]: r for r in rows}
            self.assertEqual(len(rows), 10)
            self.assertTrue(by["phonemedf:vits:10-20-0001"]["exclusion_reason"].startswith("read_error:FileNotFoundError:"))
            self.assertFalse(by["phonemedf:vits:10-20-0001"]["retained"])
            self.assertEqual(by["phonemedf:genuine:10-20-0000"]["exclusion_reason"], "label_conflict")
            self.assertEqual(by["phonemedf:xtts:10-20-0000"]["exclusion_reason"], "label_conflict")
            ok = by["phonemedf:melo:10-20-0000"]
            self.assertTrue(ok["retained"])
            shard, key = ok["canonical_path"].split("::")
            pcm = cohort.read_flac(cohort.read_from_shard(Path(d) / "shards" / shard, key))
            self.assertEqual(cohort.pcm_sha256(pcm), ok["pcm_sha256"])
            self.assertEqual(len(pcm), ok["n_samples"])
            self.assertGreater(len({v[0] for v in index.values()}), 1)
            with self.assertRaises(CohortError):
                cohort.materialize(stems, root, Path(d) / "shards", shard_bytes=10_000)

    def test_duplicate_resolution(self):
        rows = [{"recording_id": "b", "label": 1, "pcm_sha256": "h"}, {"recording_id": "a", "label": 1, "pcm_sha256": "h"},
                {"recording_id": "c", "label": 0, "pcm_sha256": "x"}, {"recording_id": "d", "label": None, "pcm_sha256": None}]
        self.assertEqual(cohort.resolve_duplicates(rows), {"b": "exact_duplicate"})


class ReadRetriesAndRepair(unittest.TestCase):
    def test_transient_io_errors_are_retried_but_missing_files_are_not(self):
        calls = []

        def flaky(path):
            calls.append(1)
            if len(calls) < 3:
                raise OSError(5, "Input/output error")
            return b"data"

        waits = []
        with mock.patch.object(cohort, "_read_bytes", flaky):
            self.assertEqual(cohort.read_bytes_with_retries("x", retries=3, backoff=2.0, sleep=waits.append), b"data")
        self.assertEqual((len(calls), waits), (3, [2.0, 4.0]))
        with mock.patch.object(cohort, "_read_bytes", mock.Mock(side_effect=OSError(5, "Input/output error"))) as m:
            with self.assertRaises(OSError):
                cohort.read_bytes_with_retries("x", retries=3, sleep=lambda s: None)
            self.assertEqual(m.call_count, 4)  # first attempt + 3 retries
        with mock.patch.object(cohort, "_read_bytes", mock.Mock(side_effect=FileNotFoundError("gone"))) as m:
            with self.assertRaises(FileNotFoundError):
                cohort.read_bytes_with_retries("x", sleep=lambda s: None)
            self.assertEqual(m.call_count, 1)

    def test_failed_rows_keep_error_class_and_message(self):
        base = {"recording_id": "phonemedf:xtts:1-2-0000", "stem": "1-2-0000", "speaker_id": "1", "chapter_id": "2",
                "component": "xtts", "label": 1, "generator": "xtts"}
        with mock.patch.object(cohort, "_read_bytes", mock.Mock(side_effect=OSError(5, "Input/output error"))), \
                mock.patch.object(cohort.time, "sleep"):
            row, blob = cohort.load_recording(base, "whatever.wav")
        self.assertIsNone(blob)
        self.assertFalse(row["retained"])
        self.assertTrue(row["exclusion_reason"].startswith("read_error:OSError:"))
        self.assertIn("Input/output error", row["exclusion_reason"])
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.wav"
            bad.write_bytes(b"this is not audio")
            row, blob = cohort.load_recording(base, bad)
        self.assertTrue(row["exclusion_reason"].startswith("decode_error:AudioError:"))

    def test_repair_reports_files_that_still_fail_and_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "root"
            (root / "dataset" / "genuine" / "wav").mkdir(parents=True)
            row = {"recording_id": "phonemedf:genuine:1-2-0000", "stem": "1-2-0000", "speaker_id": "1", "chapter_id": "2",
                   "component": "genuine", "label": 0, "generator": None, "duration_s": 0.0, "n_samples": 0, "original_sample_rate": 0,
                   "canonical_path": None, "pcm_sha256": None, "retained": False,
                   "exclusion_reason": "read_error:OSError:Input/output error", "quality_flags": ""}
            rows, result = cohort.repair([row], root, Path(d) / "shards")
            self.assertEqual((result["repaired"], len(result["still_failing"])), ([], 1))
            self.assertTrue(result["still_failing"][0]["reason"].startswith("read_error:FileNotFoundError"))
            self.assertFalse(rows[0]["retained"])
            self.assertFalse((Path(d) / "shards").exists())


def load_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "r1_data_lock.py"
    spec = importlib.util.spec_from_file_location("r1_data_lock_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepairEndToEnd(unittest.TestCase):
    # Runs the real script commands on a miniature cohort: materialize (one read fails like a Drive I/O error),
    # leakage, repair, leakage again.
    def test_materialize_repair_leakage_cycle(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as d:
            root, data_root = Path(d) / "project", Path(d) / "drive"
            script.ROOT = root
            (root / "data" / "splits").mkdir(parents=True)
            stems = make_stems(speakers=30, per=8)
            sp = splits.assign_speakers({cohort.parse_stem(s)[0] for s in stems}, seed=42)
            plan = splits.sample_stems(stems, sp, per_speaker={"train": 4, "validation": 6, "test": 6}, seed=42)
            (root / "data" / "splits" / "cohort_plan_v1.txt").write_text("\n".join(plan) + "\n")
            for protocol in ("A", "B"):
                assignments, _ = splits.build_assignments(protocol, plan, sp)
                splits.write_assignments(assignments, root / "data" / "splits" / f"protocol_{protocol}_v1.parquet")
            rng = np.random.default_rng(5)
            for component, folder in cohort.SOURCE_DIRS.items():
                (data_root / folder).mkdir(parents=True)
                for stem in plan:
                    wave = rng.uniform(-0.5, 0.5, int(rng.integers(8000, 20000))).astype("float32")
                    sf.write(data_root / folder / f"{stem}.wav", wave, RATE, subtype="PCM_16")
            victim_stem = next(s for s in plan if sp[cohort.parse_stem(s)[0]][0] == "train")
            victim = f"phonemedf:xtts:{victim_stem}"
            real_read = cohort._read_bytes

            def drive_glitch(path):
                if Path(path).parent.name == "XTTSv2" and Path(path).stem == victim_stem:
                    raise OSError(5, "Input/output error")
                return real_read(path)

            with mock.patch.object(cohort, "_read_bytes", drive_glitch), mock.patch.object(cohort.time, "sleep"):
                script.main(["materialize", "--project-root", str(data_root)])
            rows = {r["recording_id"]: r for r in cohort.read_table(root / "data" / "manifests" / "cohort_v1.parquet")}
            self.assertEqual([k for k, r in rows.items() if not r["retained"]], [victim])
            self.assertTrue(rows[victim]["exclusion_reason"].startswith("read_error:OSError:"))
            report = json.loads((root / "reports" / "r1" / "materialize.json").read_text())
            self.assertEqual(report["exclusions"], {rows[victim]["exclusion_reason"]: 1})
            self.assertEqual(script.main(["leakage"]), 0)  # the cohort without the recording is internally consistent
            shard_dir = root / "data" / "processed" / "cohort_v1"
            before = {p.name: p.read_bytes() for p in shard_dir.iterdir()}
            chunks_before = len(read_parquet(str(root / "data" / "processed" / "chunks_v1.parquet"), Chunk))

            self.assertEqual(script.main(["repair", "--project-root", str(data_root)]), 0)  # the glitch is gone

            self.assertEqual({p.name: p.read_bytes() for p in shard_dir.iterdir() if p.name in before}, before)  # nothing rewritten
            self.assertEqual(sorted(set(p.name for p in shard_dir.iterdir()) - set(before)), ["repair_00.tar"])
            fixed = {r["recording_id"]: r for r in cohort.read_table(root / "data" / "manifests" / "cohort_v1.parquet")}
            self.assertTrue(all(r["retained"] for r in fixed.values()))
            shard, key = fixed[victim]["canonical_path"].split("::")
            pcm = cohort.read_flac(cohort.read_from_shard(shard_dir / shard, key))
            self.assertEqual(cohort.pcm_sha256(pcm), fixed[victim]["pcm_sha256"])
            direct, _ = cohort.canonical_pcm(data_root / "dataset" / "tts" / "XTTSv2" / f"{victim_stem}.wav")
            np.testing.assert_array_equal(pcm, direct)
            self.assertFalse((root / "reports" / "r1" / "leakage.json").exists())  # the stale report was moved aside
            self.assertTrue(list((root / "reports" / "r1" / "superseded").glob("leakage.*")))
            self.assertTrue(list((root / "data" / "manifests" / "superseded").glob("cohort_v1.pre_repair-*")))

            # existing + repaired chunks must equal a from-scratch rebuild
            expected = set()
            for protocol in ("A", "B"):
                assignments = read_parquet(str(root / "data" / "splits" / f"protocol_{protocol}_v1.parquet"), SplitAssignment)
                expected |= {c.chunk_id for c in chunk_mod.build_chunks(protocol, assignments, fixed)}
            actual = [c.chunk_id for c in read_parquet(str(root / "data" / "processed" / "chunks_v1.parquet"), Chunk)]
            self.assertEqual(len(actual), len(set(actual)))
            self.assertEqual(set(actual), expected)
            self.assertGreater(len(actual), chunks_before)
            self.assertEqual(script.main(["leakage"]), 0)
            repair_report = json.loads(next((root / "reports" / "r1").glob("repair_*.json")).read_text())
            self.assertEqual(repair_report["repaired"], [victim])
            self.assertEqual(repair_report["retained_before"] + 1, repair_report["retained_after"])
            self.assertNotEqual(repair_report["cohort_manifest_sha256_before"], repair_report["cohort_manifest_sha256_after"])
            self.assertEqual(script.main(["repair", "--project-root", str(data_root)]), 0)  # nothing left: no change, success


if __name__ == "__main__":
    unittest.main()
