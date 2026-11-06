import copy
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from echoproof.contracts import Chunk, SplitAssignment, write_parquet
from echoproof.data import chunks as chunk_mod, cohort, splits
from echoproof.features import r2_s1
from echoproof.features.acoustic import FEATURE_NAMES, handcrafted, log_mel
from echoproof.features.stats import (CROP_SAMPLES, MelStats, crop_bank_starts, load_mel_stats, normalize_log_mel)

RATE = 16000


# ---- crop bank -------------------------------------------------------------------------------------------------------------

class CropBank(unittest.TestCase):
    def test_deterministic_increasing_in_range_and_one_per_segment(self):
        n = CROP_SAMPLES * 3 + 123
        starts = crop_bank_starts(42, "phonemedf:melo:1-2-0001", n)
        self.assertEqual(starts, crop_bank_starts(42, "phonemedf:melo:1-2-0001", n))
        self.assertEqual(len(starts), 5)
        self.assertEqual(starts, sorted(set(starts)))
        possible = n - CROP_SAMPLES + 1
        for j, s in enumerate(starts):
            self.assertTrue((j * possible) // 5 <= s < ((j + 1) * possible) // 5, (j, s))
        self.assertTrue(0 <= starts[0] and starts[-1] + CROP_SAMPLES <= n)

    def test_seed_and_recording_change_the_crops_but_k_can_vary(self):
        n = CROP_SAMPLES * 4
        base = crop_bank_starts(42, "r:a", n)
        self.assertNotEqual(base, crop_bank_starts(43, "r:a", n))
        self.assertNotEqual(base, crop_bank_starts(42, "r:b", n))
        self.assertEqual(len(crop_bank_starts(42, "r:a", n, k=3)), 3)

    def test_short_and_barely_longer_recordings(self):
        self.assertEqual(crop_bank_starts(42, "r:a", 1000), [0])
        self.assertEqual(crop_bank_starts(42, "r:a", CROP_SAMPLES), [0])
        starts = crop_bank_starts(42, "r:a", CROP_SAMPLES + 2)         # only 3 valid start positions for 5 requested crops
        self.assertEqual(len(starts), 3)
        self.assertEqual(sorted(starts), [0, 1, 2])

    def test_invalid_arguments(self):
        for bad in ((0,), (-5,), (1.5,)):
            with self.assertRaises(ValueError):
                crop_bank_starts(42, "r:a", bad[0])
        with self.assertRaises(ValueError):
            crop_bank_starts(42, "r:a", 1000, k=0)


# ---- streaming statistics --------------------------------------------------------------------------------------------------

class Statistics(unittest.TestCase):
    def test_matches_a_two_pass_computation_whatever_the_split_or_merge_order(self):
        rng = np.random.default_rng(0)
        pieces = [rng.standard_normal((n, 64)) * s + m for n, s, m in ((300, 2.0, -5.0), (1, 1.0, 0.0), (47, 0.3, 9.0), (1200, 4.0, -20.0))]
        everything = np.concatenate(pieces)
        one = MelStats()
        for p in pieces:
            one.add(p)
        mean, std = one.finalize()
        np.testing.assert_allclose(mean, everything.mean(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(std, everything.std(axis=0), rtol=1e-12)         # population std
        left, right = MelStats(), MelStats()
        for p in pieces[:2]:
            left.add(p)
        for p in pieces[2:]:
            right.add(p)
        right.merge(left)
        np.testing.assert_allclose(right.finalize()[1], std, rtol=1e-12)
        self.assertEqual((right.count, right.recordings), (everything.shape[0], 4))

    def test_numerically_stable_for_large_offsets(self):
        rng = np.random.default_rng(1)
        x = 1e6 + rng.standard_normal((5000, 64))
        s = MelStats()
        s.add(x[:2500])
        s.add(x[2500:])
        np.testing.assert_allclose(s.finalize()[1], x.std(axis=0), rtol=1e-6)

    def test_bad_input_and_too_little_data(self):
        s = MelStats()
        for bad in (np.zeros((0, 64)), np.zeros((5, 63)), np.full((5, 64), np.nan), np.zeros(64)):
            with self.assertRaises(ValueError):
                s.add(bad)
        s.add(np.ones((1, 64)))
        with self.assertRaises(ValueError):
            s.finalize()                                  # one frame has no spread

    def test_array_round_trip_and_file_loader(self):
        rng = np.random.default_rng(2)
        s = MelStats()
        s.add(rng.standard_normal((100, 64)))
        again = MelStats.from_arrays(s.to_arrays())
        np.testing.assert_array_equal(again.m2, s.m2)
        self.assertEqual((again.count, again.recordings), (100, 1))
        with tempfile.TemporaryDirectory() as d:
            mean, std = s.finalize()
            np.savez(Path(d) / "m.npz", mean=mean, std=std, frames=np.int64(100), recordings=np.int64(1))
            loaded = load_mel_stats(Path(d) / "m.npz")
            self.assertEqual(loaded["mean"].dtype, np.float32)
            frames = rng.standard_normal((10, 64)).astype(np.float32)
            out = normalize_log_mel(frames, loaded["mean"], loaded["std"])
            np.testing.assert_allclose(out, (frames - mean) / std, rtol=1e-5, atol=1e-5)
            np.savez(Path(d) / "bad.npz", mean=mean, std=np.zeros(64), frames=np.int64(1), recordings=np.int64(1))
            with self.assertRaises(ValueError):
                load_mel_stats(Path(d) / "bad.npz")
        with self.assertRaises(ValueError):
            normalize_log_mel(np.zeros((3, 10), np.float32), np.zeros(64), np.ones(64))


# ---- miniature cohort ------------------------------------------------------------------------------------------------------

def make_pcm(rng, component):
    n = int(rng.uniform(1.0, 6.5) * RATE)
    t = np.arange(n) / RATE
    f0 = rng.uniform(110, 170) if component == "genuine" else rng.uniform(180, 260)
    wave = 0.3 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.04 * rng.standard_normal(n)
    return (np.clip(wave, -1, 1) * 32767).astype(np.int16)


def build_mini_project(root: Path, *, speakers: int = 15, shard_bytes: int = 4_000_000):
    """Write a miniature locked cohort (manifest, split, chunks, shards) under `root`; return its pieces."""
    rng = np.random.default_rng(7)
    stems = [f"{1000 + s}-{200 + s}-{u:04d}" for s in range(speakers) for u in range(6)]
    sp = splits.assign_speakers({cohort.parse_stem(s)[0] for s in stems}, seed=42)
    plan = splits.sample_stems(stems, sp, per_speaker={"train": 2, "validation": 2, "test": 2}, seed=42)
    waves, rows = {}, []
    for stem in plan:
        speaker, chapter, _ = cohort.parse_stem(stem)
        for component in cohort.COMPONENTS:
            rid = cohort.recording_id(component, stem)
            waves[rid] = make_pcm(rng, component)
            rows.append({"recording_id": rid, "stem": stem, "speaker_id": speaker, "chapter_id": chapter, "component": component,
                         "label": cohort.LABEL[component], "generator": cohort.GENERATOR[component], "duration_s": len(waves[rid]) / RATE,
                         "n_samples": int(len(waves[rid])), "original_sample_rate": RATE, "canonical_path": None,
                         "pcm_sha256": cohort.pcm_sha256(waves[rid]), "retained": True, "exclusion_reason": None, "quality_flags": ""})
    shard_dir = root / "data" / "processed" / "cohort_v1"
    index = cohort.write_shards(((rid, cohort.flac_bytes(waves[rid])) for rid in sorted(waves)), shard_dir, shard_bytes=shard_bytes)
    for row in rows:
        row["canonical_path"] = f"{index[row['recording_id']][0]}::{row['recording_id']}"
    by_id = {r["recording_id"]: r for r in rows}
    assignments, split_hash = splits.build_assignments("B", plan, sp)
    chunks = chunk_mod.build_chunks("B", assignments, by_id)
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "data" / "splits").mkdir(parents=True, exist_ok=True)
    cohort.write_table(rows, root / "data" / "manifests" / "cohort_v1.parquet")
    splits.write_assignments(assignments, root / "data" / "splits" / "protocol_B_v1.parquet")
    write_parquet(chunks, str(root / "data" / "processed" / "chunks_v1.parquet"))
    return {"rows": rows, "assignments": assignments, "chunks": chunks, "waves": waves, "shard_dir": shard_dir, "split_hash": split_hash,
            "by_id": by_id}


def waveform(mini, recording_id):
    return mini["waves"][recording_id].astype(np.float32) / 32768.0


class Planning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.mini = build_mini_project(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def plan(self, **kw):
        m = self.mini
        return r2_s1.plan_b(m["rows"], m["assignments"], m["chunks"], **kw)

    def test_scope_is_b_train_plus_b_validation_only(self):
        m, plan = self.mini, self.plan()
        train = {a.recording_id for a in m["assignments"] if a.partition == "train"}
        val = {a.recording_id for a in m["assignments"] if a.partition == "validation"}
        self.assertEqual({t["recording_id"] for t in plan.train}, train)
        self.assertEqual({v["recording_id"] for v in plan.validation}, val)
        self.assertEqual(plan.counts["test_assignments_dropped"], sum(a.partition == "test" for a in m["assignments"]))
        self.assertGreater(plan.counts["test_assignments_dropped"], 0)
        everything = {t["recording_id"] for t in plan.train} | {v["recording_id"] for v in plan.validation}
        self.assertFalse(any(rid.startswith("phonemedf:chatterbox:") for rid in everything))
        self.assertEqual({t["component"] for t in plan.train}, {"genuine", "melo", "vits"})
        self.assertEqual({v["component"] for v in plan.validation}, {"genuine", "melo", "vits", "xtts"})
        self.assertEqual({c["recording_id"] for c in plan.crop_bank}, train)
        self.assertEqual(set(plan.counts["validation_chunks_by_role"]), set(splits.ROLES))

    def test_crop_bank_and_chunk_counts(self):
        plan = self.plan()
        self.assertEqual(plan.counts["train_crops"], sum(len(t["starts"]) for t in plan.train))
        self.assertTrue(all(1 <= len(t["starts"]) <= 5 for t in plan.train))
        self.assertTrue(any(len(t["starts"]) == 1 for t in plan.train) and any(len(t["starts"]) > 1 for t in plan.train))
        expected = sum(1 for c in self.mini["chunks"] if c.partition == "validation")
        self.assertEqual(plan.counts["validation_chunks"], expected)
        for crop in plan.crop_bank:
            self.assertEqual(crop["end_sample"] - crop["start_sample"], crop["valid_samples"])
            self.assertLessEqual(crop["valid_samples"], CROP_SAMPLES)
            n = self.mini["by_id"][crop["recording_id"]]["n_samples"]
            self.assertLessEqual(crop["end_sample"], n)
        self.assertEqual(len({c["crop_id"] for c in plan.crop_bank}), len(plan.crop_bank))

    def test_digest_is_deterministic_and_depends_on_the_settings(self):
        self.assertEqual(self.plan().sha256, self.plan().sha256)
        self.assertNotEqual(self.plan().sha256, self.plan(seed=43).sha256)
        self.assertNotEqual(self.plan().sha256, self.plan(k=3).sha256)

    def test_chatterbox_in_train_or_validation_is_a_hard_error(self):
        m = self.mini
        for partition, role in (("validation", "selection"), ("train", None)):
            source = next(a for a in m["assignments"] if a.partition == partition)
            rid = "phonemedf:chatterbox:" + source.recording_id.rsplit(":", 1)[1]
            self.assertEqual(next(r for r in m["rows"] if r["recording_id"] == rid)["component"], "chatterbox")
            bad = list(m["assignments"]) + [replace(source, recording_id=rid, partition=partition, validation_role=role)]
            with self.assertRaisesRegex(r2_s1.R2S1Error, "may not appear"):
                r2_s1.plan_b(m["rows"], bad, m["chunks"])

    def test_xtts_in_train_is_a_hard_error(self):
        m = self.mini
        a0 = next(a for a in m["assignments"] if a.partition == "train")
        rid = a0.recording_id.rsplit(":", 1)[1]
        xtts = f"phonemedf:xtts:{rid}"
        bad = list(m["assignments"]) + [replace(a0, recording_id=xtts)]
        with self.assertRaisesRegex(r2_s1.R2S1Error, "may not appear"):
            r2_s1.plan_b(m["rows"], bad, m["chunks"])

    def test_inconsistent_inputs_are_rejected(self):
        m = self.mini
        first = next(i for i, c in enumerate(m["chunks"]) if c.partition == "validation")
        bad = list(m["chunks"])
        c = bad[first]
        bad[first] = replace(c, end_sample=c.end_sample - 1, valid_samples=c.valid_samples - 1, end_seconds=(c.end_sample - 1) / RATE)
        with self.assertRaisesRegex(r2_s1.R2S1Error, "canonical windows"):
            r2_s1.plan_b(m["rows"], m["assignments"], bad)
        with self.assertRaisesRegex(r2_s1.R2S1Error, "group"):
            victim = next(a for a in m["assignments"] if a.partition == "train")
            r2_s1.plan_b(m["rows"], [replace(a, group_id="speaker:999999") if a is victim else a for a in m["assignments"]], m["chunks"])
        with self.assertRaisesRegex(r2_s1.R2S1Error, "twice|duplicate"):
            r2_s1.plan_b(m["rows"], list(m["assignments"]) + [m["assignments"][0]], m["chunks"])
        with self.assertRaisesRegex(r2_s1.R2S1Error, "protocol"):
            r2_s1.plan_b(m["rows"], [replace(m["assignments"][0], protocol_id="protocol:A")], m["chunks"])

    def test_unretained_recordings_are_skipped_and_reported(self):
        m = self.mini
        rid = next(a.recording_id for a in m["assignments"] if a.partition == "train")
        rows = [dict(r, retained=False, exclusion_reason="x") if r["recording_id"] == rid else r for r in m["rows"]]
        plan = r2_s1.plan_b(rows, m["assignments"], m["chunks"])
        self.assertEqual(plan.counts["unretained_skipped"], [rid])
        self.assertNotIn(rid, {t["recording_id"] for t in plan.train})


# ---- end to end ------------------------------------------------------------------------------------------------------------

class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "project"
        cls.mini = build_mini_project(cls.root)
        cls.manifest = r2_s1.build(cls.root, cls.mini["shard_dir"], cls.root / "out1")
        cls.out = cls.root / "out1"
        cls.train = dict(np.load(cls.out / r2_s1.OUTPUT_NAMES["train_features"], allow_pickle=False))
        cls.val = dict(np.load(cls.out / r2_s1.OUTPUT_NAMES["validation_features"], allow_pickle=False))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_manifest_hashes_match_the_files_and_the_inputs(self):
        m = self.manifest
        for key, info in m["outputs"].items():
            self.assertEqual(r2_s1.file_sha256(self.out / info["file"]), info["sha256"], key)
        self.assertEqual(set(m["outputs"]), set(r2_s1.OUTPUT_NAMES))
        self.assertEqual(m["inputs"]["split_sha256"], self.mini["split_hash"])
        self.assertEqual(m["inputs"]["feature_schema_hash"], r2_s1.FEATURE_SCHEMA_HASH)
        self.assertEqual(m["inputs"]["cohort_manifest_rows_digest"], r2_s1.rows_digest(self.mini["rows"]))
        self.assertTrue(m["verification"]["passed"])
        self.assertEqual(m["spot_check"]["mismatches"], [])
        self.assertLess(m["spot_check"]["streaming_vs_two_pass_max_abs"], 1e-9)
        self.assertEqual(json.loads((self.out / r2_s1.MANIFEST_NAME).read_text())["plan_sha256"], m["plan_sha256"])

    def test_every_row_equals_a_direct_recomputation_from_the_audio(self):
        from echoproof.models.lr_detector import chunk_features
        for i in range(0, len(self.train["y"]), 7):
            wave = waveform(self.mini, str(self.train["recording_id"][i]))
            start = int(self.train["start"][i])
            piece = wave[start:start + CROP_SAMPLES]
            np.testing.assert_array_equal(self.train["X"][i], handcrafted(piece))
            np.testing.assert_array_equal(self.train["X"][i], chunk_features([piece])[0])      # the production LR feature function
            self.assertEqual(int(self.train["valid_samples"][i]), len(piece))
        for i in range(0, len(self.val["y"]), 5):
            wave = waveform(self.mini, str(self.val["recording_id"][i]))
            np.testing.assert_array_equal(self.val["X"][i], handcrafted(wave[int(self.val["start"][i]):int(self.val["end"][i])]))

    def test_tables_follow_the_manifests(self):
        m = self.mini
        by_id = m["by_id"]
        self.assertEqual(self.train["X"].shape, (len(self.train["y"]), len(FEATURE_NAMES)))
        self.assertEqual(list(self.train["feature_names"]), list(FEATURE_NAMES))
        for rid, y, g, c in zip(self.train["recording_id"], self.train["y"], self.train["group_id"], self.train["component"]):
            self.assertEqual((int(y), str(c), str(g)), (by_id[str(rid)]["label"], by_id[str(rid)]["component"], f"speaker:{by_id[str(rid)]['speaker_id']}"))
        roles = {a.recording_id: a.validation_role for a in m["assignments"]}
        for rid, role in zip(self.val["recording_id"], self.val["role"]):
            self.assertEqual(str(role), roles[str(rid)])
        chunk_ids = {c.chunk_id for c in m["chunks"] if c.partition == "validation"}
        self.assertEqual(set(self.val["chunk_id"].tolist()), chunk_ids)
        self.assertEqual(set(self.val["y"].tolist()), {0, 1})
        speakers_train, speakers_val = set(self.train["group_id"].tolist()), set(self.val["group_id"].tolist())
        self.assertFalse(speakers_train & speakers_val)
        self.assertFalse(any("chatterbox" in str(r) for r in list(self.train["recording_id"]) + list(self.val["recording_id"])))

    def test_mel_statistics_cover_every_frame_of_every_train_recording(self):
        train_ids = [a.recording_id for a in self.mini["assignments"] if a.partition == "train"]
        frames = np.concatenate([log_mel(waveform(self.mini, r)).astype(np.float64) for r in sorted(train_ids)])
        stats = load_mel_stats(self.out / r2_s1.OUTPUT_NAMES["mel_stats"])
        self.assertEqual(stats["frames"], len(frames))
        self.assertEqual(stats["recordings"], len(train_ids))
        np.testing.assert_allclose(stats["mean"], frames.mean(axis=0), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(stats["std"], frames.std(axis=0), rtol=1e-5)
        self.assertEqual(self.manifest["mel_stats"]["frames"], len(frames))

    def test_crop_bank_file_matches_the_tables(self):
        bank = cohort.read_table(self.out / r2_s1.OUTPUT_NAMES["crop_bank"])
        self.assertEqual([r["crop_id"] for r in bank], self.train["crop_id"].tolist())
        self.assertEqual([r["start_sample"] for r in bank], self.train["start"].tolist())

    def test_finished_outputs_are_never_overwritten(self):
        with self.assertRaisesRegex(r2_s1.R2S1Error, "already complete"):
            r2_s1.build(self.root, self.mini["shard_dir"], self.out)

    def test_two_workers_give_identical_tables(self):
        r2_s1.build(self.root, self.mini["shard_dir"], self.root / "out_w2", workers=2)
        other_train = np.load(self.root / "out_w2" / r2_s1.OUTPUT_NAMES["train_features"], allow_pickle=False)
        other_val = np.load(self.root / "out_w2" / r2_s1.OUTPUT_NAMES["validation_features"], allow_pickle=False)
        np.testing.assert_array_equal(other_train["X"], self.train["X"])
        np.testing.assert_array_equal(other_val["X"], self.val["X"])
        np.testing.assert_array_equal(other_train["crop_id"], self.train["crop_id"])
        a = np.load(self.root / "out_w2" / r2_s1.OUTPUT_NAMES["mel_stats"], allow_pickle=False)
        b = np.load(self.out / r2_s1.OUTPUT_NAMES["mel_stats"], allow_pickle=False)
        np.testing.assert_allclose(a["std"], b["std"], rtol=1e-12)

    def test_resume_redoes_only_the_failed_shard(self):
        real = r2_s1.process_shard
        shards = []

        def flaky(job):
            shards.append(job["shard"])
            if job["shard"] == sorted(j for j in shards)[0] and not flaky.failed:
                flaky.failed = True
                raise OSError("simulated Drive read error")
            return real(job)
        flaky.failed = False
        out = self.root / "out_resume"
        with mock.patch.object(r2_s1, "process_shard", flaky):
            with self.assertRaisesRegex(r2_s1.R2S1Error, "shards failed"):
                r2_s1.build(self.root, self.mini["shard_dir"], out)
        parts = sorted((out / "parts").glob("part_*.npz"))
        self.assertGreaterEqual(len(parts), 1)
        with self.assertRaisesRegex(r2_s1.R2S1Error, "resume"):
            r2_s1.build(self.root, self.mini["shard_dir"], out)
        before = {p.name: p.stat().st_mtime_ns for p in parts}
        manifest = r2_s1.build(self.root, self.mini["shard_dir"], out, resume=True)
        self.assertEqual(sum(1 for s in manifest["shards"] if s.get("skipped")), len(parts))
        self.assertTrue(all(p.stat().st_mtime_ns == before[p.name] for p in parts))            # untouched
        resumed = np.load(out / r2_s1.OUTPUT_NAMES["train_features"], allow_pickle=False)
        np.testing.assert_array_equal(resumed["X"], self.train["X"])

    def test_missing_shard_and_corrupted_audio_fail_loudly(self):
        broken = Path(self.tmp.name) / "broken"
        shutil.copytree(self.root / "data", broken / "data")
        some = sorted((broken / "data" / "processed" / "cohort_v1").glob("shard_*.tar"))[0]
        some.unlink()
        with self.assertRaisesRegex(r2_s1.R2S1Error, "shard files not found"):
            r2_s1.build(broken, broken / "data" / "processed" / "cohort_v1", broken / "out")
        # a manifest hash that no longer matches the audio in the shard
        rows = cohort.read_table(self.root / "data" / "manifests" / "cohort_v1.parquet")
        victim = next(r for r in rows if r["recording_id"] == next(a.recording_id for a in self.mini["assignments"] if a.partition == "train"))
        victim["pcm_sha256"] = "0" * 64
        wrong = Path(self.tmp.name) / "wrong"
        shutil.copytree(self.root / "data", wrong / "data")
        (wrong / "data" / "manifests" / "cohort_v1.parquet").unlink()
        cohort.write_table(rows, wrong / "data" / "manifests" / "cohort_v1.parquet")
        with self.assertRaisesRegex(r2_s1.R2S1Error, "shards failed"):
            r2_s1.build(wrong, wrong / "data" / "processed" / "cohort_v1", wrong / "out")

    def test_split_hash_must_match_the_frozen_value(self):
        project = Path(self.tmp.name) / "frozen"
        shutil.copytree(self.root / "data", project / "data")
        (project / "reports" / "r1").mkdir(parents=True)
        (project / "reports" / "r1" / "split_report.json").write_text(json.dumps({"protocols": {"B": {"split_sha256": "f" * 64}}}))
        with self.assertRaisesRegex(r2_s1.R2S1Error, "frozen"):
            r2_s1.build(project, project / "data" / "processed" / "cohort_v1", project / "out")
        (project / "reports" / "r1" / "split_report.json").write_text(
            json.dumps({"protocols": {"B": {"split_sha256": self.mini["split_hash"]}}}))
        manifest = r2_s1.build(project, project / "data" / "processed" / "cohort_v1", project / "out")
        self.assertEqual(manifest["inputs"]["split_sha256_frozen_in_report"], self.mini["split_hash"])


class VerificationCatchesProblems(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "project"
        cls.mini = build_mini_project(cls.root, speakers=12)
        plan = r2_s1.plan_b(cls.mini["rows"], cls.mini["assignments"], cls.mini["chunks"])
        parts = Path(cls.tmp.name) / "parts"
        _, failures = r2_s1.run_jobs(plan.jobs(cls.mini["shard_dir"], parts))
        assert not failures, failures
        cls.plan = plan
        cls.tables, cls.stats, cls.frames = r2_s1.merge_parts(plan, parts)
        cls.parts = parts

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, mutate):
        tables = copy.deepcopy(self.tables)
        stats = copy.deepcopy(self.stats)
        frames = dict(self.frames)
        mutate(tables, stats, frames)
        return r2_s1.verify(self.plan, tables, stats, frames)

    def test_clean_results_pass(self):
        report = self.check(lambda *_: None)
        self.assertTrue(report["passed"], report["problems"])
        self.assertEqual(set(report["nan_columns"]) - r2_s1.EXPECTED_NAN_COLUMNS, set())

    def test_chatterbox_infinity_and_unexpected_nan_are_caught(self):
        def chatterbox(t, s, f):
            t["train"]["component"][0] = "chatterbox"
        self.assertFalse(self.check(chatterbox)["passed"])

        def inf(t, s, f):
            t["train"]["X"][0, 3] = np.inf
        self.assertFalse(self.check(inf)["passed"])

        def nan(t, s, f):
            t["validation"]["X"][0, 3] = np.nan
        self.assertTrue(any("unexpected" in p for p in self.check(nan)["problems"]))

    def test_missing_frames_leaky_speakers_and_missing_classes_are_caught(self):
        def frames(t, s, f):
            s.count -= 1
        self.assertFalse(self.check(frames)["passed"])

        def leak(t, s, f):
            t["validation"]["group_id"][0] = t["train"]["group_id"][0]
        self.assertTrue(any("both train and validation" in p for p in self.check(leak)["problems"]))

        def one_class(t, s, f):
            t["validation"]["y"][:] = 1
        self.assertFalse(self.check(one_class)["passed"])

    def test_merge_rejects_parts_from_another_plan(self):
        other = r2_s1.plan_b(self.mini["rows"], self.mini["assignments"], self.mini["chunks"], seed=1)
        with self.assertRaisesRegex(r2_s1.R2S1Error, "different plan|differ"):
            r2_s1.merge_parts(other, self.parts)

    def test_spot_check_detects_a_tampered_row(self):
        from echoproof.training.datasets import DictWaveformLoader
        loader = DictWaveformLoader({k: v.astype(np.float32) / 32768.0 for k, v in self.mini["waves"].items()})
        good = r2_s1.spot_check(self.tables, loader, n=10)
        self.assertEqual(good["mismatches"], [])
        tampered = copy.deepcopy(self.tables)
        tampered["train"]["X"] += 1e-3
        tampered["validation"]["X"] += 1e-3
        bad = r2_s1.spot_check(tampered, loader, n=10)
        self.assertGreater(len(bad["mismatches"]), 0)

    def test_outputs_cannot_overwrite_existing_files(self):
        out = Path(self.tmp.name) / "written"
        r2_s1.write_outputs(out, self.plan, self.tables, self.stats, {})
        with self.assertRaisesRegex(r2_s1.R2S1Error, "overwrite"):
            r2_s1.write_outputs(out, self.plan, self.tables, self.stats, {})


if __name__ == "__main__":
    unittest.main()
