import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from echoproof.contracts import write_parquet
from echoproof.data import chunks as chunk_mod, cohort, splits
from echoproof.features import r2_s1, r4_s1
from echoproof.features.acoustic import FEATURE_NAMES, handcrafted
from echoproof.features.stats import CROP_SAMPLES

RATE = 16000


def make_pcm(rng, component):
    n = int(rng.uniform(1.0, 6.5) * RATE)
    t = np.arange(n) / RATE
    f0 = rng.uniform(110, 170) if component == "genuine" else rng.uniform(180, 260)
    wave = 0.3 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.04 * rng.standard_normal(n)
    return (np.clip(wave, -1, 1) * 32767).astype(np.int16)


def build_mini_project(root: Path, *, speakers: int = 15, shard_bytes: int = 4_000_000):
    """A miniature locked cohort with BOTH protocol splits under `root` (A for R4-S1, B for the cross-check)."""
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
    out = {"rows": rows, "waves": waves, "shard_dir": shard_dir, "by_id": by_id}
    (root / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "data" / "splits").mkdir(parents=True, exist_ok=True)
    cohort.write_table(rows, root / "data" / "manifests" / "cohort_v1.parquet")
    all_chunks = []
    for protocol in ("A", "B"):
        assignments, split_hash = splits.build_assignments(protocol, plan, sp)
        all_chunks += chunk_mod.build_chunks(protocol, assignments, by_id)
        splits.write_assignments(assignments, root / "data" / "splits" / f"protocol_{protocol}_v1.parquet")
        out[protocol] = {"assignments": assignments, "split_hash": split_hash}
    write_parquet(all_chunks, str(root / "data" / "processed" / "chunks_v1.parquet"))
    out["chunks"] = all_chunks
    return out


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
        return r4_s1.plan_a(m["rows"], m["A"]["assignments"], m["chunks"], **kw)

    def test_scope_is_a_train_plus_a_validation_only(self):
        m, plan = self.mini, self.plan()
        train = {a.recording_id for a in m["A"]["assignments"] if a.partition == "train"}
        val = {a.recording_id for a in m["A"]["assignments"] if a.partition == "validation"}
        self.assertEqual({t["recording_id"] for t in plan.train}, train)
        self.assertEqual({v["recording_id"] for v in plan.validation}, val)
        self.assertEqual(plan.counts["test_assignments_dropped"], sum(a.partition == "test" for a in m["A"]["assignments"]))
        self.assertGreater(plan.counts["test_assignments_dropped"], 0)
        every_component = {"genuine", "melo", "vits", "xtts", "chatterbox"}
        self.assertEqual({t["component"] for t in plan.train}, every_component)
        self.assertEqual({v["component"] for v in plan.validation}, every_component)
        self.assertEqual(set(plan.counts["validation_chunks_by_role"]), set(splits.ROLES))
        self.assertTrue(all(c["crop_id"].startswith("crop:A:") for c in plan.crop_bank))

    def test_b_assignments_are_rejected(self):
        m = self.mini
        with self.assertRaisesRegex(r4_s1.R4S1Error, "protocol"):
            r4_s1.plan_a(m["rows"], m["B"]["assignments"], m["chunks"])

    def test_digest_differs_from_b_and_depends_on_settings(self):
        m = self.mini
        b = r2_s1.plan_b(m["rows"], m["B"]["assignments"], m["chunks"])
        self.assertNotEqual(self.plan().sha256, b.sha256)
        self.assertEqual(self.plan().sha256, self.plan().sha256)
        self.assertNotEqual(self.plan().sha256, self.plan(seed=43).sha256)

    def test_shared_recordings_get_the_b_crops(self):
        m = self.mini
        b = r2_s1.plan_b(m["rows"], m["B"]["assignments"], m["chunks"])
        a = self.plan()
        a_starts = {t["recording_id"]: t["starts"] for t in a.train}
        for t in b.train:
            self.assertEqual(a_starts[t["recording_id"]], t["starts"])
        self.assertGreater(len(a.train), len(b.train))

    def test_inconsistent_inputs_are_rejected(self):
        m = self.mini
        first = next(i for i, c in enumerate(m["chunks"]) if c.protocol_id == "protocol:A" and c.partition == "validation")
        bad = list(m["chunks"])
        c = bad[first]
        bad[first] = replace(c, end_sample=c.end_sample - 1, valid_samples=c.valid_samples - 1, end_seconds=(c.end_sample - 1) / RATE)
        with self.assertRaisesRegex(r4_s1.R4S1Error, "canonical windows"):
            r4_s1.plan_a(m["rows"], m["A"]["assignments"], bad)
        victim = next(a for a in m["A"]["assignments"] if a.partition == "train")
        with self.assertRaisesRegex(r4_s1.R4S1Error, "group"):
            r4_s1.plan_a(m["rows"], [replace(a, group_id="speaker:999999") if a is victim else a for a in m["A"]["assignments"]], m["chunks"])
        with self.assertRaisesRegex(r4_s1.R4S1Error, "twice|duplicate"):
            r4_s1.plan_a(m["rows"], list(m["A"]["assignments"]) + [m["A"]["assignments"][0]], m["chunks"])

    def test_unretained_recordings_are_skipped_and_reported(self):
        m = self.mini
        rid = next(a.recording_id for a in m["A"]["assignments"] if a.partition == "train")
        rows = [dict(r, retained=False, exclusion_reason="x") if r["recording_id"] == rid else r for r in m["rows"]]
        plan = r4_s1.plan_a(rows, m["A"]["assignments"], m["chunks"])
        self.assertEqual(plan.counts["unretained_skipped"], [rid])


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "project"
        cls.mini = build_mini_project(cls.root)
        # A saved B build, as it exists in the real project, so the cross-check and the gap list have something to read.
        cls.b_manifest = r2_s1.build(cls.root, cls.mini["shard_dir"], cls.root / r4_s1.R2_S1_B_DIR)
        b_plan = r2_s1.plan_b(cls.mini["rows"], cls.mini["B"]["assignments"], cls.mini["chunks"])
        index_rows = [{"kind": "crop", "recording_id": c["recording_id"], "start_sample": c["start_sample"], "end_sample": c["end_sample"]}
                      for c in b_plan.crop_bank]
        index_rows += [{"kind": "chunk", "recording_id": v["recording_id"], "start_sample": s, "end_sample": e}
                       for v in b_plan.validation for _, s, e in v["chunks"]]
        index_path = cls.root / r4_s1.R2_U1_B_INDEX
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps({"plan_sha256": "b" * 64, "rows": index_rows}), encoding="utf-8")
        cls.a_plan = r4_s1.plan_a(cls.mini["rows"], cls.mini["A"]["assignments"], cls.mini["chunks"])
        # report_path of the frozen split hash: the real build requires it.
        (cls.root / "reports" / "r1").mkdir(parents=True, exist_ok=True)
        (cls.root / "reports" / "r1" / "split_report.json").write_text(json.dumps(
            {"protocols": {"A": {"split_sha256": cls.mini["A"]["split_hash"]}, "B": {"split_sha256": cls.mini["B"]["split_hash"]}}}), encoding="utf-8")
        cls.manifest = r4_s1.build(cls.root, cls.mini["shard_dir"], cls.root / "out_a")
        cls.out = cls.root / "out_a"
        cls.train = dict(np.load(cls.out / r4_s1.OUTPUT_NAMES["train_features"], allow_pickle=False))
        cls.val = dict(np.load(cls.out / r4_s1.OUTPUT_NAMES["validation_features"], allow_pickle=False))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_manifest_hashes_match_the_files_and_the_inputs(self):
        m = self.manifest
        for key, info in m["outputs"].items():
            self.assertEqual(r2_s1.file_sha256(self.out / info["file"]), info["sha256"], key)
        self.assertEqual(set(m["outputs"]), set(r4_s1.OUTPUT_NAMES))
        self.assertEqual(m["protocol"], "A")
        self.assertEqual(m["inputs"]["split_sha256"], self.mini["A"]["split_hash"])
        self.assertTrue(m["verification"]["passed"])
        self.assertEqual(m["spot_check"]["mismatches"], [])
        self.assertEqual(m["plan_sha256"], self.a_plan.sha256)

    def test_scope_has_every_component_and_no_test_rows(self):
        self.assertEqual(set(self.train["component"].tolist()), {"genuine", "melo", "vits", "xtts", "chatterbox"})
        self.assertEqual(set(self.val["component"].tolist()), {"genuine", "melo", "vits", "xtts", "chatterbox"})
        test_ids = {a.recording_id for a in self.mini["A"]["assignments"] if a.partition == "test"}
        self.assertTrue(test_ids)
        self.assertFalse(test_ids & (set(self.train["recording_id"].tolist()) | set(self.val["recording_id"].tolist())))
        self.assertEqual(self.manifest["counts"]["test_assignments_dropped"], 0)      # build() loads with sealed=False: test rows never reach the plan

    def test_every_row_equals_a_direct_recomputation_from_the_audio(self):
        for i in range(0, len(self.train["y"]), 9):
            wave = waveform(self.mini, str(self.train["recording_id"][i]))
            start = int(self.train["start"][i])
            np.testing.assert_array_equal(self.train["X"][i], handcrafted(wave[start:start + CROP_SAMPLES]))
        for i in range(0, len(self.val["y"]), 7):
            wave = waveform(self.mini, str(self.val["recording_id"][i]))
            np.testing.assert_array_equal(self.val["X"][i], handcrafted(wave[int(self.val["start"][i]):int(self.val["end"][i])]))

    def test_mel_statistics_cover_every_frame_of_every_a_train_recording(self):
        frames = sum(r2_s1.expected_mel_frames(t["n_samples"]) for t in self.a_plan.train)
        with np.load(self.out / r4_s1.OUTPUT_NAMES["mel_stats"], allow_pickle=False) as z:
            self.assertEqual(int(z["frames"]), frames)
            self.assertEqual(int(z["recordings"]), len(self.a_plan.train))

    def test_shared_rows_equal_the_saved_b_features(self):
        cc = json.loads((self.out / r4_s1.CROSSCHECK_NAME).read_text())
        self.assertTrue(cc["passed"], cc)
        self.assertEqual(cc["train_crops"]["max_abs_difference"], 0.0)
        self.assertEqual(cc["validation_chunks"]["b_rows_missing_in_a"], 0)
        self.assertGreater(cc["train_crops"]["matched"], 0)
        self.assertEqual(self.manifest["crosscheck_vs_b_passed"], True)
        self.assertIn("mel_stats_shift_vs_b", cc)

    def test_crosscheck_detects_a_tampered_row(self):
        tables = {"train": dict(self.train), "validation": dict(self.val)}
        tables["train"]["X"] = self.train["X"].copy()
        shared = int(np.flatnonzero(np.isin(self.train["component"], ["genuine", "melo", "vits"]) & ~np.isnan(self.train["X"][:, 0]))[0])
        tables["train"]["X"][shared, 0] += 1.0
        report = r4_s1.crosscheck_against_b(tables, self.root / r4_s1.R2_S1_B_DIR)
        self.assertFalse(report["passed"])
        self.assertEqual(report["train_crops"]["rows_over_tolerance"], 1)

    def test_gap_list_is_exactly_what_b_lacks(self):
        gap = json.loads((self.out / r4_s1.GAP_NAME).read_text())
        b_cache = {(r["kind"], r["recording_id"], r["start_sample"], r["end_sample"])
                   for r in json.loads((self.root / r4_s1.R2_U1_B_INDEX).read_text())["rows"]}
        a_items = {("crop", c["recording_id"], c["start_sample"], c["end_sample"]) for c in self.a_plan.crop_bank}
        a_items |= {("chunk", v["recording_id"], s, e) for v in self.a_plan.validation for _, s, e in v["chunks"]}
        listed = {(r["kind"], r["recording_id"], r["start_sample"], r["end_sample"]) for r in gap["rows"]}
        self.assertEqual(listed, a_items - b_cache)
        self.assertTrue(listed)
        self.assertTrue(all(r["partition"] in ("train", "validation") for r in gap["rows"]))
        self.assertFalse({r["recording_id"] for r in gap["rows"]} & {a.recording_id for a in self.mini["A"]["assignments"] if a.partition == "test"})
        components = {r["component"] for r in gap["rows"]}
        self.assertTrue({"chatterbox", "xtts"} <= components)
        self.assertEqual(json.loads((self.out / "r4_s1_side_outputs.json").read_text())["manifest_sha256"], r2_s1.file_sha256(self.out / r4_s1.MANIFEST_NAME))

    def test_b_outputs_are_untouched_by_the_a_build(self):
        for info in self.b_manifest["outputs"].values():
            self.assertEqual(r2_s1.file_sha256(self.root / r4_s1.R2_S1_B_DIR / info["file"]), info["sha256"])

    def test_two_workers_give_identical_tables(self):
        out2 = self.root / "out_w2"
        r4_s1.build(self.root, self.mini["shard_dir"], out2, workers=2)
        for name in (r4_s1.OUTPUT_NAMES["train_features"], r4_s1.OUTPUT_NAMES["validation_features"], r4_s1.OUTPUT_NAMES["mel_stats"]):
            a, b = np.load(self.out / name, allow_pickle=False), np.load(out2 / name, allow_pickle=False)
            self.assertEqual(set(a.files), set(b.files))
            for key in a.files:
                if key != "plan_sha256":
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f"{name}:{key}")

    def test_finished_outputs_are_never_overwritten(self):
        with self.assertRaisesRegex(r4_s1.R4S1Error, "already complete"):
            r4_s1.build(self.root, self.mini["shard_dir"], self.out)

    def test_wrong_frozen_split_hash_is_rejected(self):
        report = self.root / "reports" / "r1" / "split_report.json"
        original = report.read_text()
        try:
            report.write_text(json.dumps({"protocols": {"A": {"split_sha256": "0" * 64}}}), encoding="utf-8")
            with self.assertRaisesRegex(r4_s1.R4S1Error, "frozen"):
                r4_s1.build(self.root, self.mini["shard_dir"], self.root / "out_bad")
        finally:
            report.write_text(original, encoding="utf-8")

    def test_missing_shard_fails_loudly(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(r4_s1.R4S1Error, "shard files not found"):
                r4_s1.build(self.root, empty, self.root / "out_missing")


class Verification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.mini = build_mini_project(Path(cls.tmp.name))
        cls.plan = r4_s1.plan_a(cls.mini["rows"], cls.mini["A"]["assignments"], cls.mini["chunks"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _tables(self):
        p = self.plan
        width = len(FEATURE_NAMES)
        train = {"crop_id": np.array([c["crop_id"] for c in p.crop_bank]), "recording_id": np.array([c["recording_id"] for c in p.crop_bank]),
                 "group_id": np.array([c["group_id"] for c in p.crop_bank]), "component": np.array([c["component"] for c in p.crop_bank]),
                 "y": np.array([c["label"] for c in p.crop_bank]), "valid_samples": np.array([c["valid_samples"] for c in p.crop_bank]),
                 "X": np.zeros((len(p.crop_bank), width))}
        rows = [(v, ch) for v in p.validation for ch in v["chunks"]]
        val = {"chunk_id": np.array([ch[0] for _, ch in rows]), "recording_id": np.array([v["recording_id"] for v, _ in rows]),
               "group_id": np.array([v["group_id"] for v, _ in rows]), "component": np.array([v["component"] for v, _ in rows]),
               "role": np.array([v["role"] for v, _ in rows]), "y": np.array([v["label"] for v, _ in rows]), "X": np.zeros((len(rows), width))}
        from echoproof.features.stats import MelStats
        stats, frames = MelStats(), {}
        rng = np.random.default_rng(1)
        for t in p.train:
            frames[t["recording_id"]] = stats.add(rng.standard_normal((r2_s1.expected_mel_frames(t["n_samples"]), 64)))
        return {"train": train, "validation": val}, stats, frames

    def test_clean_results_pass(self):
        tables, stats, frames = self._tables()
        self.assertTrue(r4_s1.verify_a(self.plan, tables, stats, frames)["passed"])

    def test_infinity_unexpected_nan_missing_component_and_test_row_are_caught(self):
        tables, stats, frames = self._tables()
        tables["train"]["X"][0, 0] = np.inf
        tables["validation"]["X"][0, 3] = np.nan
        self.assertFalse(r4_s1.verify_a(self.plan, tables, stats, frames)["passed"])
        tables, stats, frames = self._tables()
        keep = tables["train"]["component"] != "chatterbox"
        tables["train"] = {k: v[keep] for k, v in tables["train"].items()}
        report = r4_s1.verify_a(self.plan, tables, stats, frames)
        self.assertTrue(any("missing components" in p for p in report["problems"]))
        tables, stats, frames = self._tables()
        tables["validation"]["recording_id"][0] = next(a.recording_id for a in self.mini["A"]["assignments"] if a.partition == "test")
        self.assertTrue(any("test leakage" in p for p in r4_s1.verify_a(self.plan, tables, stats, frames)["problems"]))

    def test_b_prefixed_ids_and_leaky_speakers_are_caught(self):
        tables, stats, frames = self._tables()
        tables["train"]["crop_id"][0] = tables["train"]["crop_id"][0].replace("crop:A:", "crop:B:")
        self.assertTrue(any("prefix" in p for p in r4_s1.verify_a(self.plan, tables, stats, frames)["problems"]))
        tables, stats, frames = self._tables()
        tables["validation"]["group_id"][0] = tables["train"]["group_id"][0]
        self.assertTrue(any("both train and validation" in p for p in r4_s1.verify_a(self.plan, tables, stats, frames)["problems"]))


if __name__ == "__main__":
    unittest.main()
