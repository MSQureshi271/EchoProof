"""R4-U0 tests: generated FLACs and an untrained fake encoder; no research audio, model download, GPU or network.

The fixture is a miniature of the real situation: a Protocol B WavLM cache is built by the frozen R2-U1 code, R4-S1 derives the gap
of the Protocol A rows that cache lacks, and R4-U0 extracts exactly that gap. Test recordings are undecodable bytes, so any read of
one fails loudly.
"""
from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from echoproof.artifacts import file_digest
from echoproof.contracts import Chunk, SplitAssignment, write_parquet
from echoproof.data.chunks import PREPROCESSING_HASH, eval_windows
from echoproof.data.cohort import flac_bytes, pcm_sha256, write_table
from echoproof.features import r2_u1, r4_s1, r4_u0

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
A_HASH, B_HASH = "a" * 64, "b" * 64
B_TRAIN = {"genuine", "melo", "vits"}
B_VALIDATION = {"genuine", "melo", "vits", "xtts"}
COMPONENTS = ("genuine", "melo", "vits", "xtts", "chatterbox")


class FakeExtractor:
    dim = 3

    def __init__(self, fail_after=None, offset=0.0):
        self.calls, self.fail_after, self.offset = 0, fail_after, offset

    def embed(self, waves, pooling="mean"):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("injected extraction interruption")
        self.calls += 1
        out = []
        for wave in waves:
            x = np.asarray(wave, dtype=np.float32)
            base = np.array([x.mean(), np.std(x), len(x) / RATE], dtype=np.float32)
            spread = np.array([np.sqrt(np.mean(x * x)), np.max(np.abs(x)), np.std(x[:max(1, len(x) // 2)])], dtype=np.float32)
            out.append((base if pooling == "mean" else np.concatenate((base, spread))) + self.offset)
        return np.asarray(out, dtype=np.float32)


class World:
    """Cohort, A and B manifests, shards, the B cache, and the R4-S1 evidence chain, all under one temporary root."""

    def __init__(self, root):
        self.root = Path(root)
        self.rows, self.pcm, self.members = [], {}, {"shard_00.tar": [], "shard_01.tar": []}
        self.a_assign, self.b_assign, self.a_chunks, self.b_chunks = [], [], [], []
        self.identity = r2_u1.extraction_identity(processor_config={"sampling_rate": RATE, "do_normalize": False, "feature_size": 1},
                                                  versions={"numpy": np.__version__, "fixture": "1"},
                                                  source_sha256={"pretrained.py": "a" * 64}, hidden_size=3)
        for component in COMPONENTS:
            self.add(101, 1, component, "train", n=64040)
        self.add(102, 1, "genuine", "train", n=8000)
        for i, role in enumerate(("selection", "calibration", "operating")):
            for utt in range(1, 4):
                for component in COMPONENTS:
                    self.add(201 + i, utt, component, "validation", role, 70000 if (i, utt, component) == (0, 1, "genuine") else 9000 + 100 * utt)
        for component in COMPONENTS:
            self.add(301, 1, component, "test", n=10000)
        self.write()

    def add(self, speaker, utt, component, partition, role=None, n=9000):
        stem = f"{speaker}-2-{utt:04d}"
        rid = f"phonemedf:{component}:{stem}"
        group, label = f"speaker:{speaker}", int(component != "genuine")
        pcm = np.random.default_rng(100 + len(self.rows)).integers(-12000, 12000, n, dtype=np.int16).astype("<i2")
        shard = "shard_00.tar" if partition == "train" else "shard_01.tar"
        self.rows.append({"recording_id": rid, "stem": stem, "speaker_id": str(speaker), "chapter_id": "2", "component": component, "label": label,
                          "generator": None if not label else component, "duration_s": n / RATE, "n_samples": n, "original_sample_rate": RATE,
                          "canonical_path": f"{shard}::{rid}", "pcm_sha256": pcm_sha256(pcm), "retained": True, "exclusion_reason": None,
                          "quality_flags": ""})
        self.pcm[rid] = pcm
        blob = b"DO NOT DECODE A TEST RECORDING" if partition == "test" else flac_bytes(pcm)
        self.members[shard].append((rid.replace(":", "__") + ".flac", blob))
        policy = "genuine" if not label else "known_family"
        self.a_assign.append(SplitAssignment(1, "protocol:A", rid, group, partition, role, policy, "r1-v1", A_HASH))
        in_b = (partition == "train" and component in B_TRAIN) or (partition == "validation" and component in B_VALIDATION)
        if in_b:
            self.b_assign.append(SplitAssignment(1, "protocol:B", rid, group, partition, role,
                                                 "unseen_validation" if component == "xtts" else policy, "r1-v1", B_HASH))
        if partition != "train":
            for start, end in eval_windows(n):
                self.a_chunks.append(Chunk(1, f"chunk:A:{rid}:{start}", rid, group, "protocol:A", partition, start, end, RATE, start / RATE,
                                           end / RATE, end - start, 64000, PREPROCESSING_HASH))
                if in_b:
                    self.b_chunks.append(Chunk(1, f"chunk:B:{rid}:{start}", rid, group, "protocol:B", partition, start, end, RATE, start / RATE,
                                               end / RATE, end - start, 64000, PREPROCESSING_HASH))

    def write(self):
        root = self.root
        shard_dir = root / "shards"
        shard_dir.mkdir(parents=True)
        for name, entries in self.members.items():
            with tarfile.open(shard_dir / name, "w") as tar:
                for member, content in entries:
                    info = tarfile.TarInfo(member)
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))
        self.shard_dir = shard_dir
        self.paths = {"cohort": root / "data/manifests/cohort_v1.parquet", "split": root / "data/splits/protocol_A_v1.parquet",
                      "chunks": root / "data/processed/chunks_v1.parquet", "crop_bank": root / "data/features/r4_s1_A/crop_bank_A_v1.parquet",
                      "gap": root / "data/features/r4_s1_A/wavlm_cache_gap_A_v1.json"}
        for path in self.paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        write_table(self.rows, self.paths["cohort"])
        write_parquet(self.a_assign, str(self.paths["split"]))
        write_parquet(self.a_chunks, str(self.paths["chunks"]))
        self.plan_a = r4_s1.plan_a(self.rows, [a for a in self.a_assign], self.a_chunks)
        r4_s1.write_crop_bank(root / "data/features/r4_s1_A", self.plan_a)
        # B cache: built by the frozen R2-U1 code from the B part of the same recordings; its crops are the A crops of shared recordings.
        b_bank = [{**c, "crop_id": c["crop_id"].replace("crop:A", "crop:B")} for c in self.plan_a.crop_bank if c["component"] in B_TRAIN]
        b_plan = r2_u1.plan_b(self.rows, self.b_assign, self.b_chunks, b_bank, self.identity)
        self.b_dir = root / "data/features/r2_u1_B"
        self.b_manifest = r2_u1.build(b_plan, shard_dir, self.b_dir, FakeExtractor(), fixture=True, parity_n=50)
        self.b_index_path = self.b_dir / "development_index.json"
        gap = r4_s1.wavlm_gap(self.plan_a, self.b_index_path)
        self.paths["gap"].write_text(json.dumps(gap, indent=2, sort_keys=True), encoding="utf-8")
        self.gap = gap
        self.write_evidence()

    def write_evidence(self):
        root, paths = self.root, self.paths
        manifest = {"task": "R4-S1", "verification": {"passed": True}, "crosscheck_vs_b_passed": True, "plan_sha256": self.plan_a.sha256,
                    "inputs": {"cohort_manifest_file_sha256": file_digest(paths["cohort"]), "split_file_sha256": file_digest(paths["split"]),
                               "chunk_manifest_file_sha256": file_digest(paths["chunks"]), "split_sha256": A_HASH},
                    "outputs": {"crop_bank": {"file": paths["crop_bank"].name, "sha256": file_digest(paths["crop_bank"])}}}
        self.s1_manifest = manifest
        self.s1_path = root / "data/features/r4_s1_A/r4_s1_manifest.json"
        self.evidence_path = root / "reports/r4/r4_s1_colab_fixture.json"
        side = {"wavlm_cache_gap_A_v1.json": {"sha256": file_digest(paths["gap"]), "bytes": paths["gap"].stat().st_size}}
        files = {self.s1_path: manifest,
                 self.evidence_path: {"verdict": {"verified": True}, "build_manifest": {**manifest, "side_outputs": side}},
                 root / "reports/r1/split_report.json": {"protocols": {"A": {"split_sha256": A_HASH}}},
                 root / "reports/r1/leakage.json": {"passed": True, "violation_count": 0}}
        for path, payload in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")


class R4U0(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".r4-u0-test-", dir=ROOT)
        self.w = World(Path(self.tmp.name))
        self.root = self.w.root
        self.b_rows = json.loads(self.w.b_index_path.read_text(encoding="utf-8"))["rows"]

    def tearDown(self):
        self.tmp.cleanup()

    def preflight(self):
        return r4_u0.preflight(self.root, identity=self.w.identity)

    def plan_direct(self, *, assignments=None, b_rows=None, gap=None):
        return r4_u0.plan_gap(self.w.rows, assignments or [a for a in self.w.a_assign if a.partition != "test"], self.w.a_chunks,
                              self.w.plan_a.crop_bank, b_rows or self.b_rows, gap or self.w.gap, self.w.identity)

    def build(self, plan=None, extractor=None, out="a_cache", **options):
        return r4_u0.build(plan or self.preflight(), self.w.shard_dir, self.root / out, extractor or FakeExtractor(), b_dir=self.w.b_dir,
                           b_index_rows=self.b_rows, fixture=True, parity_n=10, b_parity_n=10, **options)

    # ---- planning ----
    def test_plan_is_exactly_the_gap_with_a_complete_partition_of_the_a_rows(self):
        plan = self.preflight()
        n_train = sum(1 for c in self.w.plan_a.crop_bank if c["component"] in ("xtts", "chatterbox"))
        n_chunks = sum(1 for c in self.w.a_chunks if c.partition == "validation" and c.recording_id.split(":")[1] == "chatterbox")
        self.assertEqual(len(plan.rows), n_train + n_chunks)
        self.assertEqual(plan.counts["by_component"], {"chatterbox": sum(1 for r in plan.rows if r["component"] == "chatterbox"),
                                                       "xtts": sum(1 for r in plan.rows if r["component"] == "xtts")})
        cov = plan.provenance["coverage"]
        self.assertEqual(cov["a_rows"], cov["b_cache_rows_in_a"] + cov["gap_rows"])
        self.assertEqual(cov["a_rows"], len(self.w.plan_a.crop_bank) + sum(c.partition == "validation" for c in self.w.a_chunks))
        self.assertEqual({r["partition"] for r in plan.rows}, {"train", "validation"})
        self.assertTrue(all(r["scope"] == "development" for r in plan.rows))
        self.assertEqual(plan.provenance["test_access"], "none; A test is sealed and no A test row is planned or extracted")

    def test_plan_ignores_test_chunks_and_train_chunks(self):
        extra = [Chunk(1, f"chunk:A:{r['recording_id']}:0", r["recording_id"], f"speaker:{r['speaker_id']}", "protocol:A", "train", 0,
                       min(64000, r["n_samples"]), RATE, 0.0, min(64000, r["n_samples"]) / RATE, min(64000, r["n_samples"]), 64000,
                       PREPROCESSING_HASH)
                 for r in self.w.rows if r["speaker_id"] == "102"]
        plan = r4_u0.plan_gap(self.w.rows, [a for a in self.w.a_assign if a.partition != "test"], self.w.a_chunks + extra,
                              self.w.plan_a.crop_bank, self.b_rows, self.w.gap, self.w.identity)
        self.assertEqual(plan.rows, self.preflight().rows)

    def test_a_test_assignment_is_refused_not_skipped(self):
        with self.assertRaisesRegex(r2_u1.CacheError, "A test is sealed"):
            self.plan_direct(assignments=self.w.a_assign)

    def test_preflight_rejects_any_changed_published_input(self):
        for name in ("cohort", "split", "chunks", "crop_bank", "gap"):
            original = self.w.paths[name].read_bytes()
            try:
                self.w.paths[name].write_bytes(original + b" ")
                with self.subTest(input=name), self.assertRaisesRegex(ValueError, "input file hashes"):
                    self.preflight()
            finally:
                self.w.paths[name].write_bytes(original)

    def test_preflight_requires_a_matching_verified_colab_report(self):
        original = self.w.evidence_path.read_text(encoding="utf-8")
        report = json.loads(original)
        report["verdict"]["verified"] = False
        self.w.evidence_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "matching verified Colab evidence"):
            self.preflight()
        report["verdict"]["verified"] = True
        report["build_manifest"]["plan_sha256"] = "0" * 64
        self.w.evidence_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "matching verified Colab evidence"):
            self.preflight()

    def test_preflight_rejects_failed_leakage_changed_split_and_unverified_manifest(self):
        for rel, payload, pattern in (
                ("reports/r1/leakage.json", {"passed": True, "violation_count": 2}, "leakage"),
                ("reports/r1/split_report.json", {"protocols": {"A": {"split_sha256": "c" * 64}}}, "frozen A split")):
            path = self.root / rel
            original = path.read_text(encoding="utf-8")
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(file=rel), self.assertRaisesRegex(ValueError, pattern):
                self.preflight()
            path.write_text(original, encoding="utf-8")
        manifest = dict(self.w.s1_manifest, verification={"passed": False})
        self.w.s1_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not verified"):
            self.preflight()

    def test_gap_file_must_equal_the_independently_derived_gap(self):
        missing = deepcopy(self.w.gap)
        missing["rows"].pop()
        with self.assertRaisesRegex(ValueError, "gap file differs"):
            self.plan_direct(gap=missing)
        extra = deepcopy(self.w.gap)
        row = next(r for r in self.b_rows if r["kind"] == "chunk")
        extra["rows"].append({k: row[k] for k in extra["rows"][0]})
        with self.assertRaisesRegex(ValueError, "gap file differs"):
            self.plan_direct(gap=extra)
        wrong = deepcopy(self.w.gap)
        wrong["summary"] = {**wrong["summary"], "crop|train|xtts": wrong["summary"]["crop|train|xtts"] + 1}
        with self.assertRaisesRegex(ValueError, "summary"):
            self.plan_direct(gap=wrong)
        dup = deepcopy(self.w.gap)
        dup["rows"].append(dict(dup["rows"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate row"):
            self.plan_direct(gap=dup)

    def test_gap_row_that_disagrees_with_the_manifests_is_rejected(self):
        for field, value in (("component", "melo"), ("label", 0), ("shard", "shard_09.tar"), ("group_id", "speaker:9")):
            gap = deepcopy(self.w.gap)
            gap["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.plan_direct(gap=gap)

    def test_b_cache_rows_outside_a_or_from_test_are_rejected(self):
        rows = deepcopy(self.b_rows)
        rows[0]["partition"] = "test"
        with self.assertRaisesRegex(ValueError, "test row"):
            self.plan_direct(b_rows=rows)
        rows = deepcopy(self.b_rows)
        rows[0]["end_sample"] -= 1
        with self.assertRaisesRegex(ValueError, "not A rows"):
            self.plan_direct(b_rows=rows)

    def test_label_or_speaker_mismatch_is_rejected(self):
        rows = deepcopy(self.w.rows)
        rows[0]["label"] = 1
        with self.assertRaisesRegex(ValueError, "label"):
            r4_u0.plan_gap(rows, [a for a in self.w.a_assign if a.partition != "test"], self.w.a_chunks, self.w.plan_a.crop_bank,
                           self.b_rows, self.w.gap, self.w.identity)

    # ---- build ----
    def test_build_publishes_a_verified_gap_cache_without_touching_test_audio_or_the_b_cache(self):
        b_before = {p.name: file_digest(p) for p in self.w.b_dir.rglob("*") if p.is_file()}
        plan = self.preflight()
        manifest = self.build(plan)
        self.assertEqual(manifest["status"], "fixture_verified")
        self.assertEqual(manifest["test_rows_extracted"], 0)
        self.assertTrue(manifest["identity_vs_b"]["equal"])
        self.assertEqual(manifest["b_parity"]["items"], 10)
        self.assertEqual(manifest["index"]["rows"], len(plan.rows))
        self.assertEqual({p["file"] for p in manifest["verification"]["packs"].values()}, {"parts/part_shard_00.npz", "parts/part_shard_01.npz"})
        self.assertEqual(b_before, {p.name: file_digest(p) for p in self.w.b_dir.rglob("*") if p.is_file()})
        self.assertTrue(r4_u0.verify_cache(plan, self.root / "a_cache")["passed"])
        self.assertFalse((self.root / "a_cache/.writer.lock").exists())

    def test_finished_cache_is_not_overwritten_and_resume_only_verifies(self):
        plan = self.preflight()
        self.build(plan)
        with self.assertRaisesRegex(ValueError, "finished cache exists"):
            self.build(plan)
        extractor = FakeExtractor()
        self.build(plan, extractor, resume=True)
        self.assertEqual(extractor.calls, 0)

    def test_interrupted_run_publishes_nothing_and_resume_reuses_finished_shards(self):
        plan = self.preflight()
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.build(plan, FakeExtractor(fail_after=1))
        out = self.root / "a_cache"
        self.assertFalse((out / r4_u0.MANIFEST_NAME).exists())
        done = list((out / "parts").glob("*.npz"))
        self.assertEqual(len(done), 1)
        with self.assertRaisesRegex(ValueError, "partial cache exists"):
            self.build(plan)
        manifest = self.build(plan, resume=True)
        self.assertEqual(len(manifest["packs_reused_this_session"]), 1)
        self.assertEqual(len(manifest["packs_extracted_this_session"]), 1)

    def test_a_run_that_does_not_reproduce_the_b_vectors_is_not_published(self):
        with self.assertRaisesRegex(ValueError, "does not reproduce|parity"):
            self.build(extractor=FakeExtractor(offset=0.5))
        self.assertFalse((self.root / "a_cache" / r4_u0.MANIFEST_NAME).exists())

    def test_b_parity_checks_only_rows_in_staged_shards_and_is_deterministic(self):
        plan = self.preflight()
        rows = r4_u0.b_parity_rows(self.b_rows, set(plan.shards()), 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual([r["item_id"] for r in rows], [r["item_id"] for r in r4_u0.b_parity_rows(self.b_rows, set(plan.shards()), 10)])
        self.assertTrue(all(r["scope"] == "development" for r in rows))
        with self.assertRaisesRegex(ValueError, "not enough"):
            r4_u0.b_parity_rows(self.b_rows, {"shard_99.tar"}, 10)

    def test_corrupt_or_missing_pack_is_rejected(self):
        plan = self.preflight()
        self.build(plan)
        pack = self.root / "a_cache/parts/part_shard_00.npz"
        original = pack.read_bytes()
        pack.write_bytes(original[:-20] + b"\0" * 20)
        with self.assertRaises(ValueError):
            r4_u0.verify_cache(plan, self.root / "a_cache")
        pack.write_bytes(original)
        pack.unlink()
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            r4_u0.verify_cache(plan, self.root / "a_cache")

    def test_existing_writer_lock_blocks_a_second_writer(self):
        (self.root / "a_cache").mkdir()
        (self.root / "a_cache/.writer.lock").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "writer lock"):
            self.build()

    def test_identity_difference_from_b_is_recorded_not_hidden(self):
        diff = r4_u0.identity_vs_b({**self.w.identity, "versions": {"torch": "9"}}, self.w.identity)
        self.assertFalse(diff["equal"])
        self.assertIn("versions", diff["differences"])

    # ---- reader ----
    def test_reader_returns_every_a_row_in_a_order_from_the_right_cache(self):
        self.build()
        for pooling, width in (("mean", 3), ("mean_std", 6)):
            rows, matrix = r4_u0.read_embeddings_a(self.root, pooling=pooling, a_dir=self.root / "a_cache")
            self.assertEqual(matrix.shape, (len(rows), width))
            self.assertEqual(len(rows), len(self.w.plan_a.crop_bank) + sum(c.partition == "validation" for c in self.w.a_chunks))
            self.assertTrue(np.isfinite(matrix).all())
        crops = [r["item_id"] for r in rows if r["kind"] == "crop"]
        self.assertEqual(crops, [c["crop_id"] for c in self.w.plan_a.crop_bank])
        self.assertTrue(all(r["item_id"].startswith(("crop:A:", "chunk:A:")) for r in rows))
        self.assertEqual({r["partition"] for r in rows}, {"train", "validation"})
        # every vector equals a direct extraction, whichever cache it came from
        ext = FakeExtractor()
        for r in rows[:: max(1, len(rows) // 25)]:
            pcm = self.w.pcm[r["recording_id"]].astype(np.float32) / 32768.0
            direct = ext.embed([pcm[r["start_sample"]:r["end_sample"]]], pooling="mean")[0]
            i = rows.index(r)
            rm, mm = r4_u0.read_embeddings_a(self.root, pooling="mean", a_dir=self.root / "a_cache")
            np.testing.assert_allclose(mm[i], direct, atol=1e-6)
        self.assertEqual({r["source"] for r in rows}, {"a", "b"})

    def test_reader_partition_filter_and_no_test(self):
        self.build()
        rows, _ = r4_u0.read_embeddings_a(self.root, partition="train", a_dir=self.root / "a_cache")
        self.assertEqual({r["partition"] for r in rows}, {"train"})
        with self.assertRaisesRegex(ValueError, "sealed"):
            r4_u0.read_embeddings_a(self.root, partition="test", a_dir=self.root / "a_cache")

    def test_reader_refuses_unverified_manifest_changed_index_and_changed_pack(self):
        self.build()
        a = self.root / "a_cache"
        manifest_path = a / r4_u0.MANIFEST_NAME
        original = manifest_path.read_text(encoding="utf-8")
        m = json.loads(original)
        m["b_parity"]["passed"] = False
        manifest_path.write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not verified"):
            r4_u0.read_embeddings_a(self.root, a_dir=a)
        manifest_path.write_text(original, encoding="utf-8")
        index = a / r4_u0.INDEX_NAME
        index_bytes = index.read_bytes()
        index.write_bytes(index_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "index checksum"):
            r4_u0.read_embeddings_a(self.root, a_dir=a)
        index.write_bytes(index_bytes)
        pack = a / "parts/part_shard_00.npz"
        data = pack.read_bytes()
        pack.write_bytes(data + b"x")
        with self.assertRaisesRegex(ValueError, "checksum"):
            r4_u0.read_embeddings_a(self.root, a_dir=a)

    def test_reader_fails_when_a_row_is_held_by_neither_cache(self):
        self.build()
        a = self.root / "a_cache"
        index = json.loads((a / r4_u0.INDEX_NAME).read_text(encoding="utf-8"))
        index["rows"].pop()
        (a / r4_u0.INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
        m = json.loads((a / r4_u0.MANIFEST_NAME).read_text(encoding="utf-8"))
        m["index"]["sha256"] = file_digest(a / r4_u0.INDEX_NAME)
        (a / r4_u0.MANIFEST_NAME).write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "neither cache"):
            r4_u0.read_embeddings_a(self.root, a_dir=a)

    # ---- frozen B code ----
    def test_frozen_b_sources_match_the_gr3_lock(self):
        lock = json.loads((ROOT / "locks/B_lock_gr3.json").read_text(encoding="utf-8"))
        text = json.dumps(lock)
        for rel in ("src/echoproof/features/r2_u1.py", "src/echoproof/features/r2_s1.py", "src/echoproof/features/pretrained.py"):
            self.assertIn(file_digest(ROOT / rel), text, f"{rel} differs from the hash recorded in the GR3 lock")


if __name__ == "__main__":
    unittest.main()
