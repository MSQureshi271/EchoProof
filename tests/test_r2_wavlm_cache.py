"""R2-U1 cache acceptance tests using generated FLACs and an untrained fake encoder.

No research recordings, pretrained downloads, GPU, or network are used. Temporary
artifacts stay under this repository. Real WavLM verification belongs to Colab.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from echoproof.artifacts import file_digest
from echoproof.contracts import Chunk, SplitAssignment, write_parquet
from echoproof.data.chunks import PREPROCESSING_HASH, eval_windows
from echoproof.data.cohort import flac_bytes, pcm_sha256, write_table
from echoproof.features import r2_u1


ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
SPLIT_HASH = "1" * 64


class FakeExtractor:
    """Length-sensitive deterministic features; deliberately not a speech model."""

    dim = 3

    def __init__(self, fail_after=None, offset=0.0):
        self.calls = 0
        self.lengths = []
        self.fail_after = fail_after
        self.offset = offset

    def embed(self, waves, pooling="mean"):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("injected extraction interruption")
        self.calls += 1
        result = []
        for wave in waves:
            x = np.asarray(wave, dtype=np.float32)
            self.lengths.append(len(x))
            base = np.array([x.mean(), np.std(x), len(x) / RATE], dtype=np.float32)
            spread = np.array([np.sqrt(np.mean(x * x)), np.max(np.abs(x)), np.std(x[:max(1, len(x) // 2)])],
                              dtype=np.float32)
            value = base if pooling == "mean" else np.concatenate((base, spread))
            result.append(value + self.offset)
        return np.asarray(result, dtype=np.float32)


def fixture(root):
    """Build >50 development items, all roles, long/short clips, and sealed negatives."""
    rows, assignments, chunks, bank, pcm_by_id = [], [], [], [], {}
    members = {"shard_00.tar": [], "shard_01.tar": []}

    def add(speaker, utterance, component, partition, role=None, n=9000):
        stem = f"{speaker}-2-{utterance:04d}"
        rid = f"phonemedf:{component}:{stem}"
        group = f"speaker:{speaker}"
        label = int(component != "genuine")
        rng = np.random.default_rng(100 + len(rows))
        pcm = rng.integers(-12000, 12000, n, dtype=np.int16).astype("<i2")
        shard = "shard_00.tar" if partition == "train" else "shard_01.tar"
        rows.append({"recording_id": rid, "stem": stem, "speaker_id": str(speaker), "chapter_id": "2",
                     "component": component, "label": label, "generator": None if not label else component,
                     "duration_s": n / RATE, "n_samples": n, "original_sample_rate": RATE,
                     "canonical_path": f"{shard}::{rid}", "pcm_sha256": pcm_sha256(pcm),
                     "retained": True, "exclusion_reason": None, "quality_flags": ""})
        policy = ("genuine" if not label else "unseen_validation" if component == "xtts" and partition == "validation"
                  else "unseen_final" if component == "chatterbox" and partition == "test" else "known_family")
        assignments.append(SplitAssignment(1, "protocol:B", rid, group, partition, role,
                                           policy, "r1-v1", SPLIT_HASH))
        pcm_by_id[rid] = pcm
        # Forbidden test TTS is deliberately undecodable: a scope leak fails loudly.
        blob = flac_bytes(pcm) if component != "chatterbox" else b"DO NOT DECODE CHATTERBOX"
        members[shard].append((rid.replace(":", "__") + ".flac", blob))
        if partition == "train":
            for start in ([0, 10, 20, 30, 40] if n > 64000 else [0]):
                end = min(n, start + 64000)
                bank.append({"crop_id": f"crop:B:{rid}:{start}", "recording_id": rid, "group_id": group,
                             "component": component, "label": label, "start_sample": start, "end_sample": end,
                             "valid_samples": end - start, "bank_version": "crop-bank-v1", "bank_seed": 42,
                             "bank_k": 5})
        else:
            for start, end in eval_windows(n):
                chunks.append(Chunk(1, f"chunk:B:{rid}:{start}", rid, group, "protocol:B", partition,
                                    start, end, RATE, start / RATE, end / RATE, end - start, 64000,
                                    PREPROCESSING_HASH))

    for component in ("genuine", "melo", "vits"):
        add(101, 1, component, "train", n=64040)
    add(102, 1, "genuine", "train", n=8000)
    for i, role in enumerate(("selection", "calibration", "operating")):
        for utterance in range(1, 4):
            for component in ("genuine", "melo", "vits", "xtts"):
                n = 70000 if i == 0 and utterance == 1 and component == "genuine" else 9000 + 100 * utterance
                add(201 + i, utterance, component, "validation", role, n)
    for component in ("genuine", "melo", "vits", "chatterbox"):
        add(301, 1, component, "test", n=10000)
    shard_dir = root / "shards"
    shard_dir.mkdir()
    for name, entries in members.items():
        with tarfile.open(shard_dir / name, "w") as tar:
            for member, content in entries:
                info = tarfile.TarInfo(member)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    identity = r2_u1.extraction_identity(processor_config={"sampling_rate": RATE, "do_normalize": False,
                                                         "feature_size": 1},
                                        versions={"numpy": np.__version__, "fixture": "1"},
                                        source_sha256={"pretrained.py": "a" * 64}, hidden_size=3)
    return {"rows": rows, "assignments": assignments, "chunks": chunks, "bank": bank,
            "identity": identity, "shard_dir": shard_dir, "pcm": pcm_by_id}


class CacheAcceptance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".r2-u1-test-", dir=ROOT)
        self.root = Path(self.tmp.name)
        self.data = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, *, include_test=False, **changes):
        d = {**self.data, **changes}
        return r2_u1.plan_b(d["rows"], d["assignments"], d["chunks"], d["bank"], d["identity"],
                            include_test_genuine=include_test)

    def build(self, plan=None, extractor=None, **options):
        return r2_u1.build(plan or self.plan(), self.data["shard_dir"], self.root / "cache",
                           extractor or FakeExtractor(), fixture=True, parity_n=50, **options)

    def preflight_artifacts(self):
        """Persist tiny metadata with the same frozen-evidence chain as R2-S1."""
        paths = {"cohort": self.root / "data/manifests/cohort_v1.parquet",
                 "split": self.root / "data/splits/protocol_B_v1.parquet",
                 "chunks": self.root / "data/processed/chunks_v1.parquet",
                 "crop_bank": self.root / "data/features/r2_s1_B/crop_bank_B_v1.parquet"}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        write_table(self.data["rows"], paths["cohort"])
        write_table(self.data["bank"], paths["crop_bank"])
        write_parquet(self.data["assignments"], str(paths["split"]))
        write_parquet(self.data["chunks"], str(paths["chunks"]))
        manifest = {"verification": {"passed": True},
                    "inputs": {"cohort_manifest_file_sha256": file_digest(paths["cohort"]),
                               "split_file_sha256": file_digest(paths["split"]),
                               "chunk_manifest_file_sha256": file_digest(paths["chunks"]),
                               "split_sha256": SPLIT_HASH},
                    "outputs": {"crop_bank": {"sha256": file_digest(paths["crop_bank"])}},
                    "counts": {"train_crops": len(self.data["bank"]),
                               "validation_chunks": sum(c.partition == "validation" for c in self.data["chunks"])}}
        paths.update(manifest=self.root / "data/features/r2_s1_B/r2_s1_manifest.json",
                     evidence=self.root / "reports/r2/r2_s1_colab_fixture.json",
                     split_report=self.root / "reports/r1/split_report.json",
                     leakage=self.root / "reports/r1/leakage.json")
        payloads = {"manifest": manifest,
                    "evidence": {"verdict": {"verified": True}, "build_manifest": manifest},
                    "split_report": {"protocols": {"B": {"split_sha256": SPLIT_HASH}}},
                    "leakage": {"passed": True, "violation_count": 0}}
        for name, payload in payloads.items():
            paths[name].parent.mkdir(parents=True, exist_ok=True)
            paths[name].write_text(json.dumps(payload), encoding="utf-8")
        return paths

    def test_preflight_reads_published_metadata_and_records_evidence_without_audio(self):
        paths = self.preflight_artifacts()
        plan = r2_u1.preflight(self.root, identity=self.data["identity"])
        self.assertEqual(plan.rows, self.plan().rows)
        self.assertEqual(plan.provenance["input_sha256"],
                         {name: file_digest(paths[name]) for name in ("cohort", "split", "chunks", "crop_bank")})
        self.assertEqual(plan.provenance["r2_s1_verified_evidence"],
                         [{"file": "reports/r2/r2_s1_colab_fixture.json", "sha256": file_digest(paths["evidence"])}])
        self.assertEqual(plan.provenance["test_access"], "none")
        # The metadata fixture never writes audio at the canonical project path.
        self.assertFalse((self.root / "data/processed/cohort_v1").exists())

    def test_preflight_rejects_independently_changed_input_files(self):
        paths = self.preflight_artifacts()
        for name in ("cohort", "split", "chunks", "crop_bank"):
            original = paths[name].read_bytes()
            try:
                paths[name].write_bytes(original + b"changed after publication")
                with self.subTest(input=name), self.assertRaisesRegex(ValueError, "input file hashes"):
                    r2_u1.preflight(self.root, identity=self.data["identity"])
            finally:
                paths[name].write_bytes(original)

    def test_preflight_requires_matching_verified_colab_anchor(self):
        paths = self.preflight_artifacts()
        original = paths["evidence"].read_text(encoding="utf-8")
        cases = {"missing": None, "unverified": json.loads(original), "different_manifest": json.loads(original)}
        cases["unverified"]["verdict"]["verified"] = False
        cases["different_manifest"]["build_manifest"]["counts"]["train_crops"] += 1
        for name, evidence in cases.items():
            if evidence is None:
                paths["evidence"].unlink()
            else:
                paths["evidence"].write_text(json.dumps(evidence), encoding="utf-8")
            with self.subTest(evidence=name), self.assertRaisesRegex(ValueError, "not anchored"):
                r2_u1.preflight(self.root, identity=self.data["identity"])
            paths["evidence"].write_text(original, encoding="utf-8")
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["verification"]["passed"] = False
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verification did not pass"):
            r2_u1.preflight(self.root, identity=self.data["identity"])

    def test_preflight_rejects_failed_or_nonzero_leakage_evidence(self):
        paths = self.preflight_artifacts()
        for leakage in ({"passed": False, "violation_count": 0}, {"passed": True, "violation_count": 1}):
            paths["leakage"].write_text(json.dumps(leakage), encoding="utf-8")
            with self.subTest(leakage=leakage), self.assertRaisesRegex(ValueError, "leakage evidence"):
                r2_u1.preflight(self.root, identity=self.data["identity"])

    def test_preflight_rejects_changed_frozen_split_report(self):
        paths = self.preflight_artifacts()
        paths["split_report"].write_text(json.dumps({"protocols": {"B": {"split_sha256": "2" * 64}}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen split mismatch"):
            r2_u1.preflight(self.root, identity=self.data["identity"])

    def test_default_plan_is_development_only_and_uses_exact_published_crops(self):
        plan = self.plan()
        self.assertEqual({r["partition"] for r in plan.rows}, {"train", "validation"})
        self.assertEqual({r["role"] for r in plan.rows if r["partition"] == "validation"},
                         {"selection", "calibration", "operating"})
        actual = {(r["item_id"], r["start_sample"], r["end_sample"]) for r in plan.rows if r["partition"] == "train"}
        expected = {(r["crop_id"], r["start_sample"], r["end_sample"]) for r in self.data["bank"]}
        self.assertEqual(actual, expected)
        self.assertFalse(any(r["component"] == "chatterbox" for r in plan.rows))

    def test_test_opt_in_adds_only_genuine_and_keeps_scope_distinct(self):
        plan = self.plan(include_test=True)
        tests = [r for r in plan.rows if r["partition"] == "test"]
        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0]["component"], "genuine")
        self.assertEqual(tests[0]["scope"], "test_genuine")
        self.assertTrue(all(r["scope"] == "development" for r in plan.rows if r["partition"] != "test"))

    def test_input_order_does_not_change_plan_or_content_keys(self):
        plan = self.plan()
        d = self.data
        other = self.plan(rows=list(reversed(d["rows"])), assignments=list(reversed(d["assignments"])),
                          chunks=list(reversed(d["chunks"])), bank=list(reversed(d["bank"])))
        self.assertEqual(plan.sha256, other.sha256)
        self.assertEqual(plan.rows, other.rows)

    def test_forbidden_generator_in_training_is_rejected(self):
        for component in ("xtts", "chatterbox"):
            rows = deepcopy(self.data["rows"])
            rows[0]["component"] = component
            with self.subTest(component=component), self.assertRaises(ValueError):
                self.plan(rows=rows)

    def test_duplicate_assignments_and_cohort_rows_are_rejected(self):
        for changes in ({"assignments": self.data["assignments"] + [self.data["assignments"][0]]},
                        {"rows": self.data["rows"] + [self.data["rows"][0]]}):
            with self.subTest(kind=list(changes)), self.assertRaises(ValueError):
                self.plan(**changes)

    def test_bad_speaker_group_and_label_are_rejected(self):
        assignments = list(self.data["assignments"])
        assignments[0] = replace(assignments[0], group_id="speaker:999")
        with self.assertRaises(ValueError):
            self.plan(assignments=assignments)
        rows = deepcopy(self.data["rows"])
        rows[0]["label"] = 1
        with self.assertRaises(ValueError):
            self.plan(rows=rows)

    def test_cross_partition_pcm_duplicates_are_rejected(self):
        rows = deepcopy(self.data["rows"])
        val = next(r for r in rows if r["speaker_id"] == "201")
        val["pcm_sha256"] = rows[0]["pcm_sha256"]
        with self.assertRaises(ValueError):
            self.plan(rows=rows)

    def test_missing_and_duplicate_bank_crops_are_rejected(self):
        short = next(r for r in self.data["rows"] if r["speaker_id"] == "102")
        missing = [r for r in self.data["bank"] if r["recording_id"] != short["recording_id"]]
        for bank in (missing, self.data["bank"] + [self.data["bank"][0]]):
            with self.subTest(size=len(bank)), self.assertRaises(ValueError):
                self.plan(bank=bank)

    def test_crop_bounds_and_valid_length_are_rejected(self):
        for change in ({"end_sample": 100000}, {"valid_samples": 7}, {"start_sample": -1}):
            bank = deepcopy(self.data["bank"])
            bank[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan(bank=bank)

    def test_missing_duplicate_and_wrong_preprocessing_chunks_are_rejected(self):
        for chunks in (self.data["chunks"][1:], self.data["chunks"] + [self.data["chunks"][0]],
                       [replace(self.data["chunks"][0], preprocessing_hash="b" * 64)] + self.data["chunks"][1:]):
            with self.subTest(size=len(chunks)), self.assertRaises(ValueError):
                self.plan(chunks=chunks)

    def test_canonical_paths_cannot_escape_shard_directory(self):
        for bad in ("../outside.tar::member", "C:/outside.tar::member", "shard_00.tar::../member"):
            rows = deepcopy(self.data["rows"])
            rows[0]["canonical_path"] = bad
            with self.subTest(path=bad), self.assertRaises(ValueError):
                self.plan(rows=rows)

    def test_identity_or_audio_changes_invalidate_keys_and_pooling_keys_differ(self):
        original = self.plan()
        first = original.rows[0]
        self.assertNotEqual(first["key_mean"], first["key_mean_std"])
        identity = r2_u1.extraction_identity(processor_config={"sampling_rate": RATE, "do_normalize": False,
                                                              "feature_size": 1},
                                             versions={"numpy": np.__version__, "fixture": "2"},
                                             source_sha256={"pretrained.py": "b" * 64}, hidden_size=3)
        other = self.plan(identity=identity)
        self.assertNotEqual(original.sha256, other.sha256)
        self.assertNotEqual(first["key_mean"], other.rows[0]["key_mean"])
        rows = deepcopy(self.data["rows"])
        rid = first["recording_id"]
        next(r for r in rows if r["recording_id"] == rid)["pcm_sha256"] = "f" * 64
        changed = self.plan(rows=rows)
        self.assertNotEqual(first["key_mean"], changed.rows[0]["key_mean"])
        # Valid interval/revision changes must never address the old representation.
        revision = deepcopy(self.data["identity"])
        revision["revision"] = "c" * 40
        variants = ((first["start_sample"] + 1, first["end_sample"], self.data["identity"]),
                    (first["start_sample"], first["end_sample"] - 1, self.data["identity"]),
                    (first["start_sample"], first["end_sample"], revision))
        for start, end, identity in variants:
            self.assertNotEqual(first["key_mean"],
                                r2_u1.cache_key(first["pcm_sha256"], start, end, identity, "mean"))

    def test_build_roundtrip_preserves_order_both_pooling_views_and_short_lengths(self):
        plan, encoder = self.plan(), FakeExtractor()
        self.build(plan, encoder, batch_size=7)
        rows, mean = r2_u1.read_embeddings(self.root / "cache", pooling="mean")
        rows2, combined = r2_u1.read_embeddings(self.root / "cache", pooling="mean_std")
        self.assertEqual(rows, rows2)
        self.assertEqual(len(rows), len(plan.rows))
        self.assertEqual(mean.shape, (len(rows), 3))
        self.assertEqual(combined.shape, (len(rows), 6))
        np.testing.assert_array_equal(mean, combined[:, :3])
        self.assertEqual(mean.dtype, np.float32)
        self.assertTrue(np.isfinite(combined).all())
        self.assertIn(8000, encoder.lengths)
        self.assertNotIn(0, encoder.lengths)
        r2_u1.verify_cache(plan, self.root / "cache")

    def test_reader_keeps_test_genuine_out_unless_explicitly_requested(self):
        self.build(self.plan(include_test=True))
        rows, _ = r2_u1.read_embeddings(self.root / "cache")
        self.assertFalse(any(r["partition"] == "test" for r in rows))
        rows, _ = r2_u1.read_embeddings(self.root / "cache", partition="test", include_test_genuine=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["component"], "genuine")

    def test_pcm_hash_mismatch_blocks_publication(self):
        rows = deepcopy(self.data["rows"])
        rows[0]["pcm_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            self.build(self.plan(rows=rows))
        self.assertFalse(list((self.root / "cache").glob("*manifest.json")))

    def test_waveform_length_mismatch_blocks_publication(self):
        rows = deepcopy(self.data["rows"])
        # Keep crop coordinates valid while claiming extra PCM samples.
        rows[0]["n_samples"] += 1
        rows[0]["duration_s"] = rows[0]["n_samples"] / RATE
        with self.assertRaises(ValueError):
            self.build(self.plan(rows=rows))
        self.assertFalse(list((self.root / "cache").glob("*manifest.json")))

    def test_encoder_interruption_never_publishes_a_completed_manifest(self):
        with self.assertRaises(RuntimeError):
            self.build(extractor=FakeExtractor(fail_after=0))
        cache = self.root / "cache"
        self.assertFalse(list(cache.glob("*manifest.json")))
        self.assertFalse(list(cache.glob("parts/*.npz")))
        self.build(extractor=FakeExtractor(), resume=True)
        r2_u1.verify_cache(self.plan(), cache)

    def test_corrupted_pack_is_rejected_by_verification_and_reader(self):
        plan = self.plan()
        self.build(plan)
        pack = next((self.root / "cache" / "parts").glob("*.npz"))
        with np.load(pack, allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
        arrays["mean"][0, 0] += np.float32(0.5)
        with pack.open("wb") as handle:
            np.savez(handle, **arrays)
        with self.assertRaises(ValueError):
            r2_u1.verify_cache(plan, self.root / "cache")
        with self.assertRaises(ValueError):
            r2_u1.read_embeddings(self.root / "cache")

    def test_changed_plan_cannot_resume_existing_generation(self):
        self.build()
        identity = r2_u1.extraction_identity(processor_config={"sampling_rate": RATE, "do_normalize": False,
                                                              "feature_size": 1},
                                             versions={"numpy": np.__version__, "fixture": "2"},
                                             source_sha256={"pretrained.py": "b" * 64}, hidden_size=3)
        with self.assertRaises(ValueError):
            self.build(self.plan(identity=identity), resume=True)

    def test_nonfinite_embeddings_are_rejected(self):
        class BadExtractor(FakeExtractor):
            def embed(self, waves, pooling="mean"):
                result = super().embed(waves, pooling=pooling)
                result[0, 0] = np.nan
                return result
        with self.assertRaises(ValueError):
            self.build(extractor=BadExtractor())
        self.assertFalse(list((self.root / "cache").glob("*manifest.json")))

    def test_actual_checkpoint_run_cannot_be_claimed_with_fixture_identity(self):
        with self.assertRaises(ValueError):
            r2_u1.build(self.plan(), self.data["shard_dir"], self.root / "cache", FakeExtractor(), fixture=False)

    def test_published_parity_is_50_development_items_and_explicitly_untrained(self):
        plan = self.plan(include_test=True)
        manifest = self.build(plan)
        parity = manifest["parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["items"], 50)
        self.assertEqual(len(set(parity["item_ids"])), 50)
        by_id = {row["item_id"]: row for row in plan.rows}
        self.assertTrue(all(by_id[item]["scope"] == "development" for item in parity["item_ids"]))
        self.assertEqual(set(parity["max_abs_embedding_difference"]), {"mean", "mean_std"})
        self.assertEqual(set(parity["max_abs_diagnostic_score_difference"]), {"mean", "mean_std"})
        self.assertIn("UNTRAINED", parity["diagnostic_head"])
        self.assertIn("pending R3", parity["trained_adapter_parity"])
        self.assertEqual(manifest["status"], "fixture_verified")
        self.assertFalse(manifest["research_accuracy_claim"])

    def test_completed_resume_validates_without_calling_encoder(self):
        plan = self.plan()
        first = self.build(plan)
        must_not_run = FakeExtractor(fail_after=0)
        again = self.build(plan, must_not_run, resume=True)
        self.assertEqual(first, again)
        self.assertEqual(must_not_run.calls, 0)

    def test_direct_parity_drift_blocks_final_manifest(self):
        class DriftingExtractor(FakeExtractor):
            def embed(self, waves, pooling="mean"):
                result = super().embed(waves, pooling=pooling)
                return result + np.float32(self.calls * 0.01)
        with self.assertRaisesRegex(ValueError, "parity"):
            self.build(extractor=DriftingExtractor())
        self.assertFalse(list((self.root / "cache").glob("*manifest.json")))

    def test_existing_writer_lock_is_not_silently_removed(self):
        cache = self.root / "cache"
        cache.mkdir()
        lock = cache / ".writer.lock"
        lock.write_text("another writer", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lock"):
            self.build()
        self.assertEqual(lock.read_text(encoding="utf-8"), "another writer")

    def test_committed_index_corruption_is_rejected_by_independent_verification(self):
        plan = self.plan()
        manifest = self.build(plan)
        index = self.root / "cache" / manifest["indices"]["development"]["file"]
        payload = json.loads(index.read_text(encoding="utf-8"))
        payload["rows"][0]["start_sample"] += 1
        index.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            r2_u1.verify_cache(plan, self.root / "cache")

    def test_missing_committed_pack_is_rejected(self):
        plan = self.plan()
        self.build(plan)
        next((self.root / "cache" / "parts").glob("*.npz")).unlink()
        with self.assertRaises(ValueError):
            r2_u1.verify_cache(plan, self.root / "cache")

    def test_partial_resume_reuses_completed_shard_and_rechecks_parity(self):
        plan = self.plan()
        # The bounded batch spans recordings: each fixture shard fits in one
        # batch. Interrupt the second shard after the first pack is committed.
        with self.assertRaises(RuntimeError):
            self.build(plan, FakeExtractor(fail_after=1), batch_size=64)
        packs = list((self.root / "cache" / "parts").glob("*.npz"))
        self.assertEqual(len(packs), 1)
        old_bytes = packs[0].read_bytes()
        manifest = self.build(plan, FakeExtractor(), batch_size=64, resume=True)
        self.assertEqual(manifest["packs_reused_this_session"], ["shard_00.tar"])
        self.assertEqual(manifest["packs_extracted_this_session"], ["shard_01.tar"])
        self.assertEqual(old_bytes, packs[0].read_bytes())
        self.assertTrue(manifest["parity"]["passed"])

    def test_failed_atomic_rename_leaves_no_committed_pack_or_manifest(self):
        with patch.object(r2_u1.os, "replace", side_effect=OSError("injected rename failure")):
            with self.assertRaises(OSError):
                self.build()
        cache = self.root / "cache"
        self.assertFalse(list(cache.glob("parts/*.npz")))
        self.assertFalse(list(cache.glob("*manifest.json")))
        self.assertFalse(list(cache.glob("parts/*.tmp")))
        self.build(resume=True)
        self.assertTrue(r2_u1.verify_cache(self.plan(), cache)["passed"])


if __name__ == "__main__":
    unittest.main()
