"""Contract boundaries and cross-record integrity using fabricated metadata."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from echoproof.contracts import (BundleSpecification, Chunk, ChunkScore, ContractError, DecisionRule, FeatureBatch,
    InferenceResult, PredictionRow, Recording, SplitAssignment, read_parquet,
    validate_chunk_scores, validate_relationships, write_parquet)

ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.record = Recording.from_json((ROOT / "tests/fixtures/contracts/recording.json").read_text(encoding="utf-8"))
        self.split = SplitAssignment(1, "protocol:A", self.record.recording_id, "group:g", "train", None, "known", "v1", HASH)
        self.chunk = Chunk(1, "chunk:1", self.record.recording_id, "group:g", "protocol:A", "train", 0, 16000, 16000, 0.0, 1.0, 16000, 64000, HASH)

    def test_json_round_trip_and_null_identities(self):
        restored = Recording.from_json(self.record.to_json())
        self.assertEqual(restored, self.record)
        self.assertIsNone(restored.source_speaker_id)
        with self.assertRaises(ContractError):
            restored.validate_supervised()
        replace(restored, label=0, label_verified=True, method="verified-human", method_verified=True).validate_supervised()

    def test_identity_paths_labels_and_schema_are_validated(self):
        for change in ({"original_path": "../escape.wav"}, {"original_path": "C:\\x.wav"}, {"label": True}, {"source_speaker_id": "unknown"}, {"schema_version": 2}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                replace(self.record, **change)
        value = self.record.to_dict()
        value["unexpected"] = True
        with self.assertRaises(ContractError):
            Recording.from_dict(value)

    def test_relationships_and_split_leakage(self):
        validate_relationships([self.record], [self.split], [self.chunk])
        for records, splits, chunks in (([self.record, self.record], [self.split], []), ([self.record], [self.split, self.split], []), ([self.record], [], [self.chunk]), ([self.record], [self.split], [replace(self.chunk, group_id="group:other")])):
            with self.subTest(records=records, splits=splits, chunks=chunks), self.assertRaises(ContractError):
                validate_relationships(records, splits, chunks)
        other = replace(self.record, recording_id="fixture:other")
        other_split = replace(self.split, recording_id=other.recording_id, partition="test")
        with self.assertRaises(ContractError):
            validate_relationships([self.record, other], [self.split, other_split], [])
        with self.assertRaises(ContractError):
            validate_relationships([self.record], [self.split], [], supervised=True)

    def test_chunk_offsets_and_lengths(self):
        for change in ({"end_seconds": 2.0}, {"input_samples": 100}, {"end_sample": 20000}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                replace(self.chunk, **change)

    def test_prediction_rules_scores_and_failure(self):
        rule = DecisionRule(1, "all_negative", None)
        row = PredictionRow(1, "run:1", "bundle:1", "protocol:A", self.record.recording_id, 1.0, None, rule, 0, "success", 1.0, 1)
        self.assertEqual(PredictionRow.from_json(row.to_json()), row)
        bundle = BundleSpecification(1, "bundle:1", "model-v1", HASH, rule, "genuine", "tts")
        self.assertEqual(BundleSpecification.from_json(bundle.to_json()), bundle)
        self.assertNotIn("Infinity", row.to_json())
        for change in ({"raw_score": float("nan")}, {"prediction": 1}, {"status": "failed"}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                replace(row, **change)
        with self.assertRaises(ContractError):
            DecisionRule(1, "threshold", float("inf"))
        failure = replace(row, raw_score=None, decision_rule=None, prediction=None, status="failed", chunk_count=0)
        self.assertEqual(PredictionRow.from_json(failure.to_json()), failure)

    def test_inference_failure_and_coverage(self):
        ok = InferenceResult(1, "success", "likely tts", 0.8, "raw_tts_probability", [], "v1", 2.0, 1.0, {"total": 0.1}, None)
        self.assertEqual(InferenceResult.from_json(ok.to_json()), ok)
        for change in ({"covered_duration_seconds": 3.0}, {"status": "invalid_input", "failure_reason": "bad audio"}, {"score": float("inf")}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                replace(ok, **change)
        failure = InferenceResult(1, "invalid_input", None, None, None, [], None, None, None, {}, "bad audio")
        self.assertEqual(InferenceResult.from_json(failure.to_json()), failure)

    def test_score_order_is_explicit(self):
        scores = [ChunkScore("chunk:1", 0.2), ChunkScore("chunk:2", 0.9)]
        validate_chunk_scores(["chunk:1", "chunk:2"], scores)
        with self.assertRaises(ContractError):
            validate_chunk_scores(["chunk:2", "chunk:1"], scores)
        with self.assertRaises(ContractError):
            ChunkScore("chunk:1", float("nan"))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy unavailable")
    def test_batch_lengths_masks_and_valid_finiteness(self):
        import numpy as np
        data = np.array([[[1.0], [2.0]], [[3.0], [float("nan")]]])
        batch = FeatureBatch(data, ["chunk:1", "chunk:2"], ["record:1", "record:2"], np.array([0, 1]), np.array([2, 1]), np.array([[True, True], [True, False]]), HASH)
        batch.validate()
        for change in ({"lengths": np.array([2, 2])}, {"mask": np.array([[True, True], [False, True]])}, {"recording_ids": ["record:1"]}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                replace(batch, **change).validate()
        with self.assertRaises(ContractError):
            replace(batch, values=np.full((2, 2, 1), np.nan)).validate()

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow unavailable")
    def test_parquet_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "metadata.parquet")
            rule = DecisionRule(1, "all_negative", None)
            row = PredictionRow(1, "run:1", "bundle:1", "protocol:A", self.record.recording_id, 0.7, None, rule, 0, "success", 1.0, 1)
            result = InferenceResult(1, "invalid_input", None, None, None, [], None, None, None, {}, "bad audio")
            for record in (self.record, self.split, self.chunk, rule, row, result):
                write_parquet([record], path)
                self.assertEqual(read_parquet(path, type(record)), [record])


if __name__ == "__main__":
    unittest.main()
