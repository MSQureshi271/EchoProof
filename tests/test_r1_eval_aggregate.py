"""Tests for R1-N1 recording-level score aggregation."""
import math
import unittest

from echoproof.contracts import (
    Chunk,
    ChunkScore,
    ContractError,
    Recording,
    SplitAssignment,
)
from echoproof.evaluation.aggregate import (
    RecordingScores,
    aggregate_from_contracts,
    aggregate_recordings,
)

HASH = "a" * 64


class AggregateTests(unittest.TestCase):
    def test_mean_correctness_and_order_preservation(self):
        # A: 0.2, 0.4, 0.6 -> 0.4
        # B: 0.8, 0.9 -> 0.85
        chunk_ids = ["chunk:1", "chunk:2", "chunk:3", "chunk:4", "chunk:5"]
        recording_ids = ["rec:A", "rec:B", "rec:A", "rec:B", "rec:A"]
        labels = [1, 0, 1, 0, 1]
        scores = [0.2, 0.8, 0.4, 0.9, 0.6]

        res = aggregate_recordings(chunk_ids, recording_ids, labels, scores)
        self.assertEqual(res.recording_ids, ("rec:A", "rec:B"))
        self.assertEqual(res.labels, (1, 0))
        self.assertEqual(res.chunk_counts, (3, 2))
        self.assertAlmostEqual(res.scores[0], 0.4, places=9)
        self.assertAlmostEqual(res.scores[1], 0.85, places=9)

        # Reversed arrival order preserves first-seen order
        chunk_ids_b = ["chunk:2", "chunk:4", "chunk:1", "chunk:3", "chunk:5"]
        recording_ids_b = ["rec:B", "rec:B", "rec:A", "rec:A", "rec:A"]
        labels_b = [0, 0, 1, 1, 1]
        scores_b = [0.8, 0.9, 0.2, 0.4, 0.6]

        res_b = aggregate_recordings(chunk_ids_b, recording_ids_b, labels_b, scores_b)
        self.assertEqual(res_b.recording_ids, ("rec:B", "rec:A"))
        self.assertEqual(res_b.labels, (0, 1))
        self.assertAlmostEqual(res_b.scores[0], 0.85, places=9)
        self.assertAlmostEqual(res_b.scores[1], 0.4, places=9)

    def test_expected_recording_ids(self):
        chunk_ids = ["chunk:1", "chunk:2"]
        recording_ids = ["rec:A", "rec:B"]
        labels = [1, 0]
        scores = [0.3, 0.7]

        # Exact match succeeds
        res = aggregate_recordings(chunk_ids, recording_ids, labels, scores, expected_recording_ids=["rec:B", "rec:A"])
        self.assertEqual(res.recording_ids, ("rec:A", "rec:B"))

        # Missing expected ID
        with self.assertRaises(ContractError):
            aggregate_recordings(chunk_ids, recording_ids, labels, scores, expected_recording_ids=["rec:A", "rec:B", "rec:C"])

        # Unexpected ID
        with self.assertRaises(ContractError):
            aggregate_recordings(chunk_ids, recording_ids, labels, scores, expected_recording_ids=["rec:A"])

    def test_length_mismatch_and_empty_input(self):
        with self.assertRaises(ContractError):
            aggregate_recordings([], [], [], [])

        with self.assertRaises(ContractError):
            aggregate_recordings(["chunk:1"], ["rec:A", "rec:B"], [1], [0.5])

    def test_duplicate_chunk_id(self):
        chunk_ids = ["chunk:1", "chunk:1"]
        recording_ids = ["rec:A", "rec:A"]
        labels = [1, 1]
        scores = [0.2, 0.4]
        with self.assertRaises(ContractError):
            aggregate_recordings(chunk_ids, recording_ids, labels, scores)

    def test_conflicting_labels_within_recording(self):
        chunk_ids = ["chunk:1", "chunk:2"]
        recording_ids = ["rec:A", "rec:A"]
        labels = [1, 0]
        scores = [0.2, 0.4]
        with self.assertRaises(ContractError):
            aggregate_recordings(chunk_ids, recording_ids, labels, scores)

    def test_invalid_types_and_values(self):
        valid_c = ["chunk:1"]
        valid_r = ["rec:1"]
        valid_l = [1]
        valid_s = [0.5]

        # Bools rejected
        with self.assertRaises(ContractError):
            aggregate_recordings(valid_c, valid_r, [True], valid_s)
        with self.assertRaises(ContractError):
            aggregate_recordings(valid_c, valid_r, valid_l, [True])

        # Non-finite scores
        for nf in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=nf), self.assertRaises(ContractError):
                aggregate_recordings(valid_c, valid_r, valid_l, [nf])

        # Out of [0, 1] bounds
        for oob in (-0.01, 1.01, 2.0, -10.0):
            with self.subTest(score=oob), self.assertRaises(ContractError):
                aggregate_recordings(valid_c, valid_r, valid_l, [oob])

        # Invalid labels
        for bad_l in (-1, 2, 99, "1", None):
            with self.subTest(label=bad_l), self.assertRaises(ContractError):
                aggregate_recordings(valid_c, valid_r, [bad_l], valid_s)

        # Un-namespaced IDs
        with self.assertRaises(ContractError):
            aggregate_recordings(["unnamespaced"], valid_r, valid_l, valid_s)
        with self.assertRaises(ContractError):
            aggregate_recordings(valid_c, ["unnamespaced"], valid_l, valid_s)
        with self.assertRaises(ContractError):
            aggregate_recordings(valid_c, ["unknown:1"], valid_l, valid_s)

    def test_subset_method(self):
        rec_scores = RecordingScores(
            recording_ids=("rec:1", "rec:2", "rec:3"),
            labels=(1, 0, 1),
            scores=(0.9, 0.1, 0.8),
            chunk_counts=(2, 1, 3),
            group_ids=("group:g1", "group:g2", "group:g1"),
            partitions=("train", "validation", "validation"),
            validation_roles=(None, "selection", "operating"),
        )

        sub_val = rec_scores.subset(partition="validation")
        self.assertEqual(sub_val.recording_ids, ("rec:2", "rec:3"))
        self.assertEqual(sub_val.labels, (0, 1))

        sub_role = rec_scores.subset(validation_role="operating")
        self.assertEqual(sub_role.recording_ids, ("rec:3",))
        self.assertEqual(sub_role.scores, (0.8,))

        sub_none = rec_scores.subset(partition="test")
        self.assertEqual(sub_none.recording_ids, ())

        sub_all = rec_scores.subset()
        self.assertEqual(sub_all.recording_ids, rec_scores.recording_ids)

    def test_aggregate_from_contracts(self):
        rec_1 = Recording(
            1, "rec:1", "audio/1.wav", "phonemedf", "v1",
            label=1, label_verified=True, method="xtts", method_verified=True,
            generator_family="xtts", generator_version="v2", language="en",
            source_utterance_id=None, source_speaker_id=None, target_voice_id=None,
            reference_id=None, transcript_hash=None, duration_seconds=4.0,
            sample_rate=16000, channels=1, codec="wav", sha256=HASH,
            quality_flags=[], retained=True, exclusion_reason=None, license_ref=None,
        )
        rec_2 = Recording(
            1, "rec:2", "audio/2.wav", "phonemedf", "v1",
            label=0, label_verified=True, method="genuine", method_verified=True,
            generator_family=None, generator_version=None, language="en",
            source_utterance_id=None, source_speaker_id=None, target_voice_id=None,
            reference_id=None, transcript_hash=None, duration_seconds=8.0,
            sample_rate=16000, channels=1, codec="wav", sha256=HASH,
            quality_flags=[], retained=True, exclusion_reason=None, license_ref=None,
        )

        split_1 = SplitAssignment(1, "protocol:A", "rec:1", "group:g1", "train", None, "known", "v1", HASH)
        split_2 = SplitAssignment(1, "protocol:A", "rec:2", "group:g2", "validation", "operating", "known", "v1", HASH)

        chunk_1a = Chunk(1, "chunk:1a", "rec:1", "group:g1", "protocol:A", "train", 0, 16000, 16000, 0.0, 1.0, 16000, 64000, HASH)
        chunk_2a = Chunk(1, "chunk:2a", "rec:2", "group:g2", "protocol:A", "validation", 0, 16000, 16000, 0.0, 1.0, 16000, 64000, HASH)
        chunk_2b = Chunk(1, "chunk:2b", "rec:2", "group:g2", "protocol:A", "validation", 16000, 32000, 16000, 1.0, 2.0, 16000, 64000, HASH)

        scores = [
            ChunkScore("chunk:1a", 0.7),
            ChunkScore("chunk:2a", 0.1),
            ChunkScore("chunk:2b", 0.3),
        ]

        res = aggregate_from_contracts(
            [rec_1, rec_2],
            [chunk_1a, chunk_2a, chunk_2b],
            scores,
            [split_1, split_2],
            protocol_id="protocol:A",
        )

        self.assertEqual(res.recording_ids, ("rec:1", "rec:2"))
        self.assertEqual(res.labels, (1, 0))
        self.assertAlmostEqual(res.scores[0], 0.7, places=9)
        self.assertAlmostEqual(res.scores[1], 0.2, places=9)
        self.assertEqual(res.group_ids, ("group:g1", "group:g2"))
        self.assertEqual(res.partitions, ("train", "validation"))
        self.assertEqual(res.validation_roles, (None, "operating"))

    def test_aggregate_from_contracts_wrong_score_order(self):
        rec_1 = Recording(
            1, "rec:1", "audio/1.wav", "phonemedf", "v1",
            label=1, label_verified=True, method="xtts", method_verified=True,
            generator_family=None, generator_version=None, language=None,
            source_utterance_id=None, source_speaker_id=None, target_voice_id=None,
            reference_id=None, transcript_hash=None, duration_seconds=4.0,
            sample_rate=16000, channels=1, codec="wav", sha256=HASH,
            quality_flags=[], retained=True, exclusion_reason=None, license_ref=None,
        )
        split_1 = SplitAssignment(1, "protocol:A", "rec:1", "group:g1", "train", None, "known", "v1", HASH)
        chunk_1 = Chunk(1, "chunk:1", "rec:1", "group:g1", "protocol:A", "train", 0, 16000, 16000, 0.0, 1.0, 16000, 64000, HASH)
        chunk_2 = Chunk(1, "chunk:2", "rec:1", "group:g1", "protocol:A", "train", 16000, 32000, 16000, 1.0, 2.0, 16000, 64000, HASH)

        # Swapped score chunk IDs
        scores_swapped = [ChunkScore("chunk:2", 0.5), ChunkScore("chunk:1", 0.5)]
        with self.assertRaises(ContractError):
            aggregate_from_contracts([rec_1], [chunk_1, chunk_2], scores_swapped, [split_1], protocol_id="protocol:A")

    def test_aggregate_from_contracts_label_none_rejected(self):
        rec_unlabeled = Recording(
            1, "rec:unlabeled", "audio/1.wav", "phonemedf", "v1",
            label=None, label_verified=False, method=None, method_verified=False,
            generator_family=None, generator_version=None, language=None,
            source_utterance_id=None, source_speaker_id=None, target_voice_id=None,
            reference_id=None, transcript_hash=None, duration_seconds=4.0,
            sample_rate=16000, channels=1, codec="wav", sha256=HASH,
            quality_flags=[], retained=True, exclusion_reason=None, license_ref=None,
        )
        split = SplitAssignment(1, "protocol:A", "rec:unlabeled", "group:g1", "train", None, "known", "v1", HASH)
        chunk = Chunk(1, "chunk:1", "rec:unlabeled", "group:g1", "protocol:A", "train", 0, 16000, 16000, 0.0, 1.0, 16000, 64000, HASH)
        scores = [ChunkScore("chunk:1", 0.5)]

        with self.assertRaises(ContractError):
            aggregate_from_contracts([rec_unlabeled], [chunk], scores, [split], protocol_id="protocol:A")


if __name__ == "__main__":
    unittest.main()

