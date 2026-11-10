"""R3-S1 validation-sample run on the miniature cohort: scope, determinism, report honesty and no-overwrite behaviour."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from echoproof.data import corruption as c
from echoproof.data import corruption_sample as cs
from echoproof.eda import selection
from echoproof.features.r2_s1 import R2S1Error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_r2_s1_features import build_mini_project, waveform  # noqa: E402


def _ffmpeg_ok():
    try:
        c.find_ffmpeg()
        return True
    except c.CorruptionError:
        return False


@unittest.skipUnless(_ffmpeg_ok(), "ffmpeg with libmp3lame is not available")
class SampleRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.mini = build_mini_project(cls.root)
        cls.out = cls.root / "reports" / "r3"
        cls.report = cs.run(cls.root, cls.mini["shard_dir"], cls.out, per_component=2)
        with open(cls.out / cs.TABLE_NAME, newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_report_passes_and_counts_add_up(self):
        r = self.report
        self.assertTrue(r["passed"], r["failures"])
        self.assertEqual(r["sample"]["recordings"], 8)                                        # 2 each of genuine, melo, vits, xtts
        self.assertEqual(r["sample"]["by_component"], {"genuine": 2, "melo": 2, "vits": 2, "xtts": 2})
        self.assertEqual(r["sample"]["variants_built"], r["sample"]["variants_expected"], 32)
        self.assertEqual(len(self.rows), 32)
        self.assertTrue(all(r["checks"].values()), r["checks"])

    def test_scope_is_selection_role_only_without_chatterbox_or_test(self):
        selection_ids = {a.recording_id for a in self.mini["assignments"] if a.partition == "validation" and a.validation_role == "selection"}
        parents = {row["parent_id"] for row in self.rows}
        self.assertTrue(parents <= selection_ids)
        self.assertFalse(any("chatterbox" in p for p in parents))
        self.assertIn("selection role only", self.report["scope"])
        self.assertEqual(self.report["role"], "selection")

    def test_variants_keep_parents_and_hashes_match_a_direct_call(self):
        row = next(r for r in self.rows if r["condition"] == "noise_10")
        direct = c.corrupt(waveform(self.mini, row["parent_id"]), row["parent_id"], "noise_10")
        self.assertEqual(row["variant_id"], f"{row['parent_id']}@noise_10")
        self.assertEqual(row["pcm_sha256"], direct.pcm_sha256)
        self.assertEqual(row["pass_pcm_sha256_repeat"], "True")

    def test_condition_summaries_describe_the_four_conditions(self):
        conds = self.report["conditions"]
        self.assertEqual(list(conds), list(c.CONDITION_NAMES))
        for name in ("noise_20", "noise_10"):
            self.assertEqual(conds[name]["variants"], 8)
            self.assertAlmostEqual(conds[name]["measured_snr_db"]["mean"], conds[name]["target_snr_db"], delta=0.6)
            self.assertEqual(conds[name]["white"] + conds[name]["pink"], 8)
            self.assertGreater(conds[name]["mean_delta"]["noise_floor_dbfs"], 0)               # noise raises the measured noise floor
        for name, kbps in (("mp3_32", 32), ("mp3_128", 128)):
            self.assertAlmostEqual(conds[name]["effective_kbps"]["mean"], kbps, delta=kbps * 0.15)
            self.assertGreater(conds[name]["correlation"]["min"], 0.9)

    def test_report_records_spec_ffmpeg_and_input_hashes(self):
        r = self.report
        self.assertEqual(r["spec"]["spec_sha256"], c.SPEC_SHA256)
        self.assertTrue(r["spec"]["ffmpeg"]["version"].startswith("ffmpeg version"))
        self.assertEqual(len(r["fixture"]["digest"]), 64)
        self.assertEqual(r["fixture"]["digest"], cs.fixture_digest()["digest"])               # the reproducibility fixture repeats on this machine
        self.assertEqual(r["inputs"]["split_sha256"], self.mini["split_hash"])
        self.assertEqual(len(r["inputs"]["corruption_module_sha256"]), 64)
        self.assertIn("no audio is written", r["scope"])

    def test_no_audio_is_written(self):
        written = {p.suffix for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(written, {".json", ".csv"})

    def test_finished_outputs_are_never_overwritten(self):
        with self.assertRaises(R2S1Error):
            cs.run(self.root, self.mini["shard_dir"], self.out, per_component=2)

    def test_a_failing_variant_is_recorded_and_fails_the_report(self):
        real = c.corrupt

        def flaky(wave, parent_id, condition, **kw):
            if condition == "noise_20" and parent_id.endswith("0001"):
                raise c.CorruptionError("injected")
            return real(wave, parent_id, condition, **kw)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(c, "corrupt", flaky):
            report = cs.run(self.root, self.mini["shard_dir"], Path(d), per_component=2)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["all_variants_built"])
        self.assertTrue(report["failures"] and all(f["error"] == "injected" and f["condition"] == "noise_20" for f in report["failures"]))
        self.assertEqual(report["sample"]["variants_built"], report["sample"]["variants_expected"] - len(report["failures"]))

    def test_sampling_is_deterministic_and_validates_roles(self):
        eda = selection.plan_selection(*self._inputs())
        a, b = cs.draw_sample(eda.items, 3, 42), cs.draw_sample(eda.items, 3, 42)
        self.assertEqual([i["recording_id"] for i in a], [i["recording_id"] for i in b])
        self.assertEqual(len(a), 8)                                                          # the miniature pool has only two per component
        draws = {tuple(i["recording_id"] for i in cs.draw_sample(eda.items, 1, seed)) for seed in range(1, 9)}
        self.assertGreater(len(draws), 1)                                                    # the seed changes which recording is drawn
        with self.assertRaises(R2S1Error):
            cs.draw_sample([dict(eda.items[0], role="calibration")], 1, 42)
        with self.assertRaises(R2S1Error):
            cs.draw_sample(eda.items, 0, 42)

    def test_missing_shard_is_reported_before_any_work(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(R2S1Error):
            cs.run(self.root, Path(d) / "no-shards", Path(d) / "out", per_component=1)

    def _inputs(self):
        from echoproof.data import splits
        from echoproof.data.cohort import read_table
        from echoproof.contracts import Chunk, read_parquet
        return (read_table(self.root / "data" / "manifests" / "cohort_v1.parquet"),
                splits.load_assignments(self.root / "data" / "splits" / "protocol_B_v1.parquet", sealed=False),
                read_parquet(str(self.root / "data" / "processed" / "chunks_v1.parquet"), Chunk))


if __name__ == "__main__":
    unittest.main()
