"""R4-U1a tests: Protocol A data, alignment, recipe guard, training, frozen selection and export, on synthetic embeddings.

No research audio, embedding cache, model download, GPU or network is used. Alignment is tested against the miniature A project of the R4-U0 tests.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from echoproof.artifacts import file_digest  # noqa: E402
from echoproof.training import r4_u1_wavlm_data as D, r4_u1_wavlm_train as T  # noqa: E402
from test_r4_u0_wavlm_cache import A_HASH, World  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def digest_in_lock(path, lock_text):
    """True if the file's SHA-256 is in the GR3 lock. The lock hashed the Windows (CRLF) checkout, so the same text with LF line endings (a Linux
    or Colab checkout) is accepted too: only the line endings may differ, never the content."""
    import hashlib
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return any(hashlib.sha256(variant).hexdigest() in lock_text for variant in (raw, lf, lf.replace(b"\n", b"\r\n")))

RECIPE = ROOT / "configs/r4_u1_wavlm_head_a.json"
COMPONENTS = D.COMPONENTS


def synthetic_rows(seed=0, dim=1536):
    """Separable embeddings with all five components in train and every role; speakers disjoint across train and the roles."""
    rng = np.random.default_rng(seed)
    rows, vectors = [], []

    def add(kind, partition, role, recording, group, component, n, prefix):
        label = int(component != "genuine")
        for k in range(n):
            vec = rng.normal(0, 1, dim)
            vec[:8] += 1.5 if label else -1.5
            vectors.append(vec)
            rows.append({"item_id": f"{prefix}:A:{recording}:{k}", "recording_id": recording, "group_id": group, "component": component,
                         "label": label, "partition": partition, "role": role, "kind": kind, "start_sample": k, "end_sample": k + 1,
                         "source": "a"})

    for s in range(10):
        for j, comp in enumerate(COMPONENTS):
            add("crop", "train", None, f"phonemedf:{comp}:{100 + s}-1-{j}", f"speaker:{100 + s}", comp, 1 + (s + j) % 3, "crop")
    for r_i, role in enumerate(D.ROLES):
        for s in range(4):
            for comp in COMPONENTS:
                add("chunk", "validation", role, f"phonemedf:{comp}:{200 + 10 * r_i + s}-1-0", f"speaker:{200 + 10 * r_i + s}", comp, 2, "chunk")
    return rows, np.array(vectors, dtype=np.float32)


def synthetic_data(seed=0):
    rows, matrix = synthetic_rows(seed)
    return D.build_data_a(rows, matrix, "mean_std", {"source": "synthetic"})


class DataRules(unittest.TestCase):
    def test_builds_roles_weights_and_five_components(self):
        data = synthetic_data()
        self.assertEqual(set(data.roles), set(D.ROLES))
        self.assertEqual(set(data.train.components), set(COMPONENTS))
        for split in data.roles.values():
            self.assertEqual(set(split.components), set(COMPONENTS))
        self.assertAlmostEqual(float(data.train.recording_weights.mean()), 1.0, places=5)
        self.assertEqual(data.dim, 1536)

    def test_rejects_wrong_pooling_test_partition_and_bad_rows(self):
        rows, matrix = synthetic_rows()
        with self.assertRaisesRegex(D.AlignmentError, "mean_std pooling only"):
            D.build_data_a(rows, matrix, "mean", {})
        bad = copy.deepcopy(rows)
        bad[0]["partition"] = "test"
        with self.assertRaisesRegex(D.AlignmentError, "test partition"):
            D.build_data_a(bad, matrix, "mean_std", {})
        bad = copy.deepcopy(rows)
        bad[0]["label"] = 1 - bad[0]["label"]
        with self.assertRaisesRegex(D.AlignmentError, "labels"):
            D.build_data_a(bad, matrix, "mean_std", {})
        bad = copy.deepcopy(rows)
        bad[1]["item_id"] = bad[0]["item_id"]
        with self.assertRaisesRegex(D.AlignmentError, "duplicate"):
            D.build_data_a(bad, matrix, "mean_std", {})
        nonfinite = matrix.copy()
        nonfinite[3, 3] = np.nan
        with self.assertRaisesRegex(D.AlignmentError, "non-finite"):
            D.build_data_a(rows, nonfinite, "mean_std", {})
        bad = copy.deepcopy(rows)
        bad[0]["component"] = "other"
        with self.assertRaisesRegex(D.AlignmentError, "unknown component"):
            D.build_data_a(bad, matrix, "mean_std", {})

    def test_requires_every_component_in_train_and_each_role(self):
        rows, matrix = synthetic_rows()
        keep = [i for i, r in enumerate(rows) if not (r["partition"] == "train" and r["component"] == "chatterbox")]
        with self.assertRaisesRegex(D.AlignmentError, "all five components"):
            D.build_data_a([rows[i] for i in keep], matrix[keep], "mean_std", {})
        keep = [i for i, r in enumerate(rows) if not (r["role"] == "operating" and r["component"] == "xtts")]
        with self.assertRaisesRegex(D.AlignmentError, "all five components"):
            D.build_data_a([rows[i] for i in keep], matrix[keep], "mean_std", {})

    def test_speaker_overlap_between_partitions_is_rejected(self):
        rows, matrix = synthetic_rows()
        rows = copy.deepcopy(rows)
        victim = next(r for r in rows if r["role"] == "calibration")
        victim["group_id"] = next(r for r in rows if r["partition"] == "train")["group_id"]
        with self.assertRaises(D.AlignmentError):
            D.build_data_a(rows, matrix, "mean_std", {})


class RecipeGuard(unittest.TestCase):
    def test_shipped_recipe_equals_the_b_selected_configuration_and_records_the_team_agreement(self):
        recipe = T.load_recipe(RECIPE, root=ROOT)
        b = json.loads((ROOT / T.B_RECIPE).read_text(encoding="utf-8"))
        self.assertEqual(recipe["train"], b["train"])
        self.assertEqual(recipe["model"], b["model"])
        self.assertEqual(recipe["protocol"], "A")
        self.assertEqual(recipe["variants"], ["mean_std"])
        self.assertEqual(T.load_recipe(RECIPE, root=ROOT, require_agreed=True)["status"], "team_agreed_user_reported_2026-11-15")
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".r4-u1-draft-") as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / T.B_RECIPE).write_bytes((ROOT / T.B_RECIPE).read_bytes())
            draft = json.loads(RECIPE.read_text(encoding="utf-8"))
            draft["status"] = "draft_pending_team_agreement"
            (root / "configs/a.json").write_text(json.dumps(draft), encoding="utf-8")
            with self.assertRaisesRegex(T.R4U1Error, "reported agreement"):
                T.load_recipe(root / "configs/a.json", root=root, require_agreed=True)

    def test_any_departure_from_b_is_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".r4-u1-recipe-") as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / T.B_RECIPE).write_bytes((ROOT / T.B_RECIPE).read_bytes())
            base = json.loads(RECIPE.read_text(encoding="utf-8"))
            for mutate, pattern in ((lambda r: r["train"].update(lr=0.002), "departs from the B-selected"),
                                    (lambda r: r["model"].update(hidden=64), "departs from the B-selected"),
                                    (lambda r: r.update(protocol="B"), "not an R4-U1a"),
                                    (lambda r: r.update(variants=["mean", "mean_std"]), "mean_std only"),
                                    (lambda r: r["derived_from"].update(sha256="0" * 64), "differs from the one")):
                recipe = copy.deepcopy(base)
                mutate(recipe)
                path = root / "configs" / "a.json"
                path.write_text(json.dumps(recipe), encoding="utf-8")
                with self.subTest(pattern=pattern), self.assertRaisesRegex(T.R4U1Error, pattern):
                    T.load_recipe(path, root=root)
            recipe = copy.deepcopy(base)
            recipe["status"] = "team_agreed_user_reported_2026-11-15"
            path.write_text(json.dumps(recipe), encoding="utf-8")
            self.assertEqual(T.load_recipe(path, root=root, require_agreed=True)["status"], recipe["status"])


def small_recipe(epochs=4):
    recipe = T.load_recipe(RECIPE, root=ROOT)
    recipe = copy.deepcopy(recipe)
    recipe["train"].update(max_epochs=epochs, batch_size=16, lr=0.02)
    return recipe


class Training(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT, prefix=".r4-u1-train-")
        self.out = Path(self.tmp.name)
        self.data = synthetic_data()
        self.recipe = small_recipe()

    def tearDown(self):
        self.tmp.cleanup()

    def run_train(self, name="run_mean_std", ledger=True):
        return T.train(self.data, self.recipe, self.out / name, owner="Tester", split_sha256="f" * 64,
                       ledger_path=self.out / "ledger.csv" if ledger else None)

    def test_trains_keeps_best_epoch_and_writes_ledger_row_for_protocol_a(self):
        summary = self.run_train()
        self.assertEqual(summary["epochs_run"], 4)
        self.assertEqual(summary["stop_reason"], "max_epochs")
        history = [json.loads(l) for l in (self.out / "run_mean_std/history.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(history), 4)
        best = max(history, key=lambda r: (r["val"]["macro_f1"], -r["epoch"]))
        self.assertEqual(summary["best_epoch"], best["epoch"])
        self.assertTrue(all("genuine_vs_chatterbox" in r["panels"] and "genuine_vs_xtts" in r["panels"] for r in history))
        lines = (self.out / "ledger.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn(",A,", lines[1])
        self.assertIn("wavlm_head_mean_std", lines[1])

    def test_refuses_nonempty_run_dir_missing_owner_and_other_pooling(self):
        self.run_train()
        with self.assertRaisesRegex(T.R4U1Error, "non-empty"):
            self.run_train()
        with self.assertRaisesRegex(T.R4U1Error, "owner"):
            T.train(self.data, self.recipe, self.out / "x", owner="", split_sha256="f" * 64)

    def test_selection_is_frozen_before_roles_and_exports_have_b_columns(self):
        import pyarrow.parquet as pq
        summary = self.run_train()
        evidence = T.finish(self.out, self.data, summary, self.recipe, split_sha256="f" * 64)
        selection = json.loads((self.out / "selection.json").read_text(encoding="utf-8"))
        self.assertEqual(selection["protocol"], "A")
        self.assertEqual(selection["selected"], "mean_std")
        self.assertEqual(selection["selected_epoch"], summary["best_epoch"])
        self.assertEqual(evidence["counts"]["selection"]["chunks"], len(self.data.roles["selection"]))
        for role in D.ROLES:
            chunks = pq.read_table(self.out / f"predictions_chunks_{role}.parquet").to_pylist()
            recordings = pq.read_table(self.out / f"predictions_recordings_{role}.parquet").to_pylist()
            self.assertEqual(set(chunks[0]), {"chunk_id", "recording_id", "group_id", "label", "component", "validation_role", "tts_probability"})
            self.assertEqual(set(recordings[0]), {"recording_id", "group_id", "label", "component", "validation_role", "raw_score", "chunk_count"})
            self.assertEqual({r["validation_role"] for r in chunks}, {role})
            self.assertEqual(len(chunks), len(self.data.roles[role]))
            self.assertTrue(all(0.0 <= r["tts_probability"] <= 1.0 for r in chunks))
        self.assertEqual(set(evidence["selection_panels"]), {"genuine_vs_" + c for c in COMPONENTS if c != "genuine"})

    def test_existing_selection_is_never_overwritten_and_changed_checkpoint_is_refused(self):
        summary = self.run_train()
        T.finish(self.out, self.data, summary, self.recipe, split_sha256="f" * 64)
        with self.assertRaisesRegex(T.R4U1Error, "refusing to overwrite"):
            T.finish(self.out, self.data, summary, self.recipe, split_sha256="e" * 64)
        other = self.out / "other"
        other.mkdir()
        tampered = dict(summary, best_checkpoint_sha256="0" * 64)
        with self.assertRaisesRegex(T.R4U1Error, "changed since training"):
            T.finish(other, self.data, tampered, self.recipe, split_sha256="f" * 64)
        unfinished = dict(summary, stop_reason="interrupted")
        with self.assertRaisesRegex(T.R4U1Error, "inconsistent or unfinished"):
            T.finish(other, self.data, unfinished, self.recipe, split_sha256="f" * 64)

    def test_same_seed_gives_the_same_checkpoint(self):
        a = self.run_train("run_a", ledger=False)
        b = self.run_train("run_b", ledger=False)
        self.assertEqual(a["best_epoch"], b["best_epoch"])
        self.assertEqual(a["best_macro_f1"], b["best_macro_f1"])


class Alignment(unittest.TestCase):
    """Alignment against the miniature A project: right rows pass, tampered rows and wrong counts fail."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT, prefix=".r4-u1-align-")
        self.world = World(Path(self.tmp.name))
        self.root = self.world.root
        # frozen split hash for the mini project is "a" * 64, as written by the R4-U0 fixture
        self.rows, self.matrix = self.rows_and_vectors()

    def tearDown(self):
        self.tmp.cleanup()

    def rows_and_vectors(self):
        w = self.world
        cohort = {r["recording_id"]: r for r in w.rows}
        rows = []
        roles = {a.recording_id: a for a in w.a_assign}
        for c in w.plan_a.crop_bank:
            rows.append({"item_id": c["crop_id"], "recording_id": c["recording_id"], "group_id": c["group_id"], "component": c["component"],
                         "label": c["label"], "partition": "train", "role": None, "kind": "crop", "start_sample": c["start_sample"],
                         "end_sample": c["end_sample"], "source": "a"})
        for ch in w.a_chunks:
            if ch.partition == "validation":
                rows.append({"item_id": ch.chunk_id, "recording_id": ch.recording_id, "group_id": ch.group_id,
                             "component": cohort[ch.recording_id]["component"], "label": cohort[ch.recording_id]["label"], "partition": "validation",
                             "role": roles[ch.recording_id].validation_role, "kind": "chunk", "start_sample": ch.start_sample,
                             "end_sample": ch.end_sample, "source": "a"})
        rng = np.random.default_rng(0)
        return rows, rng.normal(size=(len(rows), 1536)).astype(np.float32)

    def data(self, rows=None, matrix=None):
        return D.build_data_a(rows or self.rows, matrix if matrix is not None else self.matrix, "mean_std", {"source": "fixture"})

    def test_correct_rows_pass_and_report_counts(self):
        report = D.check_alignment_a(self.root, self.data(), expected_counts=None)
        self.assertTrue(report["passed"])
        self.assertFalse(report["model_scores_used"])
        self.assertEqual(report["counts"]["train_crops"], len(self.world.plan_a.crop_bank))

    def test_counts_must_equal_the_r4_s1_evidence(self):
        with self.assertRaisesRegex(D.AlignmentError, "differ from the R4-S1 evidence"):
            D.check_alignment_a(self.root, self.data())

    def test_missing_unknown_and_changed_rows_fail(self):
        drop = [r for i, r in enumerate(self.rows) if not (r["kind"] == "chunk" and i == next(j for j, x in enumerate(self.rows) if x["kind"] == "chunk"))]
        keep = [i for i, r in enumerate(self.rows) if r in drop]
        with self.assertRaisesRegex(D.AlignmentError, "coverage"):
            D.check_alignment_a(self.root, self.data([self.rows[i] for i in keep], self.matrix[keep]), expected_counts=None)
        wrong = copy.deepcopy(self.rows)
        crop = next(r for r in wrong if r["kind"] == "crop")
        crop["item_id"] = "crop:A:unknown:0"
        with self.assertRaisesRegex(D.AlignmentError, "not in the A crop bank"):
            D.check_alignment_a(self.root, self.data(wrong), expected_counts=None)
        wrong = copy.deepcopy(self.rows)
        chunk = next(r for r in wrong if r["kind"] == "chunk" and r["role"] == "selection")
        chunk["role"] = "operating"
        with self.assertRaisesRegex(D.AlignmentError, "validation role|speaker appears"):      # caught at build time (speaker overlap) or by alignment
            D.check_alignment_a(self.root, self.data(wrong), expected_counts=None)

    def test_changed_frozen_split_hash_is_refused(self):
        path = self.root / "reports/r1/split_report.json"
        path.write_text(json.dumps({"protocols": {"A": {"split_sha256": "c" * 64}}}), encoding="utf-8")
        with self.assertRaisesRegex(D.AlignmentError, "frozen value"):
            D.check_alignment_a(self.root, self.data(), expected_counts=None)

    def test_parity_rows_cover_all_five_components_and_are_deterministic(self):
        data = self.data()
        rows = D.parity_rows(self.root, data, 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual([r["item_id"] for r in rows], [r["item_id"] for r in D.parity_rows(self.root, data, 10)])
        self.assertEqual({r["component"] for r in rows}, set(COMPONENTS))
        for r in rows:
            self.assertEqual(data.roles["selection"].item_ids[r["position"]], r["item_id"])
            self.assertRegex(r["shard"], r"shard_\d+\.tar")


class FrozenSources(unittest.TestCase):
    def test_b_sources_are_untouched(self):
        lock = json.dumps(json.loads((ROOT / "locks/B_lock_gr3.json").read_text(encoding="utf-8")))
        for rel in ("src/echoproof/training/r3_u1_data.py", "src/echoproof/training/r3_u1_train.py", "src/echoproof/training/r3_u1_parity.py",
                    "src/echoproof/models/r3_u1_bundle.py", "src/echoproof/models/wavlm_head.py"):
            self.assertTrue(digest_in_lock(ROOT / rel, lock), f"{rel} differs from the hash recorded in the GR3 lock (beyond line endings)")


if __name__ == "__main__":
    unittest.main()
