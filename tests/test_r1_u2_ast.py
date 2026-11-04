import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from echoproof.data import chunks as chunk_mod, cohort, splits
from echoproof.gpu_budget import DEFAULT_PLAN, estimate_budget, workload_from_manifests
from echoproof.models.ast_detector import ASTDetector, ASTLoadError, HEAD_TENSORS, audit_loading, checkpoint_shapes, shape_audit
from echoproof.contracts import write_parquet

try:
    from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

RATE = 16000


class LoadingAudit(unittest.TestCase):
    def test_only_the_task_head_may_be_new_or_reshaped(self):
        ok = audit_loading({"missing": [], "unexpected": [], "mismatched": list(HEAD_TENSORS)})
        self.assertTrue(ok["ok"])
        self.assertTrue(audit_loading({"missing": ["classifier.dense.bias"], "unexpected": [], "mismatched": []})["ok"])
        bad = audit_loading({"missing": ["audio_spectrogram_transformer.encoder.layer.3.attention.attention.query.weight"],
                             "unexpected": [], "mismatched": list(HEAD_TENSORS)})
        self.assertFalse(bad["ok"])
        self.assertEqual(len(bad["non_head_new_or_reshaped"]), 1)
        self.assertFalse(audit_loading({"missing": [], "unexpected": ["something.else"], "mismatched": []})["ok"])
        self.assertFalse(audit_loading({"missing": [], "unexpected": [], "mismatched": ["audio_spectrogram_transformer.embeddings.position_embeddings"]})["ok"])

    def test_missing_safetensors_checkpoint_is_an_error_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ASTLoadError):
                checkpoint_shapes(d)


class StubModel:
    """Just enough of a model for shape_audit: a config and a state dict."""

    def __init__(self, shapes, hidden=8, labels=2):
        from types import SimpleNamespace
        self.config = SimpleNamespace(hidden_size=hidden, num_labels=labels)
        self._state = {f"t{i}": torch.zeros(*s) for i, s in enumerate(shapes)}

    def state_dict(self):
        return self._state


class ShapeAudit(unittest.TestCase):
    # The audit must not depend on parameter names, because transformers renames them between versions.
    BODY = [(16, 8), (16,), (8, 8), (8, 8), (8, 8)]

    def test_only_the_head_may_differ(self):
        model = StubModel(self.BODY + [(2, 8), (2,)])
        self.assertTrue(shape_audit(self.BODY + [(527, 8), (527,)], model)["ok"])        # the AudioSet head was replaced
        self.assertTrue(shape_audit(self.BODY + [(2, 8), (2,)], model)["ok"])            # checkpoint already had 2 labels
        report = shape_audit(self.BODY + [(527, 8), (527,)], model)
        self.assertEqual(report["model_only_shapes"], {"(2, 8)": 1, "(2,)": 1})
        self.assertEqual(report["checkpoint_only_shapes"], {"(527, 8)": 1, "(527,)": 1})

    def test_any_other_difference_is_rejected(self):
        model = StubModel(self.BODY + [(2, 8), (2,)])
        self.assertFalse(shape_audit(self.BODY[:-1] + [(527, 8), (527,)], model)["ok"])                    # a backbone tensor is missing
        self.assertFalse(shape_audit(self.BODY + [(527, 8), (527,), (4, 4)], model)["ok"])                 # extra checkpoint tensor
        self.assertFalse(shape_audit(self.BODY + [(527, 8), (300,)], model)["ok"])                         # head halves disagree
        self.assertFalse(shape_audit(self.BODY, StubModel(self.BODY + [(2, 8), (2,), (9, 9)]))["ok"])      # model has a new non-head shape


def tiny_snapshot(directory, num_labels=7):
    config = ASTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64, patch_size=16,
                       frequency_stride=10, time_stride=10, max_length=64, num_mel_bins=16, num_labels=num_labels)
    torch.manual_seed(0)
    ASTForAudioClassification(config).save_pretrained(directory)


def tiny_feature_extractor():
    return ASTFeatureExtractor(feature_size=1, sampling_rate=RATE, num_mel_bins=16, max_length=64, padding_value=0.0, do_normalize=True,
                               mean=-4.27, std=4.57, return_attention_mask=False)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyAST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tiny_snapshot(cls.tmp.name)
        cls.extractor = tiny_feature_extractor()
        try:
            cls.extractor([np.zeros(RATE, dtype=np.float32)], sampling_rate=RATE, return_tensors="pt")
            cls.features_ok = True
        except Exception:  # missing audio backend for the filterbank: structure tests still run, forward tests skip
            cls.features_ok = False

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fresh(self):
        return ASTDetector.from_dir(self.tmp.name, seed=1, feature_extractor=self.extractor)

    def needs_features(self):
        if not self.features_ok:
            self.skipTest("AST feature extractor backend unavailable")

    def test_two_class_head_is_the_only_new_or_reshaped_tensor(self):
        det = self.fresh()
        self.assertTrue(det.loading["ok"], det.loading)
        self.assertEqual(det.loading["shapes"]["model_only_shapes"], {"(2, 32)": 1, "(2,)": 1})
        self.assertEqual(det.loading["shapes"]["checkpoint_only_shapes"], {"(7, 32)": 1, "(7,)": 1})
        self.assertEqual(det.loading["non_head_new_or_reshaped"], [])
        self.assertEqual(det.loading["unexpected"], [])
        if det.loading["library_info_available"]:        # the library's own report, when this transformers version provides one
            self.assertEqual(det.loading["mismatched"], sorted(HEAD_TENSORS))
            self.assertEqual(det.loading["missing"], [])
        self.assertEqual(det.model.classifier.dense.out_features, 2)

    def test_head_initialisation_is_seeded(self):
        a = ASTDetector.from_dir(self.tmp.name, seed=5, feature_extractor=self.extractor)
        b = ASTDetector.from_dir(self.tmp.name, seed=5, feature_extractor=self.extractor)
        c = ASTDetector.from_dir(self.tmp.name, seed=6, feature_extractor=self.extractor)
        self.assertTrue(torch.equal(a.model.classifier.dense.weight, b.model.classifier.dense.weight))
        self.assertFalse(torch.equal(a.model.classifier.dense.weight, c.model.classifier.dense.weight))

    def test_stages_select_exactly_the_planned_parameters(self):
        det = self.fresh()
        layout = det.parameter_report()["layout"]
        self.assertEqual(layout["blocks"], 4)
        self.assertIn(layout["block_prefix"], ("audio_spectrogram_transformer.encoder.layer", "audio_spectrogram_transformer.layers"))
        self.assertEqual(det.parameter_report()["trainable_parameters"], 0)             # starts frozen
        det.set_stage("head")
        names = det.parameter_report()["trainable_names"]
        self.assertTrue(names and all(n.startswith("classifier.") for n in names))
        det.set_stage("partial", n_blocks=2)
        names = det.parameter_report()["trainable_names"]
        groups = {det._group(n) for n in names}
        self.assertEqual(groups, {"classifier", "final_norm", "block.02", "block.03"})
        self.assertFalse(any(n.startswith("audio_spectrogram_transformer.embeddings") for n in names))
        report = det.parameter_report()
        self.assertEqual(report["groups"]["block.00"]["trainable"], 0)
        self.assertEqual(report["groups"]["block.03"]["trainable"], report["groups"]["block.03"]["parameters"])
        self.assertEqual(sum(g["trainable"] for g in report["groups"].values()), report["trainable_parameters"])
        det.set_stage("partial", n_blocks=1)
        self.assertEqual({det._group(n) for n in det.parameter_report()["trainable_names"]}, {"classifier", "final_norm", "block.03"})
        det.set_stage("frozen")
        self.assertEqual(det.parameter_report()["trainable_parameters"], 0)
        for bad in (0, 5):
            with self.assertRaises(ValueError):
                det.set_stage("partial", n_blocks=bad)
        with self.assertRaises(ValueError):
            det.set_stage("everything")

    def test_param_groups_cover_every_trainable_parameter_once(self):
        det = self.fresh()
        det.set_stage("partial")
        groups = det.param_groups(head_lr=1e-4, backbone_lr=1e-5, weight_decay=0.01)
        self.assertEqual([g["name"] for g in groups], ["head", "backbone"])
        self.assertEqual([g["lr"] for g in groups], [1e-4, 1e-5])
        flat = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), {id(p) for p in det.parameters() if p.requires_grad})
        det.set_stage("head")
        self.assertEqual([g["name"] for g in det.param_groups(head_lr=1e-3, backbone_lr=1e-5)], ["head"])
        det.set_stage("frozen")
        with self.assertRaises(ValueError):
            det.param_groups(head_lr=1e-3, backbone_lr=1e-5)

    def test_gradients_and_updates_reach_only_the_unfrozen_parts(self):
        self.needs_features()
        det = self.fresh()
        det.set_stage("partial", n_blocks=2)
        rng = np.random.default_rng(0)
        wave = torch.from_numpy(rng.standard_normal((2, RATE)).astype(np.float32) * 0.1)
        valid = torch.tensor([RATE, RATE])
        before = det.digests()
        optimizer = torch.optim.AdamW(det.param_groups(head_lr=1e-2, backbone_lr=1e-2), weight_decay=0.0)
        det.train()
        loss = torch.nn.functional.cross_entropy(det(wave, valid), torch.tensor([0, 1]))
        loss.backward()
        for name, p in det.model.named_parameters():
            group = det._group(name)
            if group in ("classifier", "final_norm", "block.02", "block.03"):
                self.assertIsNotNone(p.grad, name)
            else:
                self.assertIsNone(p.grad, name)
        self.assertTrue(any(float(p.grad.abs().sum()) > 0 for n, p in det.model.named_parameters() if det._group(n) == "block.02"))
        optimizer.step()
        after = det.digests()
        changed = {g for g in before if before[g] != after[g]}
        self.assertEqual(changed, {"classifier", "final_norm", "block.02", "block.03"})   # frozen groups are bit-identical

    def test_forward_uses_only_valid_samples(self):
        self.needs_features()
        det = self.fresh().eval()
        rng = np.random.default_rng(1)
        short = rng.standard_normal(RATE // 2).astype(np.float32) * 0.1
        long = rng.standard_normal(RATE).astype(np.float32) * 0.1
        padded = torch.zeros(2, RATE)
        padded[0, :short.size] = torch.from_numpy(short)
        padded[1] = torch.from_numpy(long)
        with torch.no_grad():
            batched = det(padded, torch.tensor([short.size, RATE]))
            alone = det(torch.from_numpy(short)[None], torch.tensor([short.size]))
        self.assertEqual(batched.shape, (2, 2))
        torch.testing.assert_close(batched[0], alone[0], rtol=1e-4, atol=1e-5)

    def test_trainer_forward_contract_keywords(self):
        self.needs_features()
        det = self.fresh().eval()
        out = det(waveforms=torch.zeros(1, RATE) + 0.01, valid_samples=torch.tensor([RATE]))
        self.assertEqual(out.shape, (1, 2))


def synthetic_manifests(directory):
    stems = [f"{1000 + s}-{200 + s}-{u:04d}" for s in range(30) for u in range(8)]
    sp = splits.assign_speakers({cohort.parse_stem(s)[0] for s in stems}, seed=42)
    plan = splits.sample_stems(stems, sp, per_speaker={"train": 4, "validation": 6, "test": 6}, seed=42)
    rows = {}
    for stem in plan:
        for comp in cohort.COMPONENTS:
            rid = cohort.recording_id(comp, stem)
            rows[rid] = {"recording_id": rid, "n_samples": RATE * 9, "retained": True}
    chunks = []
    for protocol in ("A", "B"):
        assignments, _ = splits.build_assignments(protocol, plan, sp)
        splits.write_assignments(assignments, Path(directory) / f"protocol_{protocol}_v1.parquet")
        chunks += chunk_mod.build_chunks(protocol, assignments, rows)
    write_parquet(chunks, str(Path(directory) / "chunks.parquet"))
    return sp, plan, chunks


class Budget(unittest.TestCase):
    MEASURED = {"ast_head_train_cps": 20.0, "ast_partial_train_cps": 10.0, "ast_eval_cps": 40.0, "wavlm_cps": 40.0}

    def workload(self):
        base = {"train_recordings": 100, "train_chunks": 500, "selection_chunks": 80, "calibration_chunks": 40, "operating_chunks": 40, "test_chunks": 120}
        return {"A": {**base, "train_recordings": 150}, "B": base}

    def test_arithmetic_reserve_and_labelling(self):
        out = estimate_budget(self.MEASURED, self.workload())
        by = {(i["item"], i["protocol"]): i for i in out["items"]}
        partial_b = by[("AST partial fine-tuning", "B")]
        self.assertAlmostEqual(partial_b["hours"], 2 * 8 * 100 / 10.0 / 3600, places=3)           # 2 variants x 8 epochs x 100 recs / 10 cps
        self.assertAlmostEqual(by[("AST partial fine-tuning", "A")]["hours"], 1 * 8 * 150 / 10.0 / 3600, places=3)
        warm_a = by[("AST head warm-up", "A")]
        self.assertAlmostEqual(warm_a["hours"], 2 * 150 / 20.0 / 3600, places=3)
        self.assertEqual(by[("BiGRU training", "B")]["basis"], "assumed")
        self.assertEqual(by[("WavLM head training", "A")]["basis"], "assumed")
        self.assertEqual(partial_b["basis"], "measured")
        extraction = by[("WavLM embedding extraction (shared cache)", "A+B")]
        self.assertAlmostEqual(extraction["hours"], (150 * 4 + 160 + 120) / 40.0 / 3600, places=3)   # union uses the larger protocol-A set
        self.assertAlmostEqual(out["subtotal_hours"], out["measured_hours"] + out["assumed_hours"], places=1)
        self.assertAlmostEqual(out["total_hours_with_reserve"], out["subtotal_hours"] * 1.25, places=1)
        self.assertEqual(out["sessions_needed"], math.ceil(out["total_hours_with_reserve"] / 3.0))
        self.assertAlmostEqual(out["hours_per_account"], out["total_hours_with_reserve"] / 4, delta=0.01)  # both are rounded

    def test_faster_hardware_and_fewer_epochs_reduce_the_total(self):
        base = estimate_budget(self.MEASURED, self.workload())["total_hours_with_reserve"]
        fast = estimate_budget({**self.MEASURED, "ast_partial_train_cps": 20.0}, self.workload())["total_hours_with_reserve"]
        short = estimate_budget(self.MEASURED, self.workload(), {"ast_partial_epochs": 4})["total_hours_with_reserve"]
        self.assertLess(fast, base)
        self.assertLess(short, base)
        self.assertEqual(DEFAULT_PLAN["ast_partial_epochs"], 8)   # the default plan is not mutated by overrides

    def test_missing_or_nonpositive_measurements_are_refused(self):
        for broken in ({k: v for k, v in self.MEASURED.items() if k != "wavlm_cps"}, {**self.MEASURED, "ast_eval_cps": 0}):
            with self.assertRaises(ValueError):
                estimate_budget(broken, self.workload())

    def test_workload_counts_come_from_the_manifests(self):
        with tempfile.TemporaryDirectory() as d:
            sp, plan, chunks = synthetic_manifests(d)
            work = workload_from_manifests(d, Path(d) / "chunks.parquet")
        n_train_stems = sum(1 for s in plan if sp[cohort.parse_stem(s)[0]][0] == "train")
        self.assertEqual(work["B"]["train_recordings"], n_train_stems * 3)     # genuine + melo + vits
        self.assertEqual(work["A"]["train_recordings"], n_train_stems * 5)
        per_recording = len(chunk_mod.eval_windows(RATE * 9))
        self.assertEqual(work["A"]["train_chunks"], n_train_stems * 5 * per_recording)
        for protocol in ("A", "B"):
            w = work[protocol]
            validation = sum(c.partition == "validation" and c.protocol_id == f"protocol:{protocol}" for c in chunks)
            self.assertEqual(w["selection_chunks"] + w["calibration_chunks"] + w["operating_chunks"], validation)
            self.assertGreater(w["calibration_chunks"], 0)
            self.assertGreater(w["test_chunks"], 0)


if __name__ == "__main__":
    unittest.main()
