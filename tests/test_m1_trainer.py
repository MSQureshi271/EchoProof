import csv
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from echoproof.contracts import Chunk
from torch.utils.data import DataLoader

from echoproof.data import cohort
from echoproof.training.datasets import ChunkDataset, CropDataset, DictWaveformLoader, ShardWaveformLoader, collate, crop_start
from echoproof.training.ledger import COLUMNS, append_ledger
from echoproof.training.trainer import ResumeError, TrainConfig, Trainer, TrainingError, recording_metrics

RATE = 16000


class TinyDetector(nn.Module):
    """waveforms [B,T], valid_samples [B] -> logits [B,2]; masked mean of a conv response."""

    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, 64, stride=32)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(8, 2)

    def forward(self, waveforms, valid_samples):
        x = torch.relu(self.conv(waveforms.unsqueeze(1)))
        frames = ((valid_samples - 64) // 32 + 1).clamp(min=1)
        mask = torch.arange(x.shape[-1], device=x.device)[None, :] < frames[:, None]
        pooled = (x * mask[:, None, :]).sum(-1) / frames[:, None]
        return self.head(self.dropout(pooled))


def fixtures(n=16, seconds=1.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(RATE * seconds)) / RATE
    waves, labels = {}, {}
    for i in range(n):
        label = i % 2
        tone = np.sin(2 * np.pi * (300 if label == 0 else 3000) * t + rng.uniform(0, 6))
        waves[f"rec:{i:02d}"] = (0.4 * tone + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
        labels[f"rec:{i:02d}"] = label
    return waves, labels


def datasets(seed=42, crop=8000):
    waves, labels = fixtures()
    loader = DictWaveformLoader(waves)
    train = CropDataset(list(labels.items()), loader, seed=seed, crop_samples=crop)
    chunks = [Chunk(1, f"chunk:B:{rid}:0", rid, "speaker:1", "protocol:B", "validation", 0, 8000, RATE, 0.0, 0.5, 8000, 8000, "a" * 64)
              for rid in sorted(labels)]
    return train, ChunkDataset(chunks, labels, loader)


def make(run_dir, **overrides):
    torch.manual_seed(0)
    cfg = dict(seed=42, max_epochs=3, batch_size=4, lr=3e-3, patience=5, owner="muneeb", model_name="tiny", protocol="B")
    cfg.update(overrides)
    train, ev = datasets()
    return Trainer(TinyDetector(), TrainConfig(**cfg), train, ev, run_dir, provenance={"split_sha256": "s" * 64},
                   ledger_path=Path(run_dir).parent / "ledger.csv")


class Datasets(unittest.TestCase):
    def test_crop_is_deterministic_and_epoch_dependent(self):
        a = crop_start(42, 1, "rec:1", 100000)
        self.assertEqual(a, crop_start(42, 1, "rec:1", 100000))
        self.assertEqual(crop_start(42, 1, "rec:1", 100), 0)
        starts = {crop_start(42, e, "rec:1", 1_000_000) for e in range(1, 20)}
        self.assertGreater(len(starts), 10)
        self.assertTrue(all(0 <= s <= 1_000_000 - 64000 for s in starts))

    def test_short_recordings_are_padded_with_true_length(self):
        loader = DictWaveformLoader({"r": np.ones(1000, dtype=np.float32)})
        item = CropDataset([("r", 1)], loader, seed=1)[0]
        self.assertEqual(item["valid_samples"], 1000)
        self.assertEqual(item["waveform"].shape, (64000,))
        self.assertEqual(float(item["waveform"][1000:].sum()), 0.0)

    def test_chunk_dataset_slices_manifest_coordinates_and_rejects_mismatch(self):
        wave = np.arange(20000, dtype=np.float32)
        loader = DictWaveformLoader({"r": wave})
        good = Chunk(1, "chunk:B:r:4000", "r:1", "speaker:1", "protocol:B", "validation", 4000, 12000, RATE, 0.25, 0.75, 8000, 64000, "a" * 64)
        ds = ChunkDataset([good], {"r:1": 1}, DictWaveformLoader({"r:1": wave}))
        item = ds[0]
        self.assertEqual(float(item["waveform"][0]), 4000.0)
        self.assertEqual(item["waveform"].shape, (64000,))
        short = DictWaveformLoader({"r:1": wave[:5000]})
        with self.assertRaises(ValueError):
            ChunkDataset([good], {"r:1": 1}, short)[0]
        with self.assertRaises(ValueError):
            ChunkDataset([good], {}, loader)


class ShardLoader(unittest.TestCase):
    """ShardWaveformLoader must give every DataLoader worker its own tar descriptors."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(3)
        self.pcm, items = {}, []
        for i in range(24):
            rid = f"phonemedf:genuine:10-20-{i:04d}"
            self.pcm[rid] = rng.integers(-20000, 20000, 3000 + 7 * i).astype("<i2")
            items.append((rid, cohort.flac_bytes(self.pcm[rid])))
        index = cohort.write_shards(items, self.tmp.name, shard_bytes=20_000)
        self.assertGreater(len({v[0] for v in index.values()}), 2)  # several shards, so workers interleave reads
        self.paths = {rid: f"{shard}::{rid}" for rid, (shard, _) in index.items()}

    def tearDown(self):
        self.tmp.cleanup()

    def expected(self, rid):
        return self.pcm[rid].astype(np.float32) / 32768.0

    def test_roundtrip_and_cache(self):
        loader = ShardWaveformLoader(self.tmp.name, self.paths, cache_size=2)
        for rid in self.pcm:
            np.testing.assert_array_equal(loader(rid), self.expected(rid))
        self.assertLessEqual(len(loader._cache), 2)
        loader.close()

    def test_handles_are_reopened_in_another_process_and_dropped_on_pickle(self):
        loader = ShardWaveformLoader(self.tmp.name, self.paths)
        rid = next(iter(self.pcm))
        loader(rid)
        parent_handles = dict(loader._tars)
        self.assertTrue(parent_handles)
        loader._pid = -1  # pretend these handles were inherited from another process
        loader(rid)
        self.assertTrue(all(loader._tars[k] is not parent_handles.get(k) for k in loader._tars))
        self.assertEqual(loader._pid, os.getpid())
        clone = pickle.loads(pickle.dumps(loader))
        self.assertEqual((clone._tars, len(clone._cache)), ({}, 0))
        np.testing.assert_array_equal(clone(rid), self.expected(rid))

    def test_two_dataloader_workers_read_correct_audio(self):
        loader = ShardWaveformLoader(self.tmp.name, self.paths)
        for rid in list(self.pcm)[:5]:
            loader(rid)  # open handles in the parent first: the situation that corrupts forked workers
        crop = 4096
        dataset = CropDataset([(rid, i % 2) for i, rid in enumerate(self.pcm)], loader, seed=7, crop_samples=crop)
        for epoch in (1, 2):
            dataset.set_epoch(epoch)
            seen = {}
            for batch in DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2, collate_fn=collate,
                                    generator=torch.Generator().manual_seed(epoch)):
                for k, rid in enumerate(batch["recording_ids"]):
                    valid = int(batch["model_inputs"]["valid_samples"][k])
                    seen[rid] = batch["model_inputs"]["waveforms"][k, :valid].numpy()
            self.assertEqual(set(seen), set(self.pcm))
            for rid, got in seen.items():
                np.testing.assert_array_equal(got, self.expected(rid))  # every recording is shorter than the crop


class Metrics(unittest.TestCase):
    def test_reference_cases(self):
        m = recording_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual((m["macro_f1"], m["eer"], m["fpr"]), (1.0, 0.0, 0.0))
        m = recording_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(m["eer"], 0.5)
        self.assertEqual(m["fp"], 2)  # score >= threshold is TTS
        self.assertTrue(np.isnan(recording_metrics([1, 1], [0.9, 0.2])["eer"]))
        m = recording_metrics([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2])
        self.assertEqual((m["macro_f1"], m["eer"]), (0.0, 1.0))
        none_positive = recording_metrics([0, 1], [0.1, 0.2])  # nothing predicted TTS
        self.assertEqual((none_positive["tp"], none_positive["fn"]), (0, 1))
        self.assertAlmostEqual(none_positive["macro_f1"], (0.0 + 2 / 3) / 2)


class TrainerBehaviour(unittest.TestCase):
    def test_overfits_16_clips(self):
        with tempfile.TemporaryDirectory() as d:
            t = make(Path(d) / "run", max_epochs=40, patience=40)
            result = t.fit()
            self.assertEqual(result["best_macro_f1"], 1.0)
            self.assertIn(result["stop_reason"], ("max_epochs", "early_stopping"))
            for name in ("best.pt", "latest.pt", "predictions_recordings.parquet", "predictions_chunks.parquet", "resolved_config.json", "status.json"):
                self.assertTrue((Path(d) / "run" / name).exists(), name)
            rows = list(csv.DictReader(open(Path(d) / "ledger.csv")))
            self.assertEqual(list(rows[0]), list(COLUMNS))
            self.assertEqual((rows[0]["status"], rows[0]["model"], rows[0]["protocol"]), ("completed", "tiny", "B"))

    def test_resume_reproduces_next_epoch_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            full = make(Path(d) / "full", max_epochs=3)
            full.fit()
            two = make(Path(d) / "split", max_epochs=2)
            two.fit()
            # same config hash is required for resume: continue with max_epochs=2 config but extend via fresh Trainer
            three = make(Path(d) / "split", max_epochs=3)
            with self.assertRaises(ResumeError):  # different config (max_epochs) is refused
                three.fit(resume=True)

    def test_resume_exact_with_same_config(self):
        with tempfile.TemporaryDirectory() as d:
            full = make(Path(d) / "full", max_epochs=3)
            full.fit()
            part = make(Path(d) / "part", max_epochs=3)
            # emulate an interruption after epoch 2 by failing inside epoch 3
            original = part._train_epoch
            def boom(epoch):
                if epoch == 3:
                    raise KeyboardInterrupt
                return original(epoch)
            part._train_epoch = boom
            with self.assertRaises(KeyboardInterrupt):
                part.fit()
            resumed = make(Path(d) / "part", max_epochs=3)
            resumed.fit(resume=True)
            h = lambda name: [__import__("json").loads(l) for l in (Path(d) / name / "history.jsonl").read_text().splitlines()]
            a, b = h("full"), h("part")
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertEqual((x["epoch"], x["train_loss"], x["val"]["loss"], x["val"]["macro_f1"]), (y["epoch"], y["train_loss"], y["val"]["loss"], y["val"]["macro_f1"]))
            for k, v in torch.load(Path(d) / "full" / "latest.pt", weights_only=True)["model"].items():
                self.assertTrue(torch.equal(v, torch.load(Path(d) / "part" / "latest.pt", weights_only=True)["model"][k]), k)
            statuses = [r["status"] for r in csv.DictReader(open(Path(d) / "ledger.csv"))]
            self.assertIn("interrupted", statuses)

    def test_resume_guards(self):
        with tempfile.TemporaryDirectory() as d:
            t = make(Path(d) / "r", max_epochs=1)
            t.fit()
            with self.assertRaises(ResumeError):
                make(Path(d) / "r", max_epochs=1).fit()  # existing checkpoint without resume
            with self.assertRaises(ResumeError):
                make(Path(d) / "r", max_epochs=1, lr=1e-2).fit(resume=True)  # config changed
            with self.assertRaises(ResumeError):
                make(Path(d) / "none", max_epochs=1).fit(resume=True)  # nothing to resume
            again = make(Path(d) / "r", max_epochs=1).fit(resume=True)  # finished run returns summary
            self.assertEqual(again["epochs_run"], 1)

    def test_early_stopping_records_best_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            t = make(Path(d) / "r", max_epochs=10, patience=2, lr=1e-12)
            result = t.fit()
            self.assertEqual(result["stop_reason"], "early_stopping")
            self.assertEqual(result["epochs_run"], result["best_epoch"] + 2)

    def test_accumulation_weights_samples_including_partial_group(self):
        with tempfile.TemporaryDirectory() as d:
            t = make(Path(d) / "r")
            t.model.dropout.p = 0.0
            t._set_train_mode()
            batches = [collate([datasets()[0][i] for i in idx]) for idx in ([0, 1, 2, 3], [4, 5])]
            t.optimizer.zero_grad()
            t._accumulate(batches)
            accumulated = [p.grad.clone() for p in t.trainable]
            t.optimizer.zero_grad()
            t._accumulate([collate([datasets()[0][i] for i in range(6)])])
            for a, b in zip(accumulated, [p.grad for p in t.trainable]):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)

    def test_frozen_modules_stay_in_eval_mode(self):
        with tempfile.TemporaryDirectory() as d:
            t = make(Path(d) / "r")
            for p in t.model.conv.parameters():
                p.requires_grad = False
            t = Trainer(t.model, t.cfg, t.train_ds, t.eval_ds, Path(d) / "r2")
            t.model.conv.train()
            t._set_train_mode()
            self.assertFalse(t.model.conv.training)
            self.assertTrue(t.model.head.training and t.model.dropout.training)
            names = {n for n, p in t.model.named_parameters() if p.requires_grad}
            self.assertEqual(names, {"head.weight", "head.bias"})

    def test_optimizer_must_cover_trainable_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            train, ev = datasets()
            model = TinyDetector()
            with self.assertRaises(ValueError):
                Trainer(model, TrainConfig(), train, ev, Path(d) / "r", param_groups=[{"params": list(model.head.parameters())}])

    def test_class_strategies_and_exclusivity(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNotNone(make(Path(d) / "a", class_strategy="weights").class_weights)
            self.assertIsNotNone(make(Path(d) / "b", class_strategy="balanced_sampling").sample_weights)
            with self.assertRaises(ValueError):
                TrainConfig(class_strategy="weights+sampling").validate()
            make(Path(d) / "c", class_strategy="balanced_sampling", max_epochs=1).fit()

    def test_schedulers_run(self):
        with tempfile.TemporaryDirectory() as d:
            for name, extra in (("plateau", {}), ("cosine", {"warmup_steps": 2})):
                t = make(Path(d) / name, scheduler=name, max_epochs=3, **extra)
                t.fit()
                lrs = [__import__("json").loads(l)["lr"] for l in (Path(d) / name / "history.jsonl").read_text().splitlines()]
                if name == "cosine":
                    self.assertLess(lrs[-1], lrs[0])

    def test_failure_is_ledgered(self):
        class Broken(TinyDetector):
            def forward(self, waveforms, valid_samples):
                return super().forward(waveforms, valid_samples) * float("nan")
        with tempfile.TemporaryDirectory() as d:
            train, ev = datasets()
            t = Trainer(Broken(), TrainConfig(max_epochs=2, batch_size=4, owner="x", model_name="broken"), train, ev, Path(d) / "r", ledger_path=Path(d) / "ledger.csv")
            with self.assertRaises(TrainingError):
                t.fit()
            rows = list(csv.DictReader(open(Path(d) / "ledger.csv")))
            self.assertEqual(rows[0]["status"], "failed:TrainingError")

    def test_ledger_rejects_unknown_columns(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                append_ledger(Path(d) / "l.csv", {"bogus": 1})


if __name__ == "__main__":
    unittest.main()
