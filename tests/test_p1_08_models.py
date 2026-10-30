"""Behavioral checks for fixture-only acoustic features and baseline models."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from echoproof.features.acoustic import FEATURE_NAMES, handcrafted, log_mel
from echoproof.models.bigru import FixtureBiGRUAdapter
from echoproof.models.lr import FixtureLRAdapter


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        time = np.arange(4000, dtype=np.float32) / 16000
        cls.waves = [np.sin(time * 2 * np.pi * f).astype(np.float32) * 0.2 for f in (120, 200, 300)]
        cls.waves.append(np.zeros(4000, dtype=np.float32))
        cls.chunks = [(f"fixture:chunk-{i}", f"fixture:recording-{i}", wave) for i, wave in enumerate(cls.waves)]

    def test_handcrafted_schema_silence_and_short_audio(self):
        features = handcrafted(self.waves[-1])
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertEqual(features[-3], 0)
        self.assertTrue(np.isnan(features[-2:]).all())
        self.assertFalse(np.isinf(features).any())
        self.assertEqual(len(handcrafted(np.ones(30, dtype=np.float32))), len(FEATURE_NAMES))

    def test_log_mel_lengths_and_valid_frame_values(self):
        self.assertEqual(log_mel(self.waves[0]).shape, (1 + (4000 - 512) // 160, 64))
        self.assertEqual(log_mel(np.zeros(10, dtype=np.float32)).shape, (1, 64))
        self.assertTrue(np.isfinite(log_mel(self.waves[0])).all())
        with self.assertRaises(ValueError):
            log_mel(np.array([float('nan')]))

    def test_lr_imputes_and_reload_preserves_ordered_scores(self):
        model = FixtureLRAdapter.fit(self.waves, [0, 1, 0, 1])
        batch = model.prepare_features(self.chunks)
        self.assertTrue(np.isfinite(batch.values).all())
        scores = model.score_chunks(batch)
        self.assertEqual([s.chunk_id for s in scores], [c[0] for c in self.chunks])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "disposable.joblib"
            model.save(path)
            self.assertEqual(scores, FixtureLRAdapter.load(path).score_chunks(batch))

    def test_bigru_batch_padding_and_gradients(self):
        torch.manual_seed(42)
        chunks = [self.chunks[0], ("fixture:short", "fixture:short", self.waves[1][:1000])]
        adapter = FixtureBiGRUAdapter.from_fixtures([c[2] for c in chunks])
        batch = adapter.prepare_features(chunks)
        x = torch.tensor(batch.values)
        lengths = torch.tensor(batch.lengths)
        adapter.model.eval()
        with torch.no_grad():
            combined = adapter.model(x, lengths)
            alone = adapter.model(x[1:2, :lengths[1]], lengths[1:2])
            x[1, lengths[1]:] = 99
            changed = adapter.model(x, lengths)
        self.assertEqual(tuple(combined.shape), (2, 2))
        torch.testing.assert_close(combined[1], alone[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(combined, changed, atol=1e-5, rtol=1e-5)
        adapter.model.train()
        loss = torch.nn.functional.cross_entropy(adapter.model(x, lengths), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in adapter.model.parameters()))


if __name__ == "__main__":
    unittest.main()
