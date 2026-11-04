import re
import unittest

import numpy as np
import torch
from torch import nn

from echoproof.features.pretrained import PRETRAINED, SAMPLE_RATE, WavLMExtractor, list_trainable, pool_hidden

try:
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel
    HAVE_TRANSFORMERS = True
except ImportError:  # the extractor tests need transformers; the registry and pooling tests do not
    HAVE_TRANSFORMERS = False

TOL = 1e-4


def tiny_extractor(seed=0, do_normalize=False, batch_size=4):
    torch.manual_seed(seed)
    config = WavLMConfig(vocab_size=32, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(8,) * 7, conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                         feat_extract_norm="group", num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                         num_buckets=16, max_bucket_distance=32, apply_spec_augment=False, do_stable_layer_norm=False)
    extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0, do_normalize=do_normalize,
                                         return_attention_mask=False)
    return WavLMExtractor(WavLMModel(config), extractor, batch_size=batch_size)


def clips(lengths, seed=1):
    rng = np.random.default_rng(seed)
    return [(0.3 * np.sin(2 * np.pi * rng.uniform(100, 3000) * np.arange(n) / SAMPLE_RATE) + 0.05 * rng.standard_normal(n)).astype(np.float32)
            for n in lengths]


class Registry(unittest.TestCase):
    def test_both_models_are_pinned_to_full_commits_with_a_licence_decision(self):
        for key, spec in PRETRAINED.items():
            self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$", key)
            self.assertIn("/", spec["repo_id"])
            self.assertIn("preprocessor_config.json", spec["files"])
            self.assertIn(spec["weights_file"], spec["files"])
            self.assertTrue(spec["license"] or spec["license_status"], key)  # a licence, or an explicit unresolved status
        for key in ("wavlm", "ast"):                                                  # recorded from the first verified downloads
            self.assertRegex(PRETRAINED[key]["weights_sha256"], r"^[0-9a-f]{64}$", key)
        self.assertIsNone(PRETRAINED["wavlm"]["license"])                      # not declared upstream: stays visibly unresolved
        self.assertIn("confirm", PRETRAINED["wavlm"]["license_status"])
        self.assertEqual(PRETRAINED["ast"]["license"], "bsd-3-clause")
        self.assertEqual(PRETRAINED["wavlm"]["expected_preprocessing"]["do_normalize"], False)


class Pooling(unittest.TestCase):
    def test_padded_frames_never_contribute(self):
        hidden = torch.zeros(2, 5, 3)
        hidden[0, :3] = torch.tensor([[1.0, 2, 3], [3, 4, 5], [5, 6, 7]])
        hidden[0, 3:] = 1000.0                      # padding garbage
        hidden[1] = 2.0
        lengths = torch.tensor([3, 5])
        mean = pool_hidden(hidden, lengths, "mean")
        np.testing.assert_allclose(mean[0].numpy(), [3, 4, 5])
        np.testing.assert_allclose(mean[1].numpy(), [2, 2, 2])
        both = pool_hidden(hidden, lengths, "mean_std")
        self.assertEqual(both.shape, (2, 6))
        np.testing.assert_allclose(both[0, 3:].numpy(), np.std([[1, 2, 3], [3, 4, 5], [5, 6, 7]], axis=0), rtol=1e-6)  # population std
        np.testing.assert_allclose(both[1, 3:].numpy(), [0, 0, 0], atol=1e-7)

    def test_invalid_inputs(self):
        h = torch.zeros(2, 4, 3)
        for bad in (torch.tensor([0, 2]), torch.tensor([5, 2]), torch.tensor([2])):
            with self.assertRaises(ValueError):
                pool_hidden(h, bad)
        with self.assertRaises(ValueError):
            pool_hidden(h, torch.tensor([2, 2]), "median")


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class Extractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = tiny_extractor()

    def test_a_fresh_extractor_is_already_in_eval_mode(self):
        fresh = tiny_extractor(seed=3)           # no train()/eval() call has been made
        self.assertFalse(fresh.training)
        self.assertFalse(fresh.encoder.training)
        self.assertFalse(fresh.describe()["training_mode"])

    def test_frozen_eval_only_and_no_gradients(self):
        self.assertEqual(list_trainable(self.ex), [])
        self.ex.train()
        self.ex.encoder.train()  # even a direct call on the encoder is undone by the next train()/eval() of the wrapper
        self.ex.train()
        self.assertFalse(self.ex.training)
        before = {k: v.clone() for k, v in self.ex.encoder.state_dict().items()}
        out = self.ex.embed(clips([16000, 16000]))
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.float32)
        for k, v in self.ex.encoder.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        self.assertEqual(self.ex.describe()["trainable_parameters"], 0)

    def test_hidden_length_matches_the_encoder_output(self):
        for n in (400, 401, 1600, 8000, 16000, 64000):
            x = torch.from_numpy(clips([n])[0])[None]
            with torch.inference_mode():
                frames = self.ex.encoder(x).last_hidden_state.shape[1]
            self.assertEqual(self.ex.hidden_length(n), frames, n)
        with self.assertRaises(ValueError):
            self.ex.embed([np.zeros(100, dtype=np.float32)])      # shorter than one receptive field

    def test_single_equals_batched_for_full_and_short_crops(self):
        for length in (64000, 16000, 8000, 12345):                # full 4 s crop and short clips
            batch = clips([length] * 5, seed=length)
            together = self.ex.embed(batch)
            for pooling in ("mean", "mean_std"):
                together = self.ex.embed(batch, pooling)
                for i, w in enumerate(batch):
                    np.testing.assert_allclose(self.ex.embed([w], pooling)[0], together[i], atol=TOL, rtol=TOL)

    def test_mixed_lengths_keep_order_and_match_individual_embeddings(self):
        batch = clips([16000, 8000, 16000, 12000, 8000, 16000, 20000])
        together = self.ex.embed(batch, "mean_std")
        self.assertEqual(together.shape, (7, 64))
        for i, w in enumerate(batch):
            np.testing.assert_allclose(self.ex.embed([w], "mean_std")[0], together[i], atol=TOL, rtol=TOL)

    def test_small_batch_size_gives_the_same_result(self):
        batch = clips([16000] * 6)
        np.testing.assert_allclose(tiny_extractor(batch_size=1).embed(batch), tiny_extractor(batch_size=6).embed(batch), atol=TOL, rtol=TOL)

    def test_zero_padding_into_a_batch_would_change_embeddings(self):
        # Why the extractor buckets by length: group norm over time makes padded and unpadded results differ.
        short, long = clips([8000])[0], clips([16000], seed=9)[0]
        alone = self.ex.embed([short])[0]
        padded = torch.zeros(2, 16000)
        padded[0, :8000] = torch.from_numpy(short)
        padded[1] = torch.from_numpy(long)
        with torch.inference_mode():
            hidden = self.ex.encoder(padded).last_hidden_state
        pooled = pool_hidden(hidden, torch.tensor([self.ex.hidden_length(8000), self.ex.hidden_length(16000)]))[0].numpy()
        self.assertGreater(float(np.max(np.abs(alone - pooled))), 10 * TOL)

    def test_only_the_head_is_trainable_when_a_head_is_attached(self):
        class Model(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                self.head = nn.Sequential(nn.Linear(encoder.dim, 8), nn.GELU(), nn.Dropout(0.2), nn.Linear(8, 2))
        model = Model(tiny_extractor())
        self.assertEqual(list_trainable(model), ["head.0.bias", "head.0.weight", "head.3.bias", "head.3.weight"])
        model.train()
        self.assertFalse(model.encoder.training)   # the wrapper keeps itself in eval under a blanket train()
        self.assertFalse(model.encoder.encoder.training)
        self.assertTrue(model.head.training)

    def test_preprocessing_comes_from_the_feature_extractor(self):
        normalized, plain = tiny_extractor(do_normalize=True), tiny_extractor(do_normalize=False)
        self.assertTrue(normalized.describe()["do_normalize"])
        w = clips([16000])[0] * 0.01 + 0.2   # offset and tiny scale: normalisation visibly changes the input
        a, b = normalized.embed([w])[0], plain.embed([w])[0]
        self.assertGreater(float(np.max(np.abs(a - b))), 10 * TOL)

    def test_bad_inputs(self):
        for bad in ([np.zeros((2, 16000), dtype=np.float32)], [np.array([], dtype=np.float32)],
                    [np.full(16000, np.nan, dtype=np.float32)], []):
            with self.assertRaises(ValueError):
                self.ex.embed(bad)
        with self.assertRaises(ValueError):
            self.ex.embed(clips([16000]), "median")


if __name__ == "__main__":
    unittest.main()
