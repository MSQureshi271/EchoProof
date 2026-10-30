"""Masked temporal BiGRU for software probes and later development."""
import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from types import MappingProxyType
from echoproof.contracts import ChunkScore, FeatureBatch, validate_chunk_scores
from echoproof.features.acoustic import FEATURE_SCHEMA_HASH, log_mel


class BiGRU(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 2)

    def forward(self, inputs, lengths):
        if inputs.ndim != 3 or lengths.ndim != 1 or len(lengths) != inputs.shape[0] or (lengths < 1).any() or (lengths > inputs.shape[1]).any():
            raise ValueError("invalid sequence dimensions or lengths")
        packed = pack_padded_sequence(inputs, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        output, _ = pad_packed_sequence(encoded, batch_first=True, total_length=inputs.shape[1])
        mask = torch.arange(inputs.shape[1], device=inputs.device)[None, :] < lengths.to(inputs.device)[:, None]
        pooled = (output * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(inputs.device).unsqueeze(1)
        return self.head(self.dropout(pooled))


class FixtureBiGRUAdapter:
    def __init__(self, model, mean, scale):
        self.model = model.cpu()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.mean.shape != (64,) or self.scale.shape != (64,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("invalid fixture normalization")
        self.metadata = MappingProxyType({"scope": "software_probe", "feature_schema_hash": FEATURE_SCHEMA_HASH})

    @classmethod
    def from_fixtures(cls, waveforms):
        sequences = [log_mel(w) for w in waveforms]
        frames = np.concatenate(sequences, axis=0)
        return cls(BiGRU(), frames.mean(axis=0), np.maximum(frames.std(axis=0), 1e-5))

    def prepare_features(self, chunks):
        sequences = [(log_mel(item[2]) - self.mean) / self.scale for item in chunks]
        lengths = np.asarray([len(seq) for seq in sequences], dtype=np.int64)
        values = np.zeros((len(chunks), int(lengths.max()), 64), dtype=np.float32)
        mask = np.arange(values.shape[1])[None, :] < lengths[:, None]
        for index, seq in enumerate(sequences):
            values[index, :len(seq)] = seq
        batch = FeatureBatch(values, [c[0] for c in chunks], [c[1] for c in chunks], None, lengths, mask, FEATURE_SCHEMA_HASH)
        batch.validate()
        return batch

    def score_chunks(self, batch):
        batch.validate()
        if batch.feature_schema_hash != FEATURE_SCHEMA_HASH:
            raise ValueError("incompatible feature schema")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.as_tensor(batch.values, dtype=torch.float32), torch.as_tensor(batch.lengths, dtype=torch.int64))
            scores = torch.softmax(logits, dim=1)[:, 1].tolist()
        result = [ChunkScore(chunk_id, float(score)) for chunk_id, score in zip(batch.chunk_ids, scores)]
        validate_chunk_scores(batch.chunk_ids, result)
        return result

    def save(self, destination):
        torch.save({"scope": "software_probe", "feature_schema_hash": FEATURE_SCHEMA_HASH,
                    "mean": torch.tensor(self.mean), "scale": torch.tensor(self.scale), "state_dict": self.model.state_dict()}, destination)

    @classmethod
    def load(cls, source):
        state = torch.load(source, map_location="cpu", weights_only=True)
        if state.get("scope") != "software_probe" or state.get("feature_schema_hash") != FEATURE_SCHEMA_HASH:
            raise ValueError("incompatible fixture model")
        model = BiGRU()
        model.load_state_dict(state["state_dict"])
        return cls(model, state["mean"].numpy(), state["scale"].numpy())
