"""R3-U1 classifier head on pooled WavLM embeddings.

Linear(in_dim, hidden) -> GELU -> Dropout -> Linear(hidden, 2). `in_dim` is 768 for mean pooling and 1,536 for mean+std pooling
(the encoder's 768-wide hidden state, pooled by `features.pretrained.pool_hidden`). Class 1 is TTS: `scores` returns softmax P(TTS).
The head never sees audio; the encoder lives in the inference adapter (`models.wavlm_adapter`).
"""
from __future__ import annotations

import torch
from torch import nn

POOLING_DIM = {"mean": 768, "mean_std": 1536}


class WavLMHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        if type(in_dim) is not int or type(hidden) is not int or in_dim < 1 or hidden < 1:
            raise ValueError("in_dim and hidden must be positive integers")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.in_dim, self.hidden, self.dropout_p = in_dim, hidden, float(dropout)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(self.dropout_p), nn.Linear(hidden, 2))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.in_dim:
            raise ValueError(f"expected [batch, {self.in_dim}] embeddings, got {tuple(embeddings.shape)}")
        return self.net(embeddings)

    @torch.no_grad()
    def scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """P(TTS) per row, in eval mode semantics regardless of the current mode."""
        was_training = self.training
        self.eval()
        try:
            return torch.softmax(self(embeddings).float(), dim=1)[:, 1]
        finally:
            self.train(was_training)

    def architecture(self) -> dict:
        return {"in_dim": self.in_dim, "hidden": self.hidden, "dropout": self.dropout_p}
