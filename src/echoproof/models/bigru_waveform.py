"""Waveform-to-BiGRU model using the frozen R2-S1 log-mel definition."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from echoproof.features.acoustic import HOP, LOG_FLOOR, N_FFT, N_MELS, SAMPLE_RATE, WIN_LENGTH
from echoproof.models.bigru import BiGRU


class WaveformBiGRU(nn.Module):
    """Compute train-normalized mel features on the model device and mask padding."""

    def __init__(self, mean, std, hidden_size=128, dropout=0.3):
        super().__init__()
        import librosa

        mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
        if (mean.shape != (N_MELS,) or std.shape != (N_MELS,) or
                not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('invalid frozen mel normalization')
        mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
        self.register_buffer('mel_filter', torch.from_numpy(mel.copy()))
        self.register_buffer('window', torch.hann_window(WIN_LENGTH, periodic=True))
        self.register_buffer('mean', torch.from_numpy(mean.copy()))
        self.register_buffer('std', torch.from_numpy(std.copy()))
        self.bigru = BiGRU(input_size=N_MELS, hidden_size=hidden_size, dropout=dropout)

    def features(self, waveforms, valid_samples):
        if waveforms.ndim != 2 or valid_samples.ndim != 1 or len(valid_samples) != len(waveforms):
            raise ValueError('invalid waveform batch dimensions')
        if (not torch.isfinite(waveforms).all() or (valid_samples < 1).any() or
                (valid_samples > waveforms.shape[1]).any() or valid_samples.is_floating_point()):
            raise ValueError('invalid waveform samples or lengths')
        lengths = valid_samples.to(device=waveforms.device, dtype=torch.long)
        mask = torch.arange(waveforms.shape[1], device=waveforms.device)[None, :] < lengths[:, None]
        # Suppress arbitrary padding even for clips shorter than one FFT window.
        audio = waveforms.float().masked_fill(~mask, 0)
        if audio.shape[1] < N_FFT:
            audio = F.pad(audio, (0, N_FFT - audio.shape[1]))
        frame_lengths = 1 + (lengths.clamp(min=N_FFT) - N_FFT) // HOP
        # Keep FFT and mel arithmetic in fp32 even when the trainer uses AMP.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            spectrum = torch.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN_LENGTH,
                                  window=self.window, center=False, return_complex=True)
            power = spectrum.abs().square()
            mel = torch.matmul(self.mel_filter, power).clamp_min(LOG_FLOOR).log().transpose(1, 2)
            mel = (mel - self.mean) / self.std
        frame_mask = torch.arange(mel.shape[1], device=audio.device)[None, :] < frame_lengths[:, None]
        return mel.masked_fill(~frame_mask.unsqueeze(-1), 0), frame_lengths

    def forward(self, waveforms, valid_samples):
        features, lengths = self.features(waveforms, valid_samples)
        return self.bigru(features, lengths)
