import numpy as np
import pytest

torch = pytest.importorskip('torch')
from echoproof.features.acoustic import log_mel
from echoproof.models.bigru_waveform import WaveformBiGRU


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
@pytest.mark.parametrize('length', [100, 512, 1200, 64000])
def test_mel_matches_reference_and_ignores_padding(device, length):
    rng = np.random.default_rng(42)
    wave = rng.normal(0, 0.1, length).astype(np.float32)
    mean = np.linspace(-5, 5, 64, dtype=np.float32)
    std = np.linspace(1, 3, 64, dtype=np.float32)
    model = WaveformBiGRU(mean, std, hidden_size=4, dropout=0).to(device).eval()
    padded = np.full((2, max(length, 2000)), 0.25, dtype=np.float32)
    padded[0, :length] = wave
    padded[1] = rng.normal(0, 0.1, padded.shape[1])
    batch = torch.tensor(padded, device=device)
    with torch.no_grad():
        features, lengths = model.features(batch, torch.tensor([length, padded.shape[1]], device=device))
        alone = model(batch[:1, :length], torch.tensor([length], device=device))[0]
        together = model(batch, torch.tensor([length, padded.shape[1]], device=device))[0]
    expected = (log_mel(wave) - mean) / std
    assert lengths[0].item() == len(expected)
    np.testing.assert_allclose(features[0, :len(expected)].cpu().numpy(), expected, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(alone, together, atol=1e-5, rtol=1e-5)


def test_invalid_statistics_and_lengths():
    with pytest.raises(ValueError):
        WaveformBiGRU(np.zeros(64), np.zeros(64))
    model = WaveformBiGRU(np.zeros(64), np.ones(64), hidden_size=4)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 512), torch.tensor([0]))
