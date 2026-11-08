"""Exercise hidden64 gradient flow, padding invariance and weight reload on fixtures."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from echoproof.models.bigru_waveform import WaveformBiGRU


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_hidden64_waveform_gradients_padding_and_reload(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA fixture requires Colab GPU')
    torch.manual_seed(42)
    model = WaveformBiGRU(np.zeros(64), np.ones(64), hidden_size=64, dropout=.3).to(device)
    assert model.bigru.gru.num_layers == 2
    assert sum(p.numel() for p in model.parameters()) == 124674
    wave = torch.randn(2, 2400, device=device) * .05
    lengths = torch.tensor([1200, 2400], device=device)
    model.eval()
    with torch.no_grad():
        first = model(wave, lengths)
        changed = wave.clone()
        changed[0, 1200:] = 20
        assert torch.allclose(first, model(changed, lengths), atol=1e-5, rtol=1e-5)
        restored = WaveformBiGRU(np.zeros(64), np.ones(64), hidden_size=64, dropout=.3).to(device).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        assert torch.allclose(first, restored(wave, lengths), atol=1e-6, rtol=1e-6)
    model.train()
    before = model.bigru.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    loss = torch.nn.functional.cross_entropy(model(wave, lengths), torch.tensor([0, 1], device=device))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    assert not torch.equal(before, model.bigru.head.weight)
    assert torch.isfinite(loss)
