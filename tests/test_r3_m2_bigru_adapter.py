"""Synthetic waveform parity of the new inference bundle, on CPU and available CUDA."""
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from echoproof.artifacts import file_digest
from echoproof.bundle import BundleError
from echoproof.models.base import InferenceModelAdapter
from echoproof.models.bigru_adapter import BiGRUBundleAdapter
from echoproof.models.bigru_waveform import WaveformBiGRU
from echoproof.models.r3_m2_bundle import package_bigru


@pytest.fixture
def packaged(tmp_path):
    torch.manual_seed(42)
    model = WaveformBiGRU(np.zeros(64), np.ones(64), hidden_size=4, dropout=.3).eval()
    checkpoint = tmp_path/'best.pt'
    torch.save({'model': model.state_dict(), 'epoch': 16, 'config_hash': 'fixture-config'}, checkpoint)
    target = tmp_path/'bundle'
    args = dict(bundle_id='bundle:r3-m2-fixture-bigru',
                architecture={'hidden_size': 4, 'num_layers': 2, 'bidirectional': True, 'dropout': .3}, protocol='B',
                expected_checkpoint_sha256=file_digest(checkpoint), split_sha256='0'*64,
                expected_epoch=16, expected_config_hash='fixture-config')
    package_bigru(checkpoint, target, **args)
    return model, target, checkpoint, args


@pytest.mark.parametrize('device', ['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_bundle_parity_order_empty_batch_and_padding(packaged, device):
    model, target, _, _ = packaged
    model.to(device)
    adapter = BiGRUBundleAdapter.load(target, device=device, batch_size=2)
    assert isinstance(adapter, InferenceModelAdapter)
    assert adapter.metadata['score_type'] == 'raw_uncalibrated'
    assert adapter.score_chunks([]).shape == (0,)
    with pytest.raises(TypeError):
        adapter.metadata['model'] = 'changed'
    rng = np.random.default_rng(42)
    waves = [rng.normal(0, .05, n).astype(np.float32) for n in (100, 2500, 6400)]
    expected = []
    with torch.inference_mode():
        for wave in waves:
            logits = model(torch.tensor(wave[None], device=device), torch.tensor([len(wave)], device=device))
            expected.append(torch.softmax(logits.float(), 1)[0, 1].item())
    actual = adapter.score_chunks(waves)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(adapter.score_chunks(waves[::-1])[::-1], actual, atol=1e-5, rtol=1e-5)
    bare = BiGRUBundleAdapter.load(target/'model', device=device)
    np.testing.assert_allclose(bare.score_chunks(waves), actual, atol=1e-5, rtol=1e-5)


def test_bundle_rejects_corrupt_weights_and_wrong_declared_width(packaged, tmp_path):
    _, target, checkpoint, args = packaged
    with (target/'model/weights.pt').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(BundleError, match='Checksum'):
        BiGRUBundleAdapter.load(target)
    changed = dict(args, architecture=dict(args['architecture'], hidden_size=8))
    with pytest.raises(RuntimeError):
        package_bigru(checkpoint, tmp_path/'wrong-width', **changed)
    with pytest.raises(ValueError, match='epoch'):
        package_bigru(checkpoint, tmp_path/'wrong-epoch', **dict(args, expected_epoch=10))
