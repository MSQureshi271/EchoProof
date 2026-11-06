import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('r2_m2', Path(__file__).resolve().parents[1] / 'scripts/r2_m2_bigru.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_selection_is_balanced_deterministic_and_training_only():
    rows, assignments = [], []
    for i in range(80):
        component = 'genuine' if i % 2 == 0 else 'melo'
        rid = f'fixture:{i}'
        rows.append({'recording_id': rid, 'retained': True, 'component': component, 'label': i % 2})
        assignments.append(SimpleNamespace(recording_id=rid, protocol_id='protocol:B', partition='train' if i < 70 else 'validation'))
    chosen = mod.select_rows(rows, assignments)
    assert len(chosen) == 32
    assert sum(r['label'] for r in chosen) == 16
    assert all(int(r['recording_id'].split(':')[1]) < 70 for r in chosen)
    assert chosen == mod.select_rows(rows[::-1], assignments)


def test_padding_does_not_change_model_output():
    torch.manual_seed(42)
    a, b = np.ones((3, 64), np.float32), np.ones((9, 64), np.float32)
    one, l1 = mod.pad_sequences([a])
    batch, l2 = mod.pad_sequences([a, b])
    model = mod.BiGRU(hidden_size=4, dropout=0).eval()
    with torch.no_grad():
        assert torch.allclose(model(one, l1)[0], model(batch, l2)[0], atol=1e-5)
    assert l2.tolist() == [3, 9]
