import importlib.util
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run.py'
spec = importlib.util.spec_from_file_location('lr_calibration_experiment', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_platt_roundtrip_and_ranking():
    p = np.array([.05, .2, .4, .6, .8, .95])
    y = np.array([0, 0, 1, 0, 1, 1])
    params = module.fit('platt', y, p)
    import json
    restored = json.loads(json.dumps(params))
    a, b = module.apply('platt', params, p), module.apply('platt', restored, p)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)

def test_single_class_is_rejected():
    import pytest
    with pytest.raises(ValueError, match='both classes'):
        module.fit('platt', np.array([1, 1]), np.array([.3, .7]))

def test_operating_analysis_matches_run_and_thresholds():
    import hashlib
    import json
    root = SCRIPT.parents[1]
    run = root/'runs/initial'
    if not (run/'analysis/evidence.json').exists():
        import pytest
        pytest.skip('initial run is not synced locally')
    analysis = json.loads((run/'analysis/evidence.json').read_text(encoding='utf-8'))
    assert analysis['source_evidence_sha256'] == hashlib.sha256((run/'evidence.json').read_bytes()).hexdigest()
    for name, expected in analysis['outputs'].items():
        assert hashlib.sha256((run/'analysis'/name).read_bytes()).hexdigest() == expected
    chosen = [r for r in analysis['thresholds'] if r['rule'] == 'operating_selected']
    assert len(chosen) == 3
    assert len({(r['tp'], r['fp'], r['tn'], r['fn']) for r in chosen}) == 1

def test_notebook_code_is_valid_python():
    import ast
    import json
    notebook = json.loads((SCRIPT.parents[1]/'notebooks/LR_Calibration_Colab.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))

