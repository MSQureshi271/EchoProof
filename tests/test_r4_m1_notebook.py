import ast
import json
from pathlib import Path


def test_notebook_is_valid_and_runs_required_stages():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/'notebooks/R4_M1_Colab.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    code = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            text = ''.join(cell['source']); ast.parse(text); code.append(text)
    combined = '\n'.join(code)
    for stage in ('r4_m1_preflight.py', 'r4_m1_lr.py', 'r4_m1_bigru.py', 'r4_m1_export.py'):
        assert stage in combined
    assert 'RESUME_BIGRU' in combined and 'skipped' in combined
