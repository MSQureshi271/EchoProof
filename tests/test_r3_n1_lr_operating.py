"""Operating evaluation must preserve the frozen calibration boundary."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from echoproof.artifacts import file_digest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('r3_n1_lr_operating', ROOT/'scripts/r3_n1_lr_operating.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_reliability_counts_and_edge_values():
    bins, ece = runner.reliability([0, 1, 1], [0.0, .5, 1.0])
    assert sum(b['count'] for b in bins) == 3
    assert bins[0]['count'] == bins[5]['count'] == bins[9]['count'] == 1
    assert ece == pytest.approx(1/6)


def test_evaluation_uses_frozen_fit_and_rejects_speaker_overlap(tmp_path, monkeypatch):
    contract = {'output_root': 'runs/r3/r3_n1_lr', 'bundle_sha256': 'bundle', 'epsilon': 1e-7,
                'calibration_method': 'temperature_on_log_odds', 'temperature_bounds': [.01, 50.0],
                'calibration_acceptance': 'operating_log_loss_strictly_improves_and_brier_does_not_worsen; otherwise retain_raw_model_score',
                'calibration': {'recordings': 4, 'groups': 2, 'chunks': 4},
                'operating': {'recordings': 4, 'groups': 2, 'chunks': 4}}
    cp = tmp_path/'contract.json'; cp.write_text(json.dumps(contract))
    root = tmp_path/'runs/r3/r3_n1_lr'; export = root/'export'; fitdir = root/'fit'
    export.mkdir(parents=True); fitdir.mkdir()
    def rows(role, scores):
        return [{'recording_id': f'{role}{i}', 'group_id': f'{role}{i//2}', 'role': role,
                 'component': 'genuine' if i % 2 == 0 else 'xtts', 'label': i % 2,
                 'raw_score': scores[i], 'chunk_count': 1} for i in range(4)]
    names = {}
    for role, scores in [('calibration', [.1, .9, .2, .8]), ('operating', [.1, .9, .2, .8])]:
        name = f'predictions_recordings_{role}.parquet'; pq.write_table(pa.Table.from_pylist(rows(role, scores)), export/name)
        names[name] = file_digest(export/name)
    evidence = {'task': 'R3-N1-LR prediction export', 'status': 'complete', 'contract_sha256': file_digest(cp),
                'selected_bundle_sha256': 'bundle', 'outputs': names}
    ep = export/'evidence.json'; ep.write_text(json.dumps(evidence))
    fit = {'task': 'R3-N1-LR temperature fit', 'status': 'calibration_fit_complete', 'source_contract_sha256': file_digest(cp),
           'source_export_evidence_sha256': file_digest(ep), 'source_calibration_predictions_sha256': names['predictions_recordings_calibration.parquet'],
           'selected_bundle_sha256': 'bundle', 'fit_role': 'calibration', 'method': contract['calibration_method'],
           'epsilon': contract['epsilon'], 'bounds': contract['temperature_bounds'], 'temperature': 2.0}
    (fitdir/'temperature.json').write_text(json.dumps(fit))
    # Stand-in plot backend; the real rendering runs in Colab with matplotlib.
    mpl = types.ModuleType('matplotlib'); mpl.use = lambda *a: None
    plt = types.ModuleType('matplotlib.pyplot')
    class Plot:
        def plot(self, *a, **k): pass
        def set(self, *a, **k): pass
        def legend(self): pass
        def tight_layout(self): pass
        def savefig(self, path): Path(path).write_bytes(b'fixture')
    plt.subplots = lambda: (Plot(), Plot()); plt.close = lambda *a: None
    monkeypatch.setitem(sys.modules, 'matplotlib', mpl); monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', plt)
    monkeypatch.setattr(runner, 'ROOT', tmp_path); monkeypatch.setattr(runner, 'CONTRACT', cp)
    for name in ('scripts/r3_n1_lr_operating.py', 'src/echoproof/evaluation/threshold.py'):
        p = tmp_path/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixture')
    monkeypatch.setattr(runner, '__file__', str(tmp_path/'scripts/r3_n1_lr_operating.py'))
    output = root/'evaluation'; result = runner.evaluate(export, fitdir, output)
    assert result['temperature_accepted'] is False and result['score_type'] == 'model_score'
    assert sum(result['operating_rule'][k] for k in ('tp', 'fp', 'tn', 'fn')) == 4
    assert len(pq.read_table(output/'operating_predictions.parquet')) == 4
    with pytest.raises(FileExistsError): runner.evaluate(export, fitdir, output)
    op_rows = rows('operating', [.1, .9, .2, .8])
    op_rows[0]['group_id'] = op_rows[1]['group_id'] = 'calibration0'
    op_path = export/'predictions_recordings_operating.parquet'; pq.write_table(pa.Table.from_pylist(op_rows), op_path)
    evidence['outputs']['predictions_recordings_operating.parquet'] = file_digest(op_path)
    ep.write_text(json.dumps(evidence)); fit['source_export_evidence_sha256'] = file_digest(ep)
    (fitdir/'temperature.json').write_text(json.dumps(fit))
    with pytest.raises(ValueError, match='overlap'): runner.evaluate(export, fitdir, root/'other')
