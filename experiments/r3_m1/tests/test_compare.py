import importlib.util
from pathlib import Path
import sys
import pytest

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('experiment_compare', scripts / 'compare.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_hypothetical_stop_can_miss_later_recovery():
    scores = [.5, .6, .55, .54, .53, .9] + [.8]*14
    best = -1
    history = []
    for epoch, score in enumerate(scores, 1):
        history.append({'epoch': epoch, 'improved': score > best, 'seconds': 1, 'val': {'macro_f1': score}})
        best = max(best, score)
    result = module.summarize_history(history)
    assert result['patience3']['stop_epoch'] == 5
    assert result['patience3']['best_epoch'] == 2
    assert result['full20_best_epoch'] == 6
    assert result['gain_over_patience3'] == pytest.approx(.3)
    with pytest.raises(ValueError, match='complete'):
        module.summarize_history(history[:-1])


def test_tie_retains_original_selected_model():
    names = ['original_baseline', 'original_lower_lr', 'b01_baseline_full20', 'b01_lower_lr_full20']
    candidates = [{'id': n, 'macro_f1': .8} for n in names]
    assert module.choose(candidates)['id'] == 'original_lower_lr'
    candidates[-1]['macro_f1'] = .81
    assert module.choose(candidates)['id'] == names[-1]


def test_metrics_recomputed_from_parquet_and_bad_mean_rejected(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = ['phonemedf:genuine:1-1-1', 'phonemedf:melo:1-1-1', 'phonemedf:xtts:1-1-1']
    chunks = [{'chunk_id': 'c:'+str(i), 'recording_id': rid, 'label': int(i>0), 'tts_probability': s}
              for i,(rid,s) in enumerate(zip(ids,[.2,.8,.7]))]
    recordings = [{'recording_id': rid, 'label': int(i>0), 'raw_score': s, 'chunks': 1}
                  for i,(rid,s) in enumerate(zip(ids,[.2,.8,.7]))]
    cp, rp = tmp_path/'chunks.parquet', tmp_path/'recordings.parquet'
    pq.write_table(pa.Table.from_pylist(chunks), cp)
    pq.write_table(pa.Table.from_pylist(recordings), rp)
    expected = {r['chunk_id']: (r['recording_id'], r['label']) for r in chunks}
    assert module.metrics_from_files(cp, rp, expected)[0]['macro_f1'] == 1
    recordings[0]['raw_score'] = .3
    pq.write_table(pa.Table.from_pylist(recordings), rp)
    with pytest.raises(ValueError, match='means'):
        module.metrics_from_files(cp, rp, expected)
