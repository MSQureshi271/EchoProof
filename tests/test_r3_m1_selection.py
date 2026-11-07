from types import SimpleNamespace
import pytest

from echoproof.training.r3_m1_selection import SelectionAggregation, selection_metrics, selection_panels


def dataset(role='selection'):
    chunks = [SimpleNamespace(chunk_id='c:1', recording_id='r:1'),
              SimpleNamespace(chunk_id='c:2', recording_id='r:1'),
              SimpleNamespace(chunk_id='c:3', recording_id='r:2'),
              SimpleNamespace(chunk_id='c:4', recording_id='r:3')]
    return SimpleNamespace(validation_role=role, chunks=chunks,
                           labels_by_recording={'r:1': 0, 'r:2': 1, 'r:3': 1})


def test_aggregation_and_metrics_are_recording_level():
    rows = [('c:1', 'r:1', 0, .1), ('c:2', 'r:1', 0, .3),
            ('c:3', 'r:2', 1, .8), ('c:4', 'r:3', 1, .5)]
    recordings = SelectionAggregation(dataset())(rows)
    assert recordings[0] == ('r:1', 0, pytest.approx(.2), 2)
    assert selection_metrics([x[1] for x in recordings], [x[2] for x in recordings], .5)['macro_f1'] == 1
    panels = selection_panels(recordings, {'r:1': 'genuine', 'r:2': 'melo', 'r:3': 'xtts'})
    assert panels['known']['recordings'] == panels['unseen_xtts']['recordings'] == 2


def test_selection_rejects_other_roles_and_bad_coverage():
    with pytest.raises(ValueError, match='selection validation role'):
        SelectionAggregation(dataset('calibration'))
    agg = SelectionAggregation(dataset())
    with pytest.raises(ValueError, match='coverage'):
        agg([('c:1', 'r:1', 0, .1)])
    with pytest.raises(ValueError, match='duplicate'):
        agg([('c:1', 'r:1', 0, .1), ('c:1', 'r:1', 0, .1)])
    with pytest.raises(ValueError, match='threshold'):
        selection_metrics([0, 1], [.1, .9], .6)
