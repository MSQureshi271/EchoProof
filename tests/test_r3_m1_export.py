from types import SimpleNamespace
import pytest

from echoproof.training.r3_m1_export import aggregate_role, choose_winner


def test_choose_winner_and_tie_are_fixed():
    assert choose_winner({'baseline': .7, 'lower_lr': .71}) == 'lower_lr'
    assert choose_winner({'baseline': .7, 'lower_lr': .7}) == 'baseline'
    with pytest.raises(ValueError):
        choose_winner({'baseline': .7, 'lower_lr': float('nan')})
    with pytest.raises(ValueError):
        choose_winner({'baseline': .7})


def test_export_aggregation_requires_exact_chunks_and_valid_scores():
    dataset = SimpleNamespace(chunks=[SimpleNamespace(chunk_id='c:1', recording_id='r:1'),
                                      SimpleNamespace(chunk_id='c:2', recording_id='r:1'),
                                      SimpleNamespace(chunk_id='c:3', recording_id='r:2')],
                              labels_by_recording={'r:1': 0, 'r:2': 1})
    rows = [('c:1', 'r:1', 0, .1), ('c:2', 'r:1', 0, .3), ('c:3', 'r:2', 1, .7)]
    result = aggregate_role(dataset, rows)
    assert result[0] == ('r:1', 0, pytest.approx(.2), 2)
    with pytest.raises(ValueError, match='coverage'):
        aggregate_role(dataset, rows[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        aggregate_role(dataset, rows + [rows[0]])
    with pytest.raises(ValueError):
        aggregate_role(dataset, rows[:2] + [('c:3', 'r:2', 1, float('nan'))])
