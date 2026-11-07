"""Selection-only aggregation and metrics for R3-M1."""
from echoproof.evaluation.aggregate import aggregate_recordings
from echoproof.evaluation.metrics import compute_metrics


class SelectionAggregation:
    """Check exact selection chunk identities, labels and mean recording scores."""
    def __init__(self, dataset):
        if getattr(dataset, 'validation_role', None) != 'selection':
            raise ValueError('checkpoint selection requires the selection validation role')
        self.expected = {c.chunk_id: (c.recording_id, int(dataset.labels_by_recording[c.recording_id]))
                         for c in dataset.chunks}
        if len(self.expected) != len(dataset.chunks) or not self.expected:
            raise ValueError('empty or duplicate selection chunks')

    def __call__(self, chunk_rows):
        seen = set()
        for cid, rid, label, _ in chunk_rows:
            if cid in seen or self.expected.get(cid) != (rid, label):
                raise ValueError('duplicate, unexpected or mislabelled selection chunk')
            seen.add(cid)
        if seen != set(self.expected):
            raise ValueError('selection chunk coverage mismatch')
        result = aggregate_recordings([r[0] for r in chunk_rows], [r[1] for r in chunk_rows],
                                      [r[2] for r in chunk_rows], [r[3] for r in chunk_rows],
                                      expected_recording_ids={r[0] for r in self.expected.values()})
        return list(zip(result.recording_ids, result.labels, result.scores, result.chunk_counts))


def selection_metrics(labels, scores, threshold):
    if threshold != 0.5:
        raise ValueError('R3-M1 checkpoint selection threshold must be 0.5')
    return compute_metrics(labels, scores, threshold)


def selection_trainer(model, config, train_dataset, selection_dataset, run_dir, **kwargs):
    """Construct a trainer with the required selection-only aggregation and metric."""
    from echoproof.training.trainer import Trainer
    if config.selection_threshold != 0.5:
        raise ValueError('R3-M1 checkpoint selection threshold must be 0.5')
    if 'metrics_fn' in kwargs or 'aggregation_fn' in kwargs:
        raise ValueError('R3-M1 selection functions cannot be overridden')
    return Trainer(model, config, train_dataset, selection_dataset, run_dir,
                   metrics_fn=selection_metrics, aggregation_fn=SelectionAggregation(selection_dataset), **kwargs)


def selection_panels(recordings, components):
    """Report selection panels only; these do not override full-selection checkpoint choice."""
    panels = {}
    for name, allowed in [('known', {'genuine', 'melo', 'vits'}), ('unseen_xtts', {'genuine', 'xtts'})]:
        rows = [r for r in recordings if components[r[0]] in allowed]
        panels[name] = selection_metrics([r[1] for r in rows], [r[2] for r in rows], 0.5)
    return panels
