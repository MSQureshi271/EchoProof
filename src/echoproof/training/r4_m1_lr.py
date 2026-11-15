"""One fixed-recipe Protocol A LR fit and selection-role evaluation."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from echoproof.evaluation.aggregate import aggregate_recordings
from echoproof.evaluation.metrics import compute_metrics
from echoproof.models.lr_detector import LRDetector
from echoproof.training.r4_m1_data import build_data, file_sha256


def fit_selection(train, validation, parameters, split_sha256):
    """Fit only on train; score only selection. No other validation rows are scored."""
    mask = validation['role'] == 'selection'
    if not np.any(mask):
        raise ValueError('selection role is empty')
    selection = {k: v[mask] for k, v in validation.items()
                 if getattr(v, 'ndim', 0) and len(v) == len(mask) and k != 'feature_names'}
    model = LRDetector.fit(train['X'], train['y'], train['recording_id'],
                           **parameters, extra={'protocol': 'A', 'split_sha256': split_sha256})
    if not model.meta['diagnostics']['converged']:
        raise RuntimeError('LR fit did not converge; no model was selected')
    scores = model.predict_proba_features(selection['X'])
    rec = aggregate_recordings(selection['chunk_id'], selection['recording_id'], selection['y'], scores)
    return model, selection, scores, rec, compute_metrics(rec.labels, rec.scores, 0.5)


def run(root, shard_dir, output):
    """Run after complete A preflight, writing only into a new R4 run directory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, output = Path(root).resolve(), Path(output).resolve()
    run_root = (root/'runs/r4').resolve()
    if output.parent != run_root:
        raise ValueError('output must be a new direct child of runs/r4')
    if output.exists():
        raise FileExistsError('run directory already exists; refusing to overwrite')
    data = build_data(root, shard_dir)
    try:
        started = time.perf_counter()
        output.mkdir(parents=True, exist_ok=False)
        recipe = data['recipe']
        try:
            model, val, scores, rec, metrics = fit_selection(data['lr_train'], data['lr_validation'],
                                                              recipe['lr']['hyperparameters'], data['split_sha256'])
            model.save(output/'model')
            loaded = LRDetector.load(output/'model')
            parity = float(np.max(np.abs(scores - loaded.predict_proba_features(val['X']))))
            if parity > 1e-12:
                raise RuntimeError('LR save/reload parity failed')
            chunk_rows = [{'chunk_id': str(cid), 'recording_id': str(rid), 'group_id': str(group),
                           'component': str(component), 'label': int(y), 'role': 'selection', 'raw_score': float(score)}
                          for cid, rid, group, component, y, score in zip(val['chunk_id'], val['recording_id'], val['group_id'], val['component'], val['y'], scores)]
            metadata = {str(rid): (str(group), str(component)) for rid, group, component in zip(val['recording_id'], val['group_id'], val['component'])}
            rec_rows = [{'recording_id': rid, 'group_id': metadata[rid][0], 'component': metadata[rid][1],
                         'label': int(label), 'role': 'selection', 'raw_score': float(score), 'chunk_count': int(count)}
                        for rid, label, score, count in zip(rec.recording_ids, rec.labels, rec.scores, rec.chunk_counts)]
            pq.write_table(pa.Table.from_pylist(chunk_rows), output/'predictions_chunks_selection.parquet')
            pq.write_table(pa.Table.from_pylist(rec_rows), output/'predictions_recordings_selection.parquet')
            with (output/'coefficients.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f); writer.writerow(['feature', 'standardized_coefficient']); writer.writerows(model.coefficient_table())
            files = {str(p.relative_to(output)).replace('\\', '/'): file_sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
            evidence = {'task': 'R4-M1-LR', 'status': 'complete', 'protocol': 'A', 'created_utc': datetime.now(timezone.utc).isoformat(),
                        'recipe_sha256': file_sha256(root/'configs/r4_m1_recipes.json'), 'split_sha256': data['split_sha256'],
                        'source_sha256': {'src/echoproof/training/r4_m1_lr.py': file_sha256(Path(__file__)),
                                          'src/echoproof/training/r4_m1_data.py': file_sha256(root/'src/echoproof/training/r4_m1_data.py'),
                                          'src/echoproof/models/lr_detector.py': file_sha256(root/'src/echoproof/models/lr_detector.py')},
                        'input_sha256': data['input_sha256'], 'hyperparameters': recipe['lr']['hyperparameters'],
                        'fit_diagnostics': model.meta['diagnostics'], 'selection_metrics': metrics,
                        'selection_recordings': len(rec.recording_ids), 'selection_chunks': len(scores),
                        'reload_max_abs': parity, 'reload_tolerance': 1e-12,
                        'elapsed_seconds': round(time.perf_counter()-started, 3),
                        'scope': 'One new A fit; train-only preprocessing and selection-role evaluation; calibration and operating predictions reserved for later export; no test audio or features.',
                        'outputs': files}
            (output/'evidence.json').write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
            return evidence
        except Exception as exc:
            (output/'failure.json').write_text(json.dumps({'task': 'R4-M1-LR', 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}, indent=2)+'\n', encoding='utf-8')
            raise
    finally:
        data['loader'].close()
