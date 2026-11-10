"""Package selected models into new development bundles and verify selection-only parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from echoproof.artifacts import file_digest
from echoproof.bundle import load_bundle
from echoproof.models.bigru_adapter import BiGRUBundleAdapter
from echoproof.models.lr_adapter import LRBundleAdapter
from echoproof.models.r3_m2_bundle import package_bigru, package_lr


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def run(output, shard_dir, device):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from echoproof.evaluation.aggregate import aggregate_recordings
    from echoproof.evaluation.metrics import compute_metrics
    from echoproof.training.r3_m1_data import build_data
    from echoproof.models.lr_detector import LRDetector
    from r2_m1_lr import load_inputs

    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'runs/r3').resolve()) or not output.name.startswith('r3_m2_'):
        raise ValueError('use a new runs/r3/r3_m2_* staging directory; locked bundles are not output targets')
    if output.exists():
        raise FileExistsError('staging directory exists; inspect its evidence rather than overwriting it')
    exp = ROOT/'experiments/r3_m1'
    selected = read(exp/'reports/final_selection.json')
    expected_sha = '435d47b36431e7f0f6b6f1bdd2459c8a6d06074b79fcc7ff69631c0c93cc687e'
    if selected['id'] != 'b01_lower_lr_full20' or selected['best_epoch'] != 16 or selected['checkpoint_sha256'] != expected_sha:
        raise ValueError('selection differs from the 9 November decision; review packaging definition first')
    checkpoint = (ROOT/selected['checkpoint']).resolve()
    if not checkpoint.is_relative_to(ROOT):
        raise ValueError('checkpoint must be inside project')
    resolved = read(checkpoint.parent/'resolved_config.json')
    reference = ROOT/'runs/r2/r2_m1_20261106T140611Z'
    lr_evidence = read(reference/'evidence.json')
    handoff = read(exp/'reports/final_handoff.json')
    if handoff['selected'] != selected:
        raise ValueError('selected checkpoint and handoff disagree')
    for name, sha in handoff['outputs'].items():
        if file_digest(exp/'runs/batch01/selected'/name) != sha:
            raise ValueError('handoff predictions changed')
    data = build_data(ROOT, shard_dir)
    try:
        if data['input_sha256'] != read(exp/'configs/batch01_resolved.json')['inputs']:
            raise ValueError('parity input definitions differ from selected training run')
        output.mkdir(parents=True)
        package_lr(reference/'model', output/'lr', bundle_id='bundle:r3-m2-lr-development',
                   expected_hashes={name: lr_evidence['outputs']['model/'+name] for name in ('weights.npz', 'lr_model.json')})
        package_bigru(checkpoint, output/'bigru', bundle_id='bundle:r3-m2-bigru-development',
                      architecture={'hidden_size': 128, 'num_layers': 2, 'bidirectional': True, 'dropout': .3},
                      protocol='B', expected_checkpoint_sha256=expected_sha, split_sha256=data['split_sha256'],
                      expected_epoch=16, expected_config_hash=resolved['config_hash'])
        gru = BiGRUBundleAdapter.load(output/'bigru', device=device)
        lr = LRBundleAdapter.load(output/'lr')
        original_lr = LRDetector.load(reference/'model')
        feature_dir = ROOT/'data/features/r2_s1_B'
        lr_manifest, _, lr_selection = load_inputs(feature_dir)
        if (file_digest(feature_dir/'r2_s1_manifest.json') != lr_evidence['input_manifest_sha256']
                or lr_manifest['outputs'] != lr_evidence['input_files']):
            raise ValueError('cached LR features differ from original R2-M1 inputs')
        cached_indices = {str(cid): i for i, cid in enumerate(lr_selection['chunk_id'])}
        dataset = data['validation']['selection']
        reference_rows = pq.read_table(exp/'runs/batch01/selected/predictions_chunks_selection.parquet').to_pylist()
        expected = {r['chunk_id']: r for r in reference_rows}
        if len(expected) != len(reference_rows) or {c.chunk_id for c in dataset.chunks} != set(expected):
            raise ValueError('selection coverage mismatch')
        # LR parity samples: every selection speaker and component, shortest tail,
        # then deterministic chunk IDs up to 50. Never compute metrics on this sample.
        chosen, groups, components = set(), set(), set()
        for i, c in sorted(enumerate(dataset.chunks), key=lambda pair: pair[1].chunk_id):
            group, component = data['validation_groups'][c.recording_id], data['validation_components'][c.recording_id]
            if group not in groups or component not in components:
                chosen.add(i); groups.add(group); components.add(component)
        chosen.add(min(range(len(dataset)), key=lambda i: dataset.chunks[i].valid_samples))
        for i in sorted(range(len(dataset)), key=lambda i: dataset.chunks[i].chunk_id):
            if len(chosen) >= 50:
                break
            chosen.add(i)
        import csv
        with (reference/'predictions_chunks.csv').open(newline='', encoding='utf-8') as f:
            lr_expected = {r['chunk_id']: r for r in csv.DictReader(f)}
        actual_rows, lr_max, cached_max, extraction_drift, threshold_changes, gru_max, lr_checked = [], 0.0, 0.0, 0.0, 0, 0.0, 0
        for start in range(0, len(dataset), 64):
            items = [dataset[i] for i in range(start, min(start+64, len(dataset)))]
            waves = [x['waveform'][:x['valid_samples']] for x in items]
            scores = gru.score_chunks(waves)
            for j, (item, value) in enumerate(zip(items, scores)):
                saved = expected[item['chunk_id']]
                if saved['recording_id'] != item['recording_id'] or saved['label'] != item['label']:
                    raise ValueError('selection identity/label mismatch')
                gru_max = max(gru_max, abs(float(value)-saved['tts_probability']))
                actual_rows.append({'chunk_id': item['chunk_id'], 'recording_id': item['recording_id'],
                                    'label': item['label'], 'tts_probability': float(value)})
                if start+j in chosen:
                    r = lr_expected[item['chunk_id']]
                    if r['recording_id'] != item['recording_id'] or int(r['label']) != item['label']:
                        raise ValueError('LR reference identity differs')
                    idx = cached_indices[item['chunk_id']]
                    if str(lr_selection['recording_id'][idx]) != item['recording_id'] or int(lr_selection['y'][idx]) != item['label']:
                        raise ValueError('cached LR selection identity differs')
                    bundled_score = float(lr.score_chunks([waves[j]])[0])
                    direct_score = float(original_lr.score_chunks([waves[j]])[0])
                    cached_score = float(original_lr.predict_proba_features(lr_selection['X'][idx])[0])
                    saved_score = float(r['raw_score'])
                    lr_max = max(lr_max, abs(bundled_score-direct_score))
                    cached_max = max(cached_max, abs(cached_score-saved_score))
                    extraction_drift = max(extraction_drift, abs(direct_score-saved_score))
                    threshold_changes += (direct_score >= .5) != (saved_score >= .5)
                    lr_checked += 1
        if gru_max > 1e-5 or lr_max > 1e-12 or cached_max > 1e-12:
            raise ValueError(f'parity failed: BiGRU={gru_max}, LR same-waveform={lr_max}, LR cached={cached_max}')
        a = aggregate_recordings([r['chunk_id'] for r in actual_rows], [r['recording_id'] for r in actual_rows],
                                 [r['label'] for r in actual_rows], [r['tts_probability'] for r in actual_rows])
        metrics = compute_metrics(a.labels, a.scores, .5)
        if abs(metrics['macro_f1']-selected['macro_f1']) > 1e-12:
            raise ValueError('recording selection F1 changed after packaging')
        pq.write_table(pa.Table.from_pylist(actual_rows), output/'bigru_selection_parity.parquet')
        report = {'task': 'R3-M2 adapter packaging', 'status': 'development_bundle_parity_passed',
                  'selected_checkpoint_sha256': expected_sha, 'selection_decision_sha256': file_digest(exp/'reports/final_selection.json'),
                  'device': device, 'bigru_chunks': len(actual_rows), 'bigru_recordings': len(a.recording_ids),
                  'bigru_max_abs': gru_max, 'bigru_tolerance': 1e-5, 'bigru_selection_metrics': metrics,
                  'lr_chunks_checked': lr_checked, 'lr_max_abs': lr_max, 'lr_tolerance': 1e-12,
                  'lr_cached_vs_saved_max_abs': cached_max,
                  'lr_reextracted_vs_saved_max_abs': extraction_drift,
                  'lr_reextracted_chunk_threshold_changes_at_0_5': threshold_changes,
                  'lr_reextraction_note': 'Diagnostic sample only; feature extraction drift is reported separately from adapter parity.',
                  'review_status': 'pending', 'calibration': 'not performed; R3-N1 required',
                  'official_lock_or_bundle_replaced': False,
                  'bundle_metadata': {name: file_digest(output/name/'bundle.json') for name in ('lr', 'bigru')},
                  'artifacts': {p.relative_to(output).as_posix(): file_digest(p) for p in output.rglob('*') if p.is_file()}}
        (output/'evidence.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        return report
    finally:
        data['loader'].close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--shard-dir', required=True, type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = p.parse_args()
    print(json.dumps(run(args.output, args.shard_dir, args.device), indent=2))
