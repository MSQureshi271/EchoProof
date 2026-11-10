"""Read-only comparison of LR adapter, original model and saved selection features."""
from pathlib import Path
import argparse
import csv
import json
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
from echoproof.features.acoustic import FEATURE_NAMES, handcrafted
from echoproof.models.lr_adapter import LRBundleAdapter
from echoproof.models.lr_detector import LRDetector
from echoproof.training.r3_m1_data import build_data
from r2_m1_lr import load_inputs


def run(shards, features, bundle):
    _, _, val = load_inputs(features)
    by_id = {str(cid): i for i, cid in enumerate(val['chunk_id'])}
    reference = ROOT / 'runs/r2/r2_m1_20261106T140611Z'
    with (reference / 'predictions_chunks.csv').open(newline='', encoding='utf-8') as f:
        saved = {r['chunk_id']: r for r in csv.DictReader(f)}
    model = LRDetector.load(reference / 'model')
    adapter = LRBundleAdapter.load(bundle)
    data = build_data(ROOT, shards)
    try:
        ds = data['validation']['selection']
        chosen, groups, components = set(), set(), set()
        for i, c in sorted(enumerate(ds.chunks), key=lambda pair: pair[1].chunk_id):
            group = data['validation_groups'][c.recording_id]
            component = data['validation_components'][c.recording_id]
            if group not in groups or component not in components:
                chosen.add(i); groups.add(group); components.add(component)
        chosen.add(min(range(len(ds)), key=lambda i: ds.chunks[i].valid_samples))
        for i in sorted(range(len(ds)), key=lambda i: ds.chunks[i].chunk_id):
            if len(chosen) >= 50:
                break
            chosen.add(i)
        results = []
        for i in sorted(chosen):
            item = ds[i]
            cid = item['chunk_id']
            row = by_id[cid]
            if str(val['recording_id'][row]) != item['recording_id'] or int(val['y'][row]) != item['label']:
                raise ValueError('cached feature identity differs from selection chunk')
            wave = item['waveform'][:item['valid_samples']]
            direct = handcrafted(wave)
            cached = val['X'][row]
            differences = np.abs(direct - cached)
            finite = np.isfinite(differences)
            max_column = int(np.argmax(np.where(finite, differences, -1)))
            original = float(model.score_chunks([wave])[0])
            bundled = float(adapter.score_chunks([wave])[0])
            cached_score = float(model.predict_proba_features(cached)[0])
            saved_score = float(saved[cid]['raw_score'])
            results.append({'chunk_id': cid, 'adapter_vs_original': abs(bundled-original),
                            'original_vs_saved': abs(original-saved_score),
                            'cached_vs_saved': abs(cached_score-saved_score),
                            'feature_max_abs': float(differences[max_column]),
                            'feature_max_name': FEATURE_NAMES[max_column],
                            'nan_mask_equal': bool(np.array_equal(np.isnan(direct), np.isnan(cached))),
                            'classification_changed_at_0_5': bool((bundled >= .5) != (saved_score >= .5))})
        return {'chunks_checked': len(results),
                'environment': {n: version(n) for n in ('numpy', 'scipy', 'librosa', 'scikit-learn', 'soundfile')},
                'max_adapter_vs_original': max(x['adapter_vs_original'] for x in results),
                'max_original_vs_saved': max(x['original_vs_saved'] for x in results),
                'max_cached_vs_saved': max(x['cached_vs_saved'] for x in results),
                'all_nan_masks_equal': all(x['nan_mask_equal'] for x in results),
                'chunk_classifications_changed_at_0_5': sum(x['classification_changed_at_0_5'] for x in results),
                'largest_differences': sorted(results, key=lambda x: -x['original_vs_saved'])[:5]}
    finally:
        data['loader'].close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.shards, args.features, args.bundle), indent=2, allow_nan=False))
