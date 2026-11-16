"""Measure audio properties for the recordings in the completed R5 sealed run."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from echoproof.eda.measure import SETTINGS, measure  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(root, shards, output):
    root, shards, output = map(Path, (root, shards, output))
    run_dir = root / 'reports/final/results/r5_n1_clean_v1'
    integrity_path = run_dir / 'integrity_report.json'
    identity_path = run_dir / 'run_identity.json'
    integrity = json.loads(integrity_path.read_text(encoding='utf-8'))
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    index = shards / 'index.parquet'
    if sha(index) != identity['audio']['index_sha256']:
        raise ValueError('Audio index differs from the sealed run')
    expected = {}
    inputs = {}
    for protocol in ('A', 'B'):
        sets = []
        for model in ('LR', 'BiGRU'):
            entry = integrity['bundles'][f'{protocol}:{model}']
            path = run_dir / entry['predictions_file']
            if not entry['integrity']['passed'] or sha(path) != entry['predictions_sha256']:
                raise ValueError(f'Unverified predictions: {path}')
            inputs[path.name] = sha(path)
            df = pd.read_parquet(path, columns=['recording_id', 'decoded_duration_seconds'])
            if df.recording_id.duplicated().any() or df.decoded_duration_seconds.isna().any():
                raise ValueError(f'Invalid IDs or durations: {path}')
            durations = dict(zip(df.recording_id, df.decoded_duration_seconds))
            sets.append(set(durations))
            for rid, duration in durations.items():
                if rid in expected and abs(expected[rid] - float(duration)) > 1 / 16000:
                    raise ValueError(f'Prediction duration disagreement: {rid}')
                expected[rid] = float(duration)
        if sets[0] != sets[1]:
            raise ValueError(f'{protocol} model recording sets differ')
    if output.exists():
        raise FileExistsError(f'Output exists; inspect it: {output}')
    index_df = pq.read_table(index, columns=['recording_id', 'shard']).to_pandas()
    if index_df.recording_id.duplicated().any():
        raise ValueError('Duplicate index IDs')
    lookup = dict(zip(index_df.recording_id, index_df.shard))
    if not set(expected) <= lookup.keys():
        raise ValueError('Some prediction IDs are missing from audio index')
    by_shard = {}
    for rid in sorted(expected):
        by_shard.setdefault(lookup[rid], []).append(rid)
    rows = []
    for shard, ids in sorted(by_shard.items()):
        print(f'Measuring {shard}: {len(ids)} recordings', flush=True)
        with tarfile.open(shards / shard, 'r') as tar:
            members = {m.name: m for m in tar if m.isfile()}
            for rid in ids:
                try:
                    member = members[rid.replace(':', '__') + '.flac']
                    with tar.extractfile(member) as f:
                        samples, rate = sf.read(io.BytesIO(f.read()), dtype='float32')
                    if rate != 16000 or samples.ndim != 1:
                        raise ValueError('expected canonical 16 kHz mono audio')
                    if abs(len(samples) / rate - expected[rid]) > 1 / rate:
                        raise ValueError('duration differs from sealed prediction')
                    values, _ = measure(samples)
                    if not all(np.isfinite(v) for v in values.values()):
                        raise ValueError('non-finite measurement')
                    rows.append({'recording_id': rid, 'status': 'success', 'error': None, **values})
                except Exception as exc:
                    rows.append({'recording_id': rid, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
    output.mkdir(parents=True, exist_ok=False)
    table = output / 'recording_quality.parquet'
    pd.DataFrame(rows).to_parquet(table, index=False)
    failures = sum(r['status'] != 'success' for r in rows)
    report = {'task': 'R5-M1', 'scope': 'unique recordings in saved A/B LR and BiGRU predictions; no model inference',
              'recordings': len(rows), 'successes': len(rows) - failures, 'failures': failures,
              'prediction_sha256': inputs, 'integrity_sha256': sha(integrity_path), 'identity_sha256': sha(identity_path),
              'index_sha256': sha(index), 'measurement_source_sha256': sha(ROOT / 'src/echoproof/eda/measure.py'),
              'script_sha256': sha(Path(__file__)), 'settings': SETTINGS, 'output_sha256': sha(table),
              'numpy': np.__version__, 'soundfile': sf.__version__, 'pandas': pd.__version__}
    (output / 'measurement_report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise RuntimeError('Some recordings failed; preserve output and inspect failures before analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shards', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.shards, args.output)
