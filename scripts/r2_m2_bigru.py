"""R2-M2 disposable BiGRU overfit and padding diagnostics on B-training audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from echoproof.data import cohort, splits
from echoproof.data.chunks import eval_windows
from echoproof.features.acoustic import log_mel, FEATURE_SCHEMA_HASH
from echoproof.features.stats import load_mel_stats, normalize_log_mel
from echoproof.models.bigru import BiGRU
from echoproof.training.datasets import ShardWaveformLoader
from echoproof.training.seeding import seed_everything


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_rows(rows, assignments, count=32, seed=42):
    if count < 32 or count > 64 or count % 2:
        raise ValueError('count must be even and between 32 and 64')
    allowed = {a.recording_id for a in assignments if a.protocol_id == 'protocol:B' and a.partition == 'train'}
    groups = {0: [], 1: []}
    for row in rows:
        if row['recording_id'] in allowed and row['retained']:
            if row['component'] not in ('genuine', 'melo', 'vits') or row['label'] != (row['component'] != 'genuine'):
                raise ValueError('invalid B-training component or label')
            groups[row['label']].append(row)
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        candidates = sorted(groups[label], key=lambda r: r['recording_id'])
        if len(candidates) < count // 2:
            raise ValueError('insufficient B-training recordings')
        chosen.extend(candidates[i] for i in rng.choice(len(candidates), count // 2, replace=False))
    return sorted(chosen, key=lambda r: r['recording_id'])


def pad_sequences(sequences):
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    values = torch.zeros((len(sequences), int(lengths.max()), 64), dtype=torch.float32)
    for i, seq in enumerate(sequences):
        values[i, :len(seq)] = torch.as_tensor(seq)
    mask = torch.arange(values.shape[1])[None, :] < lengths[:, None]
    if not torch.equal(mask.sum(1), lengths):
        raise ValueError('frame mask does not match sequence lengths')
    return values, lengths


def run(root, shard_dir, report_path, count=32, seed=42, epochs=100, target=0.95, device='auto'):
    root, shard_dir, report_path = Path(root), Path(shard_dir), Path(report_path)
    if epochs < 1 or not 0 < target <= 1:
        raise ValueError('epochs and target must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if report_path.exists():
        raise FileExistsError(report_path)
    inputs = {'cohort': root / 'data/manifests/cohort_v1.parquet', 'split': root / 'data/splits/protocol_B_v1.parquet',
              'features': root / 'data/features/r2_s1_B/r2_s1_manifest.json', 'stats': root / 'data/features/r2_s1_B/mel_stats_B_v1.npz'}
    manifest = json.loads(inputs['features'].read_text(encoding='utf-8'))
    frozen = json.loads((root / 'reports/r1/split_report.json').read_text(encoding='utf-8'))['protocols']['B']['split_sha256']
    if manifest.get('protocol') != 'B' or not manifest.get('verification', {}).get('passed') or manifest['inputs']['split_sha256'] != frozen:
        raise ValueError('R2-S1 provenance or split is not verified')
    if sha(inputs['stats']) != manifest['outputs']['mel_stats']['sha256'] or manifest['inputs']['feature_schema_hash'] != FEATURE_SCHEMA_HASH:
        raise ValueError('mel statistics or feature schema mismatch')
    assignments = splits.load_assignments(inputs['split'], sealed=False)
    if {a.split_hash for a in assignments} != {frozen}:
        raise ValueError('assignment split hash mismatch')
    rows = cohort.read_table(inputs['cohort'])
    if sha(inputs['cohort']) != manifest['inputs']['cohort_manifest_file_sha256'] or sha(inputs['split']) != manifest['inputs']['split_file_sha256']:
        raise ValueError('input manifest file hash mismatch')
    chosen = select_rows(rows, assignments, count, seed)
    stats = load_mel_stats(inputs['stats'])
    loader = ShardWaveformLoader(shard_dir, {r['recording_id']: r['canonical_path'] for r in chosen})
    sequences, labels, lengths_samples = [], [], []
    try:
        for row in chosen:
            wave = loader(row['recording_id'])
            pcm = np.round(wave * 32768).astype('<i2')
            if hashlib.sha256(pcm.tobytes()).hexdigest() != row['pcm_sha256']:
                raise ValueError(f"PCM hash mismatch: {row['recording_id']}")
            start, end = eval_windows(len(wave))[0]
            clip = wave[start:end]
            mel = log_mel(clip)
            expected = 1 + (max(len(clip), 512) - 512) // 160
            if len(mel) != expected:
                raise ValueError('unexpected valid frame length')
            sequences.append(normalize_log_mel(mel, stats['mean'], stats['std']))
            labels.append(int(row['label'])); lengths_samples.append(len(clip))
    finally:
        loader.close()
    x, lengths = pad_sequences(sequences)
    mask_sums = (torch.arange(x.shape[1])[None, :] < lengths[:, None]).sum(1).tolist()
    device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)
    seed_everything(seed)
    model = BiGRU().to(device)
    model.eval()
    # A shorter valid sequence and a longer neighbour exercise packed batching and padded masks.
    short = sequences[0][:max(1, len(sequences[0]) // 2)]
    alone, la = pad_sequences([short]); paired, lp = pad_sequences([short, max(sequences, key=len)])
    with torch.no_grad():
        first = model(alone.to(device), la)[0]
        together = model(paired.to(device), lp)[0]
    difference = float((first - together).abs().max().cpu())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    x = x.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x, lengths), labels)
        if not torch.isfinite(loss):
            raise ValueError('non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, lengths)
            accuracy = float((logits.argmax(1) == labels).float().mean().cpu())
        history.append({'epoch': epoch, 'loss': float(loss.detach().cpu()), 'accuracy': accuracy})
        if accuracy >= target:
            break
    report = {'task': 'R2-M2', 'scope': 'disposable B training diagnostics only', 'seed': seed,
              'verified': bool(accuracy >= target and difference <= 1e-5), 'target_accuracy': target,
              'training_accuracy': accuracy, 'epochs': len(history), 'history': history,
              'padding_max_abs_logit_difference': difference, 'padding_tolerance': 1e-5,
              'recordings': [{'recording_id': r['recording_id'], 'label': int(r['label']), 'component': r['component']} for r in chosen],
              'sample_lengths': lengths_samples, 'frame_lengths': lengths.tolist(), 'mask_sums': mask_sums,
              'split_sha256': frozen, 'input_sha256': {k: sha(v) for k, v in inputs.items()},
              'source_sha256': {str(p.relative_to(root)): sha(p) for p in [root / 'scripts/r2_m2_bigru.py', root / 'src/echoproof/models/bigru.py', root / 'src/echoproof/features/acoustic.py']},
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': str(device),
                              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
              'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'note': 'Training-only memorization check; no checkpoint is saved and no generalization is claimed.'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--shard-dir', required=True); p.add_argument('--report', required=True)
    p.add_argument('--count', type=int, default=32); p.add_argument('--epochs', type=int, default=100); p.add_argument('--device', default='auto')
    a = p.parse_args()
    result = run(a.root, a.shard_dir, a.report, count=a.count, epochs=a.epochs, device=a.device)
    print(json.dumps({k: result[k] for k in ('verified', 'training_accuracy', 'epochs', 'padding_max_abs_logit_difference')}, indent=2))
    sys.exit(0 if result['verified'] else 1)
