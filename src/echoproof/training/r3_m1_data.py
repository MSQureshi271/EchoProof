"""Guarded Protocol B inputs for R3-M1; no test or Chatterbox audio access."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from echoproof.training.datasets import ChunkDataset, CropDataset, ShardWaveformLoader


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class VerifiedWaveformLoader:
    """Verify each decoded waveform before handing it to a dataset."""
    def __init__(self, loader, rows):
        self.loader = loader
        self.rows = {r['recording_id']: (int(r['n_samples']), r['pcm_sha256']) for r in rows}

    def __call__(self, recording_id):
        n, expected = self.rows[recording_id]
        wave = self.loader(recording_id)
        if wave.ndim != 1 or len(wave) != n or not np.isfinite(wave).all():
            raise ValueError(f'invalid waveform: {recording_id}')
        pcm = np.rint(wave * 32768).astype('<i2')
        if hashlib.sha256(pcm.tobytes()).hexdigest() != expected:
            raise ValueError(f'PCM hash mismatch: {recording_id}')
        return wave

    def close(self):
        self.loader.close()


def verify_provenance(paths, manifest, frozen, schema_hash):
    if manifest.get('protocol') != 'B' or not manifest.get('verification', {}).get('passed'):
        raise ValueError('R2-S1 manifest is not verified for B')
    inputs = manifest['inputs']
    if inputs['split_sha256'] != frozen or inputs['feature_schema_hash'] != schema_hash:
        raise ValueError('frozen split or feature schema mismatch')
    expected = {'cohort': inputs['cohort_manifest_file_sha256'], 'split': inputs['split_file_sha256'],
                'chunks': inputs['chunk_manifest_file_sha256'], 'stats': manifest['outputs']['mel_stats']['sha256']}
    actual = {name: file_sha256(paths[name]) for name in expected}
    if actual != expected:
        raise ValueError('input file hash mismatch')
    return actual


def build_data(root, shard_dir, seed=42):
    """Build B training and role-specific validation datasets from frozen artifacts.

    Reads manifest metadata only; waveform access is restricted to eligible IDs.
    The caller must close the returned loader when finished.
    """
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data import cohort, splits
    from echoproof.features.acoustic import FEATURE_SCHEMA_HASH
    from echoproof.features.r2_s1 import plan_b
    from echoproof.features.stats import load_mel_stats

    root = Path(root)
    paths = {'cohort': root / 'data/manifests/cohort_v1.parquet',
             'split': root / 'data/splits/protocol_B_v1.parquet',
             'chunks': root / 'data/processed/chunks_v1.parquet',
             'stats': root / 'data/features/r2_s1_B/mel_stats_B_v1.npz'}
    manifest_path = root / 'data/features/r2_s1_B/r2_s1_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    frozen = json.loads((root / 'reports/r1/split_report.json').read_text(encoding='utf-8'))['protocols']['B']['split_sha256']
    hashes = verify_provenance(paths, manifest, frozen, FEATURE_SCHEMA_HASH)
    assignments = splits.load_assignments(paths['split'], sealed=False)
    if not assignments or {a.split_hash for a in assignments} != {frozen}:
        raise ValueError('assignment split hash mismatch')
    rows = cohort.read_table(paths['cohort'])
    if len({r['recording_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate cohort recording')
    chunks = read_parquet(str(paths['chunks']), Chunk)
    plan = plan_b(rows, assignments, chunks)
    by_id = {r['recording_id']: r for r in rows}
    selected = [by_id[x['recording_id']] for x in plan.train + plan.validation]
    for row in selected:
        if row['component'] not in {'genuine', 'melo', 'vits', 'xtts'} or row['label'] != (row['component'] != 'genuine'):
            raise ValueError('invalid component or label')
    stats = load_mel_stats(paths['stats'])
    if stats['mean'].shape != (64,) or stats['std'].shape != (64,) or stats['recordings'] != len(plan.train):
        raise ValueError('mel statistics shape or training count mismatch')
    roles = {x['recording_id']: x['role'] for x in plan.validation}
    labels = {x['recording_id']: x['label'] for x in plan.validation}
    groups = {x['recording_id']: x['group_id'] for x in plan.validation}
    val_chunks = [c for c in chunks if c.protocol_id == 'protocol:B' and c.partition == 'validation' and c.recording_id in roles]
    if len({c.chunk_id for c in val_chunks}) != len(val_chunks):
        raise ValueError('duplicate validation chunk')
    for c in val_chunks:
        if c.group_id != groups[c.recording_id]:
            raise ValueError('chunk group mismatch')
    base = ShardWaveformLoader(shard_dir, {r['recording_id']: r['canonical_path'] for r in selected})
    loader = VerifiedWaveformLoader(base, selected)
    train = CropDataset([(x['recording_id'], x['label']) for x in plan.train], loader, seed=seed)
    validation = {}
    for role in ('selection', 'calibration', 'operating'):
        role_ids = {rid for rid, value in roles.items() if value == role}
        if not role_ids or {labels[rid] for rid in role_ids} != {0, 1}:
            raise ValueError(f'invalid validation role: {role}')
        validation[role] = ChunkDataset([c for c in val_chunks if c.recording_id in role_ids], labels, loader)
        validation[role].validation_role = role
    return {'train': train, 'validation': validation, 'loader': loader, 'mel_stats': stats,
            'validation_components': {x['recording_id']: x['component'] for x in plan.validation},
            'validation_groups': groups,
            'split_sha256': frozen, 'input_sha256': {**hashes, 'features_manifest': file_sha256(manifest_path)},
            'counts': {'train': len(train), 'validation_recordings': dict(Counter(roles.values())),
                       'validation_chunks': {role: len(ds) for role, ds in validation.items()}}}
