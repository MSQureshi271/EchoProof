"""Verified Protocol A inputs for R4-M1; no test waveform or feature access."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH
from echoproof.features.stats import load_mel_stats
from echoproof.training.datasets import ChunkDataset, CropDataset, ShardWaveformLoader
from echoproof.training.r3_m1_data import VerifiedWaveformLoader, file_sha256

ROLES = ('selection', 'calibration', 'operating')
COMPONENTS = {'genuine', 'melo', 'vits', 'xtts', 'chatterbox'}


def canonical_json_sha256(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def verify_recipe(root, recipe):
    root = Path(root)
    if recipe.get('task') != 'R4-M1' or recipe.get('protocol') != 'A':
        raise ValueError('wrong recipe task or protocol')
    item = recipe['b_lock']
    lock_path = root/item['path']
    if canonical_json_sha256(lock_path) != item['canonical_json_sha256']:
        raise ValueError('b_lock canonical content checksum mismatch')
    report = recipe['r4_s1_report']
    if file_sha256(root/report['path']) != report['sha256']:
        raise ValueError('r4_s1_report checksum mismatch')
    lock = json.loads((root/recipe['b_lock']['path']).read_text(encoding='utf-8'))
    if lock.get('lock_id') != recipe['b_lock']['lock_id'] or lock.get('protocol') != 'B':
        raise ValueError('wrong GR3 lock identity')
    for model in ('LR', 'BiGRU'):
        candidate = lock['candidates'][model]
        bundle = root/candidate['bundle_path']
        if file_sha256(bundle/'bundle.json') != candidate['bundle_json_sha256']:
            raise ValueError(f'{model} locked bundle mismatch')
        for name, details in candidate['components'].items():
            if file_sha256(bundle/name) != details['sha256']:
                raise ValueError(f'{model} locked component mismatch: {name}')
    if file_sha256(root/recipe['bigru']['source_config']) != recipe['bigru']['source_config_file_sha256']:
        raise ValueError('selected BiGRU config checksum mismatch')
    source = json.loads((root/recipe['bigru']['source_config']).read_text(encoding='utf-8'))
    if source['config_hash'] != recipe['bigru']['source_config_hash']:
        raise ValueError('selected BiGRU config identity mismatch')
    for key, value in recipe['bigru']['training'].items():
        if key not in ('optimizer', 'loss') and source['config'].get(key) != value:
            raise ValueError(f'BiGRU recipe differs from B: {key}')
    lr = json.loads((root/lock['candidates']['LR']['bundle_path']/'model/lr_model.json').read_text(encoding='utf-8'))
    if lr['hyperparameters'] != recipe['lr']['hyperparameters']:
        raise ValueError('LR recipe differs from B')
    return lock


def validate_table(table, *, kind, plan, plan_sha):
    """Check every feature row against its independently rebuilt A plan."""
    train = kind == 'train'
    if kind not in ('train', 'validation'):
        raise ValueError('unknown table kind')
    id_key = 'crop_id' if train else 'chunk_id'
    fields = (id_key, 'recording_id', 'group_id', 'component', 'y') + (() if train else ('role',))
    n = len(table['y'])
    if table['X'].shape != (n, len(FEATURE_NAMES)) or any(len(table[k]) != n for k in fields):
        raise ValueError('feature shape mismatch')
    if table['feature_names'].tolist() != list(FEATURE_NAMES) or str(table['feature_schema_hash']) != FEATURE_SCHEMA_HASH or str(table['plan_sha256']) != plan_sha:
        raise ValueError('feature schema or plan mismatch')
    if np.isinf(table['X']).any(): raise ValueError('infinite feature')
    bad = set(np.flatnonzero(np.isnan(table['X']).any(axis=0))) - {FEATURE_NAMES.index('voiced_f0_mean'), FEATURE_NAMES.index('voiced_f0_std')}
    if bad: raise ValueError('unexpected missing feature')
    expected = {}
    if train:
        for row in plan.crop_bank:
            if row['crop_id'] in expected: raise ValueError('duplicate plan crop')
            expected[row['crop_id']] = (row['recording_id'], row['group_id'], row['component'], int(row['label']))
    else:
        for row in plan.validation:
            for chunk, _, _ in row['chunks']:
                if chunk in expected: raise ValueError('duplicate plan chunk')
                expected[chunk] = (row['recording_id'], row['group_id'], row['component'], int(row['label']), row['role'])
    seen = set()
    for i in range(n):
        key = str(table[id_key][i])
        if key in seen: raise ValueError('duplicate crop or chunk')
        seen.add(key)
        got = (str(table['recording_id'][i]), str(table['group_id'][i]), str(table['component'][i]), int(table['y'][i]))
        if not train: got += (str(table['role'][i]),)
        if got != expected.get(key) or got[2] not in COMPONENTS or got[3] != (got[2] != 'genuine'):
            raise ValueError('feature row does not match A plan')
    if seen != set(expected): raise ValueError('feature coverage mismatch')


def check_groups(plan):
    groups = {}
    for role, rows in [('train', plan.train)] + [(r, [x for x in plan.validation if x['role'] == r]) for r in ROLES]:
        if not rows or {x['label'] for x in rows} != {0, 1}:
            raise ValueError(f'missing role or class: {role}')
        for row in rows:
            group = row['group_id']
            if group in groups and groups[group] != role: raise ValueError('speaker crosses partition or role')
            groups[group] = role


def build_data(root, shard_dir, recipe_path=None):
    """Verify provenance and construct A datasets. Caller must close returned loader."""
    from echoproof.contracts import Chunk, read_parquet
    from echoproof.data import cohort, splits
    from echoproof.features.r4_s1 import plan_a

    root = Path(root).resolve(); shard_dir = Path(shard_dir).resolve()
    recipe_path = Path(recipe_path) if recipe_path else root/'configs/r4_m1_recipes.json'
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    verify_recipe(root, recipe)
    inp = recipe['inputs']; feature_dir = root/inp['feature_directory']
    paths = {key: feature_dir/inp[key]['file'] for key in ('manifest', 'train_features', 'validation_features', 'crop_bank', 'mel_stats')}
    for key, path in paths.items():
        if file_sha256(path) != inp[key]['sha256']: raise ValueError(f'{key} checksum mismatch')
    for key in ('split', 'cohort', 'chunk'):
        if file_sha256(root/inp[f'{key}_file']) != inp[f'{key}_file_sha256' if key == 'split' else f'{key}_sha256']:
            raise ValueError(f'{key} checksum mismatch')
    manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
    if manifest.get('protocol') != 'A' or not manifest.get('verification', {}).get('passed'):
        raise ValueError('A feature manifest is not verified')
    if manifest['inputs']['split_sha256'] != inp['split_sha256'] or manifest['inputs']['feature_schema_hash'] != FEATURE_SCHEMA_HASH or inp['feature_schema_hash'] != FEATURE_SCHEMA_HASH:
        raise ValueError('split or feature schema mismatch')
    if manifest['inputs']['preprocessing_hash'] != inp['preprocessing_hash']:
        raise ValueError('preprocessing mismatch')
    assignments = splits.load_assignments(root/inp['split_file'], sealed=False)
    if not assignments or {a.split_hash for a in assignments} != {inp['split_sha256']}:
        raise ValueError('A split identity mismatch')
    rows = cohort.read_table(root/inp['cohort_file'])
    if len({r['recording_id'] for r in rows}) != len(rows): raise ValueError('duplicate cohort recording')
    chunks = read_parquet(str(root/inp['chunk_file']), Chunk)
    plan = plan_a(rows, assignments, chunks)
    if plan.sha256 != manifest['plan_sha256']:
        raise ValueError('rebuilt A plan hash mismatch')
    check_groups(plan)
    expected = inp['expected']
    for name in ('train_recordings', 'train_crops', 'validation_recordings', 'validation_chunks'):
        if plan.counts[name] != expected[name]: raise ValueError(f'{name} coverage mismatch')
    for role in ROLES:
        if plan.counts['validation_chunks_by_role'].get(role) != expected[f'{role}_chunks']: raise ValueError(f'{role} coverage mismatch')
    bank = cohort.read_table(paths['crop_bank'])
    if bank != plan.crop_bank: raise ValueError('crop bank differs from A plan')
    tables = {}
    for kind, key in [('train', 'train_features'), ('validation', 'validation_features')]:
        with np.load(paths[key], allow_pickle=False) as z: tables[kind] = {k: z[k] for k in z.files}
        validate_table(tables[kind], kind=kind, plan=plan, plan_sha=manifest['plan_sha256'])
    stats = load_mel_stats(paths['mel_stats'])
    if stats['mean'].shape != (64,) or stats['recordings'] != expected['train_recordings'] or stats['frames'] != manifest['mel_stats']['frames']:
        raise ValueError('A mel statistics coverage mismatch')
    by_id = {r['recording_id']: r for r in rows}
    selected = [by_id[x['recording_id']] for x in plan.train + plan.validation]
    for row in selected:
        shard = row['canonical_path'].split('::', 1)[0]
        if Path(shard).name != shard or not (shard_dir/shard).is_file(): raise ValueError(f'missing or invalid shard: {shard}')
    loader = VerifiedWaveformLoader(ShardWaveformLoader(shard_dir, {r['recording_id']: r['canonical_path'] for r in selected}), selected)
    train = CropDataset([(x['recording_id'], x['label']) for x in plan.train], loader, seed=recipe['bigru']['training']['seed'])
    labels = {x['recording_id']: x['label'] for x in plan.validation}
    roles = {x['recording_id']: x['role'] for x in plan.validation}
    groups = {x['recording_id']: x['group_id'] for x in plan.validation}
    val_chunks = [c for c in chunks if c.protocol_id == 'protocol:A' and c.partition == 'validation' and c.recording_id in roles]
    for c in val_chunks:
        if c.group_id != groups[c.recording_id]:
            raise ValueError('chunk speaker mismatch')
    validation = {}
    for role in ROLES:
        validation[role] = ChunkDataset([c for c in val_chunks if roles[c.recording_id] == role], labels, loader)
        validation[role].validation_role = role
    return {'recipe': recipe, 'train': train, 'validation': validation, 'loader': loader, 'mel_stats': stats, 'lr_train': tables['train'], 'lr_validation': tables['validation'], 'validation_components': {x['recording_id']: x['component'] for x in plan.validation}, 'validation_groups': {x['recording_id']: x['group_id'] for x in plan.validation}, 'split_sha256': inp['split_sha256'], 'input_sha256': {k: file_sha256(v) for k, v in paths.items()}, 'counts': {'train_recordings': len(train), 'validation_recordings': dict(Counter(roles.values())), 'validation_chunks': {r: len(validation[r]) for r in ROLES}}}
