from types import SimpleNamespace

import numpy as np
import pytest

from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH
from echoproof.training.r4_m1_data import canonical_json_sha256, check_groups, validate_table, verify_recipe


def fixture_plan():
    train = [{'recording_id': 'a', 'group_id': 'g1', 'component': 'genuine', 'label': 0},
             {'recording_id': 'b', 'group_id': 'g1', 'component': 'melo', 'label': 1}]
    validation = []
    for i, role in enumerate(('selection', 'calibration', 'operating')):
        for label, component in [(0, 'genuine'), (1, 'xtts')]:
            rid = f'{role}{label}'
            validation.append({'recording_id': rid, 'group_id': f'g{i+2}', 'component': component,
                               'label': label, 'role': role, 'chunks': [(f'chunk:A:{rid}:0', 0, 10)]})
    crops = [{**r, 'crop_id': f"crop:A:{r['recording_id']}:0"} for r in train]
    return SimpleNamespace(train=train, validation=validation, crop_bank=crops)


def table_for(plan):
    rows = plan.validation
    return {'X': np.zeros((len(rows), len(FEATURE_NAMES))),
            'feature_names': np.array(FEATURE_NAMES), 'feature_schema_hash': np.array(FEATURE_SCHEMA_HASH),
            'plan_sha256': np.array('plan'), 'chunk_id': np.array([r['chunks'][0][0] for r in rows]),
            'recording_id': np.array([r['recording_id'] for r in rows]),
            'group_id': np.array([r['group_id'] for r in rows]), 'component': np.array([r['component'] for r in rows]),
            'y': np.array([r['label'] for r in rows]), 'role': np.array([r['role'] for r in rows])}


def test_group_disjointness():
    plan = fixture_plan(); check_groups(plan)
    plan.validation[0]['group_id'] = 'g1'
    with pytest.raises(ValueError, match='speaker crosses'): check_groups(plan)


def test_validation_table_rejects_role_and_identity_mismatch():
    plan = fixture_plan(); table = table_for(plan)
    validate_table(table, kind='validation', plan=plan, plan_sha='plan')
    table['role'][0] = 'operating'
    with pytest.raises(ValueError, match='does not match'): validate_table(table, kind='validation', plan=plan, plan_sha='plan')


def test_duplicate_and_test_chunk_rejected():
    plan = fixture_plan(); table = table_for(plan)
    table['chunk_id'][1] = table['chunk_id'][0]
    with pytest.raises(ValueError): validate_table(table, kind='validation', plan=plan, plan_sha='plan')
    table = table_for(plan); table['chunk_id'][0] = 'chunk:A:test:0'
    with pytest.raises(ValueError): validate_table(table, kind='validation', plan=plan, plan_sha='plan')


def test_frozen_recipe_matches_local_b_artifacts():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads((root/'configs/r4_m1_recipes.json').read_text(encoding='utf-8'))
    verify_recipe(root, recipe)
    recipe['bigru']['training']['lr'] = 0.001
    with pytest.raises(ValueError, match='differs from B'): verify_recipe(root, recipe)


def test_canonical_lock_hash_ignores_format_but_detects_content(tmp_path):
    first, second = tmp_path/'first.json', tmp_path/'second.json'
    first.write_text('{"a": 1, "b": [2, 3]}', encoding='utf-8')
    second.write_text('{\n  "b": [2, 3],\n  "a": 1\n}', encoding='utf-8')
    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    second.write_text('{"a": 1, "b": [2, 4]}', encoding='utf-8')
    assert canonical_json_sha256(first) != canonical_json_sha256(second)
