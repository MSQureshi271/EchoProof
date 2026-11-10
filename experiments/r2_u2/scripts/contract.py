"""Experiment identities; no research execution or acceptance authority."""
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiments/r2_u2'
INPUTS = ('data/manifests/cohort_v1.parquet', 'data/splits/protocol_B_v1.parquet',
          'data/processed/chunks_v1.parquet', 'reports/r1/split_report.json',
          'reports/r1/leakage.json', 'reports/r1/ast_probe.json',
          'reports/r1/r1_m1_colab_20261103T182916Z.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sources():
    names = ['src/echoproof/training/trainer.py', 'src/echoproof/training/datasets.py',
             'src/echoproof/training/seeding.py', 'src/echoproof/training/ledger.py',
             'src/echoproof/artifacts.py', 'src/echoproof/models/ast_detector.py',
             'src/echoproof/features/pretrained.py']
    paths = [ROOT / p for p in names] + sorted((EXP / 'scripts').glob('*.py'))
    paths += sorted((EXP / 'tests').glob('*.py')) + sorted((EXP / 'configs').glob('*.json'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def provenance(recipe, model_identity, fixture=False, research_contract=None):
    if not fixture:
        if not research_contract or research_contract.get('policy') != 'option_b_v1':
            raise ValueError('Missing prospective experiment contract')
        reference = json.loads((EXP / 'configs/historical_recipe.json').read_text())
        if recipe != reference:
            raise ValueError('Frozen recipe changed')
        data_root = Path(research_contract['data_root']).resolve()
        for name in INPUTS:
            if sha(data_root / name) != recipe['input_sha256'][name]:
                raise ValueError('Frozen input differs: ' + name)
        if model_identity.get('revision') != 'f826b80d28226b62986cc218e5cec390b1096902':
            raise ValueError('Wrong AST revision')
        if model_identity.get('weights_sha256') != 'ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d':
            raise ValueError('Wrong AST weights')
        return {'task': 'R2-U2 Option B isolated experiment', 'sources': sources(),
                'recipe_sha256': fingerprint(recipe), 'model': model_identity,
                'inputs': recipe['input_sha256'], 'contract': research_contract,
                'fixture_only': False, 'research_accepted': False,
                'scheduler_clock': 'constant', 'split_sha256': research_contract['split_sha256']}
    reference = json.loads((EXP / 'configs/historical_recipe.json').read_text())
    return {'task': 'R2-U2 numerical engineering fixtures', 'sources': sources(),
            'frozen_input_reference': reference['input_sha256'],
            'frozen_inputs_verified_for_research': False,
            'recipe_sha256': fingerprint(recipe), 'model': model_identity,
            'fixture_only': True, 'research_accepted': False, 'scheduler_clock': 'legacy_attempts'}


def new_run():
    path = EXP / 'runs' / ('engineering_' + uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    return path
