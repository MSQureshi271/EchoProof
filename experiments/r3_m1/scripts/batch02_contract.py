"""Freeze and verify the single hidden-size-64 experiment without editing batch 01."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiments/r3_m1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_batch import sha

RUN_ID = 'b02_hidden64_full20'
PRIOR_IDS = {'original_baseline', 'original_lower_lr', 'b01_baseline_full20', 'b01_lower_lr_full20'}
ARCHITECTURE = {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3, 'bidirectional': True}
INPUT_PATHS = {'cohort': 'data/manifests/cohort_v1.parquet', 'split': 'data/splits/protocol_B_v1.parquet',
               'chunks': 'data/processed/chunks_v1.parquet', 'stats': 'data/features/r2_s1_B/mel_stats_B_v1.npz',
               'features_manifest': 'data/features/r2_s1_B/r2_s1_manifest.json'}
RESTRICTIONS = {'experimental_only': True, 'official_promotion_allowed': False,
                'allowed_audio_roles': ['train', 'selection'], 'lock_changes_allowed': False}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lock_snapshot():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'locks').glob('*_lock.json'))}


def lock_line_ending_hashes():
    """Read-only transport equivalence: only LF/CRLF may differ from approval.

    Do not parse/reserialize JSON or ignore other whitespace/content changes.
    Resolved contracts still retain exact runtime artifact hashes.
    """
    import hashlib
    result = {}
    for path in sorted((ROOT / 'locks').glob('*_lock.json')):
        lf = path.read_bytes().replace(b'\r\n', b'\n')
        result[path.relative_to(ROOT).as_posix()] = {
            hashlib.sha256(lf).hexdigest(),
            hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest(),
        }
    return result


def authorization_differences(approval, plan_hash, locks, line_ending_hashes=None):
    """Describe a sync/scope mismatch without changing the approved evidence."""
    differences = []
    if approval.get('approved_by') != 'Muneeb' or approval.get('approved') is not True:
        differences.append('approval identity/status does not match explicit Muneeb approval')
    if approval.get('plan_sha256') != plan_hash:
        differences.append(f"plan hash differs: approved={approval.get('plan_sha256')}, current={plan_hash}")
    if approval.get('restrictions') != RESTRICTIONS:
        differences.append('approval restrictions differ from the experimental-only scope')
    expected = approval.get('lock_sha256')
    if not isinstance(expected, dict):
        differences.append('approval lock snapshot is missing or malformed')
    else:
        for name in sorted(set(expected) | set(locks)):
            if name not in locks:
                differences.append(f'approved lock missing: {name} (approved={expected[name]})')
            elif name not in expected:
                differences.append(f'unapproved additional lock: {name} (current={locks[name]})')
            elif (expected[name] != locks[name]
                  and expected[name] not in (line_ending_hashes or {}).get(name, ())):
                differences.append(f'lock hash differs: {name} (approved={expected[name]}, current={locks[name]})')
    return differences


def verify_authorization():
    """Require explicit user approval for this particular post-lock diagnostic plan."""
    path = EXP / 'configs/batch02_authorization.json'
    if not path.is_file():
        raise ValueError('batch02 needs explicit Muneeb approval of the isolated post-lock experiment; no authorization file exists')
    approval = read(path)
    differences = authorization_differences(approval, sha(EXP / 'configs/batch02_plan.json'),
                                           lock_snapshot(), lock_line_ending_hashes())
    if differences:
        raise ValueError('authorization does not match this workspace:\n- ' + '\n- '.join(differences)
                         + '\nInspect sync differences. Do not edit hashes or remove locks to bypass this check.'
                         + ' If the approved scope or lock state intentionally changed, ask Muneeb again.')
    return approval


def protected_artifacts():
    """Freeze lock records and their referenced bundle files without altering them."""
    files = dict(lock_snapshot())
    for relative in list(files):
        bundle = read(ROOT / relative).get('bundle', {}).get('bundle_path')
        if bundle:
            directory = (ROOT / bundle).resolve()
            if not directory.is_relative_to(ROOT.resolve()):
                raise ValueError('locked bundle path is outside the project')
            if not directory.exists():
                # A missing referenced bundle is not evidence of a verified bundle.
                # Freeze its absence; this diagnostic must not create it either.
                files[directory.relative_to(ROOT).as_posix()+'/'] = None
                continue
            if not directory.is_dir():
                raise ValueError('locked bundle path is not a directory')
            for p in directory.rglob('*'):
                if p.is_file():
                    files[p.relative_to(ROOT).as_posix()] = sha(p)
    return files


def validate_design(plan, prior):
    if plan['task'] != 'R3-M1 batch02' or plan['experiment_id'] != RUN_ID:
        raise ValueError('unexpected batch identity')
    if plan['architecture'] != ARCHITECTURE:
        raise ValueError('only the declared hidden-size-64 architecture is eligible')
    cfg = dict(prior['configs']['b01_lower_lr_full20'])
    if plan['training_config'] != cfg:
        raise ValueError('training configuration must match the prior lower-LR full-20 run')
    if cfg['lr'] != .0003 or cfg['seed'] != 42 or cfg['max_epochs'] != 20 or cfg['patience'] != 21:
        raise ValueError('invalid matched training design')
    if plan['selection'] != prior['selection']:
        raise ValueError('selection rule must remain unchanged')
    if plan['budget'] != {'max_training_runs': 1, 'max_epochs_per_run': 20,
                          'max_training_and_validation_gpu_minutes': 60,
                          'max_audit_and_export_gpu_minutes': 15, 'count_interrupted_attempts': True}:
        raise ValueError('unexpected batch budget')
    return cfg


def validate_candidate_ids(candidates):
    if len(candidates) != 5 or {c['id'] for c in candidates} != PRIOR_IDS | {RUN_ID}:
        raise ValueError('exactly the four prior candidates and hidden64 are required')


def verify_environment(expected):
    import numpy
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required')
    actual = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__,
              'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda}
    if actual != expected:
        raise ValueError(f'execution environment differs from batch01: {actual}')


def verify_references():
    prior_path = EXP / 'configs/batch01_resolved.json'
    prior = read(prior_path)
    # Batch01's guard remains unchanged. This separate workflow requires an
    # explicit, recorded exception and can never promote a diagnostic candidate.
    verify_authorization()
    for name, expected in prior['dependencies'].items():
        if version(name) != expected:
            raise ValueError(f'dependency changed: {name}')
    for relative, expected in prior['experiment_sources'].items():
        if sha(EXP / relative) != expected:
            raise ValueError(f'batch01 source changed: {relative}')
    for relative, expected in prior['reference_sources'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'original source changed: {relative}')
    if sha(EXP / 'configs/batch01_plan.json') != prior['plan_sha256']:
        raise ValueError('batch01 plan changed')
    if sha(EXP / 'reports/audit_20261107T165239Z.json') != prior['audit_sha256']:
        raise ValueError('passed audit changed')
    original = ROOT / 'runs/r3/r3_m1_20261107T144037Z'
    if sha(original / 'experiment_contract.json') != prior['reference_contract_sha256']:
        raise ValueError('original contract changed')
    for name, expected in prior['reference_checkpoints'].items():
        if sha(original / (original.name+'-'+name) / 'best.pt') != expected:
            raise ValueError('original checkpoint changed')
    comparison = read(EXP / 'reports/batch01_comparison.json')
    if comparison['status'] != 'comparison_complete' or comparison['contract_sha256'] != sha(prior_path):
        raise ValueError('batch01 comparison does not match its contract')
    if comparison['selected'] != read(EXP / 'reports/final_selection.json'):
        raise ValueError('batch01 selected candidate differs from its decision')
    if len(comparison['candidates']) != 4 or {c['id'] for c in comparison['candidates']} != PRIOR_IDS:
        raise ValueError('batch01 candidate coverage changed')
    for relative, expected in comparison['artifact_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'batch01 evidence changed: {relative}')
    if {name: sha(ROOT / rel) for name, rel in INPUT_PATHS.items()} != prior['inputs']:
        raise ValueError('data inputs differ from batch01')
    return prior


def verify_contract(contract, path):
    path = Path(path).resolve()
    if path != (EXP / 'configs/batch02_resolved.json').resolve():
        raise ValueError('use the dedicated batch02 resolved contract path')
    if contract.get('task') != 'R3-M1 batch02' or contract.get('status') != 'frozen_not_started':
        raise ValueError('invalid batch02 contract')
    for relative, expected in contract['frozen_files'].items():
        if sha(EXP / relative) != expected:
            raise ValueError(f'frozen batch02 dependency changed: {relative}')
    if contract['protected_artifacts'] != protected_artifacts():
        raise ValueError('locked artifacts changed after contract freeze; ask Muneeb before proceeding')
    if contract['restrictions'] != RESTRICTIONS or contract['authorization'] != verify_authorization():
        raise ValueError('contract scope differs from explicit authorization')
    prior = verify_references()
    plan = read(EXP / 'configs/batch02_plan.json')
    cfg = validate_design(plan, prior)
    if contract['configs'] != {RUN_ID: cfg} or contract['architecture'] != ARCHITECTURE:
        raise ValueError('resolved design differs from plan')
    for field in ('inputs', 'environment', 'dependencies'):
        if contract[field] != prior[field]:
            raise ValueError(f'resolved {field} differs from batch01')
    if contract['budget'] != plan['budget'] or contract['selection'] != plan['selection']:
        raise ValueError('resolved budget or selection differs from plan')
    verify_environment(contract['environment'])


def prepare():
    prior = verify_references()
    verify_environment(prior['environment'])
    plan = read(EXP / 'configs/batch02_plan.json')
    cfg = validate_design(plan, prior)
    fixed = ['configs/batch02_plan.json', 'configs/batch02_authorization.json', 'configs/batch01_resolved.json',
             'reports/batch01_comparison.json', 'reports/final_selection.json']
    fixed += [p.relative_to(EXP).as_posix() for p in sorted((EXP / 'scripts').glob('*.py'))]
    fixed += [p.relative_to(EXP).as_posix() for p in sorted((EXP / 'tests').glob('test_batch02*.py'))]
    contract = {'task': 'R3-M1 batch02', 'status': 'frozen_not_started',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'configs': {RUN_ID: cfg}, 'architecture': ARCHITECTURE,
                'budget': plan['budget'], 'selection': plan['selection'],
                'frozen_files': {name: sha(EXP / name) for name in fixed},
                'inputs': prior['inputs'], 'environment': prior['environment'],
                'dependencies': prior['dependencies'], 'authorization': verify_authorization(),
                'protected_artifacts': protected_artifacts(), 'restrictions': RESTRICTIONS,
                'disclosure': 'Isolated post-lock, result-informed single-seed capacity diagnostic. Not eligible to change the locked research result or claim independent pre-lock evidence.'}
    output = EXP / 'configs/batch02_resolved.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(contract, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    path = EXP / 'configs/batch02_resolved.json'
    if args.verify:
        verify_contract(read(path), path)
        print('Batch02 contract verified')
    else:
        print(prepare())
