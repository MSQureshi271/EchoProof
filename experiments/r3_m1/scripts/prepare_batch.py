"""Validate the audit and freeze batch-01 provenance before experiment training."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from importlib.metadata import version
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'experiments/r3_m1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_audit(audit):
    required = {'original_provenance_and_checkpoint_hashes', 'baseline_early_stop_replay',
                'baseline_selection_predictions', 'lower_lr_early_stop_replay', 'lower_lr_selection_predictions',
                'scheduler_fixture', 'gpu_frontend_and_gradients_fixture', 'real_data_and_pcm',
                'gpu_full_frontend_and_frozen_normalization', 'epoch_boundary_recovery_fixtures',
                'multi_file_crash_window'}
    checks = audit.get('checks', {})
    if audit.get('task') != 'R3-M1 audit' or audit.get('overall') != 'passed' or not required <= set(checks):
        raise ValueError('a complete passed R3-M1 audit is required')
    if any(checks[name].get('status') != 'passed' for name in required):
        raise ValueError('at least one required audit check is not passed')
    return checks


def verify_firewall(root):
    if (root / 'locks/B_lock.json').exists() or (root / 'locks/A_lock.json').exists():
        raise ValueError('a protocol lock exists; this extension cannot reopen B')
    ledger = root / 'reports/ledger.csv'
    if not ledger.is_file():
        raise ValueError('main experiment ledger is missing')
    with ledger.open(newline='', encoding='utf-8') as handle:
        if any(row.get('protocol', '').strip().lower() in ('a', 'protocol:a') for row in csv.DictReader(handle)):
            raise ValueError('Protocol A ledger activity exists; inspect the firewall before continuing')
    return {'locks_absent': True, 'protocol_a_ledger_rows': 0,
            'limitation': 'local locks and ledger cannot prove no unrecorded Protocol A exposure'}


def prepare(root, audit_path, output):
    import numpy
    import torch
    root, audit_path, output = Path(root).resolve(), Path(audit_path).resolve(), Path(output).resolve()
    exp = (root / 'experiments/r3_m1').resolve()
    if not audit_path.is_relative_to(exp / 'reports') or not output.is_relative_to(exp / 'configs'):
        raise ValueError('audit and resolved config must be in the experiment directory')
    if not torch.cuda.is_available():
        raise ValueError('a CUDA GPU is required to freeze the execution environment')
    plan_path = exp / 'configs/batch01_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    checks = validate_audit(audit)
    firewall = verify_firewall(root)
    reference_path = root / plan['reference_run'] / 'experiment_contract.json'
    if sha(reference_path) != plan['reference_contract_sha256']:
        raise ValueError('original contract hash differs from plan')
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    if reference['split_sha256'] != plan['split_sha256']:
        raise ValueError('reference split mismatch')
    input_paths = {'cohort': 'data/manifests/cohort_v1.parquet', 'split': 'data/splits/protocol_B_v1.parquet',
                   'chunks': 'data/processed/chunks_v1.parquet', 'stats': 'data/features/r2_s1_B/mel_stats_B_v1.npz',
                   'features_manifest': 'data/features/r2_s1_B/r2_s1_manifest.json'}
    input_hashes = {name: sha(root / rel) for name, rel in input_paths.items()}
    if input_hashes != reference['inputs'] or input_hashes != checks['real_data_and_pcm']['details']['input_sha256']:
        raise ValueError('current inputs differ from the original run or audit')
    source_hashes = {}
    for relative, expected in reference['sources'].items():
        raw = (root / relative).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f'original source differs from reference: {relative}')
        source_hashes[relative] = actual
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != reference['environment']['gpu']:
        raise ValueError('GPU type differs from the original run')
    if torch.__version__ != reference['environment']['torch'] or numpy.__version__ != reference['environment']['numpy']:
        raise ValueError('PyTorch or NumPy differs from the original run')
    configs = {}
    for entry in plan['experiments']:
        config = dict(reference['configs'][entry['reference_config']])
        config.update(plan['training_overrides'])
        if config['lr'] != entry['lr']:
            raise ValueError('planned learning rate differs from reference')
        configs[entry['id']] = config
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    runner = exp / 'scripts/train.py'
    if not runner.is_file():
        raise ValueError('batch runner must be implemented before freezing the contract')
    contract = {'task': 'R3-M1 batch01', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'frozen_not_started', 'plan_sha256': sha(plan_path), 'audit_sha256': sha(audit_path),
                'reference_contract_sha256': sha(reference_path), 'inputs': input_hashes, 'reference_sources': source_hashes,
                'experiment_sources': {str(p.relative_to(exp)).replace('\\', '/'): sha(p)
                                       for p in sorted((exp / 'scripts').glob('*.py'))},
                'reference_checkpoints': {name: sha(root / plan['reference_run'] / (Path(plan['reference_run']).name + '-' + name) / 'best.pt')
                                          for name in ('baseline', 'lower_lr')},
                'configs': configs, 'budget': plan['budget'], 'selection': plan['selection'], 'firewall': firewall,
                'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__,
                                'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda},
                'dependencies': {name: version(name) for name in ('librosa', 'scipy', 'pyarrow', 'soundfile', 'scikit-learn')},
                'git_commit': git.stdout.strip() if git.returncode == 0 else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(contract, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    return contract


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = prepare(args.root, args.audit, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'firewall': result['firewall']}, indent=2))
