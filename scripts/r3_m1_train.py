"""Run the predeclared R3-M1 BiGRU-B baseline and learning-rate alternative."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from echoproof.training.r3_m1_data import build_data, file_sha256

EXPERIMENTS = {'baseline': 0.001, 'lower_lr': 0.0003}
SOURCES = ('scripts/r3_m1_train.py', 'src/echoproof/training/r3_m1_data.py',
           'src/echoproof/training/r3_m1_selection.py', 'src/echoproof/training/r3_m1_export.py', 'src/echoproof/training/trainer.py',
           'src/echoproof/training/datasets.py', 'src/echoproof/training/seeding.py',
           'src/echoproof/models/bigru_waveform.py', 'src/echoproof/models/bigru.py',
           'src/echoproof/features/acoustic.py', 'src/echoproof/features/r2_s1.py',
           'src/echoproof/evaluation/aggregate.py', 'src/echoproof/evaluation/metrics.py')


def freeze_contract(output, contract, resume):
    path = Path(output) / 'experiment_contract.json'
    if resume:
        if not path.is_file() or json.loads(path.read_text(encoding='utf-8')) != contract:
            raise ValueError('resume refused: experiment contract or provenance changed')
    else:
        Path(output).mkdir(parents=True, exist_ok=False)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(contract, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())


def ensure_completed_ledger(ledger, run_dir, cfg, split_hash, summary):
    """Repair the narrow crash window after the final checkpoint and before ledger append."""
    import torch
    from echoproof.training.ledger import append_ledger
    ledger = Path(ledger)
    if ledger.exists():
        with ledger.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        if any(r['run_id'] == run_dir.name and r['status'] == 'completed' for r in rows):
            return
    checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    latest = torch.load(run_dir / 'latest.pt', map_location='cpu', weights_only=True)
    metrics = checkpoint['metrics']
    append_ledger(ledger, {'run_id': run_dir.name, 'owner': cfg.owner, 'model': cfg.model_name,
                           'protocol': cfg.protocol, 'config_hash': summary['config_hash'], 'split_hash': split_hash,
                           'seed': cfg.seed, 'epochs_run': summary['epochs_run'], 'best_epoch': summary['best_epoch'],
                           'stop_reason': summary['stop_reason'], 'val_macro_f1': metrics['macro_f1'],
                           'val_eer': metrics['eer'], 'val_fpr': metrics['fpr'],
                           'train_minutes': round(latest['train_seconds'] / 60, 3), 'device': 'cuda',
                           'status': 'completed', 'notes': 'ledger recovered from completed checkpoint; peak VRAM unavailable'})


def run(root, shard_dir, output, ledger, *, resume=False):
    import numpy as np
    import torch
    from echoproof.models.bigru_waveform import WaveformBiGRU
    from echoproof.training.ledger import append_ledger
    from echoproof.training.r3_m1_export import finish
    from echoproof.training.r3_m1_selection import selection_trainer
    from echoproof.training.seeding import seed_everything
    from echoproof.training.trainer import TrainConfig

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not torch.cuda.is_available():
        raise RuntimeError('R3-M1 requires the Colab CUDA runtime')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    check = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_r3_bigru_frontend.py')],
                           cwd=ROOT, env=env, capture_output=True, text=True)
    if check.returncode != 0 or ' skipped' in check.stdout:
        raise RuntimeError(f'frontend parity preflight failed or skipped:\n{check.stdout}\n{check.stderr}')
    data = build_data(root, shard_dir, seed=42)
    try:
        configs = {name: TrainConfig(seed=42, max_epochs=20, batch_size=32, eval_batch_size=64,
                                    lr=lr, weight_decay=0.01, scheduler='plateau', plateau_patience=1,
                                    plateau_factor=0.5, patience=3, min_delta=0.0, selection_threshold=0.5,
                                    precision='fp32', class_strategy='none', num_workers=0, device='cuda',
                                    deterministic=True, owner='Muneeb', model_name='bigru', protocol='B')
                   for name, lr in EXPERIMENTS.items()}
        from dataclasses import asdict
        contract = {'task': 'R3-M1', 'protocol': 'B', 'architecture': {'input_size': 64, 'hidden_size': 128,
                    'layers': 2, 'dropout': 0.3, 'pooling': 'masked_mean'},
                    'configs': {k: asdict(v) for k, v in configs.items()},
                    'checkpoint_ties': 'earliest epoch', 'experiment_ties': 'baseline',
                    'split_sha256': data['split_sha256'], 'inputs': data['input_sha256'],
                    'sources': {p: file_sha256(ROOT / p) for p in SOURCES},
                    'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                    'numpy': np.__version__, 'cuda': torch.version.cuda,
                                    'gpu': torch.cuda.get_device_name(0)}}
        output = Path(output)
        freeze_contract(output, contract, resume)
        summaries = {}
        for name, cfg in configs.items():
            run_dir = output / f'{output.name}-{name}'
            latest = run_dir / 'latest.pt'
            if run_dir.exists() and not latest.exists():
                raise RuntimeError(f'{run_dir} has no recoverable checkpoint; preserve it and start a new run')
            seed_everything(cfg.seed)  # Must precede model initialization, not just Trainer.fit.
            started_fit = False
            try:
                model = WaveformBiGRU(data['mel_stats']['mean'], data['mel_stats']['std'], hidden_size=128, dropout=0.3)
                trainer = selection_trainer(model, cfg, data['train'], data['validation']['selection'], run_dir,
                                            ledger_path=ledger, provenance={'split_sha256': data['split_sha256'],
                                            'inputs': data['input_sha256'], 'sources': contract['sources'],
                                            'environment': contract['environment'], 'experiment': name})
                started_fit = True
                summaries[name] = trainer.fit(resume=latest.exists())
                ensure_completed_ledger(ledger, run_dir, cfg, data['split_sha256'], summaries[name])
            except BaseException as exc:
                if not started_fit:
                    append_ledger(ledger, {'run_id': run_dir.name, 'owner': 'Muneeb', 'model': 'bigru', 'protocol': 'B',
                                           'split_hash': data['split_sha256'], 'seed': cfg.seed, 'device': 'cuda',
                                           'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else f'failed:{type(exc).__name__}',
                                           'notes': str(exc)[:200]})
                raise
        return {'runs': summaries, 'evidence': finish(output, data, summaries)}
    finally:
        data['loader'].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='persistent run directory on Drive')
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run(args.root, args.shard_dir, args.output, args.ledger, resume=args.resume)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
