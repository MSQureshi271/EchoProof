"""Train the B-selected BiGRU recipe freshly on Protocol A."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import platform

from echoproof.training.r4_m1_data import build_data, file_sha256

SOURCES = ('scripts/r4_m1_bigru.py', 'src/echoproof/training/r4_m1_bigru.py', 'src/echoproof/training/r4_m1_data.py',
           'src/echoproof/training/r3_m1_data.py', 'src/echoproof/training/r3_m1_selection.py',
           'src/echoproof/training/trainer.py', 'src/echoproof/training/datasets.py',
           'src/echoproof/training/seeding.py', 'src/echoproof/models/bigru_waveform.py',
           'src/echoproof/models/bigru.py', 'src/echoproof/features/acoustic.py',
           'src/echoproof/features/r4_s1.py', 'src/echoproof/features/r2_s1.py',
           'src/echoproof/features/stats.py', 'src/echoproof/data/cohort.py',
           'src/echoproof/data/splits.py', 'src/echoproof/data/chunks.py', 'src/echoproof/contracts.py',
           'src/echoproof/evaluation/aggregate.py', 'src/echoproof/evaluation/metrics.py')


def freeze_contract(output, contract, resume):
    """Require an identical frozen run contract for any resume."""
    output = Path(output); path = output/'experiment_contract.json'
    if resume:
        if not path.is_file() or json.loads(path.read_text(encoding='utf-8')) != contract:
            raise ValueError('resume refused: run contract or provenance changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        with path.open('x', encoding='utf-8') as f:
            json.dump(contract, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())


def check_history(history, latest_epoch, best_epoch):
    """Reject crash windows where history and checkpoints are out of sync."""
    if [entry.get('epoch') for entry in history] != list(range(1, latest_epoch + 1)):
        raise ValueError('resume refused: history and latest checkpoint disagree')
    improved = [entry['epoch'] for entry in history if entry.get('improved') is True]
    if not improved or improved[-1] != best_epoch:
        raise ValueError('resume refused: history and best checkpoint disagree')


def verify_resume(run_dir):
    import pyarrow.parquet as pq
    import torch
    run_dir = Path(run_dir)
    latest = torch.load(run_dir/'latest.pt', map_location='cpu', weights_only=True)
    best = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=True)
    history = [json.loads(line) for line in (run_dir/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    check_history(history, latest['epoch'], latest['best_epoch'])
    if best['epoch'] != latest['best_epoch'] or best['config_hash'] != latest['config_hash']:
        raise ValueError('resume refused: best checkpoint does not match latest')
    for name in ('predictions_chunks.parquet', 'predictions_recordings.parquet'):
        epochs = pq.read_table(run_dir/name, columns=['epoch']).column('epoch').to_pylist()
        if not epochs or set(epochs) != {best['epoch']}:
            raise ValueError('resume refused: best predictions do not match best checkpoint')
    return latest


def make_config(recipe):
    from echoproof.training.trainer import TrainConfig
    values = dict(recipe['bigru']['training'])
    if values.pop('optimizer') != 'AdamW' or values.pop('loss') != 'cross_entropy':
        raise ValueError('unsupported optimizer or loss')
    config = TrainConfig(**values, owner=recipe['owner'], model_name='bigru', protocol='A')
    config.validate()
    return config


def run(root, shard_dir, output, *, resume=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from echoproof.models.bigru_waveform import WaveformBiGRU
    from echoproof.training.r3_m1_selection import selection_trainer
    from echoproof.training.seeding import seed_everything

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.parent != (root/'runs/r4').resolve():
        raise ValueError('output must be a direct child of runs/r4')
    if resume and not output.is_dir(): raise ValueError('resume requires an existing run directory')
    if not resume and output.exists(): raise FileExistsError('refusing to overwrite existing run')
    if not torch.cuda.is_available(): raise RuntimeError('BiGRU-A requires the CUDA runtime specified by its recipe')
    data = build_data(root, shard_dir)
    try:
        recipe = data['recipe']; cfg = make_config(recipe)
        architecture = recipe['bigru']['architecture']
        if architecture != {'input_size': 64, 'hidden_size': 128, 'num_layers': 2, 'bidirectional': True, 'dropout': .3, 'pooling': 'masked_mean'}:
            raise ValueError('unsupported BiGRU architecture')
        contract = {'task': 'R4-M1-BiGRU', 'protocol': 'A', 'recipe_sha256': file_sha256(root/'configs/r4_m1_recipes.json'),
                    'split_sha256': data['split_sha256'], 'input_sha256': data['input_sha256'],
                    'sources': {p: file_sha256(root/p) for p in SOURCES}, 'config': asdict(cfg),
                    'architecture': architecture, 'selection_role': 'selection',
                    'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
                                    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}}
        freeze_contract(output, contract, resume)
        if resume: verify_resume(output)
        seed_everything(cfg.seed)
        model = WaveformBiGRU(data['mel_stats']['mean'], data['mel_stats']['std'], hidden_size=architecture['hidden_size'], dropout=architecture['dropout'])
        trainer = selection_trainer(model, cfg, data['train'], data['validation']['selection'], output,
                                    provenance={'split_sha256': data['split_sha256'], 'recipe_sha256': contract['recipe_sha256'],
                                                'contract_sha256': file_sha256(output/'experiment_contract.json')})
        summary = trainer.fit(resume=resume)
        verify_resume(output)
        status_path = output/'status.json'
        if not status_path.exists():
            status_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
        evidence = {'task': 'R4-M1-BiGRU', 'status': 'training_complete', 'protocol': 'A',
                    'contract_sha256': file_sha256(output/'experiment_contract.json'), 'config_hash': summary['config_hash'],
                    'split_sha256': data['split_sha256'], 'epochs_run': summary['epochs_run'], 'best_epoch': summary['best_epoch'],
                    'best_selection_macro_f1': summary['best_macro_f1'], 'stop_reason': summary['stop_reason'],
                    'best_checkpoint_sha256': file_sha256(output/'best.pt'), 'latest_checkpoint_sha256': file_sha256(output/'latest.pt'),
                    'selection_predictions_sha256': {n: file_sha256(output/n) for n in ('predictions_chunks.parquet', 'predictions_recordings.parquet')},
                    'scope': 'Training and selection only; calibration, operating and test audio were not scored.',
                    'resume_limitation': 'cuDNN GRU inter-layer dropout may differ after resume; not bit-exact.'}
        evidence_path = output/'training_evidence.json'
        if evidence_path.exists():
            old = json.loads(evidence_path.read_text(encoding='utf-8'))
            if old != evidence: raise ValueError('completed training evidence differs; inspect run')
        else:
            with evidence_path.open('x', encoding='utf-8') as f: json.dump(evidence, f, indent=2, sort_keys=True, allow_nan=False)
        return evidence
    finally:
        data['loader'].close()
