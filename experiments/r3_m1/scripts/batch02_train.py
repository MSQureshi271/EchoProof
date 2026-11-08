"""Train the isolated hidden-size-64 R3-M1 candidate from the batch02 contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'experiments/r3_m1'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_guard import check_run
from prepare_batch import sha
from batch02_contract import verify_contract, ARCHITECTURE, RUN_ID, read


def verify_epoch_files(directory, epoch):
    """Refuse resume if any extra, missing or partial epoch output is present."""
    folder = directory / 'epoch_predictions'
    expected = {f'epoch_{e:02d}_{kind}.parquet' for e in range(1, epoch + 1) for kind in ('chunks', 'recordings')}
    actual = {p.name for p in folder.iterdir()} if folder.exists() else set()
    if actual != expected:
        raise ValueError('epoch prediction files differ from latest checkpoint; preserve and inspect the run')



def run(contract_path, experiment_id, shard_dir, resume):
    import numpy as np
    import torch
    from echoproof.models.bigru_waveform import WaveformBiGRU
    from echoproof.training.r3_m1_data import build_data
    from echoproof.training.r3_m1_selection import SelectionAggregation, selection_metrics
    from echoproof.training.seeding import seed_everything
    from echoproof.training.trainer import Trainer, TrainConfig
    import pyarrow as pa
    import pyarrow.parquet as pq

    contract_path = Path(contract_path).resolve()
    if not contract_path.is_relative_to((EXPERIMENT / 'configs').resolve()):
        raise ValueError('contract must be in the experiment configs folder')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    verify_contract(contract, contract_path)
    preflight = read(EXPERIMENT / 'reports/batch02/preflight.json')
    if (preflight.get('status') != 'passed' or preflight.get('contract_sha256') != sha(contract_path)
            or preflight.get('junit_sha256') != sha(EXPERIMENT / 'reports/batch02/preflight.xml')):
        raise ValueError('matching passed batch02 fixture preflight is required')
    if experiment_id not in contract['configs']:
        raise ValueError('experiment is not included in frozen batch')
    env = contract['environment']
    if (not torch.cuda.is_available() or torch.cuda.get_device_name(0) != env['gpu'] or
            torch.__version__ != env['torch'] or np.__version__ != env['numpy'] or
            torch.version.cuda != env['cuda'] or platform.python_version() != env['python']):
        raise ValueError('execution environment changed since contract freeze')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    data = build_data(ROOT, shard_dir, seed=42)
    try:
        if data['input_sha256'] != contract['inputs']:
            raise ValueError('training data differs from frozen contract')
        batch_dir = EXPERIMENT / 'runs' / 'batch02'
        batch_dir.mkdir(parents=True, exist_ok=True)
        marker = batch_dir / 'active_attempt.json'
        attempts = batch_dir / 'attempts.jsonl'
        if marker.exists() or (batch_dir / 'checks_active.json').exists():
            raise ValueError('previous attempt did not close cleanly; inspect budget and checkpoints')
        prior = [json.loads(s) for s in attempts.read_text(encoding='utf-8').splitlines()] if attempts.exists() else []
        spent = sum(float(r['elapsed_seconds']) for r in prior)
        budget = float(contract['budget']['max_training_and_validation_gpu_minutes']) * 60
        if spent >= budget:
            raise ValueError('batch training budget exhausted')
        directory = batch_dir / experiment_id
        if resume:
            if not (directory / 'latest.pt').is_file():
                raise ValueError('no latest checkpoint to resume')
            boundary = check_run(directory)
            verify_epoch_files(directory, boundary['latest_epoch'])
        elif directory.exists():
            raise FileExistsError('experiment directory already exists; use resume if safe')
        cfg = TrainConfig(**contract['configs'][experiment_id])
        if cfg.patience != 21 or cfg.max_epochs != 20 or cfg.selection_threshold != .5:
            raise ValueError('configuration violates the declared hidden-size experiment')
        with marker.open('x', encoding='utf-8') as f:
            json.dump({'experiment': experiment_id, 'contract_sha256': sha(contract_path), 'spent_seconds': spent}, f)
            f.flush()
            os.fsync(f.fileno())
        started = time.monotonic()
        status = 'failed'
        try:
            seed_everything(cfg.seed)
            model = WaveformBiGRU(data['mel_stats']['mean'], data['mel_stats']['std'], hidden_size=ARCHITECTURE['hidden_size'], dropout=ARCHITECTURE['dropout'])

            class ExperimentTrainer(Trainer):
                def _train_epoch(self, epoch):
                    if spent + time.monotonic() - started >= budget:
                        raise RuntimeError('batch time budget exhausted at epoch boundary')
                    self.current_epoch = epoch
                    return super()._train_epoch(epoch)

                def evaluate(self):
                    metrics, chunks, recordings = super().evaluate()
                    out = self.run_dir / 'epoch_predictions'
                    out.mkdir(exist_ok=True)
                    for kind, rows in [('chunks', chunks), ('recordings', recordings)]:
                        if kind == 'chunks':
                            table = {'chunk_id': [r[0] for r in rows], 'recording_id': [r[1] for r in rows],
                                     'label': [r[2] for r in rows], 'tts_probability': [r[3] for r in rows]}
                        else:
                            table = {'recording_id': [r[0] for r in rows], 'label': [r[1] for r in rows],
                                     'raw_score': [r[2] for r in rows], 'chunks': [r[3] for r in rows]}
                        path = out / f'epoch_{self.current_epoch:02d}_{kind}.parquet'
                        if path.exists():
                            raise FileExistsError(path)
                        temp = path.with_suffix('.parquet.tmp')
                        pq.write_table(pa.table(table), temp)
                        os.replace(temp, path)
                    return metrics, chunks, recordings

                def _checkpoint(self, epoch, state, finished):
                    # Preserve the best checkpoint at the first hypothetical patience-3 stop.
                    snapshot = self.run_dir / 'patience3_best.pt'
                    if state['bad_epochs'] == 3 and not snapshot.exists():
                        import shutil
                        temporary = snapshot.with_suffix('.pt.tmp')
                        shutil.copyfile(self.run_dir / 'best.pt', temporary)
                        os.replace(temporary, snapshot)
                    print(f"{experiment_id}: epoch {epoch}/20, best F1={state['best_metric']:.6f} at epoch {state['best_epoch']}", flush=True)
                    return super()._checkpoint(epoch, state, finished)

            trainer = ExperimentTrainer(model, cfg, data['train'], data['validation']['selection'], directory,
                                        ledger_path=batch_dir / 'ledger.csv',
                                        provenance={'split_sha256': data['split_sha256'], 'batch_contract_sha256': sha(contract_path),
                                                    'experiment': experiment_id, 'architecture': ARCHITECTURE}, metrics_fn=selection_metrics,
                                        aggregation_fn=SelectionAggregation(data['validation']['selection']))
            summary = trainer.fit(resume=resume)
            check_run(directory)
            verify_epoch_files(directory, summary['epochs_run'])
            verify_contract(contract, contract_path)
            status = 'completed'
            return summary
        finally:
            elapsed = time.monotonic() - started
            with attempts.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'experiment': experiment_id, 'status': status, 'elapsed_seconds': elapsed}) + '\n')
                f.flush()
                os.fsync(f.fileno())
            marker.unlink()
    finally:
        data['loader'].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--experiment', required=True, choices=[RUN_ID])
    parser.add_argument('--shard-dir', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.contract, args.experiment, args.shard_dir, args.resume), indent=2))
