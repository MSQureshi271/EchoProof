"""Read-only audit of the original R3-M1 evidence; never scores held-out audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
EXPERIMENT = ROOT / 'experiments/r3_m1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def replay(history, patience=3):
    if patience < 1 or not history:
        raise ValueError('empty history or invalid patience')
    best, best_epoch, bad, stop = -math.inf, 0, 0, None
    for index, row in enumerate(history, 1):
        score = row['val']['macro_f1']
        if row['epoch'] != index or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid epoch or macro F1')
        improved = score > best
        if bool(row['improved']) != improved:
            raise ValueError('history improvement flag differs from strict F1 rule')
        if improved:
            best, best_epoch, bad = score, index, 0
        else:
            bad += 1
        if bad >= patience:
            stop = index
            break
    return {'best_epoch': best_epoch, 'best_macro_f1': best, 'stop_epoch': stop}


def validate_coverage(train_items, selection_chunks):
    train_ids = {rid for rid, _ in train_items}
    selection_ids = {c.recording_id for c in selection_chunks}
    if len(train_ids) != len(train_items) or len({c.chunk_id for c in selection_chunks}) != len(selection_chunks):
        raise ValueError('duplicate training recording or selection chunk')
    train_components = Counter(rid.split(':')[1] for rid in train_ids)
    selection_components = Counter(rid.split(':')[1] for rid in selection_ids)
    if train_components != {'genuine': 1510, 'melo': 1510, 'vits': 1510}:
        raise ValueError('unexpected training component counts')
    if selection_components != {'genuine': 500, 'melo': 500, 'vits': 500, 'xtts': 500}:
        raise ValueError('unexpected selection component counts')
    for rid, label in train_items:
        if label != int(rid.split(':')[1] != 'genuine'):
            raise ValueError('training label disagrees with component')
    train_speakers = {rid.split(':')[-1].split('-')[0] for rid in train_ids}
    selection_speakers = {rid.split(':')[-1].split('-')[0] for rid in selection_ids}
    if (len(train_ids), len(train_speakers), len(selection_ids), len(selection_speakers), len(selection_chunks)) != (4530, 151, 2000, 25, 10029):
        raise ValueError('unexpected train or selection coverage')
    if train_speakers & selection_speakers:
        raise ValueError('speaker crosses partitions')
    return train_ids, selection_ids, train_speakers, selection_speakers


def verify_selection(run, evidence, name):
    import pyarrow.parquet as pq
    from echoproof.evaluation.metrics import compute_metrics
    directory = run / (run.name + '-' + name)
    chunks = pq.read_table(directory / 'predictions_chunks.parquet').to_pylist()
    records = pq.read_table(directory / 'predictions_recordings.parquet').to_pylist()
    if len(chunks) != 10029 or len(records) != 2000:
        raise ValueError('unexpected selection counts')
    grouped, seen = defaultdict(list), set()
    for row in chunks:
        cid, rid, label, score = row['chunk_id'], row['recording_id'], row['label'], row['tts_probability']
        if cid in seen or label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('duplicate chunk or invalid label/score')
        seen.add(cid)
        grouped[rid].append((label, score))
    if len(grouped) != len(records) or len({r['recording_id'] for r in records}) != len(records):
        raise ValueError('recording coverage mismatch')
    for row in records:
        parts = grouped[row['recording_id']]
        if (len(parts) != row['chunks'] or any(y != row['label'] for y, _ in parts) or
                abs(row['raw_score'] - math.fsum(s for _, s in parts) / len(parts)) > 1e-12):
            raise ValueError('invalid mean score or inconsistent labels')
    m = compute_metrics([r['label'] for r in records], [r['raw_score'] for r in records], 0.5)
    if abs(m['macro_f1'] - evidence['selection']['selection_macro_f1'][name]) > 1e-12:
        raise ValueError('selection metric differs from recorded result')
    return {'chunks': len(chunks), 'recordings': len(records), 'macro_f1': m['macro_f1']}


def audit(root=ROOT, shard_dir=None):
    root = Path(root)
    run = root / 'runs/r3/r3_m1_20261107T144037Z'
    evidence = json.loads((run / 'evidence.json').read_text(encoding='utf-8'))
    contract = json.loads((run / 'experiment_contract.json').read_text(encoding='utf-8'))
    result = {'task': 'R3-M1 audit', 'checks': {}, 'environment': {'python': platform.python_version()},
              'source_sha256': {}}
    checks = result['checks']

    def check(name, fn):
        try:
            checks[name] = {'status': 'passed', 'details': fn()}
        except ImportError as exc:
            checks[name] = {'status': 'not_measured', 'reason': str(exc)}
        except Exception as exc:
            checks[name] = {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}

    def provenance():
        reference = json.loads((root / 'experiments/r3_m1/configs/batch01_plan.json').read_text(encoding='utf-8'))
        if digest(run / 'experiment_contract.json') != reference['reference_contract_sha256']:
            raise ValueError('original contract hash mismatch')
        mismatches = []
        for path, expected in contract['sources'].items():
            content = (root / path).read_bytes()
            raw = hashlib.sha256(content).hexdigest()
            normalized = hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest()
            result['source_sha256'][path] = {'raw': raw, 'lf_normalized': normalized}
            if expected not in (raw, normalized):
                mismatches.append(path)
        if mismatches:
            raise ValueError(f'source hash mismatch: {mismatches}')
        for name, expected in evidence['selection']['checkpoint_sha256'].items():
            if digest(run / (run.name + '-' + name) / 'best.pt') != expected:
                raise ValueError(f'{name} checkpoint hash mismatch')
        return {'reference_contract_sha256': reference['reference_contract_sha256'],
                'recorded_input_sha256': evidence['input_sha256'], 'input_bytes_verified': False}

    check('original_provenance_and_checkpoint_hashes', provenance)
    for name in ('baseline', 'lower_lr'):
        def history_check(name=name):
            path = run / (run.name + '-' + name)
            history = [json.loads(line) for line in (path / 'history.jsonl').read_text().splitlines()]
            status = json.loads((path / 'status.json').read_text())
            replayed = replay(history)
            if (replayed['stop_epoch'] != status['epochs_run'] or replayed['best_epoch'] != status['best_epoch'] or
                    replayed['best_macro_f1'] != status['best_macro_f1'] or status['stop_reason'] != 'early_stopping'):
                raise ValueError('history and run status disagree')
            return {**replayed, 'recorded_lr_is_after_scheduler_step': True}
        check(name + '_early_stop_replay', history_check)
        check(name + '_selection_predictions', lambda name=name: verify_selection(run, evidence, name))

    def scheduler_fixture():
        import torch
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)
        rates = []
        for score in (.5, .4, .45):
            scheduler.step(score)
            rates.append(optimizer.param_groups[0]['lr'])
        if any(abs(a-b) > 1e-12 for a, b in zip(rates, (.001, .001, .0005))):
            raise ValueError(f'unexpected plateau schedule: {rates}')
        return {'scores': [.5, .4, .45], 'lr_after_scheduler': rates}

    def gpu_fixtures():
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise ImportError('CUDA GPU unavailable')
        from echoproof.features.acoustic import log_mel
        from echoproof.models.bigru_waveform import WaveformBiGRU
        torch.manual_seed(42)
        model = WaveformBiGRU(np.zeros(64), np.ones(64), hidden_size=4, dropout=0).cuda()
        rng = np.random.default_rng(42)
        waves = [np.zeros(100, dtype=np.float32), rng.normal(0, .1, 1200).astype(np.float32)]
        padded = np.full((2, 1200), .25, dtype=np.float32)
        for i, wave in enumerate(waves):
            padded[i, :len(wave)] = wave
        x = torch.tensor(padded, device='cuda')
        lengths = torch.tensor([len(w) for w in waves], device='cuda')
        model.eval()
        with torch.no_grad():
            features, frames = model.features(x, lengths)
            alone = model(x[:1, :100], lengths[:1])[0]
            together = model(x, lengths)[0]
        for i, wave in enumerate(waves):
            expected = log_mel(wave)
            np.testing.assert_allclose(features[i, :int(frames[i])].cpu().numpy(), expected, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(alone, together, atol=1e-5, rtol=1e-5)
        model.train()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = torch.nn.functional.cross_entropy(model(x, lengths), torch.tensor([0, 1], device='cuda'))
        if not torch.isfinite(loss):
            raise ValueError('non-finite loss')
        loss.backward()
        for prefix in ('bigru.gru.', 'bigru.head.'):
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
            if not grads or any(g is None or not torch.isfinite(g).all() for g in grads) or not any(g.abs().max() > 0 for g in grads):
                raise ValueError(f'invalid gradients: {prefix}')
        opt.step()
        if not any(not torch.equal(before[n], p) for n, p in model.named_parameters()):
            raise ValueError('no parameter changed')
        if any(not torch.equal(buffers[n], b) for n, b in model.named_buffers()):
            raise ValueError('normalization buffers changed')
        return {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'fixture_loss': float(loss.detach().cpu()),
                'scope': 'silence and short random fixtures, padded-neighbour and gradient checks'}

    check('scheduler_fixture', scheduler_fixture)
    check('gpu_frontend_and_gradients_fixture', gpu_fixtures)

    def real_data():
        if shard_dir is None:
            raise ImportError('shard directory not supplied')
        from echoproof.training.r3_m1_data import build_data
        data = build_data(root, shard_dir, seed=42)
        try:
            if data['input_sha256'] != evidence['input_sha256'] or data['split_sha256'] != evidence['selection']['split_sha256']:
                raise ValueError('input hashes do not match original experiment')
            selection = data['validation']['selection']
            train_ids, selection_ids, train_speakers, selection_speakers = validate_coverage(data['train'].items, selection.chunks)
            for rid in sorted(train_ids | selection_ids):
                data['loader'](rid)  # The verified loader checks decoded PCM and sample count.
            return {'train_recordings': len(train_ids), 'selection_recordings': len(selection_ids),
                    'train_speakers': len(train_speakers), 'selection_speakers': len(selection_speakers),
                    'selection_chunks': len(selection.chunks), 'pcm_verified_recordings': len(train_ids | selection_ids),
                    'input_sha256': data['input_sha256'], 'audio_scope': 'train and selection only'}
        finally:
            data['loader'].close()

    def frozen_normalization():
        if shard_dir is None:
            raise ImportError('shard directory not supplied')
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise ImportError('CUDA GPU unavailable')
        from echoproof.features.acoustic import log_mel
        from echoproof.models.bigru_waveform import WaveformBiGRU
        from echoproof.training.r3_m1_data import build_data
        data = build_data(root, shard_dir, seed=42)
        try:
            stats, ds = data['mel_stats'], data['validation']['selection']
            model = WaveformBiGRU(stats['mean'], stats['std'], hidden_size=4, dropout=0).cuda().eval()
            # Compare a full window, a short window when present, and a last/tail window.
            chunks = ds.chunks
            selected = [0, len(chunks)-1]
            short = next((i for i, c in enumerate(chunks) if c.valid_samples < c.input_samples), None)
            if short is not None:
                selected.append(short)
            diffs = []
            with torch.no_grad():
                for i in sorted(set(selected)):
                    item = ds[i]
                    wave = item['waveform'][:item['valid_samples']]
                    actual, lengths = model.features(torch.tensor(item['waveform'][None, :], device='cuda'),
                                                      torch.tensor([item['valid_samples']], device='cuda'))
                    expected = (log_mel(wave) - stats['mean']) / stats['std']
                    got = actual[0, :int(lengths[0])].cpu().numpy()
                    np.testing.assert_allclose(got, expected, atol=2e-4, rtol=2e-4)
                    diffs.append(float(np.max(np.abs(got-expected))))
            return {'selection_chunks_checked': len(diffs), 'short_chunk_found': short is not None,
                    'max_abs_difference': max(diffs), 'stats_sha256': data['input_sha256']['stats']}
        finally:
            data['loader'].close()

    def recovery_fixtures():
        import torch  # Ensure missing PyTorch is reported as not measured.
        scratch = EXPERIMENT / 'runs' / 'audit_tmp'
        scratch.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='recovery_', dir=scratch))
        env = os.environ.copy()
        env.update({'PYTHONPATH': str(root / 'src'), 'TMPDIR': str(work), 'TEMP': str(work), 'TMP': str(work),
                    'PYTHONDONTWRITEBYTECODE': '1'})
        tests = ['test_resume_exact_with_same_config', 'test_resume_guards', 'test_resume_reproduces_next_epoch_exactly']
        targets = [str(root / 'tests/test_m1_trainer.py') + '::TrainerBehaviour::' + t for t in tests]
        run_test = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *targets],
                                  cwd=work, env=env, capture_output=True, text=True)
        if run_test.returncode != 0 or ' skipped' in run_test.stdout:
            raise ValueError('recovery fixtures failed or skipped: ' + (run_test.stdout + run_test.stderr)[-2000:])
        return {'tests': tests, 'summary': run_test.stdout.strip(), 'torch': torch.__version__,
                'scope': 'ordinary-module epoch-boundary recovery; not CUDA GRU bit-exact continuation'}

    check('real_data_and_pcm', real_data)
    check('gpu_full_frontend_and_frozen_normalization', frozen_normalization)
    check('epoch_boundary_recovery_fixtures', recovery_fixtures)
    def crash_window_guard():
        import torch  # Required to inspect actual checkpoint metadata.
        from checkpoint_guard import check_run, validate_state, GuardError
        latest = {'epoch': 1, 'best_epoch': 1, 'best_metric': .8, 'config_hash': 'fixture'}
        best = {'epoch': 1, 'metrics': {'macro_f1': .8}, 'config_hash': 'fixture'}
        history = [{'epoch': 1, 'val': {'macro_f1': .8}}]
        validate_state(latest, best, history, [[1], [1]])
        interrupted = history + [{'epoch': 2, 'val': {'macro_f1': .9}}]
        try:
            validate_state(latest, best, interrupted, [[1], [1]])
        except GuardError:
            pass
        else:
            raise ValueError('guard allowed history ahead of the latest checkpoint')
        results = {name: check_run(run / (run.name + '-' + name)) for name in ('baseline', 'lower_lr')}
        return {'original_checkpoint_consistency': results, 'fixture_crash_window_rejected': True,
                'policy': 'future runner must invoke guard before resume; inconsistent artifacts are preserved and resume refused',
                'limitation': 'structural guard; no automatic repair or CUDA bit-exact resume guarantee'}
    check('multi_file_crash_window', crash_window_guard)
    statuses = [c['status'] for c in checks.values()]
    result['overall'] = 'failed' if 'failed' in statuses else 'incomplete' if 'not_measured' in statuses else 'passed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--shard-dir', type=Path, help='canonical shard directory for train/selection PCM checks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(EXPERIMENT.resolve() / 'reports'):
        raise ValueError('audit reports must be under experiments/r3_m1/reports')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = audit(args.root, args.shard_dir)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result['checks'], indent=2))
    print('Audit overall:', result['overall'], 'report:', output)
    return 1 if result['overall'] == 'failed' else 2 if result['overall'] == 'incomplete' else 0


if __name__ == '__main__':
    sys.exit(main())
