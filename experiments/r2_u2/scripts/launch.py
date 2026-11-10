"""Fresh Option B experiment only; never writes official R2/R3 outputs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from contract import EXP, sha, sources, fingerprint
from acceptance import require, validate_events, verify_software


def read(p):
    return json.loads(Path(p).read_text())


def write_new(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--software-report', required=True)
    parser.add_argument('--authorize-fresh-experiment', action='store_true')
    args = parser.parse_args()
    require(args.authorize_fresh_experiment, 'Explicit fresh experiment launch flag required')
    require(sys.platform == 'linux' and str(ROOT).startswith('/content/echoproof-'), 'Launch only from verified Colab scratch')
    data = Path(args.data_root).resolve()
    require(data != ROOT and str(data).startswith('/content/drive/'), 'Persistent private Drive project required')
    manifest = read(EXP / 'colab_source_manifest.json')
    for name, expected in manifest['files'].items():
        require(sha(ROOT / name) == expected, 'Staged source mismatch: ' + name)
    report_path = Path(args.software_report).resolve()
    require(report_path.is_relative_to(data / 'experiments/r2_u2/reports'), 'Wrong software-report location')
    software = read(report_path)
    verify_software(software, manifest['files'], 17)
    require(software.get('manifest_sha256') == sha(EXP / 'colab_source_manifest.json'), 'Software manifest checksum differs')
    for name in ('junit.xml', 'pytest.log'):
        require(sha(report_path.parent / name) == software.get('test_artifact_sha256', {}).get(name), 'Software artifact changed: ' + name)
    import xml.etree.ElementTree as ET
    cases = list(ET.parse(report_path.parent / 'junit.xml').iter('testcase'))
    require(len(cases) == 17 and not any(c.find(tag) is not None for c in cases for tag in ('failure', 'error', 'skipped')),
            'JUnit gate failed')
    import torch
    from echoproof.training.trainer import TrainConfig
    from echoproof.training.datasets import CropDataset, ChunkDataset, crop_start
    from echoproof.training.seeding import seed_everything
    from echoproof.contracts import Chunk, SplitAssignment, read_parquet
    from echoproof.data.cohort import read_table
    from echoproof.features.pretrained import pinned_snapshot, PRETRAINED
    from echoproof.models.ast_detector import ASTDetector, BACKBONE
    from numerical_trainer import NumericalTrainer
    import scope
    import importlib.metadata
    require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Tesla T4', 'Recorded T4 required')
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'pyarrow', 'pytest', 'transformers',
                 'torchaudio', 'soundfile', 'safetensors', 'huggingface-hub')}
    require(versions['transformers'] == '5.22.0', 'Pinned Transformers version required')
    require(versions['torch'] == software['environment']['torch'] and versions['numpy'] == software['environment']['numpy']
            and versions['pyarrow'] == software['environment']['pyarrow']
            and versions['pytest'] == software['environment']['pytest']
            and sys.version == software['environment']['python'], 'Training/software runtime differs')
    require(versions['torch'] == '2.14.1+cu130', 'Use the verified Account B stack')
    recipe_path = EXP / 'configs/historical_recipe.json'
    require(sha(recipe_path) == '61245ca636951484a8cba9addbdd84fec8d7f335c945d2adb046666d306c45a2', 'Frozen recipe bytes changed')
    recipe = read(recipe_path)
    for name, expected in recipe['input_sha256'].items():
        require(sha(data / name) == expected, 'Frozen input changed: ' + name)
    require(read(data / 'reports/r1/leakage.json')['passed'] is True, 'Leakage evidence failed')
    plan = scope.select_plan(read_table(data / 'data/manifests/cohort_v1.parquet'),
                             read_parquet(str(data / 'data/splits/protocol_B_v1.parquet'), SplitAssignment),
                             read_parquet(str(data / 'data/processed/chunks_v1.parquet'), Chunk))
    require(plan.counts['train_recordings'] == 4530 and plan.counts['selection_recordings'] == 2000
            and plan.counts['selection_chunks'] == 10029, 'Scope counts changed')
    require(plan.split_sha256 == read(data / 'reports/r1/split_report.json')['protocols']['B']['split_sha256'], 'Split changed')
    stage = ROOT / 'audio_shards'
    stage.mkdir(exist_ok=False)
    total = sum((data / 'data/processed/cohort_v1' / n).stat().st_size for n in plan.shards())
    require(shutil.disk_usage(ROOT).free > total + 2_000_000_000, 'Insufficient scratch space')
    shards = {}
    for name in plan.shards():
        src = data / 'data/processed/cohort_v1' / name
        before = sha(src)
        print('Staging', name, flush=True)
        with src.open('rb') as f, (stage / name).open('xb') as g:
            shutil.copyfileobj(f, g, 1024 * 1024)
        require(sha(stage / name) == before, 'Staged shard differs')
        shards[name] = before
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(recipe['train']['seed'])
    snapshot = pinned_snapshot('ast', cache_dir=ROOT / 'model_cache')
    reference = read(EXP / 'configs/model_reference.json')
    model = ASTDetector.from_dir(snapshot, device='cuda', seed=recipe['train']['seed'])
    model.model.config.id2label = {0: 'genuine', 1: 'tts'}
    model.model.config.label2id = {'genuine': 0, 'tts': 1}
    require(model.loading['ok'] and model.loading['shapes']['ok'], 'AST load audit failed')
    require(model.digests() == reference['initial_group_sha256']
            and scope.frozen_state_digest(model) == reference['initial_frozen_state_sha256'], 'Initial model reference differs')
    require(all(getattr(model.feature_extractor, k) == v for k, v in recipe['preprocessor'].items()), 'Processor differs')
    require(sha(snapshot / 'preprocessor_config.json') == reference['model']['processor_file_sha256'], 'Processor bytes differ')
    identity = {'revision': snapshot.name, 'weights_sha256': sha(snapshot / PRETRAINED['ast']['weights_file']),
                'repo_id': PRETRAINED['ast']['repo_id'], 'processor': recipe['preprocessor']}
    model.set_stage('head')
    trainability = scope.validate_head_stage(model)
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'Parameters must stay FP32')
    contract = {'policy': 'option_b_v1', 'data_root': str(data), 'split_sha256': plan.split_sha256,
                'acceptance_policy': read(EXP / 'configs/option_b_policy.json'),
                'sources': sources(), 'manifest_sha256': sha(EXP / 'colab_source_manifest.json'),
                'software_report_sha256': sha(report_path), 'environment': versions,
                'runtime': {'python': sys.version, 'gpu': torch.cuda.get_device_name(0),
                            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
                'shared_snapshot': read(EXP / 'software_sources/snapshot.json'), 'plan_sha256': plan.sha256,
                'shards': shards, 'official_promotion_authorized': False, 'resume_authorized': False,
                'skip_threshold': None, 'selection': recipe['checkpoint_policy'],
                'norm_definition': 'finite pre-clipping norms of applied updates only'}
    run = data / 'experiments/r2_u2/runs' / ('option_b_' + uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    print('NEW EXPERIMENT:', run, flush=True)
    loader = None
    hooks = []
    forwards = {'train': 0, 'eval': 0}
    result = {'research_accepted': False, 'official_promotion_authorized': False, 'status': 'experiment_failed',
              'run_id': run.name, 'created_utc': datetime.now(timezone.utc).isoformat()}
    try:
        loader = scope.CheckedWaveformLoader(stage, plan.train_rows + plan.selection_rows)
        train = CropDataset([(r['recording_id'], r['label']) for r in plan.train_rows], loader, seed=42)
        selection = ChunkDataset(plan.selection_chunks, {r['recording_id']: r['label'] for r in plan.selection_rows}, loader)
        def guard(module, inputs):
            scope.validate_head_stage(module)
            require(not any(m.training for m in getattr(module.model, BACKBONE).modules()), 'Backbone reentered training')
            require(all(p.grad is None for p in getattr(module.model, BACKBONE).parameters()), 'Backbone received gradients')
            forwards['train' if module.training else 'eval'] += 1
        hooks.append(model.register_forward_pre_hook(guard))
        cfg = TrainConfig(**recipe['train'])
        trainer = NumericalTrainer(model, cfg, train, selection, run, model_identity=identity,
                    research_contract=contract, param_groups=model.param_groups(head_lr=cfg.lr, backbone_lr=cfg.lr,
                                                                                weight_decay=cfg.weight_decay))
        # The fresh-run guard must see an empty directory. Persist inputs only
        # after initialization, exclusively and before any training begins.
        write_new(run / 'contract.json', contract)
        write_new(run / 'initial_baseline.json', {'initial_group_sha256': model.digests(),
                  'initial_frozen_state_sha256': scope.frozen_state_digest(model), 'recipe': recipe, 'model': identity})
        require(trainer.scaler.get_scale() == 65536 and trainer.scaler.get_growth_interval() == 2000
                and trainer.scaler.get_backoff_factor() == .5 and trainer.scaler.get_growth_factor() == 2,
                'Default scaler policy changed')
        summary = trainer.fit()
        latest = torch.load(run / 'latest.pt', map_location='cpu', weights_only=True)
        best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        scope._finite_tensors(latest); scope._finite_tensors(best)
        require(latest['finished'] and latest['epoch'] == 2 and summary['epochs_run'] == 2, 'Incomplete experiment')
        require(latest['config_hash'] == best['config_hash'] == trainer.config_hash, 'Checkpoint provenance differs')
        counts = validate_events(trainer.events)
        by_id = {r['recording_id']: r for r in plan.train_rows}
        for epoch in (1, 2):
            seen = []
            for event in (e for e in trainer.events if e['epoch'] == epoch):
                require(len(event['recording_ids']) == len(event['crop_ids']) == event['denominator'], 'Crop coverage mismatch')
                for rid, cid in zip(event['recording_ids'], event['crop_ids']):
                    require(rid in by_id and cid == f'crop:{epoch}:{rid}:{crop_start(42, epoch, rid, by_id[rid]["n_samples"])}', 'Crop identity differs')
                    seen.append(rid)
            require(len(seen) == len(set(seen)) == len(by_id) and set(seen) == set(by_id), 'Epoch recording coverage differs')
        require(latest['numerical_engineering']['events'] == trainer.events
                and latest['numerical_engineering']['events_sha256'] == fingerprint(trainer.events), 'Saved telemetry differs')
        disk_events = [json.loads(x) for x in (run / 'updates.jsonl').read_text().splitlines()]
        require(disk_events == trainer.events, 'Durable update log differs')
        applied = sum(c['applied'] for c in counts)
        require(latest['numerical_engineering']['attempted'] == len(disk_events)
                and latest['numerical_engineering']['applied'] == applied, 'Saved accounting differs')
        require(latest['skipped_steps'] == sum(c['scaler_skipped'] for c in counts), 'Skipped count mismatch')
        require(len(latest['optimizer']['state']) == 4
                and all(float(s['step']) == applied for s in latest['optimizer']['state'].values()), 'AdamW counters differ')
        require(all((s['exp_avg_sq'] >= 0).all().item() for s in latest['optimizer']['state'].values()), 'Invalid second moments')
        history = [json.loads(x) for x in (run / 'history.jsonl').read_text().splitlines()]
        require([h['epoch'] for h in history] == [1, 2], 'History coverage differs')
        for h, c in zip(history, counts):
            require(all(isinstance(h[k], (int, float)) and __import__('math').isfinite(h[k]) for k in ('train_loss', 'grad_norm', 'lr')),
                    'Non-finite training metric')
            require(h['grad_norm'] == c['norm_mean'], 'Norm mean differs')
            require(all(__import__('math').isfinite(v) for v in h['val'].values() if isinstance(v, (int, float))), 'Invalid validation metrics')
            require(h['train_loss'] >= 0 and h['lr'] == .0001
                    and all(0 <= h['val'][k] <= 1 for k in ('macro_f1', 'eer', 'fpr'))
                    and h['val']['loss'] >= 0, 'Metric outside valid range')
        norm_stats = latest['numerical_engineering']['norm_statistics']
        require(len(norm_stats) == 2, 'Norm summary coverage differs')
        for s, c in zip(norm_stats, counts):
            require((s['epoch'], s['count'], s['mean'], s['minimum'], s['maximum']) ==
                    (c['epoch'], c['applied'], c['norm_mean'], c['norm_min'], c['norm_max']), 'Saved norm summary differs')
        selected = max(history, key=lambda h: h['val']['macro_f1'])
        require(best['epoch'] == latest['best_epoch'] == selected['epoch'] and best['metrics'] == selected['val'], 'Selection rule differs')
        require(forwards['train'] > 0, 'No actual training-mode observation')
        proofs = {}
        for name, ck in [('latest', latest), ('best', best)]:
            model.load_state_dict(ck['model'], strict=True)
            proofs[name] = scope._state_proof(model, reference)
        metrics, chunks, recordings = trainer.evaluate()
        require(all(__import__('math').isfinite(float(v)) for v in metrics.values()), 'Invalid reload metrics')
        require(all(__import__('math').isclose(metrics[k], v, abs_tol=1e-4, rel_tol=1e-4) for k, v in best['metrics'].items()), 'Reload metrics differ')
        parity = scope._prediction_parity(run, plan, chunks, recordings, best['epoch'])
        require(loader.verified_ids == set(loader.rows), 'PCM verification incomplete')
        require(sources() == contract['sources'], 'Sources changed during experiment')
        import csv
        with (run / 'ledger.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        require(len(rows) == 1 and rows[0]['status'] == 'completed'
                and rows[0]['config_hash'] == trainer.config_hash and rows[0]['run_id'] == run.name
                and int(rows[0]['best_epoch']) == best['epoch'], 'Private completion ledger differs')
        for name, expected in recipe['input_sha256'].items():
            require(sha(data / name) == expected, 'Frozen input changed during run')
        result.update(status='experiment_checks_passed_pending_team_review', experimental_checks_passed=True,
                      final_attempt_skipped=trainer.events[-1]['scaler_skipped'],
                      overflow_events=[e for e in trainer.events if e['scaler_skipped']],
                      epoch_update_summaries=counts, summary=summary, freeze_proofs=proofs, reload_parity=parity,
                      selection_metrics=metrics, forwards=forwards, pcm_verified=len(loader.verified_ids),
                      proposed_R3_initializer='best.pt; prohibited until separate team approval')
    except BaseException:
        result.update(experimental_checks_passed=False, traceback=traceback.format_exc())
    finally:
        for hook in hooks: hook.remove()
        if loader is not None:
            loader.close()
        result['artifacts'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in run.iterdir() if p.is_file()}
        write_new(run / 'experiment_result.json', result)
        report_dir = data / 'experiments/r2_u2/reports' / run.name
        report_dir.mkdir(parents=True, exist_ok=False)
        write_new(report_dir / 'experiment_result.json', result)
        print(json.dumps({'run': str(run), 'status': result['status'], 'research_accepted': False}), flush=True)
    return 0 if result.get('experimental_checks_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
