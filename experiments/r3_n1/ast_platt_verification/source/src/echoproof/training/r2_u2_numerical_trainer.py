"""Initial experimental AMP telemetry. Shared Trainer remains unchanged.

Research fitting requires a fresh prospective contract. Official publication
remains unavailable; historical checkpoints cannot be resumed as this experiment.
The inherited history grad_norm is explicitly the APPLIED-update mean here;
this is experimental telemetry, not an amendment of official acceptance rules.
"""
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path

import torch
from echoproof.training.trainer import Trainer, TrainingError, ResumeError
from .r2_u2_contract import EXP, provenance, fingerprint, sources


class NumericalTrainer(Trainer):
    def __init__(self, model, config, train_dataset, eval_dataset, run_dir, *,
                 model_identity, fixture_only=False, resume_requested=False, research_contract=None, **kwargs):
        if not fixture_only and (resume_requested or not research_contract):
            raise ValueError('Fresh research experiment requires a prospective contract; resume is disabled')
        path = Path(run_dir).resolve()
        allowed = (EXP / 'runs').resolve() if fixture_only else Path(research_contract['data_root']).resolve() / 'experiments/r2_u2/runs'
        if not path.is_relative_to(allowed) or path == allowed:
            raise ValueError('Outputs must be isolated under experiments/r2_u2/runs/<unique_id>')
        if path.exists() and any(path.iterdir()) and not resume_requested:
            raise ResumeError('Existing run; never overwrite it')
        if 'ledger_path' in kwargs or 'provenance' in kwargs:
            raise ValueError('Experiment controls ledger and provenance')
        self.fixture_only = fixture_only
        if fixture_only:
            identity = provenance({'train': asdict(config)}, model_identity, fixture=True)
        else:
            recipe = json.loads((EXP / 'configs/historical_recipe.json').read_text())
            if asdict(config) != recipe['train']:
                raise ValueError('Research TrainConfig differs from frozen recipe')
            identity = provenance(recipe, model_identity, research_contract=research_contract)
        super().__init__(model, config, train_dataset, eval_dataset, path,
                         ledger_path=EXP / 'ledger.csv' if fixture_only else path / 'ledger.csv', provenance=identity, **kwargs)
        self.events_path = self.run_dir / 'updates.jsonl'
        self.events = []
        self.norm_statistics = []
        self.applied = 0
        self.epoch_number = 0
        self.group_sizes = []
        self.group_recordings = []
        self.group_crops = []
        self.denominator = None
        self._step_calls = 0
        self._step_hook = self.optimizer.register_step_post_hook(self._observed_step)

    def _observed_step(self, optimizer, args, kwargs):
        self._step_calls += 1

    def _accumulate(self, group):
        self.group_sizes = [len(b['labels']) for b in group]
        self.group_recordings = [rid for b in group for rid in b.get('recording_ids', [])]
        self.group_crops = [cid for b in group for cid in b.get('chunk_ids', [])]
        loss, denominator = super()._accumulate(group)
        self.denominator = denominator
        return loss, denominator

    def _record(self, event):
        before = event.pop('_calls_before', None)
        if before is not None:
            event['optimizer_step_calls_observed'] = self._step_calls - before
        self.events.append(event)
        with self.events_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())

    def _optimizer_step(self):
        grads = [p.grad for p in self.trainable if p.grad is not None]
        event = {'attempt': len(self.events) + 1, 'epoch': self.epoch_number,
                 'microbatch_sizes': self.group_sizes, 'denominator': self.denominator,
                 'recording_ids': self.group_recordings, 'crop_ids': self.group_crops,
                 'scale_before': self.scaler.get_scale(), 'scale_after': None,
                 'applied': False, 'scaler_skipped': False, 'norm_only_overflow': False,
                 'nonfinite_scaled_elements': False, 'nonfinite_unscaled_elements': False,
                 'preclip_norm': None, 'scheduler_clock': 'legacy_attempts',
                 'scheduler_advanced': False, 'lr_before': [g['lr'] for g in self.optimizer.param_groups]}
        before = self._step_calls
        event['_calls_before'] = before
        try:
            if not grads:
                raise TrainingError('No gradients')
            finite = lambda: all(torch.isfinite(g).all().item() for g in grads)
            event['nonfinite_scaled_elements'] = not finite()
            self.scaler.unscale_(self.optimizer)
            event['nonfinite_unscaled_elements'] = not finite()
            if event['nonfinite_unscaled_elements']:
                if not self.scaler.is_enabled() or not event['nonfinite_scaled_elements']:
                    raise TrainingError('Non-finite gradients without safely detected scaled overflow')
                # Do not clip rejected gradients. The original scaler already
                # recorded their nonfiniteness at unscale; let it recover.
            else:
                if self.cfg.grad_clip > 0:
                    try:
                        norm = float(torch.nn.utils.clip_grad_norm_(self.trainable, self.cfg.grad_clip,
                                                                   error_if_nonfinite=True))
                    except RuntimeError as exc:
                        if 'non-finite' not in str(exc):
                            raise
                        event['norm_only_overflow'] = True
                        raise TrainingError('Finite elements but non-finite norm; no optimizer update')
                else:
                    norm = float(torch.norm(torch.stack([g.norm() for g in grads])))
                    if not math.isfinite(norm):
                        event['norm_only_overflow'] = True
                        raise TrainingError('Finite elements but non-finite norm')
                event['preclip_norm'] = norm
                if not finite():
                    raise TrainingError('Clipping produced non-finite elements')
            self.scaler.step(self.optimizer)
            self.scaler.update()
            event['scale_after'] = self.scaler.get_scale()
            calls = self._step_calls - before
            if calls not in (0, 1):
                raise TrainingError('Unexpected optimizer execution count')
            event['applied'] = calls == 1
            if not calls:
                if not (event['nonfinite_scaled_elements'] and self.scaler.is_enabled()
                        and event['scale_after'] < event['scale_before']):
                    raise TrainingError('Unexplained missing optimizer update')
                event['scaler_skipped'] = True
                self.skipped_steps += 1
            else:
                if event['nonfinite_unscaled_elements']:
                    raise TrainingError('Unsafe optimizer update')
                self.applied += 1
                if not all(torch.isfinite(p).all().item() for p in self.trainable):
                    raise TrainingError('Non-finite updated parameter')
                for state in self.optimizer.state.values():
                    if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item() for v in state.values()):
                        raise TrainingError('Non-finite optimizer state')
            # Preserve shared Trainer scheduling. R2-U2 is constant. Changing
            # nonconstant schedules to applied-update time needs team approval.
            if self.step_scheduler is not None:
                self.step_scheduler.step()
                event['scheduler_advanced'] = True
            event['applied_clock_would_advance'] = event['applied'] and self.step_scheduler is not None
            event['scheduler_last_epoch'] = self.step_scheduler.last_epoch if self.step_scheduler else None
            event['lr_after'] = [g['lr'] for g in self.optimizer.param_groups]
            self._record(event)
            return event['preclip_norm'] if event['applied'] else None
        except BaseException as exc:
            event['error'] = str(exc)
            self._record(event)
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)

    def _train_epoch(self, epoch):
        self.epoch_number = epoch
        self.train_ds.set_epoch(epoch)
        self._set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)
        iterator = iter(self._train_loader(epoch))
        loss_total = weight_total = 0.0
        norms = []
        while True:
            group = list(itertools.islice(iterator, self.cfg.accumulation_steps))
            if not group:
                break
            loss, weight = self._accumulate(group)
            loss_total += loss
            weight_total += weight
            norm = self._optimizer_step()
            if norm is not None:
                norms.append(norm)
        if not norms:
            raise TrainingError('Epoch has no applied updates; norm statistic undefined')
        self.norm_statistics.append({'epoch': epoch, 'count': len(norms),
                                     'mean': math.fsum(norms) / len(norms),
                                     'minimum': min(norms), 'maximum': max(norms),
                                     'definition': 'finite pre-clipping norms of applied updates only'})
        # History field is inherited; its experimental definition is explicit.
        return loss_total / weight_total, math.fsum(norms) / len(norms)

    def _checkpoint(self, epoch, state, finished):
        result = super()._checkpoint(epoch, state, finished)
        result['numerical_engineering'] = {'schema': 1, 'events': self.events,
                                         'attempted': len(self.events),
                                         'events_sha256': fingerprint(self.events), 'applied': self.applied,
                                         'norm_statistics': self.norm_statistics,
                                         'scheduler_clock': 'legacy_attempts', 'research_accepted': False}
        return result

    def _load_resume(self):
        saved = torch.load(self.run_dir / 'latest.pt', map_location='cpu', weights_only=True)
        def finite_state(value):
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all().item():
                    raise ResumeError('Non-finite checkpoint tensor')
            elif isinstance(value, dict):
                for item in value.values():
                    finite_state(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    finite_state(item)
        finite_state(saved)
        proof = saved.get('numerical_engineering', {})
        disk = [json.loads(line) for line in self.events_path.read_text().splitlines()]
        history = [json.loads(line) for line in (self.run_dir / 'history.jsonl').read_text().splitlines()]
        if (proof.get('schema') != 1 or proof.get('events') != disk
                or proof.get('events_sha256') != fingerprint(disk)
                or proof.get('attempted') != len(disk)
                or any('error' in e for e in disk)
                or [h['epoch'] for h in history] != list(range(1, saved['epoch'] + 1))
                or sum(e['applied'] for e in disk) != proof.get('applied')
                or sum(e['scaler_skipped'] for e in disk) != saved['skipped_steps']
                or len(disk) != proof.get('applied', 0) + saved['skipped_steps']):
            raise ResumeError('Telemetry/history/checkpoint mismatch; preserve and inspect')
        best = torch.load(self.run_dir / 'best.pt', map_location='cpu', weights_only=True)
        finite_state(best)
        stats = proof.get('norm_statistics', [])
        if (len(stats) != len(history) or any(s['mean'] != h['grad_norm'] or not math.isfinite(s['mean'])
                or s['count'] != sum(e['applied'] for e in disk if e['epoch'] == h['epoch'])
                for s, h in zip(stats, history))):
            raise ResumeError('Applied norm statistics differ from history/events')
        if best['epoch'] != saved['best_epoch'] or best['metrics'] != history[best['epoch'] - 1]['val']:
            raise ResumeError('Best checkpoint/history mismatch')
        import pyarrow.parquet as pq
        for name in ('predictions_chunks', 'predictions_recordings'):
            epochs = pq.read_table(self.run_dir / (name + '.parquet'), columns=['epoch']).to_pydict()['epoch']
            if not epochs or set(epochs) != {best['epoch']}:
                raise ResumeError('Prediction/checkpoint mismatch')
        result = super()._load_resume()
        self.events, self.applied = disk, proof['applied']
        self.norm_statistics = proof['norm_statistics']
        return result

    def fit(self, resume=False):
        if resume and not self.fixture_only:
            raise ResumeError('Initial Option B research launch is fresh-only')
        if sources() != self.provenance['sources']:
            raise ResumeError('Experiment or shared source changed')
        torch.use_deterministic_algorithms(self.cfg.deterministic)
        torch.backends.cudnn.deterministic = self.cfg.deterministic
        torch.backends.cudnn.benchmark = False
        return super().fit(resume=resume)
