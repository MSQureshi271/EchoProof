"""Actual torch/GradScaler fixtures, CPU AMP where possible; never research audio."""
import copy
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')
pytest.importorskip('pyarrow')
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parents[1] / 'scripts')]
from contract import new_run
from numerical_trainer import NumericalTrainer
from echoproof.training.trainer import Trainer, TrainConfig, TrainingError, ResumeError
from echoproof.training.datasets import CropDataset, DictWaveformLoader


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(4, 2)

    def forward(self, waveforms, valid_samples):
        return self.dense(waveforms)


@pytest.fixture
def make():
    paths = []
    def build(scheduler='constant', resume=None, seed=42):
        ds = CropDataset([(str(i), i % 2) for i in range(18)],
                         DictWaveformLoader({str(i): np.arange(4, dtype=np.float32) / 4 for i in range(18)}),
                         seed=seed, crop_samples=4)
        path = resume or new_run()
        if path not in paths:
            paths.append(path)
        cfg = TrainConfig(seed=seed, max_epochs=2, batch_size=4, accumulation_steps=4,
                          device='cpu', scheduler=scheduler, patience=3)
        t = NumericalTrainer(Model(), cfg, ds, ds, path, model_identity={'fixture': 'linear4'},
                             fixture_only=True, resume_requested=resume is not None)
        return t
    yield build
    for path in paths:
        shutil.rmtree(path)


def gradients(t, value):
    for p in t.trainable:
        p.grad = torch.full_like(p, value)


def arm_scaler(t):
    t.scaler = torch.amp.GradScaler('cpu')
    t.scaler.scale(torch.tensor(1.0))  # initialize actual library scale state


def test_real_scaler_skip_and_scheduler_accounting(make):
    t = make(scheduler='cosine')
    arm_scaler(t)
    before = [p.detach().clone() for p in t.trainable]
    start = t.step_scheduler.last_epoch
    gradients(t, float('inf'))
    assert t._optimizer_step() is None
    e = t.events[-1]
    assert e['scaler_skipped'] and not e['applied']
    assert t.applied == 0 and t.skipped_steps == 1
    assert e['scale_after'] < e['scale_before']
    assert t.step_scheduler.last_epoch == start + 1  # deliberately retained legacy clock
    assert not e['applied_clock_would_advance']
    assert not t.optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(before, t.trainable))
    gradients(t, 1.)
    t._optimizer_step()
    assert t.applied == 1 and t.skipped_steps == 1 and len(t.events) == 2


def test_norm_only_overflow_fails_before_update(make):
    t = make()
    gradients(t, 1e30)
    with pytest.raises(TrainingError):
        t._optimizer_step()
    assert t.events[-1]['norm_only_overflow']
    assert t.skipped_steps == t.applied == 0
    assert not t.optimizer.state


def test_non_amp_nonfinite_rejected(make):
    t = make()
    gradients(t, float('nan'))
    with pytest.raises(TrainingError):
        t._optimizer_step()
    assert not t.optimizer.state


def test_partial_group_and_finite_path_equivalence(make):
    t = make()
    reference = make()
    reference.model.load_state_dict(copy.deepcopy(t.model.state_dict()))
    reference._accumulate = types.MethodType(Trainer._accumulate, reference)
    reference._optimizer_step = types.MethodType(Trainer._optimizer_step, reference)
    expected = Trainer._train_epoch(reference, 1)
    actual = t._train_epoch(1)
    assert actual == pytest.approx(expected)
    assert t.events[-1]['microbatch_sizes'] == [2]
    assert t.events[-1]['denominator'] == 2
    assert t.norm_statistics[-1]['count'] == 2
    assert all(torch.equal(a, b) for a, b in zip(t.model.parameters(), reference.model.parameters()))


def test_all_skipped_epoch_rejected(make, monkeypatch):
    t = make()
    monkeypatch.setattr(t, '_optimizer_step', lambda: None)
    with pytest.raises(TrainingError, match='no applied'):
        t._train_epoch(1)


def test_nonfinite_loss_rejected_before_update(make):
    t = make()
    with torch.no_grad():
        t.model.dense.weight.fill_(float('nan'))
    with pytest.raises(TrainingError, match='loss'):
        t._train_epoch(1)
    assert not t.optimizer.state


def test_checkpoint_resume_and_changed_provenance(make, monkeypatch):
    t = make()
    # Fixture eval records use the same interface as Trainer.evaluate.
    def evaluate(self):
        return {'macro_f1': .5, 'eer': .5, 'fpr': .5, 'loss': 1.}, [('c', 'r', 0, .4)], [('r', 0, .4, 1)]
    monkeypatch.setattr(NumericalTrainer, 'evaluate', evaluate)
    t.fit()
    resumed = make(resume=t.run_dir)
    resumed.fit(resume=True)
    assert resumed.applied == t.applied
    changed = make(resume=t.run_dir, seed=43)
    with pytest.raises(ResumeError):
        changed.fit(resume=True)
    with t.events_path.open('a') as f:
        f.write(json.dumps(t.events[-1]) + '\n')
    with pytest.raises(ResumeError):
        make(resume=t.run_dir).fit(resume=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA numerical verification pending')
def test_actual_cuda_fp16_backward_skip(make):
    t = make()
    t.model.cuda()
    t.scaler = torch.amp.GradScaler('cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        loss = t.model.dense(torch.ones(2, 4, device='cuda')).sum() * 1e5
    t.scaler.scale(loss).backward()
    t._optimizer_step()
    assert t.events[-1]['scaler_skipped'] and t.applied == 0
