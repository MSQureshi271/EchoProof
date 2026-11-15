"""Exercise the actual shared trainer with small fixture data, no audio or GPU needed."""
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')

from echoproof.training.r3_m1_selection import selection_trainer
from echoproof.training.r4_m1_bigru import verify_resume
from echoproof.training.trainer import TrainConfig


class Dataset:
    def __init__(self, validation=False):
        self.validation_role = 'selection' if validation else None
        self.items = [(f'recording:{i}', i % 2) for i in range(8)]
        self.labels_by_recording = dict(self.items)
        self.chunks = [SimpleNamespace(chunk_id=f'chunk:{rid}', recording_id=rid) for rid, _ in self.items]
        self.epoch = 0

    @property
    def labels(self):
        return [label for _, label in self.items]

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        rid, label = self.items[index]
        return {'recording_id': rid, 'chunk_id': f'chunk:{rid}', 'label': label,
                'waveform': torch.tensor([float(label), 0.25, -0.5, 0.75]).numpy(), 'valid_samples': 4}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(4, 2)
    def forward(self, waveforms, valid_samples):
        return self.linear(waveforms)


def test_training_and_completed_resume(tmp_path):
    cfg = TrainConfig(max_epochs=2, patience=3, batch_size=4, eval_batch_size=4,
                      device='cpu', protocol='A', model_name='bigru', owner='Muneeb')
    train, val = Dataset(), Dataset(validation=True)
    first = selection_trainer(Model(), cfg, train, val, tmp_path/'run', provenance={'fixture': 'frozen'})
    summary = first.fit()
    latest = verify_resume(tmp_path/'run')
    assert latest['finished'] and summary['epochs_run'] == 2
    assert 1 <= summary['best_epoch'] <= 2
    resumed = selection_trainer(Model(), cfg, train, val, tmp_path/'run', provenance={'fixture': 'frozen'})
    assert resumed.fit(resume=True)['best_epoch'] == summary['best_epoch']
    wrong = selection_trainer(Model(), cfg, train, val, tmp_path/'run', provenance={'fixture': 'changed'})
    with pytest.raises(Exception, match='different config/provenance'):
        wrong.fit(resume=True)
