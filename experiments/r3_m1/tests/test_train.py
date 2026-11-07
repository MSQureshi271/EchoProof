import importlib.util
from pathlib import Path
import sys
import pytest

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('experiment_train', scripts / 'train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_epoch_predictions_must_match_checkpoint_boundary(tmp_path):
    folder = tmp_path / 'epoch_predictions'
    folder.mkdir()
    for kind in ('chunks', 'recordings'):
        (folder / f'epoch_01_{kind}.parquet').write_bytes(b'fixture')
    module.verify_epoch_files(tmp_path, 1)
    (folder / 'epoch_02_chunks.parquet.tmp').write_bytes(b'interrupted')
    with pytest.raises(ValueError, match='differ'):
        module.verify_epoch_files(tmp_path, 1)


def test_contract_rejects_wrong_task():
    with pytest.raises(ValueError, match='invalid batch contract'):
        module.verify_contract({'task': 'other'}, Path('unused'))
