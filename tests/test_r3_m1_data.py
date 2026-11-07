import hashlib

import numpy as np
import pytest

from echoproof.training.r3_m1_data import VerifiedWaveformLoader, file_sha256, verify_provenance


def test_provenance_checks_every_input(tmp_path):
    paths = {key: tmp_path / key for key in ('cohort', 'split', 'chunks', 'stats')}
    for key, path in paths.items():
        path.write_bytes(key.encode())
    hashes = {k: file_sha256(v) for k, v in paths.items()}
    manifest = {'protocol': 'B', 'verification': {'passed': True},
                'inputs': {'split_sha256': 'frozen', 'feature_schema_hash': 'schema',
                           'cohort_manifest_file_sha256': hashes['cohort'], 'split_file_sha256': hashes['split'],
                           'chunk_manifest_file_sha256': hashes['chunks']},
                'outputs': {'mel_stats': {'sha256': hashes['stats']}}}
    assert verify_provenance(paths, manifest, 'frozen', 'schema') == hashes
    paths['chunks'].write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_provenance(paths, manifest, 'frozen', 'schema')


def test_loader_only_accepts_authorized_unmodified_audio():
    pcm = np.array([0, 123, -456], dtype='<i2')
    wave = pcm.astype(np.float32) / 32768
    rows = [{'recording_id': 'allowed', 'n_samples': len(pcm), 'pcm_sha256': hashlib.sha256(pcm.tobytes()).hexdigest()}]
    loader = VerifiedWaveformLoader({'allowed': wave}.__getitem__, rows)
    np.testing.assert_array_equal(loader('allowed'), wave)
    with pytest.raises(KeyError):
        loader('test')
    rows[0]['pcm_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='PCM hash'):
        VerifiedWaveformLoader({'allowed': wave}.__getitem__, rows)('allowed')
