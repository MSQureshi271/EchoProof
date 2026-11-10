"""Binding to the accepted R2-U2 recipe and identity helpers.

Official numerical code is a checked copy; the immutable experiment remains the
source of its approved mathematics. New settings require new experiment evidence.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / 'experiments/r2_u2/scripts/contract.py'
spec = importlib.util.spec_from_file_location('_echoproof_r2_u2_frozen_contract', PATH)
_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_impl)
EXP = _impl.EXP
def verify_official_copies():
    """Refuse divergence from the accepted implementation before making receipts."""
    import hashlib
    import json
    manifest = json.loads((EXP / 'colab_source_manifest.json').read_text())['files']
    for name in ('numerical_trainer.py', 'scope.py', 'acceptance.py'):
        original = EXP / 'scripts' / name
        relative = original.relative_to(ROOT).as_posix()
        if manifest[relative] not in {hashlib.sha256(original.read_bytes()).hexdigest(),
                                      hashlib.sha256(original.read_bytes().replace(b'\r\n', b'\n')).hexdigest()}:
            raise ValueError('Frozen R2-U2 implementation identity changed: ' + relative)
        expected = original.read_text()
        if name == 'numerical_trainer.py':
            expected = expected.replace('from contract import EXP, provenance, fingerprint, sources',
                                        'from .r2_u2_contract import EXP, provenance, fingerprint, sources')
        official = Path(__file__).with_name('r2_u2_' + name)
        if official.read_text() != expected:
            raise ValueError('Official R2-U2 copy diverged; new source-bound evidence required: ' + str(official))


def provenance(*args, **kwargs):
    verify_official_copies()
    return _impl.provenance(*args, **kwargs)
fingerprint = _impl.fingerprint
sources = _impl.sources
new_run = _impl.new_run
