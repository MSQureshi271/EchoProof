"""Strict Phase 2 audit configuration; loading never scans or creates data."""
from pathlib import Path, PureWindowsPath
import math
import yaml
from .config import ConfigError, UniqueKeyLoader

_FIELDS = {'schema_version', 'source_registry', 'sources', 'outputs', 'workers',
           'block_bytes', 'checkpoint_interval', 'hash_policy', 'header_limit_bytes',
           'decode_limit_seconds', 'sample_seed', 'sample_count', 'retries',
           'timeout_seconds', 'acquisition_ceiling_bytes', 'drive_free_bytes',
           'scratch_free_bytes', 'follow_symlinks', 'immutable_inputs'}

def _keys(obj, keys, name):
    if not isinstance(obj, dict) or set(obj) != keys:
        raise ConfigError(f'{name} must contain exactly {sorted(keys)}')

def _path(value, root):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ConfigError('paths must be nonempty relative POSIX paths')
    p = Path(value)
    if p.is_absolute() or PureWindowsPath(value).drive or any(x in ('', '.', '..') for x in value.split('/')):
        raise ConfigError('invalid relative path')
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root):
        raise ConfigError('path escapes project root')
    return resolved

def load_phase2_config(filename, project_root):
    root = Path(project_root).resolve()
    path = Path(filename).resolve()
    if not path.is_relative_to(root):
        raise ConfigError('configuration must be inside project root')
    try:
        data = yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f'cannot load configuration: {exc}') from exc
    _keys(data, _FIELDS, 'config')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ConfigError('unsupported schema version')
    _keys(data['sources'], {'genuine', 'transcripts', 'xtts', 'chatterbox', 'melo', 'vits', 'external'}, 'sources')
    _keys(data['outputs'], {'inventory', 'staging'}, 'outputs')
    registry = _path(data['source_registry'], root)
    sources = [_path(v, root) for v in data['sources'].values() if v is not None]
    if any(data['sources'][k] is None for k in ('genuine','transcripts','xtts','chatterbox','melo','vits')):
        raise ConfigError('primary roots are required')
    outputs = [_path(v, root) for v in data['outputs'].values()]
    for i, a in enumerate(outputs):
        for b in sources + [registry] + outputs[i+1:]:
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ConfigError('output overlaps another output or source')
    for key in ('workers','block_bytes','checkpoint_interval','header_limit_bytes','sample_count'):
        if type(data[key]) is not int or data[key] < 1:
            raise ConfigError(f'{key} must be a positive integer')
    for key in ('sample_seed','retries','acquisition_ceiling_bytes'):
        if type(data[key]) is not int or data[key] < 0:
            raise ConfigError(f'{key} must be a nonnegative integer')
    for key in ('drive_free_bytes','scratch_free_bytes'):
        if data[key] is not None and (type(data[key]) is not int or data[key] < 0):
            raise ConfigError(f'{key} must be null or a nonnegative integer')
    for key in ('decode_limit_seconds','timeout_seconds'):
        if type(data[key]) not in (int,float) or not math.isfinite(data[key]) or data[key] <= 0:
            raise ConfigError(f'{key} must be positive and finite')
    if data['hash_policy'] != 'sha256_full':
        raise ConfigError('unsupported hash policy')
    for key in ('follow_symlinks', 'immutable_inputs'):
        if type(data[key]) is not bool:
            raise ConfigError(f'{key} must be boolean')
    if data['follow_symlinks']:
        raise ConfigError('following source symlinks is not supported')
    return data

def require_acquisition_capacity(config, required_bytes):
    """Block acquisition until a current capacity and byte budget are supplied."""
    if type(required_bytes) is not int or required_bytes <= 0:
        raise ConfigError('required_bytes must be a positive integer')
    for key in ('acquisition_ceiling_bytes', 'drive_free_bytes', 'scratch_free_bytes'):
        value = config[key]
        if value is None or value < required_bytes:
            raise ConfigError(f'{key} is unknown or insufficient for acquisition')
