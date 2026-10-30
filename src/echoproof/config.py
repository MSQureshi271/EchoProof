"""Strict configuration for Phase 1 software probes.

Paths are resolved relative to the project root, which must contain the config
file. Validation does not create directories or open recordings.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """A configuration is malformed or unsafe."""


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("Configuration keys must be strings")
        if key in result:
            raise ConfigError(f"Duplicate key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


@dataclass(frozen=True)
class Audio:
    sample_rate: int
    crop_samples: int
    mono_policy: str


@dataclass(frozen=True)
class Paths:
    fixtures: str
    run_root: str
    asset_lock: str
    genuine_audio: str
    genuine_transcripts: str
    tts_audio: str
    staging_root: str


@dataclass(frozen=True)
class Probe:
    ast_microbatch: int
    warmup_batches: int
    measured_batches: int
    precision: str
    timing_limit_seconds: float


@dataclass(frozen=True)
class Config:
    schema_version: int
    mode: str
    seed: int
    labels: dict[str, int]
    device: str
    audio: Audio
    paths: Paths
    probe: Probe
    metric_scope: str = "software_probe"

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False) + "\n"


def _fields(value, required, where):
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    missing, unknown = required - value.keys(), value.keys() - required
    if missing or unknown:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")
    return value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _choice(value, name, choices):
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{name} must be one of {sorted(choices)}")
    return value


def _path(value, root, name):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"paths.{name} must be a nonempty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _inside(path, parent):
    return path == parent or path.is_relative_to(parent)


def load_config(config_file, project_root, *, run_root=None, device=None):
    root = Path(project_root).resolve()
    file = Path(config_file).resolve()
    if not (root / "pyproject.toml").is_file() or not _inside(file, root):
        raise ConfigError("Project root must contain pyproject.toml and the config file")
    try:
        data = yaml.load(file.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot load configuration: {exc}") from exc
    data = _fields(data, {"schema_version", "mode", "seed", "labels", "device", "audio", "paths", "probe"}, "config")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ConfigError("schema_version must be 1")
    _choice(data["mode"], "mode", {"fixture_probe"})
    _integer(data["seed"], "seed", 0)
    labels = _fields(data["labels"], {"genuine", "tts"}, "labels")
    if any(type(v) is not int for v in labels.values()) or labels != {"genuine": 0, "tts": 1}:
        raise ConfigError("labels must map genuine to 0 and tts to 1")
    selected_device = _choice(device if device is not None else data["device"], "device", {"cpu", "cuda", "auto"})
    audio = _fields(data["audio"], {"sample_rate", "crop_samples", "mono_policy"}, "audio")
    _integer(audio["sample_rate"], "audio.sample_rate")
    _integer(audio["crop_samples"], "audio.crop_samples")
    _choice(audio["mono_policy"], "audio.mono_policy", {"mean_channels"})
    names = {"fixtures", "run_root", "asset_lock", "genuine_audio", "genuine_transcripts", "tts_audio", "staging_root"}
    paths = _fields(data["paths"], names, "paths").copy()
    if run_root is not None:
        paths["run_root"] = run_root
    resolved = {name: _path(value, root, name) for name, value in paths.items()}
    # Fixture and persistent output paths have fixed boundaries; resolving also
    # catches existing symlinks that point outside those boundaries.
    fixture_boundary = (root / "tests/fixtures/generated").resolve()
    run_boundary = (root / "runs").resolve()
    config_boundary = (root / "configs").resolve()
    if any(not _inside(boundary, root) for boundary in (fixture_boundary, run_boundary, config_boundary)):
        raise ConfigError("Project output boundaries must not resolve outside the project")
    if not _inside(resolved["fixtures"], fixture_boundary):
        raise ConfigError("fixture_probe fixtures must be under tests/fixtures/generated")
    if not _inside(resolved["run_root"], run_boundary):
        raise ConfigError("run_root must be under the project's runs directory")
    if not _inside(resolved["asset_lock"], config_boundary):
        raise ConfigError("asset_lock must be under the project's configs directory")
    protected = [resolved[name] for name in ("genuine_audio", "genuine_transcripts", "tts_audio")]
    protected.append((root / "dataset").resolve())
    for name in ("fixtures", "run_root", "staging_root"):
        if any(_inside(resolved[name], p) or _inside(p, resolved[name]) for p in protected):
            raise ConfigError(f"paths.{name} overlaps a research input directory")
    if _inside(resolved["staging_root"], root) or _inside(root, resolved["staging_root"]):
        raise ConfigError("staging_root must not overlap the project root")
    probe = _fields(data["probe"], {"ast_microbatch", "warmup_batches", "measured_batches", "precision", "timing_limit_seconds"}, "probe")
    for name in ("ast_microbatch", "measured_batches"):
        _integer(probe[name], f"probe.{name}")
    _integer(probe["warmup_batches"], "probe.warmup_batches", 0)
    _choice(probe["precision"], "probe.precision", {"fp32", "amp"})
    limit = probe["timing_limit_seconds"]
    if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
        raise ConfigError("probe.timing_limit_seconds must be positive and finite")
    return Config(1, data["mode"], data["seed"], dict(labels), selected_device,
                  Audio(**audio), Paths(**{k: str(v) for k, v in resolved.items()}), Probe(**probe))
