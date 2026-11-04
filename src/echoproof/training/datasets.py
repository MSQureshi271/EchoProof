"""R1-M1 datasets: one seeded random crop per recording per epoch (training) and fixed manifest chunks (evaluation).

Both datasets read waveforms through a `loader(recording_id) -> float32 mono 16 kHz array`, so they work with
shards on Colab (`ShardWaveformLoader`), in-memory fixtures (`DictWaveformLoader`) or anything else. Models receive
raw padded waveforms plus true valid lengths; model-specific features (log-mel, processors) belong to the model side.
Crop starts depend only on (seed, epoch, recording_id), never on iteration order or worker count, so a resumed run
sees exactly the crops the uninterrupted run would have seen.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import tarfile

import numpy as np

from echoproof.training.seeding import named_seed

CROP_SAMPLES = 64000  # 4 s at 16 kHz


class DictWaveformLoader:
    def __init__(self, waveforms):
        self.waveforms = {k: np.asarray(v, dtype=np.float32) for k, v in waveforms.items()}

    def __call__(self, recording_id):
        return self.waveforms[recording_id]


class ShardWaveformLoader:
    """Reads canonical int16 FLAC members from the cohort tar shards (see data/cohort.py).

    `canonical_paths` maps recording_id -> 'shard_00.tar::recording_id' (cohort_v1.parquet column canonical_path).
    Keeps tar handles open and a small recording cache because eval chunks of one recording are read together.
    Safe with DataLoader workers: a forked worker inherits the parent's file descriptors, which share one file offset, so
    handles are tagged with the owning process id and reopened (independent descriptors) in any other process. Handles and
    cache are also dropped on pickling, which covers spawn-based workers. Not thread-safe within one process.
    """

    def __init__(self, shard_dir, canonical_paths, cache_size: int = 8):
        self.shard_dir = Path(shard_dir)
        self.paths = dict(canonical_paths)
        self.cache_size = cache_size
        self._reset()

    def _reset(self):
        self._tars, self._cache, self._pid = {}, OrderedDict(), os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tars"], state["_cache"] = {}, OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset()

    def __call__(self, recording_id):
        if self._pid != os.getpid():
            # Inherited from the parent through fork: never read through its descriptors. Drop the references without
            # closing the shared state of the parent (closing in the child only closes the child's descriptor copy).
            self._reset()
        if recording_id in self._cache:
            self._cache.move_to_end(recording_id)
            return self._cache[recording_id]
        from echoproof.data.cohort import read_flac
        shard, key = self.paths[recording_id].split("::", 1)
        if shard not in self._tars:
            self._tars[shard] = tarfile.open(self.shard_dir / shard, "r")
        handle = self._tars[shard].extractfile(key.replace(":", "__") + ".flac")
        if handle is None:
            raise KeyError(recording_id)
        wave = read_flac(handle.read()).astype(np.float32) / 32768.0
        self._cache[recording_id] = wave
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return wave

    def close(self):
        for tar in self._tars.values():
            tar.close()
        self._tars.clear()


def crop_start(seed: int, epoch: int, recording_id: str, n_samples: int, crop: int = CROP_SAMPLES) -> int:
    """Deterministic crop start; 0 when the recording is not longer than the crop."""
    if n_samples <= crop:
        return 0
    return named_seed(seed, f"crop:{epoch}:{recording_id}") % (n_samples - crop + 1)


def _pad(wave, size):
    out = np.zeros(size, dtype=np.float32)
    out[:len(wave)] = wave
    return out


class CropDataset:
    """Training view: items are (recording_id, label) pairs; call set_epoch(e) before each epoch."""

    def __init__(self, items, loader, *, seed: int, crop_samples: int = CROP_SAMPLES):
        self.items = [(str(r), int(l)) for r, l in items]
        if not self.items:
            raise ValueError("training set is empty")
        if any(l not in (0, 1) for _, l in self.items):
            raise ValueError("labels must be 0 or 1")
        self.loader, self.seed, self.crop, self.epoch = loader, seed, crop_samples, 0

    @property
    def labels(self):
        return [l for _, l in self.items]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        recording_id, label = self.items[index]
        wave = self.loader(recording_id)
        if wave.ndim != 1 or len(wave) == 0 or not np.isfinite(wave).all():
            raise ValueError(f"invalid waveform for {recording_id}")
        start = crop_start(self.seed, self.epoch, recording_id, len(wave), self.crop)
        piece = wave[start:start + self.crop]
        return {"recording_id": recording_id, "chunk_id": f"crop:{self.epoch}:{recording_id}:{start}", "label": label,
                "waveform": _pad(piece, self.crop), "valid_samples": len(piece)}


class ChunkDataset:
    """Evaluation view over manifest chunks (echoproof.contracts.Chunk) in manifest order; labels by recording."""

    def __init__(self, chunks, labels_by_recording, loader):
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("evaluation set is empty")
        missing = {c.recording_id for c in self.chunks} - set(labels_by_recording)
        if missing:
            raise ValueError(f"no label for {len(missing)} evaluation recordings")
        self.labels_by_recording, self.loader = labels_by_recording, loader

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        c = self.chunks[index]
        wave = self.loader(c.recording_id)
        piece = wave[c.start_sample:c.end_sample]
        if len(piece) != c.valid_samples:
            raise ValueError(f"chunk {c.chunk_id} does not match the audio ({len(piece)} != {c.valid_samples} samples)")
        return {"recording_id": c.recording_id, "chunk_id": c.chunk_id, "label": int(self.labels_by_recording[c.recording_id]),
                "waveform": _pad(piece, c.input_samples), "valid_samples": len(piece)}


def collate(items):
    """Batch dict: model_inputs feed `model(**model_inputs)`; the rest is bookkeeping."""
    import torch
    width = max(len(i["waveform"]) for i in items)
    waveforms = np.stack([_pad(i["waveform"], width) for i in items])
    return {"model_inputs": {"waveforms": torch.from_numpy(waveforms),
                             "valid_samples": torch.tensor([i["valid_samples"] for i in items], dtype=torch.int64)},
            "labels": torch.tensor([i["label"] for i in items], dtype=torch.int64),
            "recording_ids": [i["recording_id"] for i in items], "chunk_ids": [i["chunk_id"] for i in items]}
