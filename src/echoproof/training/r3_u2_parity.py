"""R3-U2 adapter parity: the bundle adapter must reproduce the exported selection-role chunk probabilities from raw waveforms.

The exports were scored with fp16 autocast on a CUDA device, so the gate is the adapter in fp16 on CUDA (the same numerics and code path, loaded from
the bundle, within an fp16 tolerance of 0.01). The adapter in fp32 is measured and reported, not gated: it states how far CPU/fp32 inference sits from the scores that R3-N1
will calibrate. The sample is a deterministic stratified draw over (component, short or full chunk) from the selection role, Chatterbox excluded.
Waveforms come from the verified R3-U2 chunk dataset, whose loader checks every recording's canonical PCM hash against the cohort manifest.
"""
from __future__ import annotations

import numpy as np

from echoproof.artifacts import digest

FULL_CHUNK = 64000
# fp16 results depend on the batch shape (the exports were scored in batches of 16 over the whole role, the parity sample is a subset in batches
# of 8), so the gate is a probability tolerance for fp16 numerics, not bit equality. The measured differences are reported.
ATOL_FP16 = 1e-2


class ParityError(ValueError):
    pass


def select_chunks(dataset, components: dict, n: int = 50, *, seed: int = 42) -> list:
    """Indices into `dataset.chunks`: a round-robin draw over strata, each stratum ordered by a seeded hash of the chunk id."""
    strata = {}
    for index, chunk in enumerate(dataset.chunks):
        component = components[chunk.recording_id]
        if component == "chatterbox":
            continue
        strata.setdefault((component, chunk.valid_samples < FULL_CHUNK), []).append(index)
    if n < 1 or sum(map(len, strata.values())) < n:
        raise ParityError("not enough validation chunks for the requested parity sample")
    queues = [sorted(items, key=lambda i: digest([seed, dataset.chunks[i].chunk_id])) for _, items in sorted(strata.items())]
    chosen = []
    while len(chosen) < n:
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen


def waves_for(dataset, indices) -> list:
    waves = []
    for index in indices:
        item = dataset[index]
        wave = np.asarray(item["waveform"][:int(item["valid_samples"])], dtype=np.float32)
        if len(wave) != dataset.chunks[index].valid_samples:
            raise ParityError(f"chunk {dataset.chunks[index].chunk_id} has the wrong length")
        waves.append(wave)
    return waves


def _compare(got, expected) -> dict:
    got, expected = np.asarray(got, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if got.shape != expected.shape or not np.isfinite(got).all():
        raise ParityError("invalid adapter scores")
    return {"max_abs_difference": float(np.max(np.abs(got - expected))), "mean_abs_difference": float(np.mean(np.abs(got - expected))),
            "decision_flips_at_0.5": int(np.sum((got >= .5) != (expected >= .5)))}


def parity_check(chunk_ids, expected, *, scores_fp16, others=None, atol_fp16=ATOL_FP16) -> dict:
    """`expected`: exported probabilities in the same order as `chunk_ids`. `scores_fp16` must come from the adapter in fp16 on CUDA.

    `others` maps a label (for example "fp32_cuda", "fp32_cpu") to adapter scores that are measured and reported but not gated.
    """
    if len(chunk_ids) != len(expected) or len(chunk_ids) == 0:
        raise ParityError("chunk ids and expected scores must align")
    report = {"items": len(chunk_ids), "chunk_ids": list(chunk_ids), "gate": "fp16 adapter on CUDA reproduces the exported probabilities",
              "atol_fp16": atol_fp16, "fp16": _compare(scores_fp16, expected)}
    report["passed"] = bool(report["fp16"]["max_abs_difference"] <= atol_fp16)
    if others:
        report["reported_not_gated"] = {name: _compare(scores, expected) for name, scores in others.items()}
        report["reported_note"] = ("CPU or fp32 inference differs slightly from the fp16 scores that R3-N1 calibrates; the flips are the "
                                   "chunks whose decision at 0.5 would change")
    return report
