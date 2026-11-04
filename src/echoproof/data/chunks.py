"""R1-S3 deterministic evaluation windows and chunk manifest.

Policy (IMPLEMENTATION_PLAN section 8): 4 s windows at a 2 s stride while a full window fits, plus one final
window ending at the last sample if the tail would otherwise be uncovered; recordings shorter than 4 s become one
chunk with its true valid length (right-padding happens at feature time). Audio is never repeated to fill a window.
"""
from __future__ import annotations

from echoproof.artifacts import digest
from echoproof.contracts import Chunk
from echoproof.data.cohort import CohortError, SAMPLE_RATE

WINDOW_SECONDS = 4
STRIDE_SECONDS = 2
SETTINGS = {"sample_rate": SAMPLE_RATE, "window_seconds": WINDOW_SECONDS, "stride_seconds": STRIDE_SECONDS,
            "tail_policy": "final_full_window_without_duplicate_start", "short_policy": "single_chunk_true_valid_length",
            "mono": "mean_channels", "resample": "scipy.resample_poly_once", "dtype": "int16_flac_canonical"}
PREPROCESSING_HASH = digest(SETTINGS)


def eval_windows(n_samples: int, *, window: int = WINDOW_SECONDS * SAMPLE_RATE, stride: int = STRIDE_SECONDS * SAMPLE_RATE):
    """Return [(start, end)] covering every sample exactly as the policy above describes."""
    if type(n_samples) is not int or n_samples <= 0:
        raise CohortError("recording must contain at least one sample")
    if n_samples <= window:
        return [(0, n_samples)]
    starts = list(range(0, n_samples - window + 1, stride))
    if starts[-1] + window < n_samples:
        starts.append(n_samples - window)
    return [(s, s + window) for s in starts]


def build_chunks(protocol: str, assignments, cohort_by_id, *, window: int = WINDOW_SECONDS * SAMPLE_RATE):
    """Chunk rows for every retained, assigned recording of one protocol."""
    chunks = []
    for a in assignments:
        row = cohort_by_id.get(a.recording_id)
        if row is None:
            raise CohortError(f"assignment without cohort row: {a.recording_id}")
        if not row["retained"]:
            continue
        for start, end in eval_windows(int(row["n_samples"]), window=window):
            chunks.append(Chunk(1, f"chunk:{protocol}:{a.recording_id}:{start}", a.recording_id, a.group_id, a.protocol_id,
                                a.partition, start, end, SAMPLE_RATE, start / SAMPLE_RATE, end / SAMPLE_RATE,
                                end - start, window,
                                PREPROCESSING_HASH))
    return chunks
