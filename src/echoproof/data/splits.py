"""R1-S2 speaker-grouped splits, validation roles, cohort sampling and protocol filters.

Everything here is deterministic and depends only on stem names and a seed, never on audio
or model outcomes. The independence unit is the LibriSpeech speaker (`speaker:<id>`).
Tests may only be read through `load_assignments(..., sealed=True)`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from echoproof.artifacts import digest
from echoproof.contracts import SplitAssignment, read_parquet, write_parquet
from echoproof.data.cohort import COMPONENTS, CohortError, parse_stem, recording_id, speaker_group

SPLIT_VERSION = "r1-v1"
PARTITIONS = ("train", "validation", "test")
ROLES = ("selection", "calibration", "operating")
DEFAULT_RATIOS = (0.60, 0.20, 0.20)
DEFAULT_ROLE_RATIOS = (0.50, 0.25, 0.25)
DEFAULT_PER_SPEAKER = {"train": 10, "validation": 20, "test": 20}

# Protocol filters: partition (+ role) -> {component: family_policy}
KNOWN = "known_family"
UNSEEN_VAL = "unseen_validation"
UNSEEN_FINAL = "unseen_final"
PROTOCOL_B = {
    ("train", None): {"genuine": "genuine", "melo": KNOWN, "vits": KNOWN},
    ("validation", "*"): {"genuine": "genuine", "melo": KNOWN, "vits": KNOWN, "xtts": UNSEEN_VAL},
    ("test", None): {"genuine": "genuine", "melo": KNOWN, "vits": KNOWN, "chatterbox": UNSEEN_FINAL},
}
PROTOCOL_A = {
    ("train", None): {c: ("genuine" if c == "genuine" else KNOWN) for c in COMPONENTS},
    ("validation", "*"): {c: ("genuine" if c == "genuine" else KNOWN) for c in COMPONENTS},
    ("test", None): {c: ("genuine" if c == "genuine" else KNOWN) for c in COMPONENTS},
}
PROTOCOLS = {"A": PROTOCOL_A, "B": PROTOCOL_B}


def _rank(seed: int, label: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{label}|{key}".encode()).hexdigest()


def _counts(n: int, ratios) -> list[int]:
    raw = [n * r for r in ratios]
    base = [int(x) for x in raw]
    # ties go to the later bucket (operating / test), which carry the thinnest evidence
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - base[i]), -i))
    for i in order[: n - sum(base)]:
        base[i] += 1
    return base


def assign_speakers(speakers, *, seed: int = 42, ratios=DEFAULT_RATIOS, role_ratios=DEFAULT_ROLE_RATIOS):
    """Return {speaker_id: (partition, role|None)}. Hash-ordered, so adding data never reshuffles others."""
    speakers = sorted(set(speakers), key=lambda s: _rank(seed, "partition", s))
    if len(speakers) < 10:
        raise CohortError("too few speakers for a three-way grouped split")
    n_train, n_val, n_test = _counts(len(speakers), ratios)
    train, val, test = speakers[:n_train], speakers[n_train:n_train + n_val], speakers[n_train + n_val:]
    roles = sorted(val, key=lambda s: _rank(seed, "role", s))
    n_sel, n_cal, n_op = _counts(len(roles), role_ratios)
    result = {s: ("train", None) for s in train}
    result.update({s: ("test", None) for s in test})
    for chunk, role in ((roles[:n_sel], "selection"), (roles[n_sel:n_sel + n_cal], "calibration"), (roles[n_sel + n_cal:], "operating")):
        result.update({s: ("validation", role) for s in chunk})
    return result


def sample_stems(stems, speaker_partition, *, per_speaker=DEFAULT_PER_SPEAKER, seed: int = 42):
    """Pick the same N stems per speaker for every component; hash-ordered and reproducible."""
    by_speaker = defaultdict(list)
    for stem in stems:
        by_speaker[parse_stem(stem)[0]].append(stem)
    chosen = []
    for speaker, items in sorted(by_speaker.items()):
        partition, _ = speaker_partition[speaker]
        need = per_speaker[partition]
        if len(items) < need:
            raise CohortError(f"speaker {speaker} has {len(items)} stems, needs {need}")
        chosen.extend(sorted(sorted(items, key=lambda s: _rank(seed, "stem", s))[:need]))
    return sorted(chosen)


def protocol_rows(protocol: str, stems, speaker_partition):
    """Plain-dict split rows for one protocol, before hashing."""
    rules = PROTOCOLS[protocol]
    rows = []
    for stem in stems:
        speaker = parse_stem(stem)[0]
        partition, role = speaker_partition[speaker]
        rule = rules[(partition, "*" if partition == "validation" else None)]
        for component, policy in rule.items():
            rows.append({"protocol": protocol, "recording_id": recording_id(component, stem), "stem": stem,
                         "component": component, "group_id": speaker_group(stem), "partition": partition,
                         "validation_role": role, "family_policy": policy})
    return sorted(rows, key=lambda r: r["recording_id"])


def build_assignments(protocol: str, stems, speaker_partition, *, version: str = SPLIT_VERSION):
    rows = protocol_rows(protocol, stems, speaker_partition)
    split_hash = digest([[r["recording_id"], r["group_id"], r["partition"], r["validation_role"], r["family_policy"]] for r in rows])
    policy = f"protocol_{protocol}"
    assignments = [SplitAssignment(1, f"protocol:{protocol}", r["recording_id"], r["group_id"], r["partition"],
                                   r["validation_role"], policy, version, split_hash) for r in rows]
    return assignments, split_hash


def write_assignments(assignments, path) -> None:
    write_parquet(assignments, str(path))


def load_assignments(path, *, sealed: bool = False):
    """Read assignments. Test rows are withheld unless the caller opts into sealed evaluation."""
    rows = read_parquet(str(path), SplitAssignment)
    if sealed:
        return rows
    return [r for r in rows if r.partition != "test"]


def family_policies(protocol: str, stems, speaker_partition):
    return {r["recording_id"]: r["family_policy"] for r in protocol_rows(protocol, stems, speaker_partition)}


def panel_support(rows):
    """Counts of recordings, stems and speakers per (partition, role, family_policy, label) for the precision worksheet."""
    out = defaultdict(lambda: {"recordings": 0, "speakers": set()})
    for r in rows:
        key = (r["partition"], r["validation_role"] or "-", r["family_policy"], "genuine" if r["component"] == "genuine" else "tts")
        out[key]["recordings"] += 1
        out[key]["speakers"].add(r["group_id"])
    return {"|".join(k): {"recordings": v["recordings"], "speakers": len(v["speakers"])} for k, v in sorted(out.items())}


def speaker_counts(speaker_partition):
    return dict(Counter(f"{p}/{r or '-'}" for p, r in speaker_partition.values()))
