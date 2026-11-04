"""R1-S4 leakage and protocol-integrity checks over the locked manifests.

`check_leakage` never raises on a violation: it returns every violation so the report shows all of them.
Cohort rows are dicts: recording_id, stem, speaker_id, component, label, pcm_sha256, retained.
"""
from __future__ import annotations

from collections import defaultdict


B_TRAIN_ALLOWED = {"genuine", "melo", "vits"}
B_VALIDATION_ALLOWED = {"genuine", "melo", "vits", "xtts"}
B_TEST_ALLOWED = {"genuine", "melo", "vits", "chatterbox"}


def _violation(kind, detail):
    return {"check": kind, "detail": detail}


def check_leakage(cohort_rows, assignments_by_protocol, chunks_by_protocol=None):
    cohort = {}
    violations = []
    for row in cohort_rows:
        if row["recording_id"] in cohort:
            violations.append(_violation("duplicate_cohort_recording", row["recording_id"]))
        cohort[row["recording_id"]] = row
    chunks_by_protocol = chunks_by_protocol or {}
    summary = {}
    for protocol, assignments in assignments_by_protocol.items():
        seen = set()
        speaker_slots = defaultdict(set)
        stem_slots = defaultdict(set)
        hash_slots = defaultdict(set)
        panels = defaultdict(lambda: defaultdict(set))
        for a in assignments:
            if a.recording_id in seen:
                violations.append(_violation("recording_assigned_twice", f"{protocol}:{a.recording_id}"))
            seen.add(a.recording_id)
            row = cohort.get(a.recording_id)
            if row is None:
                violations.append(_violation("assignment_without_cohort_row", f"{protocol}:{a.recording_id}"))
                continue
            slot = (a.partition, a.validation_role)
            speaker_slots[a.group_id].add(slot)
            if a.group_id != f"speaker:{row['speaker_id']}":
                violations.append(_violation("group_is_not_speaker", f"{protocol}:{a.recording_id}"))
            if not row["retained"]:
                continue
            stem_slots[row["stem"]].add(a.partition)
            if row.get("pcm_sha256"):
                hash_slots[row["pcm_sha256"]].add(a.partition)
            if protocol == "B":
                allowed = {"train": B_TRAIN_ALLOWED, "validation": B_VALIDATION_ALLOWED, "test": B_TEST_ALLOWED}[a.partition]
                if row["component"] not in allowed:
                    violations.append(_violation("b_family_not_allowed", f"{a.recording_id} in {a.partition}"))
                if row["component"] == "chatterbox" and a.partition != "test":
                    violations.append(_violation("chatterbox_outside_final_test", a.recording_id))
            label = "tts" if row["label"] == 1 else "genuine"
            panels[(a.partition, a.validation_role)]["all"].add(label)
            panels[(a.partition, a.validation_role)][label].add(row["component"])
        for speaker, slots in speaker_slots.items():
            if len(slots) > 1:
                violations.append(_violation("speaker_crosses_partitions", f"{protocol}:{speaker}:{sorted(map(str, slots))}"))
        for stem, parts in stem_slots.items():
            if len(parts) > 1:
                violations.append(_violation("stem_crosses_partitions", f"{protocol}:{stem}"))
        for digest_value, parts in hash_slots.items():
            if len(parts) > 1:
                violations.append(_violation("pcm_hash_crosses_partitions", f"{protocol}:{digest_value[:16]}"))
        for slot, groups in panels.items():
            if groups["all"] != {"genuine", "tts"}:
                violations.append(_violation("panel_missing_class", f"{protocol}:{slot}:{sorted(groups['all'])}"))
            if protocol == "B" and slot[0] == "validation" and "xtts" not in groups.get("tts", set()):
                violations.append(_violation("b_validation_role_lacks_unseen_family", f"{slot}"))
            if protocol == "B" and slot[0] == "test" and "chatterbox" not in groups.get("tts", set()):
                violations.append(_violation("b_test_lacks_unseen_final_family", f"{slot}"))
        chunks = chunks_by_protocol.get(protocol)
        if chunks is not None:
            retained_ids = {a.recording_id for a in assignments if cohort.get(a.recording_id, {}).get("retained")}
            chunked = {c.recording_id for c in chunks}
            for missing in sorted(retained_ids - chunked):
                violations.append(_violation("recording_without_chunks", f"{protocol}:{missing}"))
            by_id = {a.recording_id: a for a in assignments}
            ids = set()
            for c in chunks:
                if c.chunk_id in ids:
                    violations.append(_violation("duplicate_chunk_id", c.chunk_id))
                ids.add(c.chunk_id)
                a = by_id.get(c.recording_id)
                if a is None:
                    violations.append(_violation("orphan_chunk", c.chunk_id))
                elif a.group_id != c.group_id or a.partition != c.partition or a.protocol_id != c.protocol_id:
                    violations.append(_violation("chunk_assignment_mismatch", c.chunk_id))
                elif c.recording_id not in retained_ids:
                    violations.append(_violation("chunk_for_excluded_recording", c.chunk_id))
        summary[protocol] = {"assignments": len(assignments), "speakers": len(speaker_slots),
                             "chunks": None if chunks_by_protocol.get(protocol) is None else len(chunks_by_protocol[protocol])}
    conflicts = defaultdict(set)
    for row in cohort.values():
        if row["retained"] and row.get("pcm_sha256"):
            conflicts[row["pcm_sha256"]].add(row["label"])
    for value, labels in conflicts.items():
        if len(labels) > 1:
            violations.append(_violation("label_conflict_retained", value[:16]))
    return {"schema_version": 1, "task": "R1-S4", "passed": not violations, "violation_count": len(violations),
            "violations": violations, "summary": summary}
