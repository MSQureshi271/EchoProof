"""Read-only AST calibration artifact verification; never fits or publishes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

RUN = 'runs/r3/r3_u2_20261112T134049Z'
CHECKPOINT = 'f7efe6a547c53a310ed540f10be32fe44b148def168c2f5362cc1d22510f70ec'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    run = root / RUN
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    selection = read(run / 'selection.json')
    evidence = read(run / 'evidence.json')
    require(selection['selected'] == 'baseline' and selection['selected_epoch'] == 1
            and selection['protocol'] == 'B', 'Unexpected selected AST')
    hashes = {}

    def check(path, expected):
        actual = digest(path)
        require(actual == expected, f'Hash mismatch: {path}')
        hashes[path.relative_to(root).as_posix()] = actual

    require(selection['checkpoint_sha256']['baseline'] == CHECKPOINT, 'Selected identity changed')
    check(run / 'r3_u2_20261112T134049Z-baseline/best.pt', CHECKPOINT)
    check(run / 'experiment_contract.json', selection['contract_sha256'])
    for name, expected in evidence['output_sha256'].items():
        require(Path(name).name == name, 'Unsafe artifact path')
        check(run / name, expected)
    for key, name in [('split', 'data/splits/protocol_B_v1.parquet'),
                      ('chunks', 'data/processed/chunks_v1.parquet'),
                      ('cohort', 'data/manifests/cohort_v1.parquet')]:
        check(root / name, evidence['input_sha256'][key])
    bundle = root / 'bundles/r3_u2_ast_partial_B'
    manifest = read(bundle / 'bundle.json')
    require(manifest['extra']['source_checkpoint_sha256'] == CHECKPOINT, 'Bundle identity mismatch')
    for name, metadata in manifest['components'].items():
        path = (bundle / name).resolve()
        require(path.is_relative_to(bundle.resolve()), 'Unsafe bundle component')
        check(path, metadata['sha256'])
    groups = {}
    counts = {}
    for role in ('selection', 'calibration', 'operating'):
        chunks = pq.read_table(run / f'predictions_chunks_{role}.parquet').to_pylist()
        recordings = pq.read_table(run / f'predictions_recordings_{role}.parquet').to_pylist()
        require(len({x['chunk_id'] for x in chunks}) == len(chunks), 'Duplicate chunks')
        require(len({x['recording_id'] for x in recordings}) == len(recordings), 'Duplicate recordings')
        by_id = {}
        for row in chunks:
            require(row['validation_role'] == role, 'Wrong chunk role')
            score = row['tts_probability']
            require(math.isfinite(score) and 0 <= score <= 1, 'Invalid chunk score')
            by_id.setdefault(row['recording_id'], []).append(row)
        require(set(by_id) == {x['recording_id'] for x in recordings}, 'Recording coverage mismatch')
        for row in recordings:
            children = by_id[row['recording_id']]
            require(row['validation_role'] == role, 'Wrong recording role')
            require(row['component'] in ('genuine', 'melo', 'vits', 'xtts'), 'Forbidden component')
            require(row['label'] == int(row['component'] != 'genuine'), 'Invalid label')
            require(all(all(c[k] == row[k] for k in ('group_id', 'label', 'component')) for c in children), 'Chunk identity mismatch')
            mean = sum(c['tts_probability'] for c in children) / len(children)
            require(row['chunk_count'] == len(children) and math.isclose(mean, row['raw_score'], abs_tol=1e-12, rel_tol=0), 'Aggregation mismatch')
        groups[role] = {r['group_id'] for r in recordings}
        counts[role] = {'recordings': len(recordings), 'chunks': len(chunks)}
        require(counts[role] == evidence['counts'][role], 'Count mismatch')
    require(not any(groups[a] & groups[b] for a, b in [('selection', 'calibration'), ('selection', 'operating'), ('calibration', 'operating')]), 'Speaker overlap')
    return {'status': 'artifact_preflight_passed', 'hashes': hashes, 'counts': counts,
            'limitations': ['Canonical split membership still requires semantic validation',
                           'No model execution or waveform parity performed',
                           'No calibration fitted or policy approved']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.project_root), indent=2, allow_nan=False))
