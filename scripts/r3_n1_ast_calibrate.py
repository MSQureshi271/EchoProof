"""Isolated AST-B candidate calibration; no training, lock updates or promotion."""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from r3_n1_ast_preflight import verify, require, digest, RUN
from r3_n1_lr_operating import reliability
from echoproof.evaluation.calibrate import fit_temperature, apply_temperature
from echoproof.evaluation.metrics import compute_metrics
from echoproof.evaluation.threshold import select_threshold
from echoproof.bundle import load_bundle, write_bundle


def canonical_check(root, run):
    import pyarrow.parquet as pq
    def payload(path):
        return [json.loads(r['payload_json']) for r in pq.read_table(path).to_pylist()]
    # Metadata only: no held-out scores or audio are loaded.
    split = payload(root / 'data/splits/protocol_B_v1.parquet')
    chunks = payload(root / 'data/processed/chunks_v1.parquet')
    for role in ('calibration', 'operating', 'selection'):
        expected = {r['recording_id']: r for r in split if r['partition'] == 'validation' and r['validation_role'] == role}
        rows = pq.read_table(run / f'predictions_recordings_{role}.parquet').to_pylist()
        require(set(expected) == {r['recording_id'] for r in rows}, 'Canonical recording membership mismatch')
        require(all(r['group_id'] == expected[r['recording_id']]['group_id'] for r in rows), 'Canonical speaker mismatch')
        wanted = {r['chunk_id']: r for r in chunks if r['protocol_id'] == 'protocol:B' and r['partition'] == 'validation' and r['recording_id'] in expected}
        actual = pq.read_table(run / f'predictions_chunks_{role}.parquet').to_pylist()
        require(set(wanted) == {r['chunk_id'] for r in actual}, 'Canonical chunk membership mismatch')
        require(all(all(r[k] == wanted[r['chunk_id']][k] for k in ('recording_id', 'group_id')) for r in actual), 'Canonical chunk identity mismatch')


def accepted(raw, calibrated):
    return calibrated['log_loss'] < raw['log_loss'] and calibrated['brier_score'] <= raw['brier_score']


def execute(root):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    policy_path = root / 'configs/r3_n1_ast_candidate.json'
    policy = json.loads(policy_path.read_text())
    require(policy['acceptance'] == 'strictly_lower_operating_log_loss_and_nonworsening_brier', 'Unsupported policy')
    before = verify(root)
    run = root / RUN
    canonical_check(root, run)
    lock_hash = digest(root / 'locks/B_lock.json')
    ident = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    out = root / 'runs/r3/r3_n1_ast_candidates' / ident
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    # Persist policy and identities before fitting or evaluating any results.
    write('prospective_policy.json', policy)
    write('input_verification.json', before)
    def rows(role):
        return pq.read_table(run / f'predictions_recordings_{role}.parquet').to_pylist()
    calibration, operating = rows('calibration'), rows('operating')
    temp = fit_temperature([r['label'] for r in calibration], [r['raw_score'] for r in calibration], epsilon=policy['epsilon'], bounds=tuple(policy['bounds']))
    labels, raw = [r['label'] for r in operating], [r['raw_score'] for r in operating]
    scaled = apply_temperature(raw, temp, epsilon=policy['epsilon'])
    rm, cm = compute_metrics(labels, raw, .5), compute_metrics(labels, scaled, .5)
    use = accepted(rm, cm)
    chosen = scaled if use else raw
    rule, f1, fpr = select_threshold(labels, chosen)
    metadata = {'schema': 'r3_n1_ast_candidate_v1', 'temperature': temp, 'epsilon': policy['epsilon'],
                'temperature_accepted_by_candidate_rule': use, 'score_type': 'calibrated_probability' if use else 'model_score',
                'policy_sha256': digest(policy_path), 'checkpoint_sha256': policy['checkpoint_sha256'],
                'precision': policy['precision'], 'team_approved': False}
    original = load_bundle(root / 'bundles/r3_u2_ast_partial_B')
    spec = replace(original.spec, bundle_id='bundle:r3-n1-ast-candidate-' + ident, decision_rule=rule)
    components = dict(original.components)
    components['calibration/ast.json'] = json.dumps(metadata, allow_nan=False).encode()
    write_bundle(out / 'bundle', spec, components, extra={'stage': 'candidate_pending_review', 'officially_accepted': False})
    reloaded = load_bundle(out / 'bundle')
    meta = json.loads(reloaded.components['calibration/ast.json'].read_text())
    restored = apply_temperature(raw, meta['temperature'], epsilon=meta['epsilon']) if use else raw
    np.testing.assert_array_equal(chosen, restored)
    require([rule.decide(x) for x in chosen] == [reloaded.spec.decision_rule.decide(x) for x in restored], 'Reload decisions differ')
    pq.write_table(pa.Table.from_pylist([{**r, 'temperature_score': s, 'retained_score': c, 'decision': rule.decide(c)} for r,s,c in zip(operating,scaled,chosen)]), out / 'operating_predictions.parquet')
    bins = {name: reliability(labels, scores) for name, scores in [('raw', raw), ('temperature', scaled)]}
    write('reliability.json', bins)
    report = {'status': 'candidate_computed_not_officially_accepted', 'temperature': temp,
              'temperature_accepted_by_candidate_rule': use, 'raw_metrics_at_0_5': rm, 'temperature_metrics_at_0_5': cm,
              'decision_rule': rule.to_dict(), 'operating_macro_f1': f1, 'operating_fpr': fpr,
              'canonical_membership_verified': True, 'recording_score_reload_parity': True,
              'waveform_precision_parity': 'pending; no inference or GPU execution performed',
              'threshold_distance_raw': min(abs(x-rule.threshold) for x in chosen) if rule.threshold is not None else None,
              'policy_sha256': digest(policy_path), 'original_lock_sha256': lock_hash,
              'source_sha256': {str(p.relative_to(root)): digest(p) for p in [Path(__file__), root/'scripts/r3_n1_ast_preflight.py',root/'scripts/r3_n1_lr_operating.py',root/'src/echoproof/evaluation/calibrate.py',root/'src/echoproof/evaluation/threshold.py',root/'src/echoproof/evaluation/metrics.py']},
              'limitations': ['Operating metrics are development results, not independent test performance', 'Formal reviewer/team approval pending', 'No CPU-fp32 versus export-fp16 waveform parity at selected threshold yet']}
    require(verify(root)['hashes'] == before['hashes'], 'Inputs changed during execution')
    require(digest(root/'locks/B_lock.json') == lock_hash, 'Lock changed')
    report['outputs_sha256'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file()}
    write('evidence.json', report)
    return out, report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, required=True)
    args = p.parse_args()
    out, report = execute(args.project_root)
    print(out)
    print(json.dumps(report, indent=2))

