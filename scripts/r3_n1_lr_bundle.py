"""Package a new LR development bundle and verify frozen recording-score parity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from echoproof.artifacts import file_digest
from echoproof.bundle import load_bundle, write_bundle
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH
from echoproof.inference import load_lr_bundle

CONTRACT = ROOT/'configs/r3_n1_lr_contract.json'


def package(evaluation_dir, fit_dir, output):
    import pyarrow.parquet as pq

    c = json.loads(CONTRACT.read_text(encoding='utf-8'))
    root = (ROOT/c['output_root']).resolve()
    evaluation_dir, fit_dir, output = (Path(x).resolve() for x in (evaluation_dir, fit_dir, output))
    if any(p == root or not p.is_relative_to(root) for p in (evaluation_dir, fit_dir, output)):
        raise ValueError('handoff paths must be under the LR run root')
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing handoff')
    ev_path, fit_path = evaluation_dir/'evidence.json', fit_dir/'temperature.json'
    ev, fit = json.loads(ev_path.read_text(encoding='utf-8')), json.loads(fit_path.read_text(encoding='utf-8'))
    if (ev.get('task') != 'R3-N1-LR operating evaluation' or ev.get('status') != 'complete'
            or ev.get('source_contract_sha256') != file_digest(CONTRACT) or ev.get('source_fit_sha256') != file_digest(fit_path)
            or fit.get('source_contract_sha256') != file_digest(CONTRACT) or fit.get('selected_bundle_sha256') != c['bundle_sha256']
            or fit.get('status') != 'calibration_fit_complete' or fit.get('temperature') != ev.get('temperature')
            or ev.get('acceptance_rule') != c['calibration_acceptance']):
        raise ValueError('fit or evaluation provenance mismatch')
    accepted = ev.get('temperature_accepted')
    if type(accepted) is not bool or ev.get('score_type') != ('calibrated_probability' if accepted else 'model_score'):
        raise ValueError('inconsistent evaluation score type')
    if accepted != (ev['temperature_fixed_0_5']['log_loss'] < ev['raw_fixed_0_5']['log_loss'] and
                    ev['temperature_fixed_0_5']['brier_score'] <= ev['raw_fixed_0_5']['brier_score']):
        raise ValueError('acceptance does not match the declared rule')
    original = ROOT/c['bundle']
    if file_digest(original/'bundle.json') != c['bundle_sha256']:
        raise ValueError('selected bundle differs from contract')
    load_bundle(original)
    for name, sha in c['model_components_sha256'].items():
        if file_digest(original/'model'/name) != sha:
            raise ValueError('selected model components differ from contract')
    pred_path = evaluation_dir/'operating_predictions.parquet'
    if file_digest(pred_path) != ev['outputs']['operating_predictions.parquet']:
        raise ValueError('operating predictions checksum mismatch')
    rows = pq.read_table(pred_path).to_pylist()
    if len(rows) != c['operating']['recordings'] or len({r['recording_id'] for r in rows}) != len(rows):
        raise ValueError('operating recording coverage mismatch')
    decision = ev['operating_rule']
    rule = DecisionRule(1, decision['kind'], decision['threshold'])
    spec = BundleSpecification(1, 'bundle:r3-n1-lr-development', 'logistic_regression:B', PREPROCESSING_HASH,
                               rule, 'genuine', 'tts')
    metadata = {'schema': 'r3_n1_lr_calibration_v1', 'method': 'temperature_on_log_odds',
                'temperature': fit['temperature'], 'epsilon': fit['epsilon'], 'temperature_accepted': accepted,
                'score_type': ev['score_type'], 'source_contract_sha256': file_digest(CONTRACT),
                'source_fit_sha256': file_digest(fit_path), 'source_evaluation_sha256': file_digest(ev_path)}
    output.mkdir(parents=True, exist_ok=False)
    write_bundle(output/'bundle', spec, {'model/weights.npz': original/'model/weights.npz',
                 'model/lr_model.json': original/'model/lr_model.json',
                 'calibration/lr_calibration.json': (json.dumps(metadata, sort_keys=True)+'\n').encode()},
                 extra={'stage': 'r3_n1_lr_development', 'official_promotion': False})
    loaded = load_lr_bundle(output/'bundle')
    errors, flips = [], 0
    for row in rows:
        raw = row['raw_score']
        actual = loaded.calibrate(raw) if loaded.calibrate else raw
        errors.append(abs(actual-row['chosen_score']))
        flips += loaded.spec.decision_rule.decide(actual) != row['decision']
    max_error = max(errors)
    if max_error > 1e-12 or flips:
        raise ValueError(f'reloaded recording score/decision parity failed: {max_error}, {flips}')
    report = {'task': 'R3-N1-LR bundle handoff', 'status': 'development_handoff_passed',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'score_type': loaded.score_type,
              'temperature_accepted': accepted, 'operating_recordings_checked': len(rows),
              'recording_score_max_abs': max_error, 'score_tolerance': 1e-12, 'decision_flips': flips,
              'source_evaluation_sha256': file_digest(ev_path), 'source_fit_sha256': file_digest(fit_path),
              'bundle_json_sha256': file_digest(output/'bundle/bundle.json'), 'official_lock_or_bundle_changed': False,
              'scope': 'recording-score and decision parity using exported raw scores; waveform scoring separately verified in R3-M2'}
    (output/'evidence.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-dir', required=True, type=Path)
    p.add_argument('--fit-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(package(args.evaluation_dir, args.fit_dir, args.output), indent=2))
