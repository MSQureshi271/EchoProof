"""Run and record the fixture checks required before the diagnostic GPU fit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch02_contract import ROOT, EXP, read, verify_contract
from prepare_batch import sha


def check():
    path = EXP / 'configs/batch02_resolved.json'
    contract = read(path)
    verify_contract(contract, path)
    reports, runs = EXP / 'reports/batch02', EXP / 'runs/batch02'
    reports.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    marker = runs / 'checks_active.json'
    attempts = runs / 'check_attempts.jsonl'
    if marker.exists() or (runs / 'active_attempt.json').exists():
        raise ValueError('an active/unclosed attempt needs investigation')
    prior = [json.loads(s) for s in attempts.read_text().splitlines()] if attempts.exists() else []
    remaining = contract['budget']['max_audit_and_export_gpu_minutes']*60-sum(a['elapsed_seconds'] for a in prior)
    if remaining <= 0:
        raise ValueError('fixture-check budget exhausted')
    junit = reports / 'preflight.xml'
    with marker.open('x', encoding='utf-8') as handle:
        json.dump({'contract_sha256': sha(path), 'remaining_seconds': remaining}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    started = time.monotonic()
    status = 'failed'
    try:
        env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(EXP/'tests'),
               str(ROOT/'tests/test_r3_bigru_frontend.py'), '--junitxml', str(junit),
               '--basetemp', str(runs / ('fixture_tmp_'+str(time.time_ns())))]
        result = subprocess.run(cmd, cwd=ROOT, env=env, timeout=remaining, check=False)
        suites = ET.parse(junit).getroot().findall('.//testsuite')
        counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
        if result.returncode or counts['tests'] == 0 or any(counts[k] for k in ('failures', 'errors', 'skipped')):
            raise ValueError(f'fixture preflight did not pass completely: {counts}')
        status = 'passed'
        report = {'status': status, 'counts': counts, 'contract_sha256': sha(path), 'junit_sha256': sha(junit),
                  'elapsed_seconds': time.monotonic()-started, 'scope': 'synthetic fixtures only; no training data fit'}
        (reports / 'preflight.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        return report
    finally:
        with attempts.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'status': status, 'elapsed_seconds': time.monotonic()-started})+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        marker.unlink()


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
