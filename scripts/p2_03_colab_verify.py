"""Run the Phase 2 contract checks and record a unique verification report."""
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def run(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / 'src'))
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src')
    cmd = [sys.executable, '-m', 'unittest', 'discover', '-s', str(root / 'tests'), '-p', 'test_p2_03_contracts.py', '-v']
    result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    skipped = 'skipped=' in result.stderr or 'skipped ' in result.stderr
    report = {'task': 'P2-03', 'status': 'passed' if result.returncode == 0 and not skipped else 'needs_attention',
              'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
              'scope': 'configuration and synthetic metadata contracts; no dataset scan'}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    out = root / 'reports' / 'phase2' / ('p2_03-' + run_id)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return out, report
