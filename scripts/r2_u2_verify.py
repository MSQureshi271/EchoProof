"""Official R2-U2 artifact verifier; delegates to the pinned acceptance tool."""
import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'experiments/r2_u2/acceptance/verify_handoff.py'
if not ENTRY.is_file():
    raise SystemExit(f'R2-U2 acceptance verifier is missing: {ENTRY}')
sys.argv[0] = str(ENTRY)
runpy.run_path(str(ENTRY), run_name='__main__')
