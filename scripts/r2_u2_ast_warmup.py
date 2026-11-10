"""Official R2-U2 command entry point; delegates to hash-pinned experiment code."""
import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'experiments/r2_u2/scripts/launch.py'

if not ENTRY.is_file():
    raise SystemExit(f'R2-U2 experiment implementation is missing: {ENTRY}')

# Keep one implementation: experiment/launch.py owns source checks and provenance.
sys.path.insert(0, str(ENTRY.parent))
sys.argv[0] = str(ENTRY)
runpy.run_path(str(ENTRY), run_name='__main__')
