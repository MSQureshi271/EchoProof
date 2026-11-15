"""AST-only R4-N1 CPU calibration and candidate packaging."""
import argparse
import shutil
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from echoproof.evaluation.r4_n1_ast import run
from echoproof.models.r4_n1_ast_bundle import package

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bundle', type=Path)
    a = p.parse_args()
    if a.bundle and shutil.disk_usage(ROOT).free < 400_000_000:
        raise SystemExit('Insufficient space for a separate candidate bundle')
    result = run(ROOT, ROOT/'configs/r4_n1_ast_calibration_a.json', a.output)
    if a.bundle:
        package(ROOT, a.output, a.bundle)
    print({k:result[k] for k in ('method','parameters','accepted','macro_f1','rule','raw','calibrated')})
