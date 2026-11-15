"""Build the Protocol A lock record. Default is a dry run (prints a summary); --write creates locks/A_lock.json and refuses to overwrite.
Does not touch locks/B_lock.json or locks/B_lock_gr3.json. Refuses a dirty working tree unless --allow-dirty.
Usage: python scripts/r4_n1/a_lock.py [--write [--allow-dirty] [--out locks/A_lock.json]] [--verify locks/A_lock.json]"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from echoproof.evaluation import r4_n1_a_lock as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out", default="locks/A_lock.json")
    ap.add_argument("--verify", help="re-verify an existing lock record against the files")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args()
    root = a.root.resolve()
    if a.verify:
        record = json.loads((root / a.verify).read_text(encoding="utf-8"))
        L.verify_record(root, record)
        print("verified", a.verify, L.sha(root / a.verify))
        return
    record = L.build_lock(root, allow_dirty=a.allow_dirty or not a.write)
    L.verify_record(root, record)
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    print(f"{record['lock_id']} (dry run: {not a.write}); dirty tree: {record['code']['working_tree_dirty']} ({len(record['code']['uncommitted_entries'])} entries); {len(text)} bytes")
    for model, c in record["candidates"].items():
        o, r = c["operating_role"], c["replay"]
        print(f"  {model:6s} {c['model_version']:22s} rule >= {c['decision_rule']['threshold']:.6f}  {c['calibration']['mapping']:11s} accepted={c['calibration']['accepted']!s:5s} "
              f"macroF1 {o['macro_f1']:.4f} FPR {o['genuine_fpr']:.4f} | replay {r['recordings']} rec, flips {r['decision_flips']}, max diff {r['max_score_difference']:.2e}")
    if a.write:
        out = root / a.out
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.write_text(text + "\n", encoding="utf-8", newline="\n")
        L.verify_record(root, json.loads(out.read_text(encoding="utf-8")))
        print("written", out, L.sha(out))


main()
