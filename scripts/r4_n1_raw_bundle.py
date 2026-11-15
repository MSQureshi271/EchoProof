"""Build the R4-N1 development bundles for LR-A and BiGRU-A (raw scores with the serialized threshold). Never overwrites."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from echoproof.models.r4_n1_raw_bundles import MODELS, build_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    for model in (MODELS if args.model == "all" else [args.model]):
        evidence = build_bundle(args.root, model)
        print(json.dumps({"model": model, "bundle": evidence["bundle"], "bundle_json_sha256": evidence["bundle_json_sha256"],
                          "decision_rule": evidence["decision_rule"], "saved_score_parity": evidence["saved_score_parity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
