#!/usr/bin/env python3
"""End-to-end install check against the locked production Protocol B bundle."""
from __future__ import annotations

import json
import importlib.metadata
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def close_enough(actual, expected, tolerance=1e-8):
    return isinstance(actual, (float, int)) and math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance


def main() -> int:
    fixture = ROOT / "tests/fixtures/r4_n3/fixture.audio"
    expectation_path = ROOT / "tests/fixtures/r4_n3/expected_prediction.json"
    bundle = ROOT / "bundles/r3_n1_wavlm_head_B_calibrated"
    try:
        from echoproof import app_support
        try:
            installed_transformers = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            return fail("transformers is not installed; install requirements/app.lock.txt first")
        if installed_transformers != app_support.PINNED_TRANSFORMERS:
            return fail(f"transformers {installed_transformers} is installed; expected pinned {app_support.PINNED_TRANSFORMERS}")
        status = app_support.prepare_bundle(root=ROOT)
        if not status.ok or not status.verified_against_lock:
            return fail(f"Production bundle did not load and verify against B_lock_gr3.json: {status.message}")
        expected = json.loads(expectation_path.read_text(encoding="utf-8"))
        required = ("score", "assessment", "score_type", "threshold")
        if expected.get("status") != "verified" or any(expected.get(key) is None for key in required):
            return fail(f"{expectation_path.relative_to(ROOT)} has no recorded production expectation. Record it only after an independent verified CLI run.")
        if not fixture.is_file():
            return fail(f"Fixture not found: {fixture}")

        cli = subprocess.run([sys.executable, "-m", "echoproof", "predict", "--bundle", str(bundle), "--audio", str(fixture)],
                             cwd=ROOT, capture_output=True, text=True, check=False)
        if cli.returncode:
            return fail(f"CLI prediction failed ({cli.returncode}): {cli.stderr.strip() or cli.stdout.strip()}")
        prediction = json.loads(cli.stdout)
        if prediction.get("status") != "success":
            return fail(f"CLI prediction returned status {prediction.get('status')!r}")
        if not close_enough(prediction.get("score"), expected["score"]):
            return fail(f"CLI score {prediction.get('score')} differs from stored score {expected['score']}")
        for key in ("assessment", "score_type"):
            if prediction.get(key) != expected[key]:
                return fail(f"CLI {key} {prediction.get(key)!r} differs from stored {expected[key]!r}")
        if not close_enough(status.bundle.spec.decision_rule.threshold, expected["threshold"]):
            return fail("Loaded production threshold differs from stored expectation")

        from streamlit.testing.v1 import AppTest
        app_test = AppTest.from_file(str(ROOT / "app/main.py"), default_timeout=300).run()
        if app_test.exception:
            return fail(f"Streamlit AppTest startup exception: {app_test.exception}")
        if not app_test.file_uploader:
            return fail("Streamlit AppTest has no audio uploader")
        app_test.file_uploader[0].upload("fixture.wav", fixture.read_bytes())
        app_test.run()
        next(button for button in app_test.button if button.label.startswith("Analyze")).click()
        app_test.run()
        if app_test.exception:
            return fail(f"Streamlit AppTest prediction exception: {app_test.exception}")
        payload = json.loads(app_test.json[0].value)
        if not close_enough(payload.get("reported_score"), expected["score"]):
            return fail("Streamlit result score differs from stored fixture expectation")
        if payload.get("score_type") != expected["score_type"] or payload.get("decision_rule", {}).get("threshold") != expected["threshold"]:
            return fail("Streamlit score type or threshold differs from stored fixture expectation")
        headlines = [element.value for elements in (app_test.success, app_test.warning) for element in elements]
        decision_text = " ".join(str(value) for value in headlines).lower()
        expected_label = "synthetic" if expected["assessment"] == "synthetic" else "genuine"
        if expected_label not in decision_text:
            return fail("Streamlit decision label differs from stored fixture expectation")
        print(json.dumps({"status": "passed", "bundle": str(bundle.relative_to(ROOT)), "assessment": expected["assessment"],
                          "score": prediction["score"], "score_type": prediction["score_type"], "streamlit_apptest": "passed"}, indent=2))
        return 0
    except Exception as exc:
        return fail(f"Verification error ({type(exc).__name__}): {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
