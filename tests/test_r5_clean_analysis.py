"""Tests use fabricated predictions and never read sealed split rows or audio."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from echoproof.evaluation import r5_clean_analysis as A  # noqa: E402

spec = importlib.util.spec_from_file_location("r5_clean_analysis_script", ROOT / "scripts/r5_n1_clean_analysis.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def fixture():
    rows, entries = {}, {}
    for protocol in ("A", "B"):
        components = ["genuine", "melo", "vits", "chatterbox"] + (["xtts"] if protocol == "A" else [])
        for model in A.MODELS:
            key = f"{protocol}:{model}"
            threshold = 0.8 if model == "LR" else 0.5
            entries[key] = {"decision_rule": {"threshold": threshold}, "model_version": key}
            rows[key] = []
            for speaker in range(101, 105):
                for component in components:
                    score = 0.7 if component != "genuine" else 0.2
                    rows[key].append({"recording_id": f"phonemedf:{component}:{speaker}-1-0001",
                                      "group_id": f"speaker:{speaker}", "component": component,
                                      "label": int(component != "genuine"), "status": "success", "score": score,
                                      "assessment": "synthetic" if score >= threshold else "genuine",
                                      "score_type": "model_score" if model == "LR" else "calibrated_probability",
                                      "model_version": key, "failure_reason": None})
    return rows, entries, {"A": 20, "B": 16}


def test_panels_support_locked_thresholds_and_paired_intervals():
    rows, entries, counts = fixture()
    out = A.analyze(rows, entries, counts, n_resamples=40, seed=42)
    assert [(out[p]["genuine"], out[p]["tts"]) for p in A.PANELS] == [(4, 16), (4, 8), (4, 4)]
    for panel in out.values():
        assert panel["models"]["LR"]["metrics"]["tts_recall"] == 0
        assert panel["models"]["BiGRU"]["metrics"]["tts_recall"] == 1
        assert panel["bootstrap"]["differences"]["LR_minus_BiGRU"]["tts_recall"]["ci95"] == [-1.0, -1.0]
        assert panel["models"]["LR"]["score_type"] == "model_score"
    assert A.analyze(rows, entries, counts, n_resamples=40, seed=42) == out


def test_ranking_with_ties_matches_sklearn():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.7, 0.7, 0.1, 0.2])
    assert A._scalar_metrics(y, scores, 0.7)["roc_auc"] == roc_auc_score(y, scores)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "bad_label", "decision", "model", "group", "cross_model"])
def test_validation_refuses_bad_or_unpaired_predictions(mutation):
    rows, entries, counts = fixture()
    target = rows["A:LR"]
    if mutation == "missing": target.pop()
    elif mutation == "duplicate": target[0] = copy.deepcopy(target[1])
    elif mutation == "nonfinite": target[0]["score"] = float("nan")
    elif mutation == "bad_label": target[0]["label"] = 1
    elif mutation == "decision": target[0]["assessment"] = "synthetic"
    elif mutation == "model": target[0]["model_version"] = "wrong"
    elif mutation == "group": target[0]["group_id"] = "speaker:999"
    else:
        target[0]["recording_id"] = "phonemedf:genuine:999-1-0001"
        target[0]["group_id"] = "speaker:999"
    with pytest.raises(A.AnalysisError):
        A.analyze(rows, entries, counts, n_resamples=2)


def test_no_model_comparison_with_different_recording_sets():
    rows, entries, counts = fixture()
    rows["B:AST"].pop()
    with pytest.raises(A.AnalysisError, match="denominator"):
        A.validate_predictions(rows, entries, counts)


def test_plot_generation_on_synthetic_predictions(tmp_path):
    pytest.importorskip("matplotlib")
    rows, entries, counts = fixture()
    results = A.analyze(rows, entries, counts, n_resamples=5)
    script.plots(tmp_path, rows, entries, results)
    assert len(list(tmp_path.glob("*.png"))) == 12
    assert all(p.stat().st_size > 1000 for p in tmp_path.glob("*.png"))


def test_incomplete_run_is_refused_before_locks_or_prediction_reads(tmp_path):
    (tmp_path / "integrity_report.json").write_text(json.dumps({"status": "integrity_failed", "passed": False}))
    (tmp_path / "run_identity.json").write_text("{}")
    (tmp_path / "test_access_log.jsonl").write_text('{"event":"sealed_run_start"}\n{"event":"sealed_run_end","status":"complete"}\n')
    with pytest.raises(A.AnalysisError, match="integrity"):
        script.read_inputs(ROOT, tmp_path)


def test_prediction_hash_is_verified_before_analysis(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows, entries, _ = fixture()
    lock_files = {}
    for protocol in ("A", "B"):
        path = tmp_path / f"lock_{protocol}.json"
        path.write_text(protocol)
        lock_files[protocol] = path.name
    identity = {"entries": [], "lock_sha256": {p: script.sha(tmp_path / name) for p, name in lock_files.items()},
                "splits": {p: {"expected_split_hash": p} for p in lock_files}}
    summary = {"status": "complete", "passed": True, "bundles": {}}
    pairs = []
    for key, data in rows.items():
        protocol, model = key.split(":")
        record = {"bundle_id": key, "bundle_json_sha256": key + "-hash", "model_version": key,
                  "decision_rule": entries[key]["decision_rule"]}
        item = {"key": key, **record}
        identity["entries"].append(item)
        name = f"predictions_{protocol}_{model}.parquet"
        pq.write_table(pa.Table.from_pylist(data), tmp_path / name)
        summary["bundles"][key] = {"predictions_file": name, "bundle_json_sha256": record["bundle_json_sha256"],
                                   "model_version": key, "predictions_sha256": script.sha(tmp_path / name),
                                   "integrity": {"passed": True, "expected": len(data), "records": len(data),
                                                 "scored_ok": len(data), "problems": {}}}
        pairs.append((SimpleNamespace(key=key, protocol=protocol, model=model), record))
    report = {"protocols": {p: {"split_sha256": p, "support": {"test|-|genuine|genuine": {"recordings": 4},
              "test|-|known_family|tts": {"recordings": 16 if p == "A" else 8},
              **({"test|-|unseen_final|tts": {"recordings": 4}} if p == "B" else {})}} for p in ("A", "B")}}
    (tmp_path / "reports/r1").mkdir(parents=True)
    (tmp_path / "reports/r1/split_report.json").write_text(json.dumps(report))
    (tmp_path / "run_identity.json").write_text(json.dumps(identity))
    (tmp_path / "integrity_report.json").write_text(json.dumps(summary))
    (tmp_path / "test_access_log.jsonl").write_text('{"event":"sealed_run_start"}\n{"event":"sealed_run_end","status":"complete"}\n')
    with patch.object(script.locks, "LOCKS", lock_files), patch.object(script.locks, "SPLIT_HASHES", {"A": "A", "B": "B"}), \
         patch.object(script.locks, "verify_locks", return_value={}), patch.object(script.locks, "verify_all_bundles"), \
         patch.object(script.locks, "candidate_bundles", return_value=pairs):
        loaded, found_entries, counts, support, _ = script.read_inputs(tmp_path, tmp_path)
        script.verify_support(A.validate_predictions(loaded, found_entries, counts), support)
        path = tmp_path / "predictions_A_LR.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
        with pytest.raises(A.AnalysisError, match="prediction hash mismatch"):
            script.read_inputs(tmp_path, tmp_path)
