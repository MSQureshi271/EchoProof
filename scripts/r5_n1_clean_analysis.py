"""Analyze a completed clean sealed run without accessing audio or split rows.

python scripts/r5_n1_clean_analysis.py --run reports/final/results/r5_n1_sealed_run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from echoproof.evaluation import r5_loaders as locks  # noqa: E402
from echoproof.evaluation import r5_clean_analysis as analysis  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_inputs(root, run):
    """Verify prediction and lock provenance without opening sealed manifests."""
    import pyarrow.parquet as pq

    root, run = Path(root), Path(run)
    summary_path = run / "integrity_report.json"
    identity_path = run / "run_identity.json"
    log_path = run / "test_access_log.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    raw_log = log_path.read_bytes()
    if not raw_log.endswith(b"\n"):
        raise analysis.AnalysisError("access log is incomplete")
    events = [json.loads(line) for line in raw_log.splitlines()]
    if not events or events[0].get("event") != "sealed_run_start" or events[-1].get("event") != "sealed_run_end" or events[-1].get("status") != "complete":
        raise analysis.AnalysisError("access log does not record a completed sealed run")
    if summary.get("status") != "complete" or summary.get("passed") is not True:
        raise analysis.AnalysisError("sealed run integrity is not complete and passed")
    records = locks.verify_locks(root, deep=True)
    locks.verify_all_bundles(root, records)
    by_key = {spec.key: (spec, record) for spec, record in locks.candidate_bundles(records)}
    entries_list = identity.get("entries", [])
    entries = {entry["key"]: entry for entry in entries_list}
    if len(entries) != len(entries_list) or set(entries) != set(by_key) or set(summary.get("bundles", {})) != set(by_key):
        raise analysis.AnalysisError("run does not contain exactly the eight locked bundles")
    for protocol in ("A", "B"):
        if identity.get("lock_sha256", {}).get(protocol) != locks.sha256_file(root / locks.LOCKS[protocol]):
            raise analysis.AnalysisError(f"{protocol}: run lock hash mismatch")
        if identity.get("splits", {}).get(protocol, {}).get("expected_split_hash") != locks.SPLIT_HASHES[protocol]:
            raise analysis.AnalysisError(f"{protocol}: run split identity mismatch")
    rows, prediction_hashes = {}, {}
    for key, (spec, record) in by_key.items():
        entry, item = entries[key], summary["bundles"][key]
        for field, expected in (("bundle_id", record["bundle_id"]), ("bundle_json_sha256", record["bundle_json_sha256"]),
                                ("model_version", record["model_version"]), ("decision_rule", record["decision_rule"])):
            if entry.get(field) != expected:
                raise analysis.AnalysisError(f"{key}: run {field} differs from lock")
        expected_file = f"predictions_{spec.protocol}_{spec.model}.parquet"
        if item.get("predictions_file") != expected_file or item.get("bundle_json_sha256") != record["bundle_json_sha256"] or item.get("model_version") != record["model_version"]:
            raise analysis.AnalysisError(f"{key}: integrity provenance mismatch")
        if item.get("integrity", {}).get("passed") is not True:
            raise analysis.AnalysisError(f"{key}: integrity did not pass")
        path = run / expected_file
        prediction_hashes[key] = sha(path)
        if prediction_hashes[key] != item.get("predictions_sha256"):
            raise analysis.AnalysisError(f"{key}: prediction hash mismatch")
        rows[key] = pq.read_table(str(path)).to_pylist()
        integrity = item["integrity"]
        if (integrity.get("expected") != len(rows[key]) or integrity.get("records") != len(rows[key])
                or integrity.get("scored_ok") != len(rows[key]) or any(integrity.get("problems", {}).values())):
            raise analysis.AnalysisError(f"{key}: integrity counts or problems are inconsistent")
    report_path = root / "reports/r1/split_report.json"
    split_report = json.loads(report_path.read_text(encoding="utf-8"))
    counts, support = {}, {}
    for protocol in ("A", "B"):
        section = split_report["protocols"][protocol]
        if section["split_sha256"] != locks.SPLIT_HASHES[protocol]:
            raise analysis.AnalysisError(f"{protocol}: split report identity mismatch")
        support[protocol] = {k.split("|")[2]: v["recordings"] for k, v in section["support"].items() if k.startswith("test|-")}
        counts[protocol] = sum(support[protocol].values())
    return rows, entries, counts, support, {"predictions_sha256": prediction_hashes, "integrity_report_sha256": sha(summary_path),
                                          "run_identity_sha256": sha(identity_path), "test_access_log_sha256": sha(log_path),
                                          "split_report_sha256": sha(report_path), "lock_sha256": identity["lock_sha256"]}


def verify_support(indexed, support):
    expected = {"A_known": (support["A"]["genuine"], support["A"]["known_family"]),
                "B_known": (support["B"]["genuine"], support["B"]["known_family"]),
                "B_unseen": (support["B"]["genuine"], support["B"]["unseen_final"])}
    for panel, (genuine, tts) in expected.items():
        protocol, components = analysis.PANELS[panel]
        subset = [r for r in indexed[f"{protocol}:LR"].values() if r["component"] in components]
        if (sum(r["label"] == 0 for r in subset), sum(r["label"] == 1 for r in subset)) != (genuine, tts):
            raise analysis.AnalysisError(f"{panel}: panel support differs from split report")


def plots(out, rows, entries, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve

    for panel, (protocol, components) in analysis.PANELS.items():
        for kind in ("confusion", "roc", "pr", "reliability"):
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for ax, model in zip(axes.flat, analysis.MODELS):
                key = f"{protocol}:{model}"
                subset = [r for r in rows[key] if r["component"] in components]
                y, scores = [r["label"] for r in subset], [r["score"] for r in subset]
                item = results[panel]["models"][model]
                if kind == "confusion":
                    cm = item["metrics"]["confusion_matrix"]
                    ax.imshow(cm, cmap="Blues")
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
                    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["genuine", "TTS"], yticklabels=["genuine", "TTS"], xlabel="Predicted", ylabel="Actual")
                elif kind == "roc":
                    fpr, tpr, _ = roc_curve(y, scores)
                    ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--", color="gray")
                    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
                elif kind == "pr":
                    precision, recall, _ = precision_recall_curve(y, scores)
                    ax.plot(recall, precision)
                    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
                else:
                    bins = [b for b in item["reliability_bins"] if b["count"]]
                    ax.plot([b["mean_score"] for b in bins], [b["observed_rate"] for b in bins], marker="o")
                    ax.plot([0, 1], [0, 1], "--", color="gray")
                    ax.set(xlabel="Mean score", ylabel="Observed TTS fraction", xlim=(0, 1), ylim=(0, 1))
                ax.set_title(f"{model} ({item['score_type']})")
            fig.suptitle(f"{panel} — {kind}; n={results[panel]['recordings']}")
            fig.tight_layout()
            fig.savefig(out / f"{panel}_{kind}.png", dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="new output folder; default: analysis_clean_v1 inside run")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else args.root / args.run
    out = args.out if args.out else run / "analysis_clean_v1"
    if not out.is_absolute():
        out = args.root / out
    if out.exists():
        raise analysis.AnalysisError("analysis output already exists; refusing to overwrite")
    rows, entries, counts, support, provenance = read_inputs(args.root, run)
    verify_support(analysis.validate_predictions(rows, entries, counts), support)
    results = analysis.analyze(rows, entries, counts, n_resamples=1000, seed=42)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "task": "R5-N1", "scope": "clean A known, B known and B unseen; Protocol D not measured",
              "positive_label": "tts", "decision_rule": "score >= locked threshold", "source": provenance,
              "average_precision_implementation": "sklearn.metrics.average_precision_score",
              "analysis_code_sha256": {"scripts/r5_n1_clean_analysis.py": sha(Path(__file__)),
                                       "src/echoproof/evaluation/r5_clean_analysis.py": sha(Path(analysis.__file__)),
                                       "src/echoproof/evaluation/metrics.py": sha(args.root / "src/echoproof/evaluation/metrics.py"),
                                       "src/echoproof/evaluation/reliability.py": sha(args.root / "src/echoproof/evaluation/reliability.py")},
              "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pyarrow", "matplotlib")},
              "note": "Reliability and probability-loss values for model_score are score diagnostics, not calibrated probability claims. EER is descriptive; no test threshold is selected. Precision is defined as zero when no positives are predicted; bootstrap intervals measure speaker sampling, not seed variability.",
              "panels": results}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["panel", "model", "score_type", "recordings", "speakers", "genuine", "tts", "threshold", "macro_f1", "fpr", "tts_recall", "roc_auc", "eer", "average_precision", "brier_score", "log_loss", "ece_10_bins"]
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for panel, panel_data in results.items():
            for model, item in panel_data["models"].items():
                m = item["metrics"]
                writer.writerow({**{k: panel_data[k] for k in ("recordings", "speakers", "genuine", "tts")},
                                 **{k: m[k] for k in ("threshold", "macro_f1", "fpr", "tts_recall", "roc_auc", "eer", "average_precision", "brier_score", "log_loss")},
                                 "panel": panel, "model": model, "score_type": item["score_type"], "ece_10_bins": item["ece_10_bins"]})
    plots(out, rows, entries, results)
    manifest = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "output_hashes.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Clean analysis written to {out}; Protocol D not measured.")


if __name__ == "__main__":
    main()
