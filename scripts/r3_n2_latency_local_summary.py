"""Summarise local R3-N2 latency reports (from `scripts/r3_n2_latency_local.py`) as Markdown tables. Reads JSON only; needs numpy-free Python.

Usage: python scripts/r3_n2_latency_local_summary.py REPORT.json [REPORT.json ...] [--out summary.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL = {"lr": "LR (calibrated)", "wavlm": "WavLM (calibrated)", "bigru": "BiGRU (calibrated)"}
STAGE_ORDER = ["decode", "chunk_construction", "feature_extraction", "model_prediction", "aggregation", "calibration", "threshold_decision", "other"]


def ms(ns):
    return None if ns is None else ns / 1e6


def fmt(value, digits=1):
    return "n/a" if value is None else f"{value:,.{digits}f}"


def seconds_or_ms(value_ms):
    return "n/a" if value_ms is None else (f"{value_ms / 1000:.2f} s" if value_ms >= 1000 else f"{value_ms:.1f} ms")


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def summarise(report: dict) -> str:
    model = LABEL.get(report["model"], report["model"])
    groups = report["groups"]
    cfg = report["configuration"]
    lines = [f"### {model}", "", f"Bundle `{Path(report['bundle']['bundle_path']).as_posix()}` (bundle.json sha256 `{report['bundle']['bundle_json_sha256'][:16]}…`), "
             f"model version `{report['model_version']}`, score type `{report['score_type']}`. Warm-up {cfg['warmup_calls']} calls; timed calls per duration {cfg['measured_calls']}; "
             f"fresh-process repetitions {cfg['fresh_process_repetitions']}.", ""]
    rows = []
    for key, g in groups.items():
        if g.get("status", "").startswith("error") or "summary_ns" not in g:
            rows.append([f"{key} s", "error", "", "", "", "", g.get("errors", [""])[0] if g.get("errors") else ""])
            continue
        total = g["summary_ns"]["total"]
        ref = g["production_reference"]
        p50, p95 = ms(total["p50"]), ms(total["p95"])
        rows.append([f"{key} s", str(ref["chunks"]), str(total["count"]), seconds_or_ms(p50), seconds_or_ms(p95), seconds_or_ms(ms(total["max"])), f"{p50 / 1000 / int(key):.3f}",
                     "exact" if g["parity"]["reference_match"] and not g["parity"]["decision_flips"] else f"max delta {g['parity']['max_abs_score_delta']:.2e}, flips {g['parity']['decision_flips']}"])
    lines += ["**Warm end-to-end latency (production `analyze`, batch 1, decode to decision)**", "",
              table(["Clip", "Chunks", "Timed calls", "p50", "p95", "Max", "p50 / audio second", "Parity vs reference"], rows), ""]
    rows = []
    for key, g in groups.items():
        if "summary_ns" not in g:
            continue
        s = g["summary_ns"]
        total = ms(s["total"]["p50"])
        rows.append([f"{key} s"] + [f"{fmt(ms(s[name]['p50']), 2)} ({100 * ms(s[name]['p50']) / total:.1f}%)" if name in s else "n/a" for name in STAGE_ORDER])
    lines += ["**Stage breakdown, p50 in ms (share of the total p50)**", "", table(["Clip"] + STAGE_ORDER, rows), ""]
    rows = []
    for key, g in groups.items():
        f = g.get("fresh_process_summary_ns")
        if not f or not f["process_to_first_prediction_ns"]["count"]:
            rows.append([f"{key} s", "n/a", "n/a", "n/a", "n/a", str(len(g.get("fresh_process_errors", [])))])
            continue
        n = f["process_to_first_prediction_ns"]["count"]
        rows.append([f"{key} s", str(n), seconds_or_ms(ms(f["process_to_first_prediction_ns"]["p50"])), seconds_or_ms(ms(f["bundle_load_ns"]["p50"])),
                     seconds_or_ms(ms(f["first_prediction_ns"]["p50"])), str(len(g.get("fresh_process_errors", [])))])
    lines += ["**Cold interpreter (fresh subprocess; not a cold machine or cold disk cache), p50**", "",
              table(["Clip", "Successes", "Process start to first prediction", "Bundle load", "First prediction", "Failures"], rows), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    text = []
    for path in args.reports:
        text.append(summarise(json.loads(Path(path).read_text(encoding="utf-8"))))
    result = "\n".join(text)
    if args.out:
        Path(args.out).write_text(result, encoding="utf-8", newline="\n")
    print(result)


if __name__ == "__main__":
    main()
