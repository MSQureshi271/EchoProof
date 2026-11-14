"""R3-N1 WavLM, operating stage: the operating role is opened once, after the mapping is frozen (see r3_n1_wavlm.py and WAVLM_CALIBRATION.md)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from echoproof.artifacts import file_digest
from echoproof.contracts import DecisionRule
from echoproof.evaluation.bootstrap import paired_group_bootstrap
from echoproof.evaluation.metrics import compute_metrics
from echoproof.evaluation.r3_n1_wavlm import (WavLMCalibrationError, acceptance, apply_mapping, arrays, brier, check_role_rows, decide, decision_metrics, environment,
                                              load_json, log_loss, now, raw_equivalent_threshold, read_recordings, recipe_hash, row_metrics, rule_json,
                                              saturation, write)
from echoproof.evaluation.reliability import expected_calibration_error, reliability_bins
from echoproof.evaluation.threshold import select_threshold


def _macro_f1_at_half(labels, scores):
    return decision_metrics(labels, [1 if x >= 0.5 else 0 for x in scores])["macro_f1"]


def _genuine_fpr_at_half(labels, scores):
    return decision_metrics(labels, [1 if x >= 0.5 else 0 for x in scores])["genuine_fpr"]


def _log_loss_metric(labels, scores):
    return log_loss(labels, scores)


def _brier_metric(labels, scores):
    return brier(labels, scores)


def probability_metrics(y, scores, bins: int) -> dict:
    m = compute_metrics(y, scores, 0.5)
    return {"log_loss": m["log_loss"], "brier": m["brier_score"], "ece_equal_width": expected_calibration_error(y.tolist(), list(scores), bins),
            "roc_auc": m["roc_auc"], "eer": m["eer"]}


def speaker_diagnostics(y, comp, groups, scores, rule: DecisionRule) -> dict:
    d = decide(rule, scores)
    rows, fa_total = [], 0
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        gen, tts, xt = mask & (y == 0), mask & (y == 1), mask & (comp == "xtts")
        fa, miss = int(d[gen].sum()), int((1 - d[tts]).sum())
        rows.append({"speaker": str(g), "genuine": int(gen.sum()), "false_alarms": fa, "false_alarm_rate": fa / max(int(gen.sum()), 1),
                     "tts": int(tts.sum()), "missed_tts": miss, "xtts": int(xt.sum()), "missed_xtts": int((1 - d[xt]).sum())})
        fa_total += fa
    top = max(rows, key=lambda r: r["false_alarms"])
    return {"per_speaker": rows, "speakers_with_false_alarm_rate_at_least_half": int(sum(r["false_alarm_rate"] >= 0.5 for r in rows)),
            "total_false_alarms": fa_total, "largest_single_speaker": top["speaker"],
            "largest_single_speaker_share_of_false_alarms": (top["false_alarms"] / fa_total) if fa_total else None}


def _fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def comparison_markdown(method, params, acc, table, prob, speakers, rule, threshold) -> str:
    lines = ["# WavLM R3-N1 operating comparison", "",
             f"Selected mapping: **{method}** {params}. Calibration accepted: **{acc['accepted']}** ({acc['reason']}).", "",
             f"Deployed rule: `{rule.kind}` threshold {rule.threshold}; score type `{threshold['score_type']}`. The threshold was chosen on the operating role: "
             "a development result, not independent test performance.", "",
             "## Decision rows (full operating role)", "",
             "| Row | Rule | Macro F1 | Genuine FPR | TTS recall | TP | FP | TN | FN |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in table.items():
        f = v["full"]
        lines.append(f"| {k} | {v['rule']['kind']} {_fmt(v['rule']['threshold'])} | {_fmt(f['macro_f1'])} | {_fmt(f['genuine_fpr'])} | {_fmt(f['tts_recall'])} "
                     f"| {f['tp']} | {f['fp']} | {f['tn']} | {f['fn']} |")
    lines += ["", "## Probability quality (operating role)", "", "| Scores | Log loss | Brier | ECE (10 equal-width bins) | ROC AUC | EER |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in prob.items():
        lines.append(f"| {k} | {_fmt(v['log_loss'])} | {_fmt(v['brier'])} | {_fmt(v['ece_equal_width'])} | {_fmt(v['roc_auc'])} | {_fmt(v['eer'])} |")
    lines += ["", "## Speaker concentration of genuine false alarms at the deployed rule", "",
              f"- Speakers with a false-alarm rate of at least 50%: {speakers['speakers_with_false_alarm_rate_at_least_half']}",
              f"- Total genuine false alarms: {speakers['total_false_alarms']}; largest single speaker: {speakers['largest_single_speaker']} "
              f"(share {_fmt(speakers['largest_single_speaker_share_of_false_alarms'])})", ""]
    return "\n".join(lines)


def operating_stage(root, recipe: dict, run_dir, *, bootstrap_resamples: int | None = None, plots: bool = True) -> dict:
    root, run_dir = Path(root), Path(run_dir)
    fit_dir = run_dir / "fit"
    complete = load_json(fit_dir / "stage_complete.json")
    if complete["recipe_sha256"] != recipe_hash(recipe) or any(file_digest(fit_dir / n) != h for n, h in complete["files"].items()):
        raise WavLMCalibrationError("the frozen fit changed or does not belong to this recipe")
    calibration = load_json(fit_dir / "calibration.json")
    if calibration["operating_predictions_read"]:
        raise WavLMCalibrationError("the fit stage read the operating role")
    out = run_dir / "operating"
    out.mkdir(exist_ok=False)                      # a second run, or a crashed one, is refused: the role is opened once
    write(out / "opened.json", {"opened_utc": now(), "fit_stage_complete_sha256": file_digest(fit_dir / "stage_complete.json"),
                                "mapping_frozen_before_opening": calibration["selected_mapping"]})
    eps, bins = recipe["epsilon"], recipe["reliability"]["bins"]
    inputs = root / recipe["inputs"]["directory"]
    name = "predictions_recordings_operating.parquet"
    if file_digest(inputs / name) != recipe["inputs"]["predictions_sha256"][name]:
        raise WavLMCalibrationError("operating predictions differ from the frozen hash")
    rows = read_recordings(inputs, "operating")
    problems = check_role_rows(rows, "operating", recipe)
    if problems:
        raise WavLMCalibrationError("; ".join(problems))
    y, raw, groups, comp, ids = arrays(rows)

    method, params = calibration["selected_mapping"], calibration["selected_parameters"]
    fits = calibration["candidate_fits"]
    mapped = {c: np.array(apply_mapping(c, fits[c]["params"], raw, eps)) for c in ("temperature", "affine") if fits[c]["ok"]}
    selected = np.array(apply_mapping(method, params, raw, eps)) if method != "identity" else raw
    acc = acceptance(method, fits[method]["ok"] if method != "identity" else True, y, raw, selected)
    deployed = selected if acc["accepted"] else raw
    half = DecisionRule(1, "threshold", 0.5)
    tie = recipe["threshold_rule"]["tie_break"]

    def optimum(scores):
        return select_threshold(y.tolist(), list(scores), tie_break=tie)[0]

    rule_raw = optimum(raw)
    rule_selected = optimum(selected) if method != "identity" else None
    rule_deployed = rule_selected if acc["accepted"] else rule_raw
    rows_ab = {"A_raw_at_0_5": (raw, half), "C_raw_at_raw_optimum": (raw, rule_raw)}
    if method != "identity":
        rows_ab["B_selected_at_0_5"] = (selected, half)
        rows_ab["D_selected_at_its_optimum"] = (selected, rule_selected)
    for c, p in mapped.items():
        if c != method:
            rows_ab[f"B_{c}_reference_at_0_5"] = (p, half)
            rows_ab[f"D_{c}_reference_at_its_optimum"] = (p, optimum(p))
    table = {k: {"rule": rule_json(r), **row_metrics(y, comp, s, r)} for k, (s, r) in rows_ab.items()}

    prob = {"raw": probability_metrics(y, raw, bins)}
    if method != "identity":
        prob["selected"] = probability_metrics(y, selected, bins)
    for c, p in mapped.items():
        if c != method:
            prob[f"{c}_reference"] = probability_metrics(y, p, bins)
    score_sets = [("raw", raw)] + ([("selected", selected)] if method != "identity" else []) + [(f"{c}_reference", p) for c, p in mapped.items() if c != method]
    bin_rows = [{"scores": label, **b} for label, scores in score_sets for b in reliability_bins(y.tolist(), list(scores), bins)]
    with (out / "reliability_bins.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scores", "bin", "lower", "upper", "count", "mean_score", "observed_rate"])
        writer.writeheader()
        writer.writerows(bin_rows)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        for label in dict.fromkeys(r["scores"] for r in bin_rows):
            pts = [r for r in bin_rows if r["scores"] == label and r["count"]]
            ax.plot([r["mean_score"] for r in pts], [r["observed_rate"] for r in pts], marker="o", label=label)
        ax.set(xlabel="Mean score", ylabel="Observed TTS fraction", title="WavLM operating reliability (10 bins)", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "reliability.png", dpi=150)
        plt.close(fig)

    speakers = speaker_diagnostics(y, comp, groups, deployed, rule_deployed)
    write(out / "speaker_diagnostics.json", speakers)
    centre = rule_deployed.threshold if rule_deployed.threshold is not None else 2.0
    near = {str(w): int((np.abs(deployed - centre) <= w).sum()) for w in (0.01, 0.05)}

    n_boot, seed = bootstrap_resamples or recipe["bootstrap"]["resamples"], recipe["bootstrap"]["seed"]
    gl = groups.tolist()
    decisions = {k: [float(v) for v in decide(r, s)] for k, (s, r) in rows_ab.items() if "reference" not in k}
    boot = {"resamples": n_boot, "seed": seed, "unit": "speaker",
            "decisions_macro_f1": paired_group_bootstrap(gl, y.tolist(), decisions, _macro_f1_at_half, n_resamples=n_boot, seed=seed),
            "decisions_genuine_fpr": paired_group_bootstrap(gl, y.tolist(), decisions, _genuine_fpr_at_half, n_resamples=n_boot, seed=seed)}
    if method != "identity":
        by = {"raw": raw.tolist(), "selected": selected.tolist()}
        boot["probability_log_loss"] = paired_group_bootstrap(gl, y.tolist(), by, _log_loss_metric, n_resamples=n_boot, seed=seed)
        boot["probability_brier"] = paired_group_bootstrap(gl, y.tolist(), by, _brier_metric, n_resamples=n_boot, seed=seed)
    boot["note"] = "Conditional on the fitted mapping and the operating-selected threshold; descriptive, not an unbiased estimate."
    write(out / "bootstrap.json", boot)

    sel_name = "predictions_recordings_selection.parquet"
    if file_digest(inputs / sel_name) != recipe["inputs"]["predictions_sha256"][sel_name]:
        raise WavLMCalibrationError("selection predictions differ from the frozen hash")
    sy, sraw, sg, scomp, sids = arrays(read_recordings(inputs, "selection"))
    sdeployed = np.array(apply_mapping(method, params, sraw, eps)) if (acc["accepted"] and method != "identity") else sraw
    diag = {"role": "selection", "note": "The frozen deployed policy applied to the selection role for description; nothing is refit.",
            "deployed": row_metrics(sy, scomp, sdeployed, rule_deployed), "raw_at_0_5": row_metrics(sy, scomp, sraw, half)}
    dm_path = root / recipe["duration_matched"]["file"]
    if dm_path.exists():
        matched = set(load_json(dm_path)["ids"]["selection"])
        mask = np.array([i in matched for i in sids])
        diag["duration_matched"] = ({"recordings": int(mask.sum()), "deployed": row_metrics(sy[mask], scomp[mask], sdeployed[mask], rule_deployed),
                                     "raw_at_0_5": row_metrics(sy[mask], scomp[mask], sraw[mask], half)} if mask.any() else None)
    write(out / "selection_role_diagnostic.json", diag)

    threshold = {"task": "R3-N1-WavLM operating policy", "created_utc": now(), "selected_mapping": method, "selected_parameters": params,
                 "calibration_accepted": acc["accepted"], "acceptance": acc, "score_type": "calibrated_probability" if acc["accepted"] else "model_score",
                 "deployed_scores": "calibrated" if acc["accepted"] else "raw", "decision_rule": rule_json(rule_deployed),
                 "raw_equivalent_threshold": (raw_equivalent_threshold(method, params, rule_deployed.threshold, eps) if acc["accepted"] else rule_deployed.threshold),
                 "tie_break": tie, "epsilon": eps, "selected_on": "operating role (development result, not independent)"}
    write(out / "threshold.json", threshold)
    metrics = {"rows": table, "probability": prob, "acceptance": acc, "near_threshold_counts": near, "operating_saturation": saturation(raw, eps)}
    write(out / "metrics.json", metrics)

    import pyarrow as pa
    import pyarrow.parquet as pq
    pred_rows = [{"recording_id": ids[i], "group_id": str(groups[i]), "label": int(y[i]), "component": str(comp[i]), "raw_score": float(raw[i]),
                  "selected_score": float(selected[i]), **{f"{c}_reference_score": float(mapped[c][i]) for c in mapped if c != method},
                  "deployed_score": float(deployed[i]), "decision": int(rule_deployed.decide(float(deployed[i])))} for i in range(len(ids))]
    pq.write_table(pa.Table.from_pylist(pred_rows), out / "operating_predictions.parquet")
    (out / "comparison.md").write_text(comparison_markdown(method, params, acc, table, prob, speakers, rule_deployed, threshold), encoding="utf-8")
    evidence = {"task": "R3-N1-WavLM operating evaluation", "status": "complete", "created_utc": now(), "recipe_sha256": recipe_hash(recipe),
                "fit_stage_complete_sha256": file_digest(fit_dir / "stage_complete.json"), "source_operating_predictions_sha256": recipe["inputs"]["predictions_sha256"][name],
                "selected_mapping": method, "calibration_accepted": acc["accepted"], "score_type": threshold["score_type"], "decision_rule": rule_json(rule_deployed),
                "fit_on_operating": False, "model_retrained": False, "test_data_read": False, "official_lock_changed": False,
                "selection_note": "Acceptance and the threshold were evaluated on the operating role: development results, not independent test performance.",
                "outputs": {p.name: file_digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "evidence.json"}, "environment": environment(recipe)}
    write(out / "evidence.json", evidence)
    return {"threshold": threshold, "metrics": metrics, "evidence": evidence}
