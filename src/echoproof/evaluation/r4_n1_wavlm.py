"""R4-N1 for the Protocol A WavLM-head: input audit, mapping choice and fit (calibration role only), and the single operating-role evaluation.

The decision rules are those of `configs/r4_n1_wavlm_calibration_a.json`, which restate the B WavLM procedure (`configs/r3_n1_wavlm_calibration_b.json`)
unchanged for Protocol A. The B modules are frozen by the GR3 lock and hard-code Protocol B, so this module imports their protocol-free helpers
(mapping fits, leave-one-speaker-out comparison, selection rule, acceptance rule, metrics) and restates only the A-bound parts: the recipe guard, the
role checks (five components), the manifest join (Protocol A split, ChatterboxTTS allowed, test refused), the per-component panels and the plateau report.

Stages (each reads only what the previous one froze):

  audit      verify the saved exports, join them to the locked A manifests, recompute the quoted summaries
  fit        read the calibration role only; compare identity, temperature and affine by leave-one-speaker-out; fit and freeze the selected mapping
  operating  open the operating role once; apply the acceptance rule; choose the threshold; write the diagnostics

Nothing here loads a model, trains, reads audio, touches the A test partition or modifies an input.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from echoproof.artifacts import file_digest
from echoproof.contracts import DecisionRule
from echoproof.evaluation.bootstrap import paired_group_bootstrap
from echoproof.evaluation.metrics import compute_metrics
from echoproof.evaluation.r3_n1_wavlm import (ORDER, ROLES, WavLMCalibrationError, acceptance, apply_mapping, arrays, brier, decision_metrics, decide,
                                              environment, fit_candidate, load_json, log_loss, loso_comparison, now, raw_equivalent_threshold,
                                              read_chunks, read_recordings, recipe_hash, rule_json, saturation, select_mapping, temperature_checks, write)
from echoproof.evaluation.r3_n1_wavlm_operating import (_brier_metric, _genuine_fpr_at_half, _log_loss_metric, _macro_f1_at_half, probability_metrics,
                                                        speaker_diagnostics)
from echoproof.evaluation.reliability import expected_calibration_error, reliability_bins
from echoproof.evaluation.threshold import select_threshold

COMPONENTS = ("genuine", "melo", "vits", "xtts", "chatterbox")
GENERATORS = COMPONENTS[1:]
B_RECIPE = "configs/r3_n1_wavlm_calibration_b.json"
ADDITIVE_KEYS = {"plateau_report"}


class A_CalibrationError(WavLMCalibrationError):
    pass


# ---- recipe -------------------------------------------------------------------------------------------------------------------------------------

def load_recipe(path, *, root=None) -> dict:
    """Load the A recipe. It must be team-agreed and its procedure blocks must equal the B recipe's, so A cannot drift from the accepted procedure."""
    recipe = load_json(path)
    if recipe.get("task") != "R4-N1-WavLM" or recipe.get("protocol") != "A" or recipe.get("schema_version") != 1:
        raise A_CalibrationError("not an R4-N1 WavLM Protocol A recipe")
    if not str(recipe.get("status", "")).startswith("team_agreed"):
        raise A_CalibrationError("the recipe has not been agreed by the team")
    if recipe.get("candidates") != list(ORDER) or recipe.get("epsilon") != 1e-7 or recipe["temperature"]["bounds"] != [0.01, 50.0]:
        raise A_CalibrationError("the candidate set, epsilon and the temperature bounds are fixed")
    if recipe.get("transformers_pin") != "5.22.0":
        raise A_CalibrationError("the transformers pin is 5.22.0")
    if list(recipe.get("components", [])) != list(COMPONENTS):
        raise A_CalibrationError("Protocol A has five components")
    root = Path(root) if root is not None else Path(path).resolve().parents[1]
    b_path = root / recipe["derived_from"]["file"]
    if b_path.is_file():
        b = load_json(b_path)
        for key in recipe["derived_from"]["equal_blocks"]:
            mine = recipe.get(key)
            if isinstance(mine, dict):
                mine = {k: v for k, v in mine.items() if k not in ADDITIVE_KEYS}     # reporting additions that do not change a rule
            if mine != b.get(key):
                raise A_CalibrationError(f"recipe block {key!r} differs from the B recipe it must equal")
    return recipe


# ---- checks -------------------------------------------------------------------------------------------------------------------------------------

def check_role_rows(rows, role: str, recipe: dict) -> list[str]:
    spec, problems = recipe["roles"][role], []
    if len(rows) != spec["recordings"]:
        problems.append(f"{role}: {len(rows)} recordings, expected {spec['recordings']}")
    ids = [r["recording_id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append(f"{role}: duplicate recording ids")
    if {r["validation_role"] for r in rows} != {role}:
        problems.append(f"{role}: wrong validation_role values")
    if len({r["group_id"] for r in rows}) != spec["speakers"]:
        problems.append(f"{role}: {len({r['group_id'] for r in rows})} speakers, expected {spec['speakers']}")
    counts = Counter(r["component"] for r in rows)
    if set(counts) != set(COMPONENTS) or any(counts[c] != spec["per_component"] for c in COMPONENTS):
        problems.append(f"{role}: component counts {dict(counts)} differ from {spec['per_component']} each")
    for r in rows:
        if r["label"] not in (0, 1) or r["label"] != int(r["component"] != "genuine"):
            problems.append(f"{role}: label/component mismatch for {r['recording_id']}")
            break
    scores = np.array([r["raw_score"] for r in rows], dtype=np.float64)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        problems.append(f"{role}: non-finite or out-of-range scores")
    if any(not isinstance(r["chunk_count"], int) or r["chunk_count"] < 1 for r in rows):
        problems.append(f"{role}: invalid chunk counts")
    return problems


def plateau(y, scores) -> dict:
    """Macro F1 over every score-induced partition (`score >= u` for each distinct u, as `select_threshold` evaluates them) and the size of the tied top.

    Near-ceiling models have wide plateaus; the threshold then rests on the tie-break, not on the data, and the plateau is reported beside it."""
    y, s = np.asarray(y, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    us = np.unique(s)
    f1 = np.empty(len(us))
    for i, u in enumerate(us):
        d = decision_metrics(y, (s >= u).astype(np.int64))
        f1[i] = d["macro_f1"]
    best = float(f1.max())
    tied = np.flatnonzero(f1 >= best - 1e-12)
    return {"partitions": int(len(us)), "best_macro_f1": best, "partitions_at_best": int(len(tied)), "lowest_threshold_at_best": float(us[tied[0]]),
            "highest_threshold_at_best": float(us[tied[-1]]), "partitions_within_0_001_of_best": int((f1 >= best - 1e-3).sum())}


def panel_metrics(y, comp, scores, rule: DecisionRule) -> dict:
    """Genuine against each generator, plus the full role, at one decision rule (informational)."""
    d = decide(rule, scores)
    out = {"full": decision_metrics(y, d)}
    for generator in GENERATORS:
        mask = np.isin(comp, ["genuine", generator])
        m = decision_metrics(y[mask], d[mask])
        m["roc_auc"] = compute_metrics(y[mask], np.asarray(scores)[mask], 0.5)["roc_auc"]
        out[f"genuine_vs_{generator}"] = m
    return out


# ---- audit ----------------------------------------------------------------------------------------------------------------------------------------

def audit(root, recipe: dict, run_dir, *, join_manifests: bool = True) -> dict:
    root, run_dir = Path(root), Path(run_dir)
    inputs = root / recipe["inputs"]["directory"]
    problems, notes, checks = [], [], {}
    eps = recipe["epsilon"]

    def check(name, ok, detail=None):
        checks[name] = bool(ok)
        if not ok:
            problems.append(f"{name}: {detail if detail is not None else 'failed'}")

    evidence = load_json(inputs / "evidence.json")
    selection = load_json(inputs / "selection.json")
    check("selection_json_hash", file_digest(inputs / "selection.json") == recipe["inputs"]["selection_json_sha256"])
    check("evidence_json_hash", file_digest(inputs / "evidence.json") == recipe["inputs"]["evidence_json_sha256"])
    check("recipe_hashes_equal_evidence_hashes", evidence["output_sha256"] == recipe["inputs"]["predictions_sha256"])
    for name, expected in recipe["inputs"]["predictions_sha256"].items():
        check(f"hash {name}", file_digest(inputs / name) == expected)
    check("selection_is_mean_std_at_the_recorded_epoch", selection["selected"] == recipe["model"]["selected"] and selection["selected_epoch"] == recipe["model"]["epoch"])
    check("selection_protocol_is_A", selection.get("protocol") == "A")
    check("checkpoint_hash", selection["checkpoint_sha256"][selection["selected"]] == recipe["model"]["checkpoint_sha256"])
    check("split_hash", selection["split_sha256"] == recipe["model"]["split_sha256"])

    bundle_dir = root / recipe["model"]["source_bundle"]
    check("source_bundle_json_hash", file_digest(bundle_dir / "bundle.json") == recipe["model"]["source_bundle_json_sha256"])
    manifest = load_json(bundle_dir / "bundle.json")
    check("source_bundle_components", all(file_digest(bundle_dir / k) == v["sha256"] for k, v in manifest["components"].items()))
    head = load_json(bundle_dir / "model/wavlm_head.json")
    check("bundle_checkpoint_identity", head["source_checkpoint_sha256"] == recipe["model"]["checkpoint_sha256"] and head["pooling"] == "mean_std")
    check("bundle_split_identity", head["split_sha256"] == recipe["model"]["split_sha256"])
    check("bundle_is_uncalibrated_A", manifest.get("extra", {}).get("stage") == "development_uncalibrated" and manifest["model_version"] == recipe["model"]["model_version"])

    by_role, readable = {}, True
    for role in ROLES:
        try:
            rows = read_recordings(inputs, role)
        except Exception as exc:                                       # a corrupt or altered file is a recorded failure, not a crash
            check(f"read recordings {role}", False, f"{type(exc).__name__}: {exc}")
            readable = False
            continue
        by_role[role] = rows
        for p in check_role_rows(rows, role, recipe):
            check(f"rows {p}", False)
    if not readable:
        return _finish_audit(run_dir, recipe, checks, problems, notes, {}, {}, {}, {})
    speakers = {role: {r["group_id"] for r in rows} for role, rows in by_role.items()}
    check("speakers_disjoint_across_roles", len(set().union(*speakers.values())) == sum(len(v) for v in speakers.values()))
    check("recording_ids_disjoint_across_roles", len({r["recording_id"] for rows in by_role.values() for r in rows}) == sum(len(v) for v in by_role.values()))

    chunk_report = {}
    for role in ROLES:
        try:
            chunks = read_chunks(inputs, role)
        except Exception as exc:
            check(f"read chunks {role}", False, f"{type(exc).__name__}: {exc}")
            continue
        by_recording = defaultdict(list)
        for c in chunks:
            by_recording[c["recording_id"]].append(c["tts_probability"])
        ids = [c["chunk_id"] for c in chunks]
        worst = max((abs(float(np.mean(by_recording[r["recording_id"]])) - r["raw_score"]) for r in by_role[role]), default=0.0)
        check(f"chunks {role} count", len(chunks) == recipe["roles"][role]["chunks"], f"{len(chunks)} vs {recipe['roles'][role]['chunks']}")
        check(f"chunks {role} unique", len(set(ids)) == len(ids))
        check(f"chunks {role} counts per recording", all(len(by_recording[r["recording_id"]]) == r["chunk_count"] for r in by_role[role]))
        check(f"chunks {role} recording score is the mean of its chunk probabilities", worst < 1e-9, f"max difference {worst}")
        chunk_report[role] = {"chunks": len(chunks), "max_abs_mean_difference": worst}

    if problems:
        return _finish_audit(run_dir, recipe, checks, problems, notes, {}, {}, {}, chunk_report)
    ys, ss, *_ = arrays(by_role["selection"])
    selection_f1 = compute_metrics(ys, ss, 0.5)["macro_f1"]
    check("selection_macro_f1_reproduces", abs(selection_f1 - selection["selection_macro_f1"][selection["selected"]]) < 1e-12, str(selection_f1))

    summary, sat, errors = {}, {}, {}
    for role in ("calibration", "operating"):
        y, s, *_ = arrays(by_role[role])
        m = compute_metrics(y, s, 0.5)
        summary[role] = {"macro_f1": m["macro_f1"], "genuine_fpr": m["fpr"], "eer": m["eer"], "roc_auc": m["roc_auc"]}
        expected = recipe["expected_raw_summary_at_0_5"][role]
        d = recipe["expected_raw_summary_at_0_5"]["decimals"]
        check(f"summary {role} equals the recorded figures", all(round(summary[role][k], d) == expected[k] for k in expected), str(summary[role]))
    for role in ROLES:
        y, s, g, comp, ids = arrays(by_role[role])
        sat[role] = saturation(s, eps)
        check(f"saturation {role} equals the recorded counts", sat[role] == recipe["expected_saturation"][role], str(sat[role]))
        errors[role] = [ids[i] for i in range(len(ids)) if (y[i] == 0 and s[i] >= 1.0 - eps) or (y[i] == 1 and s[i] <= eps)]
    if join_manifests:
        from echoproof.data import cohort, splits
        cohort_rows = {r["recording_id"]: r for r in cohort.read_table(str(root / recipe["split"]["cohort"]))}
        assignments = {a.recording_id: a for a in splits.load_assignments(root / recipe["split"]["file"], sealed=False)}   # test rows are withheld
        bad = Counter()
        for role, rows in by_role.items():
            for r in rows:
                a, c = assignments.get(r["recording_id"]), cohort_rows.get(r["recording_id"])
                if a is None or c is None:
                    bad["unknown recording"] += 1
                    continue
                if a.partition != "validation" or a.validation_role != role:
                    bad["role or partition"] += 1
                if a.group_id != r["group_id"]:
                    bad["group"] += 1
                if int(c["label"]) != r["label"] or c["component"] != r["component"]:
                    bad["label or component"] += 1
        check("exports match the locked A cohort and split manifests", not bad, dict(bad))
    else:
        notes.append("manifest join skipped (test mode)")
    return _finish_audit(run_dir, recipe, checks, problems, notes, summary, sat, errors, chunk_report)


def _finish_audit(run_dir, recipe, checks, problems, notes, summary, sat, errors, chunk_report) -> dict:
    result = {"task": "R4-N1-WavLM input audit", "created_utc": now(), "passed": not problems, "checks": checks, "problems": problems, "notes": notes,
              "raw_summary_at_0_5": summary, "saturation": sat, "saturated_recordings_that_are_errors": errors, "chunk_checks": chunk_report,
              "recipe_sha256": recipe_hash(recipe), "environment": environment(recipe), "model_scores_used": False, "operating_role_used_for_fitting": False,
              "test_partition_read": False}
    write(Path(run_dir) / "input_audit.json", result)
    if problems:
        raise A_CalibrationError("input audit failed: " + "; ".join(problems[:5]))
    return result


# ---- fit stage (calibration role only) ----------------------------------------------------------------------------------------------------------

def fit_stage(root, recipe: dict, run_dir) -> dict:
    root, run_dir = Path(root), Path(run_dir)
    audit_path = run_dir / "input_audit.json"
    audit_record = load_json(audit_path)
    if not audit_record.get("passed") or audit_record["recipe_sha256"] != recipe_hash(recipe):
        raise A_CalibrationError("the input audit did not pass for this recipe")
    inputs = root / recipe["inputs"]["directory"]
    name = "predictions_recordings_calibration.parquet"
    if file_digest(inputs / name) != recipe["inputs"]["predictions_sha256"][name]:
        raise A_CalibrationError("calibration predictions changed since the audit")
    rows = read_recordings(inputs, "calibration")                      # the operating file is never opened in this stage
    y, s, g, comp, ids = arrays(rows)
    eps = recipe["epsilon"]
    out = run_dir / "fit"
    out.mkdir(exist_ok=False)

    comparison = loso_comparison(y, s, g, recipe)
    final_fits = {c: fit_candidate(c, y, s, recipe) for c in ORDER}
    failures = dict(comparison["failures"])
    for c in ("temperature", "affine"):
        if not final_fits[c]["ok"]:
            failures.setdefault(c, f"final fit failed: {final_fits[c]['failure_reason']}")
    temp_check = None
    if final_fits["temperature"]["params"]:
        temp_check = temperature_checks(y, s, final_fits["temperature"]["params"]["temperature"], recipe["temperature"]["bounds"], eps)
        if not temp_check["agrees"]:
            failures.setdefault("temperature", "the shared and the independent temperature fit disagree")
    speaker_losses = {c: comparison["per_speaker_log_loss"][c] for c in ORDER if c not in failures}
    speaker_losses["identity"] = comparison["per_speaker_log_loss"]["identity"]
    rule = select_mapping(speaker_losses, failures)
    selected = rule["selected"]
    comparison.update({"selection": rule, "order": list(ORDER), "failures_after_final_fit": failures})
    write(out / "loso_comparison.json", comparison)

    candidate_fits = {c: {"ok": final_fits[c]["ok"], "failure_reason": final_fits[c]["failure_reason"], "params": final_fits[c]["params"],
                          "diagnostics": final_fits[c].get("diagnostics"), "warnings": final_fits[c].get("warnings")} for c in ORDER}
    in_sample = {}
    for c in ORDER:
        if c not in failures:
            p = apply_mapping(c, final_fits[c]["params"], s, eps)
            in_sample[c] = {"log_loss": log_loss(y, p), "brier": brier(y, p), "ece_equal_width_10": expected_calibration_error(y.tolist(), p)}
    calibration = {"task": "R4-N1-WavLM mapping fit", "created_utc": now(), "selected_mapping": selected, "selected_parameters": final_fits[selected]["params"],
                   "selection": rule, "candidate_fits": candidate_fits, "temperature_reference": {"fit": final_fits["temperature"]["params"], "checks": temp_check},
                   "in_sample_calibration_role": in_sample, "epsilon": eps, "fit_role": "calibration", "fit_recordings": int(len(y)),
                   "fit_speakers": int(len(set(g.tolist()))), "saturation_in_fit": saturation(s, eps),
                   "calibration_errors_at_0_5": int(((s >= 0.5) != (y == 1)).sum()),
                   "operating_predictions_read": False, "test_data_read": False, "model_retrained": False, "threshold_selected": False,
                   "source_calibration_predictions_sha256": recipe["inputs"]["predictions_sha256"][name], "recipe_sha256": recipe_hash(recipe),
                   "input_audit_sha256": file_digest(audit_path), "environment": environment(recipe)}
    write(out / "calibration.json", calibration)
    complete = {"task": "R4-N1-WavLM fit stage complete", "created_utc": now(), "selected_mapping": selected,
                "files": {n: file_digest(out / n) for n in ("loso_comparison.json", "calibration.json")}, "recipe_sha256": recipe_hash(recipe),
                "note": "The mapping is frozen. The operating role has not been read."}
    write(out / "stage_complete.json", complete)
    return calibration


# ---- operating stage ----------------------------------------------------------------------------------------------------------------------------

def _fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def comparison_markdown(method, params, acc, table, prob, speakers, rule, threshold, plateaus) -> str:
    lines = ["# WavLM-A R4-N1 operating comparison", "",
             f"Selected mapping: **{method}** {params}. Calibration accepted: **{acc['accepted']}** ({acc['reason']}).", "",
             f"Deployed rule: `{rule.kind}` threshold {rule.threshold}; score type `{threshold['score_type']}`. The threshold was chosen on the operating role: "
             "a development result, not independent test performance. Protocol A has no unseen generator.", "",
             "## Decision rows (full operating role)", "",
             "| Row | Rule | Macro F1 | Genuine FPR | TTS recall | TP | FP | TN | FN |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in table.items():
        f = v["full"]
        lines.append(f"| {k} | {v['rule']['kind']} {_fmt(v['rule']['threshold'])} | {_fmt(f['macro_f1'])} | {_fmt(f['genuine_fpr'])} | {_fmt(f['tts_recall'])} "
                     f"| {f['tp']} | {f['fp']} | {f['tn']} | {f['fn']} |")
    lines += ["", "## Threshold plateau (partitions that reach the maximum operating macro F1)", "", "| Scores | Partitions | Best macro F1 | At best | Lowest threshold at best | Highest threshold at best |",
              "|---|---:|---:|---:|---:|---:|"]
    for k, v in plateaus.items():
        lines.append(f"| {k} | {v['partitions']} | {_fmt(v['best_macro_f1'])} | {v['partitions_at_best']} | {_fmt(v['lowest_threshold_at_best'], 6)} | {_fmt(v['highest_threshold_at_best'], 6)} |")
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
        raise A_CalibrationError("the frozen fit changed or does not belong to this recipe")
    calibration = load_json(fit_dir / "calibration.json")
    if calibration["operating_predictions_read"]:
        raise A_CalibrationError("the fit stage read the operating role")
    out = run_dir / "operating"
    out.mkdir(exist_ok=False)                      # a second run, or a crashed one, is refused: the role is opened once
    write(out / "opened.json", {"opened_utc": now(), "fit_stage_complete_sha256": file_digest(fit_dir / "stage_complete.json"),
                                "mapping_frozen_before_opening": calibration["selected_mapping"]})
    eps, bins = recipe["epsilon"], recipe["reliability"]["bins"]
    inputs = root / recipe["inputs"]["directory"]
    name = "predictions_recordings_operating.parquet"
    if file_digest(inputs / name) != recipe["inputs"]["predictions_sha256"][name]:
        raise A_CalibrationError("operating predictions differ from the frozen hash")
    rows = read_recordings(inputs, "operating")
    problems = check_role_rows(rows, "operating", recipe)
    if problems:
        raise A_CalibrationError("; ".join(problems))
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
    table = {k: {"rule": rule_json(r), **panel_metrics(y, comp, s, r)} for k, (s, r) in rows_ab.items()}
    plateaus = {"raw": plateau(y, raw)}
    if method != "identity":
        plateaus["selected"] = plateau(y, selected)

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
        ax.set(xlabel="Mean score", ylabel="Observed TTS fraction", title="WavLM-A operating reliability (10 bins)", xlim=(0, 1), ylim=(0, 1))
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
        raise A_CalibrationError("selection predictions differ from the frozen hash")
    sy, sraw, sg, scomp, sids = arrays(read_recordings(inputs, "selection"))
    sdeployed = np.array(apply_mapping(method, params, sraw, eps)) if (acc["accepted"] and method != "identity") else sraw
    write(out / "selection_role_diagnostic.json", {"role": "selection", "note": "The frozen deployed policy applied to the selection role for description; nothing is refit.",
                                                   "deployed": panel_metrics(sy, scomp, sdeployed, rule_deployed), "raw_at_0_5": panel_metrics(sy, scomp, sraw, half)})

    threshold = {"task": "R4-N1-WavLM operating policy", "created_utc": now(), "selected_mapping": method, "selected_parameters": params,
                 "calibration_accepted": acc["accepted"], "acceptance": acc, "score_type": "calibrated_probability" if acc["accepted"] else "model_score",
                 "deployed_scores": "calibrated" if acc["accepted"] else "raw", "decision_rule": rule_json(rule_deployed),
                 "raw_equivalent_threshold": (raw_equivalent_threshold(method, params, rule_deployed.threshold, eps) if acc["accepted"] else rule_deployed.threshold),
                 "plateau": plateaus["selected" if acc["accepted"] and method != "identity" else "raw"],
                 "tie_break": tie, "epsilon": eps, "selected_on": "operating role (development result, not independent)"}
    write(out / "threshold.json", threshold)
    write(out / "metrics.json", {"rows": table, "probability": prob, "acceptance": acc, "plateaus": plateaus, "near_threshold_counts": near,
                                 "operating_saturation": saturation(raw, eps)})

    import pyarrow as pa
    import pyarrow.parquet as pq
    pred_rows = [{"recording_id": ids[i], "group_id": str(groups[i]), "label": int(y[i]), "component": str(comp[i]), "raw_score": float(raw[i]),
                  "selected_score": float(selected[i]), **{f"{c}_reference_score": float(mapped[c][i]) for c in mapped if c != method},
                  "deployed_score": float(deployed[i]), "decision": int(rule_deployed.decide(float(deployed[i])))} for i in range(len(ids))]
    pq.write_table(pa.Table.from_pylist(pred_rows), out / "operating_predictions.parquet")
    (out / "comparison.md").write_text(comparison_markdown(method, params, acc, table, prob, speakers, rule_deployed, threshold, plateaus), encoding="utf-8")
    evidence = {"task": "R4-N1-WavLM operating evaluation", "status": "complete", "created_utc": now(), "recipe_sha256": recipe_hash(recipe),
                "fit_stage_complete_sha256": file_digest(fit_dir / "stage_complete.json"), "source_operating_predictions_sha256": recipe["inputs"]["predictions_sha256"][name],
                "selected_mapping": method, "calibration_accepted": acc["accepted"], "score_type": threshold["score_type"], "decision_rule": rule_json(rule_deployed),
                "fit_on_operating": False, "model_retrained": False, "test_data_read": False, "official_lock_changed": False,
                "selection_note": "Acceptance and the threshold were evaluated on the operating role: development results, not independent test performance.",
                "outputs": {p.name: file_digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "evidence.json"}, "environment": environment(recipe)}
    write(out / "evidence.json", evidence)
    return {"threshold": threshold, "metrics": {"rows": table, "probability": prob, "plateaus": plateaus}, "evidence": evidence}
