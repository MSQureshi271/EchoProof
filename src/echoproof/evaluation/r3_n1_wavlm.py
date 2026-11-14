"""R3-N1 for the selected WavLM head: input audit, mapping choice and fit (calibration role only), and the single operating-role evaluation.

Every decision rule is the one frozen in `configs/r3_n1_wavlm_calibration_b.json` (see WAVLM_CALIBRATION.md). The stages are separate on purpose:

  audit      verify the saved exports, join them to the locked manifests, recompute the quoted summaries
  fit        read the calibration role only; compare identity, temperature and affine by leave-one-speaker-out; fit and freeze the selected mapping
  operating  open the operating role once; apply the acceptance rule; choose the threshold; write the diagnostics

Nothing here loads a model, trains, reads audio, touches the test partition or modifies an input.
"""
from __future__ import annotations

import csv
import importlib.metadata as metadata
import json
import math
import platform
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from echoproof.artifacts import digest, file_digest
from echoproof.contracts import DecisionRule
from echoproof.evaluation.affine_calibration import apply_affine, fit_affine
from echoproof.evaluation.calibrate import apply_temperature, fit_temperature
from echoproof.evaluation.metrics import compute_metrics
from echoproof.evaluation.reliability import expected_calibration_error, reliability_bins
from echoproof.evaluation.threshold import select_threshold

ROLES = ("selection", "calibration", "operating")
ORDER = ("identity", "temperature", "affine")
COMPONENTS = ("genuine", "melo", "vits", "xtts")
RECORDING_COLUMNS = ["recording_id", "group_id", "label", "component", "validation_role", "raw_score", "chunk_count"]
CHUNK_COLUMNS = ["chunk_id", "recording_id", "group_id", "label", "component", "validation_role", "tts_probability"]
TEMPERATURE_AGREEMENT = 1e-4
OBJECTIVE_AGREEMENT = 1e-10


class WavLMCalibrationError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not serialisable: {type(value).__name__}")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---- recipe -----------------------------------------------------------------------------------------------------------------------------------

def load_recipe(path) -> dict:
    recipe = load_json(path)
    if recipe.get("task") != "R3-N1-WavLM" or recipe.get("protocol") != "B" or recipe.get("schema_version") != 1:
        raise WavLMCalibrationError("not an R3-N1 WavLM Protocol B recipe")
    if not str(recipe.get("status", "")).startswith("owner_confirmed"):
        raise WavLMCalibrationError("the recipe has not been confirmed by the owner")
    if recipe.get("candidates") != list(ORDER):
        raise WavLMCalibrationError("the candidate set is fixed: identity, temperature, affine")
    if recipe.get("epsilon") != 1e-7 or recipe["temperature"]["bounds"] != [0.01, 50.0]:
        raise WavLMCalibrationError("epsilon and the temperature bounds are fixed by the plan")
    if recipe.get("transformers_pin") != "5.22.0":
        raise WavLMCalibrationError("the transformers pin is 5.22.0")
    return recipe


def recipe_hash(recipe: dict) -> str:
    return digest(recipe)


def environment(recipe=None) -> dict:
    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None
    env = {"python": platform.python_version(), "platform": platform.platform(), "numpy": version("numpy"), "scipy": version("scipy"),
           "pyarrow": version("pyarrow"), "matplotlib": version("matplotlib"), "transformers": version("transformers"), "torch": version("torch")}
    pin = (recipe or {}).get("transformers_pin")
    env["transformers_pin"] = pin
    env["transformers_matches_pin"] = None if env["transformers"] is None else env["transformers"] == pin
    return env


# ---- data helpers -------------------------------------------------------------------------------------------------------------------------------

def read_table(path):
    import pyarrow.parquet as pq
    return pq.read_table(path)


def read_recordings(directory, role: str) -> list[dict]:
    table = read_table(Path(directory) / f"predictions_recordings_{role}.parquet")
    if table.column_names != RECORDING_COLUMNS:
        raise WavLMCalibrationError(f"{role}: unexpected recording columns {table.column_names}")
    return table.to_pylist()


def read_chunks(directory, role: str) -> list[dict]:
    table = read_table(Path(directory) / f"predictions_chunks_{role}.parquet")
    if table.column_names != CHUNK_COLUMNS:
        raise WavLMCalibrationError(f"{role}: unexpected chunk columns {table.column_names}")
    return table.to_pylist()


def arrays(rows):
    return (np.array([r["label"] for r in rows], dtype=np.int64), np.array([r["raw_score"] for r in rows], dtype=np.float64),
            np.array([r["group_id"] for r in rows]), np.array([r["component"] for r in rows]), [r["recording_id"] for r in rows])


def check_role_rows(rows, role: str, recipe: dict) -> list[str]:
    """Structural checks on one role's recording rows; returns the list of problems (empty if clean)."""
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


def saturation(scores: np.ndarray, epsilon: float) -> dict:
    high, low = scores >= 1.0 - epsilon, scores <= epsilon
    return {"high": int(high.sum()), "exactly_one": int((scores == 1.0).sum()), "low": int(low.sum()),
            "distinct_before": int(len(np.unique(scores))), "distinct_after": int(len(np.unique(np.clip(scores, epsilon, 1.0 - epsilon))))}


def log_loss(y, p, epsilon=1e-7) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), epsilon, 1.0 - epsilon)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def decision_metrics(labels, decisions) -> dict:
    """Confusion-based metrics for any 0/1 decisions, including the all-negative rule."""
    y, d = np.asarray(labels, dtype=np.int64), np.asarray(decisions, dtype=np.int64)
    tp, fp = int(((d == 1) & (y == 1)).sum()), int(((d == 1) & (y == 0)).sum())
    tn, fn = int(((d == 0) & (y == 0)).sum()), int(((d == 0) & (y == 1)).sum())
    f1_tts = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_gen = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "macro_f1": (f1_tts + f1_gen) / 2.0, "genuine_fpr": fp / (fp + tn) if fp + tn else 0.0,
            "tts_recall": tp / (tp + fn) if tp + fn else 0.0, "accuracy": (tp + tn) / len(y)}


def decide(rule: DecisionRule, scores) -> np.ndarray:
    return np.array([rule.decide(float(s)) for s in scores], dtype=np.int64)


# ---- mappings -----------------------------------------------------------------------------------------------------------------------------------

def apply_mapping(method: str, params: dict, scores, epsilon: float) -> list[float]:
    if method == "identity":
        return [float(s) for s in scores]
    if method == "temperature":
        return apply_temperature(list(scores), params["temperature"], epsilon=epsilon)
    if method == "affine":
        return apply_affine(list(scores), params["a"], params["b"], epsilon=epsilon)
    raise WavLMCalibrationError(f"unknown mapping: {method}")


def raw_equivalent_threshold(method: str, params: dict, tau: float, epsilon: float):
    """Raw-score threshold whose decisions equal `score >= tau` on the mapped scale (None when it is not meaningful)."""
    if method == "identity" or tau is None or not 0.0 < tau < 1.0:
        return None
    z = math.log(tau / (1.0 - tau))
    value = z * params["temperature"] if method == "temperature" else (z - params["b"]) / params["a"]
    return float(1.0 / (1.0 + math.exp(-value))) if abs(value) < 700 else None


def temperature_checks(y, s, temperature: float, bounds, epsilon: float) -> dict:
    """Independent re-fit with a stable objective, objective values at the interesting points, boundary status and recorded warnings."""
    from scipy.optimize import minimize_scalar
    y, s = np.asarray(y, dtype=np.int64), np.asarray(s, dtype=np.float64)
    z = np.log(np.clip(s, epsilon, 1 - epsilon) / (1 - np.clip(s, epsilon, 1 - epsilon)))
    m = 1.0 - 2.0 * y

    def objective(t):
        return float(np.mean(np.logaddexp(0.0, m * z / t)))
    independent = minimize_scalar(objective, bounds=tuple(bounds), method="bounded")
    return {"independent_temperature": float(independent.x), "independent_success": bool(independent.success), "difference": abs(float(independent.x) - temperature),
            "objective_at_fit": objective(temperature), "objective_independent": float(independent.fun),
            "objective_difference": abs(objective(temperature) - float(independent.fun)), "objective_at_identity": objective(1.0),
            "objective_at_lower_bound": objective(bounds[0]), "objective_at_upper_bound": objective(bounds[1]),
            "inside_bounds": bool(bounds[0] + 1e-6 < temperature < bounds[1] - 1e-6),
            "agrees": bool(abs(float(independent.x) - temperature) <= TEMPERATURE_AGREEMENT and abs(objective(temperature) - float(independent.fun)) <= OBJECTIVE_AGREEMENT)}


def fit_candidate(method: str, y, s, recipe: dict) -> dict:
    """Fit one mapping. Returns {'ok', 'failure_reason', 'params', ...}; never raises for an optimiser failure."""
    eps = recipe["epsilon"]
    if method == "identity":
        return {"ok": True, "failure_reason": None, "params": {}}
    if method == "temperature":
        bounds = recipe["temperature"]["bounds"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(all="ignore"):
                try:
                    t = float(fit_temperature(y.tolist(), s.tolist(), epsilon=eps, bounds=tuple(bounds)))
                except Exception as exc:                                            # reported, never silent
                    return {"ok": False, "failure_reason": f"fit raised {type(exc).__name__}: {exc}", "params": {}}
        inside = bounds[0] + recipe["temperature"]["bound_tolerance"] < t < bounds[1] - recipe["temperature"]["bound_tolerance"]
        return {"ok": bool(math.isfinite(t) and inside), "failure_reason": None if math.isfinite(t) and inside else "temperature at a bound or non-finite",
                "params": {"temperature": t}, "warnings": [str(w.message) for w in caught]}
    if method == "affine":
        cfg = recipe["affine"]
        fit = fit_affine(y.tolist(), s.tolist(), epsilon=eps, u_bounds=tuple(cfg["u_bounds"]), b_bounds=tuple(cfg["b_bounds"]), reg=cfg["regularisation"],
                         start=tuple(cfg["start"]))
        return {"ok": fit["ok"], "failure_reason": fit["failure_reason"], "params": {"a": fit["a"], "b": fit["b"], "u": fit["u"]}, "diagnostics": fit}
    raise WavLMCalibrationError(f"unknown mapping: {method}")


# ---- selection of the mapping (calibration role only) --------------------------------------------------------------------------------------------

def per_speaker(y, p, groups) -> dict:
    out = {}
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        out[str(g)] = {"log_loss": log_loss(y[mask], p[mask]), "brier": brier(y[mask], p[mask]), "recordings": int(mask.sum())}
    return out


def select_mapping(speaker_losses: dict, failures: dict) -> dict:
    """One-standard-error rule, simplest first. `speaker_losses[candidate]` is the list of per-held-out-speaker log losses."""
    eligible = {c: v for c, v in speaker_losses.items() if c not in failures}
    if "identity" not in eligible:
        raise WavLMCalibrationError("identity must always be eligible")
    means = {c: float(np.mean(v)) for c, v in eligible.items()}
    best = min(means, key=lambda c: (means[c], ORDER.index(c)))
    losses = np.asarray(eligible[best], dtype=np.float64)
    se = float(np.std(losses, ddof=1) / math.sqrt(len(losses))) if len(losses) > 1 else 0.0
    limit = means[best] + se
    chosen = next(c for c in ORDER if c in means and means[c] <= limit)
    return {"selected": chosen, "best": best, "means": means, "standard_error_of_best": se, "limit": limit, "excluded": dict(failures)}


def loso_comparison(y, s, groups, recipe: dict) -> dict:
    """Leave-one-speaker-out comparison of identity, temperature and affine on the calibration role."""
    eps, speakers = recipe["epsilon"], sorted(set(groups.tolist()))
    oof = {c: np.empty(len(y)) for c in ORDER}
    folds, failures = [], {}
    for g in speakers:
        held = groups == g
        record = {"held_out_speaker": str(g), "recordings": int(held.sum())}
        for c in ORDER:
            fit = fit_candidate(c, y[~held], s[~held], recipe)
            record[c] = {"ok": fit["ok"], "failure_reason": fit["failure_reason"], "params": fit["params"]}
            if not fit["ok"]:
                failures.setdefault(c, f"fit failed on the fold holding out {g}: {fit['failure_reason']}")
                oof[c][held] = np.nan
                continue
            oof[c][held] = apply_mapping(c, fit["params"], s[held], eps)
            record[c]["log_loss"] = log_loss(y[held], oof[c][held])
            record[c]["brier"] = brier(y[held], oof[c][held])
        folds.append(record)
    per = {c: ([f[c]["log_loss"] for f in folds] if c not in failures else None) for c in ORDER}
    pooled = {c: ({"log_loss": log_loss(y, oof[c]), "brier": brier(y, oof[c])} if c not in failures else None) for c in ORDER}
    return {"folds": folds, "per_speaker_log_loss": per, "pooled_out_of_fold": pooled, "failures": failures, "speakers": [str(g) for g in speakers]}


# ---- audit stage -----------------------------------------------------------------------------------------------------------------------------------

def audit(root, recipe: dict, run_dir, *, join_manifests: bool = True) -> dict:
    root, run_dir = Path(root), Path(run_dir)
    inputs = root / recipe["inputs"]["directory"]
    problems, notes = [], []
    eps = recipe["epsilon"]
    checks = {}

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
    check("selection_is_mean_std_epoch_14", selection["selected"] == recipe["model"]["selected"] and selection["selected_epoch"] == recipe["model"]["epoch"])
    check("checkpoint_hash", selection["checkpoint_sha256"][selection["selected"]] == recipe["model"]["checkpoint_sha256"])
    check("split_hash", selection["split_sha256"] == recipe["model"]["split_sha256"])

    bundle_dir = root / recipe["model"]["source_bundle"]
    check("source_bundle_json_hash", file_digest(bundle_dir / "bundle.json") == recipe["model"]["source_bundle_json_sha256"])
    manifest = load_json(bundle_dir / "bundle.json")
    check("source_bundle_components", all(file_digest(bundle_dir / k) == v["sha256"] for k, v in manifest["components"].items()))
    head = load_json(bundle_dir / "model/wavlm_head.json")
    check("bundle_checkpoint_identity", head["source_checkpoint_sha256"] == recipe["model"]["checkpoint_sha256"] and head["pooling"] == "mean_std")
    check("bundle_split_identity", head["split_sha256"] == recipe["model"]["split_sha256"])
    check("bundle_is_uncalibrated", manifest.get("extra", {}).get("stage") == "development_uncalibrated" and manifest["model_version"] == recipe["model"]["model_version"])

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
    selection_rows = by_role["selection"]
    ys, ss, *_ = arrays(selection_rows)
    selection_f1 = compute_metrics(ys, ss, 0.5)["macro_f1"]
    check("selection_macro_f1_reproduces", abs(selection_f1 - selection["selection_macro_f1"][selection["selected"]]) < 1e-12, str(selection_f1))

    summary, sat = {}, {}
    for role in ("calibration", "operating"):
        y, s, *_ = arrays(by_role[role])
        m = compute_metrics(y, s, 0.5)
        summary[role] = {"macro_f1": m["macro_f1"], "genuine_fpr": m["fpr"], "eer": m["eer"], "roc_auc": m["roc_auc"]}
        expected = recipe["expected_raw_summary_at_0_5"][role]
        d = recipe["expected_raw_summary_at_0_5"]["decimals"]
        check(f"summary {role} equals the plan's figures", all(round(summary[role][k], d) == expected[k] for k in expected), str(summary[role]))
    errors = {}
    for role in ROLES:
        y, s, g, comp, ids = arrays(by_role[role])
        sat[role] = saturation(s, eps)
        check(f"saturation {role} equals the plan's counts", sat[role] == recipe["expected_saturation"][role], str(sat[role]))
        wrong = [ids[i] for i in range(len(ids)) if (y[i] == 0 and s[i] >= 1.0 - eps) or (y[i] == 1 and s[i] <= eps)]
        errors[role] = wrong
    if join_manifests:
        from echoproof.data import cohort, splits
        cohort_rows = {r["recording_id"]: r for r in cohort.read_table(str(root / "data/manifests/cohort_v1.parquet"))}
        assignments = {a.recording_id: a for a in splits.load_assignments(root / "data/splits/protocol_B_v1.parquet", sealed=False)}
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
                if c["component"] == "chatterbox":
                    bad["chatterbox"] += 1
        check("exports match the locked cohort and split manifests", not bad, dict(bad))
    else:
        notes.append("manifest join skipped (test mode)")
    return _finish_audit(run_dir, recipe, checks, problems, notes, summary, sat, errors, chunk_report)


def _finish_audit(run_dir, recipe, checks, problems, notes, summary, sat, errors, chunk_report) -> dict:
    result = {"task": "R3-N1-WavLM input audit", "created_utc": now(), "passed": not problems, "checks": checks, "problems": problems, "notes": notes,
              "raw_summary_at_0_5": summary, "saturation": sat, "saturated_recordings_that_are_errors": errors, "chunk_checks": chunk_report,
              "recipe_sha256": recipe_hash(recipe), "environment": environment(recipe), "model_scores_used": False, "operating_role_used_for_fitting": False,
              "test_partition_read": False, "chatterbox_read": False}
    write(Path(run_dir) / "input_audit.json", result)
    if problems:
        raise WavLMCalibrationError("input audit failed: " + "; ".join(problems[:5]))
    return result


# ---- fit stage (calibration role only) -------------------------------------------------------------------------------------------------------------

def fit_stage(root, recipe: dict, run_dir) -> dict:
    root, run_dir = Path(root), Path(run_dir)
    audit_path = run_dir / "input_audit.json"
    audit_record = load_json(audit_path)
    if not audit_record.get("passed") or audit_record["recipe_sha256"] != recipe_hash(recipe):
        raise WavLMCalibrationError("the input audit did not pass for this recipe")
    inputs = root / recipe["inputs"]["directory"]
    name = "predictions_recordings_calibration.parquet"
    if file_digest(inputs / name) != recipe["inputs"]["predictions_sha256"][name]:
        raise WavLMCalibrationError("calibration predictions changed since the audit")
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
    calibration = {"task": "R3-N1-WavLM mapping fit", "created_utc": now(), "selected_mapping": selected, "selected_parameters": final_fits[selected]["params"],
                   "selection": rule, "candidate_fits": candidate_fits, "temperature_reference": {"fit": final_fits["temperature"]["params"], "checks": temp_check},
                   "in_sample_calibration_role": in_sample, "epsilon": eps, "fit_role": "calibration", "fit_recordings": int(len(y)),
                   "fit_speakers": int(len(set(g.tolist()))), "saturation_in_fit": saturation(s, eps),
                   "operating_predictions_read": False, "test_data_read": False, "model_retrained": False, "threshold_selected": False,
                   "source_calibration_predictions_sha256": recipe["inputs"]["predictions_sha256"][name], "recipe_sha256": recipe_hash(recipe),
                   "input_audit_sha256": file_digest(audit_path), "environment": environment(recipe)}
    write(out / "calibration.json", calibration)
    complete = {"task": "R3-N1-WavLM fit stage complete", "created_utc": now(), "selected_mapping": selected,
                "files": {n: file_digest(out / n) for n in ("loso_comparison.json", "calibration.json")}, "recipe_sha256": recipe_hash(recipe),
                "note": "The mapping is frozen. The operating role has not been read."}
    write(out / "stage_complete.json", complete)
    return calibration


# ---- acceptance rule -------------------------------------------------------------------------------------------------------------------------------

def acceptance(method: str, fit_ok: bool, y, raw, calibrated) -> dict:
    """The pre-declared rule of WAVLM_CALIBRATION.md Section 7, applied once on the operating role."""
    if method == "identity":
        return {"accepted": False, "reason": "identity was selected: nothing to accept", "valid_fit": True, "log_loss_strictly_lower": False, "brier_not_higher": True}
    cal = np.asarray(calibrated, dtype=np.float64)
    valid = bool(fit_ok and np.isfinite(cal).all() and (cal >= 0).all() and (cal <= 1).all())
    raw_ll, cal_ll, raw_br, cal_br = log_loss(y, raw), log_loss(y, cal), brier(y, raw), brier(y, cal)
    lower, not_higher = bool(cal_ll < raw_ll), bool(cal_br <= raw_br)
    accepted = valid and lower and not_higher
    reasons = []
    if not valid:
        reasons.append("invalid fit")
    if not lower:
        reasons.append("log loss not strictly lower")
    if not not_higher:
        reasons.append("Brier score higher")
    return {"accepted": accepted, "reason": "all conditions hold" if accepted else "; ".join(reasons), "valid_fit": valid, "log_loss_strictly_lower": lower,
            "brier_not_higher": not_higher, "raw_log_loss": raw_ll, "calibrated_log_loss": cal_ll, "raw_brier": raw_br, "calibrated_brier": cal_br}


def row_metrics(y, comp, scores, rule: DecisionRule) -> dict:
    """Full-role and known/unseen panel metrics for one score set under one decision rule."""
    d = decide(rule, scores)
    out = {"full": decision_metrics(y, d)}
    for name, allowed in (("known", {"genuine", "melo", "vits"}), ("unseen_xtts", {"genuine", "xtts"})):
        mask = np.isin(comp, list(allowed))
        out[name] = decision_metrics(y[mask], d[mask])
    return out


def rule_json(rule: DecisionRule) -> dict:
    return {"kind": rule.kind, "threshold": rule.threshold}
