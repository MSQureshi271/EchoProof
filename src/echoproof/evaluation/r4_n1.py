"""Protocol A calibration orchestration for R4-N1 (LR-A and BiGRU-A)."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from echoproof.artifacts import digest, file_digest
from echoproof.contracts import DecisionRule
from echoproof.evaluation.affine_calibration import apply_affine, fit_affine
from echoproof.evaluation.calibrate import apply_temperature, fit_temperature
from echoproof.evaluation.metrics import compute_metrics
from echoproof.evaluation.reliability import expected_calibration_error, reliability_bins

ROLES = ("selection", "calibration", "operating")
ORDER = ("identity", "temperature", "affine")
COMPONENTS = ("genuine", "melo", "vits", "xtts", "chatterbox")
RECORDING_COLUMNS = ["recording_id", "group_id", "component", "role", "label", "raw_score", "chunk_count"]
CHUNK_COLUMNS = ["chunk_id", "recording_id", "group_id", "component", "role", "label", "raw_score"]


class R4N1Error(ValueError):
    """An R4-N1 recipe, input, or stage contract failed."""


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def load_recipe(path: Path) -> dict:
    try:
        recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R4N1Error(f"cannot read calibration recipe: {exc}") from exc
    if recipe.get("schema_version") != 1 or recipe.get("task") != "R4-N1-Calibration-A" or recipe.get("protocol") != "A":
        raise R4N1Error("not an R4-N1 Protocol A recipe")
    if recipe.get("candidates") != list(ORDER):
        raise R4N1Error("candidate order must be identity, temperature, affine")
    if recipe.get("epsilon") != 1e-7 or recipe.get("roles", {}).get("calibration", {}).get("recordings") != 1200:
        raise R4N1Error("predeclared A epsilon or role contract changed")
    if recipe.get("threshold_rule", {}).get("tie_break_order") != ["higher macro F1", "lower genuine FPR", "closer to 0.5", "smaller threshold"]:
        raise R4N1Error("predeclared A threshold tie-break changed")
    if Path(recipe.get("output_root", "")).as_posix() != "runs/r4/r4_n1_lr":
        raise R4N1Error("LR-A output_root must be runs/r4/r4_n1_lr")
    if recipe.get("models", {}).get("lr", {}).get("verification_state") != "verified_local":
        raise R4N1Error("LR-A inputs are not marked verified in the recipe")
    return recipe


def log_loss(labels: Sequence[int], scores: Sequence[float], epsilon: float = 1e-7) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(scores, dtype=np.float64), epsilon, 1.0 - epsilon)
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise R4N1Error("invalid inputs to log loss")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def brier(labels: Sequence[int], scores: Sequence[float]) -> float:
    y, p = np.asarray(labels, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise R4N1Error("invalid inputs to Brier score")
    return float(np.mean((p - y) ** 2))


def apply_mapping(method: str, params: dict, scores: Sequence[float], epsilon: float) -> list[float]:
    if method == "identity":
        return [float(score) for score in scores]
    if method == "temperature":
        return apply_temperature(scores, params["temperature"], epsilon=epsilon)
    if method == "affine":
        return apply_affine(scores, params["a"], params["b"], epsilon=epsilon)
    raise R4N1Error(f"unknown mapping {method!r}")


def fit_candidate(method: str, labels, scores, recipe: dict) -> dict:
    """Fit one mapping and normalize helper failures into an auditable result."""
    if method == "identity":
        return {"ok": True, "failure_reason": None, "params": {}, "objective": log_loss(labels, scores, recipe["epsilon"])}
    try:
        if method == "temperature":
            cfg = recipe["temperature"]
            value = float(fit_temperature(labels, scores, epsilon=recipe["epsilon"], bounds=tuple(cfg["bounds"])))
            lo, hi = cfg["bounds"]
            valid = math.isfinite(value) and lo + cfg["bound_tolerance"] < value < hi - cfg["bound_tolerance"]
            return {"ok": valid, "failure_reason": None if valid else "temperature non-finite or at/near a bound",
                    "params": {"temperature": value}, "objective": log_loss(labels, apply_temperature(scores, value, epsilon=recipe["epsilon"]), recipe["epsilon"]) if valid else None}
        if method == "affine":
            cfg = recipe["affine"]
            fit = fit_affine(labels, scores, epsilon=recipe["epsilon"], u_bounds=tuple(cfg["u_bounds"]), b_bounds=tuple(cfg["b_bounds"]),
                             reg=cfg["regularisation"], start=tuple(cfg["start"]))
            params = {"a": fit.get("a"), "u": fit.get("u"), "b": fit.get("b")}
            valid = bool(fit.get("ok")) and all(v is not None and math.isfinite(float(v)) for v in params.values()) and math.isfinite(float(fit["objective_at_fit"]))
            return {"ok": valid, "failure_reason": fit.get("failure_reason") if not valid else None, "params": params,
                    "objective": fit.get("objective_at_fit"), "diagnostics": fit}
    except Exception as exc:
        return {"ok": False, "failure_reason": f"fit raised {type(exc).__name__}: {exc}", "params": {}, "objective": None}
    raise R4N1Error(f"unknown mapping {method!r}")


def loso_comparison(labels, scores, groups, recipe: dict) -> dict:
    """Leave-one-speaker-out comparison; each held-out speaker has equal weight."""
    y, s, g = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64), np.asarray(groups)
    if not (len(y) == len(s) == len(g)) or len(set(g.tolist())) < 2:
        raise R4N1Error("LOSO needs aligned rows from at least two speakers")
    folds, losses, failures = [], {method: [] for method in ORDER}, {}
    speakers = sorted(set(g.tolist()))
    for speaker in speakers:
        hold = g == speaker
        row = {"held_out_speaker": str(speaker), "recordings": int(hold.sum()), "candidates": {}}
        for method in ORDER:
            fit = fit_candidate(method, y[~hold].tolist(), s[~hold].tolist(), recipe)
            candidate = {"ok": fit["ok"], "failure_reason": fit["failure_reason"], "params": fit["params"]}
            if fit["ok"]:
                try:
                    mapped = apply_mapping(method, fit["params"], s[hold].tolist(), recipe["epsilon"])
                    value = log_loss(y[hold], mapped, recipe["epsilon"])
                    candidate["held_out_log_loss"] = value
                    losses[method].append(value)
                except Exception as exc:
                    candidate.update({"ok": False, "failure_reason": f"held-out application failed: {type(exc).__name__}: {exc}"})
            if not candidate["ok"]:
                failures.setdefault(method, f"failed for held-out speaker {speaker}: {candidate['failure_reason']}")
            row["candidates"][method] = candidate
        folds.append(row)
    for method in ORDER:
        if method in failures:
            losses[method] = None
    return {"folds": folds, "per_speaker_log_loss": losses, "failures": failures, "speakers": [str(x) for x in speakers]}


def select_mapping(speaker_losses: dict, failures: dict) -> dict:
    eligible = {m: values for m, values in speaker_losses.items() if m not in failures and values is not None and len(values)}
    if "identity" not in eligible:
        raise R4N1Error("identity must remain eligible")
    means = {m: float(np.mean(values)) for m, values in eligible.items()}
    best = min(means, key=lambda m: (means[m], ORDER.index(m)))
    vals = np.asarray(eligible[best], dtype=np.float64)
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
    limit = means[best] + se
    chosen = next(m for m in ORDER if m in means and means[m] <= limit)
    return {"selected": chosen, "best": best, "means": means, "standard_error_of_best": se, "limit": limit,
            "eligible_order": [m for m in ORDER if m in means], "excluded": dict(failures)}


def _parquet_rows(path: Path, expected_columns: list[str]) -> list[dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    if table.column_names != expected_columns:
        raise R4N1Error(f"{path}: unexpected Parquet columns {table.column_names}; expected {expected_columns}")
    return table.to_pylist()


def _input_paths(root: Path, model: dict) -> tuple[Path, Path]:
    return root / model["export_directory"], root / model["training_evidence_path"]


def _check_recordings(rows: list[dict], role: str, recipe: dict) -> list[str]:
    expected = recipe["roles"][role]
    problems = []
    if len(rows) != expected["recordings"]:
        problems.append(f"{role}: recordings {len(rows)} != {expected['recordings']}")
    ids = [r["recording_id"] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append(f"{role}: duplicate recording ids")
    if {r["role"] for r in rows} != {role}:
        problems.append(f"{role}: role column mismatch")
    if len({r["group_id"] for r in rows}) != expected["speakers"]:
        problems.append(f"{role}: speaker count mismatch")
    counts = Counter(r["component"] for r in rows)
    if set(counts) != set(COMPONENTS) or any(counts[c] != expected["per_component"] for c in COMPONENTS):
        problems.append(f"{role}: component counts mismatch: {dict(counts)}")
    for r in rows:
        if r["label"] != recipe["protocol_contract"]["components"].get(r["component"], {}).get("label"):
            problems.append(f"{role}: label/component mismatch for {r['recording_id']}")
            break
        score = r["raw_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            problems.append(f"{role}: invalid score for {r['recording_id']}")
            break
        if not isinstance(r["chunk_count"], int) or r["chunk_count"] < 1:
            problems.append(f"{role}: invalid chunk_count for {r['recording_id']}")
            break
    return problems


def _check_chunk_parity(recordings: list[dict], chunks: list[dict], role: str, recipe: dict) -> float:
    expected = recipe["roles"][role]["chunks"]
    if len(chunks) != expected:
        raise R4N1Error(f"{role}: chunks {len(chunks)} != {expected}")
    chunk_ids = [r["chunk_id"] for r in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise R4N1Error(f"{role}: duplicate chunk ids")
    by_rec: dict[str, list[float]] = defaultdict(list)
    rec_by_id = {r["recording_id"]: r for r in recordings}
    for r in chunks:
        if r["role"] != role:
            raise R4N1Error(f"{role}: chunk role mismatch")
        parent = rec_by_id.get(r["recording_id"])
        if parent is None or any(r[column] != parent[column] for column in ("group_id", "component", "label", "role")):
            raise R4N1Error(f"{role}: chunk/recording metadata mismatch for {r['recording_id']}")
        by_rec[r["recording_id"]].append(float(r["raw_score"]))
    if set(by_rec) != {r["recording_id"] for r in recordings}:
        raise R4N1Error(f"{role}: chunk and recording identities differ")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for vals in by_rec.values() for v in vals):
        raise R4N1Error(f"{role}: invalid chunk score")
    if any(len(by_rec[r["recording_id"]]) != r["chunk_count"] for r in recordings):
        raise R4N1Error(f"{role}: chunk_count mismatch")
    return max((abs(float(np.mean(by_rec[r["recording_id"]])) - float(r["raw_score"])) for r in recordings), default=0.0)


def audit_inputs(root: Path, recipe: dict, model_name: str = "lr") -> dict:
    """Audit only selection and calibration contents; do not open operating Parquets."""
    if model_name not in ("lr", "bigru"):
        raise R4N1Error(f"unsupported A model {model_name!r}; choose lr or bigru")
    model = recipe["models"][model_name]
    if model.get("verification_state") != "verified_local":
        raise R4N1Error(f"{model_name} inputs are not verified")
    export_dir, training_evidence_path = _input_paths(root, model)
    export_evidence_path = root / model["export_evidence_path"]
    checks, problems = {}, []

    def check(name: str, condition: bool, detail: str = ""):
        checks[name] = bool(condition)
        if not condition:
            problems.append(f"{name}: {detail or 'failed'}")

    check("training_evidence_sha256", training_evidence_path.is_file() and file_digest(training_evidence_path) == model["training_evidence_sha256"])
    check("export_evidence_sha256", export_evidence_path.is_file() and file_digest(export_evidence_path) == model["export_evidence_sha256"])
    selected_path = root / model["selected_model_path"]
    check("selected_model_sha256", selected_path.is_file() and file_digest(selected_path) == model["selected_model_sha256"])
    model_metadata_path = root / model["model_metadata_path"] if model.get("model_metadata_path") else None
    if model_metadata_path is not None:
        check("model_metadata_exists", model_metadata_path.is_file())
    if not all(checks.values()):
        return {"passed": False, "checks": checks, "problems": problems, "operating_predictions_content_read": False}
    train_ev = json.loads(training_evidence_path.read_text(encoding="utf-8"))
    export_ev = json.loads(export_evidence_path.read_text(encoding="utf-8"))
    model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8")) if model_metadata_path else None
    expected_train_status = "training_complete" if model_name == "bigru" else "complete"
    check("training_complete", train_ev.get("status") == expected_train_status and train_ev.get("protocol") == "A")
    check("export_complete", export_ev.get("status") == "complete" and export_ev.get("protocol") == "A" and export_ev.get("model") == model_name)
    check("split_hash", train_ev.get("split_sha256") == recipe["protocol_contract"]["split_sha256"] == export_ev.get("split_sha256"))
    if model_metadata is not None:
        check("model_metadata_protocol_split", model_metadata.get("protocol") == "A" and model_metadata.get("split_sha256") == recipe["protocol_contract"]["split_sha256"])
    check("training_evidence_link", export_ev.get("training_evidence_sha256") == model["training_evidence_sha256"])
    train_run_dir = Path(model["training_evidence_path"]).parent
    selected_relative = Path(model["selected_model_path"]).relative_to(train_run_dir).as_posix()
    train_model_hash = train_ev.get("best_checkpoint_sha256") if model_name == "bigru" else train_ev.get("selected_model_sha256", train_ev.get("outputs", {}).get(selected_relative))
    check("selected_model_hash", train_model_hash == model["selected_model_sha256"] and export_ev.get("selected_model_sha256") == model["selected_model_sha256"])
    check("r4_s1_manifest_hash", export_ev.get("input_sha256", {}).get("manifest") == recipe["protocol_contract"]["r4_s1_manifest"]["sha256"])
    check("export_hash_map", export_ev.get("outputs") == model["prediction_sha256"])
    for rel, expected_hash in model["prediction_sha256"].items():
        role = next((r for r in ROLES if rel.endswith(f"_{r}.parquet")), None)
        if role == "operating":
            # Check presence only; actual bytes/schema/rows are deferred until fit freeze.
            check(f"operating_file_present:{rel}", (export_dir / rel).is_file())
            continue
        path = export_dir / rel
        check(f"prediction_sha256:{rel}", path.is_file() and file_digest(path) == expected_hash)
    if problems:
        return {"passed": False, "checks": checks, "problems": problems, "operating_predictions_content_read": False}

    rows_by_role, chunk_max_diffs = {}, {}
    for role in ("selection", "calibration"):
        rec = _parquet_rows(export_dir / f"predictions_recordings_{role}.parquet", RECORDING_COLUMNS)
        chunks = _parquet_rows(export_dir / f"predictions_chunks_{role}.parquet", CHUNK_COLUMNS)
        role_problems = _check_recordings(rec, role, recipe)
        if role_problems:
            problems.extend(role_problems)
            continue
        try:
            chunk_max_diffs[role] = _check_chunk_parity(rec, chunks, role, recipe)
        except R4N1Error as exc:
            problems.append(str(exc))
        rows_by_role[role] = rec
    if not problems:
        groups = {role: {row["group_id"] for row in rows} for role, rows in rows_by_role.items()}
        check("selection_calibration_speakers_disjoint", not (groups["selection"] & groups["calibration"]))
        check("selection_calibration_recordings_disjoint", not ({r["recording_id"] for r in rows_by_role["selection"]} & {r["recording_id"] for r in rows_by_role["calibration"]}))
    return {"passed": not problems, "checks": checks, "problems": problems, "recipe_sha256": digest(recipe),
            "roles": {r: {"recordings": len(rows), "speakers": len({x['group_id'] for x in rows})} for r, rows in rows_by_role.items()},
            "chunk_mean_max_abs_difference": chunk_max_diffs, "selection_and_calibration_groups": sorted({r["group_id"] for rows in rows_by_role.values() for r in rows}),
            "operating_predictions_content_read": False, "operating_parquet_sha256_verified": False,
            "operating_expected_sha256": {name: model["prediction_sha256"][name] for name in model["prediction_sha256"] if name.endswith("_operating.parquet")},
            "r4_s1_manifest_verification": "export-evidence hash binding only; manifest file is documented but absent locally"}


def fit_stage(root: Path, recipe: dict, run_dir: Path, audit: dict, model_name: str = "lr") -> dict:
    """Fit LOSO and final mappings on calibration rows only, then freeze them."""
    if not audit.get("passed") or audit.get("operating_predictions_content_read") or audit.get("recipe_sha256") != digest(recipe):
        raise R4N1Error("input audit did not pass or opened operating predictions")
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "input_audit.json", audit)
    model = recipe["models"][model_name]
    export_dir = root / model["export_directory"]
    cal_rows = _parquet_rows(export_dir / "predictions_recordings_calibration.parquet", RECORDING_COLUMNS)
    y = np.asarray([r["label"] for r in cal_rows], dtype=np.int64)
    s = np.asarray([r["raw_score"] for r in cal_rows], dtype=np.float64)
    groups = np.asarray([r["group_id"] for r in cal_rows])
    comparison = loso_comparison(y, s, groups, recipe)
    final_fits = {method: fit_candidate(method, y.tolist(), s.tolist(), recipe) for method in ORDER}
    failures = dict(comparison["failures"])
    for method in ORDER:
        if not final_fits[method]["ok"]:
            failures.setdefault(method, f"full calibration refit failed: {final_fits[method]['failure_reason']}")
    losses = comparison["per_speaker_log_loss"]
    rule = select_mapping(losses, failures)
    selected = rule["selected"]
    comparison.update({"selection": rule, "candidate_order": list(ORDER), "failures_after_final_fit": failures,
                       "fit_role": "calibration", "operating_predictions_read": False, "test_data_read": False})
    fit_dir = run_dir / "fit"
    fit_dir.mkdir()
    write_json(fit_dir / "loso_comparison.json", comparison)
    cal_result = {"task": f"R4-N1 {model_name.upper()}-A mapping fit", "model": model_name, "selected_mapping": selected, "selected_parameters": final_fits[selected]["params"],
                  "selection": rule, "candidate_fits": final_fits, "fit_role": "calibration", "fit_recordings": len(cal_rows),
                  "fit_speakers": len(set(groups.tolist())), "epsilon": recipe["epsilon"], "operating_predictions_read": False,
                  "test_data_read": False, "model_retrained": False, "threshold_selected": False,
                  "source_calibration_predictions_sha256": model["prediction_sha256"]["predictions_recordings_calibration.parquet"],
                  "recipe_sha256": digest(recipe), "input_audit_sha256": file_digest(run_dir / "input_audit.json")}
    write_json(fit_dir / "calibration.json", cal_result)
    files = {name: file_digest(fit_dir / name) for name in ("loso_comparison.json", "calibration.json")}
    complete = {"task": "R4-N1 LR-A fit stage complete", "files": files, "recipe_sha256": digest(recipe),
                "mapping_frozen": True, "mapping": selected, "operating_predictions_read": False}
    write_json(fit_dir / "stage_complete.json", complete)
    return cal_result


def _decision_metrics(labels, decisions) -> dict:
    y, d = np.asarray(labels, dtype=np.int64), np.asarray(decisions, dtype=np.int64)
    tp = int(((y == 1) & (d == 1)).sum())
    fp = int(((y == 0) & (d == 1)).sum())
    tn = int(((y == 0) & (d == 0)).sum())
    fn = int(((y == 1) & (d == 0)).sum())
    f1_tts = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_gen = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "macro_f1": (f1_tts + f1_gen) / 2,
            "genuine_fpr": fp / (fp + tn) if fp + tn else 0.0}


def select_threshold_a(labels: Sequence[int], scores: Sequence[float], tolerance: float = 1e-12) -> tuple[DecisionRule, dict]:
    """R4 A threshold rule; final numeric tie prefers the smaller threshold."""
    y, s = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    if len(y) != len(s) or not len(y) or set(y.tolist()) != {0, 1} or not np.isfinite(s).all() or (s < 0).any() or (s > 1).any():
        raise R4N1Error("invalid threshold inputs; need aligned finite [0,1] scores and both classes")
    candidates = []
    for threshold in np.unique(s):
        d = (s >= threshold).astype(np.int64)
        met = _decision_metrics(y, d)
        candidates.append({"kind": "threshold", "threshold": float(threshold), **met, "dist_05": abs(float(threshold) - 0.5)})
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    all_neg_macro = (2 * n_neg / (2 * n_neg + n_pos)) / 2
    candidates.append({"kind": "all_negative", "threshold": None, "macro_f1": all_neg_macro, "genuine_fpr": 0.0,
                       "tp": 0, "fp": 0, "tn": n_neg, "fn": n_pos, "dist_05": 1.0})
    rounded_f1 = lambda c: round(c["macro_f1"] / tolerance) * tolerance
    rounded_fpr = lambda c: round(c["genuine_fpr"] / tolerance) * tolerance
    best_f1 = max(rounded_f1(c) for c in candidates)
    f1_tied = [c for c in candidates if rounded_f1(c) == best_f1]
    best_fpr = min(rounded_fpr(c) for c in f1_tied)
    fpr_tied = [c for c in f1_tied if rounded_fpr(c) == best_fpr]
    best_dist = min(round(c["dist_05"] / tolerance) * tolerance for c in fpr_tied)
    dist_tied = [c for c in fpr_tied if round(c["dist_05"] / tolerance) * tolerance == best_dist]
    # All-negative is an explicit candidate. If all earlier criteria tie, a numeric threshold
    # is preferred because it has a defined threshold; among numeric candidates choose smaller.
    best = min(dist_tied, key=lambda c: (c["threshold"] is None, c["threshold"] if c["threshold"] is not None else math.inf))
    rule = DecisionRule(1, "all_negative", None) if best["kind"] == "all_negative" else DecisionRule(1, "threshold", best["threshold"])
    plateau = {"macro_f1_max": float(max(c["macro_f1"] for c in candidates)),
               "thresholds_tied_at_max_macro_f1": sum(c["kind"] == "threshold" and rounded_f1(c) == best_f1 for c in candidates),
               "all_negative_tied_at_max_macro_f1": any(c["kind"] == "all_negative" and rounded_f1(c) == best_f1 for c in candidates),
               "thresholds_remaining_after_fpr_tie_break": sum(c["kind"] == "threshold" and rounded_f1(c) == best_f1 and rounded_fpr(c) == best_fpr for c in candidates),
               "thresholds_remaining_after_closeness_tie_break": sum(c["kind"] == "threshold" and rounded_f1(c) == best_f1 and rounded_fpr(c) == best_fpr and round(c["dist_05"] / tolerance) * tolerance == best_dist for c in candidates),
               "tolerance": tolerance}
    return rule, {"macro_f1": best["macro_f1"], "genuine_fpr": best["genuine_fpr"], "plateau": plateau, "candidate_count": len(candidates)}


def acceptance(method: str, fit: dict, labels, raw_scores, mapped_scores) -> dict:
    if method == "identity":
        return {"accepted": False, "reason": "identity mapping; retain raw scores", "valid_fit": True}
    raw_ll, mapped_ll = log_loss(labels, raw_scores), log_loss(labels, mapped_scores)
    raw_br, mapped_br = brier(labels, raw_scores), brier(labels, mapped_scores)
    params = fit.get("params", {})
    valid = bool(fit.get("ok")) and all(v is None or math.isfinite(float(v)) for v in params.values()) and np.isfinite(mapped_scores).all() and (np.asarray(mapped_scores) >= 0).all() and (np.asarray(mapped_scores) <= 1).all()
    ll_lower, br_not_higher = mapped_ll < raw_ll, mapped_br <= raw_br
    accepted = valid and ll_lower and br_not_higher
    reasons = []
    if not valid: reasons.append("invalid fit")
    if not ll_lower: reasons.append("operating log loss not strictly lower")
    if not br_not_higher: reasons.append("operating Brier score higher")
    return {"accepted": accepted, "reason": "all conditions hold" if accepted else "; ".join(reasons), "valid_fit": valid,
            "raw_log_loss": raw_ll, "mapped_log_loss": mapped_ll, "log_loss_strictly_lower": ll_lower,
            "raw_brier": raw_br, "mapped_brier": mapped_br, "brier_not_higher": br_not_higher}


def _bootstrap_operating(labels, groups, decisions, *, resamples: int, seed: int, alpha: float) -> dict:
    y, g, d = np.asarray(labels, dtype=np.int64), np.asarray(groups), np.asarray(decisions, dtype=np.int64)
    unique = sorted(set(g.tolist()))
    by_group = {speaker: np.flatnonzero(g == speaker) for speaker in unique}
    rng = np.random.RandomState(seed)
    samples = {"macro_f1": [], "genuine_fpr": []}
    for _ in range(resamples):
        chosen = rng.choice(len(unique), size=len(unique), replace=True)
        idx = np.concatenate([by_group[unique[i]] for i in chosen])
        if len(set(y[idx].tolist())) < 2:
            continue
        metrics = _decision_metrics(y[idx], d[idx])
        for name in samples: samples[name].append(metrics[name])
    if not samples["macro_f1"]:
        raise R4N1Error("no valid speaker bootstrap replicates")
    return {name: {"estimate": _decision_metrics(y, d)[name], "ci_lower": float(np.percentile(vals, 100 * alpha / 2)),
                   "ci_upper": float(np.percentile(vals, 100 * (1 - alpha / 2)))} for name, vals in samples.items()} | {
                   "resamples_requested": resamples, "valid_resamples": len(samples["macro_f1"]), "seed": seed, "alpha": alpha, "unit": "speaker"}


def _probability_summary(labels, scores, bins: int) -> dict:
    metrics = compute_metrics(labels, scores, 0.5)
    return {"log_loss": metrics["log_loss"], "brier": metrics["brier_score"], "ece": expected_calibration_error(labels, scores, bins),
            "roc_auc": metrics["roc_auc"], "eer": metrics["eer"], "reliability_bins": reliability_bins(labels, scores, bins)}


def _plot_reliability(series: dict[str, list[dict]], destination: Path, model_name: str = "lr") -> None:
    """Write a reliability diagram from the already-computed equal-width bins."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite reliability plot {destination}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise R4N1Error("matplotlib is required to write the predeclared reliability plot") from exc
    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="Perfect calibration")
    for name, bins in series.items():
        points = [b for b in bins if b["count"] and b["mean_score"] is not None and b["observed_rate"] is not None]
        ax.plot([b["mean_score"] for b in points], [b["observed_rate"] for b in points], marker="o", linewidth=1.5, label=name)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted TTS probability", ylabel="Observed TTS frequency",
           title=f"{model_name.upper()}-A operating-role reliability")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(destination, dpi=160, format="png")
    finally:
        plt.close(fig)


def operating_stage(root: Path, recipe: dict, run_dir: Path, model_name: str = "lr") -> dict:
    fit_dir = run_dir / "fit"
    complete_path = fit_dir / "stage_complete.json"
    if not complete_path.is_file():
        raise R4N1Error("fit stage is not complete; operating predictions remain unopened")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("recipe_sha256") != digest(recipe) or not complete.get("mapping_frozen"):
        raise R4N1Error("fit stage does not freeze a mapping for this recipe")
    if any(file_digest(fit_dir / name) != value for name, value in complete["files"].items()):
        raise R4N1Error("frozen fit artifact hash mismatch")
    out = run_dir / "operating"
    out.mkdir(exist_ok=False)
    write_json(out / "opened.json", {"mapping_frozen_before_opening": complete["mapping"], "fit_stage_complete_sha256": file_digest(complete_path)})
    model = recipe["models"][model_name]
    export_dir = root / model["export_directory"]
    evidence = json.loads((root / model["export_evidence_path"]).read_text(encoding="utf-8"))
    rel = "predictions_recordings_operating.parquet"
    path = export_dir / rel
    expected_hash = model["prediction_sha256"][rel]
    chunk_rel = "predictions_chunks_operating.parquet"
    chunk_expected_hash = model["prediction_sha256"][chunk_rel]
    if (evidence.get("outputs", {}).get(rel) != expected_hash or file_digest(path) != expected_hash
            or evidence.get("outputs", {}).get(chunk_rel) != chunk_expected_hash
            or file_digest(export_dir / chunk_rel) != chunk_expected_hash):
        raise R4N1Error("operating prediction hash mismatch")
    rows = _parquet_rows(path, RECORDING_COLUMNS)
    problems = _check_recordings(rows, "operating", recipe)
    if problems:
        raise R4N1Error("; ".join(problems))
    groups = {r["group_id"] for r in rows}
    fit_audit = json.loads((run_dir / "input_audit.json").read_text(encoding="utf-8"))
    if groups & set(fit_audit["selection_and_calibration_groups"]):
        raise R4N1Error("operating speakers overlap selection/calibration speakers")
    chunks = _parquet_rows(export_dir / "predictions_chunks_operating.parquet", CHUNK_COLUMNS)
    diff = _check_chunk_parity(rows, chunks, "operating", recipe)

    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    raw = np.asarray([r["raw_score"] for r in rows], dtype=np.float64)
    grp = np.asarray([r["group_id"] for r in rows])
    comp = np.asarray([r["component"] for r in rows])
    fit = json.loads((fit_dir / "calibration.json").read_text(encoding="utf-8"))
    method, params = fit["selected_mapping"], fit["selected_parameters"]
    selected_fit = fit["candidate_fits"][method]
    mapped = np.asarray(apply_mapping(method, params, raw.tolist(), recipe["epsilon"]), dtype=np.float64)
    accepted = acceptance(method, selected_fit, y, raw, mapped)
    deployed = mapped if accepted["accepted"] else raw
    rule, threshold = select_threshold_a(y.tolist(), deployed.tolist(), recipe["threshold_rule"]["metric_tie_tolerance"])
    decisions = np.asarray([rule.decide(float(x)) for x in deployed], dtype=np.int64) if rule.kind == "threshold" else np.zeros(len(y), dtype=np.int64)
    bins = recipe["reliability"]["bins"]
    probability = {"raw": _probability_summary(y.tolist(), raw.tolist(), bins),
                   "mapped": _probability_summary(y.tolist(), mapped.tolist(), bins),
                   "deployed": _probability_summary(y.tolist(), deployed.tolist(), bins)}
    reliability_path = out / "reliability.png"
    _plot_reliability({name: values["reliability_bins"] for name, values in probability.items()}, reliability_path, model_name)
    per_component = {}
    for generator in COMPONENTS[1:]:
        mask = (comp == "genuine") | (comp == generator)
        per_component[generator] = {"recordings": int(mask.sum()), "metrics": compute_metrics(y[mask].tolist(), deployed[mask].tolist(), float(rule.threshold) if rule.threshold is not None else 0.5) if rule.kind == "threshold" else _decision_metrics(y[mask], np.zeros(int(mask.sum()), dtype=np.int64))}
    speaker_rows = []
    for speaker in sorted(set(grp.tolist())):
        mask = grp == speaker
        genuine = mask & (y == 0)
        fa = int(decisions[genuine].sum())
        speaker_rows.append({"speaker": str(speaker), "genuine_recordings": int(genuine.sum()), "false_alarms": fa,
                             "false_alarm_rate": fa / int(genuine.sum()) if genuine.any() else None})
    bootstrap = _bootstrap_operating(y, grp, decisions, resamples=recipe["bootstrap"]["resamples"], seed=recipe["bootstrap"]["seed"], alpha=recipe["bootstrap"]["alpha"])
    report = {"task": f"R4-N1 {model_name.upper()}-A operating acceptance and threshold", "model": model_name, "protocol": "A", "mapping": method,
              "mapping_accepted": accepted["accepted"], "score_type": "calibrated_probability" if accepted["accepted"] else "model_score",
              "threshold_score_source": "mapped" if accepted["accepted"] else "raw", "acceptance": accepted,
              "threshold_rule": {"kind": rule.kind, "threshold": rule.threshold, "decision": "score >= threshold"}, "threshold_metrics": threshold,
              "operating_recordings": len(rows), "operating_speakers": len(groups), "operating_chunk_mean_max_abs_difference": diff,
              "probability_metrics": probability, "reliability_plot": {"path": reliability_path.relative_to(run_dir).as_posix(), "sha256": file_digest(reliability_path),
              "series": list(probability)}, "per_component": per_component, "per_speaker_false_alarms": speaker_rows,
              "speaker_bootstrap": bootstrap, "operating_predictions_sha256": expected_hash, "recipe_sha256": digest(recipe),
              "test_data_read": False, "operating_opened_after_fit_freeze": True}
    write_json(out / "operating_results.json", report)
    write_json(out / "stage_complete.json", {"task": "R4-N1 LR-A operating stage complete", "recipe_sha256": digest(recipe),
                "operating_results_sha256": file_digest(out / "operating_results.json"), "test_data_read": False})
    return report


def resolve_run_dir(root: Path, recipe: dict, run_dir: Path | None = None, model_name: str = "lr") -> Path:
    root = Path(root).resolve()
    default = recipe["output_root"] if model_name == "lr" else "runs/r4/r4_n1_bigru"
    output = Path(run_dir) if run_dir is not None else Path(default)
    return output if output.is_absolute() else root / output


def run_lr(root: Path, recipe_path: Path, run_dir: Path | None = None) -> dict:
    return run_model(root, recipe_path, "lr", run_dir)


def run_bigru(root: Path, recipe_path: Path, run_dir: Path | None = None) -> dict:
    return run_model(root, recipe_path, "bigru", run_dir)


def run_model(root: Path, recipe_path: Path, model_name: str, run_dir: Path | None = None) -> dict:
    root = Path(root).resolve()
    recipe = load_recipe(recipe_path)
    if model_name not in ("lr", "bigru"):
        raise R4N1Error(f"unsupported A model {model_name!r}; choose lr or bigru")
    output = resolve_run_dir(root, recipe, run_dir, model_name)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run {output}")
    audit = audit_inputs(root, recipe, model_name)
    if not audit["passed"]:
        raise R4N1Error(f"{model_name.upper()}-A input audit failed: " + "; ".join(audit["problems"][:8]))
    fit_stage(root, recipe, output, audit, model_name)
    result = operating_stage(root, recipe, output, model_name)
    return {"run_dir": str(output), "status": "complete", "model": model_name, "protocol": "A", "selected_mapping": result["mapping"],
            "mapping_accepted": result["mapping_accepted"], "threshold_rule": result["threshold_rule"]}

