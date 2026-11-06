"""R2-S3 step 2 (local, no audio): the shortcut probe.

Question: can *nuisance variables alone* (how long, how loud, how much silence, what sample rate) tell genuine speech from synthetic speech?
If they can, a detector may be learning the recording habits of a generator instead of the speech itself.

Method (fixed before looking at selection results):
  * Train on B train recordings, score on B selection-role validation recordings. Features are standardised with train statistics only.
  * Primary model, as the plan names it: logistic regression on duration, RMS, silence fraction, original sample rate. Constant columns are
    dropped (the sample rate is 16 kHz everywhere, so it cannot carry information) and reported.
  * Wider sets add the measurements R2-S2 flagged (leading/trailing silence, noise floor and level measures, high-band shares), and each
    measurement is also tried alone. A shallow gradient-boosted model on all measurements is a supplementary check for non-linear shortcuts.
  * LR strength is chosen by speaker-grouped cross-validation on train only. Selection data never influences any choice.
  * Duration-matched comparison: recordings are binned by duration (1 s bins); in every bin both classes keep the same number of recordings,
    chosen by a seeded hash. The probe is repeated on matched selection data, with the model trained on all train data and on matched train data.
  * Uncertainty: speaker-clustered bootstrap (R1-N1 `paired_group_bootstrap`, 1,000 resamples, seed 42); with only 25 selection speakers the
    intervals are wide, and the verdict uses their lower end.
Verdict rule: a variable set *separates* the classes if the lower end of the 95% interval of selection ROC-AUC exceeds SHORTCUT_AUC_FLOOR (0.60);
it is *marginal* if the point estimate reaches 0.60 but the lower end does not; otherwise it does *not separate*.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from echoproof.eda import measure, selection
from echoproof.evaluation.bootstrap import paired_group_bootstrap
from echoproof.evaluation.metrics import compute_metrics

TASK = "R2-S3"
SHORTCUT_AUC_FLOOR = 0.60
SEED = 42
N_RESAMPLES = 1000
BIN_SECONDS = 1.0
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
PRIMARY = ("duration_s", "rms_dbfs", "silence_fraction", "sample_rate")
WITH_EDGES = PRIMARY + ("leading_silence_s", "trailing_silence_s")
WITH_LEVELS = WITH_EDGES + ("noise_floor_dbfs", "active_level_dbfs", "peak", "clip_fraction", "dc_offset")
ALL_MEASURED = WITH_LEVELS + ("hf_ratio_4k", "hf_ratio_7k")
FEATURE_SETS = {"primary (plan: duration, RMS, silence fraction, sample rate)": PRIMARY, "plus leading/trailing silence": WITH_EDGES,
                "plus level measures": WITH_LEVELS, "all measurements (adds high-band shares)": ALL_MEASURED}
SINGLE_FEATURES = tuple(c for c in ALL_MEASURED if c != "sample_rate")
CONFIGS = {"A": "train on all train, score all selection", "B": "train on all train, score duration-matched selection",
           "C": "train on duration-matched train, score duration-matched selection"}
OUTPUT_JSON = "r2_s3_shortcut_probe.json"
OUTPUT_MEMO = "r2_s3_shortcut_memo.md"
MATCHED_NAME = "duration_matched_B_v1.json"


class ProbeError(RuntimeError):
    pass


# ---- data --------------------------------------------------------------------------------------------------------------------

def _verified_frame(directory: Path, manifest_name: str, output_key: str):
    import pandas as pd
    manifest_path = directory / manifest_name
    if not manifest_path.exists():
        raise ProbeError(f"{manifest_path} not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["outputs"][output_key]
    if hashlib.sha256((directory / spec["file"]).read_bytes()).hexdigest() != spec["sha256"]:
        raise ProbeError(f"{spec['file']} does not match the hash in {manifest_name}")
    return pd.read_parquet(directory / spec["file"]), manifest


def load_data(root, train_dir=None, selection_dir=None) -> dict:
    from echoproof.data.cohort import read_table
    root = Path(root)
    train, train_manifest = _verified_frame(Path(train_dir) if train_dir else root / "data" / "eda" / "r2_s2_B_train", measure.MANIFEST_NAME, "stats")
    sel, sel_manifest = _verified_frame(Path(selection_dir) if selection_dir else root / "data" / "eda" / "r2_s3_B_selection",
                                        selection.MANIFEST_NAME, "stats")
    if train_manifest["inputs"]["split_sha256"] != sel_manifest["inputs"]["split_sha256"]:
        raise ProbeError("train and selection measurements were made under different split hashes")
    if train_manifest["settings"] != sel_manifest["settings"]:
        raise ProbeError("train and selection measurements use different measurement settings")
    if set(train["component"]) != {"genuine", "melo", "vits"} or set(sel["component"]) != set(selection.SELECTION_COMPONENTS):
        raise ProbeError("unexpected components in the measurement files")
    if set(sel["role"]) != {"selection"}:
        raise ProbeError("the selection file contains other validation roles")
    if set(train["group_id"]) & set(sel["group_id"]):
        raise ProbeError("a speaker appears in train and selection")
    rate = {r["recording_id"]: int(r["original_sample_rate"]) for r in read_table(root / "data" / "manifests" / "cohort_v1.parquet")}
    for frame in (train, sel):
        frame["sample_rate"] = frame["recording_id"].map(rate).astype(float)
        if frame["sample_rate"].isna().any():
            raise ProbeError("a recording has no sample rate in the cohort manifest")
    return {"train": train.reset_index(drop=True), "selection": sel.reset_index(drop=True),
            "train_manifest": train_manifest, "selection_manifest": sel_manifest}


# ---- duration matching -------------------------------------------------------------------------------------------------------

def duration_matched(frame, *, seed: int = SEED, width: float = BIN_SECONDS) -> list[str]:
    """Recording ids such that, within every duration bin, genuine and synthetic recordings are equally many."""
    bins = np.floor(frame["duration_s"].to_numpy() / width).astype(int)
    keep = []
    for b in sorted(set(bins)):
        rows = frame[bins == b]
        zero, one = rows[rows["label"] == 0], rows[rows["label"] == 1]
        n = min(len(zero), len(one))
        for part in (zero, one):
            ranked = sorted(part["recording_id"], key=lambda r: hashlib.sha256(f"{seed}|r2_s3_match|{r}".encode()).hexdigest())
            keep.extend(ranked[:n])
    return sorted(keep)


# ---- models ------------------------------------------------------------------------------------------------------------------

def _balanced_weights(y):
    y = np.asarray(y)
    return np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1)) * len(y)


def fit_lr(frame, columns):
    """Standardised, class-balanced logistic regression; C picked by speaker-grouped CV on this frame only. Returns (predict_fn, info)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X, y, groups = frame[list(columns)].to_numpy(float), frame["label"].to_numpy(), frame["group_id"].to_numpy()
    spread = X.std(axis=0)
    kept = [c for c, s in zip(columns, spread) if s > 0]
    dropped = [c for c, s in zip(columns, spread) if s == 0]
    if not kept:                                   # nothing varies in train: the variables carry no information, the score is a constant
        info = {"features_used": [], "constant_columns_dropped": dropped, "C": None, "grouped_cv_auc_by_C": {}, "standardised_coefficients": {},
                "constant_model": True}
        return (lambda other: np.full(len(other), 0.5)), info
    X = X[:, [columns.index(c) for c in kept]]
    mean, std = X.mean(axis=0), X.std(axis=0)
    Z = (X - mean) / std
    cv = {}
    for C in C_GRID:
        scores = []
        for tr, te in GroupKFold(n_splits=5).split(Z, y, groups):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], y[tr], sample_weight=_balanced_weights(y[tr]))
            scores.append(roc_auc_score(y[te], m.predict_proba(Z[te])[:, 1]))
        cv[C] = float(np.mean(scores))
    best_c = max(C_GRID, key=lambda c: (cv[c], -c))
    model = LogisticRegression(C=best_c, max_iter=2000).fit(Z, y, sample_weight=_balanced_weights(y))

    def predict(other):
        return model.predict_proba((other[kept].to_numpy(float) - mean) / std)[:, 1]
    info = {"features_used": kept, "constant_columns_dropped": dropped, "C": best_c, "grouped_cv_auc_by_C": {str(k): v for k, v in cv.items()},
            "standardised_coefficients": dict(zip(kept, [float(v) for v in model.coef_[0]]))}
    return predict, info


def fit_gbm(frame, columns):
    from sklearn.ensemble import GradientBoostingClassifier
    cols = [c for c in columns if frame[c].std() > 0]
    model = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.1, random_state=SEED)
    model.fit(frame[cols].to_numpy(float), frame["label"].to_numpy(), sample_weight=_balanced_weights(frame["label"]))
    importance = dict(sorted(zip(cols, [float(v) for v in model.feature_importances_]), key=lambda kv: -kv[1]))
    return (lambda other: model.predict_proba(other[cols].to_numpy(float))[:, 1]), {"features_used": cols, "importance": importance}


# ---- evaluation --------------------------------------------------------------------------------------------------------------

def _auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def evaluate(frame, scores, *, seed: int = SEED, n_resamples: int = N_RESAMPLES) -> dict:
    """Recording-level metrics at 0.5 plus a speaker-clustered bootstrap interval of ROC-AUC, overall and genuine vs each generator."""
    labels = frame["label"].to_numpy().astype(int)
    scores = np.clip(np.asarray(scores, float), 0.0, 1.0)
    metrics = compute_metrics(labels.tolist(), scores.tolist(), 0.5)
    boot = paired_group_bootstrap(frame["group_id"].tolist(), labels.tolist(), {"m": scores.tolist()}, _auc, n_resamples=n_resamples, seed=seed)
    interval = boot["models"]["m"] if "models" in boot else boot["m"]
    out = {"n": int(len(frame)), "n_genuine": int((labels == 0).sum()), "n_synthetic": int((labels == 1).sum()), "roc_auc": metrics["roc_auc"],
           "auc_ci95": [float(interval["ci_lower"]), float(interval["ci_upper"])], "macro_f1_at_0.5": metrics["macro_f1"],
           "genuine_recall": metrics["genuine_recall"], "eer": metrics["eer"], "per_generator_auc": {}}
    for comp in sorted(set(frame["component"]) - {"genuine"}):
        mask = ((frame["component"] == comp) | (frame["component"] == "genuine")).to_numpy()
        out["per_generator_auc"][comp] = _auc(labels[mask], scores[mask])
    out["verdict"] = verdict(out["roc_auc"], out["auc_ci95"][0])
    out["reversed"] = bool(out["auc_ci95"][1] < 0.5)      # annotation only, added after the first results; never changes the verdict
    return out


def verdict(auc_value: float, ci_low: float) -> str:
    if ci_low > SHORTCUT_AUC_FLOOR:
        return "separates"
    if auc_value >= SHORTCUT_AUC_FLOOR:
        return "marginal"
    return "does not separate"


def run_probe(data: dict, *, n_resamples: int = N_RESAMPLES) -> dict:
    train, sel = data["train"], data["selection"]
    ids_train, ids_sel = duration_matched(train), duration_matched(sel)
    train_m, sel_m = train[train["recording_id"].isin(ids_train)].reset_index(drop=True), sel[sel["recording_id"].isin(ids_sel)].reset_index(drop=True)
    matching = {}
    for name, full, matched in (("train", train, train_m), ("selection", sel, sel_m)):
        matching[name] = {"recordings_full": int(len(full)), "recordings_matched": int(len(matched)),
                          "genuine_matched": int((matched["label"] == 0).sum()), "synthetic_matched": int((matched["label"] == 1).sum()),
                          "synthetic_components_matched": {k: int(v) for k, v in matched["component"].value_counts().sort_index().items()},
                          "median_duration_s_matched": {k: float(v) for k, v in matched.groupby("label")["duration_s"].median().items()},
                          "duration_auc_before": _auc(full["label"], full["duration_s"]), "duration_auc_after": _auc(matched["label"], matched["duration_s"])}
    results = {}
    for set_name, cols in FEATURE_SETS.items():
        predict_full, info_full = fit_lr(train, list(cols))
        predict_matched, info_matched = fit_lr(train_m, list(cols))
        results[set_name] = {"columns": list(cols), "model_all_train": info_full, "model_matched_train": info_matched,
                             "A": evaluate(sel, predict_full(sel), n_resamples=n_resamples),
                             "B": evaluate(sel_m, predict_full(sel_m), n_resamples=n_resamples),
                             "C": evaluate(sel_m, predict_matched(sel_m), n_resamples=n_resamples)}
    singles = {}
    for col in SINGLE_FEATURES:
        predict_full, info = fit_lr(train, [col])
        predict_matched, _ = fit_lr(train_m, [col])
        singles[col] = {"coefficient": info["standardised_coefficients"].get(col, 0.0), "constant_in_train": bool(info.get("constant_model", False)),
                        "A": evaluate(sel, predict_full(sel), n_resamples=n_resamples),
                        "B": evaluate(sel_m, predict_full(sel_m), n_resamples=n_resamples),
                        "C": evaluate(sel_m, predict_matched(sel_m), n_resamples=n_resamples)}
    gbm_predict, gbm_info = fit_gbm(train, list(ALL_MEASURED))
    gbm_matched_predict, _ = fit_gbm(train_m, list(ALL_MEASURED))
    supplementary = {"gradient_boosting_all_measurements": {**gbm_info, "A": evaluate(sel, gbm_predict(sel), n_resamples=n_resamples),
                                                           "B": evaluate(sel_m, gbm_predict(sel_m), n_resamples=n_resamples),
                                                           "C": evaluate(sel_m, gbm_matched_predict(sel_m), n_resamples=n_resamples)}}
    references = {"random_scores_auc": 0.5, "majority_class_macro_f1_at_0.5": float(compute_metrics(
        sel["label"].astype(int).tolist(), [1.0] * len(sel), 0.5)["macro_f1"])}
    return {"matching": matching, "feature_sets": results, "single_features": singles, "supplementary": supplementary, "references": references,
            "matched_ids": {"train": ids_train, "selection": ids_sel}}


# ---- conclusions and memo ----------------------------------------------------------------------------------------------------

def conclusions(result: dict) -> dict:
    """Plain facts for the memo, derived from the numbers."""
    separating = {name: {k: entry[k]["verdict"] for k in "ABC"} for name, entry in result["feature_sets"].items()}
    single = {col: {k: e[k]["verdict"] for k in "ABC"} for col, e in result["single_features"].items()}
    survivors = [col for col, v in single.items() if v["A"] in ("separates", "marginal") and v["C"] in ("separates", "marginal")]
    lost_after_matching = [col for col, v in single.items() if v["A"] in ("separates", "marginal") and v["C"] == "does not separate"]
    primary = result["feature_sets"][next(iter(FEATURE_SETS))]
    any_set = any(v["A"] == "separates" for v in separating.values())
    return {"any_feature_set_separates_on_all_selection": any_set, "feature_set_verdicts": separating, "single_feature_verdicts": single,
            "single_features_that_survive_duration_matching": survivors, "single_features_lost_after_matching": lost_after_matching,
            "primary_model_verdicts": {k: primary[k]["verdict"] for k in "ABC"}}


def _fmt(entry):
    return f"{entry['roc_auc']:.3f} [{entry['auc_ci95'][0]:.3f}, {entry['auc_ci95'][1]:.3f}]"


REVERSAL_NOTE = ("*Reversed* (added after the first results, not part of the verdict rule): the whole 95% interval lies below "
                 "0.50, so the variable separates the classes in the opposite direction on selection than the one learned on train. "
                 "The rule above calls this 'does not separate', but it is not harmless: the variable differs between classes and the relationship does not carry over.")


def memo_text(result: dict, concl: dict, data: dict) -> str:
    m = result["matching"]
    lines = ["# R2-S3 shortcut probe: do nuisance variables separate genuine from synthetic speech?", "",
             "Protocol B. Trained on B train, scored on B selection validation. No calibration, operating, test or Chatterbox data is used.", ""]
    primary_name = next(iter(FEATURE_SETS))
    p = result["feature_sets"][primary_name]
    lines += ["## Answer", ""]
    verdicts = concl["feature_set_verdicts"]
    sep_sets = [n for n, v in verdicts.items() if v["A"] == "separates"]
    if sep_sets:
        lines.append(f"**Yes.** Metadata alone separates the classes on selection validation: {len(sep_sets)} of {len(verdicts)} variable sets have an AUC "
                     f"interval whose lower end is above {SHORTCUT_AUC_FLOOR:.2f}. The plan's primary model (duration, RMS, silence fraction, sample rate) reaches "
                     f"AUC {_fmt(p['A'])} ({p['A']['verdict']}).")
    else:
        lines.append(f"**No variable set clears the {SHORTCUT_AUC_FLOOR:.2f} floor with confidence.** The plan's primary model reaches AUC {_fmt(p['A'])} "
                     f"({p['A']['verdict']}).")
    surv, lost = concl["single_features_that_survive_duration_matching"], concl["single_features_lost_after_matching"]
    lines.append("")
    lines.append(f"After duration matching the primary model scores AUC {_fmt(p['C'])} ({p['C']['verdict']}); "
                 f"single variables that still separate or are marginal on matched data: {', '.join(surv) if surv else 'none'}; "
                 f"variables that separated before matching and no longer do: {', '.join(lost) if lost else 'none'}.")
    lines += ["", "## Method (fixed before the selection results were seen)", "",
              "- Logistic regression, standardised with train statistics, class-balanced, strength chosen by speaker-grouped cross-validation on train only.",
              "- Constant columns are dropped; the sample rate is 16,000 Hz for every recording, so it carries no information "
              f"(dropped from the primary model: {', '.join(p['model_all_train']['constant_columns_dropped']) or 'none'}).",
              f"- Duration matching: 1 s bins, equal genuine and synthetic counts per bin, chosen by seeded hash. Train {m['train']['recordings_full']} -> "
              f"{m['train']['recordings_matched']} recordings; selection {m['selection']['recordings_full']} -> {m['selection']['recordings_matched']}. "
              f"Duration alone has AUC {m['selection']['duration_auc_before']:.3f} on full selection and {m['selection']['duration_auc_after']:.3f} after matching.",
              f"- Intervals: speaker-clustered bootstrap, {N_RESAMPLES} resamples, seed {SEED}, 25 selection speakers. Verdict rule: *separates* if the lower end exceeds "
              f"{SHORTCUT_AUC_FLOOR:.2f}; *marginal* if the estimate reaches it but the lower end does not.", "",
              "## Results: variable sets (ROC-AUC [95% interval], verdict)", "",
              "| Variable set | A: all selection | B: matched selection, model on all train | C: matched selection, model on matched train |", "|---|---|---|---|"]
    for name, entry in result["feature_sets"].items():
        lines.append(f"| {name} | " + " | ".join(f"{_fmt(entry[k])} {entry[k]['verdict']}" for k in "ABC") + " |")
    lines += ["", "## Results: each variable alone", "", "| Variable | Coefficient (A model) | A | B | C |", "|---|---|---|---|---|"]
    for col, entry in sorted(result["single_features"].items(), key=lambda kv: -kv[1]["A"]["roc_auc"]):
        cells = [f"{_fmt(entry[k])} " + ("reversed" if entry[k].get("reversed") else entry[k]["verdict"]) for k in "ABC"]
        lines.append(f"| {col} | {entry['coefficient']:+.2f} | " + " | ".join(cells) + " |")
    flipped = [c for c, e in result["single_features"].items() if e["A"].get("reversed")]
    lines += ["", REVERSAL_NOTE + (f" Reversed on all selection data: {', '.join(flipped)}." if flipped else " No variable reversed.")]
    g = result["supplementary"]["gradient_boosting_all_measurements"]
    top = list(g["importance"].items())[:3]
    lines += ["", "## Supplementary: shallow gradient boosting on all measurements", "",
              "| A | B | C |", "|---|---|---|", "| " + " | ".join(f"{_fmt(g[k])} {g[k]['verdict']}" for k in "ABC") + " |", "",
              "Most used variables: " + ", ".join(f"{k} ({v:.2f})" for k, v in top) + ".", "",
              "## Which generator drives it (AUC, genuine vs each generator, primary model, A)", ""]
    lines.append(", ".join(f"{k} {v:.3f}" for k, v in p["A"]["per_generator_auc"].items()) + ".")
    lines += ["", "## What this means for the detectors", ""]
    if sep_sets or surv:
        lines += ["- Report every B model on **both** the full selection set and the duration-matched selection set; the matched ids are in "
                  f"`data/eda/r2_s3_shortcut/{MATCHED_NAME}`. A model whose score drops sharply on the matched set is partly using these variables.",
                  "- Do not describe a metadata-separable result as speech-quality evidence. Each separating variable should be named in the report's limitations.",
                  "- Candidate remedies to decide at GR2, not applied here: trimming leading and trailing low-energy frames before cropping, and loudness-normalising crops. "
                  "Either changes the locked preprocessing and needs a decision-log entry."]
    else:
        lines += ["- No nuisance variable separated the classes with confidence; keep the duration-matched selection set as a robustness check anyway."]
    lines += ["", "## Limits of this probe", "",
              "- A low AUC shows these variables are not enough on their own; it does not prove the absence of other shortcuts (for example speaker or text effects).",
              "- Variables are measured with the R2-S2 definitions (documented in `eda/measure.py`), not an independent tool.",
              "- 25 selection speakers make the intervals wide; XTTS is a generator unseen in training, so its AUC measures transfer of the shortcut, not fit.",
              "- Matching removes genuine recordings that are longer than almost all synthetic ones, so matched results use fewer genuine recordings."]
    return "\n".join(lines) + "\n"


def make_figure(result: dict, out_base: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    plt.rcParams.update({"svg.hashsalt": "echoproof-r2-s3", "svg.fonttype": "none", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    names = list(result["feature_sets"])
    colors = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}
    labels = {"A": "all selection", "B": "matched, model on all train", "C": "matched, model on matched train"}
    fig, ax = plt.subplots(figsize=(11, 4.8))
    height = 0.25
    for k, key in enumerate("ABC"):
        for i, name in enumerate(names):
            e = result["feature_sets"][name][key]
            y = i + (k - 1) * height
            ax.barh(y, e["roc_auc"] - 0.5, left=0.5, height=height * 0.9, color=colors[key], label=labels[key] if i == 0 else None)
            ax.errorbar(e["roc_auc"], y, xerr=[[e["roc_auc"] - e["auc_ci95"][0]], [e["auc_ci95"][1] - e["roc_auc"]]], color="black", capsize=2, linewidth=1)
    ax.axvline(0.5, color="#444444", linewidth=1)
    ax.axvline(SHORTCUT_AUC_FLOOR, color="#CC0000", linestyle="--", linewidth=1)
    ax.text(SHORTCUT_AUC_FLOOR + 0.005, len(names) - 0.45, f"floor {SHORTCUT_AUC_FLOOR:.2f}", color="#CC0000", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([textwrap.fill(n, 34) for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0.1, 1.0)
    ax.set_xlabel("Selection ROC-AUC of a metadata-only logistic regression (0.5 = chance), 95% speaker-clustered interval")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    p = result["feature_sets"][names[0]]
    top = max(result["feature_sets"].items(), key=lambda kv: kv[1]["A"]["roc_auc"])
    lines = (f"Plan's primary model: AUC {p['A']['roc_auc']:.2f} on all selection, {p['B']['roc_auc']:.2f} / {p['C']['roc_auc']:.2f} after duration matching; best variable set "
             f"('{top[0].split(' (')[0]}') reaches {top[1]['A']['roc_auc']:.2f} / {top[1]['B']['roc_auc']:.2f} / {top[1]['C']['roc_auc']:.2f}.",
             f"Bars whose interval stays right of the dashed {SHORTCUT_AUC_FLOOR:.2f} floor count as separating the classes without any speech content.")
    text = "\n".join(textwrap.fill(line, 120, subsequent_indent="   ") for line in lines)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.text(0.01, 0.012, text, fontsize=8.5, ha="left", va="bottom")
    for ext in ("png", "svg"):
        kwargs = {"dpi": 150} if ext == "png" else {}
        fig.savefig(f"{out_base}.{ext}", metadata={"Software": None} if ext == "png" else {"Date": None}, **kwargs)
    plt.close(fig)
    return {"figure": Path(out_base).name, "files": [Path(out_base).name + ".png", Path(out_base).name + ".svg"], "interpretation": list(lines)}


def run(root, *, train_dir=None, selection_dir=None, report_dir=None, matched_dir=None, n_resamples: int = N_RESAMPLES) -> dict:
    root = Path(root)
    report_dir = Path(report_dir) if report_dir else root / "reports" / "r2"
    matched_dir = Path(matched_dir) if matched_dir else root / "data" / "eda" / "r2_s3_shortcut"
    data = load_data(root, train_dir, selection_dir)
    result = run_probe(data, n_resamples=n_resamples)
    concl = conclusions(result)
    report_dir.mkdir(parents=True, exist_ok=True)
    matched_dir.mkdir(parents=True, exist_ok=True)
    matched = {"task": TASK, "bin_seconds": BIN_SECONDS, "seed": SEED, "train_plan_sha256": data["train_manifest"]["plan_sha256"],
               "selection_scope_sha256": data["selection_manifest"]["scope_sha256"], "ids": result.pop("matched_ids")}
    (matched_dir / MATCHED_NAME).write_text(json.dumps(matched, indent=1, sort_keys=True), encoding="utf-8")
    figure = make_figure(result, report_dir / "r2_s3_shortcut_auc")
    (report_dir / OUTPUT_MEMO).write_text(memo_text(result, concl, data), encoding="utf-8")
    report = {"task": TASK, "settings": {"seed": SEED, "bootstrap_resamples": n_resamples, "auc_floor": SHORTCUT_AUC_FLOOR, "bin_seconds": BIN_SECONDS,
                                         "c_grid": list(C_GRID), "configs": CONFIGS},
              "inputs": {"train_stats_sha256": data["train_manifest"]["outputs"]["stats"]["sha256"],
                         "selection_stats_sha256": data["selection_manifest"]["outputs"]["stats"]["sha256"],
                         "split_sha256": data["train_manifest"]["inputs"]["split_sha256"],
                         "matched_ids_sha256": hashlib.sha256((matched_dir / MATCHED_NAME).read_bytes()).hexdigest()},
              "results": result, "conclusions": concl, "figure": figure}
    (report_dir / OUTPUT_JSON).write_text(json.dumps(report, indent=2, sort_keys=True, default=float), encoding="utf-8")
    return report
