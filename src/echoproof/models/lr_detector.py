"""R1-M2 production Logistic Regression: imputer -> scaler -> LR, with an eight-trial grid runner.

The fitted pipeline is stored as plain float64 arrays (weights.npz + lr_model.json) and scored with NumPy, so a saved model
needs neither pickle (no code execution on load) nor a matching scikit-learn version, and reload parity is exact.
Feature choice (recorded in every saved model): the existing 67-column schema in `echoproof.features.acoustic` -
13 MFCC and 13 delta-MFCC (mean, std), six spectral/energy families (mean, std) and three voicing columns.
The plan's 20-MFCC starting schema and an explicit voicing-validity flag are not used: voiced_fraction == 0 already marks
"no voiced frames" and the NaN F0 statistics are median-imputed from training data only.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from echoproof.artifacts import digest, file_digest
from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH, handcrafted

LR_SCHEMA = {"name": "echoproof_lr_detector", "version": 1}
GRID_C = (0.01, 0.1, 1.0, 10.0)
GRID_CLASS_WEIGHT = ("none", "balanced")
_ARRAYS = ("medians", "mean", "scale", "coef", "intercept")


class LRModelError(ValueError):
    """Invalid training data or an incompatible/tampered saved LR model."""


def recording_weights(recording_ids, labels, class_weight: str = "none"):
    """Per-chunk sample weights: every recording carries equal total weight, then optional class balancing.

    Each recording's chunks sum to (chunks / recordings), so the total weight equals the chunk count and C keeps the meaning
    it has for an unweighted fit. class_weight='balanced' multiplies each recording by N / (2 * N_class) with N counted in
    *recordings* (scikit-learn's own 'balanced' would count chunks and bring duration bias back); the total is unchanged.
    Labels must be constant within a recording.
    """
    if class_weight not in GRID_CLASS_WEIGHT:
        raise LRModelError("class_weight must be 'none' or 'balanced'")
    ids, y = np.asarray(recording_ids), np.asarray(labels)
    if ids.ndim != 1 or y.shape != ids.shape or len(ids) == 0:
        raise LRModelError("recording_ids and labels must be equal-length, non-empty 1-D arrays")
    uniq, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    recording_label = np.zeros(len(uniq), dtype=np.int64)
    recording_label[inverse] = y
    if (recording_label[inverse] != y).any():
        raise LRModelError("a recording has chunks with different labels")
    weights = (len(ids) / len(uniq)) / counts[inverse]
    if class_weight == "balanced":
        per_class = np.bincount(recording_label, minlength=2)
        if (per_class == 0).any():
            raise LRModelError("balanced class weights require both classes")
        weights = weights * (len(uniq) / (2.0 * per_class[recording_label[inverse]]))
    return weights.astype(np.float64)


def chunk_features(waveforms):
    """Stack the handcrafted vector for each chunk waveform (NaN is allowed for unvoiced F0 statistics)."""
    return np.stack([handcrafted(np.asarray(w, dtype=np.float32)) for w in waveforms])


def _sigmoid(z):
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    e = np.exp(z[~positive])
    out[~positive] = e / (1.0 + e)
    return out


class LRDetector:
    """Fitted imputer + scaler + binary LR held as arrays. Output is P(TTS), the class-1 probability (classes_ == [0, 1])."""

    def __init__(self, arrays, meta):
        self.arrays = {k: np.asarray(arrays[k], dtype=np.float64) for k in _ARRAYS}
        self.meta = dict(meta)
        self._validate()

    # ---- fitting ------------------------------------------------------------------------------------------------
    @staticmethod
    def fit_pipeline(X, y, recording_ids, *, C: float = 1.0, class_weight: str = "none", max_iter: int = 2000, seed: int = 42):
        """Fit the scikit-learn pipeline on training data only; returns (pipeline, diagnostics)."""
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or len(X) != len(y):
            raise LRModelError(f"expected an [n, {len(FEATURE_NAMES)}] feature matrix with one label per row")
        if np.isinf(X).any():
            raise LRModelError("features contain infinities")
        if set(np.unique(y).tolist()) != {0, 1}:
            raise LRModelError("training requires both classes (0=genuine, 1=tts)")
        weights = recording_weights(recording_ids, y, class_weight)
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("scaler", StandardScaler()),
                             ("classifier", LogisticRegression(C=C, max_iter=max_iter, random_state=seed))])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X, y, scaler__sample_weight=weights, classifier__sample_weight=weights)
        classifier = pipeline.named_steps["classifier"]
        if list(classifier.classes_) != [0, 1]:
            raise LRModelError("unexpected class order")
        n_iter = int(np.max(classifier.n_iter_))
        converged = n_iter < max_iter and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        return pipeline, {"converged": bool(converged), "n_iter": n_iter, "max_iter": max_iter, "chunks": int(len(y)),
                          "recordings": int(len(np.unique(np.asarray(recording_ids)))), "weight_sum": float(weights.sum())}

    @classmethod
    def from_pipeline(cls, pipeline, diagnostics, *, C, class_weight, seed=42, extra=None):
        classifier = pipeline.named_steps["classifier"]
        if list(classifier.classes_)[1] != 1:  # a binary coef_ describes classes_[1]; check rather than assume
            raise LRModelError("class 1 must be the positive column")
        scaler = pipeline.named_steps["scaler"]
        arrays = {"medians": pipeline.named_steps["imputer"].statistics_, "mean": scaler.mean_,
                  "scale": np.where(np.asarray(scaler.scale_) == 0, 1.0, scaler.scale_),
                  "coef": classifier.coef_[0], "intercept": classifier.intercept_}
        meta = {"schema": LR_SCHEMA, "feature_names": list(FEATURE_NAMES), "feature_schema_hash": FEATURE_SCHEMA_HASH,
                "label_map": {"genuine": 0, "tts": 1},
                "hyperparameters": {"C": C, "class_weight": class_weight, "max_iter": diagnostics["max_iter"], "seed": seed},
                "diagnostics": diagnostics, "weighting": "equal weight per recording; total weight = chunk count",
                "feature_choice": "13 MFCC + 13 delta (mean,std) + 6 spectral/energy families (mean,std) + 3 voicing columns = 67",
                **(extra or {})}
        return cls(arrays, meta)

    @classmethod
    def fit(cls, X, y, recording_ids, *, C: float = 1.0, class_weight: str = "none", max_iter: int = 2000, seed: int = 42, extra=None):
        pipeline, diagnostics = cls.fit_pipeline(X, y, recording_ids, C=C, class_weight=class_weight, max_iter=max_iter, seed=seed)
        return cls.from_pipeline(pipeline, diagnostics, C=C, class_weight=class_weight, seed=seed, extra=extra)

    # ---- validation and inference -------------------------------------------------------------------------------
    def _validate(self):
        n = len(FEATURE_NAMES)
        for key, shape in {"medians": (n,), "mean": (n,), "scale": (n,), "coef": (n,), "intercept": (1,)}.items():
            if self.arrays[key].shape != shape or not np.isfinite(self.arrays[key]).all():
                raise LRModelError(f"invalid or non-finite array: {key}")
        if (self.arrays["scale"] <= 0).any():
            raise LRModelError("scale must be positive")
        if self.meta.get("schema") != LR_SCHEMA or self.meta.get("feature_schema_hash") != FEATURE_SCHEMA_HASH \
                or list(self.meta.get("feature_names", ())) != list(FEATURE_NAMES):
            raise LRModelError("incompatible LR model or feature schema")
        if self.meta.get("label_map") != {"genuine": 0, "tts": 1}:
            raise LRModelError("unexpected label map")

    @property
    def metadata(self):
        return MappingProxyType({"scope": "production", "model": "logistic_regression", "feature_schema_hash": FEATURE_SCHEMA_HASH,
                                 "hyperparameters": json.dumps(self.meta["hyperparameters"], sort_keys=True)})

    def predict_proba_features(self, X):
        """P(TTS) per feature row; NaN is replaced by the training median, as the fitted imputer does."""
        X = np.array(X, dtype=np.float64, ndmin=2)
        if X.shape[1] != len(FEATURE_NAMES) or np.isinf(X).any():
            raise LRModelError("feature matrix has the wrong width or contains infinities")
        a = self.arrays
        X = np.where(np.isnan(X), a["medians"], X)
        return _sigmoid(((X - a["mean"]) / a["scale"]) @ a["coef"] + a["intercept"][0])

    def score_chunks(self, waveforms):
        """P(TTS) per chunk waveform (mono 16 kHz float, true valid samples only)."""
        return self.predict_proba_features(chunk_features(waveforms))

    def coefficient_table(self):
        """(feature, standardized coefficient) by decreasing magnitude: an association, not a causal explanation."""
        return sorted(zip(FEATURE_NAMES, self.arrays["coef"].tolist()), key=lambda p: (-abs(p[1]), p[0]))

    # ---- persistence: weights.npz (no pickle) + lr_model.json ----------------------------------------------------
    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        weights, meta = directory / "weights.npz", directory / "lr_model.json"
        if weights.exists() or meta.exists():
            raise LRModelError(f"refusing to overwrite an existing LR model in {directory}")
        np.savez(weights, **self.arrays)
        meta.write_text(json.dumps(self.meta, indent=2, sort_keys=True), encoding="utf-8")
        return {"weights.npz": file_digest(weights), "lr_model.json": file_digest(meta)}

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        try:
            with np.load(directory / "weights.npz", allow_pickle=False) as data:
                if set(data.files) != set(_ARRAYS):
                    raise LRModelError("unexpected arrays in weights.npz")
                arrays = {k: data[k] for k in _ARRAYS}
            meta = json.loads((directory / "lr_model.json").read_text(encoding="utf-8"))
        except LRModelError:
            raise
        except (OSError, ValueError) as exc:
            raise LRModelError(f"cannot read LR model: {exc}") from exc
        return cls(arrays, meta)


# ---- eight-trial grid runner ---------------------------------------------------------------------------------------

def aggregate_recordings(recording_ids, labels, scores):
    """Mean chunk probability per recording, in first-seen order -> (labels, scores)."""
    index, sums, counts, out_labels = {}, [], [], []
    for rid, label, score in zip(recording_ids, labels, scores):
        if rid not in index:
            index[rid] = len(index)
            sums.append(0.0)
            counts.append(0)
            out_labels.append(int(label))
        sums[index[rid]] += float(score)
        counts[index[rid]] += 1
    return out_labels, [s / c for s, c in zip(sums, counts)]


def _rank_key(trial):
    """Higher macro F1, then lower EER, lower FPR, then grid order (lower C first, 'none' before 'balanced')."""
    def last_if_nan(value):
        return float("inf") if value is None or value != value else value
    m = trial["metrics"]
    return (-m["macro_f1"], last_if_nan(m["eer"]), last_if_nan(m["fpr"]), trial["order"])


def run_lr_grid(train, selection, *, grid_c=GRID_C, grid_class_weight=GRID_CLASS_WEIGHT, max_iter=2000, seed=42, threshold=0.5,
                protocol="", split_hash=None, run_prefix="lr", ledger_path=None, owner="", metrics_fn=None):
    """Fit every (C, class_weight) trial on `train`, score `selection` per recording, pick one, ledger every trial.

    train / selection: dicts with X [n, 67], y [n] (the recording's label for each chunk) and recording_ids [n].
    Selection is recording-level macro F1 at a fixed threshold (never tuned). A failed trial is ledgered, not hidden.
    metrics_fn(labels, scores, threshold) defaults to the neural trainer's recording_metrics (imports torch lazily).
    """
    if metrics_fn is None:
        from echoproof.training.trainer import recording_metrics as metrics_fn
    from echoproof.training.ledger import append_ledger
    trials = []
    for order, (c, cw) in enumerate((c, cw) for c in grid_c for cw in grid_class_weight):
        trial = {"order": order, "C": c, "class_weight": cw, "run_id": f"{run_prefix}-t{order}"}
        config_hash = digest({"model": "lr", "C": c, "class_weight": cw, "max_iter": max_iter, "seed": seed,
                              "protocol": protocol, "split": split_hash})
        try:
            model = LRDetector.fit(train["X"], train["y"], train["recording_ids"], C=c, class_weight=cw, max_iter=max_iter,
                                   seed=seed, extra={"protocol": protocol, "split_sha256": split_hash})
            labels, scores = aggregate_recordings(selection["recording_ids"], selection["y"], model.predict_proba_features(selection["X"]))
            trial.update(model=model, metrics=metrics_fn(labels, scores, threshold), diagnostics=model.meta["diagnostics"], status="completed")
        except Exception as exc:  # noqa: BLE001 - a failed trial is evidence too
            trial.update(model=None, metrics=None, diagnostics=None, status=f"failed:{type(exc).__name__}", error=str(exc)[:200])
        if ledger_path:
            m, d = trial["metrics"] or {}, trial["diagnostics"] or {}
            append_ledger(ledger_path, {"run_id": trial["run_id"], "owner": owner, "model": "lr", "protocol": protocol,
                                        "config_hash": config_hash, "split_hash": split_hash, "seed": seed, "epochs_run": d.get("n_iter"),
                                        "val_macro_f1": m.get("macro_f1"), "val_eer": m.get("eer"), "val_fpr": m.get("fpr"),
                                        "device": "cpu", "status": trial["status"],
                                        "notes": f"C={c};class_weight={cw};converged={d.get('converged')};error={trial.get('error', '')}"})
        trials.append(trial)
    completed = [t for t in trials if t["status"] == "completed"]
    if not completed:
        raise LRModelError("every grid trial failed")
    best = min(completed, key=_rank_key)
    return {"trials": trials, "selected": best["order"], "selected_model": best["model"],
            "selected_config": {"C": best["C"], "class_weight": best["class_weight"]}}
