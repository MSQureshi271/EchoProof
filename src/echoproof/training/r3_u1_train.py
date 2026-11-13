"""R3-U1 training, selection and export for the WavLM head on cached embeddings (Protocol B).

Discipline copied from R3-M1: the epoch and the pooling are chosen from selection-role recording macro F1 at fixed 0.5 alone; the choice is frozen in
`selection.json`; only then are the calibration and operating roles scored. Raw uncalibrated scores are exported in the same columns R3-N1 reads for the
BiGRU. Calibration, thresholds and any model comparison beyond the selection role belong to later tasks.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from echoproof.artifacts import digest, file_digest
from echoproof.evaluation.aggregate import aggregate_recordings
from echoproof.evaluation.metrics import compute_metrics
from echoproof.models.wavlm_head import POOLING_DIM, WavLMHead
from echoproof.training.r3_u1_data import POOLINGS, ROLES, EmbeddingData
from echoproof.training.seeding import named_seed

THRESHOLD = 0.5
TASK = "R3-U1"
PANELS = {"known": {"genuine", "melo", "vits"}, "unseen_xtts": {"genuine", "xtts"}}
REQUIRED_TRAIN = {"optimizer": "AdamW", "lr": 0.001, "max_epochs": 20}


class R3U1Error(ValueError):
    pass


def load_recipe(path) -> dict:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("task") != TASK or recipe.get("protocol") != "B" or recipe.get("schema_version") != 1:
        raise R3U1Error("not an R3-U1 Protocol B recipe")
    train = recipe["train"]
    for key, value in REQUIRED_TRAIN.items():
        if train.get(key) != value:
            raise R3U1Error(f"recipe departs from the plan: train.{key} must be {value!r}")
    if recipe.get("variants") != list(POOLINGS) or recipe["model"]["in_dim"] != POOLING_DIM:
        raise R3U1Error("recipe variants or input widths differ from the plan")
    if recipe["selection"]["role"] != "selection" or "0.5" not in recipe["selection"]["metric"]:
        raise R3U1Error("selection must use the selection role at fixed 0.5")
    if train["early_stopping"] is not False or type(train["batch_size"]) is not int or train["batch_size"] < 1:
        raise R3U1Error("invalid training settings")
    return recipe


def recipe_hash(recipe: dict) -> str:
    return digest(recipe)


# ---- scoring and selection metrics -------------------------------------------------------------------------------------------------------

def chunk_scores(model, X: np.ndarray, *, device="cpu", batch=4096) -> np.ndarray:
    import torch
    model.eval()
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out[i:i + batch] = model.scores(torch.from_numpy(X[i:i + batch]).to(device)).double().cpu().numpy()
    if not np.isfinite(out).all() or (out < 0).any() or (out > 1).any():
        raise R3U1Error("model produced invalid probabilities")
    return out


def aggregate(split, probabilities):
    """Arithmetic mean of chunk probabilities per recording; exact recording coverage is enforced."""
    expected = set(split.recordings())
    return aggregate_recordings(split.item_ids, split.recording_ids, split.labels.tolist(), [float(p) for p in probabilities],
                                expected_recording_ids=expected)


def selection_metrics(labels, scores) -> dict:
    return compute_metrics(list(labels), list(scores), THRESHOLD)


def panel_metrics(split, recordings) -> dict:
    comp = {rid: info[1] for rid, info in split.recordings().items()}
    out = {}
    for name, allowed in PANELS.items():
        idx = [i for i, rid in enumerate(recordings.recording_ids) if comp[rid] in allowed]
        out[name] = selection_metrics([recordings.labels[i] for i in idx], [recordings.scores[i] for i in idx])
    return out


def evaluate_selection(model, data: EmbeddingData, *, device="cpu"):
    split = data.roles["selection"]
    recordings = aggregate(split, chunk_scores(model, split.X, device=device))
    return selection_metrics(recordings.labels, recordings.scores), recordings


# ---- training --------------------------------------------------------------------------------------------------------------------------------

def _atomic_torch_save(obj, path: Path):
    import torch
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_variant(data: EmbeddingData, recipe: dict, run_dir, *, owner: str, split_sha256: str, device="cpu", run_id=None,
                  ledger_path=None, progress=None) -> dict:
    """Train one pooling variant for the full epoch budget and keep the best selection epoch. Refuses a non-empty run directory."""
    import torch
    import torch.nn.functional as F
    from echoproof.training.ledger import append_ledger

    if not owner or not isinstance(owner, str):
        raise R3U1Error("an owner is required so runs are not attributed to the wrong person")
    say = progress or (lambda *_: None)
    cfg, mcfg = recipe["train"], recipe["model"]
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise R3U1Error(f"refusing to reuse a non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    pooling = data.pooling
    if data.dim != mcfg["in_dim"][pooling]:
        raise R3U1Error("embedding width differs from the recipe")
    chash = recipe_hash({"recipe": recipe, "pooling": pooling})
    torch.manual_seed(named_seed(cfg["seed"], f"r3_u1/{pooling}/init"))
    model = WavLMHead(data.dim, hidden=mcfg["hidden"], dropout=mcfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    X = torch.from_numpy(data.train.X)
    y = torch.from_numpy(data.train.labels)
    w = torch.from_numpy(data.train.recording_weights if cfg["recording_equal_weights"] else np.ones(len(y), dtype=np.float32))
    shuffle = torch.Generator().manual_seed(named_seed(cfg["seed"], f"r3_u1/{pooling}/shuffle"))
    resolved = {"recipe": recipe, "recipe_sha256": recipe_hash(recipe), "pooling": pooling, "device": device, "torch": torch.__version__,
                "data_identity": data.identity, "train_crops": len(y), "train_recordings": len(data.train.recordings()),
                "selection_recordings": len(data.roles["selection"].recordings()), "config_hash": chash}
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    best = {"macro_f1": -1.0, "epoch": 0}
    history_path = run_dir / "history.jsonl"
    started = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        order = torch.randperm(len(y), generator=shuffle)
        total, seen = 0.0, 0
        for start in range(0, len(y), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            xb, yb, wb = X[idx].to(device), y[idx].to(device), w[idx].to(device)
            loss = (F.cross_entropy(model(xb), yb, reduction="none") * wb).mean()
            if not torch.isfinite(loss):
                raise R3U1Error(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        metrics, recordings = evaluate_selection(model, data, device=device)
        improved = metrics["macro_f1"] > best["macro_f1"]                       # strict: ties keep the earliest epoch
        row = {"epoch": epoch, "train_loss": total / seen, "improved": bool(improved), "seconds": round(time.perf_counter() - t0, 3),
               "val": {k: metrics[k] for k in ("macro_f1", "accuracy", "roc_auc", "eer", "fpr", "fnr", "log_loss") if k in metrics},
               "panels": {k: {m: v[m] for m in ("macro_f1", "roc_auc", "eer", "fpr", "fnr") if m in v}
                          for k, v in panel_metrics(data.roles["selection"], recordings).items()},
               "panels_note": "informational only; never used to choose the epoch"}
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        if improved:
            best = {"macro_f1": metrics["macro_f1"], "epoch": epoch, "metrics": metrics}
            _atomic_torch_save({"epoch": epoch, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()
                                            if isinstance(v, (int, float))},
                                "config": resolved["recipe"], "config_hash": chash, "pooling": pooling,
                                "architecture": model.architecture()}, run_dir / "best.pt")
        say(f"{pooling} epoch {epoch}/{cfg['max_epochs']}: selection macro F1 {metrics['macro_f1']:.4f} (best {best['macro_f1']:.4f} at {best['epoch']})")
    minutes = (time.perf_counter() - started) / 60
    summary = {"pooling": pooling, "best_epoch": best["epoch"], "best_macro_f1": best["macro_f1"], "config_hash": chash, "epochs_run": cfg["max_epochs"],
               "stop_reason": "max_epochs", "best_checkpoint": str(run_dir / "best.pt"), "best_checkpoint_sha256": file_digest(run_dir / "best.pt"),
               "run_dir": str(run_dir), "device": device, "train_minutes": round(minutes, 3), "owner": owner}
    (run_dir / "status.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if ledger_path is not None:
        m = best["metrics"]
        append_ledger(ledger_path, {"run_id": run_id or f"r3_u1_{pooling}", "owner": owner, "model": f"wavlm_head_{pooling}", "protocol": "B",
                                    "config_hash": chash, "split_hash": split_sha256, "seed": cfg["seed"], "epochs_run": cfg["max_epochs"],
                                    "best_epoch": best["epoch"], "stop_reason": "max_epochs", "val_macro_f1": best["macro_f1"],
                                    "val_eer": m["eer"], "val_fpr": m["fpr"], "train_minutes": round(minutes, 3), "peak_vram_mb": "",
                                    "device": device, "status": "completed", "notes": f"pooling={pooling}; draft recipe pending team review"})
    return summary


# ---- selection, freeze and export --------------------------------------------------------------------------------------------------------

def choose_winner(f1_by_pooling: dict) -> str:
    if set(f1_by_pooling) != set(POOLINGS):
        raise R3U1Error("both pooling variants are required")
    scores = {k: float(v) for k, v in f1_by_pooling.items()}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise R3U1Error("invalid selection score")
    return "mean_std" if scores["mean_std"] > scores["mean"] else "mean"       # exact tie: mean


def _write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def finish(output, datas: dict, summaries: dict, recipe: dict, *, split_sha256: str, device="cpu") -> dict:
    """Select from checkpoint metadata alone, freeze it, then score all three validation roles with the winner."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    output = Path(output)
    if set(summaries) != set(POOLINGS) or set(datas) != set(POOLINGS):
        raise R3U1Error("both completed runs and both embedding sets are required")
    checkpoints, f1, hashes = {}, {}, {}
    for pooling, summary in summaries.items():
        path = Path(summary["best_checkpoint"])
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if (ckpt["config_hash"] != summary["config_hash"] or ckpt["epoch"] != summary["best_epoch"] or ckpt["pooling"] != pooling
                or ckpt["metrics"]["macro_f1"] != summary["best_macro_f1"] or summary["stop_reason"] != "max_epochs"):
            raise R3U1Error(f"inconsistent or unfinished checkpoint: {pooling}")
        if file_digest(path) != summary["best_checkpoint_sha256"]:
            raise R3U1Error(f"checkpoint changed since training: {pooling}")
        checkpoints[pooling], f1[pooling], hashes[pooling] = ckpt, ckpt["metrics"]["macro_f1"], summary["best_checkpoint_sha256"]
    winner = choose_winner(f1)
    selection = {"task": TASK, "protocol": "B", "selected": winner, "selection_threshold": THRESHOLD, "selection_macro_f1": f1,
                 "checkpoint_sha256": hashes, "selected_epoch": summaries[winner]["best_epoch"], "split_sha256": split_sha256,
                 "recipe_sha256": recipe_hash(recipe), "tie_rule": "mean pooling", "recipe_status": recipe.get("status")}
    selection_path = output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise R3U1Error("existing selection differs; refusing to overwrite")
    else:
        _write_json(selection_path, selection)
    data = datas[winner]
    mcfg = recipe["model"]
    model = WavLMHead(data.dim, hidden=mcfg["hidden"], dropout=mcfg["dropout"])
    model.load_state_dict(checkpoints[winner]["model"], strict=True)
    model.to(device).eval()
    products, counts, panels = {}, {}, None
    for role in ROLES:                                              # the winner was frozen above, before these roles were scored
        split = data.roles[role]
        probs = chunk_scores(model, split.X, device=device)
        recordings = aggregate(split, probs)
        info = split.recordings()
        if role == "selection":
            actual = selection_metrics(recordings.labels, recordings.scores)["macro_f1"]
            if abs(actual - f1[winner]) > 1e-12:
                raise R3U1Error("reloaded checkpoint does not reproduce the saved selection metric")
            panels = panel_metrics(split, recordings)
        chunk_table = pa.Table.from_pylist([{"chunk_id": cid, "recording_id": rid, "group_id": gid, "label": int(label), "component": comp,
                                             "validation_role": role, "tts_probability": float(p)}
                                            for cid, rid, gid, comp, label, p in zip(split.item_ids, split.recording_ids, split.group_ids,
                                                                                     split.components, split.labels.tolist(), probs)])
        rec_table = pa.Table.from_pylist([{"recording_id": rid, "group_id": info[rid][2], "label": int(label), "component": info[rid][1],
                                           "validation_role": role, "raw_score": float(score), "chunk_count": int(n)}
                                          for rid, label, score, n in zip(recordings.recording_ids, recordings.labels, recordings.scores,
                                                                          recordings.chunk_counts)])
        for kind, table in (("chunks", chunk_table), ("recordings", rec_table)):
            path = output / f"predictions_{kind}_{role}.parquet"
            tmp = path.with_suffix(".parquet.tmp")
            pq.write_table(table, tmp)
            os.replace(tmp, path)
            products[path.name] = file_digest(path)
        counts[role] = {"chunks": len(split), "recordings": len(info)}
    evidence = {"task": TASK, "status": "validation_predictions_complete", "selection": selection, "selection_sha256": file_digest(selection_path),
                "output_sha256": products, "counts": counts, "selection_panels": panels, "score_type": "raw uncalibrated TTS score",
                "data_identity": data.identity, "note": "Calibration and operating predictions are not selection metrics. Raw scores need R3-N1."}
    _write_json(output / "evidence.json", evidence)
    return evidence
