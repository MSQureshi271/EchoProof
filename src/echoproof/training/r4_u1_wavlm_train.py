"""R4-U1a training, selection and export for the WavLM head on the Protocol A embeddings.

Same discipline and numerics as `r3_u1_train` (a frozen B source, hard-coded to Protocol B): the epoch is chosen from selection-role recording macro F1 at
fixed 0.5 alone, the choice is frozen in `selection.json`, and only then are the calibration and operating roles scored. Only the B-selected mean_std
variant is trained, so there is no pooling choice. Selection-role panels are per component, for information only. Raw scores are exported uncalibrated in the
columns R4-N1 reads. The generic scoring helpers are imported from `r3_u1_train`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from echoproof.artifacts import digest, file_digest
from echoproof.models.wavlm_head import WavLMHead
from echoproof.training.r3_u1_train import (THRESHOLD, _atomic_torch_save, _write_json, aggregate, chunk_scores, selection_metrics)
from echoproof.training.r4_u1_wavlm_data import POOLING, ROLES, EmbeddingData
from echoproof.training.seeding import named_seed

TASK = "R4-U1a"
B_RECIPE = "configs/r3_u1_wavlm_head_b.json"
PANEL_METRICS = ("macro_f1", "roc_auc", "eer", "fpr", "fnr")


class R4U1Error(ValueError):
    pass


def recipe_hash(recipe: dict) -> str:
    return digest(recipe)


def load_recipe(path, *, root=None, require_agreed: bool = False) -> dict:
    """The A recipe must equal the B recipe's model and train blocks. `require_agreed` (used by `train`) also needs the team's reported agreement."""
    path = Path(path)
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if recipe.get("task") != TASK or recipe.get("protocol") != "A" or recipe.get("schema_version") != 1:
        raise R4U1Error("not an R4-U1a Protocol A recipe")
    root = Path(root) if root is not None else path.resolve().parents[1]
    b_path = root / B_RECIPE
    derived = recipe.get("derived_from", {})
    if derived.get("file") != B_RECIPE or file_digest(b_path) != derived.get("sha256"):
        raise R4U1Error("the B recipe is missing or differs from the one this recipe was derived from")
    b = json.loads(b_path.read_text(encoding="utf-8"))
    if b.get("protocol") != "B" or recipe["train"] != b["train"] or recipe["model"] != b["model"]:
        raise R4U1Error("the A recipe departs from the B-selected configuration (model or train block differs)")
    if recipe.get("variants") != [POOLING] or recipe["selection"]["role"] != "selection" or "0.5" not in recipe["selection"]["metric"]:
        raise R4U1Error("Protocol A trains mean_std only and selects on the selection role at fixed 0.5")
    if require_agreed and not str(recipe.get("status", "")).startswith("team_agreed"):
        raise R4U1Error(f"recipe status {recipe.get('status')!r}: training needs the team's reported agreement recorded in the status field")
    return recipe


def component_panels(split, recordings) -> dict:
    """Recording-level metrics of genuine versus each generator on the selection role. Informational only."""
    comp = {rid: info[1] for rid, info in split.recordings().items()}
    out = {}
    for generator in sorted({c for c in comp.values() if c != "genuine"}):
        idx = [i for i, rid in enumerate(recordings.recording_ids) if comp[rid] in ("genuine", generator)]
        m = selection_metrics([recordings.labels[i] for i in idx], [recordings.scores[i] for i in idx])
        out["genuine_vs_" + generator] = {k: m[k] for k in PANEL_METRICS if k in m}
    return out


def evaluate_selection(model, data: EmbeddingData, *, device="cpu"):
    split = data.roles["selection"]
    recordings = aggregate(split, chunk_scores(model, split.X, device=device))
    return selection_metrics(recordings.labels, recordings.scores), recordings


def train(data: EmbeddingData, recipe: dict, run_dir, *, owner: str, split_sha256: str, device="cpu", run_id=None, ledger_path=None,
          progress=None) -> dict:
    """Train mean_std for the full epoch budget and keep the best selection epoch. Refuses a non-empty run directory."""
    import torch
    import torch.nn.functional as F
    from echoproof.training.ledger import append_ledger

    if not owner or not isinstance(owner, str):
        raise R4U1Error("an owner is required so runs are not attributed to the wrong person")
    if data.pooling != POOLING:
        raise R4U1Error("Protocol A trains mean_std only")
    say = progress or (lambda *_: None)
    cfg, mcfg = recipe["train"], recipe["model"]
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise R4U1Error(f"refusing to reuse a non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if data.dim != mcfg["in_dim"][POOLING]:
        raise R4U1Error("embedding width differs from the recipe")
    chash = recipe_hash({"recipe": recipe, "pooling": POOLING})
    torch.manual_seed(named_seed(cfg["seed"], f"r4_u1/{POOLING}/init"))
    model = WavLMHead(data.dim, hidden=mcfg["hidden"], dropout=mcfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    X, y = torch.from_numpy(data.train.X), torch.from_numpy(data.train.labels)
    w = torch.from_numpy(data.train.recording_weights if cfg["recording_equal_weights"] else np.ones(len(y), dtype=np.float32))
    shuffle = torch.Generator().manual_seed(named_seed(cfg["seed"], f"r4_u1/{POOLING}/shuffle"))
    resolved = {"recipe": recipe, "recipe_sha256": recipe_hash(recipe), "pooling": POOLING, "device": device, "torch": torch.__version__,
                "data_identity": data.identity, "train_crops": len(y), "train_recordings": len(data.train.recordings()),
                "selection_recordings": len(data.roles["selection"].recordings()), "config_hash": chash}
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    best = {"macro_f1": -1.0, "epoch": 0}
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
                raise R4U1Error(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        metrics, recordings = evaluate_selection(model, data, device=device)
        improved = metrics["macro_f1"] > best["macro_f1"]                       # strict: ties keep the earliest epoch
        row = {"epoch": epoch, "train_loss": total / seen, "improved": bool(improved), "seconds": round(time.perf_counter() - t0, 3),
               "val": {k: metrics[k] for k in ("macro_f1", "accuracy", "roc_auc", "eer", "fpr", "fnr", "log_loss") if k in metrics},
               "panels": component_panels(data.roles["selection"], recordings), "panels_note": "informational only; never used to choose the epoch"}
        with open(run_dir / "history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        if improved:
            best = {"macro_f1": metrics["macro_f1"], "epoch": epoch, "metrics": metrics}
            _atomic_torch_save({"epoch": epoch, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                                "config": resolved["recipe"], "config_hash": chash, "pooling": POOLING, "architecture": model.architecture()},
                               run_dir / "best.pt")
        say(f"epoch {epoch}/{cfg['max_epochs']}: selection macro F1 {metrics['macro_f1']:.4f} (best {best['macro_f1']:.4f} at {best['epoch']})")
    minutes = (time.perf_counter() - started) / 60
    summary = {"pooling": POOLING, "best_epoch": best["epoch"], "best_macro_f1": best["macro_f1"], "config_hash": chash, "epochs_run": cfg["max_epochs"],
               "stop_reason": "max_epochs", "best_checkpoint": str(run_dir / "best.pt"), "best_checkpoint_sha256": file_digest(run_dir / "best.pt"),
               "run_dir": str(run_dir), "device": device, "train_minutes": round(minutes, 3), "owner": owner}
    (run_dir / "status.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if ledger_path is not None:
        m = best["metrics"]
        append_ledger(ledger_path, {"run_id": run_id or "r4_u1_wavlm_head_A", "owner": owner, "model": "wavlm_head_mean_std", "protocol": "A",
                                    "config_hash": chash, "split_hash": split_sha256, "seed": cfg["seed"], "epochs_run": cfg["max_epochs"],
                                    "best_epoch": best["epoch"], "stop_reason": "max_epochs", "val_macro_f1": best["macro_f1"],
                                    "val_eer": m["eer"], "val_fpr": m["fpr"], "train_minutes": round(minutes, 3), "peak_vram_mb": "",
                                    "device": device, "status": "completed",
                                    "notes": f"pooling={POOLING}; B-selected configuration; recipe status {recipe.get('status')}"})
    return summary


def finish(output, data: EmbeddingData, summary: dict, recipe: dict, *, split_sha256: str, device="cpu") -> dict:
    """Freeze the selection from checkpoint metadata alone, then score all three validation roles with the selected head."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    output = Path(output)
    path = Path(summary["best_checkpoint"])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if (ckpt["config_hash"] != summary["config_hash"] or ckpt["epoch"] != summary["best_epoch"] or ckpt["pooling"] != POOLING
            or ckpt["metrics"]["macro_f1"] != summary["best_macro_f1"] or summary["stop_reason"] != "max_epochs"):
        raise R4U1Error("inconsistent or unfinished checkpoint")
    if file_digest(path) != summary["best_checkpoint_sha256"]:
        raise R4U1Error("checkpoint changed since training")
    f1 = ckpt["metrics"]["macro_f1"]
    selection = {"task": TASK, "protocol": "A", "selected": POOLING, "selection_threshold": THRESHOLD, "selection_macro_f1": {POOLING: f1},
                 "checkpoint_sha256": {POOLING: summary["best_checkpoint_sha256"]}, "selected_epoch": summary["best_epoch"],
                 "split_sha256": split_sha256, "recipe_sha256": recipe_hash(recipe), "recipe_status": recipe.get("status"),
                 "selection_rule": "best selection-role recording macro F1 at 0.5, earliest epoch on ties; mean_std fixed by the B selection"}
    selection_path = output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise R4U1Error("existing selection differs; refusing to overwrite")
    else:
        _write_json(selection_path, selection)
    mcfg = recipe["model"]
    model = WavLMHead(data.dim, hidden=mcfg["hidden"], dropout=mcfg["dropout"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    products, counts, panels = {}, {}, None
    for role in ROLES:                                              # the selection was frozen above, before these roles were scored
        split = data.roles[role]
        probs = chunk_scores(model, split.X, device=device)
        recordings = aggregate(split, probs)
        info = split.recordings()
        if role == "selection":
            if abs(selection_metrics(recordings.labels, recordings.scores)["macro_f1"] - f1) > 1e-12:
                raise R4U1Error("reloaded checkpoint does not reproduce the saved selection metric")
            panels = component_panels(split, recordings)
        chunk_table = pa.Table.from_pylist([{"chunk_id": cid, "recording_id": rid, "group_id": gid, "label": int(label), "component": comp,
                                             "validation_role": role, "tts_probability": float(p)}
                                            for cid, rid, gid, comp, label, p in zip(split.item_ids, split.recording_ids, split.group_ids,
                                                                                     split.components, split.labels.tolist(), probs)])
        rec_table = pa.Table.from_pylist([{"recording_id": rid, "group_id": info[rid][2], "label": int(label), "component": info[rid][1],
                                           "validation_role": role, "raw_score": float(score), "chunk_count": int(n)}
                                          for rid, label, score, n in zip(recordings.recording_ids, recordings.labels, recordings.scores,
                                                                          recordings.chunk_counts)])
        for kind, table in (("chunks", chunk_table), ("recordings", rec_table)):
            out = output / f"predictions_{kind}_{role}.parquet"
            tmp = out.with_suffix(".parquet.tmp")
            pq.write_table(table, tmp)
            os.replace(tmp, out)
            products[out.name] = file_digest(out)
        counts[role] = {"chunks": len(split), "recordings": len(info)}
    evidence = {"task": TASK, "status": "validation_predictions_complete", "selection": selection, "selection_sha256": file_digest(selection_path),
                "output_sha256": products, "counts": counts, "selection_panels": panels, "score_type": "raw uncalibrated TTS score",
                "data_identity": data.identity, "note": "Calibration and operating predictions are not selection metrics. Raw scores need R4-N1."}
    _write_json(output / "evidence.json", evidence)
    return evidence
