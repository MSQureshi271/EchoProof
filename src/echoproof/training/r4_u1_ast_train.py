"""R4-U1b: AST-A, a fresh Protocol A head-only warm-up followed by the B-selected partial fine-tune, with the adaptation evidence.

The shared Trainer runs the loop, fp16 scaling, schedules, resume and ledger rows, exactly as for R2-U2 and R3-U2. `r3_u2_train` is a frozen B source
(Protocol B recipe, two experiments, B panels, protocol "B" in its configs), so what is generic is imported (load_warmup, check_partial_stage,
gradient_proof, proof_batch, state_group_digests, score_amp, adaptation_evidence) and what is bound to B is restated here:
  * both recipes must equal their B counterparts (the warm-up and the baseline fine-tune); the one allowed difference is `warmup_steps`;
  * the warm-up trains the classifier head only on A and records `warmup_result.json`, the initializer and epoch-0 reference of the fine-tune;
  * the fine-tune proves its initializer (SHA-256, pretrained groups), the partial stage, non-zero gradients and changed checksums exactly in the
    unfrozen groups, reproduces the warm-up's selection macro F1 at epoch 0 (within 0.003), and checks the checksums again after training;
  * the selection is frozen before the calibration and operating roles are scored; exports use the BiGRU column layout, raw and uncalibrated;
  * panels are per component (genuine versus each generator), informational only, since A has no unseen generator.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

from echoproof.artifacts import digest, file_digest
from echoproof.training import r3_u2_train as R3

WARMUP_TASK = "R4-U1b-warmup"
TASK = "R4-U1b"
ROLES = ("selection", "calibration", "operating")
PARITY_ATOL = 1e-3
B_WARMUP = "configs/r2_u2_ast_b.json"
B_FINETUNE = "configs/r3_u2_ast_ft_b.json"
A_STEPS_FORMULA = {"crops": 7550}
PANEL_METRICS = ("macro_f1", "roc_auc", "eer", "fpr", "fnr")


class R4U1Error(ValueError):
    pass


def recipe_hash(recipe: dict) -> str:
    return digest(recipe)


def _require_b(root, rel, derived):
    path = Path(root) / rel
    if derived.get("file") != rel or not path.is_file() or file_digest(path) != derived.get("sha256"):
        raise R4U1Error(f"the B recipe {rel} is missing or differs from the one this recipe was derived from")
    return json.loads(path.read_text(encoding="utf-8"))


def warmup_steps_for(crops: int, train: dict) -> int:
    """About 5 percent of the optimiser steps (the B proportion), by the Trainer's own step arithmetic."""
    steps = math.ceil(math.ceil(crops / train["batch_size"]) / train["accumulation_steps"]) * train["max_epochs"]
    return round(0.05 * steps)


def load_warmup_recipe(path, *, root=None, require_agreed=False) -> dict:
    path = Path(path)
    recipe = json.loads(path.read_text(encoding="utf-8"))
    root = Path(root) if root is not None else path.resolve().parents[1]
    if recipe.get("task") != WARMUP_TASK or recipe.get("protocol") != "A" or recipe.get("schema_version") != 1:
        raise R4U1Error("not an R4-U1b warm-up Protocol A recipe")
    b = _require_b(root, B_WARMUP, recipe.get("derived_from", {}))
    b_train = {k: v for k, v in b["train"].items() if k not in ("owner", "model_name", "protocol")}
    if recipe["train"] != b_train:
        raise R4U1Error("the A warm-up train block departs from the B R2-U2 recipe")
    if recipe["model"]["stage"] != "head" or recipe["selection"]["role"] != "selection" or "0.5" not in recipe["selection"]["metric"]:
        raise R4U1Error("the warm-up trains the head only and selects on the selection role at fixed 0.5")
    if require_agreed and not str(recipe.get("status", "")).startswith("team_agreed"):
        raise R4U1Error(f"recipe status {recipe.get('status')!r}: training needs the team's reported agreement recorded in the status field")
    return recipe


def load_finetune_recipe(path, *, root=None, require_agreed=False) -> dict:
    path = Path(path)
    recipe = json.loads(path.read_text(encoding="utf-8"))
    root = Path(root) if root is not None else path.resolve().parents[1]
    if recipe.get("task") != TASK or recipe.get("protocol") != "A" or recipe.get("schema_version") != 1:
        raise R4U1Error("not an R4-U1b Protocol A recipe")
    b = _require_b(root, B_FINETUNE, recipe.get("derived_from", {}))
    differs = {k for k in set(b["train"]) | set(recipe["train"]) if b["train"].get(k) != recipe["train"].get(k)}
    if differs - {"warmup_steps"}:
        raise R4U1Error(f"the A fine-tune train block departs from the B recipe: {sorted(differs - {'warmup_steps'})}")
    if recipe["train"]["warmup_steps"] != warmup_steps_for(A_STEPS_FORMULA["crops"], recipe["train"]):
        raise R4U1Error("warmup_steps must be 5 percent of the A optimiser steps (the B proportion)")
    if recipe["model"] != b["model"] or recipe["selection"]["role"] != "selection" or "0.5" not in recipe["selection"]["metric"]:
        raise R4U1Error("the model block departs from B, or selection is not the selection role at fixed 0.5")
    if set(recipe["experiments"]) != {"baseline"} or recipe["experiments"]["baseline"] != b["experiments"]["baseline"]:
        raise R4U1Error("Protocol A runs the B-selected baseline experiment only")
    if recipe["init"]["pretrained_group_sha256"] != b["init"]["pretrained_group_sha256"] or len(recipe["init"]["pretrained_group_sha256"]) != 14:
        raise R4U1Error("the pretrained group checksums differ from the pinned B values")
    if require_agreed and not str(recipe.get("status", "")).startswith("team_agreed"):
        raise R4U1Error(f"recipe status {recipe.get('status')!r}: training needs the team's reported agreement recorded in the status field")
    return recipe


def warmup_train_config(recipe: dict, *, owner: str, overrides: dict | None = None):
    from echoproof.training.trainer import TrainConfig
    if not owner or not isinstance(owner, str):
        raise R4U1Error("an owner is required so runs are not attributed to the wrong person")
    cfg = TrainConfig(**recipe["train"], owner=owner, model_name="ast_frozen", protocol="A")
    return replace(cfg, **overrides) if overrides else cfg


def finetune_train_config(recipe: dict, *, owner: str, overrides: dict | None = None):
    from echoproof.training.trainer import TrainConfig
    if not owner or not isinstance(owner, str):
        raise R4U1Error("an owner is required so runs are not attributed to the wrong person")
    t, e = recipe["train"], recipe["experiments"]["baseline"]
    cfg = TrainConfig(seed=t["seed"], max_epochs=t["max_epochs"], batch_size=t["batch_size"], accumulation_steps=t["accumulation_steps"],
                      eval_batch_size=t["eval_batch_size"], lr=e["head_lr"], weight_decay=t["weight_decay"], grad_clip=t["grad_clip"],
                      scheduler=t["scheduler"], warmup_steps=t["warmup_steps"], patience=t["patience"], min_delta=t["min_delta"],
                      selection_threshold=t["selection_threshold"], precision=t["precision"], class_strategy=t["class_strategy"],
                      num_workers=t["num_workers"], device=t["device"], deterministic=t["deterministic"], owner=owner,
                      model_name="ast_partial", protocol="A")
    return replace(cfg, **overrides) if overrides else cfg


def component_panels(recordings, components) -> dict:
    """Recording-level metrics of genuine versus each generator. Informational only. `recordings` rows are (id, label, score, chunks)."""
    from echoproof.training.r3_m1_selection import selection_metrics
    out = {}
    for generator in sorted({components[r[0]] for r in recordings} - {"genuine"}):
        rows = [r for r in recordings if components[r[0]] in ("genuine", generator)]
        m = selection_metrics([r[1] for r in rows], [r[2] for r in rows], 0.5)
        out["genuine_vs_" + generator] = {k: m[k] for k in PANEL_METRICS if k in m}
    return out


def _write_json(path, value):
    R3._write_json(path, value)


# ---- warm-up --------------------------------------------------------------------------------------------------------------------------------

def run_warmup(recipe, data, *, snapshot, run_dir, ledger, owner, pins, feature_extractor=None, resume=False, overrides=None, provenance=None,
               progress=None) -> dict:
    """Head-only warm-up on A. The backbone must stay equal to the pinned pretrained groups at the start, at the end and in best.pt."""
    import torch
    from echoproof.models.ast_detector import ASTDetector
    from echoproof.training.r3_m1_selection import selection_trainer

    say = progress or (lambda *_: None)
    run_dir = Path(run_dir)
    resuming = bool(resume and (run_dir / "latest.pt").exists())
    if run_dir.exists() and any(run_dir.iterdir()) and not resuming:
        raise R4U1Error(f"refusing to reuse a non-empty run directory without a recoverable checkpoint: {run_dir}")
    cfg = warmup_train_config(recipe, owner=owner, overrides=overrides)
    detector = ASTDetector.from_dir(snapshot, device="cpu", seed=cfg.seed, feature_extractor=feature_extractor)
    detector.set_stage("head")
    stage = detector.parameter_report()
    if not stage["trainable_names"] or any(not n.startswith("classifier.") for n in stage["trainable_names"]):
        raise R4U1Error("the warm-up must train the classifier head only")
    start = detector.digests()
    backbone = {g: h for g, h in start.items() if g != "classifier"}
    if backbone != pins:
        raise R4U1Error(f"the loaded backbone differs from the pinned pretrained groups: {sorted(g for g in pins if backbone.get(g) != pins[g])}")
    prov = {"split_sha256": data["split_sha256"], "inputs": data["input_sha256"], "recipe_sha256": recipe_hash(recipe), "stage": "head",
            **(provenance or {})}
    trainer = selection_trainer(detector, cfg, data["train"], data["validation"]["selection"], run_dir, ledger_path=ledger, provenance=prov)
    say("warm-up: training the head on A")
    summary = trainer.fit(resume=resuming)
    best_path = Path(summary["best_checkpoint"])
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    best_groups = R3.state_group_digests(detector, best["model"])
    final = detector.digests()
    problems = []
    if any(final[g] != start[g] for g in backbone) or any(best_groups[g] != start[g] for g in backbone):
        problems.append("a backbone group changed during the head-only warm-up")
    if final["classifier"] == start["classifier"] or best_groups["classifier"] == start["classifier"]:
        problems.append("the classifier head did not change")
    result = {"task": WARMUP_TASK, "protocol": "A", "recipe_sha256": recipe_hash(recipe), "recipe_status": recipe.get("status"),
              "split_sha256": data["split_sha256"], "input_sha256": data["input_sha256"], "best_epoch": summary["best_epoch"],
              "best_macro_f1": summary["best_macro_f1"], "best_checkpoint": best_path.name, "best_checkpoint_sha256": file_digest(best_path),
              "checkpoint_epoch": best["epoch"], "epochs_run": summary["epochs_run"], "stop_reason": summary["stop_reason"],
              "config_hash": summary["config_hash"], "skipped_steps": trainer.skipped_steps,
              "selection_macro_f1_by_epoch": [json.loads(l)["val"]["macro_f1"] for l in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()],
              "group_sha256_start": start, "group_sha256_final": final, "group_sha256_best": best_groups,
              "backbone_unchanged": not problems or all("classifier" in p for p in problems), "problems": problems, "pins_equal_at_start": True}
    _write_json(run_dir / "warmup_result.json", result)
    if problems:
        raise R4U1Error(f"warm-up evidence failed: {problems}")
    return result


def load_warmup_result(run_dir, recipe_warm=None) -> dict:
    """The recorded warm-up result, with its best checkpoint re-hashed. The initializer of the fine-tune comes only from here."""
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "warmup_result.json").read_text(encoding="utf-8"))
    if result.get("task") != WARMUP_TASK or result.get("protocol") != "A" or result.get("problems"):
        raise R4U1Error("not a clean A warm-up result")
    best = run_dir / result["best_checkpoint"]
    if file_digest(best) != result["best_checkpoint_sha256"]:
        raise R4U1Error("the warm-up checkpoint changed since its result was written")
    if recipe_warm is not None and result["recipe_sha256"] != recipe_hash(recipe_warm):
        raise R4U1Error("the warm-up recipe changed since the warm-up ran")
    result["best_checkpoint_path"] = str(best)
    return result


# ---- fine-tune ------------------------------------------------------------------------------------------------------------------------------

def run_finetune(recipe, warm, data, *, snapshot, run_dir, ledger, owner, feature_extractor=None, resume=False, overrides=None, provenance=None,
                 progress=None) -> dict:
    """Partial fine-tune from the A warm-up: proofs, epoch-0 reproduction, training, post-training checksums. Fresh run directory only."""
    from echoproof.training.r3_m1_selection import selection_trainer

    say = progress or (lambda *_: None)
    run_dir = Path(run_dir)
    latest = run_dir / "latest.pt"
    resuming = bool(resume and latest.exists())
    if run_dir.exists() and any(run_dir.iterdir()) and not resuming:
        raise R4U1Error(f"refusing to reuse a non-empty run directory without a recoverable checkpoint: {run_dir}")
    cfg = finetune_train_config(recipe, owner=owner, overrides=overrides)
    exp = recipe["experiments"]["baseline"]
    pins = recipe["init"]["pretrained_group_sha256"]
    init_recipe = {"init": {"checkpoint_sha256": warm["best_checkpoint_sha256"], "epoch": warm["checkpoint_epoch"], "pretrained_group_sha256": pins},
                   "model": recipe["model"]}
    detector, init = R3.build_detector(init_recipe, warm["best_checkpoint_path"], snapshot=snapshot, feature_extractor=feature_extractor,
                                       device=cfg.device, pins=pins, seed=cfg.seed)
    start_digests = init["group_sha256_after_load"]
    prov = {"split_sha256": data["split_sha256"], "inputs": data["input_sha256"], "experiment": "baseline", "recipe_sha256": recipe_hash(recipe),
            "init_checkpoint_sha256": warm["best_checkpoint_sha256"], **(provenance or {})}
    groups = detector.param_groups(head_lr=exp["head_lr"], backbone_lr=exp["backbone_lr"], weight_decay=cfg.weight_decay)
    trainer = selection_trainer(detector, cfg, data["train"], data["validation"]["selection"], run_dir, param_groups=groups, ledger_path=ledger,
                                provenance=prov)
    proof_path, epoch0_path = run_dir / "gradient_proof.json", run_dir / "epoch0_metrics.json"
    if not resuming:
        say("fine-tune: gradient proof")
        proof = R3.gradient_proof(detector, R3.proof_batch(data["train"]), exp, weight_decay=cfg.weight_decay,
                                  n_blocks=recipe["model"]["unfrozen_blocks"], device=trainer.device)
        _write_json(proof_path, proof)
        say("fine-tune: epoch-0 evaluation of the loaded A warm-up")
        metrics, chunk_rows, recordings = trainer.evaluate()
        R3._write_chunk_predictions(run_dir / "epoch0_predictions_chunks.parquet", chunk_rows)
        target, tolerance = warm["best_macro_f1"], recipe["init"]["warmup_epoch0_tolerance"]
        drift = abs(metrics["macro_f1"] - target)
        _write_json(epoch0_path, {"metrics": metrics, "panels": component_panels(recordings, data["validation_components"]),
                                  "recorded_warmup_macro_f1": target, "abs_difference": drift, "tolerance": tolerance,
                                  "within_tolerance": drift <= tolerance})
        if drift > tolerance:
            raise R4U1Error(f"the loaded warm-up scores macro F1 {metrics['macro_f1']:.6f}, not the recorded {target:.6f}: wrong weights or preprocessing")
        if detector.digests() != start_digests:
            raise R4U1Error("the model changed before training started")
    say("fine-tune: training")
    summary = trainer.fit(resume=resuming)
    evidence = R3.adaptation_evidence("baseline", recipe, detector, trainer, summary, init, run_dir, resumed=resuming)
    evidence.update(task=TASK, protocol="A", init_source="A warm-up result", warmup_best_macro_f1=warm["best_macro_f1"])
    _write_json(run_dir / "adaptation_evidence.json", evidence)
    return {**summary, "adaptation_evidence": str(run_dir / "adaptation_evidence.json"), "skipped_steps": evidence["skipped_steps"]}


def paired_comparison(dataset, adapted_rows, adapted_recordings, epoch0_path, components) -> dict:
    """The loaded A warm-up (epoch 0) and the adapted model on exactly the same selection chunks and recordings, per component."""
    import pyarrow.parquet as pq
    from echoproof.training.r3_m1_export import aggregate_role
    from echoproof.training.r3_m1_selection import selection_metrics

    table = pq.read_table(epoch0_path).to_pylist()
    base_rows = [(r["chunk_id"], r["recording_id"], r["label"], r["tts_probability"]) for r in table]
    if {r[0] for r in base_rows} != {r[0] for r in adapted_rows}:
        raise R4U1Error("warm-up and adapted predictions cover different chunks")
    base_recordings = aggregate_role(dataset, base_rows)
    base, adapted = {r[0]: r for r in base_recordings}, {r[0]: r for r in adapted_recordings}
    if set(base) != set(adapted):
        raise R4U1Error("warm-up and adapted predictions cover different recordings")
    out = {"recordings": len(adapted), "chunks": len(adapted_rows)}
    for label, recs in (("warmup", base_recordings), ("adapted", adapted_recordings)):
        out[label] = {"all": selection_metrics([r[1] for r in recs], [r[2] for r in recs], 0.5), **component_panels(recs, components)}
    fixed = broken = both_right = both_wrong = 0
    for rid in sorted(adapted):
        label = adapted[rid][1]
        ok_base, ok_adapted = (base[rid][2] >= 0.5) == bool(label), (adapted[rid][2] >= 0.5) == bool(label)
        fixed += (not ok_base) and ok_adapted
        broken += ok_base and not ok_adapted
        both_right += ok_base and ok_adapted
        both_wrong += (not ok_base) and (not ok_adapted)
    out["decision_changes_at_0.5"] = {"fixed_by_adaptation": int(fixed), "broken_by_adaptation": int(broken), "right_both": int(both_right),
                                      "wrong_both": int(both_wrong)}
    out["macro_f1_change"] = out["adapted"]["all"]["macro_f1"] - out["warmup"]["all"]["macro_f1"]
    return out


def finish(run_dir, data, summary, recipe, warm, *, snapshot, feature_extractor=None, device="cuda", contract_path=None, overrides=None) -> dict:
    """Freeze the selection from checkpoint metadata alone, then score all three validation roles and compare with the warm-up."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from echoproof.models.ast_detector import ASTDetector
    from echoproof.training.r3_m1_export import aggregate_role
    from echoproof.training.r3_m1_selection import selection_metrics

    run_dir = Path(run_dir)
    path = Path(summary["best_checkpoint"])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    f1 = ckpt["metrics"]["macro_f1"]
    if (ckpt["config_hash"] != summary["config_hash"] or ckpt["epoch"] != summary["best_epoch"] or f1 != summary["best_macro_f1"]
            or summary["stop_reason"] != "max_epochs"):
        raise R4U1Error("inconsistent or unfinished checkpoint")
    selection = {"task": TASK, "protocol": "A", "selected": "baseline", "selection_threshold": 0.5, "selection_macro_f1": {"baseline": f1},
                 "checkpoint_sha256": {"baseline": file_digest(path)}, "selected_epoch": summary["best_epoch"], "split_sha256": data["split_sha256"],
                 "recipe_sha256": recipe_hash(recipe), "recipe_status": recipe.get("status"),
                 "init_checkpoint_sha256": warm["best_checkpoint_sha256"], "warmup_recipe_sha256": warm["recipe_sha256"],
                 "selection_rule": "best selection-role recording macro F1 at 0.5, earliest epoch on ties; the baseline experiment is fixed by the B selection"}
    if contract_path is not None:
        selection["contract_sha256"] = file_digest(contract_path)
    selection_path = run_dir / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise R4U1Error("existing selection differs; refusing to overwrite")
    else:
        _write_json(selection_path, selection)
    cfg = finetune_train_config(recipe, owner="export", overrides=overrides)
    detector = ASTDetector.from_dir(snapshot, device="cpu", seed=cfg.seed, feature_extractor=feature_extractor)
    detector.load_state_dict(ckpt["model"], strict=True)
    detector.to(device).eval()
    products, counts, panels, parity, paired = {}, {}, None, None, None
    components, groups = data["validation_components"], data["validation_groups"]
    for role in ROLES:                                            # the selection is on disk before the calibration and operating audio is scored
        dataset = data["validation"][role]
        rows = R3.score_amp(detector, dataset, device, cfg.precision, cfg.eval_batch_size)
        recordings = aggregate_role(dataset, rows)
        if role == "selection":
            actual = selection_metrics([r[1] for r in recordings], [r[2] for r in recordings], 0.5)["macro_f1"]
            if abs(actual - f1) > PARITY_ATOL:
                raise R4U1Error(f"reloaded checkpoint scores selection macro F1 {actual}, saved {f1}")
            saved = {r["chunk_id"]: r["tts_probability"] for r in pq.read_table(run_dir / "predictions_chunks.parquet").to_pylist()}
            if set(saved) != {r[0] for r in rows}:
                raise R4U1Error("saved best-epoch predictions cover different chunks")
            diff = max(abs(saved[r[0]] - r[3]) for r in rows)
            if diff > PARITY_ATOL:
                raise R4U1Error(f"reloaded checkpoint does not reproduce its saved predictions (max difference {diff})")
            parity = {"max_abs_probability_difference_vs_saved_best_epoch": diff, "macro_f1_reloaded": actual, "macro_f1_saved": f1, "atol": PARITY_ATOL}
            panels = component_panels(recordings, components)
            paired = paired_comparison(dataset, rows, recordings, run_dir / "epoch0_predictions_chunks.parquet", components)
        chunk_table = pa.Table.from_pylist([{"chunk_id": cid, "recording_id": rid, "group_id": groups[rid], "label": int(label),
                                             "component": components[rid], "validation_role": role, "tts_probability": float(score)}
                                            for cid, rid, label, score in rows])
        rec_table = pa.Table.from_pylist([{"recording_id": rid, "group_id": groups[rid], "label": int(label), "component": components[rid],
                                           "validation_role": role, "raw_score": float(score), "chunk_count": int(n)}
                                          for rid, label, score, n in recordings])
        for kind, table in (("chunks", chunk_table), ("recordings", rec_table)):
            out = run_dir / f"predictions_{kind}_{role}.parquet"
            tmp = out.with_suffix(".parquet.tmp")
            pq.write_table(table, tmp)
            os.replace(tmp, out)
            products[out.name] = file_digest(out)
        counts[role] = {"chunks": len(rows), "recordings": len(recordings)}
    evidence = {"task": TASK, "status": "validation_predictions_complete", "selection": selection, "selection_sha256": file_digest(selection_path),
                "input_sha256": data["input_sha256"], "output_sha256": products, "counts": counts, "selection_panels": panels,
                "reload_parity": parity, "paired_warmup_vs_adapted": paired, "score_type": "raw uncalibrated TTS score",
                "adaptation_evidence": str(run_dir / "adaptation_evidence.json"),
                "note": "Calibration and operating predictions are not selection metrics. Raw scores need R4-N1. Single seed."}
    _write_json(run_dir / "evidence.json", evidence)
    return evidence
