"""R3-U2: AST partial fine-tuning from the R2-U2 warm-up checkpoint (Protocol B), with the adaptation evidence the plan requires.

What this adds to the shared Trainer (which is used unchanged for the loop, fp16 scaling, cosine schedule, resume and ledger rows):
  * a fresh run: only the checkpoint's `model` weights are loaded (strictly, after its SHA-256 and the untouched pretrained groups are verified);
    no optimizer, scaler, scheduler or RNG state is restored from the warm-up run;
  * the partial stage (last two blocks, final norm, classifier) is checked, and optimiser groups are rebuilt (backbone / head learning rates);
  * a gradient proof on a copy of the model: non-zero finite gradients and changed checksums exactly in the unfrozen groups, bit-identical frozen groups;
  * an epoch-0 evaluation of the loaded warm-up that must reproduce the R2-U2 selection macro F1, and whose predictions are the paired comparison;
  * after training, the checkpoints' group checksums prove which groups moved;
  * a selection that is frozen before calibration and operating audio is scored, raw uncalibrated exports in the BiGRU column layout, and a paired
    warm-up versus adapted comparison on the same selection recordings.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from echoproof.artifacts import digest, file_digest

TASK = "R3-U2"
EXPERIMENTS = ("baseline", "lower_backbone_lr")
ROLES = ("selection", "calibration", "operating")
PARITY_ATOL = 1e-3


class R3U2Error(ValueError):
    pass


# ---- recipe ---------------------------------------------------------------------------------------------------------------------------------

REQUIRED = {"weight_decay": 0.01, "scheduler": "cosine", "selection_threshold": 0.5, "class_strategy": "none", "precision": "fp16",
            "seed": 42}


def load_recipe(path) -> dict:
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    if recipe.get("task") != TASK or recipe.get("protocol") != "B" or recipe.get("schema_version") != 1:
        raise R3U2Error("not an R3-U2 Protocol B recipe")
    train = recipe["train"]
    for key, value in REQUIRED.items():
        if train.get(key) != value:
            raise R3U2Error(f"recipe departs from the plan: train.{key} must be {value!r}")
    if train["batch_size"] * train["accumulation_steps"] != 16:
        raise R3U2Error("effective batch size must be 16")
    if not 6 <= train["max_epochs"] <= 10 or train["patience"] < train["max_epochs"] or train["warmup_steps"] < 1:
        raise R3U2Error("epochs must be 6 to 10, early stopping disabled, and a warm-up present")
    if recipe["model"]["unfrozen_blocks"] != 2 or recipe["model"]["stage"] != "partial":
        raise R3U2Error("the plan unfreezes exactly the last two blocks")
    if set(recipe["experiments"]) != set(EXPERIMENTS):
        raise R3U2Error("expected the baseline and one alternative")
    base = recipe["experiments"]["baseline"]
    if base["backbone_lr"] != 1e-5 or base["head_lr"] != 1e-4:
        raise R3U2Error("baseline learning rates must be backbone 1e-5 and head 1e-4")
    if recipe["selection"]["role"] != "selection" or "0.5" not in recipe["selection"]["metric"]:
        raise R3U2Error("selection must use the selection role at fixed 0.5")
    if len(recipe["init"]["checkpoint_sha256"]) != 64:
        raise R3U2Error("the initial checkpoint hash is required")
    return recipe


def recipe_hash(recipe: dict) -> str:
    return digest(recipe)


def train_config(recipe: dict, name: str, *, owner: str, overrides: dict | None = None):
    from echoproof.training.trainer import TrainConfig
    if not owner or not isinstance(owner, str):
        raise R3U2Error("an owner is required so runs are not attributed to the wrong person")
    t, e = recipe["train"], recipe["experiments"][name]
    cfg = TrainConfig(seed=t["seed"], max_epochs=t["max_epochs"], batch_size=t["batch_size"], accumulation_steps=t["accumulation_steps"],
                      eval_batch_size=t["eval_batch_size"], lr=e["head_lr"], weight_decay=t["weight_decay"], grad_clip=t["grad_clip"],
                      scheduler=t["scheduler"], warmup_steps=t["warmup_steps"], patience=t["patience"], min_delta=t["min_delta"],
                      selection_threshold=t["selection_threshold"], precision=t["precision"], class_strategy=t["class_strategy"],
                      num_workers=t["num_workers"], device=t["device"], deterministic=t["deterministic"], owner=owner,
                      model_name="ast_partial", protocol="B")
    return replace(cfg, **overrides) if overrides else cfg


# ---- model: warm-up weights into the partial stage -----------------------------------------------------------------------------------------

def expected_groups(detector, n_blocks: int) -> list:
    return sorted(["classifier", "final_norm"] + [f"block.{i:02d}" for i in range(detector.n_blocks - n_blocks, detector.n_blocks)])


def load_warmup(detector, checkpoint, *, expected_sha: str, expected_epoch: int, pins: dict) -> dict:
    """Load only the `model` weights, strictly. Everything else in the checkpoint is ignored by design."""
    import torch
    checkpoint = Path(checkpoint)
    if file_digest(checkpoint) != expected_sha:
        raise R3U2Error("the warm-up checkpoint differs from the pinned SHA-256")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("epoch") != expected_epoch or "model" not in ckpt:
        raise R3U2Error("unexpected warm-up checkpoint epoch or layout")
    if any(not torch.isfinite(v).all() for v in ckpt["model"].values() if v.is_floating_point()):
        raise R3U2Error("the warm-up checkpoint contains non-finite weights")
    detector.load_state_dict(ckpt["model"], strict=True)
    groups = detector.digests()
    differing = sorted(g for g, h in pins.items() if groups.get(g) != h)
    if differing:
        raise R3U2Error(f"pretrained groups differ from the pinned backbone after loading: {differing}")
    return {"checkpoint_sha256": expected_sha, "epoch": ckpt["epoch"], "warmup_metrics": ckpt.get("metrics"), "group_sha256_after_load": groups,
            "loaded": "model weights only, strict=True; optimizer, scaler, scheduler and RNG state not restored"}


def check_partial_stage(detector, n_blocks: int) -> dict:
    prefixes = detector.stage_prefixes("partial", n_blocks)
    for name, p in detector.model.named_parameters():
        if p.requires_grad != any(name.startswith(x) for x in prefixes):
            raise R3U2Error(f"trainable flag differs from the partial stage: {name}")
    report = detector.parameter_report()
    groups = sorted(g for g, v in report["groups"].items() if v["trainable"])
    if groups != expected_groups(detector, n_blocks):
        raise R3U2Error(f"unexpected trainable groups: {groups}")
    if any(v["trainable"] not in (0, v["parameters"]) for v in report["groups"].values()):
        raise R3U2Error("a group is partially trainable")
    return report


def build_detector(recipe: dict, checkpoint, *, snapshot, feature_extractor=None, device="cpu", pins=None, seed=42):
    from echoproof.models.ast_detector import ASTDetector
    init, n = recipe["init"], recipe["model"]["unfrozen_blocks"]
    detector = ASTDetector.from_dir(snapshot, device="cpu", seed=seed, feature_extractor=feature_extractor)
    info = load_warmup(detector, checkpoint, expected_sha=init["checkpoint_sha256"], expected_epoch=init["epoch"],
                       pins=init["pretrained_group_sha256"] if pins is None else pins)
    detector.set_stage("partial", n)
    info["stage"] = check_partial_stage(detector, n)
    detector.to(device)
    return detector, info


def _set_train_mode(model):
    """The shared Trainer's rule: fully frozen modules stay in eval mode while the rest trains."""
    model.train()
    for module in model.modules():
        params = list(module.parameters())
        if params and not any(p.requires_grad for p in params):
            module.eval()


def gradient_proof(detector, batch, exp: dict, *, weight_decay: float, n_blocks: int, device) -> dict:
    """One optimiser step on a deep copy: non-zero finite gradients and changed checksums exactly in the unfrozen groups."""
    import torch
    import torch.nn.functional as F
    clone = copy.deepcopy(detector).to(device)
    _set_train_mode(clone)
    before = clone.digests()
    optimizer = torch.optim.AdamW(clone.param_groups(head_lr=exp["head_lr"], backbone_lr=exp["backbone_lr"], weight_decay=weight_decay))
    inputs = {k: v.to(device) for k, v in batch["model_inputs"].items()}
    labels = batch["labels"].to(device)
    loss = F.cross_entropy(clone(**inputs).float(), labels)
    if not torch.isfinite(loss):
        raise R3U2Error("non-finite loss in the gradient proof")
    loss.backward()
    norms = {}
    for name, p in clone.model.named_parameters():
        if p.grad is not None:
            group = clone._group(name)
            norms[group] = norms.get(group, 0.0) + float(p.grad.detach().double().pow(2).sum())
    norms = {g: math.sqrt(v) for g, v in sorted(norms.items())}
    optimizer.step()
    after = clone.digests()
    expected = expected_groups(detector, n_blocks)
    changed = sorted(g for g in after if after[g] != before[g])
    with_grad = sorted(g for g, v in norms.items() if v > 0 and math.isfinite(v))
    frozen = sorted(set(after) - set(expected))
    result = {"expected_updated_groups": expected, "groups_with_gradients": with_grad, "gradient_norms": norms,
              "groups_changed_by_one_step": changed, "frozen_groups_bit_identical": [g for g in frozen if after[g] == before[g]],
              "frozen_groups": frozen, "loss": float(loss.detach()), "samples": int(labels.numel()),
              "ok": with_grad == expected and changed == expected and all(after[g] == before[g] for g in frozen)}
    del clone, optimizer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    if not result["ok"]:
        raise R3U2Error(f"gradient proof failed: {result}")
    return result


def proof_batch(train_ds, per_class: int = 4):
    """A fixed, class-balanced micro-batch (first items of each class in dataset order) for the gradient proof."""
    from echoproof.training.datasets import collate
    train_ds.set_epoch(1)
    picks = {0: [], 1: []}
    for index, (_, label) in enumerate(train_ds.items):
        if len(picks[label]) < per_class:
            picks[label].append(index)
        if all(len(v) == per_class for v in picks.values()):
            break
    if any(len(v) < per_class for v in picks.values()):
        raise R3U2Error("not enough training items of each class for the gradient proof")
    return collate([train_ds[i] for i in picks[0] + picks[1]])


def state_group_digests(detector, state: dict) -> dict:
    """Group checksums of a saved `model` state dict, computed exactly as `ASTDetector.digests` does for live parameters."""
    hashers = {}
    for name, p in sorted(detector.model.named_parameters()):
        tensor = state["model." + name]
        hashers.setdefault(detector._group(name), hashlib.sha256()).update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {k: v.hexdigest() for k, v in sorted(hashers.items())}


# ---- scoring helpers ----------------------------------------------------------------------------------------------------------------------------

def score_amp(model, dataset, device, precision: str, batch_size: int):
    """Chunk probabilities with the same autocast as the Trainer's evaluation, so a reloaded checkpoint reproduces its saved scores."""
    import torch
    from torch.utils.data import DataLoader
    from echoproof.training.datasets import collate
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    device = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch["model_inputs"].items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(**inputs)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            if not np.isfinite(probs).all():
                raise R3U2Error("non-finite score")
            rows.extend(zip(batch["chunk_ids"], batch["recording_ids"], batch["labels"].tolist(), probs.tolist()))
    return rows


def _write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_chunk_predictions(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({"chunk_id": [r[0] for r in rows], "recording_id": [r[1] for r in rows], "label": [int(r[2]) for r in rows],
                      "tts_probability": [float(r[3]) for r in rows]})
    tmp = Path(path).with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, path)


# ---- one experiment -------------------------------------------------------------------------------------------------------------------------------

def run_experiment(name, recipe, data, *, checkpoint, snapshot, output, ledger, owner, feature_extractor=None, pins=None, resume=False,
                   overrides=None, provenance=None, progress=None) -> dict:
    """Train one experiment for the full budget. `output` is the folder shared by both experiments; the run lives in `<output>/<output name>-<name>`."""
    import torch
    from echoproof.training.r3_m1_selection import selection_panels, selection_trainer

    say = progress or (lambda *_: None)
    output = Path(output)
    run_dir = output / f"{output.name}-{name}"
    latest = run_dir / "latest.pt"
    resuming = bool(resume and latest.exists())
    if run_dir.exists() and any(run_dir.iterdir()) and not resuming:
        raise R3U2Error(f"refusing to reuse a non-empty run directory without a recoverable checkpoint: {run_dir}")
    cfg = train_config(recipe, name, owner=owner, overrides=overrides)
    exp = recipe["experiments"][name]
    detector, init = build_detector(recipe, checkpoint, snapshot=snapshot, feature_extractor=feature_extractor, device=cfg.device, pins=pins,
                                    seed=cfg.seed)
    start_digests = init["group_sha256_after_load"]
    prov = {"split_sha256": data["split_sha256"], "inputs": data["input_sha256"], "experiment": name, "recipe_sha256": recipe_hash(recipe),
            "init_checkpoint_sha256": recipe["init"]["checkpoint_sha256"], **(provenance or {})}
    groups = detector.param_groups(head_lr=exp["head_lr"], backbone_lr=exp["backbone_lr"], weight_decay=cfg.weight_decay)
    trainer = selection_trainer(detector, cfg, data["train"], data["validation"]["selection"], run_dir, param_groups=groups,
                                ledger_path=ledger, provenance=prov)
    proof_path, epoch0_path = run_dir / "gradient_proof.json", run_dir / "epoch0_metrics.json"
    if not resuming:
        say(f"{name}: gradient proof")
        proof = gradient_proof(detector, proof_batch(data["train"]), exp, weight_decay=cfg.weight_decay,
                               n_blocks=recipe["model"]["unfrozen_blocks"], device=trainer.device)
        _write_json(proof_path, proof)
        say(f"{name}: epoch-0 evaluation of the loaded warm-up")
        metrics, chunk_rows, recordings = trainer.evaluate()
        _write_chunk_predictions(run_dir / "epoch0_predictions_chunks.parquet", chunk_rows)
        panels = selection_panels(recordings, data["validation_components"])
        target, tolerance = recipe["init"]["warmup_selection_macro_f1"], recipe["init"]["warmup_epoch0_tolerance"]
        drift = None if target is None else abs(metrics["macro_f1"] - target)
        _write_json(epoch0_path, {"metrics": metrics, "panels": panels, "recorded_warmup_macro_f1": target, "abs_difference": drift,
                                  "tolerance": tolerance, "within_tolerance": None if drift is None else drift <= tolerance})
        if drift is not None and drift > tolerance:
            raise R3U2Error(f"the loaded warm-up scores macro F1 {metrics['macro_f1']:.6f}, not the recorded {target:.6f}: wrong weights or preprocessing")
        # the evaluation and the proof must not have touched the model or the trainer's own parameters
        if detector.digests() != start_digests:
            raise R3U2Error("the model changed before training started")
    say(f"{name}: training")
    summary = trainer.fit(resume=resuming)
    evidence = adaptation_evidence(name, recipe, detector, trainer, summary, init, run_dir, resumed=resuming)
    return {**summary, "adaptation_evidence": str(run_dir / "adaptation_evidence.json"), "skipped_steps": evidence["skipped_steps"]}


def adaptation_evidence(name, recipe, detector, trainer, summary, init, run_dir, *, resumed) -> dict:
    import torch
    n = recipe["model"]["unfrozen_blocks"]
    expected = expected_groups(detector, n)
    start = init["group_sha256_after_load"]
    final = detector.digests()
    best = torch.load(summary["best_checkpoint"], map_location="cpu", weights_only=True)
    best_groups = state_group_digests(detector, best["model"])
    frozen = sorted(set(start) - set(expected))
    problems = []
    if any(final[g] != start[g] for g in frozen) or any(best_groups[g] != start[g] for g in frozen):
        problems.append("a frozen group changed")
    if any(final[g] == start[g] for g in expected):
        problems.append("an unfrozen group did not change by the end of training")
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    steps = trainer.steps_per_epoch * trainer.cfg.max_epochs
    evidence = {"task": TASK, "experiment": name, "recipe_sha256": recipe_hash(recipe), "config_hash": summary["config_hash"],
                "config": asdict(trainer.cfg), "resumed": bool(resumed),
                "initial_checkpoint": {k: init[k] for k in ("checkpoint_sha256", "epoch", "warmup_metrics", "loaded")},
                "trainable_groups": expected, "frozen_groups": frozen,
                "group_sha256_start": start, "group_sha256_final_epoch": final, "group_sha256_best_checkpoint": best_groups,
                "frozen_groups_unchanged_final_and_best": not any("frozen" in p for p in problems),
                "unfrozen_groups_changed_final": [g for g in expected if final[g] != start[g]],
                "unfrozen_groups_changed_best": [g for g in expected if best_groups[g] != start[g]],
                "best_epoch": summary["best_epoch"], "best_macro_f1": summary["best_macro_f1"], "epochs_run": summary["epochs_run"],
                "selection_macro_f1_by_epoch": [h["val"]["macro_f1"] for h in history], "train_loss_by_epoch": [h["train_loss"] for h in history],
                "learning_rate_at_epoch_end": [h["lr"] for h in history], "peak_vram_mb": max((h["peak_vram_mb"] or 0) for h in history),
                "skipped_steps": trainer.skipped_steps, "optimizer_steps_per_epoch": trainer.steps_per_epoch, "optimizer_steps_total": steps,
                "scheduler_clock": "one step per optimiser attempt, including fp16 GradScaler-skipped steps (shared Trainer convention)",
                "gradient_proof_file": "gradient_proof.json", "epoch0_file": "epoch0_metrics.json", "problems": problems}
    _write_json(run_dir / "adaptation_evidence.json", evidence)
    if problems:
        raise R3U2Error(f"adaptation evidence failed: {problems}")
    return evidence


# ---- selection, freeze, paired comparison and export ----------------------------------------------------------------------------------------

def choose_winner(f1_by_experiment: dict) -> str:
    if set(f1_by_experiment) != set(EXPERIMENTS):
        raise R3U2Error("both experiments are required")
    scores = {k: float(v) for k, v in f1_by_experiment.items()}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
        raise R3U2Error("invalid selection score")
    return "lower_backbone_lr" if scores["lower_backbone_lr"] > scores["baseline"] else "baseline"     # exact tie: baseline


def finish(output, data, summaries, recipe, *, snapshot, feature_extractor=None, device="cuda", contract_path=None, overrides=None) -> dict:
    """Choose from checkpoint metadata alone, freeze it, then score all three validation roles with the winner and compare with the warm-up."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from echoproof.models.ast_detector import ASTDetector
    from echoproof.training.r3_m1_export import aggregate_role
    from echoproof.training.r3_m1_selection import selection_metrics, selection_panels

    output = Path(output)
    if set(summaries) != set(EXPERIMENTS):
        raise R3U2Error("both completed runs are required")
    checkpoints, f1, hashes = {}, {}, {}
    for name, summary in summaries.items():
        path = Path(summary["best_checkpoint"])
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if (ckpt["config_hash"] != summary["config_hash"] or ckpt["epoch"] != summary["best_epoch"]
                or ckpt["metrics"]["macro_f1"] != summary["best_macro_f1"] or summary["stop_reason"] != "max_epochs"):
            raise R3U2Error(f"inconsistent or unfinished checkpoint: {name}")
        checkpoints[name], f1[name], hashes[name] = ckpt, ckpt["metrics"]["macro_f1"], file_digest(path)
    winner = choose_winner(f1)
    selection = {"task": TASK, "protocol": "B", "selected": winner, "selection_threshold": 0.5, "selection_macro_f1": f1,
                 "checkpoint_sha256": hashes, "selected_epoch": summaries[winner]["best_epoch"], "split_sha256": data["split_sha256"],
                 "recipe_sha256": recipe_hash(recipe), "tie_rule": "baseline", "recipe_status": recipe.get("status"),
                 "init_checkpoint_sha256": recipe["init"]["checkpoint_sha256"]}
    if contract_path is not None:
        selection["contract_sha256"] = file_digest(contract_path)
    selection_path = output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise R3U2Error("existing selection differs; refusing to overwrite")
    else:
        _write_json(selection_path, selection)
    # The winner is on disk before any calibration or operating audio is scored.
    cfg = train_config(recipe, winner, owner="export", overrides=overrides)
    detector = ASTDetector.from_dir(snapshot, device="cpu", seed=cfg.seed, feature_extractor=feature_extractor)
    detector.load_state_dict(checkpoints[winner]["model"], strict=True)
    detector.to(device).eval()
    run_dir = Path(summaries[winner]["run_dir"])
    products, counts, panels, parity, paired = {}, {}, None, None, None
    components, groups = data["validation_components"], data["validation_groups"]
    for role in ROLES:
        dataset = data["validation"][role]
        rows = score_amp(detector, dataset, device, cfg.precision, cfg.eval_batch_size)
        recordings = aggregate_role(dataset, rows)
        if role == "selection":
            actual = selection_metrics([r[1] for r in recordings], [r[2] for r in recordings], 0.5)["macro_f1"]
            if abs(actual - f1[winner]) > PARITY_ATOL:
                raise R3U2Error(f"reloaded checkpoint scores selection macro F1 {actual}, saved {f1[winner]}")
            saved = {r["chunk_id"]: r["tts_probability"] for r in pq.read_table(run_dir / "predictions_chunks.parquet").to_pylist()}
            if set(saved) != {r[0] for r in rows}:
                raise R3U2Error("saved best-epoch predictions cover different chunks")
            diff = max(abs(saved[r[0]] - r[3]) for r in rows)
            if diff > PARITY_ATOL:
                raise R3U2Error(f"reloaded checkpoint does not reproduce its saved predictions (max difference {diff})")
            parity = {"max_abs_probability_difference_vs_saved_best_epoch": diff, "macro_f1_reloaded": actual, "macro_f1_saved": f1[winner],
                      "atol": PARITY_ATOL}
            panels = selection_panels(recordings, components)
            paired = paired_comparison(dataset, rows, recordings, run_dir / "epoch0_predictions_chunks.parquet", components)
        chunk_table = pa.Table.from_pylist([{"chunk_id": cid, "recording_id": rid, "group_id": groups[rid], "label": int(label),
                                             "component": components[rid], "validation_role": role, "tts_probability": float(score)}
                                            for cid, rid, label, score in rows])
        rec_table = pa.Table.from_pylist([{"recording_id": rid, "group_id": groups[rid], "label": int(label), "component": components[rid],
                                           "validation_role": role, "raw_score": float(score), "chunk_count": int(n)}
                                          for rid, label, score, n in recordings])
        for kind, table in (("chunks", chunk_table), ("recordings", rec_table)):
            path = output / f"predictions_{kind}_{role}.parquet"
            tmp = path.with_suffix(".parquet.tmp")
            pq.write_table(table, tmp)
            os.replace(tmp, path)
            products[path.name] = file_digest(path)
        counts[role] = {"chunks": len(rows), "recordings": len(recordings)}
    evidence = {"task": TASK, "status": "validation_predictions_complete", "selection": selection, "selection_sha256": file_digest(selection_path),
                "input_sha256": data["input_sha256"], "output_sha256": products, "counts": counts, "selection_panels": panels,
                "reload_parity": parity, "paired_warmup_vs_adapted": paired, "score_type": "raw uncalibrated TTS score",
                "adaptation_evidence": {n: str(Path(s["run_dir"]) / "adaptation_evidence.json") for n, s in summaries.items()},
                "note": "Calibration and operating predictions are not selection metrics. Raw scores need R3-N1. Single seed."}
    _write_json(output / "evidence.json", evidence)
    return evidence


def paired_comparison(dataset, adapted_rows, adapted_recordings, epoch0_path, components) -> dict:
    """The loaded warm-up (epoch 0) and the adapted model on exactly the same selection chunks and recordings."""
    import pyarrow.parquet as pq
    from echoproof.training.r3_m1_export import aggregate_role
    from echoproof.training.r3_m1_selection import selection_metrics, selection_panels

    table = pq.read_table(epoch0_path).to_pylist()
    base_rows = [(r["chunk_id"], r["recording_id"], r["label"], r["tts_probability"]) for r in table]
    if {r[0] for r in base_rows} != {r[0] for r in adapted_rows}:
        raise R3U2Error("warm-up and adapted predictions cover different chunks")
    base_recordings = aggregate_role(dataset, base_rows)
    base = {r[0]: r for r in base_recordings}
    adapted = {r[0]: r for r in adapted_recordings}
    if set(base) != set(adapted):
        raise R3U2Error("warm-up and adapted predictions cover different recordings")
    ids = sorted(adapted)
    out = {"recordings": len(ids), "chunks": len(adapted_rows)}
    for label, recs in (("warmup", base_recordings), ("adapted", adapted_recordings)):
        out[label] = {"all": selection_metrics([r[1] for r in recs], [r[2] for r in recs], 0.5), **selection_panels(recs, components)}
    fixed = broken = both_right = both_wrong = 0
    for rid in ids:
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
