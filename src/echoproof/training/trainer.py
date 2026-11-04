"""R1-M1 shared, model-agnostic neural trainer for BiGRU, WavLM head and AST (IMPLEMENTATION_PLAN section 9).

Model contract: `model(waveforms=[B,T] float, valid_samples=[B] int) -> logits [B,2]` (class 1 = TTS). Feature
extraction (log-mel, processors) lives inside the model so the same trainer serves every approach. Datasets come from
`echoproof.training.datasets`.

What this guarantees (each has a test):
  * gradient accumulation weights every *sample* equally, including a final partial group;
  * fp16 gradients are unscaled before clipping;
  * frozen submodules stay in eval mode (no dropout/batch-norm updates) while the rest trains;
  * checkpoint selection uses recording-level macro F1 at a fixed threshold (never tuned per epoch);
  * `latest.pt` resumes model, optimizer, scheduler, scaler, RNG, epoch and early-stopping state exactly, and an
    incompatible resume is refused instead of silently starting over;
  * every run, including failures, appends a ledger row.
Known limitation (measured on a Colab T4, reports/r1/r1_m1_colab_*.json): resume is bit-exact for ordinary modules, but
cuDNN keeps a private dropout state for nn.GRU's inter-layer dropout that no checkpoint can capture, so a resumed BiGRU with
dropout > 0 continues correctly with different dropout masks (loss differs ~1e-4 from an uninterrupted run).
To adapt the optimizer after unfreezing (AST stage 2), create a new Trainer with new `param_groups`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import itertools
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from echoproof.artifacts import digest
from echoproof.training.datasets import collate
from echoproof.training.ledger import append_ledger
from echoproof.training.seeding import named_seed, seed_everything, seed_worker


class TrainingError(RuntimeError):
    """Non-finite loss/gradients/scores or an invalid training state."""


class ResumeError(RuntimeError):
    """Resume refused: missing, incompatible or ambiguous checkpoint state."""


@dataclass
class TrainConfig:
    seed: int = 42
    max_epochs: int = 20
    batch_size: int = 32
    accumulation_steps: int = 1
    eval_batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    scheduler: str = "constant"      # constant | plateau | cosine
    plateau_patience: int = 1
    plateau_factor: float = 0.5
    warmup_steps: int = 0            # cosine only (optimizer steps)
    patience: int = 3
    min_delta: float = 0.0
    selection_threshold: float = 0.5
    precision: str = "fp32"          # fp32 | fp16 (cuda, loss-scaled) | bf16
    class_strategy: str = "none"     # none | weights | balanced_sampling (never both)
    num_workers: int = 0
    device: str = "auto"
    deterministic: bool = True
    owner: str = ""
    model_name: str = ""
    protocol: str = ""

    def validate(self):
        ints = ("max_epochs", "batch_size", "accumulation_steps", "eval_batch_size", "patience")
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in ints):
            raise ValueError(f"{ints} must be positive integers")
        if type(self.seed) is not int or self.seed < 0 or self.num_workers < 0 or self.warmup_steps < 0:
            raise ValueError("invalid seed, num_workers or warmup_steps")
        if not (self.lr > 0 and math.isfinite(self.lr) and self.weight_decay >= 0 and self.grad_clip >= 0):
            raise ValueError("invalid lr, weight_decay or grad_clip")
        if self.scheduler not in ("constant", "plateau", "cosine"):
            raise ValueError("scheduler must be constant, plateau or cosine")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self.class_strategy not in ("none", "weights", "balanced_sampling"):
            raise ValueError("class_strategy must be none, weights or balanced_sampling")
        if not 0 < self.selection_threshold < 1:
            raise ValueError("selection_threshold must be in (0, 1)")


# ------------------------------------------------------------------ recording-level metrics
# Minimal, dependency-free versions for checkpoint selection. The evaluation package (R1-N1) is the authority for
# reporting; inject `metrics_fn` to use it. Direction: label 1 = TTS is positive, prediction is `score >= threshold`.

def _eer(labels, scores):
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0.0, np.cumsum(y == 1)[distinct] / pos]
    fpr = np.r_[0.0, np.cumsum(y == 0)[distinct] / neg]
    d = (1 - tpr) - fpr
    i = int(np.argmax(d <= 0))
    if d[i] == 0:
        return float(fpr[i])
    t = d[i - 1] / (d[i - 1] - d[i])
    return float(fpr[i - 1] + t * (fpr[i] - fpr[i - 1]))


def recording_metrics(labels, scores, threshold=0.5):
    labels, scores = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    pred = scores >= threshold
    tp, fp = int((pred & (labels == 1)).sum()), int((pred & (labels == 0)).sum())
    tn, fn = int((~pred & (labels == 0)).sum()), int((~pred & (labels == 1)).sum())
    f1 = lambda a, b, c: 0.0 if 2 * a + b + c == 0 else 2 * a / (2 * a + b + c)  # noqa: E731
    nan = float("nan")
    return {"macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2, "accuracy": (tp + tn) / len(labels),
            "fpr": fp / (fp + tn) if fp + tn else nan, "fnr": fn / (fn + tp) if fn + tp else nan,
            "eer": _eer(labels, scores), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "recordings": int(len(labels))}


# ------------------------------------------------------------------ state helpers

def _atomic_save(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(obj, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _rng_state():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"name": name, "keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos),
                      "has_gauss": int(has_gauss), "cached": float(cached)},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    p = state["python"]
    random.setstate((p[0], tuple(p[1]), p[2]))
    n = state["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _signature(model):
    return [(n, list(p.shape)) for n, p in model.named_parameters()]


class Trainer:
    def __init__(self, model, config: TrainConfig, train_dataset, eval_dataset, run_dir, *, param_groups=None,
                 ledger_path=None, provenance=None, metrics_fn=None):
        config.validate()
        self.cfg, self.model, self.train_ds, self.eval_ds = config, model, train_dataset, eval_dataset
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path, self.provenance = ledger_path, dict(provenance or {})
        self.metrics_fn = metrics_fn or recording_metrics
        want = config.device
        self.device = torch.device("cuda" if want == "auto" and torch.cuda.is_available() else "cpu" if want == "auto" else want)
        if config.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 requires CUDA; use bf16 or fp32 on CPU")
        self.model.to(self.device)

        self.class_weights = None
        counts = np.bincount(train_dataset.labels, minlength=2)
        if config.class_strategy != "none" and (counts == 0).any():
            raise ValueError("class handling requires both classes in the training set")
        self.sample_weights = None
        if config.class_strategy == "weights":
            self.class_weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=self.device)
        elif config.class_strategy == "balanced_sampling":
            self.sample_weights = [float(1.0 / counts[l]) for l in train_dataset.labels]

        params = param_groups if param_groups is not None else [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        optimized = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        orphans = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in optimized]
        frozen_in_opt = [n for n, p in model.named_parameters() if not p.requires_grad and id(p) in optimized]
        if orphans or frozen_in_opt:
            raise ValueError(f"optimizer mismatch; trainable but missing: {orphans[:3]}, frozen but present: {frozen_in_opt[:3]}")
        self.trainable = [p for g in self.optimizer.param_groups for p in g["params"]]
        if not self.trainable:
            raise ValueError("no trainable parameters")

        batches = math.ceil(len(train_dataset) / config.batch_size)
        self.steps_per_epoch = math.ceil(batches / config.accumulation_steps)
        self.step_scheduler, self.epoch_scheduler = None, None
        if config.scheduler == "plateau":
            self.epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=config.plateau_factor, patience=config.plateau_patience)
        elif config.scheduler == "cosine":
            total = self.steps_per_epoch * config.max_epochs
            warm = config.warmup_steps

            def factor(step):
                if warm and step < warm:
                    return (step + 1) / warm
                return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
            self.step_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        self.config_hash = digest({"config": asdict(config), "provenance": self.provenance})
        self.skipped_steps = 0

    # -------------------------------------------------------------- modes and forward
    def _set_train_mode(self):
        """model.train() would reactivate dropout/batch-norm in frozen blocks; put fully frozen modules back in eval."""
        self.model.train()
        for module in self.model.modules():
            params = list(module.parameters())
            if params and not any(p.requires_grad for p in params):
                module.eval()

    def _autocast(self):
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.cfg.precision)
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None)

    def _move(self, batch):
        return {k: v.to(self.device) for k, v in batch["model_inputs"].items()}, batch["labels"].to(self.device)

    def _accumulate(self, group):
        """Backward over one optimizer step's micro-batches; loss = sum of per-sample losses / total sample weight."""
        weight = self.class_weights
        denominator = sum(float((weight[b["labels"]] if weight is not None else torch.ones(len(b["labels"]))).sum()) for b in group)
        loss_total = 0.0
        for batch in group:
            inputs, labels = self._move(batch)
            with self._autocast():
                logits = self.model(**inputs)
            loss_sum = F.cross_entropy(logits.float(), labels, weight=weight, reduction="sum")
            if not torch.isfinite(loss_sum):
                raise TrainingError("non-finite training loss")
            self.scaler.scale(loss_sum / denominator).backward()
            loss_total += float(loss_sum.detach())
        return loss_total, denominator

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)  # must precede clipping so the threshold applies to true gradients
        if self.cfg.grad_clip > 0:
            norm = float(torch.nn.utils.clip_grad_norm_(self.trainable, self.cfg.grad_clip))
        else:
            norm = float(torch.norm(torch.stack([p.grad.norm() for p in self.trainable if p.grad is not None])))
        if not math.isfinite(norm):
            if self.scaler.is_enabled():
                self.skipped_steps += 1  # loss scaler will skip this step and lower the scale
            else:
                raise TrainingError("non-finite gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.step_scheduler is not None:
            self.step_scheduler.step()
        return norm

    # -------------------------------------------------------------- epochs
    def _train_loader(self, epoch):
        generator = torch.Generator().manual_seed(named_seed(self.cfg.seed, f"loader:{epoch}"))
        kwargs = dict(batch_size=self.cfg.batch_size, collate_fn=collate, num_workers=self.cfg.num_workers,
                      worker_init_fn=seed_worker, generator=generator, drop_last=False)
        if self.sample_weights is not None:
            sampler = WeightedRandomSampler(self.sample_weights, len(self.train_ds), replacement=True, generator=generator)
            return DataLoader(self.train_ds, sampler=sampler, **kwargs)
        return DataLoader(self.train_ds, shuffle=True, **kwargs)

    def _train_epoch(self, epoch):
        self.train_ds.set_epoch(epoch)
        self._set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)
        iterator, total_loss, total_weight, norms = iter(self._train_loader(epoch)), 0.0, 0.0, []
        while True:
            group = list(itertools.islice(iterator, self.cfg.accumulation_steps))
            if not group:
                break
            loss, weight = self._accumulate(group)
            total_loss, total_weight = total_loss + loss, total_weight + weight
            norms.append(self._optimizer_step())
        return total_loss / total_weight, float(np.mean(norms))

    def evaluate(self):
        self.model.eval()
        loader = DataLoader(self.eval_ds, batch_size=self.cfg.eval_batch_size, shuffle=False, collate_fn=collate,
                            num_workers=self.cfg.num_workers, worker_init_fn=seed_worker)
        chunk_rows, loss_sum, count = [], 0.0, 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = self._move(batch)
                with self._autocast():
                    logits = self.model(**inputs)
                logits = logits.float()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                if not np.isfinite(probs).all():
                    raise TrainingError("non-finite evaluation score")
                loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
                count += len(labels)
                chunk_rows += list(zip(batch["chunk_ids"], batch["recording_ids"], batch["labels"].tolist(), probs.tolist()))
        grouped = {}
        for _, rid, label, p in chunk_rows:
            grouped.setdefault(rid, [label, []])[1].append(p)
        recordings = [(rid, v[0], float(np.mean(v[1])), len(v[1])) for rid, v in grouped.items()]
        metrics = self.metrics_fn([r[1] for r in recordings], [r[2] for r in recordings], self.cfg.selection_threshold)
        metrics["loss"] = loss_sum / count
        return metrics, chunk_rows, recordings

    def _write_predictions(self, epoch, chunk_rows, recordings):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for name, table in (("predictions_chunks", {"chunk_id": [r[0] for r in chunk_rows], "recording_id": [r[1] for r in chunk_rows],
                                                    "label": [r[2] for r in chunk_rows], "tts_probability": [r[3] for r in chunk_rows]}),
                            ("predictions_recordings", {"recording_id": [r[0] for r in recordings], "label": [r[1] for r in recordings],
                                                        "raw_score": [r[2] for r in recordings], "chunks": [r[3] for r in recordings]})):
            table = pa.table({**table, "epoch": [epoch] * len(next(iter(table.values())))})
            tmp = self.run_dir / f"{name}.parquet.tmp"
            pq.write_table(table, str(tmp))
            os.replace(tmp, self.run_dir / f"{name}.parquet")

    # -------------------------------------------------------------- checkpoints
    def _checkpoint(self, epoch, state, finished):
        return {"epoch": epoch, "finished": finished, "stop_reason": state["stop_reason"], "best_metric": state["best_metric"],
                "best_epoch": state["best_epoch"], "bad_epochs": state["bad_epochs"], "train_seconds": state["train_seconds"],
                "skipped_steps": self.skipped_steps, "config": asdict(self.cfg), "config_hash": self.config_hash,
                "signature": _signature(self.model), "trainable": sorted(n for n, p in self.model.named_parameters() if p.requires_grad),
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": (self.step_scheduler or self.epoch_scheduler).state_dict() if (self.step_scheduler or self.epoch_scheduler) else None,
                "scaler": self.scaler.state_dict(), "rng": _rng_state()}

    def _load_resume(self):
        path = self.run_dir / "latest.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt["config_hash"] != self.config_hash:
            raise ResumeError("checkpoint was made with a different config/provenance; start a new run directory")
        if ckpt["signature"] != [list(x) for x in _signature(self.model)] and ckpt["signature"] != _signature(self.model):
            raise ResumeError("checkpoint architecture differs from the current model")
        if ckpt["trainable"] != sorted(n for n, p in self.model.named_parameters() if p.requires_grad):
            raise ResumeError("checkpoint trainable set differs from the current model")
        try:
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, RuntimeError) as exc:
            raise ResumeError(f"checkpoint state does not fit this optimizer/model: {exc}") from exc
        scheduler = self.step_scheduler or self.epoch_scheduler
        if scheduler is not None and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.skipped_steps = ckpt["skipped_steps"]
        _restore_rng(ckpt["rng"])
        return ckpt

    # -------------------------------------------------------------- run
    def fit(self, resume: bool = False):
        latest = self.run_dir / "latest.pt"
        started = time.monotonic()
        state = {"best_metric": -math.inf, "best_epoch": 0, "bad_epochs": 0, "stop_reason": None, "train_seconds": 0.0}
        start_epoch = 0
        if latest.exists() and not resume:
            raise ResumeError("run directory already contains a checkpoint; pass resume=True or use a new directory")
        if resume and not latest.exists():
            raise ResumeError("resume requested but no latest.pt exists")
        if resume:
            ckpt = self._load_resume()
            start_epoch = ckpt["epoch"]
            state.update({k: ckpt[k] for k in state})
            if ckpt["finished"]:
                return self._summary(state, start_epoch)
        else:
            if self.cfg.deterministic and self.device.type == "cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            seed_everything(self.cfg.seed)
            if not self.cfg.deterministic:
                torch.use_deterministic_algorithms(False)
            (self.run_dir / "resolved_config.json").write_text(json.dumps(
                {"config": asdict(self.cfg), "provenance": self.provenance, "config_hash": self.config_hash, "device": str(self.device),
                 "total_parameters": sum(p.numel() for p in self.model.parameters()),
                 "trainable_parameters": sum(p.numel() for p in self.trainable),
                 "trainable_names": sorted(n for n, p in self.model.named_parameters() if p.requires_grad)}, indent=2, sort_keys=True), encoding="utf-8")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch = start_epoch
        try:
            for epoch in range(start_epoch + 1, self.cfg.max_epochs + 1):
                tick = time.monotonic()
                train_loss, grad_norm = self._train_epoch(epoch)
                metrics, chunk_rows, recordings = self.evaluate()
                score = metrics["macro_f1"]
                improved = score > state["best_metric"] + self.cfg.min_delta
                if improved:
                    state.update(best_metric=score, best_epoch=epoch, bad_epochs=0)
                    _atomic_save({"epoch": epoch, "model": self.model.state_dict(), "metrics": metrics, "config": asdict(self.cfg),
                                  "config_hash": self.config_hash}, self.run_dir / "best.pt")
                    self._write_predictions(epoch, chunk_rows, recordings)
                else:
                    state["bad_epochs"] += 1
                if self.epoch_scheduler is not None:
                    self.epoch_scheduler.step(score)
                state["train_seconds"] += time.monotonic() - tick
                entry = {"epoch": epoch, "train_loss": train_loss, "grad_norm": grad_norm, "lr": self.optimizer.param_groups[0]["lr"],
                         "val": {k: v for k, v in metrics.items()}, "improved": improved, "seconds": time.monotonic() - tick,
                         "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if self.device.type == "cuda" else None,
                         "skipped_steps": self.skipped_steps}
                with open(self.run_dir / "history.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, sort_keys=True, allow_nan=True) + "\n")
                if state["bad_epochs"] >= self.cfg.patience:
                    state["stop_reason"] = "early_stopping"
                elif epoch == self.cfg.max_epochs:
                    state["stop_reason"] = "max_epochs"
                _atomic_save(self._checkpoint(epoch, state, state["stop_reason"] is not None), latest)
                if state["stop_reason"]:
                    break
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else f"failed:{type(exc).__name__}"
            self._ledger(state, epoch, status, started, notes=str(exc)[:200])
            raise
        summary = self._summary(state, epoch)
        _atomic_text(json.dumps(summary, indent=2, sort_keys=True), self.run_dir / "status.json")
        best = json.loads((self.run_dir / "history.jsonl").read_text().splitlines()[state["best_epoch"] - 1])["val"] if state["best_epoch"] else {}
        self._ledger(state, epoch, "completed", started, best=best)
        return summary

    def _summary(self, state, epochs_run):
        return {"epochs_run": epochs_run, "best_epoch": state["best_epoch"], "best_macro_f1": state["best_metric"],
                "stop_reason": state["stop_reason"], "config_hash": self.config_hash, "run_dir": str(self.run_dir),
                "best_checkpoint": str(self.run_dir / "best.pt"), "latest_checkpoint": str(self.run_dir / "latest.pt")}

    def _ledger(self, state, epochs_run, status, started, best=None, notes=None):
        if not self.ledger_path:
            return
        best = best or {}
        append_ledger(self.ledger_path, {
            "run_id": self.run_dir.name, "owner": self.cfg.owner, "model": self.cfg.model_name, "protocol": self.cfg.protocol,
            "config_hash": self.config_hash, "split_hash": self.provenance.get("split_sha256"), "seed": self.cfg.seed,
            "epochs_run": epochs_run, "best_epoch": state["best_epoch"] or None, "stop_reason": state["stop_reason"],
            "val_macro_f1": best.get("macro_f1"), "val_eer": best.get("eer"), "val_fpr": best.get("fpr"),
            "train_minutes": round((state["train_seconds"] + (time.monotonic() - started)) / 60, 3) if status != "completed" else round(state["train_seconds"] / 60, 3),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if self.device.type == "cuda" else None,
            "device": str(self.device), "status": status, "notes": notes})
