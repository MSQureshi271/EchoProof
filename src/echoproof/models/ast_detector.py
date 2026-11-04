"""R1-U2 Audio Spectrogram Transformer wrapper: 2-class head, freeze/unfreeze stages, loading audit.

Model contract (shared with echoproof.training.trainer): forward(waveforms=[B,T], valid_samples=[B]) -> logits [B,2], class 1 = TTS.
Only the valid samples of each clip are passed to the pinned AST feature extractor, which then applies its own documented
padding/truncation to the fixed 1,024-frame grid (checkpoint normalisation included). The grid is NOT changed here: a shorter
grid needs compatible positional embeddings and is a separate, validated experiment. Because short clips are padded to the grid,
padding length can act as a duration cue; that is audited in EDA (see IMPLEMENTATION_PLAN section 8).

Stages:
  frozen   - nothing trainable (inference / feature checks)
  head     - classifier.* only (head warm-up with a frozen backbone)
  partial  - classifier.*, the backbone's final LayerNorm, and the last n encoder blocks (default 2)
Rebuild the optimizer (new Trainer with `param_groups`) whenever the trainable set changes.

Version note: transformers renames parameters between major versions (4.x: `...encoder.layer.N.attention.attention.query`;
5.x: `...layers.N.attention.q_proj`, with the Hub checkpoint's old names converted while loading). Block and final-norm locations
are therefore discovered from the live model, the loading audit does not compare raw checkpoint names, and checkpoints saved from
this model use whatever names the pinned transformers version produces: keep the transformers version pinned in the environment lock.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import torch
from torch import nn

from echoproof.features.pretrained import SAMPLE_RATE, list_trainable, pinned_snapshot

BACKBONE = "audio_spectrogram_transformer"
HEAD_TENSORS = ("classifier.dense.weight", "classifier.dense.bias")   # the only tensors allowed to be new or reshaped
STAGES = ("frozen", "head", "partial")


class ASTLoadError(RuntimeError):
    """The checkpoint does not load into the 2-class model the way the plan requires, or the model layout is unrecognised."""


def checkpoint_shapes(path) -> list:
    """Tensor shapes stored in the snapshot's model.safetensors (names are deliberately not used; see the version note)."""
    file = Path(path) / "model.safetensors"
    if not file.exists():
        raise ASTLoadError(f"{file} not found; the audit needs the safetensors checkpoint")
    from safetensors import safe_open
    with safe_open(str(file), framework="pt") as handle:
        return [tuple(handle.get_slice(k).get_shape()) for k in handle.keys() if not k.endswith("position_ids")]


def shape_audit(ckpt_shapes, model) -> dict:
    """Name-independent audit: compare the multiset of tensor shapes in the checkpoint with the model's.

    ok only if the model's extra shapes are exactly the new head ([num_labels, hidden] and [num_labels]) and the checkpoint's extra
    shapes, if any, are the old head with the same number of labels ([L, hidden] and [L]) - i.e. nothing but the task head differs.
    """
    hidden, labels = int(model.config.hidden_size), int(model.config.num_labels)
    want = Counter(tuple(v.shape) for k, v in model.state_dict().items() if not k.endswith("position_ids"))
    have = Counter(ckpt_shapes)
    model_only, ckpt_only = want - have, have - want
    head_shapes = {(labels, hidden): 1, (labels,): 1}
    ok_model = all(head_shapes.get(s, 0) >= n for s, n in model_only.items())
    old_labels = {s[0] for s in ckpt_only if len(s) >= 1}
    ok_ckpt = (not ckpt_only) or (len(old_labels) == 1 and set(ckpt_only) == {(next(iter(old_labels)), hidden), (next(iter(old_labels)),)}
                                  and all(n == 1 for n in ckpt_only.values()))
    return {"ok": bool(ok_model and ok_ckpt), "model_only_shapes": {str(k): v for k, v in model_only.items()},
            "checkpoint_only_shapes": {str(k): v for k, v in ckpt_only.items()}, "tensors_in_model": sum(want.values()),
            "tensors_in_checkpoint": sum(have.values())}


def audit_loading(report: dict) -> dict:
    """Name-based audit of the library's own loading report: only the task-head tensors may be new/reshaped; nothing unexpected."""
    head = set(HEAD_TENSORS)
    extra_new = sorted((set(report["missing"]) | set(report["mismatched"])) - head)
    return {**report, "ok": not extra_new and not report["unexpected"], "non_head_new_or_reshaped": extra_new}


def _library_report(info):
    """Normalise transformers' output_loading_info dict (None if this version does not provide one)."""
    if not isinstance(info, dict) or not any(k in info for k in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        return None
    mismatched = [m[0] if isinstance(m, (tuple, list)) else m for m in info.get("mismatched_keys", [])]
    return {"missing": sorted(info.get("missing_keys", [])), "unexpected": sorted(info.get("unexpected_keys", [])), "mismatched": sorted(mismatched)}


class ASTDetector(nn.Module):
    def __init__(self, model, feature_extractor, loading=None):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.loading = loading
        self._layer_prefix, self._n_layers = self._find_blocks()
        self._norm_prefix = self._find_final_norm()
        self.set_stage("frozen")

    # ---- layout discovery --------------------------------------------------------------------------------------------
    def _find_blocks(self):
        backbone = getattr(self.model, BACKBONE)
        encoder = getattr(backbone, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            return f"{BACKBONE}.encoder.layer", len(encoder.layer)
        if hasattr(backbone, "layers"):
            return f"{BACKBONE}.layers", len(backbone.layers)
        raise ASTLoadError("cannot find the encoder blocks in this transformers version; update ASTDetector._find_blocks")

    def _find_final_norm(self):
        backbone = getattr(self.model, BACKBONE)
        norms = [n for n, m in backbone.named_children() if isinstance(m, nn.LayerNorm)]
        if len(norms) != 1:
            raise ASTLoadError(f"expected exactly one final LayerNorm under {BACKBONE}, found {norms}")
        return f"{BACKBONE}.{norms[0]}."

    # ---- construction ------------------------------------------------------------------------------------------------
    @classmethod
    def from_dir(cls, path, *, device: str = "cpu", seed: int = 42, feature_extractor=None, num_labels: int = 2):
        """Load a local snapshot with a fresh `num_labels` head. The head is initialised after seeding torch with `seed`."""
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        torch.manual_seed(seed)
        try:
            result = ASTForAudioClassification.from_pretrained(path, num_labels=num_labels, ignore_mismatched_sizes=True, output_loading_info=True)
        except TypeError:  # a version that rejects the argument: the name-independent shape audit below still applies
            torch.manual_seed(seed)
            result = ASTForAudioClassification.from_pretrained(path, num_labels=num_labels, ignore_mismatched_sizes=True)
        model, info = result if isinstance(result, tuple) else (result, None)
        shapes = shape_audit(checkpoint_shapes(path), model)
        library = _library_report(info)
        audited = audit_loading(library) if library is not None else {"missing": [], "unexpected": [], "mismatched": [], "ok": True,
                                                                       "non_head_new_or_reshaped": []}
        loading = {**audited, "source": "transformers loading info + shape audit" if library is not None else "shape audit only (no loading info)",
                   "library_info_available": library is not None, "shapes": shapes, "ok": bool(audited["ok"] and shapes["ok"])}
        if not loading["ok"]:
            raise ASTLoadError(f"unexpected load result (only {HEAD_TENSORS} may be new or reshaped): {loading}")
        extractor = feature_extractor or ASTFeatureExtractor.from_pretrained(path)
        return cls(model.to(device), extractor, loading)

    @classmethod
    def from_pinned(cls, *, device: str = "cpu", seed: int = 42, cache_dir=None):
        return cls.from_dir(pinned_snapshot("ast", cache_dir=cache_dir), device=device, seed=seed)

    # ---- freeze / unfreeze -------------------------------------------------------------------------------------------
    @property
    def n_blocks(self) -> int:
        return self._n_layers

    def last_blocks(self, n: int):
        if not 1 <= n <= self._n_layers:
            raise ValueError(f"n_blocks must be between 1 and {self._n_layers}")
        return [f"{self._layer_prefix}.{i}." for i in range(self._n_layers - n, self._n_layers)]

    def stage_prefixes(self, stage: str, n_blocks: int = 2):
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        if stage == "frozen":
            return []
        head = ["classifier."]
        return head if stage == "head" else head + [self._norm_prefix] + self.last_blocks(n_blocks)

    def set_stage(self, stage: str, n_blocks: int = 2):
        prefixes = self.stage_prefixes(stage, n_blocks)
        for name, p in self.model.named_parameters():
            p.requires_grad_(any(name.startswith(x) for x in prefixes))
        self.stage, self.stage_blocks = stage, n_blocks
        if not prefixes:
            return []
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        for x in prefixes:  # a prefix that matches nothing means the layout is not what this code assumes
            if not any(n.startswith(x) for n in trainable):
                raise ASTLoadError(f"stage '{stage}': no parameter starts with '{x}'")
        return trainable

    def _group(self, name: str) -> str:
        if name.startswith("classifier."):
            return "classifier"
        if name.startswith(self._norm_prefix):
            return "final_norm"
        marker = self._layer_prefix + "."
        if name.startswith(marker):
            return "block." + name[len(marker):].split(".")[0].zfill(2)
        return "embeddings" if name.startswith(f"{BACKBONE}.embeddings.") else "other"

    def parameter_report(self) -> dict:
        groups = {}
        for name, p in self.model.named_parameters():
            g = groups.setdefault(self._group(name), {"parameters": 0, "trainable": 0, "tensors": 0})
            g["parameters"] += p.numel()
            g["tensors"] += 1
            g["trainable"] += p.numel() if p.requires_grad else 0
        total = sum(g["parameters"] for g in groups.values())
        trainable = sum(g["trainable"] for g in groups.values())
        return {"stage": self.stage, "total_parameters": total, "trainable_parameters": trainable,
                "trainable_fraction": trainable / total, "groups": dict(sorted(groups.items())),
                "layout": {"block_prefix": self._layer_prefix, "final_norm_prefix": self._norm_prefix, "blocks": self._n_layers},
                "trainable_names": sorted(n for n, p in self.model.named_parameters() if p.requires_grad)}

    def param_groups(self, *, head_lr: float, backbone_lr: float, weight_decay: float = 0.01):
        """Optimizer groups for the Trainer: new head at head_lr, unfrozen pretrained parts at backbone_lr."""
        head, backbone = [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                (head if name.startswith("classifier.") else backbone).append(p)
        groups = [{"params": ps, "lr": lr, "weight_decay": weight_decay, "name": label}
                  for label, ps, lr in (("head", head, head_lr), ("backbone", backbone, backbone_lr)) if ps]
        if not groups:
            raise ValueError("no trainable parameters; call set_stage first")
        return groups

    def digests(self) -> dict:
        """SHA-256 of each parameter group's bytes: proves which groups changed (and which did not) after training steps."""
        hashers = {}
        for name, p in sorted(self.model.named_parameters()):
            hashers.setdefault(self._group(name), hashlib.sha256()).update(p.detach().cpu().contiguous().numpy().tobytes())
        return {k: v.hexdigest() for k, v in sorted(hashers.items())}

    # ---- forward -----------------------------------------------------------------------------------------------------
    def extract_features(self, waveforms, valid_samples):
        """The CPU feature step alone (also used by the probe to time it separately from the GPU step)."""
        clips = [w[:int(n)].detach().float().cpu().numpy() for w, n in zip(waveforms, valid_samples.tolist())]
        return self.feature_extractor(clips, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"]

    def forward(self, waveforms, valid_samples):
        features = self.extract_features(waveforms, valid_samples)
        return self.model(input_values=features.to(waveforms.device)).logits

    def trainable_names(self):
        return list_trainable(self)
