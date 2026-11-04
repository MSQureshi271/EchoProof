"""R1-U2 GPU-hours budget for the eight final fits plus extraction and sealed-evaluation scoring.

estimate_budget() is a pure function of (measured throughputs, manifest workload, plan). Every line item says whether its rate was
`measured` (on the T4 probe) or `assumed` (not yet measured; replace after the owning task measures it). The budget uses the maximum
epochs, so early stopping can only make it smaller. A reserve (default 25%, from the plan) is added on top.
"""
from __future__ import annotations

from math import ceil
from pathlib import Path

DEFAULT_PLAN = {
    "ast_warmup_epochs": 2,                    # head-only epochs, once per protocol (variants branch from the warm-up checkpoint)
    "ast_partial_epochs": 8,                   # maximum partial fine-tuning epochs per variant
    "ast_variants": {"B": 2, "A": 1},          # B: baseline + one targeted alternative; A reuses the B-selected configuration
    "wavlm_crops_per_train_recording": 4,      # fixed crop bank for the cached-embedding head
    "wavlm_head_fits": {"B": 2, "A": 1},
    "wavlm_head_minutes_per_fit": 5.0,         # ASSUMED: a two-layer head on cached embeddings
    "bigru_fits": {"B": 2, "A": 1},
    "bigru_epochs": 20,
    "bigru_minutes_per_epoch": 1.5,            # ASSUMED until R2-M2/R3-M1 measure it on real chunks
    "corruption_conditions": 4,                # MP3 64/128 kbps, noise 20/10 dB
}


def workload_from_manifests(splits_dir, chunks_path) -> dict:
    """Counts per protocol from the locked manifests: training recordings and chunks per partition / validation role."""
    from echoproof.contracts import Chunk, SplitAssignment, read_parquet
    chunks = read_parquet(str(chunks_path), Chunk)
    out = {}
    for protocol in ("A", "B"):
        assignments = read_parquet(str(Path(splits_dir) / f"protocol_{protocol}_v1.parquet"), SplitAssignment)
        role = {a.recording_id: a.validation_role for a in assignments}
        mine = [c for c in chunks if c.protocol_id == f"protocol:{protocol}"]
        w = {"train_recordings": len({c.recording_id for c in mine if c.partition == "train"}),
             "train_chunks": sum(c.partition == "train" for c in mine), "selection_chunks": 0, "calibration_chunks": 0,
             "operating_chunks": 0, "test_chunks": sum(c.partition == "test" for c in mine)}
        for c in mine:
            if c.partition == "validation":
                w[f"{role[c.recording_id]}_chunks"] += 1
        out[protocol] = w
    return out


def _item(name, protocol, units, rate_per_second, basis, detail):
    return {"item": name, "protocol": protocol, "hours": units / rate_per_second / 3600 if rate_per_second else 0.0,
            "basis": basis, "detail": detail}


def estimate_budget(measured: dict, workload: dict, plan: dict | None = None, *, reserve: float = 0.25, session_hours: float = 3.0,
                    accounts: int = 4) -> dict:
    plan = {**DEFAULT_PLAN, **(plan or {})}
    need = ("ast_head_train_cps", "ast_partial_train_cps", "ast_eval_cps", "wavlm_cps")
    if any(not measured.get(k) or measured[k] <= 0 for k in need):
        raise ValueError(f"measured throughputs required and positive: {need}")
    m, items = measured, []
    for protocol in ("B", "A"):
        w = workload[protocol]
        variants = plan["ast_variants"][protocol]
        eval_cps = m["ast_eval_cps"]
        epochs = plan["ast_warmup_epochs"] + plan["ast_partial_epochs"]
        items.append(_item("AST head warm-up", protocol, plan["ast_warmup_epochs"] * w["train_recordings"], m["ast_head_train_cps"], "measured",
                           f"{plan['ast_warmup_epochs']} epochs x {w['train_recordings']} recordings, once per protocol"))
        items.append(_item("AST partial fine-tuning", protocol, variants * plan["ast_partial_epochs"] * w["train_recordings"],
                           m["ast_partial_train_cps"], "measured", f"{variants} variant(s) x {plan['ast_partial_epochs']} epochs x {w['train_recordings']} recordings"))
        items.append(_item("AST selection-validation each epoch", protocol,
                           (plan["ast_warmup_epochs"] + variants * plan["ast_partial_epochs"]) * w["selection_chunks"], eval_cps, "measured",
                           f"{plan['ast_warmup_epochs'] + variants * plan['ast_partial_epochs']} epochs x {w['selection_chunks']} chunks"))
        items.append(_item("AST calibration + operating + test scoring", protocol,
                           variants * (w["calibration_chunks"] + w["operating_chunks"] + w["test_chunks"]), eval_cps, "measured",
                           f"{variants} fit(s) x {w['calibration_chunks'] + w['operating_chunks'] + w['test_chunks']} chunks"))
        items.append(_item("AST corrupted-test scoring (protocol D)", protocol,
                           variants * plan["corruption_conditions"] * w["test_chunks"], eval_cps, "measured",
                           f"{variants} fit(s) x {plan['corruption_conditions']} conditions x {w['test_chunks']} test chunks"))
        items.append({"item": "BiGRU training", "protocol": protocol, "basis": "assumed",
                      "hours": plan["bigru_fits"][protocol] * plan["bigru_epochs"] * plan["bigru_minutes_per_epoch"] / 60,
                      "detail": f"{plan['bigru_fits'][protocol]} fit(s) x {plan['bigru_epochs']} epochs x {plan['bigru_minutes_per_epoch']} min"})
        items.append({"item": "WavLM head training", "protocol": protocol, "basis": "assumed",
                      "hours": plan["wavlm_head_fits"][protocol] * plan["wavlm_head_minutes_per_fit"] / 60,
                      "detail": f"{plan['wavlm_head_fits'][protocol]} fit(s) x {plan['wavlm_head_minutes_per_fit']} min"})
    # WavLM embeddings are cached once per (recording, window); Protocol A's recordings contain Protocol B's, so use the larger set.
    union = {k: max(workload["A"][k], workload["B"][k]) for k in workload["A"]}
    val = union["selection_chunks"] + union["calibration_chunks"] + union["operating_chunks"]
    crops = union["train_recordings"] * plan["wavlm_crops_per_train_recording"]
    items.append(_item("WavLM embedding extraction (shared cache)", "A+B", crops + val + union["test_chunks"], m["wavlm_cps"], "measured",
                       f"{crops} train-crop-bank + {val} validation + {union['test_chunks']} test chunks"))
    items.append(_item("WavLM corrupted-test extraction (protocol D)", "A+B", plan["corruption_conditions"] * union["test_chunks"], m["wavlm_cps"], "measured",
                       f"{plan['corruption_conditions']} conditions x {union['test_chunks']} test chunks"))
    subtotal = sum(i["hours"] for i in items)
    measured_hours = sum(i["hours"] for i in items if i["basis"] == "measured")
    total = subtotal * (1 + reserve)
    for i in items:
        i["hours"] = round(i["hours"], 3)
    return {"items": items, "subtotal_hours": round(subtotal, 2), "measured_hours": round(measured_hours, 2),
            "assumed_hours": round(subtotal - measured_hours, 2), "reserve_fraction": reserve, "total_hours_with_reserve": round(total, 2),
            "sessions_needed": ceil(total / session_hours), "session_hours": session_hours,
            "hours_per_account": round(total / accounts, 2), "accounts": accounts,
            "note": "Maximum epochs are budgeted; early stopping can only reduce this. Assumed items must be replaced by measurements."}
