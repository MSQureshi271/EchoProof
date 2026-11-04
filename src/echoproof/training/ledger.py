"""Experiment ledger (reports/ledger.csv): one appended row per run, failed runs included."""
import csv
import os
from pathlib import Path

COLUMNS = ("run_id", "owner", "model", "protocol", "config_hash", "split_hash", "seed", "epochs_run", "best_epoch",
           "stop_reason", "val_macro_f1", "val_eer", "val_fpr", "train_minutes", "peak_vram_mb", "device", "status", "notes")


def append_ledger(path, row: dict) -> None:
    """Append one row; creates the header on first use; unknown keys are rejected so the columns stay shared."""
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown ledger columns: {sorted(unknown)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
