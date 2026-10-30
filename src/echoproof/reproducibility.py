"""Fixture-only P1-06 run and fresh-process verification."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml
from .artifacts import (atomic_write_new, canonical_json, create_run, digest, file_digest,
                        make_manifest, write_json)
from .environment import capture_environment
from .training.seeding import seed_everything


def _subprocess(code, args=()):
    result = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Fresh process failed: {result.stderr[-1200:]}")
    return json.loads(result.stdout)


def _seeded_sample(seed):
    code = """import json, random, sys
import numpy as np
import torch
from echoproof.training.seeding import seed_everything, loader_generator, seed_worker
seeds=seed_everything(int(sys.argv[1]))
loader=torch.utils.data.DataLoader(torch.arange(8), batch_size=4, shuffle=True, num_workers=0, generator=loader_generator(int(sys.argv[1])), worker_init_fn=seed_worker)
print(json.dumps({'seeds':seeds, 'python':random.random(), 'numpy':np.random.random(4).tolist(), 'torch':torch.rand(4).tolist(), 'loader':torch.cat(list(loader)).tolist()}))
"""
    return _subprocess(code, [seed])


def storage_probe(config, project_root):
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    run_id, folder = create_run(config.paths.run_root)
    try:
        seeds = seed_everything(config.seed)
        resolved = asdict(config)
        config_hash = digest(resolved)
        changed = dict(resolved)
        changed["seed"] = config.seed + 1
        if digest(changed) == config_hash:
            raise RuntimeError("config hash did not change after seed change")
        first, second = _seeded_sample(config.seed), _seeded_sample(config.seed)
        # Both processes use the same interpreter and CPU. Equality corresponds
        # to zero tolerance here; cross-device equivalence is not asserted.
        if first != second:
            raise RuntimeError("fresh seeded CPU fixture samples did not match")
        env = capture_environment(project_root)
        env.update({"schema_version": 1, "run_id": run_id, "task": "P1-06", "started_utc": started_utc,
                    "config_hash": config_hash, "contract_source_sha256": file_digest(Path(__file__).with_name("contracts.py")),
                    "seeds": seeds, "device_policy": config.device, "probe_device": "cpu", "precision_requested": config.probe.precision, "probe_precision": "fp32",
                    "loader_workers": 0, "ast_microbatch": config.probe.ast_microbatch})
        atomic_write_new(folder / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=True).encode("utf-8"))
        write_json(folder / "environment.json", env)
        atomic_write_new(folder / "events.jsonl", (canonical_json({"schema_version": 1, "event": "fixture_probe_started", "timestamp_utc": started_utc}) + "\n" + canonical_json({"schema_version": 1, "event": "fixture_probe_completed"}) + "\n").encode("utf-8"))
        write_json(folder / "metrics.json", {"schema_version": 1, "scope": "software_probe", "seeded_runs_equal": True,
                   "tolerance": 0, "config_hash_changed": True, "fixture_output": first})
        names = ["resolved_config.yaml", "environment.json", "events.jsonl", "metrics.json"]
        write_json(folder / "artifact_manifest.json", make_manifest(folder, names))
        verification = _subprocess("import json,sys; from echoproof.artifacts import verify_manifest; print(json.dumps(verify_manifest(sys.argv[1])))", [folder])
        write_json(folder / "status.json", {"schema_version": 1, "status": "success", "run_id": run_id,
                   "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
                   "verification": verification})
        return {"status": "success", "run_id": run_id, "run_dir": str(folder), "verification": verification}
    except Exception:
        # No success marker is written when work or fresh-process verification fails.
        raise
