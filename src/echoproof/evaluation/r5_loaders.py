"""R5-N1: a dedicated loader registry for the eight locked bundles (Option 2 of `docs/r5_bundle_loading_options.md`).

Nothing here edits a shared or locked file. The registry maps each bundle named in `locks/B_lock_gr3.json` and `locks/A_lock.json` to its verified
loader, its device and precision policy and its fail-closed rules, and it can verify both lock files and every bundle against the lock record before
any model is loaded. Scoring is not done here: the sealed runner calls the unchanged shared `predict_recording` / `analyze`.

Fail-closed rules: any hash difference, any unknown or missing lock candidate, any threshold override, any device the bundle is not verified on,
and any loaded bundle whose version, id or decision rule differs from the lock record raises. Nothing is skipped or defaulted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from echoproof.bundle import load_bundle
from echoproof.contracts import ContractError
from echoproof.evaluation import r4_n1_a_lock as a_lock

B_LOCK = a_lock.B_LOCK                      # locks/B_lock_gr3.json
A_LOCK = "locks/A_lock.json"
B_LOCK_SHA256 = a_lock.B_LOCK_SHA256        # of the CRLF form it was written in; an LF checkout is accepted
A_LOCK_SHA256 = "8613ff1e22e7bfa41d8547231c2b77f8066249646268a165b996e1392a1b29d1"   # as committed in 70d11ac (docs/r4_n1_a_lock.md quotes the prefix 8613ff1e22e7bfa4)
B_SPLIT_HASH = "0854de6b00ae059a858d60f18f95df3f1501f579ea3e2fdefbd8de076e2034d7"
A_SPLIT_HASH = a_lock.SPLIT_SHA256
SPLIT_FILES = {"A": "data/splits/protocol_A_v1.parquet", "B": "data/splits/protocol_B_v1.parquet"}
SPLIT_HASHES = {"A": A_SPLIT_HASH, "B": B_SPLIT_HASH}
LOCKS = {"B": B_LOCK, "A": A_LOCK}
MODELS = ("LR", "BiGRU", "WavLM", "AST")

# The B BiGRU bundle's calibration review is "pending". Its loader refuses that unless told otherwise; the sealed run tells it so, explicitly, because
# the plan scores every locked bundle once and a pending review is a recorded open item, not a reason to leave a locked candidate out.
ALLOW_PENDING_REVIEW_BIGRU_B = True
ACCEPTED_REVIEW_STATUSES = ("pending", "approved")


class LockError(ValueError):
    """A lock file, bundle or registry entry differs from what the lock records, or the lock itself is not the locked one."""


class DevicePolicyError(ValueError):
    """The requested device is not one the bundle is verified on."""


@dataclass(frozen=True)
class LoaderSpec:
    key: str                  # "B:LR" ...
    protocol: str             # "A" | "B"
    model: str                # key of the candidate in the lock
    model_version: str
    loader: str               # the verified loader that is called
    via_router: bool          # True when the shared router `load_inference_bundle` has a route for this bundle
    requires_cuda: bool       # verified on CUDA fp16 only
    precision: str | None     # "fp16" for AST, None otherwise
    cpu_only: bool = False    # LR features are CPU-only
    allow_pending_review: bool = False


def _spec(protocol, model, version, loader, **kw):
    return LoaderSpec(f"{protocol}:{model}", protocol, model, version, loader, kw.pop("via_router", False), kw.pop("requires_cuda", False), kw.pop("precision", None), **kw)


REGISTRY: dict[str, LoaderSpec] = {s.key: s for s in (
    _spec("B", "LR", "logistic_regression:B", "echoproof.inference.load_inference_bundle", via_router=True, cpu_only=True),
    _spec("B", "BiGRU", "bigru:B", "echoproof.models.r3_n1_bigru_bundle.load_calibrated_bigru_bundle", allow_pending_review=ALLOW_PENDING_REVIEW_BIGRU_B),
    _spec("B", "WavLM", "wavlm_head:B", "echoproof.inference.load_inference_bundle", via_router=True),
    _spec("B", "AST", "ast_partial:B", "echoproof.models.r3_n1_ast_platt_bundle.load_ast_platt_candidate", requires_cuda=True, precision="fp16"),
    _spec("A", "WavLM", "wavlm_head:A", "echoproof.models.r4_n1_wavlm_bundle.load_wavlm_bundle_a"),
    _spec("A", "AST", "ast_partial:A", "echoproof.models.r4_n1_ast_bundle.load_ast_a_bundle", requires_cuda=True, precision="fp16"),
    _spec("A", "LR", "logistic_regression:A", "echoproof.models.r4_n1_raw_bundles.load_lr_bundle_a", cpu_only=True),
    _spec("A", "BiGRU", "bigru:A", "echoproof.models.r4_n1_raw_bundles.load_bigru_bundle_a"),
)}


# ---- hashing and locks -------------------------------------------------------------------------------------------------------------------------

def sha256_file(path) -> str:
    return a_lock.sha(path)


def file_matches(path, expected: str) -> bool:
    """The file hashes to `expected` in its stored form or in its CRLF or LF form (a Windows and a Colab checkout may differ in line endings)."""
    data = Path(path).read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    return any(hashlib.sha256(f).hexdigest() == expected for f in (data, lf, lf.replace(b"\n", b"\r\n")))


def _need(condition, message):
    if not condition:
        raise LockError(message)


def read_lock(root, protocol: str) -> dict:
    path = Path(root) / LOCKS[protocol]
    _need(path.is_file(), f"lock file is missing: {LOCKS[protocol]}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_b_lock(root) -> dict:
    """The Protocol B lock is the locked file, names exactly the four registry candidates and the pinned split. Returns the record."""
    root = Path(root)
    path = root / B_LOCK
    _need(path.is_file(), f"lock file is missing: {B_LOCK}")
    _need(file_matches(path, B_LOCK_SHA256), "the Protocol B lock file differs from the locked hash")
    record = read_lock(root, "B")
    _need(record.get("protocol") == "B", "the B lock does not say Protocol B")
    _need(set(record["candidates"]) == set(MODELS), "the B lock candidates are not exactly LR, BiGRU, WavLM, AST")
    _need(record["split"]["split_hash"] == B_SPLIT_HASH, "the B lock names a different split hash")
    _need(record["test_access_audit"]["test_partition_accessed"] is False, "the B lock admits test access")
    return record


def verify_a_lock(root, *, deep: bool = True) -> dict:
    """The Protocol A lock is the locked file and (deep) every file it hashes still matches (`r4_n1_a_lock.verify_record`)."""
    root = Path(root)
    path = root / A_LOCK
    _need(path.is_file(), f"lock file is missing: {A_LOCK}")
    _need(file_matches(path, A_LOCK_SHA256), "the Protocol A lock file differs from the locked hash")
    record = read_lock(root, "A")
    _need(record.get("protocol") == "A" and record.get("lock_id") == a_lock.LOCK_ID, "the A lock identity is wrong")
    _need(set(record["candidates"]) == set(MODELS), "the A lock candidates are not exactly LR, BiGRU, WavLM, AST")
    _need(record["split"]["split_hash"] == A_SPLIT_HASH, "the A lock names a different split hash")
    _need(record["test_access_audit"]["test_partition_accessed"] is False, "the A lock admits test access")
    if deep:
        try:
            a_lock.verify_record(root, record)
        except Exception as exc:           # ALockError, BundleError, KeyError, OSError ...: all are a refusal
            raise LockError(f"the A lock no longer verifies: {exc}") from exc
    return record


def verify_locks(root, *, deep: bool = True) -> dict:
    return {"B": verify_b_lock(root), "A": verify_a_lock(root, deep=deep)}


def candidate_bundles(records: dict) -> list[tuple[LoaderSpec, dict]]:
    """Pair every registry entry with its lock candidate. The registry and the two locks must agree exactly: nothing missing, nothing extra."""
    expected = {(s.protocol, s.model) for s in REGISTRY.values()}
    found = {(p, m) for p, rec in records.items() for m in rec["candidates"]}
    _need(expected == found, f"registry and locks differ: only in registry {sorted(expected - found)}, only in locks {sorted(found - expected)}")
    out = []
    for spec in REGISTRY.values():
        record = records[spec.protocol]["candidates"][spec.model]
        _need(record["model_version"] == spec.model_version, f"{spec.key}: the lock says model version {record['model_version']!r}, the registry {spec.model_version!r}")
        out.append((spec, record))
    return out


def verify_bundle(root, spec: LoaderSpec, record: dict):
    """Re-verify one bundle against its lock record: bundle.json hash, every component hash, version, id, preprocessing hash and decision rule."""
    path = Path(root) / record["bundle_path"]
    _need(path.is_dir(), f"{spec.key}: bundle folder is not present: {record['bundle_path']}")
    _need(sha256_file(path / "bundle.json") == record["bundle_json_sha256"], f"{spec.key}: bundle.json differs from the lock")
    bundle = load_bundle(path)                                    # verifies every component hash against bundle.json
    _need({k: v["sha256"] for k, v in record["components"].items()} == dict(bundle.component_hashes), f"{spec.key}: components differ from the lock")
    _need(bundle.spec.model_version == spec.model_version and bundle.spec.bundle_id == record["bundle_id"], f"{spec.key}: identity differs from the lock")
    _need(bundle.spec.preprocessing_hash == record["preprocessing_hash"], f"{spec.key}: preprocessing hash differs from the lock")
    _need(bundle.spec.decision_rule.to_dict() == record["decision_rule"], f"{spec.key}: decision rule differs from the lock")
    if spec.allow_pending_review:
        status = bundle.raw_metadata.get("extra", {}).get("review_status")
        _need(status in ACCEPTED_REVIEW_STATUSES, f"{spec.key}: review status {status!r} is not one the explicit pending-review allowance covers")
    return bundle


def verify_all_bundles(root, records: dict) -> dict:
    return {spec.key: verify_bundle(root, spec, record) for spec, record in candidate_bundles(records)}


# ---- device and precision policy -----------------------------------------------------------------------------------------------------------------

def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def choose_device(spec: LoaderSpec, requested: str | None = None, *, cuda: bool | None = None) -> str:
    """AST bundles: CUDA fp16 only, never a fallback. LR: CPU. Others: the requested device, else CUDA when present, else CPU."""
    cuda = cuda_available() if cuda is None else cuda
    if spec.requires_cuda:
        if not cuda:
            raise DevicePolicyError(f"{spec.key} is verified on CUDA fp16 only and no CUDA device is available")
        if requested is not None and not str(requested).startswith("cuda"):
            raise DevicePolicyError(f"{spec.key} is verified on CUDA fp16 only; {requested!r} is refused")
        return requested or "cuda"
    if spec.cpu_only:
        return "cpu"
    if requested is not None:
        if str(requested).startswith("cuda") and not cuda:
            raise DevicePolicyError(f"{spec.key}: {requested!r} requested but no CUDA device is available")
        return requested
    return "cuda" if cuda else "cpu"


# ---- loading -----------------------------------------------------------------------------------------------------------------------------------

def load_entry(spec: LoaderSpec, bundle_path, *, device: str, threshold: float | None = None):
    """Call the verified loader for one registry entry. No threshold override is accepted for any of the eight bundles."""
    if threshold is not None:
        raise ContractError(f"{spec.key}: a threshold override is refused; the bundle's serialized decision rule is the official one")
    path = Path(bundle_path)
    if spec.key == "B:LR":
        from echoproof.inference import load_inference_bundle
        return load_inference_bundle(path)
    if spec.key == "B:WavLM":
        if str(device).startswith("cpu"):
            from echoproof.inference import load_inference_bundle      # the shared route (CPU only: the router has no device argument)
            return load_inference_bundle(path)
        from echoproof.models.r3_n1_wavlm_bundle import load_wavlm_bundle   # the function the router calls, with a device
        return load_wavlm_bundle(path, device=device)
    if spec.key == "B:BiGRU":
        from echoproof.models.r3_n1_bigru_bundle import load_calibrated_bigru_bundle
        return load_calibrated_bigru_bundle(path, device=device, allow_pending_review=ALLOW_PENDING_REVIEW_BIGRU_B)
    if spec.key == "B:AST":
        from echoproof.models.r3_n1_ast_platt_bundle import load_ast_platt_candidate
        return load_ast_platt_candidate(path, device=device, precision="fp16")
    if spec.key == "A:WavLM":
        from echoproof.models.r4_n1_wavlm_bundle import load_wavlm_bundle_a
        return load_wavlm_bundle_a(path, device=device)
    if spec.key == "A:AST":
        from echoproof.models.r4_n1_ast_bundle import load_ast_a_bundle
        return load_ast_a_bundle(path, device=device, precision="fp16")
    if spec.key == "A:LR":
        from echoproof.models.r4_n1_raw_bundles import load_lr_bundle_a
        return load_lr_bundle_a(path)
    if spec.key == "A:BiGRU":
        from echoproof.models.r4_n1_raw_bundles import load_bigru_bundle_a
        return load_bigru_bundle_a(path, device=device)
    raise LockError(f"no loader is registered for {spec.key!r}")


def check_loaded(spec: LoaderSpec, record: dict, loaded) -> None:
    """The object returned by the loader is the locked bundle: not a mock, same version, id and decision rule as the lock record."""
    _need(not loaded.is_mock, f"{spec.key}: the loader returned a mock bundle")
    _need(loaded.spec.model_version == spec.model_version and loaded.spec.bundle_id == record["bundle_id"], f"{spec.key}: loaded identity differs from the lock")
    _need(loaded.spec.decision_rule.to_dict() == record["decision_rule"], f"{spec.key}: loaded decision rule differs from the lock")


def load_verified(root, spec: LoaderSpec, record: dict, *, device: str | None = None, threshold: float | None = None, cuda: bool | None = None):
    """Verify the bundle against the lock, apply the device policy, load it, and check what came back. Returns (LoadedBundle, device)."""
    if threshold is not None:
        raise ContractError(f"{spec.key}: a threshold override is refused")
    chosen = choose_device(spec, device, cuda=cuda)
    verify_bundle(root, spec, record)
    loaded = load_entry(spec, Path(root) / record["bundle_path"], device=chosen)
    check_loaded(spec, record, loaded)
    return loaded, chosen
