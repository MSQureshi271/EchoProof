"""Build and verify the Protocol A lock record (`locks/A_lock.json`) for the four R4-N1 development bundles.

The record binds each bundle (identity, component hashes, serialized decision rule) to the evidence that produced it: the calibration or threshold run,
the operating-role results at the rule, and the replay or parity report. Every fact is read from files and re-verified (hashes, flags, thresholds);
nothing is typed in except the limitations and the user-reported policy items. The lock does not select a deployment model, does not touch the Protocol B
lock and does not read any Protocol A test data. Modelled on `scripts/r3_all/gr3_lock.py`; B sources are not edited.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from echoproof.bundle import load_bundle

LOCK_ID = "lock:protocol_a_gr4_v1"
SPLIT_SHA256 = "a9075367c89295a953eab1de7dc9f2380b14b3f628c2fabca42aaff94a6201a5"
B_LOCK = "locks/B_lock_gr3.json"
B_LOCK_SHA256 = "6262d24a8d36338ebe8c9004e0f428c17d85239a668394eff81ca88b276f5937"
ROLE_COUNTS = {"selection": 2500, "calibration": 1200, "operating": 1300}
MODELS = ("WavLM", "AST", "LR", "BiGRU")
BUNDLES = {"WavLM": "bundles/r4_n1_wavlm_head_A_calibrated", "AST": "bundles/r4_n1_ast_A_candidate_20261115",
           "LR": "bundles/r4_n1_lr_A_raw_threshold", "BiGRU": "bundles/r4_n1_bigru_A_raw_threshold"}
MODEL_VERSIONS = {"WavLM": "wavlm_head:A", "AST": "ast_partial:A", "LR": "logistic_regression:A", "BiGRU": "bigru:A"}


class ALockError(ValueError):
    """The evidence for the A lock is missing, inconsistent or unverified."""


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def b_lock_matches(root) -> bool:
    """The Protocol B lock is unchanged. Its recorded hash is of the CRLF form it was written in; a checkout may hold LF, so either form is accepted."""
    data = (Path(root) / B_LOCK).read_bytes()
    forms = (data, data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    return any(hashlib.sha256(f).hexdigest() == B_LOCK_SHA256 for f in forms)


def jload(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def need(condition, message):
    if not condition:
        raise ALockError(message)


def ref(root: Path, rel: str) -> dict:
    """A hashed reference to a project file."""
    path = Path(root) / rel
    need(path.is_file(), f"missing evidence file: {rel}")
    return {"path": rel, "sha256": sha(path)}


def bundle_record(root: Path, model: str) -> dict:
    root, rel = Path(root), BUNDLES[model]
    path = root / rel
    need(path.is_dir(), f"{model}: bundle folder is not present locally: {rel}")
    bundle = load_bundle(path)                                   # verifies every component hash
    need(bundle.spec.model_version == MODEL_VERSIONS[model], f"{model}: model version {bundle.spec.model_version!r}")
    extra = bundle.raw_metadata.get("extra", {})
    need(extra.get("promoted", False) is False and extra.get("locked", False) is False, f"{model}: bundle claims promotion or a lock")
    rule = bundle.spec.decision_rule.to_dict()
    need(rule.get("kind") == "threshold" and isinstance(rule.get("threshold"), float), f"{model}: no threshold rule")
    return {"bundle_path": rel, "bundle_id": bundle.spec.bundle_id, "model_version": bundle.spec.model_version, "bundle_json_sha256": sha(path / "bundle.json"),
            "preprocessing_hash": bundle.spec.preprocessing_hash, "decision_rule": rule,
            "components": {k: {"sha256": h, "bytes": Path(bundle.components[k]).stat().st_size} for k, h in sorted(bundle.component_hashes.items())},
            "bundle_extra": {k: extra[k] for k in ("stage", "protocol", "promoted", "locked", "score_type", "calibration_accepted") if k in extra}}


def _wavlm(root: Path, record: dict) -> dict:
    runs = sorted(p for p in (root / "runs/r4/r4_n1_wavlm").iterdir() if p.is_dir() and (p / "handoff/handoff.json").is_file())
    need(len(runs) == 1, "WavLM: expected exactly one finished run folder")
    run = runs[0]
    rel = run.relative_to(root).as_posix()
    handoff, replay = jload(run / "handoff/handoff.json"), jload(run / "handoff/replay.json")
    threshold, metrics, evidence = jload(run / "operating/threshold.json"), jload(run / "operating/metrics.json"), jload(run / "operating/evidence.json")
    need(handoff["bundle_json_sha256"] == record["bundle_json_sha256"] == replay["bundle_json_sha256"], "WavLM: hand-off or replay is for a different bundle")
    need(handoff["replay_passed"] is True and replay["passed"] is True and handoff["a_test_read"] is False and handoff["locks_changed"] is False, "WavLM: replay failed or test data read")
    need(evidence["status"] == "complete" and evidence["test_data_read"] is False and evidence["fit_on_operating"] is False, "WavLM: operating evidence is incomplete")
    need(threshold["decision_rule"]["threshold"] == record["decision_rule"]["threshold"], "WavLM: threshold differs from the bundle")
    full = metrics["rows"]["D_selected_at_its_optimum"]["full"]
    raw = metrics["rows"]["A_raw_at_0_5"]["full"]
    tiers = replay["tiers"]
    return {
        "calibration": {"mapping": threshold["selected_mapping"], "parameters": threshold["selected_parameters"], "accepted": threshold["calibration_accepted"], "score_type": threshold["score_type"]},
        "operating_role": {**{k: full[k] for k in ("macro_f1", "genuine_fpr", "tts_recall", "tp", "fp", "tn", "fn")}, "raw_at_0_5": {k: raw[k] for k in ("macro_f1", "genuine_fpr", "fp", "fn")},
                           "threshold_plateau": threshold["plateau"], "selected_on": threshold["selected_on"]},
        "replay": {"kind": "waveform replay through the bundle (2 tiers), Colab T4", "passed": True, "recordings": replay["recordings"], "roles": replay["roles"],
                   "tolerance": replay["tolerance"], "max_score_difference": max(t["max_score_difference"] for t in tiers.values()),
                   "decision_flips": len({f["recording_id"] for f in replay["flips"]}), "environment": replay["environment"]},
        "evidence": {"run": rel, **{n: ref(root, f"{rel}/{n}") for n in ("handoff/handoff.json", "handoff/replay.json", "handoff/bundle_packaging.json", "operating/evidence.json",
                                                                          "operating/threshold.json", "operating/metrics.json", "fit/calibration.json")},
                     "recipe": ref(root, "configs/r4_n1_wavlm_calibration_a.json"), "source_bundle_json": ref(root, "bundles/r4_u1_wavlm_head_A/bundle.json")}}


def _ast(root: Path, record: dict) -> dict:
    run = "runs/r4/r4_n1_ast/candidate_20261115_approved_recipe"
    op, fit, ver = jload(root / run / "operating.json"), jload(root / run / "fit.json"), jload(root / run / "local_verification.json")
    receipts = sorted((root / "reports/r4/r4_n1_ast_parity").glob("*.json"))
    need(len(receipts) == 1, "AST: expected exactly one parity receipt")
    parity = jload(receipts[0])
    need(parity["bundle_sha256"] == record["bundle_json_sha256"], "AST: the parity receipt is for a different bundle")
    need(parity["calibration_sha256"] == record["components"]["calibration/ast_a.json"]["sha256"], "AST: the parity receipt is for a different calibration")
    need(parity["passed"] is True and not parity["decision_flips"], "AST: parity did not pass")
    need(op["rule"]["threshold"] == record["decision_rule"]["threshold"], "AST: threshold differs from the bundle")
    need(ver["saved_scores_exact"] is True and ver["saved_decisions_exact"] is True and ver["B_lock_bytes_unchanged"] is True, "AST: local verification failed")
    m = op["operating_metrics"]
    return {
        "calibration": {"mapping": op["method"], "parameters": op["parameters"], "accepted": op["accepted"], "score_type": op["score_type"]},
        "operating_role": {"macro_f1": m["macro_f1"], "genuine_fpr": m["fpr"], "tts_recall": m["tts_recall"], "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
                           "threshold_plateau": op["threshold_plateau"], "selected_on": "operating role (development result, not independent)",
                           "brier_difference_bootstrap_includes_zero": bool(op["bootstrap"]["brier"]["differences"]["calibrated_vs_raw"]["ci_lower"] < 0 < op["bootstrap"]["brier"]["differences"]["calibrated_vs_raw"]["ci_upper"])},
        "replay": {"kind": "sampled CUDA fp16 parity (all chunks of 50 stratified selection recordings), recipient run on Colab", "passed": True, "recordings": 50, "chunks": parity["items"],
                   "tolerance": parity["atol_fp16"], "max_score_difference": parity["fp16"]["max_abs_difference"], "decision_flips": len(parity["decision_flips"]),
                   "approved_flag_in_receipt": parity["approved"], "reference_precision": parity["reference_precision"],
                   "operating_scores_recomputed_from_saved_calibration": ver["operating_recordings_recomputed"]},
        "evidence": {"run": run, **{n: ref(root, f"{run}/{n}") for n in ("operating.json", "fit.json", "local_verification.json", "prospective.json", "decisions.json")},
                     "parity_receipt": ref(root, receipts[0].relative_to(root).as_posix()), "recipe": ref(root, "configs/r4_n1_ast_calibration_a.json"),
                     "source_bundle_json": ref(root, "bundles/r4_u1_ast_partial_A/bundle.json")}}


def _raw(root: Path, model: str, record: dict) -> dict:
    key = model.lower()
    run = f"runs/r4/r4_n1_{key}"
    res, pack, par = jload(root / run / "operating/operating_results.json"), jload(root / run / "handoff/bundle_packaging.json"), jload(root / run / "handoff/waveform_parity.json")
    for name, doc in (("packaging", pack), ("parity", par)):
        need(doc["bundle_json_sha256"] == record["bundle_json_sha256"], f"{model}: {name} report is for a different bundle")
    need(par["passed"] is True and par["test_partition_read"] is False and pack["test_partition_read"] is False, f"{model}: parity failed or test data read")
    need(res["test_data_read"] is False and res["mapping"] == "identity" and res["mapping_accepted"] is False, f"{model}: unexpected calibration outcome")
    need(res["threshold_rule"]["threshold"] == record["decision_rule"]["threshold"], f"{model}: threshold differs from the bundle")
    replay = pack["saved_score_parity"]
    need(replay["status"] == "passed", f"{model}: saved-score replay failed")
    return {
        "calibration": {"mapping": "identity", "parameters": {}, "accepted": False, "score_type": "model_score"},
        "operating_role": {k: replay[k] for k in ("macro_f1", "genuine_fpr", "tp", "fp", "tn", "fn")} | {"tts_recall": replay["tp"] / (replay["tp"] + replay["fn"]),
                           "threshold_plateau": res["threshold_metrics"]["plateau"], "selected_on": "operating role (development result, not independent)"},
        "replay": {"kind": "waveform parity through the bundle (2 tiers)", "passed": True, "recordings": par["recordings"], "roles": par["roles"], "tolerance": par["tolerance"],
                   "max_score_difference": max(t["max_score_difference"] for t in par["tiers"].values()), "decision_flips": len({f["recording_id"] for f in par["flips"]}),
                   "flips_explained": all(f["within_tolerance_of_threshold"] for f in par["flips"]), "environment": par["environment"]},
        "evidence": {"run": run, **{n: ref(root, f"{run}/{n}") for n in ("operating/operating_results.json", "handoff/bundle_packaging.json", "handoff/waveform_parity.json",
                                                                          "fit/calibration.json", "input_audit.json")},
                     "recipe": ref(root, "configs/r4_n1_calibration_a.json")}}


def git(root: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


LIMITATIONS = [
    "All four bundles are development bundles: none is promoted, none has an independent review record, and the calibration recipe was accepted by the team on the project owner's report (no individual approval record in the repository).",
    "Every threshold was chosen on the operating role, so the operating figures are development results, not independent test performance. The A test partition was never read, and no model choice used it.",
    "Protocol A has no unseen generator and every operating figure is in-distribution; nothing here measures generalisation (that is R5, on the sealed test partition).",
    "The operating role has 260 genuine recordings from 13 speakers: 0 false alarms still allows a true rate of about 1.9%. WavLM-A separates the operating role perfectly; its threshold equals the lowest TTS operating score and the real margin is an interval (0.5689, 0.6254], not a point.",
    "Candidate thresholds are observed operating scores, so the recording that sits on a threshold is decided by float rounding: the Colab parity runs had one explained flip each for LR-A and BiGRU-A (the flipped recording is within tolerance of the threshold).",
    "LR-A and BiGRU-A keep raw model scores (calibration rejected). All BiGRU-A false alarms come from one speaker (speaker:6454, 19 of 20 recordings); LR-A is weak overall (genuine FPR 18.1%).",
    "AST-A calibration was accepted by the point-estimate rule, but the bootstrap interval of the Brier difference includes zero. AST-A is verified on CUDA fp16 only, by a sampled parity (all chunks of 50 selection recordings) whose receipt carries approved=false; its loader refuses CPU and other precisions.",
    "Parity checks reuse canonical audio from the shards and so do not test resampling of original files; the verified LR feature path used librosa 0.11.0 on Colab.",
    "Single seed per model. BiGRU-A is the best of 20 selection epochs and its run was resumed once (not bit-exact); recorded source-hash differences for acoustic.py and bigru_waveform.py remain.",
    "No latency or memory measurement was made for any A bundle, and the app and CLI cannot route A bundles yet (R4-N2). Production stays with the Protocol B lock (locks/B_lock_gr3.json), which this lock does not change.",
]


def build_lock(root, *, allow_dirty: bool = False) -> dict:
    root = Path(root)
    need(b_lock_matches(root), "the Protocol B lock changed")
    candidates = {}
    for model in MODELS:
        record = bundle_record(root, model)
        facts = _wavlm(root, record) if model == "WavLM" else _ast(root, record) if model == "AST" else _raw(root, model, record)
        need(facts["operating_role"]["macro_f1"] > 0 and facts["replay"]["passed"] is True, f"{model}: evidence is not usable")
        need(record["bundle_extra"].get("score_type") in (None, facts["calibration"]["score_type"]), f"{model}: score type differs between bundle and evidence")
        candidates[model] = {**record, **facts}
    status = git(root, "status", "--porcelain")
    need(allow_dirty or not status, "the working tree is not clean; commit first or pass --allow-dirty (the dirty entries are then hashed in the record)")
    src = {p.relative_to(root).as_posix(): sha(p) for p in sorted((root / "src/echoproof").rglob("*.py"))}
    scripts = {p.relative_to(root).as_posix(): sha(p) for p in sorted((root / "scripts").glob("r4_*.py"))}
    return {
        "schema_version": 1, "lock_id": LOCK_ID, "protocol": "A", "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "supersedes_note": "locks/B_lock_gr3.json (Protocol B, production candidates) is not changed by this lock.",
        "b_lock": {"path": B_LOCK, "sha256": B_LOCK_SHA256, "sha256_of_file_as_checked_out": sha(root / B_LOCK), "unchanged": True,
                   "note": "the recorded hash is of the CRLF form the lock was written in; the checkout may hold LF"},
        "split": {"split_hash": SPLIT_SHA256, "split_version": "protocol_A_v1", "split_file_sha256": sha(root / "data/splits/protocol_A_v1.parquet"),
                  "roles": ROLE_COUNTS, "components": ["genuine", "melo", "vits", "xtts", "chatterbox"],
                  "r4_s1_manifest": {"path": "data/features/r4_s1_A/r4_s1_manifest.json", "sha256": sha(root / "data/features/r4_s1_A/r4_s1_manifest.json")}},
        "calibration_procedure": {"recipe_status": "team-agreed on the project owner's report, 15 November 2026; no individual approval record",
                                  "summary": "identity, temperature and affine on log-odds; leave-one-speaker-out per-speaker log loss on the calibration role; simplest within one standard error; strict acceptance on the operating role; macro-F1 threshold with the contract tie-breaks; no FPR cap, prevalence correction or abstention",
                                  "outcomes": {m: {"mapping": c["calibration"]["mapping"], "accepted": c["calibration"]["accepted"], "score_type": c["calibration"]["score_type"]} for m, c in candidates.items()}},
        "candidates": candidates,
        "production_selection": {"selected": None, "note": "This lock does not select a deployment model. Production remains the Protocol B WavLM bundle of locks/B_lock_gr3.json; any use of an A bundle is decided in R4-N2 or later."},
        "review_status": "no independent review recorded for R4-S1, R4-U0, R4-U1, R4-M1 or R4-N1",
        "code": {"git_head": git(root, "rev-parse", "HEAD"), "working_tree_dirty": bool(status), "uncommitted_entries_sha256": hashlib.sha256(status.encode()).hexdigest(),
                 "uncommitted_entries": status.splitlines(), "src_echoproof_sha256": src, "scripts_r4_sha256": scripts, "python": sys.version.split()[0]},
        "test_access_audit": {"audit_status": "partially_verified", "test_partition_accessed": False,
                              "note": "Every run and report named here records that no A test recording was read, and the split loader withholds test rows by default. Earlier work by other owners was not re-audited and filesystem access was not monitored (as in the B locks)."},
        "limitations": LIMITATIONS}


def verify_record(root, record: dict) -> None:
    """Re-hash every file the record names and re-check the bundles; raises ALockError on any difference, missing candidate or flag."""
    root = Path(root)
    need(record.get("lock_id") == LOCK_ID and record.get("protocol") == "A" and record["split"]["split_hash"] == SPLIT_SHA256, "wrong lock identity or split")
    need(record["test_access_audit"]["test_partition_accessed"] is False, "the record admits test access")
    need(set(record["candidates"]) == set(MODELS), "the four candidates are not all present")
    need(record["b_lock"]["sha256"] == B_LOCK_SHA256 and b_lock_matches(root), "the Protocol B lock changed")
    for model, c in record["candidates"].items():
        bundle = load_bundle(root / c["bundle_path"])
        need(sha(root / c["bundle_path"] / "bundle.json") == c["bundle_json_sha256"], f"{model}: bundle.json changed")
        need({k: v["sha256"] for k, v in c["components"].items()} == dict(bundle.component_hashes), f"{model}: components differ from the record")
        need(bundle.spec.decision_rule.to_dict() == c["decision_rule"], f"{model}: decision rule differs")
        need(c["replay"]["passed"] is True and (c["replay"]["decision_flips"] == 0 or c["replay"].get("flips_explained") is True), f"{model}: unexplained replay flips")
        for name, item in c["evidence"].items():
            if isinstance(item, dict):
                need(sha(root / item["path"]) == item["sha256"], f"{model}: evidence file changed: {item['path']}")
