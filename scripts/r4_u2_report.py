"""Build Usman's R4-U2 draft from local saved evidence; no model execution."""
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/final"
USED = {}


def read(path):
    p = ROOT / path
    USED[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding="utf-8-sig"))


def verify(path, expected):
    p = ROOT / path
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(f"Evidence hash mismatch: {path}")
    USED[path] = actual


def main():
    run = "runs/r4/r4_u1_ast_ft_A_20261115T102637Z"
    report = read("reports/r4/r4_u1_ast_colab_20261115T122234Z.json")
    for name, field in (("evidence.json", "evidence_sha256"), ("selection.json", "selection_sha256"),
                        ("adaptation_evidence.json", "adaptation_evidence_sha256")):
        verify(f"{run}/{name}", report[field])
    parity_path = "reports/r4/" + report["parity_report"]
    verify(parity_path, report["parity_report_sha256"])
    parity = read(parity_path)
    evidence = read(f"{run}/evidence.json")
    adaptation = read(f"{run}/adaptation_evidence.json")
    resolved = read(f"{run}/resolved_config.json")
    selection = read(f"{run}/selection.json")
    contract = read(f"{run}_contract.json")
    for recorded, local in (("scripts/r4_u1_ast.py", "scripts/r4_u1_ast (2).py"),
                            ("src/echoproof/training/r4_u1_ast_train.py", "src/echoproof/training/r4_u1_ast_train.py"),
                            ("src/echoproof/training/r4_u1_ast_data.py", "src/echoproof/training/r4_u1_ast_data.py")):
        expected = contract["sources"][recorded]
        if hashlib.sha256((ROOT / local).read_bytes()).hexdigest() == expected:
            verify(local, expected)
        else:
            # Git checkout can translate LF to CRLF. Verify the actual committed
            # blob, never silently normalize a research source fingerprint.
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            blob = subprocess.check_output(["git", "show", f"{revision}:{recorded}"], cwd=ROOT)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError(f"Neither local source nor Git blob matches: {recorded}")
            USED[f"git:{revision}:{recorded}"] = expected
    for group in adaptation["frozen_groups"]:
        if not (adaptation["group_sha256_start"][group] == adaptation["group_sha256_best_checkpoint"][group]
                == adaptation["group_sha256_final_epoch"][group]):
            raise ValueError(f"Frozen group changed: {group}")
    verify(f"{run}/best.pt", parity["checkpoint_sha256"])
    for name, sha in evidence["output_sha256"].items():
        verify(f"{run}/{name}", sha)
    for name, item in parity["bundle_components"].items():
        verify("bundles/r4_u1_ast_partial_A/" + name, item["sha256"])
    if not parity["passed"] or adaptation["problems"] or not adaptation["frozen_groups_unchanged_final_and_best"]:
        raise ValueError("Adaptation/parity evidence does not support the draft")
    if not adaptation["unfrozen_groups_changed_best"] or not adaptation["unfrozen_groups_changed_final"]:
        raise ValueError("Missing changed-group evidence")
    wb = read("reports/r3/r3_u1_B/selection.json")
    wa = read("reports/r4/r4_u1_wavlm_A/selection.json")
    ab = read("reports/r3/r3_u2_20261112T134049Z.json")
    recipe = read("configs/r3_u1_wavlm_head_b.json")
    totals = []
    # Explicit paths avoid silently including unrelated teammate experiments.
    histories = [*(f"reports/r3/r3_u1_B/run_{v}/history.jsonl" for v in ("mean", "mean_std")),
                 "reports/r4/r4_u1_wavlm_A/run_mean_std/history.jsonl",
                 *(f"runs/r3/r3_u2_20261112T134049Z/r3_u2_20261112T134049Z-{v}/history.jsonl" for v in ("baseline", "lower_backbone_lr")),
                 "runs/r4/r4_u1_ast_warmup_A_20261115T065205Z/history.jsonl",
                 "runs/r4/r4_u1_ast_ft_A_20261115T074952Z/history.jsonl", f"{run}/history.jsonl"]
    for path in histories:
        p = ROOT / path
        if not p.is_file():
            totals.append({"path": path, "status": "missing"})
            continue
        USED[path] = hashlib.sha256(p.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        seconds = sum(row["seconds"] for row in rows)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Invalid timing: {path}")
        totals.append({"path": path, "epochs_recorded": len(rows), "epoch_loop_seconds": seconds,
                       "peak_vram_mb": max(row.get("peak_vram_mb", 0) for row in rows),
                       "device_kind": "CPU" if "wavlm" in path or "r3_u1_B" in path else "CUDA",
                       "status": "partial historical attempt" if "074952Z" in path else "saved epoch history"})
    facts = {"status": "draft_for_peer_review", "task": "R4-U2", "sources_sha256": USED,
             "ast_a_run": run, "ast_a_selected_epoch": selection["selected_epoch"],
             "ast_a_macro_f1": report["selection_macro_f1"]["baseline"], "compute": totals,
             "limits": ["timings cover recorded epoch loops, including validation; not complete project compute",
                        "no billing, account allocation, energy or full failed-attempt census verified",
                        "validation estimates, single seed, no sealed test results"]}
    metrics = evidence["paired_warmup_vs_adapted"]
    draft = f'''# R4-U2 — pretrained models and compute (draft for peer review)

Owner: Muhammad Usman Murtaza. Scope: page 8 and the compute portion of page 9 in REVISED_PLAN.md, R4-M2 / R4-U2. Pagination is provisional until the team assembles the report. The training-procedure portion of page 9 belongs to Muneeb. This draft does not establish formal run acceptance or a protocol lock.

## Page 8 — pretrained speech representations

EchoProof uses microsoft/wavlm-base-plus and MIT/ast-finetuned-audioset-10-10-0.4593 for recording-level genuine-versus-TTS detection. Canonical audio is mono 16 kHz. Evaluation uses four-second chunks and their arithmetic mean P(TTS) per recording. Checkpoint selection uses the selection role at threshold 0.5; calibration and operating roles have separate downstream purposes. Protocol B tests transfer to generators excluded from B training; Protocol A includes all five components in development. Their validation scores measure different conditions and should not be interpreted as a controlled improvement from B to A. [REVISED_PLAN.md, R1-U1, R3-U1/U2, Phase R4; docs/r3_u1_wavlm_head.md; docs/r3_u2_ast_finetune.md]

### Frozen WavLM encoder and trained classifier

The encoder stays frozen and produces cached embeddings. A 128-unit GELU classifier with dropout {recipe['model']['dropout']} and two output logits is trained with AdamW (learning rate 0.001, weight decay 0.01), batch 128, constant schedule and seed 42. Recording crop weights prevent recordings with more crops from dominating. B compares mean pooling (768 features) with mean plus standard deviation (1536 features); mean+std was selected. The head contains 98,690 parameters for mean and 196,994 for mean+std, calculated from the two linear layers including biases; these exclude the frozen encoder. [configs/r3_u1_wavlm_head_b.json; src/echoproof/models/wavlm_head.py]

B mean selection macro F1 was {wb['selection_macro_f1']['mean']:.6f}; mean+std was {wb['selection_macro_f1']['mean_std']:.6f}, selected at epoch {wb['selected_epoch']}. A reused mean+std and selected epoch {wa['selected_epoch']} with macro F1 {wa['selection_macro_f1']['mean_std']:.6f}. These are raw-score selection results. Direct waveform/cache parity and bundle checks support the exported representation path; sample parity does not prove equality on every possible recording. [reports/r3/r3_u1_B/selection.json; reports/r4/r4_u1_wavlm_A/selection.json; docs/r4_u1_wavlm_head.md]

### AST warm-up and partial adaptation

AST first trains its classifier while the backbone is frozen. Partial adaptation then trains blocks 10 and 11, the final normalization and classifier; embeddings and blocks 0–9 remain frozen. A fresh optimizer uses backbone learning rate 1e-5 and head learning rate 1e-4, weight decay 0.01, microbatch 8 with accumulation 2, FP16 with GradScaler, and warm-up followed by cosine decay. A uses 189 warm-up steps over eight epochs. Training takes one seeded four-second crop per training recording per epoch. [configs/r4_u1_ast_ft_a.json; completed run resolved_config.json and adaptation_evidence.json]

The A run records {resolved['total_parameters']:,} total and {resolved['trainable_parameters']:,} trainable parameters ({100*resolved['trainable_parameters']/resolved['total_parameters']:.2f}%). Its saved checksums verify unchanged frozen groups in the final and selected states and changed unfrozen groups. A separate gradient proof is retained. This is evidence of actual backbone adaptation, beyond training a new classifier. [runs/r4/r4_u1_ast_ft_A_20261115T102637Z/{{resolved_config,gradient_proof,adaptation_evidence}}.json]

| Selection-validation result | Warm-up macro F1 | Adapted macro F1 | Selected epoch |
|---|---:|---:|---:|
| AST-B baseline | {ab['adaptation']['baseline']['epoch0_macro_f1']:.6f} | {ab['adaptation']['baseline']['best_macro_f1']:.6f} | {ab['adaptation']['baseline']['best_epoch']} |
| AST-A | {adaptation['warmup_best_macro_f1']:.6f} | {report['selection_macro_f1']['baseline']:.6f} | {report['selected_epoch']} |

The paired A comparison covers {metrics['recordings']:,} identical selection recordings: {metrics['decision_changes_at_0.5']['fixed_by_adaptation']} decisions were corrected and {metrics['decision_changes_at_0.5']['broken_by_adaptation']} became incorrect at 0.5. Fifty selection chunks, including 25 short chunks and all five components, passed CUDA FP16 adapter parity with maximum score difference {parity['fp16']['max_abs_difference']}. FP32 CPU maximum difference was {parity['reported_not_gated']['fp32_cpu']['max_abs_difference']:.6f}, with zero sampled decision flips at 0.5; that does not establish parity at a future calibrated threshold. [reports/r4/r4_u1_ast_colab_20261115T122234Z.json; linked parity report]

Four cumulative GradScaler skips are recorded in A fine-tuning. The shared Trainer can report Infinity in epoch gradient norms, and its scheduler uses optimizer attempts rather than confirmed updates. These limitations remain visible; the isolated R2-U2 AMP-aware reporting correction did not change the shared Trainer or automatically apply to AST fine-tuning. No claim of eliminated overflow is made. [docs/r2_u2_acceptance.md; docs/r3_u2_ast_finetune.md; A history and adaptation evidence]

## Page 9 contribution — measured compute and limitations

AST-A ran on a Tesla T4 with torch 2.14.1+cu130 and transformers 5.22.1. The saved A fine-tuning history records peak allocated VRAM of {adaptation['peak_vram_mb']:.1f} MiB; this is a framework-reported allocation, not total GPU occupancy. WavLM heads train on CPU from cached representations, while encoder extraction and parity require separate accounting. [A Colab report; docs/r4_u1_wavlm_head.md]

| Saved history | Recorded epochs | Epoch-loop minutes | Device | Scope |
|---|---:|---:|---|---|
'''
    for item in totals:
        if "epoch_loop_seconds" in item:
            draft += f"| `{item['path']}` | {item['epochs_recorded']} | {item['epoch_loop_seconds']/60:.2f} | {item['device_kind']} | {item['status']} |\n"
    draft += '''
Epoch-loop timings include validation executed inside the training loop. They exclude unrecorded setup, model downloads, data staging, gradient-proof passes, packaging and later parity; they are neither billed GPU time nor a complete wall-clock total. The interrupted A attempt is shown separately and is not the selected run. R2 extraction, original failed warm-up work, R2-U2 Option B, feasibility runs and other retries also consumed resources; this table is a bounded subtotal and must not be labelled total project compute. A consolidated account of those stages and actual paid costs remains unavailable to this draft. Do not infer zero cost from missing billing evidence.

Single-seed selection estimates can be optimistic. The model/epoch choices use development data; these figures do not replace sealed evaluation, confidence intervals or R5 error analysis. No additional fitting, prediction generation or held-out test access was performed to prepare this draft.

## Review and integration checklist

- Peer reviewer must verify the descriptions against source and the linked evidence; no reviewer approval is implied here.
- Reconcile the compute contribution with Muneeb's page 9 training section and the team's account/time records. Preserve missing measurements as limitations if unavailable.
- Add the draft to the final report's numbering and bibliography; retain the source-bound facts file for audit.
- Final test, calibration, robustness and deployment claims belong to the relevant approved later-stage evidence and must not be inferred from these selection scores.
'''
    OUT.mkdir(parents=True, exist_ok=True)
    USED["scripts/r4_u2_report.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "r4_u2_pages_8_9_draft.md").write_text(draft, encoding="utf-8")
    (OUT / "r4_u2_facts.json").write_text(json.dumps(facts, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Verified {len(USED)} evidence identities; wrote draft and facts to {OUT}")


if __name__ == "__main__":
    main()
