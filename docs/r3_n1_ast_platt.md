# AST-B Platt calibration candidate

Completed local candidate: `runs/r3/r3_n1_ast_platt_candidates/20261114T150254Z-046f980d/`.
Fit: a=1.391048549348291, b=1.0973216440921831, epsilon=1e-7.
Operating threshold: 0.6316596289764391 (TTS when score >= threshold).
Operating log loss 0.18087007, Brier 0.05253199, macro F1 0.92283241.
The fit reproduced the earlier full-calibration Platt parameters; the threshold
was selected afresh with the shared procedure. This is not independent validation.
Local software receipt: `reports/r3/r3_n1_ast_platt_software_20261114T150254Z.json`.
Seven local tests passed; complete operating-score save/reload parity passed.
Recipient Colab verification subsequently passed on Tesla T4 with FP16, torch
2.14.1+cu130 and transformers 5.22.1: seven tests, 12 real operating recordings
and 57 chunks, including four recordings nearest the threshold. Chunk scores,
recording scores and Platt scores matched exactly; all sampled decisions matched.
Original artifacts were unchanged. The immutable receipt is published at
`docs/evidence/r3_n1_ast_platt/recipient_verification.json`; report SHA-256:
`645cb7bd093edb82f9e8d5e9083c31988ab9cb7f4253a16c0937ef64e1cb260d`.
Status: implementation and sampled CUDA/FP16 verification complete; independent
integration disposition and explicit promotion remain separate. CPU FP32 is not verified.

Authority and the scoped methodology deviation are recorded in the 14 November
2026 decision-log entry. The user reports team approval for this method; formal
recipient/integration approval is separate. Historical statistical gates did not
pass. Neither older results nor their status are amended.

`configs/r3_n1_ast_platt_candidate.json` binds the policy. The selected AST is
baseline epoch 1, checkpoint SHA-256
`f7efe6a547c53a310ed540f10be32fe44b148def168c2f5362cc1d22510f70ec`.
`scripts/r3_n1_ast_platt.py` verifies the original run, bundle and frozen metadata,
canonical split/chunk membership and historical exploratory source/artifact hashes.
It reuses that verified exploration's Platt fit, fitting only 960 calibration
recordings. The 1,040 operating recordings supply diagnostics and the shared
macro-F1 threshold search. No pooled fit, search over methods, test scoring or
model training occurs. Outputs are new directories; existing artifacts are hashed
again after execution.

Run from project root with the existing dependencies:

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_r3_n1_ast_platt.py -v
.venv/Scripts/python.exe -B scripts/r3_n1_ast_platt.py
```

Each candidate contains the pre-fit policy/source/input/environment receipt,
`fit.json`, `operating_predictions.parquet`, reliability JSON/SVG, metrics and
verification evidence, an independently loadable bundle, and file hashes.
The runner never updates an official lock or published bundle.

Explicit integration API:

```python
from echoproof.models.r3_n1_ast_platt_bundle import load_ast_platt_candidate
from echoproof.inference import predict_recording
bundle = load_ast_platt_candidate(candidate_directory / 'bundle',
                                  device='cuda', precision='fp16')
result = predict_recording(audio_bytes, bundle)
```

The existing inference pipeline decodes and chunks audio, scores chunks, computes
their mean, applies the Platt transform once, then applies the saved decision rule.
The existing AST adapter and shared inference implementation are unchanged.
Candidate loading is explicit; this does not alter automatic app dispatch or
authorize a public probability claim beyond the measured development domain.

The reference exports use CUDA FP16. CPU FP32 loading remains blocked pending
precision/decision verification. The recipient Colab receipt now verifies actual
AST waveform scoring on 12 recordings. Local synthetic-waveform tests exercised
real decoding and downstream logic with a fixture scorer; those tests alone did
not prove model parity. The Colab sample does not prove parity on every recording.
Do not claim FP32 or deployment acceptance without verification and approval.

The newer WavLM commit adds `load_inference_bundle` routing for LR and WavLM;
it does not route AST. This candidate uses the explicit loader above. Adding AST
to automatic app routing is a separate integration action, not silently included
in this publication. Historical source snapshots remain unchanged. Rebuilding a
candidate requires the full original exploratory artifacts and matching source
pins; curated Git evidence alone is not a replacement for the Drive artifacts.
