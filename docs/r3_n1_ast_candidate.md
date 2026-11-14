# R3-N1 AST-B candidate calibration

The prospective policy is `configs/r3_n1_ast_candidate.json`, authorized by
Muhammad Usman Murtaza on 14 November 2026 before execution. This is not formal
team acceptance or independent review. Log loss must strictly improve and Brier
must not worsen on operating validation; otherwise retain raw model scores.
Temperature is fitted only on calibration recordings after mean chunk aggregation.
The retained score receives the shared contract threshold search.

The local B lock describes provisional GR2 LR, not a final four-model GR3 seal.
It is preserved. No claim is made about unseen remote locks or test-access history.

Run `python scripts/r3_n1_ast_calibrate.py --project-root .` to produce a new,
exclusive directory under `runs/r3/r3_n1_ast_candidates/`. Never replace existing
outputs. The selected AST is baseline epoch 1 of `r3_u2_20261112T134049Z`.
Original run exports, checkpoint and bundle remain unchanged.

The implementation reuses shared temperature, metrics, threshold and bundle APIs
and the LR reliability-bin helper, not LR model selection or BiGRU calibrators.
It verifies hashes, split recording membership, canonical chunk membership,
speaker separation and aggregation before fitting. Saved policy, input hashes,
reliability bins, operating predictions and evidence accompany a candidate bundle.

`echoproof.models.r3_n1_ast_bundle.load_ast_candidate` explicitly loads the
candidate into the existing LoadedBundle interface. It does not change automatic
app dispatch or an official lock. FP16 is the recorded export reference; CPU FP32
loading is intentionally blocked pending waveform parity at the fitted threshold.
Recording-score reload parity is not waveform parity. Reliability JSON is plotting
data; a rendered reliability plot and full inference validation remain to finish.

Focused pytest fixtures are in `tests/test_r3_n1_ast_candidate.py`. Local pytest
was unavailable at implementation time; do not describe these tests as passed.
Formal review, complete regression testing, waveform precision/decision parity
and explicit team acceptance are required before promotion.
