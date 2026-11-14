# AST-B alternative calibrators — exploratory only

Prospective design authorized by the user on 14 November 2026. No changes to the
official R3-N1 recipe, candidate acceptance, checkpoint, threshold, or bundles.
Operating outcomes have already been examined. Neither operating evaluation nor
pooled cross-validation is independent confirmation; no promotion is authorized.

Use selected baseline epoch 1 from r3_u2_20261112T134049Z. Inputs are raw canonical
recording probabilities and labels, grouped by speaker. No audio, model execution,
test scores, Protocol A scores, or training are needed.

Fixed candidates: raw, existing temperature implementation (bounds .01–50),
positive-slope Platt, nonnegative-slope beta, and isotonic. Epsilon is 1e-7.
Platt and beta minimize equal-recording-weight binary NLL using analytic gradients,
L-BFGS-B, identity initialization, maxiter 2000, ftol 1e-12, gtol 1e-8.
Platt slope has numerical lower bound 1e-8; beta slopes have lower bound zero.
No regularization, tuning, or outcome-driven restarts. Nonconvergence stops the run.
Isotonic uses PAVA block event frequencies: these are the binary binomial-NLL
minimizers as well as squared-loss isotonic solutions. sklearn supplies the PAVA
fit and linear interpolation between fitted knots, constant extrapolation; output
is clipped to [1e-7,1-1e-7]. Report knot count and number of fitted probability levels.
All scores use epsilon clipping for NLL; Brier uses actual returned probabilities.

CV: calibration LOSO (12 deterministic folds, not meaningfully seed-dependent),
calibration grouped 4-fold (3 speakers each), pooled grouped 5-fold (5 speakers each),
seeds 42/43/44. Grouped folds shuffle sorted speaker IDs with numpy default_rng(seed),
then array_split. Verify both classes in training and validation, disjoint speakers,
and exactly one OOF prediction per recording. All methods share the folds.
Pooled data are calibration plus operating only: 25 speakers, 2000 recordings.

Metrics: NLL, Brier, AUROC, ECE with ten fixed equal-width bins and ten equal-count
bins. Equal-count bins use stable score sorting with recording-ID order as tie-break,
then array_split; tied probabilities can straddle bins. Report bin counts and ranges.
OOF AUROC can change even for strictly monotone methods because folds use different
maps. For a single full-fit map ranking is preserved except clipping/numerical ties;
isotonic can remove partitions by ties. No threshold-optimized OOF claims are made.

Uncertainty: paired speaker-cluster percentile bootstrap, 1000 resamples, seed 42,
same sampled speakers for all methods within a dataset. Report candidate-minus-raw
NLL, Brier, both ECE definitions. Recompute ECE per replicate. Fits/OOF predictions
are fixed: this does not capture refitting uncertainty, overlapping-CV dependence,
or multiple-comparison correction. Three CV seeds are not independent experiments.

Fit every candidate on all calibration data, evaluate operating once; no model
selection. Report fixed saved threshold and separate shared operating-threshold
search, fixed before these results. The latter is post-hoc exploratory, not a new
official threshold. Report current candidate rule (strict NLL improvement AND
nonworsening Brier) without accepting anything. Include per-speaker loss changes
and leave-one-speaker-out changes in the overall difference as influence diagnostics.

Meaningful-improvement gates: pooled OOF NLL and Brier lower in all three seeds;
paired NLL bootstrap upper limit below zero in all three seeds (conservative reading);
both point-estimate directions agree on operating. Parameter min/max/std and
optimizer boundary/convergence diagnostics are reported. The user did not define
a numerical parameter-stability tolerance: stability requires review, not an
invented cutoff. No candidate can be declared reliably better with that gate unresolved.

Outputs remain under this experiment, in unique runs. Save exact fold membership,
OOF/full-fit predictions, parameters, reliability tables, source/input hashes,
environment, verification results, and summary. Preserve originals and verify hashes
again afterward. Formal follow-up approval would be needed even for promising results.
