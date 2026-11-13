# LR calibration experiment

Experimental only. Compare raw scores, temperature scaling, and Platt scaling on
recording-level log-odds from the selected LR model. No model retraining, test
data, Protocol A, official bundle changes, or lock changes are permitted.

Use leave-one-speaker-out predictions on the calibration role, refitting both
calibrators in every fold. Select the method with lowest pooled out-of-fold log
loss; ties favor raw, then temperature. Report Brier and speaker-level results.
Fit the chosen method on all calibration recordings, then evaluate the operating
role once. Operating outcomes have already been seen in prior work and are not
an independent final test. Selection is exploratory and does not authorize
promotion. The calibration role has only 12 speakers; uncertainty can be large.

Platt fits an unregularized logistic map to clipped recording log-odds, with
positive slope required to preserve ranking. Temperature is fitted with the
existing project's bounds and clipping. Identical recording weights apply.
