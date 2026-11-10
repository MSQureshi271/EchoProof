# R3-M2 isolated LR ablation

Muneeb approved this experimental-only exception on 9 November 2026. The authorization binds `configs/r3_m2_ablation.json` and the current lock hashes. It permits two fixed-hyperparameter fits: the full 67-feature control and the 60-feature variant without the declared energy/activity-proxy group. The allowance is 60 minutes, checked at fit boundaries and after completion; an individual fit is not forcibly killed at the deadline. Failed/interrupted attempts count, and an existing run directory prevents automatic restart.

Run through `notebooks/R3_M2_Colab.ipynb` after its fixture and development-bundle parity checks pass. The notebook requires the frozen batch01 runtime and the verified Drive data. No scientific fit has been performed locally.

Outputs go to `experiments/r3_m2/runs/ablation01` with a separate ledger. Training uses only B train, and comparison uses only B selection. The script verifies the original LR's scores and coefficient table before fitting. No calibration, operating or test outcome selects a variant. No official model, existing bundle or lock is overwritten; any future adoption requires a separate team decision.

See `reports/r3/r3_m2_plan.md` for the implementation and validation status. Subhan's review remains pending.
