# R4-M1 Colab workflow

`notebooks/R4_M1_Colab.ipynb` runs tests, A preflight, LR-A fit, BiGRU-A fit, and both validation exports in order. It requires a CUDA runtime, the project's existing dependencies, the canonical shard tar directory, the R4-S1 A feature files, and the frozen GR3 lock and LR/BiGRU bundle components on Drive. Sync the complete current project code and tests, including `configs/r4_m1_recipes.json`, before starting.

Set `ROOT`, `SHARDS`, and a persistent `RUN_ID` in the notebook. Outputs are unique direct children of `runs/r4`. The notebook refuses skipped tests or existing incomplete LR and export directories. If BiGRU training was interrupted, keep the same run ID, set `RESUME_BIGRU = True`, and rerun; the trainer verifies the contract and checkpoint state. Preserve inconsistent partial outputs for inspection. Resume is not bit-exact for cuDNN GRU dropout.

The notebook validates completed LR/export files on rerun, but an existing BiGRU directory requires explicit `RESUME_BIGRU=True`, including a completed run being rechecked. It does not update the ledger; record reviewed runs during the R4-M1 handoff step. No training, GPU integration test, or Drive preflight was performed locally because the local environment lacks PyTorch and the Drive data.
