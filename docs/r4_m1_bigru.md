# R4-M1 BiGRU-A training

`scripts/r4_m1_bigru.py` verifies the A inputs, freezes a run contract and trains a freshly initialized BiGRU on Protocol A. It uses the selected B recipe (hidden size 128, dropout 0.3, learning rate 0.0003, 20 epochs, patience 21, seed 42) with A training mel statistics. Each epoch is selected by recording-level macro F1 at raw threshold 0.5 on the selection role, with earliest epoch retained on ties. Calibration and operating audio are not scored during this step.

On a CUDA Colab runtime, with the R4-S1 Drive inputs and canonical shards available:

```bash
PYTHONPATH=src python scripts/r4_m1_bigru.py --root /path/to/project --shard-dir /path/to/shards --output /path/to/project/runs/r4/UNIQUE_BIGRU_RUN
```

For an interrupted run, use the same command and add `--resume`. Resume requires an unchanged recipe, input hashes, code hashes and runtime identity, plus matching history, latest checkpoint, best checkpoint and best-epoch prediction files. If a crash leaves inconsistent artifacts, preserve the run for inspection rather than overwrite it. A resumed GPU run is not guaranteed bit-exact: cuDNN's GRU inter-layer dropout state is not captured in the checkpoint.

The run writes the frozen contract, `latest.pt`, `best.pt`, `history.jsonl`, best selection predictions, status and training evidence. R4-M1 Step 5 will handle the remaining validation exports and handoff checks; the ledger is recorded in Step 8. Real training requires the Drive data and a CUDA runtime and has not yet been run by this implementation step.
