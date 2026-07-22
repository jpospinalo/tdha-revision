# Performance

Computational optimizations in the pipeline. None of them changes the methodological workflow or the reported metrics.

## Connectivity is built once

Connectivity is generated during data preparation and reused across all folds and repetitions, so identical matrices are never recomputed. In `run_queue.py --in-process`, configurations that share data and windowing also reuse it through the in-memory cache. Building it is cheap anyway: from ~40 ms (12 ROIs) to ~2 s (116 ROIs).

## Windowing

Windows come straight from the physical specification, converted to samples with the site TR once, with no per-fold recomputation.

## Early stopping

Training stops on the inner validation loss with `restore_best_weights`, so the best epoch is recovered whenever training stops. The default `--patience 25` cuts the epochs that run after the loss has plateaued — the dominant cost within a run. Raise it only if convergence curves show late gains.

## Mixed precision

Every architecture declares a `float32` output, so the loss and sigmoid stay stable under `mixed_float16`. `--mixed-precision` enables it on GPU and speeds up the large configurations (39/116 ROIs, transformer, brainnetcnn). It only shifts the low-order digits of the metrics, so its use is recorded in `config.json`.

## Batch execution

`run_queue.py --in-process` runs a whole batch in one process: TensorFlow starts once instead of once per run. The default subprocess mode stays for long queues on unstable sessions, where isolating each run lets one failure stop without taking the rest down.

## Configuration and aggregation

Parameters live in a single configuration per run, which keeps executions consistent and errors rare. Metrics are written during the run and aggregated only after all repetitions finish.
