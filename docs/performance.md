# Performance

Computational optimizations in the pipeline. None of them changes the methodological workflow or the reported metrics.

## Connectivity is built once — except three fold-local representations

Connectivity is generated during data preparation and reused across all folds and repetitions, so identical matrices are never recomputed. In `run_queue.py --in-process`, configurations that share data and windowing also reuse it through the in-memory cache. Building it is cheap anyway: from ~40 ms (12 ROIs) to ~2 s (116 ROIs).

This does not hold for `ordered_scaled`, `permuted_scaled`, and `tangent`: their value depends on which subjects fall in `fit` for a given fold, so `run_config()` recomputes them once per fold via a `fold_transform` hook (see `methodology.md`). For the `_scaled` variants this is a cheap array rescale. For `tangent` it additionally fits a Ledoit-Wolf covariance and a geometric (Fréchet) reference per fold — more than the other representations, though still small next to a training run's cost at 12 ROIs; this has not been benchmarked at 116 ROIs, where the covariance estimation and the reference iteration both grow with ROI count.

## Windowing

Windows come straight from the physical specification, converted to samples with the site TR once, with no per-fold recomputation.

## Early stopping

Training stops on the inner validation series chosen by `--early-stopping-monitor` (`val_loss` by default, or `val_bce`) with `restore_best_weights`, so the best epoch is recovered whenever training stops. The default `--patience 25` cuts the epochs that run after the monitored series has plateaued — the dominant cost within a run, independent of which monitor is selected. Raise it only if convergence curves show late gains.

Recording the extra `bce`/`val_bce` metric adds a second scalar computation per batch alongside the existing `accuracy` metric — negligible next to the forward/backward pass itself, and it does not change what `model.fit` optimizes.

## Mixed precision

Every architecture declares a `float32` output, so the loss and sigmoid stay stable under `mixed_float16`. `--mixed-precision` enables it on GPU and speeds up the large configurations (39/116 ROIs, transformer, brainnetcnn). It only shifts the low-order digits of the metrics, so its use is recorded in `config.json`.

## Batch execution

`run_queue.py --in-process` runs a whole batch in one process: TensorFlow starts once instead of once per run. The default subprocess mode stays for long queues on unstable sessions, where isolating each run lets one failure stop without taking the rest down.

## Configuration and aggregation

Parameters live in a single configuration per run, which keeps executions consistent and errors rare. Metrics are written during the run and aggregated only after all repetitions finish.
