# Performance

This document summarizes the computational optimizations implemented in the current version of the pipeline.

The objective of these optimizations is to improve execution efficiency while preserving the methodological behavior and reproducibility of the experiments.

---

# Design Principles

Performance improvements were implemented under the following principles:

- Preserve the scientific behavior of the pipeline.
- Maintain reproducibility across executions.
- Reduce unnecessary computations.
- Improve scalability for large experimental batches.

No optimization modifies the methodological workflow or the reported evaluation metrics.

---

# Connectivity Computation

Connectivity representations are generated only once during data preparation and subsequently reused throughout the training process.

This design avoids repeated computation of identical connectivity matrices across cross-validation folds and experimental repetitions.

---

# Temporal Windowing

Temporal windows are generated directly from the physical window specification.

The implementation automatically converts window duration and overlap into samples using the corresponding repetition time (TR), avoiding redundant calculations during experiment execution.

---

# Experimental Configuration

Experimental parameters are centralized in a single configuration file.

This approach simplifies experiment management, reduces manual configuration errors and ensures consistent execution across multiple experiments.

---

# Modular Execution

The pipeline separates data preparation, model training and result aggregation into independent modules.

This modular design allows each stage to execute independently and prevents unnecessary repetition of processing steps.

---

# Early Stopping

Training uses early stopping on the inner validation loss with `restore_best_weights`, so the best epoch is recovered regardless of when training stops. The default patience (`--patience 25`) stops folds once the validation loss has plateaued instead of always running every epoch, which is the dominant compute cost within a run. Increase it only if convergence curves show late improvements.

# Mixed Precision

Every architecture declares a `float32` output so the loss and sigmoid stay numerically stable, which makes the pipeline safe to run under `mixed_float16`. The optional `--mixed-precision` flag enables it on GPU and accelerates the larger configurations (39 and 116 ROIs, transformer, CNN). It changes only the low-order digits of the metrics, so whether it was used is recorded in `config.json`.

# Batch Execution

`run_queue.py --in-process` runs an entire batch of configurations inside a single process. TensorFlow starts once instead of once per run, and configurations that share data and windowing reuse the already-built connectivity sequences through the in-memory cache. The default subprocess mode is kept for long queues on unstable sessions, where process isolation lets a single failure stop without affecting the rest.

# Result Aggregation

Performance metrics are computed during experiment execution and aggregated only after all repetitions have finished.

This separation minimizes intermediate processing while preserving complete experiment traceability.

---

# Reproducible Execution

Execution settings, experimental parameters and generated metrics are stored together with the experiment outputs.

This organization allows experiments to be reproduced without additional manual configuration.

---

# Summary

The current implementation incorporates several optimizations that improve computational efficiency while maintaining methodological consistency.

The primary optimization strategies include:

- reuse of generated connectivity representations;
- centralized experiment configuration;
- modular execution of the experimental workflow;
- standardized aggregation of experimental results.

These optimizations improve scalability and maintainability without affecting the scientific validity of the experimental pipeline.