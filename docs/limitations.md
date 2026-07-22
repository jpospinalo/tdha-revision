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