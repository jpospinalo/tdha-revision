# Validation

This document summarizes the validation and verification procedures performed on the current implementation of the pipeline.

The objective is to document the aspects of the software that have been reviewed to ensure methodological consistency, reproducibility and compatibility between modules.

---

# Validation Summary

| Area | Status |
|------|:------:|
| Software architecture | ✅ |
| Module compatibility | ✅ |
| Data processing | ✅ |
| Functional connectivity generation | ✅ |
| Windowing | ✅ |
| Model training | ✅ |
| Performance evaluation | ✅ |
| Result aggregation | ✅ |
| Reproducibility | ✅ |

---

# Software Architecture

The interaction between the main modules of the pipeline has been verified to ensure that data structures and execution flow remain consistent throughout the experimental process.

The following interfaces have been validated:

- `data.py` → `run_experiment.py`
- `run_experiment.py` → `compile_results.py`
- `run_queue.py` → `run_experiment.py`

The exported outputs of each module are compatible with the expected inputs of the subsequent stage.

---

# Data Processing

The data preparation stage has been verified to ensure that:

- ROI time series are correctly loaded.
- Subjects preserve their associated metadata.
- Site-specific repetition times (TR) are correctly handled.
- Generated data structures are consistent across the pipeline.

---

# Functional Connectivity

Connectivity generation has been verified for both supported representations.

## Static Connectivity

The pipeline correctly generates one connectivity matrix per subject using the complete ROI time series.

## Dynamic Connectivity

The pipeline correctly generates sequences of connectivity matrices using temporal windows defined in physical time.

The resulting connectivity sequences preserve temporal order and are compatible with sequential learning models.

---

# Temporal Windowing

The temporal windowing implementation has been verified to ensure that:

- window duration is correctly converted from seconds to samples;
- overlap is consistently applied;
- different repetition times produce equivalent physical windows;
- rectangular and Gaussian window functions are correctly supported.

---

# Model Training

The training workflow has been reviewed to verify that:

- data partitions remain isolated throughout training;
- validation data are used exclusively for model selection;
- class weights are computed using the training partition only;
- experimental configurations are consistently applied across repetitions.

---

# Performance Evaluation

The evaluation stage has been verified to ensure consistent computation and reporting of classification metrics.

The generated metrics are exported together with the corresponding experimental configuration, allowing complete traceability of every experiment.

---

# Result Aggregation

The aggregation process has been verified to ensure that:

- compatible experiments are grouped correctly;
- descriptive statistics are computed consistently;
- experimental configurations remain associated with the aggregated results.

---

# Reproducibility

The current implementation incorporates several mechanisms that improve experiment reproducibility.

These include:

- deterministic random seeds;
- configuration export;
- experiment metadata preservation;
- standardized output structure.

Together, these mechanisms allow experiments to be repeated using the same execution settings.

---

# Validation Status

The current implementation has been reviewed with respect to the software architecture, methodological consistency and experimental workflow.

The validations documented in this file reflect the state of the current implementation and provide a reference for future maintenance and extension of the pipeline.

# Major Improvements Implemented

The current version of the pipeline incorporates the following methodological improvements:

- Physical window specification based on time.
- Site-specific repetition time support.
- Static and dynamic connectivity representations.
- Optional Gaussian window weighting.
- Optional Fisher transformation.
- Centralized experimental configuration.
- Improved experiment reproducibility.
- Standardized result aggregation.