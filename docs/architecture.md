# Software Architecture

This document describes the overall architecture of the ADHD-200 experimental pipeline and the interaction between its main components.

The objective of the pipeline is to transform resting-state fMRI ROI time series into connectivity representations that can be used to train and evaluate machine learning and deep learning models under a reproducible experimental framework.

---

# Pipeline Overview

The pipeline is organized into four main stages:

```
ADHD-200 Dataset
        │
        ▼
ROI Time Series
        │
        ▼
Data Preparation
        │
        ▼
Connectivity Generation
        │
        ▼
Model Training and Evaluation
        │
        ▼
Experiment Aggregation
```

Each stage has a well-defined responsibility and communicates with the next through standardized data structures.

---

# Pipeline Components

## Data Preparation (`data.py`)

This module is responsible for preparing the functional MRI data before model training.

Its responsibilities include:

- loading ROI time series;
- generating static or dynamic functional connectivity representations;
- handling site-specific repetition times (TR);
- creating temporal windows from physical window specifications;
- applying optional preprocessing operations such as Fisher transformation and Gaussian window weighting;
- producing the final tensors used during model training.

---

## Model Training (`run_experiment.py`)

This module controls the execution of a complete experiment.

Its responsibilities include:

- loading experimental configurations;
- creating training and validation folds;
- training machine learning or deep learning models;
- selecting the best epoch using the validation set;
- evaluating the trained model;
- exporting metrics and experiment metadata.

---

## Experiment Execution (`run_queue.py`)

This module automates the execution of multiple experiments.

It generates combinations of experimental configurations and sequentially executes the corresponding training runs while preserving a consistent experimental structure.

---

## Result Aggregation (`compile_results.py`)

This module consolidates the results produced by multiple experiments.

It computes descriptive statistics across repetitions and exports aggregated summaries that facilitate comparison between experimental configurations. It also refuses to aggregate runs that are not comparable (different seeds, split fingerprints, or BOLD hashes) and provides repeated-measures ANOVA with paired post-hoc tests.

---

## Model Registry (`kerasmodels/`)

This package holds the neural architectures as a registry. Each module registers a `build(n_windows, n_features, **hyperparameters)` function that returns an **uncompiled** `keras.Model` with a single sigmoid output. Compilation (optimizer, learning rate, loss, metrics) is performed by `run_experiment.py`, so architectures stay decoupled from training hyperparameters.

The registered architectures are:

- `lstm`, `gru` — recurrent models over the window sequence (order-sensitive).
- `cnn1d` — 1D convolution along the time (window) axis.
- `transformer` — self-attention encoder, with an optional learned positional encoding (`positional=False` turns it into an order-invariant set model).
- `deepsets` — a shared per-window MLP followed by symmetric pooling; order-invariant by construction.
- `brainnetcnn` — topological convolution over the connectivity matrix (edge-to-edge, edge-to-node, node-to-graph filters); reconstructs the symmetric matrix internally from the vectorized upper triangle, intended for the static representation.

New architectures are added by dropping a module in the package and importing it in `__init__.py`; they become available as `--model <name>` without touching any other file.

---

## Environment Verification (`verify_setup.py`)

This module checks the repository and environment before any experiment: file structure, BOLD payloads and shapes, ROI-set consistency, sequence construction, cross-validation partitions (leakage checks), and that every registered architecture builds. It is meant to run right after cloning.

---

# Design Principles

The implementation follows several design principles.

## Modular organization

Each component performs a single well-defined task, reducing dependencies between modules.

## Reproducibility

Experimental configurations are stored together with the generated results, allowing experiments to be reproduced.

## Extensibility

New connectivity representations, preprocessing strategies, models or evaluation metrics can be incorporated without modifying the overall workflow.

## Dataset independence

Although the current implementation targets the ADHD-200 dataset, the processing pipeline is sufficiently modular to support datasets with equivalent ROI time series.

---

# Information Flow

The complete workflow can be summarized as:

```
ROI Time Series
      │
      ▼
Connectivity Generation
      │
      ▼
Feature Representation
      │
      ▼
Cross Validation
      │
      ▼
Model Training
      │
      ▼
Performance Evaluation
      │
      ▼
Result Aggregation
```

Each stage consumes only the outputs generated by the previous stage, ensuring a clear separation of responsibilities throughout the pipeline.