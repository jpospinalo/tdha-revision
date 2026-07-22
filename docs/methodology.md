# Methodology

This document summarizes the methodological decisions implemented in the current version of the experimental pipeline.

The objective is to provide a concise description of how data are processed, how experiments are executed, and how results are generated.

---

# Dataset

The pipeline was developed for the ADHD-200 dataset using preprocessed resting-state functional MRI (rs-fMRI) ROI time series.

Each subject is represented by a multivariate temporal signal, where rows correspond to time points and columns correspond to regions of interest (ROIs).

---

# Functional Connectivity Representation

The pipeline supports two connectivity representations.

## Static Connectivity

A single functional connectivity matrix is computed using the complete ROI time series of each subject.

This representation captures the overall functional relationships between brain regions during the entire acquisition period.

## Dynamic Connectivity

The ROI time series are divided into overlapping temporal windows. A functional connectivity matrix is computed for each window, producing a sequence of connectivity matrices that preserves temporal information.

The resulting sequence constitutes the input for sequential learning models.

---

# Temporal Windowing

Dynamic connectivity is generated using physical window specifications rather than a fixed number of samples.

Window length and overlap are defined in seconds and automatically converted to samples using the repetition time (TR) associated with each acquisition site.

This approach guarantees consistent temporal windows across datasets acquired with different sampling rates.

The implementation supports both rectangular and Gaussian temporal windows.

---

# Connectivity Estimation

Functional connectivity is estimated using Pearson correlation between ROI time series within each temporal window.

An optional Fisher Z transformation can be applied after correlation estimation to improve statistical properties.

---

# Experimental Configuration

All experimental parameters are defined through a configuration file.

The configuration includes:

- connectivity representation;
- temporal window parameters;
- preprocessing options;
- model architecture;
- training parameters;
- evaluation settings.

The configuration used in each experiment is stored together with the generated results to facilitate reproducibility.

---

# Model Training

Experiments are executed using repeated stratified cross-validation.

For each training split:

- the model is trained using the training partition;
- model selection is performed using the corresponding validation partition;
- the selected model is evaluated on the test partition.

The process is repeated across all folds and repetitions defined in the experimental configuration.

---

# Performance Evaluation

Model performance is evaluated using classification metrics computed independently for each experimental repetition.

Individual experiment results are preserved and subsequently aggregated to obtain descriptive statistics across repetitions.

---

# Reproducibility

The pipeline incorporates several mechanisms to improve experiment reproducibility.

These include:

- deterministic random seeds;
- configuration export;
- experiment metadata storage;
- standardized result aggregation.

These mechanisms allow experiments to be reproduced using the same configuration and execution settings.