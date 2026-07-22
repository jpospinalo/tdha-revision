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

# Model Input Representations

From the dynamic connectivity sequence, the pipeline can build several model inputs, selected with `--representation`:

- `ordered`: the connectivity windows in their real temporal order (the sequence itself).
- `permuted`: the same windows shuffled within each subject. Used as a control: if `ordered` and `permuted` perform equally, temporal order carries no discriminative signal.
- `mean` / `mean_std`: order-invariant summaries of the windows (mean, or mean concatenated with standard deviation per connection).
- `static`: a single connectivity matrix over the whole series, with no windowing.

The `permuted`, `mean`, and `mean_std` representations exist to test whether the temporal ordering of resting-state windows contributes signal, which determines whether order-sensitive architectures (recurrent, positional transformer) are appropriate over order-invariant ones (`deepsets`, transformer without positional encoding, `static`).

---

# Temporal Windowing

Dynamic connectivity is generated using physical window specifications rather than a fixed number of samples.

Window length and overlap are defined in seconds and automatically converted to samples using the repetition time (TR) associated with each acquisition site. This guarantees consistent temporal windows across sites acquired with different sampling rates.

Window length respects the standard dynamic-connectivity lower bound: the window must exceed the longest wavelength retained in the signal (Leonardi & Van De Ville, 2015). Because the ADHD-200 ATHENA preprocessing band-pass filters at 0.009 Hz, that floor is approximately 111 s, so the default physical window is 120 s where the scan length allows it. Sites whose acquisitions are too short for a valid window (OHSU, 185 s) default to the static representation.

The implementation supports both rectangular and Gaussian temporal windows, and an optional Fisher Z transformation of the correlations.

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

Experiments are executed using repeated stratified cross-validation. Several architectures are available (`lstm`, `gru`, `cnn1d`, `transformer`, `deepsets`, `brainnetcnn`); the architecture is decoupled from training, which is configured centrally.

Each outer fold is split further to avoid leakage in epoch selection:

- an inner partition (a fraction of the outer training set) is held out to select the training epoch via early stopping on its loss;
- the model is trained on the remaining training data with class weights computed only from that partition;
- the outer validation fold is used **only once**, for the final evaluation, never to decide anything.

Training uses early stopping with `restore_best_weights`, so the reported model is the best inner-validation epoch regardless of when training stops. This inner/outer split is a nested structure for epoch selection; it is not a nested cross-validation for hyperparameter search.

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