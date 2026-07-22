# Limitations

This document summarizes the current limitations of the experimental pipeline.

These limitations define the scope of the current implementation and should be considered when interpreting experimental results or extending the software.

---

# Experimental Scope

The pipeline has been designed and validated for resting-state fMRI ROI time series from the ADHD-200 dataset.

Although its modular design facilitates adaptation to similar datasets, compatibility with other data sources has not been explicitly validated.

---

# Functional Connectivity

The current implementation estimates functional connectivity using Pearson correlation.

Alternative connectivity measures are not included in the current version of the pipeline.

---

# Model Selection

Epoch selection (early stopping) is performed on an inner partition held out from each outer training fold, and the outer fold is used only for final evaluation. This nesting isolates epoch selection from the test data, but the pipeline does not implement a full nested cross-validation for hyperparameter or architecture search: hyperparameters are fixed by the user, not tuned on an inner loop.

---

# Hyperparameter Optimization

Experimental parameters are defined by the user through the configuration file.

The pipeline does not include automatic hyperparameter optimization or architecture search.

---

# Supported Models

The current implementation registers six architectures: `lstm`, `gru`, `cnn1d`, `transformer`, `deepsets` and `brainnetcnn`. All consume the vectorized connectivity representation and return a single sigmoid output; `brainnetcnn` reconstructs the connectivity matrix internally.

Additional architectures can be registered in `kerasmodels/` without modifying the training modules, provided they follow the same build contract.

---

# Computational Requirements

Dynamic functional connectivity produces a larger number of connectivity matrices than static connectivity.

Consequently, execution time and memory requirements increase with the number and size of temporal windows.

---

# Summary

The current implementation provides a reproducible and modular framework for evaluating static and dynamic functional connectivity representations using the experimental methodology adopted in this project.

The limitations described in this document define the current scope of the software and should be considered when extending the pipeline or interpreting its results.