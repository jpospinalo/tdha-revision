# Methodology

How data are processed, how experiments run, and how results are produced.

## Dataset

ADHD-200 preprocessed rs-fMRI ROI time series. Each subject is a multivariate signal with time points on one axis and regions of interest (ROIs) on the other.

## Connectivity

**Static** — one connectivity matrix per subject, computed over the whole series.

**Dynamic** — the series is cut into overlapping windows, one connectivity matrix per window, giving a sequence that keeps temporal information.

Connectivity is estimated as the Pearson correlation between ROIs within each window, with an optional Fisher Z transform afterwards.

## Model input representations

From the dynamic sequence, `--representation` selects what the model receives:

- `ordered` — the windows in real temporal order (the sequence itself).
- `permuted` — the same windows shuffled within each subject. A control: if `ordered` and `permuted` score the same, temporal order carries no signal.
- `mean` / `mean_std` — order-invariant summaries (mean, or mean concatenated with per-connection standard deviation).
- `static` — a single matrix over the whole series, no windowing.

`permuted`, `mean`, and `mean_std` exist to test whether the ordering of resting-state windows contributes signal, which decides whether order-sensitive architectures (recurrent, positional transformer) are worth using over order-invariant ones (`deepsets`, transformer without positional encoding, `static`).

## Temporal windowing

Windows are specified in physical time (seconds) and converted to samples using each site's TR, so the same window covers the same duration regardless of sampling rate.

Window length respects the dynamic-connectivity lower bound: it must exceed the longest wavelength retained in the signal (Leonardi & Van De Ville, 2015). ATHENA band-pass filters at 0.009 Hz, which puts that floor near 111 s, so the default physical window is 120 s where the scan allows it. Sites too short for a valid window (OHSU, 185 s) default to the static representation.

Both rectangular and Gaussian windows are supported.

## Training

Repeated stratified cross-validation. Architectures (`lstm`, `gru`, `cnn1d`, `transformer`, `deepsets`, `brainnetcnn`) are decoupled from training, which is configured centrally.

Each outer fold is split again to keep epoch selection honest:

- an inner partition of the outer training set is held out to pick the epoch by early stopping on its loss;
- the model trains on the rest, with class weights computed only from it;
- the outer fold is touched **once**, for the final evaluation.

`restore_best_weights` returns the best inner-validation epoch. This nesting is for epoch selection only — it is not a nested cross-validation for hyperparameter search.

## Evaluation and reproducibility

Classification metrics are computed per repetition, stored per run, and aggregated afterwards. Reproducibility rests on fixed seeds (identical partitions across machines), the configuration and metadata exported with every run, and a standardized output layout.
