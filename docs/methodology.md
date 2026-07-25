# Methodology

How data are processed, how experiments run, and how results are produced.

## Dataset

ADHD-200 preprocessed rs-fMRI ROI time series. Each subject is a multivariate signal with time points on one axis and regions of interest (ROIs) on the other.

## Connectivity

**Static** — one connectivity matrix per subject, computed over the whole series.

**Dynamic** — the series is cut into overlapping windows, one connectivity matrix per window, giving a sequence that keeps temporal information.

Connectivity is estimated as the Pearson correlation between ROIs within each window, with an optional Fisher Z transform afterwards. Two full-series representations (`partial`, `shrunk`) replace the raw sample correlation with a Ledoit-Wolf shrinkage estimate instead: Dadi et al. (2019, *Benchmarking functional connectome-based predictive models for resting-state fMRI*, >240 pipelines evaluated) find that raw sample correlation is the weakest connectivity estimator for prediction, and that shrinkage regularization improves on it — more so as the number of ROIs grows relative to the number of timepoints, since sample correlation's estimation variance grows with that ratio.

A third representation, `tangent`, implements their strongest performer: tangent-space parametrization (nilearn's `ConnectivityMeasure(kind="tangent")`). Unlike `partial`/`shrunk`, tangent projection is not subject-local — every subject's coordinates depend on a group reference (a Fréchet mean over covariances), so it cannot be precomputed once per subject before cross-validation without leaking the outer fold into that reference. See "Fold-local representations" below for how this is kept fold-local.

## Model input representations

From the dynamic sequence, `--representation` selects what the model receives:

- `ordered` — the windows in real temporal order (the sequence itself).
- `permuted` — the same windows shuffled within each subject. A control: if `ordered` and `permuted` score the same, temporal order carries no signal.
- `mean` / `mean_std` — order-invariant summaries (mean, or mean concatenated with per-connection standard deviation).
- `static` — a single Pearson matrix over the whole series, no windowing.
- `partial` — a single regularized partial-correlation matrix (Ledoit-Wolf shrinkage) over the whole series; isolates direct connections and stays well-conditioned when timepoints < ROIs.
- `shrunk` — a single regularized *full*-correlation matrix (Ledoit-Wolf shrinkage) over the whole series; same question as `static`, more stable estimate.
- `hybrid` — static connectivity concatenated per connection with the mean, standard deviation, and mean absolute change of the windows; order-invariant.
- `ordered_scaled` / `permuted_scaled` — the same windows as `ordered`/`permuted`, each connection rescaled to zero mean / unit variance, fit-only. A rough reference point for `tangent`, not a clean isolation of its geometry: besides the rescaling step, the two also differ in temporality (`ordered_scaled` is windowed/dynamic, `tangent` is whole-series/static) and in covariance estimator (sample correlation per window vs. the Ledoit-Wolf estimate `tangent` uses). A result that only shows up in `tangent` and not in `ordered_scaled` is consistent with the tangent-space geometry mattering, but doesn't rule out the estimator or temporality differences as the cause.
- `tangent` — tangent-space projection built by nilearn's `ConnectivityMeasure(kind="tangent")` from the raw (whole-series) BOLD signal, fold-local. nilearn does **not** start from a Pearson correlation matrix: by default it standardizes (z-scores) each ROI's time series and estimates a Ledoit-Wolf covariance (`cov_estimator=None` resolves internally to `LedoitWolf`), then projects each subject's covariance against a geometric (Fréchet) mean fit only on that fold's `fit` subjects. `nilearn_version`, `tangent_cov_estimator`, and `tangent_standardize` are recorded per run under `windowing_diagnostics` in `config.json`. See "Fold-local representations" below.

`static`, `partial`, `shrunk`, `mean`, and `tangent` all produce one matrix (or
matrix-equivalent vector) per subject. `static`, `partial`, `shrunk`, and `mean` are
interchangeable as `brainnetcnn` input; `tangent` is not (see below). `ordered` and
`permuted` (and their `_scaled` variants) also work with `brainnetcnn`, but it treats
their windows as fixed input channels, not as a modeled sequence — the first layer learns
one weight per window, with no recurrence or attention across them, so it does not
exploit temporal order even when fed a dynamic representation. `mean_std` and `hybrid`
concatenate multiple statistics per connection and cannot be reshaped back into a square
matrix, so they only work with vector models (`lstm`, `gru`, `cnn1d`, `transformer`,
`deepsets`).

`permuted`, `mean`, `mean_std`, and `permuted_scaled` exist to test whether the ordering of resting-state windows contributes signal, which decides whether order-sensitive architectures (recurrent, positional transformer) are worth using over order-invariant ones (`deepsets`, transformer without positional encoding, `static`).

## Fold-local representations

`ordered_scaled`, `permuted_scaled`, and `tangent` cannot be built once per subject
before cross-validation the way every other representation is: their values (a
per-connection mean/sd, or a geometric reference) depend on *which subjects fall in
`fit`* for a given fold, and fitting that on anything outside `fit` would leak the
outer/inner validation subjects into training. `run_experiment.py` handles this with a
`fold_transform` hook: `build_representation()` still returns an untransformed base
(the raw `ordered`/`permuted` tensor, or the raw BOLD series for `tangent`) once per
run, and `run_config()` calls the fold-specific transform — fit only on `fit_idx`,
applied without refitting to `inner_val_idx`/`outer_val_idx` — inside the fold loop,
before every use of the data (training, epoch selection, and the outer evaluation).
Every historical representation uses no transform (`fold_transform=None`), so its
tensor is exactly the one indexed directly, with no behavior change.

`tangent` additionally requires nilearn and does not accept `--fisher-z` (its
coefficients are not Pearson correlations) or `brainnetcnn` (they are not
topologically interpretable edge weights, unlike a correlation matrix) — both
combinations are rejected explicitly.

## Temporal windowing

Windows are specified in physical time (seconds) and converted to samples using each site's TR, so the same window covers the same duration regardless of sampling rate.

Window length respects the dynamic-connectivity lower bound: it must exceed the longest wavelength retained in the signal (Leonardi & Van De Ville, 2015). ATHENA band-pass filters at 0.009 Hz, which puts that floor near 111 s, so the **recommended** physical window is 120 s where the scan allows it — pass it explicitly with `--window-seconds 120` (the notebook's example configs already do). Sites too short for a valid window (OHSU, 185 s) default to the static representation.

This is a recommendation, not the CLI's actual default: `run_experiment.py`, when no window/step arguments are given at all, falls back to `--window 70 --step 2` (`windowing_preset: "legacy_70_2"` in `config.json`) for backward compatibility with early runs, not to 120 s. Always pass `--window-seconds`/`--step-seconds` explicitly rather than relying on the bare default.

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

`compile_results.py` reports two views of validation performance. Per-fold mean±sd
(`val_*_mean`/`val_*_sd`) averages the metric computed on each outer fold separately —
noisy when folds are small (e.g. ~18 subjects per fold at `n_splits=10`). Per-repetition
out-of-fold metrics (`oof_*_mean`/`oof_*_sd`) instead pool every fold's predictions
within a repetition — in a `RepeatedStratifiedKFold`, each repetition's folds partition
the whole sample exactly once — and score AUC, F1-macro, balanced accuracy, log-loss,
and Brier on that pooled repetition before averaging across repetitions. This is less
sensitive to individual small-fold sampling variance.

Paired comparisons (`--stats`) additionally correct the resampled t-test for the fact
that folds from a repeated k-fold are not independent observations — their training sets
overlap heavily, which a naive paired t-test ignores and which inflates its significance
(Nadeau & Bengio, 2003). The corrected variance estimate is used for the Holm-corrected
significance verdict; the naive paired p-value is kept in the output only as a
reference.
