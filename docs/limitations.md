# Limitations

The scope of the current implementation, to keep in mind when interpreting results or extending the code.

## Dataset

Designed and validated for ADHD-200 rs-fMRI ROI time series. The modular design should adapt to similar datasets, but that has not been tested.

## Connectivity

Pearson correlation (`static`, `ordered`, `permuted`, `mean`, `mean_std`,
`ordered_scaled`, `permuted_scaled`), two Ledoit-Wolf shrinkage estimators (`partial`:
regularized partial correlation, isolates direct connections; `shrunk`: regularized full
correlation — same question as `static`, more stable with short series or many ROIs),
and tangent-space parametrization (`tangent`, via nilearn), which Dadi et al. (2019,
*Benchmarking functional connectome-based predictive models for resting-state fMRI*)
find to be the strongest predictor in their benchmark of connectivity measures. No
mutual information or other nonlinear measures.

`tangent` is only implemented in its **static** form (one projection per subject,
against a fit-only group reference) — not the dynamic, per-window version (a tangent
coordinate per window, forming a sequence). The static form was chosen first because it
needed far less new machinery (no change to the fold loop's data-dependency structure
beyond a single fit-only transform) and because it isolates the geometric question from
the temporal-order question; a dynamic tangent representation, if pursued later, adds a
second axis of complexity (per-window projection) on top of one (dynamic vs. static)
that has not yet shown a clear benefit for this dataset — see `validation.md` for the
`ordered` vs. `static` results this decision was based on. The reference geometric mean
is nilearn's tested implementation, not a hand-rolled one, to keep the SPD-matrix
algebra (matrix log/exp, Fréchet mean) off this project's maintenance surface.

ATHENA (the upstream ADHD-200 preprocessing pipeline) does not scrub high-motion
volumes — it only regresses motion parameters, WM/CSF signal, and a polynomial drift.
Residual micromotion can still bias correlation estimates (Power et al. 2012;
Satterthwaite et al. 2012). Mean framewise displacement per subject is not currently
joined into this pipeline, so its effect on the connectivity matrices is not tested here.

## Model selection

Epoch selection (early stopping) runs on an inner partition of each outer training fold; the outer fold is used only for the final evaluation. This isolates epoch selection from the test data, but it is not a full nested cross-validation: hyperparameters are fixed by the user, not tuned on an inner loop. Which inner-validation series is watched (`--early-stopping-monitor`: `val_loss`, the default, or `val_bce`) is itself configurable and part of the run's identity, but this only changes which epoch gets selected, never the optimized objective (`binary_crossentropy` + L2) or which partition is read from. Runs from `config_schema_version` 1-3 recorded `best_epoch`/`best_monitor_value` as the global minimum of the monitored series, which is not always the epoch `EarlyStopping` actually restored once `min_delta` or `start_from_epoch` are nonzero; `config_schema_version >= 4` reads these directly off the `EarlyStopping` instance and enforces them against a second, independent evaluation (`restored_monitor_value`): if the two don't match, `run_experiment.py` raises before the outer fold is ever used, rather than merely flagging the mismatch afterward — see `methodology.md`. This only affects that metadata's precision, not the external metrics of earlier runs, which were already computed on whichever weights Keras had restored; `compile_results.py` still compiles schema 1-3 runs descriptively, printing a once-per-run notice that they fall outside the formal `--stats-by early_stopping_monitor` A/B rather than silently including or silently dropping them.

## Supported models

Six architectures are registered: `lstm`, `gru`, `cnn1d`, `transformer`, `deepsets`, `brainnetcnn`. All take the vectorized connectivity representation and return a single sigmoid output; `brainnetcnn` reconstructs the matrix internally. New ones follow the same build contract in `kerasmodels/`.

`brainnetcnn` requires its input to reconstruct into a valid square matrix
(`n_features = r·(r-1)/2` for some integer `r`). It works with any single-matrix
representation (`static`, `partial`, `shrunk`, `mean`) and with `ordered`/`permuted`
and their `_scaled` variants (each window becomes a channel, not a modeled time step —
see `methodology.md`). It does **not** work with `mean_std` or `hybrid`: both
concatenate multiple statistics per connection, so the feature count no longer
corresponds to a triangle and construction fails. It also does not work with `tangent`,
rejected explicitly even though its feature count does reconstruct into a square matrix:
tangent coefficients are geometric deviations from a reference, not edge weights, so
`brainnetcnn`'s topology-respecting filters would be operating on a quantity they were
not designed for.

## Computational cost

Dynamic connectivity produces many more matrices than static, so time and memory grow with the number and size of the windows.
