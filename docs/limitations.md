# Limitations

The scope of the current implementation, to keep in mind when interpreting results or extending the code.

## Dataset

Designed and validated for ADHD-200 rs-fMRI ROI time series. The modular design should adapt to similar datasets, but that has not been tested.

## Connectivity

Pearson correlation (`static`, `ordered`, `permuted`, `mean`, `mean_std`) and two
Ledoit-Wolf shrinkage estimators: `partial` (regularized partial correlation, isolates
direct connections) and `shrunk` (regularized full correlation — same question as
`static`, more stable with short series or many ROIs). No tangent-space parametrization
yet, which Dadi et al. (2019, *Benchmarking functional connectome-based predictive
models for resting-state fMRI*) find to be the strongest predictor in their benchmark of
connectivity measures; no mutual information or other nonlinear measures.

ATHENA (the upstream ADHD-200 preprocessing pipeline) does not scrub high-motion
volumes — it only regresses motion parameters, WM/CSF signal, and a polynomial drift.
Residual micromotion can still bias correlation estimates (Power et al. 2012;
Satterthwaite et al. 2012). Mean framewise displacement per subject is not currently
joined into this pipeline, so its effect on the connectivity matrices is not tested here.

## Model selection

Epoch selection (early stopping) runs on an inner partition of each outer training fold; the outer fold is used only for the final evaluation. This isolates epoch selection from the test data, but it is not a full nested cross-validation: hyperparameters are fixed by the user, not tuned on an inner loop.

## Supported models

Six architectures are registered: `lstm`, `gru`, `cnn1d`, `transformer`, `deepsets`, `brainnetcnn`. All take the vectorized connectivity representation and return a single sigmoid output; `brainnetcnn` reconstructs the matrix internally. New ones follow the same build contract in `kerasmodels/`.

`brainnetcnn` requires its input to reconstruct into a valid square matrix
(`n_features = r·(r-1)/2` for some integer `r`). It works with any single-matrix
representation (`static`, `partial`, `shrunk`, `mean`) and with `ordered`/`permuted`
(each window becomes a channel, not a modeled time step — see `methodology.md`). It
does **not** work with `mean_std` or `hybrid`: both concatenate multiple statistics per
connection, so the feature count no longer corresponds to a triangle and construction
fails.

## Computational cost

Dynamic connectivity produces many more matrices than static, so time and memory grow with the number and size of the windows.
