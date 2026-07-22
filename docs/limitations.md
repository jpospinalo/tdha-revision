# Limitations

The scope of the current implementation, to keep in mind when interpreting results or extending the code.

## Dataset

Designed and validated for ADHD-200 rs-fMRI ROI time series. The modular design should adapt to similar datasets, but that has not been tested.

## Connectivity

Only Pearson correlation. No alternative measures (partial correlation, mutual information, etc.).

## Model selection

Epoch selection (early stopping) runs on an inner partition of each outer training fold; the outer fold is used only for the final evaluation. This isolates epoch selection from the test data, but it is not a full nested cross-validation: hyperparameters are fixed by the user, not tuned on an inner loop.

## Supported models

Six architectures are registered: `lstm`, `gru`, `cnn1d`, `transformer`, `deepsets`, `brainnetcnn`. All take the vectorized connectivity representation and return a single sigmoid output; `brainnetcnn` reconstructs the matrix internally. New ones follow the same build contract in `kerasmodels/`.

## Computational cost

Dynamic connectivity produces many more matrices than static, so time and memory grow with the number and size of the windows.
