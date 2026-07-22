# Software Architecture

The pipeline turns rs-fMRI ROI time series into connectivity representations and trains classifiers on them, keeping every run reproducible.

## Stages

```
ROI time series → data preparation → connectivity generation → cross-validated training → aggregation
```

Each stage consumes only the previous stage's output.

## Modules

**`data.py`** — loads ROI time series, builds static or dynamic connectivity, resolves the site-specific TR, cuts temporal windows from physical (seconds) specifications, and optionally applies a Fisher Z transform or Gaussian window weighting.

**`run_experiment.py`** — runs one experiment: builds the cross-validation folds, trains the model, selects the epoch on an inner split, evaluates on the outer fold, and writes the per-fold metrics, `config.json`, and a derived `resumen.md`.

**`run_queue.py`** — expands a grid of configurations and runs them, either as one subprocess per configuration or, with `--in-process`, all in a single process.

**`compile_results.py`** — collects the runs under `results/runs/`, tabulates them, refuses to aggregate runs that are not comparable (different seed, split fingerprint, or BOLD hash), and runs a repeated-measures ANOVA with paired post-hoc tests.

**`kerasmodels/`** — the architecture registry. Each module registers a `build(n_windows, n_features, **hyperparameters)` that returns an **uncompiled** `keras.Model` with a single sigmoid output; `run_experiment.py` compiles it, so architectures carry no training hyperparameters. Registered:

- `lstm`, `gru` — recurrent, order-sensitive.
- `cnn1d` — 1D convolution along the window axis.
- `transformer` — self-attention; `positional=False` makes it order-invariant.
- `deepsets` — per-window MLP plus symmetric pooling, order-invariant by construction.
- `brainnetcnn` — edge-to-edge / edge-to-node filters over the connectivity matrix, which it reconstructs internally from the vectorized upper triangle; meant for the static representation.

A new module imported in `__init__.py` becomes available as `--model <name>`.

**`verify_setup.py`** — pre-flight checks after cloning: file structure, BOLD shapes, ROI-set consistency, sequence construction, cross-validation leakage, and that every architecture builds.

## Design

Each module does one thing and hands standardized structures to the next, so a connectivity measure, representation, model, or metric can be swapped without touching the rest. Every run stores its full configuration next to its results — that is what makes runs reproducible and portable to other datasets with equivalent ROI time series.
