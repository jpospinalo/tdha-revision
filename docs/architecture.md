# Software Architecture

The pipeline turns rs-fMRI ROI time series into connectivity representations and trains classifiers on them, keeping every run reproducible.

## Stages

```
ROI time series → data preparation → connectivity generation → cross-validated training → aggregation
```

Each stage consumes only the previous stage's output.

## Modules

**`data.py`** — loads ROI time series, builds static or dynamic connectivity, resolves the site-specific TR, cuts temporal windows from physical (seconds) specifications, and optionally applies a Fisher Z transform or Gaussian window weighting.

**`run_experiment.py`** — runs one experiment: builds the cross-validation folds, trains the model, selects the epoch on an inner split, evaluates on the outer fold, and writes the per-fold metrics, `config.json`, and a derived `resumen.md`. Most representations are built once, globally, before the fold loop. Three (`ordered_scaled`, `permuted_scaled`, `tangent`) instead go through a `fold_transform` hook that `run_config()` calls once per fold, fit only on that fold's `fit_idx` and applied without refitting to `inner_val`/`outer_val` — see `methodology.md`'s "Fold-local representations". When no `fold_transform` is given (every historical representation), the fold loop indexes the base tensor exactly as before.

**`run_queue.py`** — expands a grid of configurations and runs them, either as one subprocess per configuration or, with `--in-process`, all in a single process.

**`compile_results.py`** — collects the runs under `results/runs/`, tabulates them, and warns when runs are not comparable (different seed, split fingerprint, BOLD hash, dirty git tree, or duplicated configs); `--strict-comparability` turns those warnings into a hard failure instead of just printing them. It also computes per-repetition out-of-fold metrics (pooling each repetition's fold predictions before scoring AUC/F1/balanced-accuracy/log-loss/Brier — less noisy than averaging per-fold metrics). Paired comparisons (`--stats`) print a repeated-measures ANOVA labeled explicitly as exploratory — it treats folds as independent and does not correct for the overlap between a repeated k-fold's training sets, so it plays no role in the significance verdict. That verdict instead comes from a Nadeau-Bengio corrected resampled t-test (which does account for that overlap) plus Holm correction across contrasts; the naive paired t-test and Wilcoxon are kept in the output only as reference. Runs are paired by an explicit `(repeat, fold)` merge, not by position after sorting — it fails loudly on missing columns, duplicated keys within a run, or mismatched fold sets between runs, instead of silently pairing the wrong rows.

**`kerasmodels/`** — the architecture registry. Each module registers a `build(n_windows, n_features, **hyperparameters)` that returns an **uncompiled** `keras.Model` with a single sigmoid output; `run_experiment.py` compiles it, so architectures carry no training hyperparameters. Registered:

- `lstm`, `gru` — recurrent, order-sensitive.
- `cnn1d` — 1D convolution along the window axis.
- `transformer` — self-attention; `positional=False` makes it order-invariant.
- `deepsets` — per-window MLP plus symmetric pooling, order-invariant by construction.
- `brainnetcnn` — edge-to-edge / edge-to-node filters over the connectivity matrix, which it reconstructs internally from the vectorized upper triangle. Works with any single-matrix representation (`static`, `partial`, `shrunk`, `mean`) and with `ordered`/`permuted`/their `_scaled` variants (windows as channels); incompatible with `mean_std`, `hybrid`, and `tangent` (see `limitations.md`).

A new module imported in `__init__.py` becomes available as `--model <name>`.

**`verify_setup.py`** — pre-flight checks after cloning: file structure, BOLD shapes, ROI-set consistency, sequence construction, cross-validation leakage, and that every architecture builds.

## Design

Each module does one thing and hands standardized structures to the next, so a connectivity measure, representation, model, or metric can be swapped without touching the rest. Every run stores its full configuration next to its results — that is what makes runs reproducible and portable to other datasets with equivalent ROI time series.
