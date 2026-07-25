# Validation

What has been checked in the current implementation, for consistency, reproducibility, and compatibility between modules.

| Area | Status |
|------|:------:|
| Module interfaces | ✅ |
| Data processing | ✅ |
| Connectivity generation | ✅ |
| Windowing | ✅ |
| Model training | ✅ |
| Performance evaluation | ✅ |
| Result aggregation | ✅ |
| Reproducibility | ✅ |

## Module interfaces

The outputs of `data.py → run_experiment.py`, `run_experiment.py → compile_results.py`, and `run_queue.py → run_experiment.py` match the inputs each next stage expects.

## Data processing

ROI time series load correctly, subjects keep their metadata, site TRs are handled, and the resulting structures stay consistent through the pipeline.

## Connectivity

Static produces one matrix per subject over the whole series; dynamic produces window sequences in temporal order, from physically-defined windows. Both are checked for symmetry, diagonal, and value range. The Ledoit-Wolf shrinkage estimators (`partial`, `shrunk`) are checked in `verify_setup.py` for shape, finiteness, and range, same as the Pearson-based ones.

`ordered_scaled`, `permuted_scaled`, and `tangent` are checked in `verify_setup.py` for two additional properties, not just shape/finiteness: that the fit-only statistic (mean/sd for the scaled variants, the geometric reference for `tangent`) reproduces the expected value on `fit`, and — the leakage check — that perturbing subjects outside `fit` does not change the output for subjects inside `fit`. This was also verified against real NYU/NeuroIMAGE data (not just synthetic fixtures) when these representations were added, together with a bit-exact regression check confirming every historical representation's output tensor is unchanged.

## Windowing

Seconds convert to samples correctly, overlap is applied consistently, different TRs yield equivalent physical windows, and both rectangular and Gaussian windows work.

## Training

Partitions stay isolated; the inner partition is used only for epoch selection and the outer fold only for the final evaluation; class weights come from the training partition alone; and every registered architecture builds with a single sigmoid output.

`--early-stopping-monitor val_bce` is checked in `verify_setup.py --full` beyond the exit code: `config.json` records `config_schema_version=3` and the requested monitor; `history.csv` carries `bce`/`inner_val_bce` with no NaN/inf; `metrics_val.csv` carries `early_stopping_monitor`/`best_monitor_value`; `best_epoch` and `best_monitor_value` are reconstructed independently from `history.csv`, per `(repeat, fold)`, as the epoch that minimizes `inner_val_bce` — not `val_loss` — confirming `EarlyStopping` restored the weights it was actually configured to watch; and `predictions_val.csv` still covers every subject exactly once per repetition. The `val_loss` path (the historical default) is checked the same way it always was, by exit code and printed accuracy.

## Evaluation and aggregation

Metrics are computed and exported with their configuration, so every run is traceable. Aggregation groups only comparable runs and keeps each one tied to its configuration.

## Reproducibility

Fixed seeds, exported configuration and metadata, and a standardized output layout let a run be repeated under the same settings.

## Improvements in this version

Physical (time-based) windowing with site-specific TR; static and dynamic connectivity; order-invariant and order-permuted representations for temporal-order controls; two Ledoit-Wolf shrinkage connectivity estimators (`partial`, `shrunk`) alongside raw Pearson; tangent-space connectivity (`tangent`, via nilearn) with a fold-local reference; per-fold rescaling controls (`ordered_scaled`, `permuted_scaled`) to separate a scaling effect from a representation effect; an architecture registry with six models, including an order-invariant baseline (`deepsets`) and a topological matrix model (`brainnetcnn`); shorter default early-stopping patience and optional mixed precision; a configurable early-stopping monitor (`val_loss`/`val_bce`, `config_schema_version=3`) that isolates whether selecting the epoch on predictive BCE alone, instead of total regularized loss, changes outcomes — without altering the optimized objective; single-process batch execution; centralized configuration; standardized aggregation; per-repetition out-of-fold metrics; a Nadeau-Bengio corrected paired significance test for repeated k-fold comparisons, explicitly paired by `(repeat, fold)` rather than by row position; and `--strict-comparability` to fail instead of only warn when compiled runs are not comparable.
