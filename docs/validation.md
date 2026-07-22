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

Static produces one matrix per subject over the whole series; dynamic produces window sequences in temporal order, from physically-defined windows. Both are checked for symmetry, diagonal, and value range.

## Windowing

Seconds convert to samples correctly, overlap is applied consistently, different TRs yield equivalent physical windows, and both rectangular and Gaussian windows work.

## Training

Partitions stay isolated; the inner partition is used only for epoch selection and the outer fold only for the final evaluation; class weights come from the training partition alone; and every registered architecture builds with a single sigmoid output.

## Evaluation and aggregation

Metrics are computed and exported with their configuration, so every run is traceable. Aggregation groups only comparable runs and keeps each one tied to its configuration.

## Reproducibility

Fixed seeds, exported configuration and metadata, and a standardized output layout let a run be repeated under the same settings.

## Improvements in this version

Physical (time-based) windowing with site-specific TR; static and dynamic connectivity; order-invariant and order-permuted representations for temporal-order controls; an architecture registry with six models, including an order-invariant baseline (`deepsets`) and a topological matrix model (`brainnetcnn`); shorter default early-stopping patience and optional mixed precision; single-process batch execution; centralized configuration; and standardized aggregation.
