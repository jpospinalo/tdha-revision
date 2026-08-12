# Documentation

Technical documentation for the ADHD-200 dynamic functional connectivity pipeline. This folder mixes two kinds of content — documents describing the pipeline and paper as they currently stand, and historical records of decisions, plans, and provenance from earlier phases. Both are kept: past decisions are **not** left out, because several of them (ROI-panel provenance, `class_weight` chronology, reviewer-response commitments) are still load-bearing for the manuscript.

## Current documentation

Describes the implementation and the paper's reference configuration as they stand today.

| Document | Description |
|----------|-------------|
| `architecture.md` | Modules and data flow. |
| `methodology.md` | How data are processed and experiments run. Includes a note distinguishing general/legacy runner defaults from the manuscript's actual campaign configuration. |
| `validation.md` | What has been verified in the implementation. |
| `performance.md` | Computational optimizations. |
| `limitations.md` | Scope and known limitations of the implementation (not the manuscript's Limitations section, which does not exist yet — see `finalization/limitations_handoff.md` below). |
| `paper_reference_configuration.md` | **Canonical source** for the experimental configuration actually used in the manuscript — cohort, ROI panels, windowing, validation, class weighting, environments, official run IDs. Takes priority over any other document if they disagree. |
| `paper_environment.md` | The three software environments that produced results used in the manuscript, and which result came from which. |
| `data_provenance/` | Provenance, access mechanism, and hash of external data sources not versioned in this repository (e.g. the ADHD-200 phenotypic file). |
| `guia-experimentacion-colaborativa.md` | Operational guide for the Colab notebook (Spanish): running, validating, downloading, and pushing a run. Carries a historical-guide notice: some of its example configurations predate the paper's campaign and do not match it. |
| `Guia_implementacion_baseline_ML.md` | Guide for the logistic-regression baseline used in Table 5 (`src/run_baseline_ml.py`). |

## Historical records / provenance / decisions

Reflect an earlier state of the project, a decision that was made and should not be re-litigated silently, or a process record. Kept because later documents reference them, not because they describe the current pipeline.

| Document | Description |
|----------|-------------|
| `PLAN_RESPUESTA_REVISORES.md` | **Approved and frozen plan** for answering the reviewers: evidence available without retraining, the ten-run campaign, mandatory transparency statements, and the checks required before resubmission. |
| `auditoria-metricas.md` | One-time audit (July 2026) of an earlier manuscript against an external predecessor repository (`TDHA-fMRI`) and its `results/rois/`/`results/architectures/` layout — neither exists in this repository anymore. Kept because it documents why several later fixes (single architecture per ROI comparison, non-overwritable run folders, the current `results/runs/<roi_set>/` layout) were made; not a description of the current pipeline. |
| `Concepto_LOSO_armonizacion_multisitio.md` | Earlier stage of the LOSO/harmonization discussion. Marked `SUPERSEDED / HISTORICAL RECORD` at the top; not a specification for any future LOSO work. |
| `finalization/` | Process records from the Methods/Results finalization phase: baseline freezes (`f0_freeze/`), gate resolutions (`f1_gates.md`), the handoff package for Discussion/Limitations (`limitations_handoff.md`), per-round terminology-fix reports (`f3_terminologia/`, `f6_refs/`), and the QA reports for each execution batch (`informe_qa_tanda1.md`). Do not update these retroactively to match later conclusions — they document the checkpoint they were written at. |

## Exported vs. canonical figures

`analysis/roi_comparison/outputs/figures/` is the canonical source for manuscript figures. `docs/manuscrito_revisado/figure_style_r2/figures/` holds byte-identical export copies of the restyled Figures 2–4 actually embedded in the current manuscript (R8) — those are a packaging convenience, not a second source of truth. If they ever diverge, `analysis/roi_comparison/outputs/figures/` wins. (A repository cleanup in August 2026 removed an older, unrelated duplicate pair — `figure3_v5_auc_by_roi_count.*` / `figure4_v6_sensitivity.*` — that had been sitting loose in `docs/manuscrito_revisado/`; their canonical copies remain untouched in `analysis/roi_comparison/outputs/figures/`.)

This folder complements the source code and the manuscript. It summarizes the relevant decisions without documenting every function.
