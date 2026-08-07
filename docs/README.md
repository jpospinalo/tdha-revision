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
| `auditoria-metricas.md` | Audit of the manuscript figures against the versioned results. |
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
| `modificaciones_eventanado.md` | Integration notes for the windowing changes. |
| `Concepto_LOSO_armonizacion_multisitio.md` | Earlier stage of the LOSO/harmonization discussion. Marked `SUPERSEDED / HISTORICAL RECORD` at the top; not a specification for any future LOSO work. |
| `finalization/` | Process records from the Methods/Results finalization phase: gate resolutions (`f1_gates.md`), the handoff package for Discussion/Limitations (`limitations_handoff.md`), per-round terminology-fix reports (`f3_terminologia/`, `f6_refs/`), and the QA report for the first execution batch (`informe_qa_tanda1.md`). Do not update these retroactively to match later conclusions — they document the checkpoint they were written at. |

## Exported vs. canonical figures

`analysis/roi_comparison/outputs/figures/` is the canonical source for manuscript figures. `docs/manuscrito_revisado/` may contain byte-identical copies of some of these — those are export snapshots bundled with the manuscript package, not a second source of truth. If they ever diverge, `analysis/roi_comparison/outputs/figures/` wins.

This folder complements the source code and the manuscript. It summarizes the relevant decisions without documenting every function.
