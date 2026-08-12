#!/usr/bin/env python3
"""Derive Table 6 (main text) and the Supplementary LOSO tables from the
frozen loso_static_v1 canonical outputs. Reads only canonical, congelado
artifacts under analysis/loso/outputs and results/loso/_design; hard-codes
no scientific figures. Part of phase LOSO_METHODS_RESULTS_INTEGRATION_V3_2_1,
CP4.

Outputs (written under docs/manuscrito_revisado/loso_integration_v3_2_1/):
  - loso_table6_source.csv     (16 rows, unrounded canonical values + display strings)
  - s_loso_design.csv          (source data for Supplementary Table S_LOSO_Design)
  - s_loso_fullmetrics.csv     (source data for Supplementary Table S_LOSO_FullMetrics)
  - s_loso_contrasts.csv       (source data for Supplementary Table S_LOSO_Contrasts, 12 rows)
  - s_loso_seeds.csv           (source data for Supplementary Table S_LOSO_Seeds)
  - s_loso_convergence.csv     (source data for Supplementary Table S_LOSO_Convergence)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_OUT = REPO_ROOT / "analysis" / "loso" / "outputs"
DESIGN_DIR = REPO_ROOT / "results" / "loso" / "_design"
PHASE_DIR = REPO_ROOT / "docs" / "manuscrito_revisado" / "loso_integration_v3_2_1"

SUMMARY_CSV = ANALYSIS_OUT / "loso_metrics_summary.csv"
CONTRASTS_CSV = ANALYSIS_OUT / "loso_contrasts.csv"
CONVERGENCE_CSV = ANALYSIS_OUT / "loso_convergence_summary.csv"
DESIGN_JSON = DESIGN_DIR / "loso_static_v1_design.json"

SITE_ORDER = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}"


def fmt_cell(point: float, lo: float, hi: float) -> str:
    return f"{fmt_pct(point)}% [{fmt_pct(lo)}, {fmt_pct(hi)}]"


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    contrasts = pd.read_csv(CONTRASTS_CSV)
    convergence = pd.read_csv(CONVERGENCE_CSV)
    design = json.loads(DESIGN_JSON.read_text())

    if len(summary) != 16:
        raise SystemExit(f"STOP: loso_metrics_summary.csv has {len(summary)} rows, expected 16.")
    if len(contrasts) != 12:
        raise SystemExit(f"STOP: loso_contrasts.csv has {len(contrasts)} rows, expected 12.")

    rotation_sizes = design["rotation_sizes"]
    site_class_counts = design["site_class_counts"]

    # ---------------------------------------------------------------
    # Table 6 source (main text): 4 rows x 4 model/ROI columns
    # ---------------------------------------------------------------
    rows = []
    for site in SITE_ORDER:
        held_out_n = rotation_sizes[site]["test"]
        fit_n = rotation_sizes[site]["fit"]
        inner_n = rotation_sizes[site]["inner_val"]
        row = {
            "held_out_site": site,
            "held_out_n": held_out_n,
            "fit_n": fit_n,
            "inner_val_n": inner_n,
        }
        for roi in ["12", "116"]:
            for model, col_prefix in [("brainnetcnn", f"bnn{roi}"), ("logreg", f"logreg{roi}")]:
                sub = summary[
                    (summary["held_out_site"] == site)
                    & (summary["roi_set"].astype(str) == roi)
                    & (summary["model"] == model)
                ]
                if len(sub) != 1:
                    raise SystemExit(
                        f"STOP: expected exactly 1 row for {site}/{roi}/{model}, found {len(sub)}."
                    )
                r = sub.iloc[0]
                row[f"{col_prefix}_auc_point"] = r["auc_point"]
                row[f"{col_prefix}_auc_ci_low"] = r["auc_ci_low"]
                row[f"{col_prefix}_auc_ci_high"] = r["auc_ci_high"]
                row[f"{col_prefix}_display"] = fmt_cell(
                    r["auc_point"], r["auc_ci_low"], r["auc_ci_high"]
                )
        rows.append(row)

    table6 = pd.DataFrame(rows)
    table6.to_csv(PHASE_DIR / "loso_table6_source.csv", index=False)

    # ---------------------------------------------------------------
    # S_LOSO_Design
    # ---------------------------------------------------------------
    design_rows = []
    for site in SITE_ORDER:
        source_sites = [s for s in SITE_ORDER if s != site]
        design_rows.append(
            {
                "held_out_site": site,
                "held_out_n": rotation_sizes[site]["test"],
                "held_out_control_n": site_class_counts[site]["control"],
                "held_out_adhd_n": site_class_counts[site]["adhd"],
                "source_sites": ", ".join(source_sites),
                "fit_n": rotation_sizes[site]["fit"],
                "inner_val_n": rotation_sizes[site]["inner_val"],
                "participant_characteristics_cross_reference": "NOT AVAILABLE IN FROZEN SCOPE",
            }
        )
    pd.DataFrame(design_rows).to_csv(PHASE_DIR / "s_loso_design.csv", index=False)

    # ---------------------------------------------------------------
    # S_LOSO_FullMetrics (16 rows, all secondary metrics, no new CIs)
    # ---------------------------------------------------------------
    full_cols = [
        "held_out_site",
        "roi_set",
        "model",
        "auc_point",
        "auc_ci_low",
        "auc_ci_high",
        "balanced_accuracy_point",
        "f1_macro_point",
        "sensitivity_point",
        "specificity_point",
    ]
    summary[full_cols].to_csv(PHASE_DIR / "s_loso_fullmetrics.csv", index=False)

    # ---------------------------------------------------------------
    # S_LOSO_Contrasts (12 rows, full contrast table)
    # ---------------------------------------------------------------
    contrasts.to_csv(PHASE_DIR / "s_loso_contrasts.csv", index=False)

    # ---------------------------------------------------------------
    # S_LOSO_Seeds (BrainNetCNN seed dispersion only; logreg has none)
    # ---------------------------------------------------------------
    seed_cols = ["held_out_site", "roi_set", "model", "seed_sd", "seed_min", "seed_max"]
    bnn_only = summary[summary["model"] == "brainnetcnn"][seed_cols]
    bnn_only.to_csv(PHASE_DIR / "s_loso_seeds.csv", index=False)

    # ---------------------------------------------------------------
    # S_LOSO_Convergence
    # ---------------------------------------------------------------
    convergence.to_csv(PHASE_DIR / "s_loso_convergence.csv", index=False)

    print("Wrote loso_table6_source.csv, s_loso_design.csv, s_loso_fullmetrics.csv, "
          "s_loso_contrasts.csv, s_loso_seeds.csv, s_loso_convergence.csv")
    print(f"Table 6 rows: {len(table6)} (expected 4)")
    for site in SITE_ORDER:
        print(f"  {site}: held_out_n={rotation_sizes[site]['test']}")


if __name__ == "__main__":
    main()
