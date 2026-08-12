#!/usr/bin/env python3
"""Restyle Figures 2, 3, 4 of the manuscript, reusing only already-computed,
frozen canonical outputs -- no new modeling, no new bootstrap, no new AUC
values reported anywhere. This script is intentionally separate from (and
never imports/executes) the regression-protected canonical scripts
(run_statistical_analysis.py, generate_manuscript_figures_v6.py); it only
reads their already-written, frozen output artifacts.

Figure 2 (mean out-of-fold AUC by ROI count and site):
  - reused data: analysis/roi_comparison/outputs/data/metrics_by_repeat.csv
    (per-repetition AUC, grey dots) and
    analysis/roi_comparison/outputs/tables/descriptive_performance.csv
    (mean + bootstrap CI, black dot). Both already exist and are unchanged
    by this script.
  - new style: individual per-repetition points + black mean/CI, faceted by
    site (matches analysis/roi_comparison/outputs/figures/paired_roi_profiles.svg
    style, with English labels appropriate for the manuscript).

Figure 3 (sensitivity forest plot):
  - reused data: analysis/roi_comparison/outputs/tables/figure4_v6_audit.csv
    (point/ci_low/ci_high per site/condition, already frozen).
  - same layout/grouping/skip-logic as build_figure_sensitivity() in
    generate_manuscript_figures_v6.py, reproduced here read-only from its
    already-exported CSV (this script does not call or modify that script).
  - new style: black/grey monochrome markers instead of navy, matching
    Figure 2's harmonized palette. No individual per-repetition points
    added (scope decision -- would require deriving per-repetition deltas
    for all 13 conditions x 2 sites from raw predictions, a materially
    larger and separately-verifiable task not requested for this pass).

Figure 4 (ROC curves):
  - recomputed directly from the 16 already-frozen predictions_val.csv
    files under results/runs/<roi>/<site>_..._control_baseline*/ for the
    primary within-site BrainNetCNN runs, using the same method already
    described in the existing (blue) Figure 4 caption: mean predicted
    probability per participant across the five repetitions, one ROC curve
    per site/ROI. AUC recomputed only as a QA cross-check against
    descriptive_performance.csv (small differences vs. the repetition-
    averaged AUC are expected and already present in the existing figure,
    since it is a different aggregation method from Table 4's -- not a new
    discrepancy introduced here). New style: grey-to-black sequential
    palette instead of blue, harmonized typography.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path("/sessions/laughing-wizardly-archimedes/mnt/tdha-revision")
ROI_OUT = REPO_ROOT / "analysis" / "roi_comparison" / "outputs"
OUT_DIR = REPO_ROOT / "docs" / "manuscrito_revisado" / "figure_style_r2" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_ORDER = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]
ROI_ORDER = [12, 18, 39, 116]

# ---------------------------------------------------------------------------
# Shared harmonized style (black/grey monochrome, matches paired_roi_profiles
# style and the new Figure 2)
# ---------------------------------------------------------------------------
_PREFERRED_SANS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
_available = {f.name for f in fm.fontManager.ttflist}
FONT_FAMILY = next((f for f in _PREFERRED_SANS if f in _available), "DejaVu Sans")

COLOR_POINT = "#000000"
COLOR_INDIV = "#8C8C8C"
COLOR_TEXT = "#2B2B2B"
COLOR_AXES = "#595959"
COLOR_REF_LINE = "#C7C7C7"
COLOR_GROUP_HEADER = "#000000"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 8.5,
    "axes.edgecolor": COLOR_AXES,
    "axes.labelcolor": COLOR_TEXT,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_AXES,
    "ytick.color": COLOR_AXES,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def _errorbar(ax, y, est, lo, hi, color=COLOR_POINT, markersize=4.6, elinewidth=1.2, capsize=2.7, capthick=1.2):
    # Horizontal forest-plot point: category position on y, delta value (with
    # its CI) on x. Matches generate_manuscript_figures_v6.py's _errorbar
    # exactly (xerr, not yerr) -- copied here read-only, not imported.
    xerr_lo = est - lo
    xerr_hi = hi - est
    _, caplines, barlinecols = ax.errorbar(
        est, y, xerr=[[xerr_lo], [xerr_hi]],
        fmt="o", color=color, ecolor=color, elinewidth=elinewidth, capsize=capsize,
        capthick=capthick, markersize=markersize, zorder=3,
        markeredgecolor="white", markeredgewidth=0.6,
    )
    for cap in caplines:
        cap.set_solid_capstyle("round")
    for coll in barlinecols:
        coll.set_capstyle("round")


# ===========================================================================
# FIGURE 2 -- restyled, reusing metrics_by_repeat.csv + descriptive_performance.csv
# ===========================================================================
def build_figure2():
    metrics_by_repeat = pd.read_csv(ROI_OUT / "data" / "metrics_by_repeat.csv")
    desc = pd.read_csv(ROI_OUT / "tables" / "descriptive_performance.csv")

    all_lows = desc["auc_bilateral_ci_low"].tolist()
    all_highs = desc["auc_bilateral_ci_high"].tolist()
    all_reps = metrics_by_repeat["auc"].tolist()
    margin = 0.03
    y_min = max(0.0, min(all_lows + all_reps) - margin)
    y_max = min(1.0, max(all_highs + all_reps) + margin)

    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.2), sharey=True)
    jitter = np.linspace(-0.15, 0.15, 5)
    x = np.arange(len(ROI_ORDER))

    for ax, site in zip(axes, SITE_ORDER):
        means, los, his = [], [], []
        for roi in ROI_ORDER:
            row = desc[(desc["site"] == site) & (desc["roi_set"] == roi)].iloc[0]
            means.append(row["mean_auc"])
            los.append(row["auc_bilateral_ci_low"])
            his.append(row["auc_bilateral_ci_high"])
        means_arr, los_arr, his_arr = np.array(means), np.array(los), np.array(his)
        yerr_low = np.clip(means_arr - los_arr, 0, None)
        yerr_high = np.clip(his_arr - means_arr, 0, None)
        ax.errorbar(
            x, means, yerr=[yerr_low, yerr_high],
            fmt="o", color=COLOR_POINT, ecolor=COLOR_POINT, markersize=4.6,
            elinewidth=1.2, capsize=2.7, capthick=1.2, zorder=3,
            markeredgecolor="white", markeredgewidth=0.6,
        )
        for i, roi in enumerate(ROI_ORDER):
            sub = metrics_by_repeat[(metrics_by_repeat["site"] == site) & (metrics_by_repeat["roi_set"] == roi)]
            reps = sub.sort_values("repeat")["auc"].to_numpy()
            assert len(reps) == 5, (site, roi, len(reps))
            ax.scatter(np.full(5, x[i]) + jitter, reps, color=COLOR_INDIV, s=8, zorder=2, linewidths=0)
        ax.axhline(0.5, color=COLOR_REF_LINE, linestyle=(0, (1, 1.6)), linewidth=0.9, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in ROI_ORDER], fontsize=8.3)
        ax.set_title(r"$\mathbf{" + site.replace(" ", r"\ ") + "}$", fontsize=9.5, pad=4, color=COLOR_TEXT)
        ax.set_xlabel("ROI panel", fontsize=8.5)
        ax.set_ylim(y_min, y_max)
        _style_axes(ax)
        ax.tick_params(axis="y", labelsize=8.3)

    axes[0].set_ylabel("Out-of-fold AUC", fontsize=8.9)
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.subplots_adjust(wspace=0.12)

    png_path = OUT_DIR / "figure2_auc_by_roi_restyled.png"
    svg_path = OUT_DIR / "figure2_auc_by_roi_restyled.svg"
    fig.savefig(png_path, dpi=320)
    fig.savefig(svg_path)
    plt.close(fig)

    # QA: black dot + CI must equal descriptive_performance.csv exactly (same
    # numbers already reported in Table 4); this is a pure re-plot.
    for site in SITE_ORDER:
        for roi in ROI_ORDER:
            row = desc[(desc["site"] == site) & (desc["roi_set"] == roi)].iloc[0]
            reps = metrics_by_repeat[(metrics_by_repeat["site"] == site) & (metrics_by_repeat["roi_set"] == roi)]["auc"]
            assert abs(reps.mean() - row["mean_auc"]) < 1e-9, (site, roi, reps.mean(), row["mean_auc"])
    print(f"Figure 2 written: {png_path} ; QA (mean of 5 reps == Table 4 mean_auc) PASSED for all 16 cells")
    return png_path, svg_path


# ===========================================================================
# FIGURE 3 -- harmonized palette only, reusing figure4_v6_audit.csv verbatim
# ===========================================================================
FIGURE_LABELS = {
    "12 ROIs": "12 ROIs",
    "18 ROIs": "18 ROIs",
    "39 ROIs": "39 ROIs",
    "116 ROIs": "116 ROIs",
    "Static connectivity": "Static FC",
    "Static connectivity (DeepSets)": "Static FC (DeepSets)",
    "Static connectivity (LSTM)": "Static FC (LSTM)",
    "LSTM-128": "LSTM (128 units)",
    "GRU-151": "GRU (151 units)",
    "Window 140 s / step 12 s": "140 / 12",
    "Window 120 s / step 24 s": "120 / 24",
    "Window 60 s / step 12 s (BrainNetCNN)": "60 / 12",
    "Window 60 s / step 12 s (GRU)": "60 / 12 (GRU)",
    "ROI panel": "ROI count",
    "Signal representation": "Connectivity",
    "Model architecture": "Architecture",
    "Windowing": "Window / step",
}


def disp(label):
    return FIGURE_LABELS.get(label, label)


PANEL_A_GROUPS = [
    ("ROI panel", ["18 ROIs", "39 ROIs", "116 ROIs"]),
    ("Signal representation", [
        "Static connectivity",
        "Static connectivity (DeepSets)",
        "Static connectivity (LSTM)",
    ]),
    ("Model architecture", ["LSTM-128", "GRU-151"]),
    ("Windowing", [
        "Window 140 s / step 12 s",
        "Window 120 s / step 24 s",
        "Window 60 s / step 12 s (BrainNetCNN)",
        "Window 60 s / step 12 s (GRU)",
    ]),
]
RESTRICTED_TO = {
    "LSTM-128", "GRU-151",
    "Window 140 s / step 12 s", "Window 120 s / step 24 s",
    "Window 60 s / step 12 s (BrainNetCNN)", "Window 60 s / step 12 s (GRU)",
    "Static connectivity (DeepSets)", "Static connectivity (LSTM)",
}
RESTRICTED_SITES = {"NYU", "Peking"}


def _y_positions_panel_a():
    ROW_H = 1.0
    HEADER_H = 0.62
    GROUP_GAP = 0.20
    cursor = 0.0
    positions, header_y = {}, {}
    first = True
    for group, conds in PANEL_A_GROUPS:
        if not first:
            cursor -= GROUP_GAP
        first = False
        cursor -= HEADER_H
        header_y[group] = cursor
        for cond in conds:
            cursor -= ROW_H
            positions[cond] = cursor
    offset = -min(list(positions.values()) + list(header_y.values()))
    positions = {k: v + offset for k, v in positions.items()}
    header_y = {k: v + offset for k, v in header_y.items()}
    span = max(list(positions.values()) + list(header_y.values())) + 0.75
    return positions, header_y, span


def build_figure3():
    audit = pd.read_csv(ROI_OUT / "tables" / "figure4_v6_audit.csv")
    evaluated = audit[audit["status"] == "evaluated"]
    records_a = {
        (r["site"], r["condition"]): (r["point"], r["ci_low"], r["ci_high"])
        for _, r in evaluated.iterrows()
    }

    all_vals = []
    for (site, label), (est, lo, hi) in records_a.items():
        all_vals.extend([lo, hi])
    max_abs = max(abs(min(all_vals)), abs(max(all_vals)))
    pad = 0.005
    limit = np.ceil((max_abs + pad) / 0.05) * 0.05
    x_lo, x_hi = -limit, limit
    xticks = [t for t in [-0.20, 0.0, 0.20] if x_lo < t < x_hi] or [round(-limit / 2, 2), 0.0, round(limit / 2, 2)]

    pos_a, header_y_a, span_a = _y_positions_panel_a()
    row_labels_a = [lab for _, labs in PANEL_A_GROUPS for lab in labs]

    V5_SPAN = 10.83
    V5_HEIGHT = 3.6
    fig_height = V5_HEIGHT * (span_a / V5_SPAN)

    fig = plt.figure(figsize=(6.5, fig_height))
    gs = fig.add_gridspec(
        nrows=1, ncols=5, width_ratios=[0.82, 1, 1, 1, 1],
        wspace=0.14,
        left=0.03, right=0.99, top=1 - (0.14 * 3.6 / fig_height), bottom=0.16 * 3.6 / fig_height,
    )
    axes = [fig.add_subplot(gs[0, c]) for c in (1, 2, 3, 4)]

    n_drawn = 0
    for ax, site in zip(axes, SITE_ORDER):
        ax.axvline(0.0, color=COLOR_REF_LINE, linestyle=(0, (1, 1.6)), linewidth=0.9, zorder=1)
        for label in row_labels_a:
            y = pos_a[label]
            key = (site, label)
            not_evaluated = (label in RESTRICTED_TO) and (site not in RESTRICTED_SITES)
            if not_evaluated or key not in records_a:
                continue
            est, lo, hi = records_a[key]
            _errorbar(ax, y, est, lo, hi)
            n_drawn += 1
        ax.set_title(r"$\mathbf{" + site.replace(" ", r"\ ") + "}$", fontsize=9.5, pad=4, color=COLOR_TEXT)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xticks(xticks)
        # Axis title states "percentage points"; tick labels must therefore
        # show the x100 percentage-point scale, not raw decimal AUC units
        # (R3 fix, plan V3.2.1 R2 section 6 -- the underlying tick positions,
        # x_lo/x_hi, and all plotted points/CIs/lines are unchanged, only the
        # displayed label strings).
        ax.set_xticklabels([
            "0" if abs(t) < 1e-12 else f"{'−' if t < 0 else '+'}{abs(t) * 100:.0f}"
            for t in xticks
        ])
        ax.set_ylim(-0.6, span_a)
        _style_axes(ax)
        ax.set_yticks([pos_a[lab] for lab in row_labels_a])
        ax.tick_params(axis="x", labelsize=8.3)
        ax.tick_params(axis="y", pad=3.2)
        if ax is axes[0]:
            ax.set_yticklabels([disp(lab) for lab in row_labels_a], fontsize=8.3)
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)

    for group_label, y in header_y_a.items():
        axes[0].text(
            -0.06, y, disp(group_label), fontsize=8.0, style="italic", fontweight="semibold",
            color=COLOR_GROUP_HEADER, ha="right", va="center",
            transform=axes[0].get_yaxis_transform(), clip_on=False,
        )

    center_x = (axes[0].get_position().x0 + axes[3].get_position().x1) / 2
    bottom_of_axes = min(ax.get_position().y0 for ax in axes)
    fig.text(center_x, max(0.01, bottom_of_axes - 0.055 * 3.6 / fig_height),
              "Δ out-of-fold AUC (percentage points; comparator by row, see caption)",
              fontsize=8.9, ha="center", va="top")

    png_path = OUT_DIR / "figure3_sensitivity_restyled.png"
    svg_path = OUT_DIR / "figure3_sensitivity_restyled.svg"
    fig.savefig(png_path, dpi=320, pad_inches=0.03)
    fig.savefig(svg_path, pad_inches=0.03)
    plt.close(fig)

    assert n_drawn == len(evaluated) == 32, f"expected 32 drawn points, drew {n_drawn}, evaluated rows={len(evaluated)}"
    print(f"Figure 3 written: {png_path} ; drew {n_drawn}/32 evaluated points, all values reused verbatim from figure4_v6_audit.csv")
    return png_path, svg_path


# ===========================================================================
# FIGURE 4 -- ROC curves recomputed from raw frozen predictions, new palette
# ===========================================================================
RUN_DIRS = {
    (12, "NYU"): "NYU_rois12_w60s6_brainnetcnn_control_baseline_v13_3e220e5c",
    (12, "Peking"): "Peking_rois12_w60s6_brainnetcnn_control_baseline_v13_bc841110",
    (12, "NeuroIMAGE"): "NeuroIMAGE_rois12_w61s6_brainnetcnn_control_baseline_v13_2b729a8c",
    (12, "OHSU"): "OHSU_rois12_w48s5_brainnetcnn_control_baseline_v13_1a7c37ce",
    (18, "NYU"): "NYU_rois18_w60s6_brainnetcnn_control_baseline_v13_662d71a9",
    (18, "Peking"): "Peking_rois18_w60s6_brainnetcnn_control_baseline_v13_0bf7fa0e",
    (18, "NeuroIMAGE"): "NeuroIMAGE_rois18_w61s6_brainnetcnn_control_baseline_v13_93342cf0",
    (18, "OHSU"): "OHSU_rois18_w48s5_brainnetcnn_2ce6c48e",
    (39, "NYU"): "NYU_rois39_w60s6_brainnetcnn_control_base_line_1521c348",
    (39, "Peking"): "Peking_rois39_w60s6_brainnetcnn_control_baseline_v13_396e34d2",
    (39, "NeuroIMAGE"): "NeuroIMAGE_rois39_w61s6_brainnetcnn_control_baseline_v13_dc028168",
    (39, "OHSU"): "OHSU_rois39_w48s5_brainnetcnn_control_baseline_v13_299719fe",
    (116, "NYU"): "NYU_rois116_w60s6_brainnetcnn_control_baseline_v13_160b89cd",
    (116, "Peking"): "Peking_rois116_w60s6_brainnetcnn_240732d1",
    (116, "NeuroIMAGE"): "NeuroIMAGE_rois116_w61s6_brainnetcnn_control_baseline_v13_669d72bd",
    (116, "OHSU"): "OHSU_rois116_w48s5_brainnetcnn_control_baseline_v13_f82f17b4",
}

# Sequential grey -> black palette (harmonized with Figures 2/3), replacing
# the previous sequential blue palette; darkest = largest ROI panel, as in
# the original (light 12 ROIs -> dark 116 ROIs).
ROC_PALETTE = {12: "#BFBFBF", 18: "#8C8C8C", 39: "#4D4D4D", 116: "#000000"}


def _roc_points(y_true, y_score):
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    return fpr, tpr


def _auc_trapz(fpr, tpr):
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def build_figure4():
    desc = pd.read_csv(ROI_OUT / "tables" / "descriptive_performance.csv")
    runs_root = REPO_ROOT / "results" / "runs"

    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.9), sharey=True)
    qa_rows = []

    for ax, site in zip(axes, SITE_ORDER):
        for roi in ROI_ORDER:
            run_dir = RUN_DIRS[(roi, site)]
            path = runs_root / str(roi) / run_dir / "predictions_val.csv"
            assert path.exists(), f"missing predictions file: {path}"
            df = pd.read_csv(path)
            # Mean predicted probability per participant across the five
            # repetitions (same method already used for the existing,
            # blue-palette Figure 4, per its caption).
            per_subj = df.groupby("subject_id").agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
            y_true = per_subj["y_true"].to_numpy()
            y_prob = per_subj["y_prob"].to_numpy()
            fpr, tpr = _roc_points(y_true, y_prob)
            recomputed_auc = _auc_trapz(fpr, tpr)
            table4_auc = desc[(desc["site"] == site) & (desc["roi_set"] == roi)]["mean_auc"].iloc[0]
            qa_rows.append({
                "site": site, "roi_set": roi, "n_participants": len(per_subj),
                "recomputed_auc_mean_prob_method": recomputed_auc,
                "table4_auc_mean_of_5_reps_method": table4_auc,
                "abs_diff": abs(recomputed_auc - table4_auc),
            })
            ax.step(fpr, tpr, where="post", color=ROC_PALETTE[roi], linewidth=1.3, label=f"{roi} ROIs")
        ax.plot([0, 1], [0, 1], color=COLOR_REF_LINE, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_title(r"$\mathbf{" + site.replace(" ", r"\ ") + "}$", fontsize=9.5, pad=4, color=COLOR_TEXT)
        ax.set_xlabel("1 − Specificity", fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _style_axes(ax)
        ax.tick_params(labelsize=7.8)

    axes[0].set_ylabel("Sensitivity", fontsize=8.9)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0),
                         frameon=False, fontsize=8.3, handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    png_path = OUT_DIR / "figure4_roc_restyled.png"
    svg_path = OUT_DIR / "figure4_roc_restyled.svg"
    fig.savefig(png_path, dpi=320, bbox_extra_artists=(legend,), bbox_inches="tight")
    fig.savefig(svg_path, bbox_extra_artists=(legend,), bbox_inches="tight")
    plt.close(fig)

    qa = pd.DataFrame(qa_rows)
    qa_path = OUT_DIR.parent / "figure4_roc_qa_recompute.csv"
    qa.to_csv(qa_path, index=False)
    print(qa.to_string())
    print(f"Figure 4 written: {png_path}")
    print(f"QA comparison saved: {qa_path}")
    print(f"Max abs diff between recomputed (mean-prob-then-ROC) AUC and Table 4 (mean-of-5-reps) AUC: {qa['abs_diff'].max():.4f}")
    return png_path, svg_path, qa


if __name__ == "__main__":
    build_figure2()
    build_figure3()
    build_figure4()
