#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v6: extends v5's sensitivity figure (build_figure4() in v5 -- the figure
embedded as "Figure 3" in the current manuscript numbering) with six new
rows, added as part of the natural sensitivity-analysis narrative (not as a
separate reviewer-response figure):

  - Signal representation: + "Static FC (DeepSets)", + "Static FC (LSTM)"
    (within-architecture static-vs-windowed contrasts; existing
    "Static connectivity" row is relabeled "Static FC (BrainNetCNN)" for
    disambiguation only -- its value is untouched).
  - Model architecture: + "GRU (151 units)" (units chosen so its parameter
    count, 99,359, matches the LSTM's, 99,969, to <1%).
  - Windowing: + "60 / 12 (BrainNetCNN)", + "60 / 12 (GRU)".

The logistic-regression / DeepSets-vs-BrainNetCNN-static algorithm
comparison is NOT added to this figure, consistent with the v9.7 decision
(still in force) to keep logistic-regression-type comparisons in Table 5
text only, not in the figure.

All PRE-EXISTING rows (ROI panel x3, Static connectivity, LSTM-128, the two
existing Windowing rows) are reproduced with byte-identical point/CI values
to v5 -- verified numerically in main() before export. Only the new rows'
values are freshly computed here.

New rows' data provenance: real out-of-fold predictions from the
n_repeats=5, n_splits=10, seed=42 "reviewer_sensitivity" experiment runs
under results/runs/12/, executed via run_reviewer_sensitivity.sh. AUC is
aggregated by repetition and bootstrap CIs use the same participant-level,
class-stratified, paired-by-subject resampling (10,000 resamples, NumPy
PCG64, seed 42) as every other number in Table 5 / Figure 3 -- see
Statistical Analysis in Methods. No retraining, no new resampling method.

Provenance update (2026-08-07): Peking's six reviewer_sensitivity runs
(static LSTM, static DeepSets, GRU 120/12, GRU 60/12, DeepSets 120/12,
BrainNetCNN 60/12) were originally executed without --class-weight,
violating Peking's prespecified class_weight=True policy (Gate G2). They
were superseded by six corrected runs tagged
reviewer_sensitivity_weighted_fix, sharing the identical
split_fingerprint=1e9626ad3839ff46 (same partitions, only class_weight
changed). This script now selects the *_weighted_fix runs for Peking; the
original unweighted Peking runs remain on disk as provenance only. NYU,
NeuroIMAGE, and OHSU reviewer_sensitivity runs are unaffected and unchanged.
"""
from pathlib import Path
import atexit
import glob
import os
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
OUT_DIR = REPO_ROOT / "analysis" / "roi_comparison" / "outputs"
TABLES_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR = REPO_ROOT / "results" / "runs" / "12"

# Baseline de regresión para el chequeo "las filas pre-existentes de v5 no
# cambiaron" (ver main()). No es figure4_v5_audit.csv: ese archivo nunca se
# comprometió a git (verificado con `git log --all` y `git rev-list --all
# --objects`, 2026-08-07) y desapareció del working tree en la limpieza del
# repositorio del 2026-08-06 (c2c78f0) sin que este chequeo se hubiera
# actualizado. Este CSV lo sustituye: son las 28 filas "pre-existentes"
# (ROI panel x3, Static connectivity, LSTM-128, Window 140/12, Window
# 120/24 -- para los 4 sitios, con las filas restringidas a NYU/Peking
# marcadas not_evaluated en NeuroIMAGE/OHSU) extraídas directamente del
# figure4_v6_audit.csv canónico vigente, que por construcción las reproduce
# con los mismos valores que v5 (ver docstring del módulo). No son valores
# reconstruidos a mano: se derivaron programáticamente filtrando las 5
# condiciones que v6 añadió (Static connectivity (DeepSets)/(LSTM),
# GRU-151, Window 60/12 BrainNetCNN/GRU).
REGRESSION_REFERENCE = (
    REPO_ROOT / "analysis" / "roi_comparison" / "config" / "figure4_v5_regression_reference.csv"
)

# ---------------------------------------------------------------------------
# Figure-only display labels (extends v5's FIGURE_LABELS)
# ---------------------------------------------------------------------------
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


def disp(label: str) -> str:
    return FIGURE_LABELS.get(label, label)


# ---------------------------------------------------------------------------
# Shared style -- copied verbatim from v5 (no aesthetic changes in this
# script: same palette, fonts, marker/line geometry).
# ---------------------------------------------------------------------------
_PREFERRED_SANS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
_available = {f.name for f in fm.fontManager.ttflist}
FONT_FAMILY = next((f for f in _PREFERRED_SANS if f in _available), "DejaVu Sans")

COLOR_POINT = "#1B4B66"
COLOR_TEXT = "#2B2B2B"
COLOR_TEXT_MUTED = "#6E6E6E"
COLOR_AXES = "#595959"
COLOR_REF_LINE = "#C7C7C7"
COLOR_GROUP_HEADER = "#1B4B66"

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

SITE_ORDER = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def _errorbar(ax, y, est, lo, hi, color=COLOR_POINT, markersize=4.6, elinewidth=1.2, capsize=2.7, capthick=1.2):
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
# New-data bootstrap (same methodology as Statistical Analysis in Methods)
# ===========================================================================
def _find1(pattern):
    m = glob.glob(pattern)
    assert len(m) == 1, (pattern, m)
    return Path(m[0]) / "predictions_val.csv"


def _load_preds(path):
    return pd.read_csv(path)


def _auc_by_repeat(df):
    aucs = []
    for rep, sub in df.groupby("repeat"):
        yt = sub["y_true"].to_numpy()
        yp = sub["y_prob"].to_numpy()
        pos, neg = yp[yt == 1], yp[yt == 0]
        ranks = rankdata(np.concatenate([pos, neg]))
        U = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
        aucs.append(U / (len(pos) * len(neg)))
    return float(np.mean(aucs))


def _paired_bootstrap(path_a, path_b, n_iter=10000, seed=42):
    df_a, df_b = _load_preds(path_a), _load_preds(path_b)
    reps = sorted(df_a["repeat"].unique())
    assert reps == sorted(df_b["repeat"].unique())
    point_a, point_b = _auc_by_repeat(df_a), _auc_by_repeat(df_b)
    per_rep = []
    for rep in reps:
        sa = df_a[df_a["repeat"] == rep].set_index("subject_id")
        sb = df_b[df_b["repeat"] == rep].set_index("subject_id")
        subs = sorted(sa.index)
        assert subs == sorted(sb.index), f"subject mismatch at repeat {rep}"
        yt = sa.loc[subs, "y_true"].to_numpy()
        ypa = sa.loc[subs, "y_prob"].to_numpy()
        ypb = sb.loc[subs, "y_prob"].to_numpy()
        idx_pos = np.where(yt == 1)[0]
        idx_neg = np.where(yt == 0)[0]
        per_rep.append((idx_pos, idx_neg, ypa, ypb))
    rng = np.random.Generator(np.random.PCG64(seed))
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        ra, rb = [], []
        for idx_pos, idx_neg, ypa, ypb in per_rep:
            sp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            sn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            n_pos = len(sp)

            def auc(yp):
                vals = np.concatenate([yp[sp], yp[sn]])
                r = rankdata(vals)
                U = r[:n_pos].sum() - n_pos * (n_pos + 1) / 2
                return U / (n_pos * len(sn))

            ra.append(auc(ypa))
            rb.append(auc(ypb))
        deltas[i] = np.mean(ra) - np.mean(rb)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point_a - point_b, lo, hi


def _new_condition_records():
    """(site, technical_label) -> (delta, ci_low, ci_high), for the six new
    rows only. Comparator differs by row (see docstring / Table 5 note):
      - GRU-151: vs the 12-ROI windowed BrainNetCNN reference (same
        convention as the existing LSTM-128 row).
      - Static connectivity (DeepSets/LSTM): static vs that architecture's
        own windowed (120s/12s) run (within-architecture, not vs the
        primary reference).
      - Window 60s/12s (BrainNetCNN): vs the 12-ROI windowed BrainNetCNN
        reference (same convention as the existing 140/12 and 120/24 rows).
      - Window 60s/12s (GRU): vs that architecture's own 120s/12s run
        (within-architecture).
    """
    R = RUNS_DIR
    ref = {
        "NYU": _find1(f"{R}/NYU_rois12_w60s6_brainnetcnn_control_baseline_v13_*"),
        "Peking": _find1(f"{R}/Peking_rois12_w60s6_brainnetcnn_control_baseline_v13_*"),
    }
    # NOTE (2026-08-07, Peking class-weight correction): the six Peking
    # reviewer_sensitivity runs originally lacked --class-weight, violating
    # the prespecified site policy (class_weight=True for Peking; see Gate
    # G2, docs/finalization/f1_gates.md). They were superseded by six
    # corrected runs tagged reviewer_sensitivity_weighted_fix (same
    # split_fingerprint=1e9626ad3839ff46, i.e. identical partitions -- only
    # class_weight changed). The historical (unweighted) Peking runs are
    # retained in results/runs/ for provenance but must never be selected
    # here: the patterns below pin Peking explicitly to the weighted_fix
    # tag so _find1's single-match assertion cannot silently resolve to the
    # superseded run. NYU is not affected and keeps the original pattern.
    gru_w120 = {
        "NYU": _find1(f"{R}/NYU_rois12_w60s6_gru_reviewer_sensitivity_*"),
        "Peking": _find1(f"{R}/Peking_rois12_w60s6_gru_reviewer_sensitivity_weighted_fix_*"),
    }
    gru_w60 = {
        "NYU": _find1(f"{R}/NYU_rois12_w30s6_gru_reviewer_sensitivity_*"),
        "Peking": _find1(f"{R}/Peking_rois12_w30s6_gru_reviewer_sensitivity_weighted_fix_*"),
    }
    bnn_w60 = {
        "NYU": _find1(f"{R}/NYU_rois12_w30s6_brainnetcnn_reviewer_sensitivity_*"),
        "Peking": _find1(f"{R}/Peking_rois12_w30s6_brainnetcnn_reviewer_sensitivity_weighted_fix_*"),
    }
    lstm_static = {
        "NYU": _find1(f"{R}/NYU_rois12_static_lstm_reviewer_sensitivity_*"),
        "Peking": _find1(f"{R}/Peking_rois12_static_lstm_reviewer_sensitivity_weighted_fix_*"),
    }
    lstm_w120 = {
        "NYU": _find1(f"{R}/NYU_rois12_w60s6_lstm_rev32_lstm128_ordered_*"),
        "Peking": _find1(f"{R}/Peking_rois12_w60s6_lstm_rev32_lstm128_ordered_*"),
    }
    deepsets_static = {
        "NYU": _find1(f"{R}/NYU_rois12_static_deepsets_reviewer_sensitivity_*"),
        "Peking": _find1(f"{R}/Peking_rois12_static_deepsets_reviewer_sensitivity_weighted_fix_*"),
    }
    deepsets_w120 = {
        "NYU": _find1(f"{R}/NYU_rois12_w60s6_deepsets_reviewer_sensitivity_*"),
        "Peking": _find1(f"{R}/Peking_rois12_w60s6_deepsets_reviewer_sensitivity_weighted_fix_*"),
    }

    records = {}
    for site in ["NYU", "Peking"]:
        d, lo, hi = _paired_bootstrap(gru_w120[site], ref[site])
        records[(site, "GRU-151")] = (d, lo, hi)

        d, lo, hi = _paired_bootstrap(lstm_static[site], lstm_w120[site])
        records[(site, "Static connectivity (LSTM)")] = (d, lo, hi)

        d, lo, hi = _paired_bootstrap(deepsets_static[site], deepsets_w120[site])
        records[(site, "Static connectivity (DeepSets)")] = (d, lo, hi)

        d, lo, hi = _paired_bootstrap(bnn_w60[site], ref[site])
        records[(site, "Window 60 s / step 12 s (BrainNetCNN)")] = (d, lo, hi)

        d, lo, hi = _paired_bootstrap(gru_w60[site], gru_w120[site])
        records[(site, "Window 60 s / step 12 s (GRU)")] = (d, lo, hi)

    return records


# ===========================================================================
# Sensitivity figure (v5's build_figure4 -- manuscript "Figure 3"), extended
# ===========================================================================
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


def _load_panel_a_deltas():
    """v5's existing-row loading logic, verbatim, plus the six new rows."""
    primary = pd.read_csv(TABLES_DIR / "primary_12_vs_116.csv")
    secondary = pd.read_csv(TABLES_DIR / "secondary_pairwise_comparisons.csv")
    secondary = secondary[secondary["metric"] == "auc"]
    boot = pd.read_csv(TABLES_DIR / "manuscript_bootstrap_10k.csv")

    records = {}
    for _, r in primary.iterrows():
        records[(r["site"], "116 ROIs")] = (
            -r["delta_auc"], -r["bilateral_ci_high"], -r["bilateral_ci_low"],
        )
    for tech_label, contrast in [("18 ROIs", "12-18"), ("39 ROIs", "12-39")]:
        sub = secondary[secondary["contrast"] == contrast]
        for _, r in sub.iterrows():
            records[(r["site"], tech_label)] = (
                -r["estimate"], -r["bilateral_ci_high"], -r["bilateral_ci_low"],
            )
    boot_map = {
        "static": "Static connectivity",
        "lstm128": "LSTM-128",
        "window140_12": "Window 140 s / step 12 s",
        "window120_24": "Window 120 s / step 24 s",
    }
    sens = boot[~boot["contrast_id"].str.startswith("baseline")]
    for _, r in sens.iterrows():
        site, cond = r["contrast_id"].split("__")
        tech_label = boot_map[cond]
        records[(site, tech_label)] = (r["delta"], r["delta_ci_low"], r["delta_ci_high"])

    records.update(_new_condition_records())
    return records


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


def build_figure_sensitivity(png_path, svg_path, pdf_path):
    records_a = _load_panel_a_deltas()
    audit_rows = []

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

    # Figure height scales with the number of rows/groups so that row
    # spacing, font size, and marker geometry stay visually identical to
    # v5 (v5's height 3.6in was tuned for span_a ~ 10.83; scale
    # proportionally for the taller v6 layout instead of hardcoding).
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

    for ax, site in zip(axes, SITE_ORDER):
        ax.axvline(0.0, color=COLOR_REF_LINE, linestyle=(0, (1, 1.6)), linewidth=0.9, zorder=1)
        for label in row_labels_a:
            y = pos_a[label]
            key = (site, label)
            not_evaluated = (label in RESTRICTED_TO) and (site not in RESTRICTED_SITES)
            if not_evaluated or key not in records_a:
                audit_rows.append({"figure": "Figure3", "site": site, "condition": label,
                                    "point": None, "ci_low": None, "ci_high": None, "status": "not_evaluated"})
                continue
            est, lo, hi = records_a[key]
            _errorbar(ax, y, est, lo, hi)
            audit_rows.append({"figure": "Figure3", "site": site, "condition": label,
                                "point": est, "ci_low": lo, "ci_high": hi, "status": "evaluated"})
        ax.set_title(
            r"$\mathbf{" + site.replace(" ", r"\ ") + "}$",
            fontsize=9.5, pad=4, color=COLOR_TEXT,
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_xticks(xticks)
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

    EXPORT_DPI = 320
    fig.savefig(png_path, dpi=EXPORT_DPI, pad_inches=0.03)
    fig.savefig(svg_path, pad_inches=0.03)
    fig.savefig(pdf_path, pad_inches=0.03)
    plt.close(fig)
    return pd.DataFrame(audit_rows), png_path, svg_path, pdf_path, (x_lo, x_hi)


def main():
    final_csv = TABLES_DIR / "figure4_v6_audit.csv"
    final_png = FIG_DIR / "figure4_v6_sensitivity.png"
    final_svg = FIG_DIR / "figure4_v6_sensitivity.svg"
    final_pdf = FIG_DIR / "figure4_v6_sensitivity.pdf"

    # Todo se calcula y serializa primero en un directorio de staging dentro
    # de outputs/ (mismo patrón que run_statistical_analysis.py), y solo se
    # promueve a las rutas finales (os.replace) si la comprobación de
    # regresión contra REGRESSION_REFERENCE pasa. Así nunca queda una
    # escritura parcial en las rutas canónicas si la validación falla.
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(OUT_DIR)))
    # Si el proceso termina antes de la promoción final (excepción, assert
    # fallido, SystemExit), este staging queda huérfano; el atexit lo limpia
    # siempre, incluso si nunca se llega a la promoción.
    atexit.register(lambda: shutil.rmtree(staging_dir, ignore_errors=True))
    (staging_dir / "tables").mkdir(parents=True, exist_ok=True)
    (staging_dir / "figures").mkdir(parents=True, exist_ok=True)
    staging_csv = staging_dir / "tables" / final_csv.name
    staging_png = staging_dir / "figures" / final_png.name
    staging_svg = staging_dir / "figures" / final_svg.name
    staging_pdf = staging_dir / "figures" / final_pdf.name

    audit, png, svg, pdf, xr = build_figure_sensitivity(staging_png, staging_svg, staging_pdf)
    audit.to_csv(staging_csv, index=False)

    # Numeric check: every PRE-EXISTING (site, condition) pair must be
    # byte-identical to v5 -- this script must only ADD rows, never change
    # the value of an existing one. Baseline: REGRESSION_REFERENCE (see
    # comment at its definition for why it is not figure4_v5_audit.csv).
    off = pd.read_csv(REGRESSION_REFERENCE)
    merged = audit.merge(off, on=["site", "condition"], suffixes=("_v6", "_v5"))
    assert len(merged) == len(off), (
        f"esperaba encontrar las {len(off)} filas de REGRESSION_REFERENCE en el audit "
        f"recién calculado, encontré {len(merged)} -- ¿cambió algún nombre de condición/sitio?"
    )
    for col in ["point", "ci_low", "ci_high"]:
        diff = (merged[f"{col}_v6"].fillna(0) - merged[f"{col}_v5"].fillna(0)).abs().max()
        print(f"Max abs diff vs v5 on shared rows, {col}: {diff}")
        assert diff < 1e-12, f"PRE-EXISTING ROW CHANGED -- HALT ({col})"
    assert (merged["status_v6"] == merged["status_v5"]).all(), "PRE-EXISTING ROW STATUS CHANGED -- HALT"

    n_new_evaluated = len(audit) - len(merged)
    print(f"Total rows: {len(audit)} (reference had {len(off)}); new rows: {len(audit) - len(off.merge(audit[['site','condition']].drop_duplicates(), on=['site','condition']))}")
    print("Evaluated:", (audit["status"] == "evaluated").sum(), " Not evaluated:", (audit["status"] == "not_evaluated").sum())

    # Validación superada: promover staging -> rutas canónicas finales.
    os.replace(staging_csv, final_csv)
    os.replace(staging_png, final_png)
    os.replace(staging_svg, final_svg)
    os.replace(staging_pdf, final_pdf)
    shutil.rmtree(staging_dir, ignore_errors=True)
    print("Outputs:", final_png, final_svg, final_pdf)

    from PIL import Image
    im = Image.open(final_png)
    w, h = im.size
    print(f"PNG: {w}x{h}px, effective dpi at 6.5in display width = {w/6.5:.1f}")


if __name__ == "__main__":
    main()
