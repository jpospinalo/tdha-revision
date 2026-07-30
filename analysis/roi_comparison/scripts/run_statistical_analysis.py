#!/usr/bin/env python3
"""Fase 2 del análisis comparativo de ROIs: estimación puntual, bootstrap
pareado por sujeto, contrastes, análisis de errores y figuras.

Lee exclusivamente:
  analysis_config.json, run_manifest.csv,
  outputs/data/subject_scores.csv, outputs/data/metrics_by_repeat.csv,
  outputs/tables/comparability_audit.csv (deben estar todas en PASS).

No vuelve a leer predicciones crudas ni recalcula una definición alternativa
de las métricas: reutiliza ``metrics_from_arrays`` del primer script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_analysis_dataset import metrics_from_arrays  # noqa: E402

SECONDARY_METRICS = ["balanced_accuracy", "f1_macro", "sensitivity", "specificity"]
ALL_BOOTSTRAP_METRICS = ["auc"] + SECONDARY_METRICS
SECONDARY_CONTRASTS = [(12, 18), (12, 39), (18, 39), (18, 116), (39, 116)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root))
    except ValueError:
        return str(Path(path).resolve())


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["roi_set"] = df["roi_set"].astype(int)
    df["include"] = df["include"].str.lower().map({"true": True, "false": False})
    return df[df["include"]].copy()


def check_audit_all_pass(audit_path: Path) -> None:
    audit = pd.read_csv(audit_path)
    if len(audit) != 16 or not (audit["status"] == "PASS").all():
        bad = audit[audit["status"] != "PASS"]
        raise SystemExit(
            f"comparability_audit.csv no está en PASS para las 16 corridas "
            f"(ver {audit_path}); filas con problema:\n{bad}"
        )
    if (audit["reconciliation_status"] != "PASS").any():
        raise SystemExit("comparability_audit.csv: reconciliación con README no está en PASS en todas las filas.")


def point_estimates(metrics_by_repeat: pd.DataFrame, site_order: list, roi_order: list) -> pd.DataFrame:
    """Media de las cinco métricas por repetición, por sitio y tamaño (8.1)."""
    grp = metrics_by_repeat.groupby(["site", "roi_set"])
    agg = grp[["auc"] + SECONDARY_METRICS + ["accuracy"]].mean().reset_index()
    n = grp["n_subjects"].first().reset_index(name="n_subjects")
    out = agg.merge(n, on=["site", "roi_set"])
    out["site"] = pd.Categorical(out["site"], categories=site_order, ordered=True)
    out["roi_set"] = pd.Categorical(out["roi_set"], categories=roi_order, ordered=True)
    return out.sort_values(["site", "roi_set"]).reset_index(drop=True)


def build_prob_tensor(subject_scores: pd.DataFrame, site: str, roi_order: list) -> tuple[np.ndarray, np.ndarray, list]:
    """Para un sitio: matriz (n_subjects, n_roi, 5) de y_prob y vector y_true,
    ordenados por subject_id ascendente (texto). Verifica que y_true e
    identidad de sujeto sean idénticos entre los cuatro tamaños (ya validado
    en el primer script, se recomprueba aquí por autoconsistencia)."""
    site_df = subject_scores[subject_scores["site"] == site]
    ref_roi = roi_order[0]
    ref = site_df[site_df["roi_set"] == ref_roi].sort_values("subject_id", key=lambda s: s.astype(str))
    subject_ids = ref["subject_id"].tolist()
    y_true = ref["y_true"].to_numpy()
    n = len(subject_ids)
    prob_cols = [f"y_prob_r{r}" for r in range(1, 6)]

    tensor = np.empty((n, len(roi_order), 5), dtype=np.float64)
    for j, roi in enumerate(roi_order):
        sub = site_df[site_df["roi_set"] == roi].set_index("subject_id").loc[subject_ids]
        if not np.array_equal(sub["y_true"].to_numpy(), y_true):
            raise SystemExit(f"sitio {site} ROI {roi}: y_true no coincide con la referencia ROI {ref_roi}")
        tensor[:, j, :] = sub[prob_cols].to_numpy()
    return tensor, y_true, subject_ids


def bootstrap_site(tensor: np.ndarray, y_true: np.ndarray, roi_order: list, n_iter: int, seed: int,
                    rng: np.random.Generator | None = None) -> dict:
    """Bootstrap estratificado por clase, pareado entre tamaños/repeticiones/métricas.

    Devuelve dict[(roi, metric)] -> np.ndarray de forma (n_iter,) con la media
    de las 5 repeticiones de esa métrica, para cada remuestreo.

    ``rng`` es un parámetro opcional de inyección de dependencia (por defecto
    None: se construye ``numpy.random.default_rng(seed)`` como especifica el
    plan 5.6). Pasar un generador ya construido permite reanudar un bootstrap
    exactamente donde quedó (el objeto es mutable y conserva su estado interno
    entre llamadas) sin alterar en nada el resultado de una ejecución continua
    con la misma semilla; se usa únicamente para verificar el pipeline en
    entornos con límite de tiempo por invocación, nunca desde la interfaz de
    línea de comandos documentada.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    control_idx = np.flatnonzero(y_true == 0)
    adhd_idx = np.flatnonzero(y_true == 1)
    n_control, n_adhd = len(control_idx), len(adhd_idx)
    n_roi = len(roi_order)
    n_metrics = len(ALL_BOOTSTRAP_METRICS)

    draws = np.empty((n_iter, n_roi, n_metrics), dtype=np.float64)

    for it in range(n_iter):
        boot_control = control_idx[rng.integers(0, n_control, size=n_control)]
        boot_adhd = adhd_idx[rng.integers(0, n_adhd, size=n_adhd)]
        boot_idx = np.concatenate([boot_control, boot_adhd])
        y_boot = y_true[boot_idx]

        for j in range(n_roi):
            repeat_vals = np.empty((5, n_metrics), dtype=np.float64)
            for r in range(5):
                probs = tensor[boot_idx, j, r]
                m = metrics_from_arrays(y_boot, probs)
                repeat_vals[r, :] = [m[name] for name in ALL_BOOTSTRAP_METRICS]
            draws[it, j, :] = repeat_vals.mean(axis=0)

    result = {}
    for j, roi in enumerate(roi_order):
        for k, metric in enumerate(ALL_BOOTSTRAP_METRICS):
            result[(roi, metric)] = draws[:, j, k]
    return result


def bilateral_ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def generate_d3_narrative(primary_df: pd.DataFrame) -> str:
    """Regla narrativa obligatoria de la sección 10.3. Deriva signos y
    solapamiento directamente de ``primary_12_vs_116.csv``; nunca hardcodea
    el patrón conocido durante la planificación."""
    signs = {}
    for _, row in primary_df.iterrows():
        d = row["delta_auc"]
        signs[row["site"]] = "positiva" if d > 0 else ("negativa" if d < 0 else "nula")
    sign_summary = "; ".join(f"{site}: {s}" for site, s in signs.items())

    lows = primary_df["bilateral_ci_low"].to_numpy()
    highs = primary_df["bilateral_ci_high"].to_numpy()
    common_overlap = float(np.max(lows)) <= float(np.min(highs))

    closing = (
        "La ausencia de una región común a los cuatro intervalos, o la ausencia de "
        "solapamiento en algún par, no demuestra por sí sola heterogeneidad estadística. "
        "Este análisis no estima ni contrasta heterogeneidad ni un efecto común."
    )

    if common_overlap:
        text = (
            f"Las estimaciones puntuales de la diferencia AUC 12−116 difieren en magnitud y "
            f"signo entre sitios: {sign_summary}. Sin embargo, los intervalos presentan una "
            f"región de solapamiento común y la precisión disponible no permite determinar si "
            f"este patrón refleja heterogeneidad real o variabilidad de muestreo. Los sitios se "
            f"presentan por separado; el análisis no estima ni contrasta un efecto común y "
            f"tampoco afirma heterogeneidad estadística."
        )
        return text

    sites = primary_df["site"].tolist()
    pair_lines = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            si, sj = sites[i], sites[j]
            lo_i, hi_i = lows[i], highs[i]
            lo_j, hi_j = lows[j], highs[j]
            overlaps = max(lo_i, lo_j) <= min(hi_i, hi_j)
            pair_lines.append(f"{si}-{sj}: {'con' if overlaps else 'sin'} solapamiento")
    pairs_summary = "; ".join(pair_lines)
    text = (
        f"Las estimaciones puntuales de la diferencia AUC 12−116 difieren en magnitud y signo "
        f"entre sitios: {sign_summary}. Los cuatro intervalos no comparten una región común; "
        f"por pares: {pairs_summary}. {closing}"
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config = load_config(args.config)
    if config.get("noninferiority_margin") is not None or config.get("noninferiority_margin_rationale") is not None:
        print("ERROR: noninferiority_margin / noninferiority_margin_rationale deben ser null (D2).", file=sys.stderr)
        return 1

    site_order = config["site_order"]
    roi_order = config["roi_order"]
    n_iter = config["bootstrap_iterations"]
    seed = config["bootstrap_seed"]
    ci_level = config["ci_level"]
    if abs(ci_level - 0.95) > 1e-9:
        raise SystemExit("ci_level distinto de 0.95 no está soportado por esta implementación (D2/plan 5.6).")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    for d in (tables_dir, data_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    audit_path = output_dir / "tables" / "comparability_audit.csv"
    check_audit_all_pass(audit_path)

    manifest = load_manifest(args.manifest)
    subject_scores = pd.read_csv(input_dir / "subject_scores.csv", dtype={"subject_id": str})
    metrics_by_repeat = pd.read_csv(input_dir / "metrics_by_repeat.csv")

    out_files = {
        "descriptive_performance.csv": tables_dir / "descriptive_performance.csv",
        "primary_12_vs_116.csv": tables_dir / "primary_12_vs_116.csv",
        "precision_diagnostics.csv": tables_dir / "precision_diagnostics.csv",
        "secondary_pairwise_comparisons.csv": tables_dir / "secondary_pairwise_comparisons.csv",
        "secondary_metric_intervals.csv": tables_dir / "secondary_metric_intervals.csv",
        "error_analysis_long.csv": data_dir / "error_analysis_long.csv",
        "subject_error_profiles.csv": data_dir / "subject_error_profiles.csv",
        "error_analysis_summary.csv": tables_dir / "error_analysis_summary.csv",
        "paired_roi_profiles.svg": figures_dir / "paired_roi_profiles.svg",
        "paired_roi_profiles.png": figures_dir / "paired_roi_profiles.png",
        "primary_contrast_forest.svg": figures_dir / "primary_contrast_forest.svg",
        "primary_contrast_forest.png": figures_dir / "primary_contrast_forest.png",
        "analysis_manifest.json": output_dir / "analysis_manifest.json",
    }
    if not args.overwrite:
        existing = [str(p) for p in out_files.values() if p.exists()]
        if existing:
            print("ERROR: las siguientes salidas ya existen; use --overwrite para reemplazarlas:", file=sys.stderr)
            for p in existing:
                print(f"  - {p}", file=sys.stderr)
            return 1

    # ---- 8.1 Estimación puntual -------------------------------------------------
    pts = point_estimates(metrics_by_repeat, site_order, roi_order)

    # ---- 8.2 Bootstrap por sitio --------------------------------------------------
    site_timings = {}
    bootstrap_by_site = {}
    for site in site_order:
        t0 = time.perf_counter()
        tensor, y_true, subject_ids = build_prob_tensor(subject_scores, site, roi_order)
        draws = bootstrap_site(tensor, y_true, roi_order, n_iter, seed)
        bootstrap_by_site[site] = draws
        elapsed = time.perf_counter() - t0
        site_timings[site] = elapsed
        print(f"[bootstrap] {site}: n_subjects={len(subject_ids)} tiempo={elapsed:.1f}s "
              f"({elapsed / n_iter * 1000:.2f} ms/iter)")

    total_time = sum(site_timings.values())

    # ---- descriptive_performance.csv (16 filas) -----------------------------------
    desc_rows = []
    for site in site_order:
        for roi in roi_order:
            row = pts[(pts["site"] == site) & (pts["roi_set"] == roi)].iloc[0]
            auc_lo, auc_hi = bilateral_ci(bootstrap_by_site[site][(roi, "auc")])
            desc_rows.append({
                "site": site, "roi_set": roi, "n_subjects": int(row["n_subjects"]),
                "mean_auc": row["auc"], "auc_bilateral_ci_low": auc_lo, "auc_bilateral_ci_high": auc_hi,
                "balanced_accuracy": row["balanced_accuracy"], "f1_macro": row["f1_macro"],
                "sensitivity": row["sensitivity"], "specificity": row["specificity"],
            })
    descriptive_performance = pd.DataFrame(desc_rows)
    descriptive_performance.to_csv(out_files["descriptive_performance.csv"], index=False)

    # ---- primary_12_vs_116.csv y precision_diagnostics.csv (4 filas cada una) -----
    primary_rows = []
    precision_rows = []
    for site in site_order:
        auc12 = pts[(pts["site"] == site) & (pts["roi_set"] == 12)]["auc"].iloc[0]
        auc116 = pts[(pts["site"] == site) & (pts["roi_set"] == 116)]["auc"].iloc[0]
        delta = auc12 - auc116
        boot_delta = bootstrap_by_site[site][(12, "auc")] - bootstrap_by_site[site][(116, "auc")]
        lo, hi = bilateral_ci(boot_delta)
        se = float(boot_delta.std(ddof=1))
        n_subjects = int(pts[(pts["site"] == site) & (pts["roi_set"] == 12)]["n_subjects"].iloc[0])
        primary_rows.append({
            "site": site, "auc_12": auc12, "auc_116": auc116,
            "delta_auc": delta, "bilateral_ci_low": lo, "bilateral_ci_high": hi,
        })
        precision_rows.append({
            "site": site, "n_subjects": n_subjects, "delta_auc": delta,
            "bootstrap_standard_error": se, "bilateral_ci_low": lo, "bilateral_ci_high": hi,
            "bilateral_interval_width": hi - lo,
        })
    primary_df = pd.DataFrame(primary_rows)
    primary_df.to_csv(out_files["primary_12_vs_116.csv"], index=False)
    precision_df = pd.DataFrame(precision_rows)
    precision_df.to_csv(out_files["precision_diagnostics.csv"], index=False)
    d3_narrative = generate_d3_narrative(primary_df)
    print("\n[narrativa D3, sección 10.3]\n" + d3_narrative + "\n")

    # ---- secondary_pairwise_comparisons.csv (100 filas) ----------------------------
    sec_rows = []
    for site in site_order:
        for left, right in SECONDARY_CONTRASTS:
            for metric in ALL_BOOTSTRAP_METRICS:
                left_pt = pts[(pts["site"] == site) & (pts["roi_set"] == left)][metric].iloc[0]
                right_pt = pts[(pts["site"] == site) & (pts["roi_set"] == right)][metric].iloc[0]
                diff_pt = left_pt - right_pt
                boot_diff = bootstrap_by_site[site][(left, metric)] - bootstrap_by_site[site][(right, metric)]
                lo, hi = bilateral_ci(boot_diff)
                sec_rows.append({
                    "site": site, "metric": metric, "contrast": f"{left}-{right}",
                    "left_roi": left, "right_roi": right,
                    "estimate": diff_pt, "bilateral_ci_low": lo, "bilateral_ci_high": hi,
                })
    secondary_pairwise = pd.DataFrame(sec_rows)
    secondary_pairwise.to_csv(out_files["secondary_pairwise_comparisons.csv"], index=False)

    # ---- secondary_metric_intervals.csv (64 filas) ---------------------------------
    interval_rows = []
    for site in site_order:
        for roi in roi_order:
            for metric in SECONDARY_METRICS:
                pt = pts[(pts["site"] == site) & (pts["roi_set"] == roi)][metric].iloc[0]
                lo, hi = bilateral_ci(bootstrap_by_site[site][(roi, metric)])
                interval_rows.append({
                    "site": site, "roi_set": roi, "metric": metric,
                    "estimate": pt, "bilateral_ci_low": lo, "bilateral_ci_high": hi,
                })
    secondary_intervals = pd.DataFrame(interval_rows)
    secondary_intervals.to_csv(out_files["secondary_metric_intervals.csv"], index=False)

    # ---- 9. Análisis de errores (12 vs 116) -----------------------------------------
    error_rows = []
    for site in site_order:
        site_df = subject_scores[subject_scores["site"] == site]
        s12 = site_df[site_df["roi_set"] == 12].set_index("subject_id")
        s116 = site_df[site_df["roi_set"] == 116].set_index("subject_id")
        subject_ids = sorted(s12.index.tolist(), key=str)
        for sid in subject_ids:
            y_true = int(s12.loc[sid, "y_true"])
            for r in range(1, 6):
                p12 = float(s12.loc[sid, f"y_prob_r{r}"])
                p116 = float(s116.loc[sid, f"y_prob_r{r}"])
                pred12 = int(p12 >= 0.5)
                pred116 = int(p116 >= 0.5)
                c12 = pred12 == y_true
                c116 = pred116 == y_true
                if c12 and c116:
                    category = "both_correct"
                elif c12 and not c116:
                    category = "correct_12_only"
                elif not c12 and c116:
                    category = "correct_116_only"
                else:
                    category = "both_incorrect"
                error_rows.append({
                    "site": site, "subject_id": sid, "y_true": y_true, "repeat": r,
                    "y_prob_12": p12, "y_prob_116": p116, "pred_12": pred12, "pred_116": pred116,
                    "correct_12": c12, "correct_116": c116, "category": category,
                    "probability_difference_12_minus_116": p12 - p116,
                })
    error_long = pd.DataFrame(error_rows)
    error_long.to_csv(out_files["error_analysis_long.csv"], index=False)

    profile_rows = []
    for (site, sid), sub in error_long.groupby(["site", "subject_id"], sort=False):
        sub = sub.sort_values("repeat")
        cat_counts = sub["category"].value_counts()
        preds12 = sub["pred_12"].tolist()
        preds116 = sub["pred_116"].tolist()
        profile_rows.append({
            "site": site, "subject_id": sid, "y_true": int(sub["y_true"].iloc[0]),
            "correct_12_count": int(sub["correct_12"].sum()),
            "correct_116_count": int(sub["correct_116"].sum()),
            "both_correct_count": int(cat_counts.get("both_correct", 0)),
            "correct_12_only_count": int(cat_counts.get("correct_12_only", 0)),
            "correct_116_only_count": int(cat_counts.get("correct_116_only", 0)),
            "both_incorrect_count": int(cat_counts.get("both_incorrect", 0)),
            "y_prob_12_mean": float(sub["y_prob_12"].mean()), "y_prob_12_sd": float(sub["y_prob_12"].std(ddof=0)),
            "y_prob_116_mean": float(sub["y_prob_116"].mean()), "y_prob_116_sd": float(sub["y_prob_116"].std(ddof=0)),
            "n_positive_predictions_12": int(sum(preds12)), "n_positive_predictions_116": int(sum(preds116)),
            "unstable_pred_12": len(set(preds12)) > 1, "unstable_pred_116": len(set(preds116)) > 1,
        })
    subject_error_profiles = pd.DataFrame(profile_rows)
    subject_error_profiles.to_csv(out_files["subject_error_profiles.csv"], index=False)

    summary = (error_long.groupby(["site", "y_true", "category"]).size()
               .reset_index(name="count"))
    summary.to_csv(out_files["error_analysis_summary.csv"], index=False)

    # ---- 10.1 Perfiles por ROI ------------------------------------------------------
    all_lows, all_highs = [], []
    for site in site_order:
        for roi in roi_order:
            lo, hi = bilateral_ci(bootstrap_by_site[site][(roi, "auc")])
            all_lows.append(lo)
            all_highs.append(hi)
    margin = 0.03
    y_min = max(0.0, min(all_lows) - margin)
    y_max = min(1.0, max(all_highs) + margin)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    jitter = np.linspace(-0.15, 0.15, 5)
    for ax, site in zip(axes, site_order):
        means, los, his = [], [], []
        for roi in roi_order:
            m = pts[(pts["site"] == site) & (pts["roi_set"] == roi)]["auc"].iloc[0]
            lo, hi = bilateral_ci(bootstrap_by_site[site][(roi, "auc")])
            means.append(m)
            los.append(lo)
            his.append(hi)
        x = np.arange(len(roi_order))
        means_arr, los_arr, his_arr = np.array(means), np.array(los), np.array(his)
        # Clip defensivo: en una cola bootstrap muy asimétrica la media puntual
        # (calculada sobre los datos reales, no sobre los remuestreos) podría en
        # principio caer fuera de su propio intervalo percentil; sin el clip,
        # errorbar recibiría una longitud negativa. No ocurre con los datos
        # reales de este análisis (verificado), pero se deja como salvaguarda.
        yerr_low = np.clip(means_arr - los_arr, 0, None)
        yerr_high = np.clip(his_arr - means_arr, 0, None)
        ax.errorbar(x, means, yerr=[yerr_low, yerr_high],
                     fmt="o", color="black", capsize=4, zorder=3)
        for i, roi in enumerate(roi_order):
            sub = metrics_by_repeat[(metrics_by_repeat["site"] == site) & (metrics_by_repeat["roi_set"] == roi)]
            reps = sub.sort_values("repeat")["auc"].to_numpy()
            ax.scatter(np.full(5, x[i]) + jitter, reps, color="gray", s=10, zorder=2)
        ax.axhline(0.5, color="lightgray", linestyle="--", linewidth=1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in roi_order])
        ax.set_title(site)
        ax.set_xlabel("ROIs")
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel("AUC OOF (media de 5 repeticiones)")
    fig.suptitle("Perfiles AUC por tamaño de ROI (barras: IC bootstrap bilateral 95%)")
    fig.tight_layout()
    fig.savefig(out_files["paired_roi_profiles.svg"])
    fig.savefig(out_files["paired_roi_profiles.png"], dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(site_order))
    deltas = [primary_df[primary_df["site"] == s]["delta_auc"].iloc[0] for s in site_order]
    los2 = [primary_df[primary_df["site"] == s]["bilateral_ci_low"].iloc[0] for s in site_order]
    his2 = [primary_df[primary_df["site"] == s]["bilateral_ci_high"].iloc[0] for s in site_order]
    deltas_arr, los2_arr, his2_arr = np.array(deltas), np.array(los2), np.array(his2)
    xerr_low = np.clip(deltas_arr - los2_arr, 0, None)
    xerr_high = np.clip(his2_arr - deltas_arr, 0, None)
    ax2.errorbar(deltas, y_pos, xerr=[xerr_low, xerr_high],
                  fmt="o", color="black", capsize=4)
    ax2.axvline(0, color="lightgray", linestyle="--", linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(site_order)
    ax2.set_xlabel("Diferencia AUC (12 − 116), IC bootstrap bilateral 95%")
    ax2.set_title("Contraste principal por sitio (sin efecto combinado)")
    fig2.tight_layout()
    fig2.savefig(out_files["primary_contrast_forest.svg"])
    fig2.savefig(out_files["primary_contrast_forest.png"], dpi=150)
    plt.close(fig2)

    # ---- analysis_manifest.json ------------------------------------------------------
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
    except Exception:
        commit = None
    try:
        status_lines = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root).decode().splitlines()
    except Exception:
        status_lines = []
    changed_paths = [line[3:] for line in status_lines]
    non_analysis_changes = [p for p in changed_paths if not p.startswith("analysis/")]

    import sklearn
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }

    plan_path = repo_root / "analysis" / "roi_comparison" / "analysis_plan.md"
    plan_sha256 = sha256_file(plan_path) if plan_path.exists() else None

    run_hashes = []
    for _, row in manifest.sort_values(["site", "roi_set"]).iterrows():
        run_dir = repo_root / row["relative_path"]
        run_hashes.append({
            "site": row["site"], "roi_set": int(row["roi_set"]), "run_id": row["run_id"],
            "config_json_sha256": sha256_file(run_dir / "config.json"),
            "predictions_val_csv_sha256": sha256_file(run_dir / "predictions_val.csv"),
            "folds_csv_sha256": sha256_file(run_dir / "folds.csv"),
        })

    output_hashes = {}
    for name, path in out_files.items():
        if name == "analysis_manifest.json":
            continue
        output_hashes[name] = {"path": relative_to_repo(path, repo_root), "sha256": sha256_file(path)}
    output_hashes["subject_scores.csv"] = {
        "path": relative_to_repo(input_dir / "subject_scores.csv", repo_root),
        "sha256": sha256_file(input_dir / "subject_scores.csv"),
    }
    output_hashes["metrics_by_repeat.csv"] = {
        "path": relative_to_repo(input_dir / "metrics_by_repeat.csv", repo_root),
        "sha256": sha256_file(input_dir / "metrics_by_repeat.csv"),
    }
    output_hashes["comparability_audit.csv"] = {
        "path": relative_to_repo(audit_path, repo_root), "sha256": sha256_file(audit_path),
    }

    prob_matrix_bytes = sum(
        build_prob_tensor(subject_scores, site, roi_order)[0].nbytes for site in site_order
    )

    manifest_out = {
        "plan_version": "5.6",
        "plan_sha256": plan_sha256,
        "analysis_config_sha256": sha256_file(args.config),
        "run_manifest_sha256": sha256_file(args.manifest),
        "results_readme_sha256": sha256_file(repo_root / "results" / "README.md"),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "command": " ".join(sys.argv),
        "repo_commit": commit,
        "changed_paths_outside_analysis": non_analysis_changes,
        "versions": versions,
        "runs": run_hashes,
        "bootstrap": {
            "iterations": n_iter, "seed": seed, "rng": config["bootstrap_rng"],
            "seed_scope": config["bootstrap_seed_scope"], "method": config["bootstrap_method"],
            "quantile_method": config["bootstrap_quantile_method"], "ci_level": ci_level,
        },
        "timing_seconds": {**site_timings, "total": total_time},
        "prob_matrix_bytes_float64": int(prob_matrix_bytes),
        "d1_d5_resolution": {
            "D1_primary_metric": "mean_repeat_oof_auc",
            "D1_secondary_metrics": SECONDARY_METRICS,
            "D1_audit_metric": "accuracy",
            "D2_noninferiority_margin": None,
            "D2_noninferiority_margin_rationale": None,
            "D2_scope": "estimacion_pura_sin_dictamen_binario",
            "D3_pooling": "ninguno; cada sitio se presenta por separado; no se afirma efecto comun ni heterogeneidad",
            "D4_bootstrap": "pareado, estratificado por sujeto, condicionado a predicciones/entrenamientos/particiones existentes",
            "D5_estimand": "desempeno medio del pipeline de validacion cruzada (no ensamble de probabilidades, no modelo final desplegado)",
        },
        "reconciliation_status": "PASS (16/16, ver comparability_audit.csv)",
        "preregistration_status": (
            "No es una preinscripcion prospectiva ciega a los resultados. El plan 5.6 se "
            "cerro despues de una revision de factibilidad en la que ya eran visibles las "
            "diferencias medias y varianzas por sitio entre 12 y 116 ROIs (plan, secciones "
            "1 y 2). Los resultados se presentan como estimacion con apoyo exploratorio, no "
            "como confirmacion definitiva; una afirmacion confirmatoria requeriria una "
            "cohorte o conjunto de datos externo no usado en estas decisiones."
        ),
        "d3_narrative_generated": d3_narrative,
        "figure_vertical_limits": {"auc_panel_y_min": y_min, "auc_panel_y_max": y_max},
        "outputs": output_hashes,
        "results_read_only": len(non_analysis_changes) == 0,
        "limitation": (
            "Los intervalos bootstrap estan condicionados a las predicciones, entrenamientos y "
            "cinco particiones de validacion cruzada existentes; no capturan la variabilidad de "
            "un reentrenamiento o reparticion completos."
        ),
    }
    with open(out_files["analysis_manifest.json"], "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2, ensure_ascii=False, sort_keys=False)

    print(f"Análisis estadístico completo. Tiempo total bootstrap: {total_time:.1f}s "
          f"({total_time / 60:.1f} min) para {n_iter} iteraciones x {len(site_order)} sitios.")
    print(f"  descriptive_performance.csv: {len(descriptive_performance)} filas")
    print(f"  primary_12_vs_116.csv: {len(primary_df)} filas")
    print(f"  precision_diagnostics.csv: {len(precision_df)} filas")
    print(f"  secondary_pairwise_comparisons.csv: {len(secondary_pairwise)} filas")
    print(f"  secondary_metric_intervals.csv: {len(secondary_intervals)} filas")
    print(f"  error_analysis_long.csv: {len(error_long)} filas")
    print(f"  subject_error_profiles.csv: {len(subject_error_profiles)} filas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
