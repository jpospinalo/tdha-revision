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
import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_analysis_dataset import (  # noqa: E402
    EXPECTED_SUBJECT_COUNTS,
    REQUIRED_ARTIFACTS,
    ValidationError,
    metrics_from_arrays,
    resolve_and_check_output_paths,
    validate_analysis_config,
)

SECONDARY_METRICS = ["balanced_accuracy", "f1_macro", "sensitivity", "specificity"]
ALL_BOOTSTRAP_METRICS = ["auc"] + SECONDARY_METRICS
SECONDARY_CONTRASTS = [(12, 18), (12, 39), (18, 39), (18, 116), (39, 116)]

# Hash canónico del plan 5.6 aprobado por el equipo (analysis_plan.md debe
# ser una copia byte por byte). Un plan ausente o con hash distinto debe
# detener la ejecución antes de iniciar el bootstrap (CORRECCIONES_V19 §5).
CANONICAL_PLAN_SHA256 = "199857a46006a082d97f6a055ffdaaa075fd25be87bbb4147e806aae28367163"


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


EXPECTED_SUBJECT_SCORES_COLUMNS = [
    "site", "roi_set", "subject_id", "y_true",
    "y_prob_r1", "y_prob_r2", "y_prob_r3", "y_prob_r4", "y_prob_r5",
    "y_prob_mean", "y_prob_sd", "n_positive_predictions",
]
EXPECTED_METRICS_BY_REPEAT_COLUMNS = [
    "site", "roi_set", "repeat", "n_subjects", "n_control", "n_adhd",
    "auc", "balanced_accuracy", "f1_macro", "sensitivity", "specificity", "accuracy",
]
PHASE2_NUMERIC_METRIC_COLUMNS = ["auc", "balanced_accuracy", "f1_macro", "sensitivity", "specificity", "accuracy"]


def check_audit_all_pass(audit_path: Path, site_order: list, roi_order: list) -> None:
    """CORRECCIONES_V19 §8: revalida comparability_audit.csv en la Fase 2,
    de forma independiente de lo que ya validó la Fase 1 -- protege contra un
    archivo intermedio corrupto o editado a mano entre ambas fases."""
    audit = pd.read_csv(audit_path)
    expected_combos = {(s, r) for s in site_order for r in roi_order}
    actual_combos = set(zip(audit["site"], audit["roi_set"]))
    if len(audit) != 16:
        raise ValidationError(f"comparability_audit.csv: {len(audit)} filas, se esperaban 16 (ver {audit_path})")
    if audit.duplicated(subset=["site", "roi_set"]).any():
        raise ValidationError("comparability_audit.csv: combinaciones (site, roi_set) duplicadas")
    missing, extra = expected_combos - actual_combos, actual_combos - expected_combos
    if missing or extra:
        raise ValidationError(
            f"comparability_audit.csv: combinaciones (site, roi_set) inesperadas "
            f"(faltan={missing}, sobran={extra})"
        )
    if not (audit["status"] == "PASS").all():
        bad = audit[audit["status"] != "PASS"]
        raise ValidationError(
            f"comparability_audit.csv no está en PASS para las 16 corridas "
            f"(ver {audit_path}); filas con problema:\n{bad}"
        )
    if (audit["reconciliation_status"] != "PASS").any():
        raise ValidationError("comparability_audit.csv: reconciliación con README no está en PASS en todas las filas.")


def validate_subject_scores_for_phase2(
    subject_scores: pd.DataFrame, site_order: list, roi_order: list, expected_subject_counts: dict
) -> None:
    """CORRECCIONES_V19 §8: revalida subject_scores.csv antes del bootstrap,
    de forma independiente de la Fase 1."""
    if list(subject_scores.columns) != EXPECTED_SUBJECT_SCORES_COLUMNS:
        raise ValidationError(
            f"subject_scores.csv: columnas {list(subject_scores.columns)}, "
            f"se esperaba {EXPECTED_SUBJECT_SCORES_COLUMNS}"
        )
    if len(subject_scores) != 1860:
        raise ValidationError(f"subject_scores.csv: {len(subject_scores)} filas, se esperaban 1860")
    if subject_scores.duplicated(subset=["site", "roi_set", "subject_id"]).any():
        raise ValidationError("subject_scores.csv: combinaciones (site, roi_set, subject_id) duplicadas")

    prob_cols = [f"y_prob_r{r}" for r in range(1, 6)]
    probs = subject_scores[prob_cols].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        raise ValidationError("subject_scores.csv: probabilidades no finitas (NaN/inf) en columnas científicas")
    if (probs < 0).any() or (probs > 1).any():
        raise ValidationError("subject_scores.csv: probabilidades fuera de [0,1]")
    if not set(subject_scores["y_true"].unique()).issubset({0, 1}):
        raise ValidationError(f"subject_scores.csv: y_true fuera de {{0,1}}: {sorted(subject_scores['y_true'].unique())}")
    if subject_scores[["y_true"] + prob_cols].isna().any().any():
        raise ValidationError("subject_scores.csv: NaN en columnas científicas (y_true o y_prob_r*)")

    for site in site_order:
        sub = subject_scores[subject_scores["site"] == site]
        n_expected = expected_subject_counts[site]
        for roi in roi_order:
            n = len(sub[sub["roi_set"] == roi])
            if n != n_expected:
                raise ValidationError(
                    f"subject_scores.csv: {site}/{roi}: {n} filas, se esperaban {n_expected} (una por sujeto)"
                )
        ref_roi = roi_order[0]
        ref = sub[sub["roi_set"] == ref_roi].set_index("subject_id")["y_true"]
        for roi in roi_order[1:]:
            cur = sub[sub["roi_set"] == roi].set_index("subject_id")["y_true"]
            if set(cur.index) != set(ref.index):
                raise ValidationError(
                    f"subject_scores.csv: sujetos distintos entre ROI {ref_roi} y {roi} en sitio {site}"
                )
            if not (cur.reindex(ref.index) == ref).all():
                raise ValidationError(
                    f"subject_scores.csv: y_true distinto entre ROI {ref_roi} y {roi} en sitio {site}"
                )


def validate_metrics_by_repeat_for_phase2(metrics_by_repeat: pd.DataFrame, site_order: list, roi_order: list) -> None:
    """CORRECCIONES_V19 §8: revalida metrics_by_repeat.csv antes del
    bootstrap, de forma independiente de la Fase 1."""
    if list(metrics_by_repeat.columns) != EXPECTED_METRICS_BY_REPEAT_COLUMNS:
        raise ValidationError(
            f"metrics_by_repeat.csv: columnas {list(metrics_by_repeat.columns)}, "
            f"se esperaba {EXPECTED_METRICS_BY_REPEAT_COLUMNS}"
        )
    if len(metrics_by_repeat) != 80:
        raise ValidationError(f"metrics_by_repeat.csv: {len(metrics_by_repeat)} filas, se esperaban 80")
    if metrics_by_repeat.duplicated(subset=["site", "roi_set", "repeat"]).any():
        raise ValidationError("metrics_by_repeat.csv: combinaciones (site, roi_set, repeat) duplicadas")
    for site in site_order:
        for roi in roi_order:
            sub = metrics_by_repeat[(metrics_by_repeat["site"] == site) & (metrics_by_repeat["roi_set"] == roi)]
            reps = sorted(sub["repeat"].tolist())
            if reps != [1, 2, 3, 4, 5]:
                raise ValidationError(f"metrics_by_repeat.csv: {site}/{roi}: repeticiones {reps}, se esperaba 1..5")
            n_control = int(sub["n_control"].iloc[0])
            n_adhd = int(sub["n_adhd"].iloc[0])
            n_subjects = int(sub["n_subjects"].iloc[0])
            if n_control + n_adhd != n_subjects:
                raise ValidationError(
                    f"metrics_by_repeat.csv: {site}/{roi}: n_control+n_adhd={n_control + n_adhd} != "
                    f"n_subjects={n_subjects}"
                )
    vals = metrics_by_repeat[PHASE2_NUMERIC_METRIC_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValidationError("metrics_by_repeat.csv: valores no finitos (NaN/inf) en columnas de métricas")
    if (vals < 0).any() or (vals > 1).any():
        raise ValidationError("metrics_by_repeat.csv: valores de métricas fuera de [0,1]")


def validate_autoconsistency(
    subject_scores: pd.DataFrame, metrics_by_repeat: pd.DataFrame, site_order: list, roi_order: list,
    atol: float = 1e-12,
) -> None:
    """CORRECCIONES_V19 §8: recalcula las seis métricas de cada una de las 80
    combinaciones (site, roi_set, repeat) directamente desde
    subject_scores.csv y las compara con metrics_by_repeat.csv dentro de una
    tolerancia absoluta de 1e-12. Detiene la ejecución antes del bootstrap si
    alguna métrica fue alterada entre la Fase 1 y la Fase 2."""
    for site in site_order:
        for roi in roi_order:
            sub = subject_scores[(subject_scores["site"] == site) & (subject_scores["roi_set"] == roi)]
            y_true = sub["y_true"].to_numpy()
            for r in range(1, 6):
                y_prob = sub[f"y_prob_r{r}"].to_numpy()
                recomputed = metrics_from_arrays(y_true, y_prob)
                row_df = metrics_by_repeat[
                    (metrics_by_repeat["site"] == site) & (metrics_by_repeat["roi_set"] == roi)
                    & (metrics_by_repeat["repeat"] == r)
                ]
                if len(row_df) != 1:
                    raise ValidationError(
                        f"autoconsistencia: {len(row_df)} filas para {site}/{roi}/repeat {r}, se esperaba 1"
                    )
                row = row_df.iloc[0]
                for metric_name in PHASE2_NUMERIC_METRIC_COLUMNS:
                    diff = abs(float(recomputed[metric_name]) - float(row[metric_name]))
                    if diff > atol:
                        raise ValidationError(
                            f"autoconsistencia falló para {site}/{roi}/repeat {r}/{metric_name}: "
                            f"recalculado={recomputed[metric_name]!r} vs metrics_by_repeat.csv={row[metric_name]!r} "
                            f"(diferencia {diff!r} > tolerancia {atol!r})"
                        )


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
                    rng: np.random.Generator | None = None, site: str | None = None,
                    progress_every: int = 1000) -> dict:
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

    ``site`` y ``progress_every`` solo controlan un mensaje de progreso
    impreso cada ``progress_every`` iteraciones (CORRECCIONES_V19 §10). La
    condición de impresión no llama al RNG, no cambia el orden de los
    bucles, no cambia los índices bootstrap y no guarda los remuestreos
    crudos: el resultado con y sin mensajes es numéricamente idéntico.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    control_idx = np.flatnonzero(y_true == 0)
    adhd_idx = np.flatnonzero(y_true == 1)
    n_control, n_adhd = len(control_idx), len(adhd_idx)
    n_roi = len(roi_order)
    n_metrics = len(ALL_BOOTSTRAP_METRICS)
    site_label = site if site is not None else "?"

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

        if progress_every and (it + 1) % progress_every == 0:
            pct = 100.0 * (it + 1) / n_iter
            print(f"[bootstrap] {site_label} · {it + 1}/{n_iter} iteraciones · {pct:.0f}%")

    result = {}
    for j, roi in enumerate(roi_order):
        for k, metric in enumerate(ALL_BOOTSTRAP_METRICS):
            result[(roi, metric)] = draws[:, j, k]
    return result


def bilateral_ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def compute_primary_delta(auc_12: float, auc_116: float) -> float:
    """Convención de signo del contraste principal: siempre 12 menos 116
    (sección 14, control 8 de las instrucciones v1.5)."""
    return auc_12 - auc_116


PRECISION_DIAGNOSTICS_COLUMNS = [
    "site", "n_subjects", "delta_auc", "bootstrap_standard_error",
    "bilateral_ci_low", "bilateral_ci_high", "bilateral_interval_width",
]


def build_precision_diagnostics_row(
    site: str, n_subjects: int, delta_auc: float, se: float, lo: float, hi: float
) -> dict:
    """Fila de precision_diagnostics.csv: deliberadamente sin cuantiles ni
    semi-amplitudes unilaterales (D2, sin margen de no inferioridad)."""
    return {
        "site": site, "n_subjects": n_subjects, "delta_auc": delta_auc,
        "bootstrap_standard_error": se, "bilateral_ci_low": lo, "bilateral_ci_high": hi,
        "bilateral_interval_width": hi - lo,
    }


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


def compute_input_hash_inventory(repo_root: Path, manifest: pd.DataFrame) -> dict:
    """CORRECCIONES_V19 §4/§11: inventario determinista de SHA-256 de los
    insumos protegidos, capturado al inicio y al final de la ejecución
    productiva. ``results_read_only`` se calcula comparando dos llamadas a
    esta función (antes/después), nunca a partir de la disponibilidad de
    Git. Excluye ``__pycache__`` y archivos ocultos de sistema (por ejemplo
    ``.DS_Store``) bajo ``src/``: son artefactos derivados que cambian sin
    que el código fuente cambie, y su inclusión generaría falsos positivos.
    """
    results_readme_path = repo_root / "results" / "README.md"
    run_artifacts = {}
    for _, row in manifest.sort_values(["site", "roi_set"]).iterrows():
        run_dir = repo_root / row["relative_path"]
        key = f"{row['site']}_{row['roi_set']}"
        run_artifacts[key] = {
            a: sha256_file(run_dir / a) for a in REQUIRED_ARTIFACTS if (run_dir / a).exists()
        }

    requirements_path = repo_root / "requirements.txt"
    notebook_path = repo_root / "tdha_experimentos.ipynb"
    src_dir = repo_root / "src"
    src_hashes = {}
    if src_dir.is_dir():
        for p in sorted(src_dir.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or p.name.startswith("."):
                continue
            src_hashes[str(p.relative_to(repo_root))] = sha256_file(p)

    return {
        "results_readme_sha256": sha256_file(results_readme_path) if results_readme_path.exists() else None,
        "run_artifacts": run_artifacts,
        "protected_files": {
            "requirements_txt_sha256": sha256_file(requirements_path) if requirements_path.exists() else None,
            "tdha_experimentos_ipynb_sha256": sha256_file(notebook_path) if notebook_path.exists() else None,
            "src": src_hashes,
        },
    }


def hash_inventory_fingerprint(inventory: dict) -> str:
    """Huella agregada determinista de un inventario de hashes (para
    comparar antes/después en una sola cadena, además de la comparación
    campo por campo)."""
    canonical = json.dumps(inventory, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_git_status(repo_root: Path) -> tuple[str | None, list[str] | None, str]:
    """(commit, status_lines, provenance_status). Si Git no está disponible
    -por ejemplo, ejecutando desde un ZIP en Colab- devuelve (None, None,
    "unavailable"): CORRECCIONES_V19 §11 exige no inventar un commit ficticio
    ni una lista vacía en ese caso (una lista vacía se leería, incorrectamente,
    como "no hay cambios")."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
        status_lines = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().splitlines()
        return commit, status_lines, "available"
    except Exception:
        return None, None, "unavailable"


def build_run_hashes(manifest: pd.DataFrame, repo_root: Path) -> list[dict]:
    """CORRECCIONES_V19 §11.1: config_hash, run_id y los tres SHA-256 ya
    registrados, para cada una de las 16 corridas del manifiesto."""
    run_hashes = []
    for _, row in manifest.sort_values(["site", "roi_set"]).iterrows():
        run_dir = repo_root / row["relative_path"]
        run_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        run_hashes.append({
            "site": row["site"], "roi_set": int(row["roi_set"]), "run_id": row["run_id"],
            "config_hash": run_cfg.get("config_hash"),
            "config_json_sha256": sha256_file(run_dir / "config.json"),
            "predictions_val_csv_sha256": sha256_file(run_dir / "predictions_val.csv"),
            "folds_csv_sha256": sha256_file(run_dir / "folds.csv"),
        })
    return run_hashes


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

    # CORRECCIONES_V19 §11: estado de Git capturado al inicio, antes de
    # cualquier cómputo (se recaptura al final para distinguir cambios
    # preexistentes de cambios nuevos).
    git_commit_before, git_status_before, git_provenance_status = get_git_status(repo_root)

    plan_path = repo_root / "analysis" / "roi_comparison" / "analysis_plan.md"
    if not plan_path.exists():
        print(f"ERROR: no se encontró el plan canónico en {plan_path}.", file=sys.stderr)
        return 1
    plan_sha256_early = sha256_file(plan_path)
    if plan_sha256_early != CANONICAL_PLAN_SHA256:
        print(
            f"ERROR: analysis_plan.md no coincide con el hash canónico del plan 5.6 aprobado.\n"
            f"  esperado: {CANONICAL_PLAN_SHA256}\n"
            f"  actual:   {plan_sha256_early}",
            file=sys.stderr,
        )
        return 1

    config = load_config(args.config)
    try:
        validate_analysis_config(config)
    except ValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
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
    manifest = load_manifest(args.manifest)
    subject_scores = pd.read_csv(input_dir / "subject_scores.csv", dtype={"subject_id": str})
    metrics_by_repeat = pd.read_csv(input_dir / "metrics_by_repeat.csv")

    # CORRECCIONES_V19 §4/§11: inventario de hashes de insumos, capturado
    # antes de iniciar el bootstrap (se recaptura al final).
    input_hash_inventory_before = compute_input_hash_inventory(repo_root, manifest)

    # CORRECCIONES_V19 §8: revalidar todas las entradas derivadas de la Fase 1
    # antes de iniciar el bootstrap; no producir ninguna tabla científica si
    # algo falla aquí. Independiente de lo que ya validó build_analysis_dataset.py.
    try:
        check_audit_all_pass(audit_path, site_order, roi_order)
        validate_subject_scores_for_phase2(subject_scores, site_order, roi_order, EXPECTED_SUBJECT_COUNTS)
        validate_metrics_by_repeat_for_phase2(metrics_by_repeat, site_order, roi_order)
        validate_autoconsistency(subject_scores, metrics_by_repeat, site_order, roi_order)
    except ValidationError as e:
        print(f"ERROR: validación de entradas de Fase 2 falló, no se produjo ninguna tabla científica: {e}",
              file=sys.stderr)
        return 1

    # CORRECCIONES_V19 §9.1/§9.3: preflight completo sobre las rutas FINALES,
    # antes de calcular nada; luego se calcula y serializa todo en un
    # directorio de staging dentro de output_dir, y solo se promueve a las
    # rutas finales (os.replace) si absolutamente todo -- tablas, figuras y
    # el manifiesto -- terminó de generarse sin errores.
    out_files_final = {
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
        "secondary_contrast_18_vs_116_forest.svg": figures_dir / "secondary_contrast_18_vs_116_forest.svg",
        "secondary_contrast_18_vs_116_forest.png": figures_dir / "secondary_contrast_18_vs_116_forest.png",
        "secondary_contrast_39_vs_116_forest.svg": figures_dir / "secondary_contrast_39_vs_116_forest.svg",
        "secondary_contrast_39_vs_116_forest.png": figures_dir / "secondary_contrast_39_vs_116_forest.png",
        "contrasts_vs_116_forest.svg": figures_dir / "contrasts_vs_116_forest.svg",
        "contrasts_vs_116_forest.png": figures_dir / "contrasts_vs_116_forest.png",
        "analysis_manifest.json": output_dir / "analysis_manifest.json",
    }
    try:
        resolve_and_check_output_paths(list(out_files_final.values()), output_dir)
    except ValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not args.overwrite:
        existing = [str(p) for p in out_files_final.values() if p.exists()]
        if existing:
            print("ERROR: las siguientes salidas ya existen; use --overwrite para reemplazarlas:", file=sys.stderr)
            for p in existing:
                print(f"  - {p}", file=sys.stderr)
            return 1

    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(output_dir)))
    # Si el proceso termina (excepción no capturada, SystemExit, etc.) antes
    # de la promoción final, este directorio de staging queda huérfano; el
    # atexit lo limpia siempre, incluso si nunca se llega a la promoción.
    atexit.register(lambda: shutil.rmtree(staging_dir, ignore_errors=True))
    (staging_dir / "tables").mkdir(parents=True, exist_ok=True)
    (staging_dir / "data").mkdir(parents=True, exist_ok=True)
    (staging_dir / "figures").mkdir(parents=True, exist_ok=True)
    # A partir de aquí, todo el cuerpo de la función escribe en el staging
    # (out_files), nunca directamente en las rutas finales.
    out_files = {
        name: staging_dir / Path(final_path).relative_to(output_dir)
        for name, final_path in out_files_final.items()
    }

    # ---- 8.1 Estimación puntual -------------------------------------------------
    pts = point_estimates(metrics_by_repeat, site_order, roi_order)

    # ---- 8.2 Bootstrap por sitio --------------------------------------------------
    site_timings = {}
    bootstrap_by_site = {}
    for site in site_order:
        t0 = time.perf_counter()
        tensor, y_true, subject_ids = build_prob_tensor(subject_scores, site, roi_order)
        draws = bootstrap_site(tensor, y_true, roi_order, n_iter, seed, site=site)
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
        delta = compute_primary_delta(auc12, auc116)
        boot_delta = bootstrap_by_site[site][(12, "auc")] - bootstrap_by_site[site][(116, "auc")]
        lo, hi = bilateral_ci(boot_delta)
        se = float(boot_delta.std(ddof=1))
        n_subjects = int(pts[(pts["site"] == site) & (pts["roi_set"] == 12)]["n_subjects"].iloc[0])
        primary_rows.append({
            "site": site, "auc_12": auc12, "auc_116": auc116,
            "delta_auc": delta, "bilateral_ci_low": lo, "bilateral_ci_high": hi,
        })
        precision_rows.append(build_precision_diagnostics_row(site, n_subjects, delta, se, lo, hi))
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

    # ---- 10.4 Contrastes secundarios frente a 116 (18 y 39): figuras exploratorias --
    # Estas figuras NO forman parte del contraste principal (12 frente a 116, unico
    # preespecificado como primario en el plan 5.6, seccion 8). Visualizan dos de los
    # cinco contrastes secundarios ya calculados en secondary_pairwise_comparisons.csv
    # (18-116 y 39-116), con el mismo bootstrap pareado, sin declaraciones de
    # significancia y sin correccion por comparaciones multiples -- exactamente como
    # exige el plan para los contrastes secundarios. No se reescala ni se reinterpreta
    # el contraste principal.
    def _forest_series(df_subset: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = df_subset.set_index("site").loc[site_order]
        return d[value_col].to_numpy(), d["bilateral_ci_low"].to_numpy(), d["bilateral_ci_high"].to_numpy()

    def _draw_forest_panel(ax, est: np.ndarray, lo: np.ndarray, hi: np.ndarray, title: str) -> None:
        y_pos_local = np.arange(len(site_order))
        xerr = [np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)]
        ax.errorbar(est, y_pos_local, xerr=xerr, fmt="o", color="black", capsize=4)
        ax.axvline(0, color="lightgray", linestyle="--", linewidth=1)
        ax.set_yticks(y_pos_local)
        ax.set_yticklabels(site_order)
        ax.set_title(title)
        ax.set_xlabel("Diferencia AUC, IC bootstrap bilateral 95%")

    secondary_auc = secondary_pairwise[secondary_pairwise["metric"] == "auc"]

    individual_specs = [
        ("18-116", "Contraste 18 − 116 (secundaria, exploratoria)", "secondary_contrast_18_vs_116_forest"),
        ("39-116", "Contraste 39 − 116 (secundaria, exploratoria)", "secondary_contrast_39_vs_116_forest"),
    ]
    for contrast_label, title, fname_stub in individual_specs:
        sub = secondary_auc[secondary_auc["contrast"] == contrast_label]
        est, lo, hi = _forest_series(sub, "estimate")
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        _draw_forest_panel(ax_s, est, lo, hi, title)
        fig_s.tight_layout()
        fig_s.savefig(out_files[f"{fname_stub}.svg"])
        fig_s.savefig(out_files[f"{fname_stub}.png"], dpi=150)
        plt.close(fig_s)

    # Figura combinada: 12-116 (primaria) junto a 18-116 y 39-116 (secundarias),
    # en la misma escala horizontal para que sean directamente comparables.
    combo_specs = [
        ("12 − 116 (primaria)", primary_df.rename(columns={"delta_auc": "estimate"})),
        ("18 − 116 (secundaria, exploratoria)", secondary_auc[secondary_auc["contrast"] == "18-116"]),
        ("39 − 116 (secundaria, exploratoria)", secondary_auc[secondary_auc["contrast"] == "39-116"]),
    ]
    combo_data = []
    all_lo_combo, all_hi_combo = [], []
    for title, df_c in combo_specs:
        est, lo, hi = _forest_series(df_c, "estimate")
        combo_data.append((title, est, lo, hi))
        all_lo_combo.append(lo)
        all_hi_combo.append(hi)
    xmin = float(np.min(np.concatenate(all_lo_combo))) - 0.02
    xmax = float(np.max(np.concatenate(all_hi_combo))) + 0.02

    fig_c, axes_c = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax_c, (title, est, lo, hi) in zip(axes_c, combo_data):
        _draw_forest_panel(ax_c, est, lo, hi, title)
        ax_c.set_xlim(xmin, xmax)
    fig_c.suptitle("Contraste de cada tamaño de ROI frente a 116, por sitio (sin efecto combinado)")
    fig_c.tight_layout()
    fig_c.savefig(out_files["contrasts_vs_116_forest.svg"])
    fig_c.savefig(out_files["contrasts_vs_116_forest.png"], dpi=150)
    plt.close(fig_c)

    # ---- analysis_manifest.json ------------------------------------------------------
    # CORRECCIONES_V19 §11: estado de Git y inventario de hashes AL FINAL,
    # para compararlos contra las capturas "before" del inicio de main().
    git_commit_after, git_status_after, git_provenance_status_after = get_git_status(repo_root)
    # Si Git estuvo disponible al inicio pero no al final (o viceversa), no
    # hay una comparación de estado coherente posible; se registra tal cual.
    if git_provenance_status != git_provenance_status_after:
        git_provenance_status = "inconsistent_before_after"

    if git_status_before is None or git_status_after is None:
        # CORRECCIONES_V19 §11: sin Git no se inventan listas vacías ni un
        # commit ficticio; los campos quedan explícitamente en null.
        changed_paths_before = None
        changed_paths_after = None
        non_analysis_changes = None
    else:
        changed_paths_before = [line[3:] for line in git_status_before]
        changed_paths_after = [line[3:] for line in git_status_after]
        non_analysis_changes = [p for p in changed_paths_after if not p.startswith("analysis/")]

    input_hash_inventory_after = compute_input_hash_inventory(repo_root, manifest)
    input_hash_inventory_fingerprint_before = hash_inventory_fingerprint(input_hash_inventory_before)
    input_hash_inventory_fingerprint_after = hash_inventory_fingerprint(input_hash_inventory_after)
    results_inputs_unchanged = (
        input_hash_inventory_before["results_readme_sha256"] == input_hash_inventory_after["results_readme_sha256"]
        and input_hash_inventory_before["run_artifacts"] == input_hash_inventory_after["run_artifacts"]
    )
    protected_files_unchanged = (
        input_hash_inventory_before["protected_files"] == input_hash_inventory_after["protected_files"]
    )
    # CORRECCIONES_V19 §11: results_read_only se basa exclusivamente en la
    # comparación de hashes antes/después, nunca en si Git estaba disponible
    # ni en su salida (una lista de cambios vacía por ausencia de Git nunca
    # debe leerse como "sin cambios").
    results_read_only = results_inputs_unchanged

    import sklearn
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }

    # plan_path/plan_sha256_early ya se verificaron contra CANONICAL_PLAN_SHA256
    # al inicio de main(), antes de cualquier cómputo; se reutilizan aquí.
    plan_sha256 = plan_sha256_early

    run_hashes = build_run_hashes(manifest, repo_root)

    # CORRECCIONES_V19 §9.3: los hashes se calculan sobre los archivos de
    # staging (los únicos que existen en este punto), pero la ruta registrada
    # en el manifiesto es la ruta final prevista, no la de staging.
    output_hashes = {}
    for name, staged_path in out_files.items():
        if name == "analysis_manifest.json":
            continue
        output_hashes[name] = {
            "path": relative_to_repo(out_files_final[name], repo_root),
            "sha256": sha256_file(staged_path),
        }
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
        "git_provenance_status": git_provenance_status,
        "repo_commit_before": git_commit_before,
        "repo_commit_after": git_commit_after,
        "changed_paths_before": changed_paths_before,
        "changed_paths_after": changed_paths_after,
        "changed_paths_outside_analysis": non_analysis_changes,
        "input_hash_inventory_fingerprint_before": input_hash_inventory_fingerprint_before,
        "input_hash_inventory_fingerprint_after": input_hash_inventory_fingerprint_after,
        "input_hash_inventory_before": input_hash_inventory_before,
        "input_hash_inventory_after": input_hash_inventory_after,
        "protected_files_unchanged": protected_files_unchanged,
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
        "results_read_only": results_read_only,
        "results_read_only_method": (
            "comparacion de hashes SHA-256 de results/README.md y de los 7 artefactos "
            "oficiales de las 16 corridas, antes y despues de la ejecucion (no se infiere "
            "de la disponibilidad de Git; ver input_hash_inventory_fingerprint_before/after)"
        ),
        "limitation": (
            "Los intervalos bootstrap estan condicionados a las predicciones, entrenamientos y "
            "cinco particiones de validacion cruzada existentes; no capturan la variabilidad de "
            "un reentrenamiento o reparticion completos."
        ),
    }
    with open(out_files["analysis_manifest.json"], "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2, ensure_ascii=False, sort_keys=False)

    # CORRECCIONES_V19 §9.3: promover TODO (tablas, figuras y el manifiesto)
    # solo ahora que todo el cálculo y todas las serializaciones de staging
    # terminaron sin errores. analysis_manifest.json se promueve al final,
    # después de todas las demás salidas.
    non_manifest = [k for k in out_files if k != "analysis_manifest.json"]
    for name in non_manifest:
        out_files_final[name].parent.mkdir(parents=True, exist_ok=True)
        os.replace(out_files[name], out_files_final[name])
    out_files_final["analysis_manifest.json"].parent.mkdir(parents=True, exist_ok=True)
    os.replace(out_files["analysis_manifest.json"], out_files_final["analysis_manifest.json"])
    shutil.rmtree(staging_dir, ignore_errors=True)

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
