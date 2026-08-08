#!/usr/bin/env python3
"""Analyzer congelado de la campaña ``loso_static_v1`` (Secciones 49-57).

Este script se implementa y se prueba con fixtures sintéticas ANTES de ver
resultados reales (Sección 49). Después del freeze científico
(``CP4``/Sección 63), no se modifica su lógica salvo para corregir un bug
declarado explícitamente (Sección 66/CP7) — nunca en silencio.

Solo lee corridas formales ya almacenadas bajo ``results/loso/<run_id>/``; no
entrena, no selecciona hiperparámetros, no ejecuta GPU, no modifica esos
resultados (contrato de ``analysis/loso/README.md``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results" / "loso"
CONFIG_PATH = REPO_ROOT / "analysis" / "loso" / "config" / "loso_analysis_config.json"
OUTPUT_DIR = REPO_ROOT / "analysis" / "loso" / "outputs"

SITES = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]
ROI_SETS = ["12", "116"]
MODELS = ("brainnetcnn", "logreg")
BNN_SEEDS = [42, 43, 44, 45, 46]

EXPECTED_PREDICTIONS_TOTAL = 5580
EXPECTED_PREDICTIONS_BY_SITE = {"NYU": 2124, "Peking": 2196, "NeuroIMAGE": 468, "OHSU": 792}
EXPECTED_RUNS_TOTAL = 48
EXPECTED_METRICS_SUMMARY_ROWS = 16
EXPECTED_CONTRASTS_ROWS = 12


def _sha_file(path: Path, *, length: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _sha_bytes(data: bytes, *, length: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:length]


def load_analysis_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "campaign_id", "site_order", "roi_order", "models", "brainnet_seeds",
        "primary_metric", "bootstrap_iterations", "bootstrap_seed", "ci_level",
    }
    missing = required.difference(config)
    if missing:
        raise SystemExit(f"STOP: loso_analysis_config.json incompleto: faltan {sorted(missing)}.")
    return config


# ---------------------------------------------------------------------------
# Descubrimiento de corridas formales (Sección 42: ignora .staging y _design)
# ---------------------------------------------------------------------------


def discover_formal_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    dirs = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "_design" or child.name.startswith(".staging"):
            continue
        dirs.append(child)
    return dirs


def load_formal_run(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"STOP: {run_dir} no tiene config.json.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("formal", False):
        raise SystemExit(f"STOP: {run_dir} no está marcado formal=true.")
    predictions_path = run_dir / "predictions_test.csv"
    if not predictions_path.exists():
        raise SystemExit(f"STOP: {run_dir} no tiene predictions_test.csv.")
    predictions = pd.read_csv(predictions_path)
    metrics_test_path = run_dir / "metrics_test.csv"
    metrics_test = pd.read_csv(metrics_test_path) if metrics_test_path.exists() else pd.DataFrame()
    return {
        "run_dir": run_dir,
        "config": config,
        "predictions": predictions,
        "metrics_test": metrics_test,
    }


def discover_runs(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    return [load_formal_run(d) for d in discover_formal_run_dirs(results_root)]


# ---------------------------------------------------------------------------
# Manifest (Sección 56)
# ---------------------------------------------------------------------------


def _expected_condition_keys() -> list[tuple[str, str, str, int | None]]:
    keys: list[tuple[str, str, str, int | None]] = []
    for site in SITES:
        for roi in ROI_SETS:
            for seed in BNN_SEEDS:
                keys.append((site, roi, "brainnetcnn", seed))
            keys.append((site, roi, "logreg", None))
    return keys


def build_manifest(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    for run in runs:
        config = run["config"]
        key = (
            config["held_out_site"],
            str(config["roi_set"]),
            config["model"],
            config.get("model_seed"),
        )
        if key in by_key:
            raise SystemExit(f"STOP: manifest ambiguo — más de una corrida formal para {key}.")
        by_key[key] = run

    expected_keys = set(_expected_condition_keys())
    present_keys = set(by_key)
    missing = expected_keys - present_keys
    unexpected = present_keys - expected_keys
    if missing:
        raise SystemExit(f"STOP: faltan corridas formales para {sorted(missing)}.")
    if unexpected:
        raise SystemExit(f"STOP: corridas formales inesperadas (glob ambiguo): {sorted(unexpected)}.")
    if len(by_key) != EXPECTED_RUNS_TOTAL:
        raise SystemExit(f"STOP: manifest tiene {len(by_key)} corridas; se esperaban {EXPECTED_RUNS_TOTAL}.")

    entries = []
    for key, run in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or 0)):
        held_out_site, roi_set, model, seed = key
        config = run["config"]
        entries.append(
            {
                "held_out_site": held_out_site,
                "roi_set": roi_set,
                "model": model,
                "model_seed": seed,
                "run_id": config["run_id"],
                "identity_hash": config.get("identity_hash"),
                "config_sha256": _sha_file(run["run_dir"] / "config.json"),
                "predictions_sha256": _sha_file(run["run_dir"] / "predictions_test.csv"),
                "split_sha256": config.get("split_manifest_file_sha256"),
                "training_source_sha": config.get("training_source_git_sha"),
                "environment_signature": config.get("environment_signature"),
                "status": "formal",
            }
        )
    return {"campaign_id": "loso_static_v1", "n_runs": len(entries), "runs": entries}


# ---------------------------------------------------------------------------
# Predicciones largas (Secciones 54-55)
# ---------------------------------------------------------------------------


def build_predictions_long(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frames = []
    for run in runs:
        df = run["predictions"].copy()
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)

    if len(long_df) != EXPECTED_PREDICTIONS_TOTAL:
        raise SystemExit(
            f"STOP: loso_predictions_long tiene {len(long_df)} filas; se esperaban "
            f"{EXPECTED_PREDICTIONS_TOTAL}."
        )
    for site, expected_n in EXPECTED_PREDICTIONS_BY_SITE.items():
        n = int((long_df["held_out_site"] == site).sum())
        if n != expected_n:
            raise SystemExit(f"STOP: {site} tiene {n} filas en predictions_long; se esperaban {expected_n}.")
    if not ((long_df["y_prob"] >= 0) & (long_df["y_prob"] <= 1)).all():
        raise SystemExit("STOP: y_prob fuera de [0, 1] en predictions_long.")
    if not np.isfinite(long_df["y_prob"]).all():
        raise SystemExit("STOP: y_prob no finito en predictions_long.")
    return long_df


def build_metrics_by_run(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        config = run["config"]
        metrics_row = run["metrics_test"].iloc[0].to_dict() if len(run["metrics_test"]) else {}
        rows.append(
            {
                "held_out_site": config["held_out_site"],
                "roi_set": str(config["roi_set"]),
                "model": config["model"],
                "model_seed": config.get("model_seed"),
                "run_id": config["run_id"],
                **metrics_row,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_RUNS_TOTAL:
        raise SystemExit(f"STOP: metrics_by_run tiene {len(df)} filas; se esperaban {EXPECTED_RUNS_TOTAL}.")
    return df


# ---------------------------------------------------------------------------
# Bootstrap (Secciones 51-52) y contrastes (Sección 53)
# ---------------------------------------------------------------------------


def _canonical_site_order(predictions_long: pd.DataFrame, site: str) -> np.ndarray:
    """subject_key ascendente para el sitio, común a todas las condiciones."""

    keys = predictions_long.loc[predictions_long["held_out_site"] == site, "subject_key"].unique()
    return np.sort(keys)


def _condition_arrays(
    predictions_long: pd.DataFrame, *, site: str, roi_set: str, model: str,
    seed: int | None, subject_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        (predictions_long["held_out_site"] == site)
        & (predictions_long["roi_set"].astype(str) == str(roi_set))
        & (predictions_long["model"] == model)
    )
    if model == "brainnetcnn":
        mask &= predictions_long["model_seed"] == seed
    sub = predictions_long.loc[mask].set_index("subject_key").reindex(subject_order)
    if sub["y_true"].isna().any():
        raise SystemExit(f"STOP: faltan predicciones para {site}/{roi_set}/{model}/seed={seed}.")
    return sub["y_true"].to_numpy(dtype=np.int64), sub["y_prob"].to_numpy(dtype=np.float64)


def _assert_y_true_consistent(predictions_long: pd.DataFrame, site: str, subject_order: np.ndarray) -> np.ndarray:
    """Sección 51 punto 2: y_true idéntico entre condiciones/seeds para el sitio."""

    sub = predictions_long.loc[predictions_long["held_out_site"] == site]
    pivot = sub.pivot_table(index="subject_key", values="y_true", aggfunc="nunique")
    if (pivot["y_true"] != 1).any():
        raise SystemExit(f"STOP: y_true inconsistente entre condiciones para {site}.")
    y_true = sub.drop_duplicates("subject_key").set_index("subject_key").reindex(subject_order)["y_true"]
    return y_true.to_numpy(dtype=np.int64)


def compute_site_bootstrap(
    predictions_long: pd.DataFrame, site: str, analysis_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Un conjunto de resamples por sitio, reutilizado por TODAS las condiciones
    y contrastes de ese sitio (Sección 51 punto 8)."""

    subject_order = _canonical_site_order(predictions_long, site)
    y_true = _assert_y_true_consistent(predictions_long, site, subject_order)

    control_pos = np.flatnonzero(y_true == 0)
    adhd_pos = np.flatnonzero(y_true == 1)
    n_iter = int(analysis_config["bootstrap_iterations"])
    rng = np.random.Generator(np.random.PCG64(int(analysis_config["bootstrap_seed"])))

    draws = np.empty((n_iter, len(subject_order)), dtype=np.int64)
    for i in range(n_iter):
        resampled_control = rng.choice(control_pos, size=control_pos.size, replace=True)
        resampled_adhd = rng.choice(adhd_pos, size=adhd_pos.size, replace=True)
        draws[i] = np.concatenate([resampled_control, resampled_adhd])

    condition_probs: dict[tuple[str, str, int | None], np.ndarray] = {}
    for roi_set in ROI_SETS:
        for seed in BNN_SEEDS:
            _, y_prob = _condition_arrays(
                predictions_long, site=site, roi_set=roi_set, model="brainnetcnn",
                seed=seed, subject_order=subject_order,
            )
            condition_probs[(roi_set, "brainnetcnn", seed)] = y_prob
        _, y_prob = _condition_arrays(
            predictions_long, site=site, roi_set=roi_set, model="logreg",
            seed=None, subject_order=subject_order,
        )
        condition_probs[(roi_set, "logreg", None)] = y_prob

    return {"subject_order": subject_order, "y_true": y_true, "draws": draws, "condition_probs": condition_probs}


def _auc_or_nan(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def condition_point_estimate(
    site_ctx: Mapping[str, Any], *, roi_set: str, model: str,
) -> tuple[float, dict[str, float] | None]:
    """Punto estimado sobre los datos observados (NO sobre el bootstrap)."""

    y_true = site_ctx["y_true"]
    if model == "brainnetcnn":
        seed_aucs = {
            seed: _auc_or_nan(y_true, site_ctx["condition_probs"][(roi_set, "brainnetcnn", seed)])
            for seed in BNN_SEEDS
        }
        point = float(np.mean(list(seed_aucs.values())))
        dispersion = {
            "seed_sd": float(np.std(list(seed_aucs.values()), ddof=1)),
            "seed_min": float(np.min(list(seed_aucs.values()))),
            "seed_max": float(np.max(list(seed_aucs.values()))),
        }
        return point, dispersion
    y_prob = site_ctx["condition_probs"][(roi_set, "logreg", None)]
    return _auc_or_nan(y_true, y_prob), None


def condition_bootstrap_replicates(site_ctx: Mapping[str, Any], *, roi_set: str, model: str) -> np.ndarray:
    """Sección 51.1/51.2: metric-then-mean por seed para BNN; una métrica para logistic."""

    draws = site_ctx["draws"]
    y_true_full = site_ctx["y_true"]
    n_iter = draws.shape[0]

    if model == "brainnetcnn":
        per_seed = np.empty((len(BNN_SEEDS), n_iter), dtype=np.float64)
        for s, seed in enumerate(BNN_SEEDS):
            y_prob_full = site_ctx["condition_probs"][(roi_set, "brainnetcnn", seed)]
            for i in range(n_iter):
                idx = draws[i]
                per_seed[s, i] = _auc_or_nan(y_true_full[idx], y_prob_full[idx])
        return per_seed.mean(axis=0)

    y_prob_full = site_ctx["condition_probs"][(roi_set, "logreg", None)]
    replicates = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = draws[i]
        replicates[i] = _auc_or_nan(y_true_full[idx], y_prob_full[idx])
    return replicates


def percentile_ci(replicates: np.ndarray, *, ci_level: float) -> tuple[float, float]:
    alpha = (1.0 - ci_level) * 100 / 2.0
    lo, hi = np.percentile(replicates, [alpha, 100 - alpha], method="linear")
    return float(lo), float(hi)


def build_metrics_summary(
    predictions_long: pd.DataFrame, analysis_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    ci_level = float(analysis_config["ci_level"])
    rows = []
    site_contexts: dict[str, dict[str, Any]] = {}
    replicate_cache: dict[tuple[str, str, str], np.ndarray] = {}

    for site in SITES:
        ctx = compute_site_bootstrap(predictions_long, site, analysis_config)
        site_contexts[site] = ctx
        for roi_set in ROI_SETS:
            for model in MODELS:
                point, dispersion = condition_point_estimate(ctx, roi_set=roi_set, model=model)
                replicates = condition_bootstrap_replicates(ctx, roi_set=roi_set, model=model)
                replicate_cache[(site, roi_set, model)] = replicates
                lo, hi = percentile_ci(replicates, ci_level=ci_level)
                row = {
                    "held_out_site": site,
                    "roi_set": roi_set,
                    "model": model,
                    "auc_point": point,
                    "auc_ci_low": lo,
                    "auc_ci_high": hi,
                }
                if dispersion is not None:
                    row.update(dispersion)
                rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_METRICS_SUMMARY_ROWS:
        raise SystemExit(f"STOP: metrics_summary tiene {len(df)} filas; se esperaban {EXPECTED_METRICS_SUMMARY_ROWS}.")
    return df, {"contexts": site_contexts, "replicates": replicate_cache}


CONTRAST_SPECS = (
    ("dimensionality", "brainnetcnn", "116", "brainnetcnn", "12"),
    ("model_family_at_12", "logreg", "12", "brainnetcnn", "12"),
    ("model_family_at_116", "logreg", "116", "brainnetcnn", "116"),
)


def build_contrasts(
    predictions_long: pd.DataFrame, analysis_config: Mapping[str, Any], bootstrap_state: Mapping[str, Any],
) -> pd.DataFrame:
    ci_level = float(analysis_config["ci_level"])
    contexts = bootstrap_state["contexts"]
    replicates = bootstrap_state["replicates"]

    rows = []
    for name, model_a, roi_a, model_b, roi_b in CONTRAST_SPECS:
        for site in SITES:
            ctx = contexts[site]
            point_a, _ = condition_point_estimate(ctx, roi_set=roi_a, model=model_a)
            point_b, _ = condition_point_estimate(ctx, roi_set=roi_b, model=model_b)
            rep_a = replicates[(site, roi_a, model_a)]
            rep_b = replicates[(site, roi_b, model_b)]
            diff_replicates = rep_a - rep_b  # mismo draw por construcción (Sección 51 p.8)
            lo, hi = percentile_ci(diff_replicates, ci_level=ci_level)
            rows.append(
                {
                    "contrast": name,
                    "held_out_site": site,
                    "condition_a": f"{model_a}_{roi_a}",
                    "condition_b": f"{model_b}_{roi_b}",
                    "delta_point": point_a - point_b,
                    "delta_ci_low": lo,
                    "delta_ci_high": hi,
                }
            )
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_CONTRASTS_ROWS:
        raise SystemExit(f"STOP: contrasts tiene {len(df)} filas; se esperaban {EXPECTED_CONTRASTS_ROWS}.")
    return df


def build_bootstrap_manifest(
    analysis_config: Mapping[str, Any], predictions_long_hash: str, summary_hash: str,
) -> dict[str, Any]:
    import sklearn

    return {
        "analysis_config_sha256": _sha_bytes(
            json.dumps(analysis_config, sort_keys=True).encode("utf-8")
        ),
        "analysis_source_git_sha": None,  # se completa en main() con git_info()
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "bootstrap_seed": analysis_config["bootstrap_seed"],
        "rng": analysis_config["bootstrap_rng"],
        "iterations": analysis_config["bootstrap_iterations"],
        "percentile_method": "linear",
        "subject_ordering": analysis_config["bootstrap_subject_order"],
        "input_predictions_sha256": predictions_long_hash,
        "output_summary_sha256": summary_hash,
    }


# ---------------------------------------------------------------------------
# Reporte y main()
# ---------------------------------------------------------------------------


def build_report(
    *, manifest: Mapping[str, Any], metrics_summary: pd.DataFrame, contrasts: pd.DataFrame,
) -> str:
    lines = [
        "# LOSO_STATIC_V1_REPORT",
        "",
        "Campaña `loso_static_v1`: 4 sitios held-out x 2 ROI sets x 2 familias de "
        "modelo, conectividad estática, sin harmonización ni ponderación de "
        "clase/sitio. Ver `PLAN_FINAL_LOSO_STATIC_V1_IA_REVISADO.md` para el diseño "
        "completo.",
        "",
        f"## Completitud: {manifest['n_runs']}/48 corridas formales",
        "",
        "## AUC por condición (95% CI, percentil, sin ajustar)",
        "",
        metrics_summary.to_markdown(index=False),
        "",
        "## Contrastes preespecificados (95% CI)",
        "",
        contrasts.to_markdown(index=False),
        "",
        "## Terminología",
        "",
        "cross-site transport / LOSO cross-site evaluation / internal-external "
        "validation-style analysis / held-out site / transportability. Evitar sin "
        "matiz: external validation, independent validation, generalizes across "
        "sites, clinically validated, robust biomarker.",
        "",
        "## Caveat NYU",
        "",
        "La configuración de BrainNetCNN se desarrolló/fijó históricamente usando "
        "NYU antes de la evaluación multisitio. La rotación con NYU held-out es "
        "'development-site held-out re-evaluation within the LOSO campaign', no "
        "una evaluación en un sitio totalmente ajeno al desarrollo del modelo.",
        "",
        "## No harmonización / no ponderación",
        "",
        "class_weight=false, site_weighting=false, sample_weight=false, "
        "harmonization=none en las 48 corridas formales.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    analysis_config = load_analysis_config()
    runs = discover_runs()
    manifest = build_manifest(runs)
    predictions_long = build_predictions_long(runs)
    metrics_by_run = build_metrics_by_run(runs)
    metrics_summary, bootstrap_state = build_metrics_summary(predictions_long, analysis_config)
    contrasts = build_contrasts(predictions_long, analysis_config, bootstrap_state)

    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(OUTPUT_DIR)))
    try:
        (staging_dir / "loso_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        predictions_long.to_csv(staging_dir / "loso_predictions_long.csv", index=False)
        metrics_by_run.to_csv(staging_dir / "loso_metrics_by_run.csv", index=False)
        metrics_summary.to_csv(staging_dir / "loso_metrics_summary.csv", index=False)
        contrasts.to_csv(staging_dir / "loso_contrasts.csv", index=False)

        predictions_hash = _sha_file(staging_dir / "loso_predictions_long.csv")
        summary_hash = _sha_file(staging_dir / "loso_metrics_summary.csv")
        bootstrap_manifest = build_bootstrap_manifest(analysis_config, predictions_hash, summary_hash)
        try:
            sys.path.insert(0, str(REPO_ROOT / "src"))
            from run_experiment import git_info

            bootstrap_manifest["analysis_source_git_sha"] = git_info()["commit"]
        except Exception:
            pass
        (staging_dir / "loso_bootstrap_manifest.json").write_text(
            json.dumps(bootstrap_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        report = build_report(manifest=manifest, metrics_summary=metrics_summary, contrasts=contrasts)
        (staging_dir / "LOSO_STATIC_V1_REPORT.md").write_text(report, encoding="utf-8")

        for name in (
            "loso_manifest.json", "loso_predictions_long.csv", "loso_metrics_by_run.csv",
            "loso_metrics_summary.csv", "loso_contrasts.csv", "loso_bootstrap_manifest.json",
            "LOSO_STATIC_V1_REPORT.md",
        ):
            final_path = OUTPUT_DIR / name
            staged_path = staging_dir / name
            if final_path.exists():
                final_path.unlink()
            Path(staged_path).replace(final_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Analysis outputs escritos en {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
