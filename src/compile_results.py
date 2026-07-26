#!/usr/bin/env python3
"""Compila corridas de ``results/runs`` del proyecto TDHA.

Compatible con configuraciones históricas y con el esquema v4 de
``run_experiment.py``: representación estática/dinámica, ventanas en TR o
segundos, solapamiento, Fisher z, ventanas gaussianas, y el monitor de
early stopping (``val_loss``/``val_bce``; ausente en v1/v2 se interpreta como
``val_loss`` con ``min_delta=1e-5``). Los esquemas 1-3 calculaban ``best_epoch``
como el mínimo global de la serie monitoreada (``np.argmin``), que no siempre
coincide con la época cuyos pesos restauró ``EarlyStopping`` cuando
``min_delta > 0`` o ``start_from_epoch > 0`` — sus métricas externas siguen
siendo válidas (se calcularon sobre los pesos ya restaurados por Keras), pero
``best_epoch``/``best_monitor_value`` de esas corridas son un metadato
aproximado, no una reconstrucción exacta, y no participan en la comparación
A/B formal por ``early_stopping_monitor`` (requiere esquema 4 y
``early_stopping_ab_hash``).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "results" / "runs"
COUNT_COLUMNS = {"true_positives", "true_negatives", "false_positives", "false_negatives"}
META_COLUMNS = {
    "fold", "repeat", "n_epochs", "best_epoch", "n_fit", "n_inner_val",
    "n_outer_val", "class_weight_0", "class_weight_1", "best_monitor_value",
    "restored_monitor_value",
}
PREFERRED_METRICS = [
    "loss", "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "f1", "f1_macro", "auc",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se pudo leer {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path} no contiene un objeto JSON.")
    return obj


def _windowing(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("windowing")
    return value if isinstance(value, dict) else {}


def _diagnostics(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("windowing_diagnostics")
    return value if isinstance(value, dict) else {}


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _suffix_from_metrics(path: Path) -> str:
    return path.stem.removeprefix("metrics_val")


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=[np.number]).columns
    discovered = [c for c in numeric if c not in META_COLUMNS and c not in COUNT_COLUMNS]
    ordered = [c for c in PREFERRED_METRICS if c in discovered]
    ordered.extend(c for c in discovered if c not in ordered)
    return ordered


def _safe_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if series.notna().sum() > 1 else float("nan")


def _pct(value: float, metric: str) -> float:
    return value if metric == "loss" else value * 100.0


# Métricas OOF que no son pérdidas se expresan en porcentaje, igual que el resto de
# la tabla; log_loss y brier se dejan en su escala natural.
OOF_METRICS = ("auc", "f1_macro", "balanced_accuracy", "log_loss", "brier")
OOF_LOSS_METRICS = frozenset({"log_loss", "brier"})


def oof_metrics_per_repetition(predictions: pd.DataFrame) -> pd.DataFrame | None:
    """Métricas out-of-fold agrupando las predicciones de cada repetición.

    En un ``RepeatedStratifiedKFold`` cada repetición cubre a todos los sujetos
    exactamente una vez: cada sujeto aparece en la validación externa de un único
    pliegue dentro de la repetición. Agrupar los pliegues de una repetición reconstruye
    una predicción out-of-fold para toda la muestra, y repetir el proceso por cada
    repetición da varias estimaciones sobre las que calcular media y dispersión.

    Esto es más estable y menos sesgado que promediar métricas calculadas pliegue a
    pliegue: el AUC o el F1 de un pliegue con ~18 sujetos es muy ruidoso, mientras que
    el de la muestra completa (agrupada) tiene mucha menos varianza de muestreo.

    Devuelve una fila por repetición con AUC, F1 macro, exactitud balanceada, log-loss
    y Brier, o ``None`` si el archivo de predicciones no tiene el formato esperado.
    """

    needed = {"repeat", "y_true", "y_prob"}
    if predictions is None or predictions.empty or not needed.issubset(predictions.columns):
        return None
    from sklearn.metrics import (
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    rows: list[dict[str, Any]] = []
    for repeat, group in predictions.groupby("repeat"):
        y = pd.to_numeric(group["y_true"], errors="coerce").to_numpy()
        p = pd.to_numeric(group["y_prob"], errors="coerce").to_numpy()
        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask].astype(int)
        p = p[mask].astype(float)
        if y.size == 0:
            continue
        pred = (p >= 0.5).astype(int)
        both_classes = np.unique(y).size > 1
        rows.append({
            "repeat": int(repeat),
            "n": int(y.size),
            "auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
        })
    return pd.DataFrame(rows) if rows else None


def summarize(run_dir: Path, cfg: dict[str, Any], suffix: str = "") -> dict[str, Any] | None:
    val_path = run_dir / f"metrics_val{suffix}.csv"
    train_path = run_dir / f"metrics_train{suffix}.csv"
    if not val_path.exists() or not train_path.exists():
        return None

    val = pd.read_csv(val_path)
    train = pd.read_csv(train_path)
    if val.empty or train.empty:
        return None

    w = _windowing(cfg)
    d = _diagnostics(cfg)
    git = cfg.get("git") if isinstance(cfg.get("git"), dict) else {}
    representation = cfg.get("representation") or ("static" if cfg.get("window") is None else "ordered")
    mode = "static" if representation in ("static", "partial", "shrunk", "tangent") else "dynamic"

    row: dict[str, Any] = {
        "run_id": str(cfg.get("run_id", run_dir.name)) + suffix,
        "base_run_id": cfg.get("run_id", run_dir.name),
        "subset_suffix": suffix or None,
        "config_schema_version": cfg.get("config_schema_version", 1),
        "config_hash": cfg.get("config_hash"),
        "site": cfg.get("site"),
        "roi_set": cfg.get("roi_set"),
        "n_subjects": cfg.get("n_subjects"),
        "n_rois": cfg.get("n_rois"),
        "n_timepoints": cfg.get("n_timepoints"),
        "n_windows": cfg.get("n_windows", d.get("n_windows")),
        "model": cfg.get("model"),
        "seed": cfg.get("seed"),
        "n_splits": cfg.get("n_splits"),
        "n_repeats": cfg.get("n_repeats"),
        "n_folds": int(len(val)),
        "split_fingerprint": cfg.get("split_fingerprint"),
        "representation": representation,
        "representation_seed": cfg.get("representation_seed"),
        "connectivity_mode": mode,
        "windowing_preset": cfg.get("windowing_preset"),
        "window_tr": _first(w.get("window_tr"), cfg.get("window")),
        "step_tr": _first(w.get("step_tr"), cfg.get("step")),
        "window_seconds": w.get("window_seconds"),
        "step_seconds": w.get("step_seconds"),
        "requested_window_seconds": w.get("requested_window_seconds"),
        "requested_step_seconds": w.get("requested_step_seconds"),
        "requested_overlap": w.get("requested_overlap"),
        "effective_overlap": _first(w.get("effective_overlap"), d.get("effective_overlap")),
        "window_shape": _first(w.get("shape"), cfg.get("window_shape"), "rectangular"),
        "gaussian_sigma": _first(w.get("gaussian_sigma"), cfg.get("gaussian_sigma")),
        "fisher_z": bool(cfg.get("fisher_z", w.get("fisher_z", False))),
        "constant_policy": cfg.get("constant_policy", "zero"),
        "random_subset": cfg.get("random_subset"),
        "start_from_epoch": cfg.get("start_from_epoch"),
        # Ausente en corridas de esquema 1/2: se interpretan como el
        # comportamiento que tenía el callback antes de que este campo
        # existiera (monitor fijo en val_loss, min_delta fijo en 1e-5).
        "early_stopping_monitor": cfg.get("early_stopping_monitor", "val_loss"),
        "early_stopping_min_delta": cfg.get("early_stopping_min_delta", 1e-5),
        # None en esquemas < 4: esas corridas no pueden participar en el A/B
        # formal por monitor (ver check_comparability_ab() / --stats-by).
        "early_stopping_ab_hash": cfg.get("early_stopping_ab_hash"),
        "bold_hash": cfg.get("bold_hash"),
        "atlas_hash": cfg.get("atlas_hash"),
        "roi_indices_hash": cfg.get("roi_indices_hash"),
        "data_code_hash": cfg.get("data_code_hash"),
        "runner_code_hash": cfg.get("runner_code_hash"),
        "usuario": git.get("user"),
        "commit": (git.get("commit") or "")[:8],
        "arbol_limpio": git.get("clean"),
        "timestamp": cfg.get("timestamp"),
        "methodological_warnings": " | ".join(cfg.get("methodological_warnings", []) or []),
    }

    metrics = sorted(set(_metric_columns(val)).intersection(_metric_columns(train)),
                     key=lambda x: PREFERRED_METRICS.index(x) if x in PREFERRED_METRICS else 999)
    for metric in metrics:
        va = pd.to_numeric(val[metric], errors="coerce")
        tr = pd.to_numeric(train[metric], errors="coerce")
        row[f"train_{metric}_mean"] = _pct(float(tr.mean()), metric)
        row[f"val_{metric}_mean"] = _pct(float(va.mean()), metric)
        row[f"val_{metric}_sd"] = _pct(_safe_std(va), metric)
        row[f"val_{metric}_median"] = _pct(float(va.median()), metric)
        row[f"val_{metric}_min"] = _pct(float(va.min()), metric)
        row[f"val_{metric}_max"] = _pct(float(va.max()), metric)

    # Alias históricos.
    for metric in metrics:
        row[f"train_{metric}"] = row[f"train_{metric}_mean"]
        row[f"val_{metric}"] = row[f"val_{metric}_mean"]
    if "accuracy" in metrics:
        row["gap_acc"] = row["train_accuracy_mean"] - row["val_accuracy_mean"]
    row["epoca_media"] = float(val["best_epoch"].mean()) if "best_epoch" in val else np.nan
    row["epoca_sd"] = _safe_std(pd.to_numeric(val["best_epoch"], errors="coerce")) if "best_epoch" in val else np.nan

    # Métricas OOF por repetición (revisión 3.3): estimación agrupada, menos ruidosa
    # que promediar métricas pliegue a pliegue. Aditivo: si no hay predicciones OOF la
    # fila conserva exactamente las mismas columnas de antes.
    pred_path = run_dir / f"predictions_val{suffix}.csv"
    if pred_path.exists():
        try:
            predictions = pd.read_csv(pred_path)
        except (OSError, pd.errors.ParserError):
            predictions = None
        oof = oof_metrics_per_repetition(predictions)
        if oof is not None and not oof.empty:
            row["oof_n_repeats"] = int(len(oof))
            for metric in OOF_METRICS:
                series = pd.to_numeric(oof[metric], errors="coerce")
                scale_as = "loss" if metric in OOF_LOSS_METRICS else metric
                row[f"oof_{metric}_mean"] = _pct(float(series.mean()), scale_as)
                row[f"oof_{metric}_sd"] = _pct(_safe_std(series), scale_as)
    return row


def _validate_schema4_artifacts(run_dir: Path, suffix: str) -> list[str]:
    """Comprueba, para corridas de ``config_schema_version >= 4``, que los
    artefactos que sustentan el gate no circular de ``restored_monitor_value``
    existan y sean coherentes (ver ``methodology.md``, 'Early-stopping
    monitor'). No es retroactivo: nunca se invoca para esquemas 1-3, que no
    produjeron estos artefactos.
    """
    problems: list[str] = []
    paths = {
        "metrics_train": run_dir / f"metrics_train{suffix}.csv",
        "metrics_val": run_dir / f"metrics_val{suffix}.csv",
        "history": run_dir / f"history{suffix}.csv",
        "predictions_val": run_dir / f"predictions_val{suffix}.csv",
        "folds": run_dir / f"folds{suffix}.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            problems.append(f"falta {path.name}")
            continue
        try:
            frames[name] = pd.read_csv(path)
        except Exception as exc:
            problems.append(f"{path.name}: no se pudo leer ({type(exc).__name__}: {exc})")
    if problems:
        return problems

    # Columnas exigidas por archivo, y de esas, cuáles deben además ser
    # finitas. No se exige (ni se audita finitud en) el resto de columnas:
    # varias son opcionales por diseño (p. ej. class_weight_0/1 solo se
    # rellenan con --class-weight) y NaN ahí es legítimo, no un fallo. Los
    # identificadores estructurales (fold/repeat/subject_id/split/epoch) se
    # exigen en los cinco artefactos, no solo en metrics_train/val/history:
    # sin ellos, predictions_val.csv y folds.csv podían faltar por completo
    # sus columnas y aun así "pasar" la validación.
    metrics_common = {
        "fold", "repeat", "n_epochs", "best_epoch",
        "early_stopping_monitor", "best_monitor_value", "restored_monitor_value",
    }
    required_by_file = {
        "metrics_train": metrics_common,
        "metrics_val": metrics_common,
        "history": {"fold", "repeat", "epoch", "loss", "inner_val_loss", "bce", "inner_val_bce"},
        "predictions_val": {"fold", "repeat", "subject_id", "y_true", "y_prob"},
        "folds": {"fold", "repeat", "subject_id", "split"},
    }
    # subject_id, split y early_stopping_monitor son texto y no se convierten
    # a número; el resto de columnas obligatorias sí debe ser finito.
    finite_by_file = {
        "metrics_train": {"fold", "repeat", "n_epochs", "best_epoch", "best_monitor_value", "restored_monitor_value"},
        "metrics_val": {"fold", "repeat", "n_epochs", "best_epoch", "best_monitor_value", "restored_monitor_value"},
        "history": {"fold", "repeat", "epoch", "loss", "inner_val_loss", "bce", "inner_val_bce"},
        "predictions_val": {"fold", "repeat", "y_true", "y_prob"},
        "folds": {"fold", "repeat"},
    }
    for name, required in required_by_file.items():
        missing = required - set(frames[name].columns)
        if missing:
            problems.append(f"{name}{suffix}.csv: faltan columnas {sorted(missing)}")

    for name, frame in frames.items():
        if frame.empty:
            problems.append(f"{name}{suffix}.csv está vacío")

    for name, columns in finite_by_file.items():
        frame = frames.get(name)
        if frame is None or frame.empty:
            continue
        present = [c for c in columns if c in frame.columns]
        if not present:
            continue
        # frame[present].stack() descartaría los NaN por defecto (dropna=True)
        # antes de que np.isfinite() los viera — así es como un
        # best_monitor_value=NaN pasaba inadvertido. to_numeric + coerce
        # conserva cada posición (incluida una celda vacía o con texto no
        # numérico, que también se vuelve NaN) para que isfinite() la audite.
        numeric = frame[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if numeric.size == 0 or not np.isfinite(numeric).all():
            problems.append(f"{name}{suffix}.csv: valores no finitos o no numéricos en {sorted(present)}")

    return problems


def collect(root: str | Path, *, strict: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    notices: list[str] = []
    seen_schema_notice: set[str] = set()
    root = Path(root)
    for cfg_path in sorted(root.glob("*/config.json")):
        try:
            cfg = _read_json(cfg_path)
            run_dir = cfg_path.parent
            val_files = sorted(run_dir.glob("metrics_val*.csv"))
            if not val_files:
                errors.append(f"{run_dir.name}: corrida incompleta, sin metrics_val*.csv")
                continue
            schema = cfg.get("config_schema_version", 1)
            for path in val_files:
                suffix = _suffix_from_metrics(path)
                train_path = run_dir / f"metrics_train{suffix}.csv"
                if not train_path.exists():
                    errors.append(
                        f"{run_dir.name}{suffix}: corrida incompleta, falta {train_path.name}"
                    )
                    continue
                row = summarize(run_dir, cfg, suffix)
                if row is None:
                    # summarize() solo devuelve None si metrics_train/metrics_val
                    # están vacíos (ya comprobamos que existen arriba) — antes esto
                    # se descartaba en silencio, incluso bajo strict=True.
                    errors.append(
                        f"{run_dir.name}{suffix}: metrics_train{suffix}.csv o "
                        f"metrics_val{suffix}.csv están vacíos; summarize() no pudo "
                        "compilar la corrida"
                    )
                    continue
                if schema >= 4:
                    problems = _validate_schema4_artifacts(run_dir, suffix)
                    if problems:
                        errors.append(f"{run_dir.name}{suffix}: " + "; ".join(problems))
                        continue
                elif run_dir.name not in seen_schema_notice:
                    seen_schema_notice.add(run_dir.name)
                    notices.append(
                        f"{run_dir.name}: esquema {schema}, compila de forma descriptiva; "
                        "best_epoch/best_monitor_value pueden no coincidir exactamente con "
                        "lo que EarlyStopping restauró y quedan fuera de la comparación A/B "
                        "formal por early_stopping_monitor (ver docs/limitations.md)."
                    )
                rows.append(row)
        except Exception as exc:  # informa el archivo defectuoso sin perder las demás corridas
            errors.append(f"{cfg_path}: {exc}")
    if strict and errors:
        raise ValueError("Errores durante la compilación:\n- " + "\n- ".join(errors))
    frame = pd.DataFrame(rows)
    frame.attrs["collection_warnings"] = errors
    frame.attrs["collection_notices"] = notices
    return frame


def check_comparability(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if df.empty:
        return problems

    for site, group in df.groupby("site", dropna=False):
        hashes = set(group["bold_hash"].dropna()) if "bold_hash" in group else set()
        if len(hashes) > 1:
            problems.append(f"las señales BOLD de {site} difieren entre corridas ({len(hashes)} hashes)")

    hard = [
        ("seed", "semilla"), ("n_splits", "n_splits"), ("n_repeats", "n_repeats"),
        ("split_fingerprint", "particiones externas/internas"),
    ]
    for column, label in hard:
        if column in df and df[column].dropna().nunique() > 1:
            problems.append(f"{label} distintos; la comparación deja de ser estrictamente pareada")

    if "arbol_limpio" in df and df["arbol_limpio"].eq(False).any():
        dirty = df.loc[df["arbol_limpio"].eq(False), "run_id"].astype(str).tolist()
        problems.append(f"corridas realizadas con cambios sin confirmar: {dirty}")
    if df.duplicated(subset=[c for c in ["config_hash", "subset_suffix"] if c in df], keep=False).any():
        problems.append("se detectaron configuraciones duplicadas")
    return problems


def _check_early_stopping_ab(base: pd.DataFrame) -> None:
    """Guardas específicas de ``--stats-by early_stopping_monitor``.

    ``fixed`` en ``main()`` ya iguala site/model/roi_set/representation/seed/
    n_splits/n_repeats, y ``split_fingerprint`` se verifica aparte — pero eso
    deja sin fijar la ventana, los hiperparámetros de arquitectura, lr, batch,
    epochs, patience, min_delta, class_weight, fisher_z, o los hashes de datos/
    código: dos corridas podrían diferir en cualquiera de esos ejes y aun así
    pasar ese filtro. ``early_stopping_ab_hash`` (identity completa del
    esquema 4 menos ``early_stopping_monitor``) es la única garantía de que
    "idénticas salvo el monitor" es realmente cierto, no una lista de columnas
    que puede quedarse corta.
    """

    schema = pd.to_numeric(base.get("config_schema_version"), errors="coerce")
    if schema.isna().any() or (schema < 4).any():
        offending = base.loc[schema.isna() | (schema < 4), "base_run_id"].tolist()
        raise SystemExit(
            "La comparación A/B formal por early_stopping_monitor exige "
            "config_schema_version >= 4 en todas las corridas (best_epoch de "
            f"esquemas anteriores no reconstruye lo que restauró EarlyStopping): {offending}"
        )

    if "early_stopping_ab_hash" not in base or base["early_stopping_ab_hash"].isna().any():
        missing = (
            base.loc[base["early_stopping_ab_hash"].isna(), "base_run_id"].tolist()
            if "early_stopping_ab_hash" in base else base["base_run_id"].tolist()
        )
        raise SystemExit(
            f"Falta early_stopping_ab_hash en: {missing}. No se puede confirmar que las "
            "corridas sean idénticas salvo el monitor."
        )
    ab_hashes = set(base["early_stopping_ab_hash"].dropna())
    if len(ab_hashes) != 1:
        raise SystemExit(
            f"Las corridas seleccionadas tienen {len(ab_hashes)} valores distintos de "
            "early_stopping_ab_hash: no son idénticas salvo el monitor de early stopping "
            "(difieren en ventana, arquitectura, hiperparámetros, min_delta, o hashes de "
            "datos/código). Ajuste los filtros hasta dejar solo el par comparable."
        )

    monitors = set(base["early_stopping_monitor"].dropna())
    if monitors != {"val_loss", "val_bce"}:
        raise SystemExit(
            "La comparación A/B formal requiere exactamente los monitores "
            f"'val_loss' y 'val_bce'; se encontraron: {sorted(monitors)}."
        )


def methodological_group_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "site", "roi_set", "model", "representation", "connectivity_mode",
        "window_tr", "step_tr", "window_seconds", "step_seconds",
        "effective_overlap", "window_shape", "gaussian_sigma", "fisher_z",
        "constant_policy", "early_stopping_monitor", "early_stopping_min_delta",
    ]
    return [c for c in candidates if c in df.columns]


def aggregate_table(df: pd.DataFrame, group_by: Iterable[str] | None = None) -> pd.DataFrame:
    groups = list(group_by or methodological_group_columns(df))
    metric_cols = [
        c for c in df.columns
        if c.endswith("_mean") and (c.startswith("val_") or c.startswith("oof_"))
    ]
    if not groups or not metric_cols:
        return pd.DataFrame()
    aggregations: dict[str, list[str]] = {c: ["mean", "std", "min", "max", "median"] for c in metric_cols}
    result = df.groupby(groups, dropna=False).agg(aggregations)
    result.columns = [f"{a}_{b}" for a, b in result.columns]
    result = result.reset_index()
    counts = df.groupby(groups, dropna=False).size().rename("n_runs").reset_index()
    return counts.merge(result, on=groups, how="left")


def corrected_resampled_ttest(
    a: np.ndarray, b: np.ndarray, n_splits: int
) -> tuple[float, float]:
    """t-test remuestreado corregido de Nadeau & Bengio (2003) para k-fold repetido.

    Los pliegues de una validación cruzada repetida **no** son independientes: sus
    conjuntos de entrenamiento se solapan fuertemente, de modo que la varianza de las
    diferencias entre modelos está subestimada. Un t-test pareado ingenuo que trate los
    ``n_splits × n_repeats`` pliegues como observaciones independientes produce
    p-valores demasiado optimistas (error tipo I inflado).

    La corrección reemplaza ``s²/J`` por ``s²·(1/J + ρ/(1-ρ))``, con
    ``ρ = 1/n_splits`` la proporción de datos en validación en cada pliegue. Con ``J``
    pliegues y ``df = J − 1`` grados de libertad.

    Devuelve el estadístico t corregido y su p-valor a dos colas.
    """

    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n_folds = diff.size
    if n_folds < 2 or n_splits < 2:
        return float("nan"), float("nan")
    mean = float(diff.mean())
    var = float(diff.var(ddof=1))
    if var == 0.0:
        return (float("inf") if mean else 0.0), (0.0 if mean else 1.0)
    rho = 1.0 / n_splits
    corrected_var = var * (1.0 / n_folds + rho / (1.0 - rho))
    t = mean / math.sqrt(corrected_var)
    from scipy import stats

    p = float(2.0 * stats.t.sf(abs(t), df=n_folds - 1))
    return t, p


def _infer_n_splits(frame: pd.DataFrame) -> int | None:
    """Deduce k (pliegues por repetición) a partir de la columna ``repeat``."""

    if "repeat" not in frame or frame["repeat"].dropna().empty:
        return None
    n_repeats = int(frame["repeat"].nunique())
    if n_repeats == 0 or len(frame) % n_repeats != 0:
        return None
    return len(frame) // n_repeats


def paired_stats(root: str | Path, runs: dict[Any, str], metric: str = "accuracy") -> None:
    from scipy import stats
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multitest import multipletests

    values: dict[Any, np.ndarray] = {}
    splits: set[int] = set()
    reference_index: pd.Index | None = None
    for key, run_id in runs.items():
        frame = pd.read_csv(Path(root) / run_id / "metrics_val.csv")
        required = {"repeat", "fold", metric}
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{run_id}: faltan columnas requeridas {sorted(missing_columns)}"
            )
        if frame.duplicated(["repeat", "fold"]).any():
            duplicated = frame.loc[
                frame.duplicated(["repeat", "fold"], keep=False),
                ["repeat", "fold"],
            ]
            raise ValueError(
                f"{run_id}: claves repeat/fold duplicadas:\n"
                f"{duplicated.to_string(index=False)}"
            )
        # Pareamos por (repeat, fold) explícitamente en vez de confiar en que
        # ordenar por 'fold' y comparar solo la longitud alinee las mismas
        # particiones entre corridas — esa suposición se cumple hoy porque el
        # generador numera 'fold' de forma global y única, pero no está
        # garantizada por el formato del archivo.
        series = (
            frame.set_index(["repeat", "fold"])[metric]
            .sort_index()
            .astype(float)
        )
        if reference_index is None:
            reference_index = series.index
        elif not series.index.equals(reference_index):
            missing = reference_index.difference(series.index).tolist()
            extra = series.index.difference(reference_index).tolist()
            raise ValueError(
                f"{run_id}: los pliegues no coinciden con la corrida de referencia. "
                f"Faltantes={missing}; adicionales={extra}"
            )
        if not np.isfinite(series.to_numpy()).all():
            raise ValueError(f"{run_id}: la métrica {metric!r} contiene NaN o infinitos.")
        values[key] = series.to_numpy()
        k = _infer_n_splits(frame)
        if k is not None:
            splits.add(k)

    n_splits = splits.pop() if len(splits) == 1 else None
    if len(splits) > 1:
        print("\nAVISO: las corridas usan distinto n_splits; se omite la corrección "
              "de Nadeau-Bengio (requiere el mismo k).")

    long = pd.concat([
        pd.DataFrame({"value": v, "group": str(k), "fold": np.arange(1, len(v) + 1)})
        for k, v in values.items()
    ], ignore_index=True)
    print("\nANOVA de medidas repetidas (EXPLORATORIO — trata cada pliegue como "
          "observación independiente y NO corrige la dependencia entre pliegues de "
          "una k-fold repetida; no participa en el veredicto de significancia, que "
          "usa el t-test corregido de Nadeau-Bengio más abajo):\n")
    print(AnovaRM(long, "value", "fold", within=["group"]).fit())

    rows = []
    for a, b in itertools.combinations(values, 2):
        _, p_t = stats.ttest_rel(values[a], values[b])
        try:
            _, p_w = stats.wilcoxon(values[a], values[b])
        except ValueError:
            p_w = np.nan
        entry = {"grupo_1": a, "grupo_2": b,
                 "dif_pp": (values[b].mean() - values[a].mean()) * 100,
                 "p_t_pareada": p_t, "p_wilcoxon": p_w}
        if n_splits is not None:
            _, p_nb = corrected_resampled_ttest(values[a], values[b], n_splits)
            entry["p_nadeau_bengio"] = p_nb
        rows.append(entry)
    result = pd.DataFrame(rows)

    # La corrección de Holm y el veredicto de significancia se aplican sobre el
    # p-valor corregido (Nadeau-Bengio) cuando está disponible, porque el t-test
    # ingenuo sobre pliegues correlacionados sobreestima la significancia. Se conserva
    # ``p_t_pareada`` en la tabla solo como referencia.
    base_p = "p_nadeau_bengio" if "p_nadeau_bengio" in result else "p_t_pareada"
    result["p_holm"] = multipletests(result[base_p], method="holm")[1]
    result["significativo"] = result["p_holm"] < 0.05
    if n_splits is not None:
        print(f"\nContrastes pareados (corrección de varianza Nadeau-Bengio, k={n_splits}, "
              f"Holm sobre {base_p}):\n")
    else:
        print("\nContrastes pareados con corrección de Holm "
              "(sin Nadeau-Bengio: no se pudo deducir k):\n")
    print(result.round(4).to_string(index=False))


def _filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    for attr, column in [("site", "site"), ("model", "model"),
                         ("representation", "representation")]:
        value = getattr(args, attr, None)
        if value is not None:
            df = df[df[column].astype(str) == str(value)]
    if args.roi_set:
        allowed = {str(x) for x in args.roi_set}
        df = df[df["roi_set"].astype(str).isin(allowed)]
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--site")
    parser.add_argument("--model")
    parser.add_argument("--roi-set", nargs="*")
    parser.add_argument("--representation")
    parser.add_argument("--out", help="CSV detallado por corrida")
    parser.add_argument("--aggregate-out", help="CSV agregado por configuración metodológica")
    parser.add_argument("--strict", action="store_true", help="fallar ante corridas incompletas o archivos inválidos")
    parser.add_argument(
        "--strict-comparability", action="store_true",
        help="fallar (en vez de solo avisar) si check_comparability() detecta "
        "incompatibilidades entre las corridas seleccionadas (seeds, hashes, árbol sucio, "
        "configuraciones duplicadas)",
    )
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--stats-metric", default="accuracy")
    parser.add_argument(
        "--stats-by",
        choices=["roi_set", "representation", "early_stopping_monitor"],
        default="roi_set",
        help="dimensión a comparar de forma pareada: subconjuntos de ROIs "
        "(por defecto), representaciones (fija un solo roi_set), o el monitor "
        "de early stopping val_loss/val_bce (fija site/model/roi_set/"
        "representation/seed/n_splits/n_repeats)",
    )
    args = parser.parse_args(argv)

    df = collect(args.root, strict=args.strict)
    warnings = df.attrs.get("collection_warnings", [])
    notices = df.attrs.get("collection_notices", [])
    if df.empty:
        raise SystemExit(f"No se encontraron corridas completas en {args.root}")
    df = _filter(df, args)
    if df.empty:
        raise SystemExit("Ninguna corrida coincide con los filtros.")

    display_cols = [
        "run_id", "site", "roi_set", "model", "representation", "window_seconds",
        "window_tr", "step_tr", "effective_overlap", "window_shape", "fisher_z",
        "early_stopping_monitor", "n_folds", "n_windows", "val_accuracy_mean",
        "val_accuracy_sd", "val_f1_macro_mean", "val_auc_mean", "oof_auc_mean",
        "oof_f1_macro_mean", "gap_acc", "commit",
    ]
    print(df[[c for c in display_cols if c in df]].round(4).to_string(index=False))

    if warnings:
        print("\nAVISOS DE RECOLECCIÓN")
        for warning in warnings:
            print(f"  · {warning}")
    if notices:
        print("\nCORRIDAS HISTÓRICAS (esquema < 4, fuera del A/B formal por monitor)")
        for notice in notices:
            print(f"  · {notice}")
    problems = check_comparability(df)
    if problems:
        print("\nAVISOS DE COMPARABILIDAD")
        for problem in problems:
            print(f"  · {problem}")
        if args.strict_comparability:
            raise SystemExit(
                "Comparación abortada (--strict-comparability): " + "; ".join(problems)
            )
    else:
        print("\nLas corridas seleccionadas son compatibles para comparación pareada.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nTabla detallada guardada en {args.out}")
    if args.aggregate_out:
        aggregated = aggregate_table(df)
        Path(args.aggregate_out).parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(args.aggregate_out, index=False)
        print(f"Tabla agregada guardada en {args.aggregate_out}")

    if args.stats:
        base = df[df["random_subset"].isna()] if "random_subset" in df else df
        group_col = args.stats_by
        # Todo lo que no sea la dimensión a comparar debe quedar fijo.
        if group_col == "representation":
            fixed = ["site", "model", "roi_set"]  # comparar representaciones dentro de un mismo roi_set
        elif group_col == "early_stopping_monitor":
            # comparar val_loss vs val_bce dentro de la misma configuración
            # metodológica; seed/n_splits/n_repeats se verifican aparte por
            # claridad del mensaje, aunque split_fingerprint ya los garantiza.
            fixed = ["site", "model", "roi_set", "representation", "seed", "n_splits", "n_repeats"]
        else:
            fixed = ["site", "model", "representation"]  # comparar ROIs dentro de una misma representación
        for column in fixed:
            if column in base and base[column].dropna().nunique() > 1:
                raise SystemExit(
                    f"Para --stats por {group_col}, filtre un solo {column} "
                    f"(hay {base[column].nunique()}: use --{column.replace('_', '-')})."
                )
        if "split_fingerprint" in base and base["split_fingerprint"].dropna().nunique() > 1:
            raise SystemExit("Las corridas no comparten la misma huella de particiones.")

        if group_col == "early_stopping_monitor":
            _check_early_stopping_ab(base)

        # Antes de construir 'runs': si hay dos corridas con el mismo valor de
        # group_col, la comprensión de diccionario de abajo se quedaría
        # silenciosamente con la última y compararía menos pares de los que el
        # usuario cree, sin avisar. Se detecta y aborta explícitamente.
        dup_mask = base[group_col].duplicated(keep=False)
        if dup_mask.any():
            dup_values = sorted(base.loc[dup_mask, group_col].dropna().unique().tolist())
            dup_ids = base.loc[dup_mask, "base_run_id"].tolist()
            raise SystemExit(
                f"Hay más de una corrida para el/los mismo(s) valor(es) de {group_col} "
                f"{dup_values}: {dup_ids}. Filtre hasta dejar una sola corrida por valor "
                "antes de comparar (--stats necesita exactamente un run_id por grupo)."
            )

        order = "n_rois" if group_col == "roi_set" else group_col
        runs = {
            row[group_col]: row.base_run_id
            for _, row in base.sort_values(order).iterrows()
        }
        if len(runs) < 2:
            raise SystemExit(f"Se requieren al menos dos valores de {group_col}.")
        paired_stats(args.root, runs, args.stats_metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
