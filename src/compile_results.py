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
OOF_METRICS = ("accuracy", "auc", "f1_macro", "balanced_accuracy", "log_loss", "brier")
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

    Devuelve una fila por repetición con exactitud (accuracy), AUC, F1 macro, exactitud
    balanceada, log-loss y Brier, o ``None`` si el archivo de predicciones no tiene el
    formato esperado.
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
            "accuracy": float((pred == y).mean()),
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
    env = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
    representation = cfg.get("representation") or ("static" if cfg.get("window") is None else "ordered")
    mode = "static" if representation in ("static", "partial", "shrunk", "tangent") else "dynamic"
    # Claves ordenadas para que dos corridas con la misma arquitectura y los mismos
    # hiperparámetros produzcan el mismo texto sin importar en qué orden Python haya
    # recorrido el diccionario — así arch_json sirve para comparar por igualdad exacta
    # sin abrir una columna por hiperparámetro (que variaría de un modelo a otro).
    arch_json = json.dumps(cfg.get("arch") or {}, sort_keys=True, separators=(",", ":"))

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
        "n_random_sets": cfg.get("n_random_sets"),
        "exclude_roi_set": cfg.get("exclude_roi_set"),
        "arch_json": arch_json,
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
        "epochs": cfg.get("epochs"),
        "patience": cfg.get("patience"),
        "clipnorm": cfg.get("clipnorm"),
        "inner_val_frac": cfg.get("inner_val_frac"),
        "class_weight": cfg.get("class_weight"),
        "deterministic": cfg.get("deterministic"),
        "mixed_precision": cfg.get("mixed_precision"),
        "start_from_epoch": cfg.get("start_from_epoch"),
        # Versiones descriptivas del entorno de ejecución: no participan en
        # config_hash ni en la comparabilidad (dos corridas con Keras 3.12 y 3.13
        # pueden seguir siendo el mismo experimento), pero explican una diferencia
        # de rendimiento si aparece.
        "python_version": env.get("python"),
        "tensorflow_version": env.get("tensorflow"),
        "keras_version": env.get("keras"),
        "sklearn_version": env.get("scikit_learn"),
        "gpu": ", ".join(env["gpu"]) if isinstance(env.get("gpu"), list) else env.get("gpu"),
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


def _parse_schema_version(cfg: dict[str, Any]) -> int | None:
    """Devuelve ``config_schema_version`` como ``int`` válido (``>= 1``), o
    ``None`` si el campo es de un tipo inaceptable (texto, booleano, decimal)
    o menor que 1. Ausente se interpreta como el valor histórico ``1``.

    Único lugar donde se hace esta conversión — la usan tanto
    ``validate_run_artifacts()`` como ``collect()``, para que un
    ``config_schema_version`` corrupto no dependa de que cada llamador repita
    la misma comprobación de tipo antes de comparar contra el entero 4.
    """
    schema_raw = cfg.get("config_schema_version", 1)
    if (
        isinstance(schema_raw, bool)
        or not isinstance(schema_raw, (int, np.integer))
        or int(schema_raw) < 1
    ):
        return None
    return int(schema_raw)


def validate_run_artifacts(
    run_dir: Path, suffix: str = "", cfg: dict[str, Any] | None = None
) -> list[str]:
    """Valida una sola carpeta de corrida de punta a punta.

    Es la fuente única de estas reglas — no hay una segunda implementación
    en otro archivo: ``collect(strict=True)`` la aplica sobre cada corrida
    de esquema >= 4 que compila, ``verify_setup.py --full`` la aplica sobre
    las corridas que acaba de entrenar, y el notebook la aplica sobre la
    carpeta que se acaba de producir antes de descargarla o subirla.

    Con ``config_schema_version < 4`` (corridas anteriores al monitor
    configurable y a ``restored_monitor_value``) solo comprueba que los
    cinco artefactos existan, sean legibles y no estén vacíos: exigir el
    resto retroactivamente rechazaría corridas históricas que nunca
    produjeron esos campos. Ver ``docs/limitations.md``.

    Con ``config_schema_version >= 4`` añade: columnas estructurales en los
    cinco artefactos, valores finitos en los campos numéricos obligatorios,
    ``n_splits * n_repeats`` filas únicas de ``(fold, repeat)`` en
    ``metrics_train``/``metrics_val``, las mismas claves ``(fold, repeat)``
    en ambos, una serie de épocas completa por pliegue en ``history``, que
    el valor registrado en ``best_epoch`` coincida con ``best_monitor_value``
    y que ``restored_monitor_value`` esté cerca de ``best_monitor_value``
    (la prueba no circular de qué pesos restauró ``EarlyStopping`` — ver
    ``methodology.md``), ``y_true`` en {0,1} y ``y_prob`` en [0,1], un solo
    ``subject_id`` por repetición en las predicciones OOF y cobertura de
    ``n_subjects``, las particiones ``fit``/``inner_val``/``outer_val``
    disjuntas por pliegue, que los sujetos de ``predictions_val.csv``
    coincidan con el ``outer_val`` de ``folds.csv``, y que los tamaños de
    partición coincidan con ``n_fit``/``n_inner_val``/``n_outer_val``.
    """
    problems: list[str] = []
    if cfg is None:
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            return [f"falta config.json en {run_dir}"]
        try:
            cfg = _read_json(cfg_path)
        except ValueError as exc:
            return [str(exc)]

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
    for name, frame in frames.items():
        if frame.empty:
            problems.append(f"{name}{suffix}.csv está vacío")
    if problems:
        return problems

    # config_schema_version debe validarse antes de comparar con el entero 4:
    # un valor textual o booleano hacía que "schema < 4" lanzara TypeError
    # (o, con un booleano, comparara silenciosamente contra 0/1).
    schema = _parse_schema_version(cfg)
    if schema is None:
        return [
            "config.json: config_schema_version debe ser un entero >= 1; "
            f"se recibió {cfg.get('config_schema_version', 1)!r}"
        ]
    if schema < 4:
        return problems

    # n_splits/n_repeats/n_subjects activan las comprobaciones de cobertura
    # de más abajo (filas esperadas, unión de particiones); si faltan o son
    # inválidos, esas comprobaciones se desactivaban en silencio en vez de
    # señalar el problema. Se acumulan junto con el resto de columnas
    # estructurales (no se corta en el primer campo inválido) para que un
    # config.json con los tres campos rotos los reporte los tres a la vez.
    campo_specs = (("n_splits", 2), ("n_repeats", 1), ("n_subjects", 1))
    for campo, minimo in campo_specs:
        valor = cfg.get(campo)
        if (
            campo not in cfg
            or isinstance(valor, bool)
            or not isinstance(valor, (int, np.integer))
            or int(valor) < minimo
        ):
            problems.append(
                f"config.json: {campo} debe ser un entero >= {minimo}; "
                f"se recibió {valor!r}"
            )

    # Columnas exigidas por archivo, y de esas, cuáles deben además ser
    # finitas. No se exige (ni se audita finitud en) el resto de columnas:
    # varias son opcionales por diseño (p. ej. class_weight_0/1 solo se
    # rellenan con --class-weight) y NaN ahí es legítimo, no un fallo. Los
    # identificadores estructurales (fold/repeat/subject_id/split/epoch) se
    # exigen en los cinco artefactos, no solo en metrics_train/val/history:
    # sin ellos, predictions_val.csv y folds.csv podían faltar por completo
    # sus columnas y aun así "pasar" la validación. n_fit/n_inner_val/
    # n_outer_val se exigen aparte de las particiones porque son el
    # contraste independiente contra los tamaños reales de folds.csv más
    # abajo — sin ellos ese contraste no puede hacerse.
    metrics_common = {
        "fold", "repeat", "n_epochs", "best_epoch", "n_fit", "n_inner_val", "n_outer_val",
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
        "metrics_train": {"fold", "repeat", "n_epochs", "best_epoch", "n_fit", "n_inner_val",
                          "n_outer_val", "best_monitor_value", "restored_monitor_value"},
        "metrics_val": {"fold", "repeat", "n_epochs", "best_epoch", "n_fit", "n_inner_val",
                        "n_outer_val", "best_monitor_value", "restored_monitor_value"},
        "history": {"fold", "repeat", "epoch", "loss", "inner_val_loss", "bce", "inner_val_bce"},
        "predictions_val": {"fold", "repeat", "y_true", "y_prob"},
        "folds": {"fold", "repeat"},
    }
    # Columnas de texto que no pueden quedar vacías/nulas — subject_id y
    # split son identificadores estructurales igual que fold/repeat, solo
    # que no numéricos, así que se validan aparte de finite_by_file.
    text_by_file = {
        "metrics_train": {"early_stopping_monitor"},
        "metrics_val": {"early_stopping_monitor"},
        "predictions_val": {"subject_id"},
        "folds": {"subject_id", "split"},
    }

    for name, required in required_by_file.items():
        missing = required - set(frames[name].columns)
        if missing:
            problems.append(f"{name}{suffix}.csv: faltan columnas {sorted(missing)}")

    for name, columns in finite_by_file.items():
        frame = frames.get(name)
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

    for name, columns in text_by_file.items():
        frame = frames.get(name)
        for col in (c for c in columns if c in frame.columns):
            vacio = frame[col].isna() | (frame[col].astype(str).str.strip() == "")
            if vacio.any():
                problems.append(f"{name}{suffix}.csv: columna {col!r} tiene {int(vacio.sum())} valores vacíos")

    # A partir de aquí las comprobaciones son semánticas (cruzan varios
    # archivos, comparan valores entre sí) y asumen columnas presentes con
    # tipos correctos. Si alguna comprobación estructural de arriba ya
    # encontró un problema, se corta aquí: seguir adelante sobre datos que ya
    # se sabe que están corruptos (p. ej. un texto donde se esperaba un
    # número) es lo que antes dejaba escapar un TypeError de np.isfinite()
    # en vez de un diagnóstico legible.
    if problems:
        return problems

    train, val, hist = frames["metrics_train"], frames["metrics_val"], frames["history"]
    pred, folds = frames["predictions_val"], frames["folds"]

    try:
        # n_splits/n_repeats/n_subjects ya se validaron como enteros en rango
        # más arriba (y si no lo eran, la función ya retornó antes de llegar
        # aquí) — se usan directamente, sin un "if n_splits and n_repeats"
        # que trataría un 0 inválido como "no lo sé" y apagaría en silencio
        # la comprobación de expected_rows.
        n_splits, n_repeats, n_subjects = (
            int(cfg["n_splits"]), int(cfg["n_repeats"]), int(cfg["n_subjects"]),
        )
        expected_rows = n_splits * n_repeats

        # Claves únicas por archivo (sin filas duplicadas). Antes solo se
        # comprobaba esto para metrics_train/metrics_val; una fila repetida en
        # folds.csv o una época repetida en history.csv pasaban inadvertidas
        # porque las comprobaciones siguientes construían conjuntos (que
        # deduplican) en vez de contar filas.
        key_cols_by_file = {
            "metrics_train": ["repeat", "fold"],
            "metrics_val": ["repeat", "fold"],
            "history": ["repeat", "fold", "epoch"],
            "predictions_val": ["repeat", "subject_id"],
            "folds": ["repeat", "fold", "subject_id"],
        }
        for name, frame in frames.items():
            cols = key_cols_by_file[name]
            if not set(cols) <= set(frame.columns):
                continue
            n_total, n_unique = len(frame), len(frame[cols].drop_duplicates())
            if n_total != n_unique:
                problems.append(
                    f"{name}{suffix}.csv: {n_total} filas pero {n_unique} claves "
                    f"{tuple(cols)} únicas (hay filas duplicadas)"
                )

        # (repeat, fold) esperado: n_splits × n_repeats pares únicos en
        # metrics_val, y ese mismo conjunto —sin faltantes ni sobrantes— en
        # metrics_train, history, predictions_val y folds. Antes solo se
        # comparaban metrics_train y metrics_val entre sí; un fold ausente
        # por completo en folds.csv (o en predictions_val.csv) no lo
        # detectaba nada, porque el resto del código solo agrupaba por
        # (fold, repeat) presentes y nunca notaba los que faltaban.
        fold_repeat_by_file: dict[str, set] = {}
        for name, cols in (("metrics_train", ["repeat", "fold"]), ("history", ["repeat", "fold"]),
                          ("predictions_val", ["repeat", "fold"]), ("folds", ["repeat", "fold"])):
            frame = frames[name]
            if set(cols) <= set(frame.columns):
                fold_repeat_by_file[name] = set(map(tuple, frame[cols].to_numpy()))
        reference = set(map(tuple, val[["repeat", "fold"]].to_numpy()))
        if len(reference) != expected_rows:
            problems.append(
                f"metrics_val{suffix}.csv: {len(reference)} pares (repeat, fold) únicos, "
                f"se esperaban {expected_rows} (n_splits={n_splits} × n_repeats={n_repeats})"
            )
        for name, keys in fold_repeat_by_file.items():
            if keys != reference:
                problems.append(
                    f"{name}{suffix}.csv no cubre los mismos pares (repeat, fold) que "
                    f"metrics_val{suffix}.csv: solo en metrics_val="
                    f"{sorted(reference - keys)}, solo en {name}={sorted(keys - reference)}"
                )

        # Repeticiones cubiertas por las predicciones OOF frente a las de las
        # métricas — un desajuste aquí sobreviviría a la comprobación de
        # (repeat, fold) de arriba si predictions_val.csv, por error, trajera
        # una repetición completa que metrics_val.csv no tiene con las mismas
        # claves exactas de fold.
        if {"repeat"} <= set(pred.columns):
            pred_repeats = set(pred["repeat"].unique())
            metric_repeats = {r for r, _ in reference}
            if pred_repeats != metric_repeats:
                problems.append(
                    f"predictions_val{suffix}.csv cubre repeticiones {sorted(pred_repeats)}, "
                    f"metrics_val{suffix}.csv cubre {sorted(metric_repeats)}"
                )

        # Serie de épocas completa por pliegue, best_epoch en rango, y su valor en
        # history coincide con best_monitor_value (no circular: restored_monitor_value
        # viene de una reevaluación del modelo, no de leer esta misma fila).
        for _, row in val.iterrows():
            fold_v, repeat_v = row["fold"], row["repeat"]
            n_epochs_fold, best_epoch = int(row["n_epochs"]), int(row["best_epoch"])
            fold_hist = hist[(hist["fold"] == fold_v) & (hist["repeat"] == repeat_v)]
            epochs_presentes = set(fold_hist["epoch"].dropna().astype(int))
            epochs_esperadas = set(range(1, n_epochs_fold + 1))
            if epochs_presentes != epochs_esperadas:
                problems.append(
                    f"f{fold_v}r{repeat_v}: history{suffix}.csv no cubre las épocas "
                    f"1..{n_epochs_fold} (faltan {sorted(epochs_esperadas - epochs_presentes)})"
                )
            if not (1 <= best_epoch <= n_epochs_fold):
                problems.append(f"f{fold_v}r{repeat_v}: best_epoch={best_epoch} fuera de [1, {n_epochs_fold}]")
                continue
            monitor = row.get("early_stopping_monitor")
            inner_col = {"val_loss": "inner_val_loss", "val_bce": "inner_val_bce"}.get(monitor)
            best_value = float(row["best_monitor_value"])
            if inner_col is None or inner_col not in hist.columns:
                continue
            recorded = fold_hist.loc[fold_hist["epoch"] == best_epoch, inner_col]
            if recorded.empty:
                problems.append(f"f{fold_v}r{repeat_v}: no hay fila de history{suffix}.csv para epoch={best_epoch}")
            elif abs(float(recorded.iloc[0]) - best_value) > 1e-6:
                problems.append(
                    f"f{fold_v}r{repeat_v}: history[{inner_col}][{best_epoch}]={float(recorded.iloc[0])} "
                    f"!= best_monitor_value={best_value}"
                )
            restored_value = float(row["restored_monitor_value"])
            if not np.isfinite(restored_value):
                problems.append(f"f{fold_v}r{repeat_v}: restored_monitor_value no finito")
            elif abs(restored_value - best_value) > 1e-4:
                problems.append(
                    f"f{fold_v}r{repeat_v}: restored_monitor_value={restored_value} se "
                    f"aleja de best_monitor_value={best_value} (pesos no consistentes)"
                )

        # Rango de etiquetas y probabilidades en las predicciones OOF.
        y_true = pd.to_numeric(pred["y_true"], errors="coerce")
        y_prob = pd.to_numeric(pred["y_prob"], errors="coerce")
        fuera_de_clase = sorted(y_true[~y_true.isin([0, 1])].dropna().unique().tolist())
        if fuera_de_clase:
            problems.append(f"predictions_val{suffix}.csv: y_true fuera de {{0,1}}: {fuera_de_clase}")
        if not y_prob.between(0, 1).all():
            problems.append(f"predictions_val{suffix}.csv: y_prob tiene valores fuera de [0, 1]")

        # Cobertura de n_subjects por repetición (los duplicados de subject_id
        # ya se descartaron arriba, en la comprobación de clave única).
        # n_subjects ya se validó como entero >= 1 más arriba.
        for repeat, group in pred.groupby("repeat"):
            if group["subject_id"].nunique() != n_subjects:
                problems.append(
                    f"predictions_val{suffix}.csv, repetición {repeat}: "
                    f"{group['subject_id'].nunique()} sujetos, se esperaban {n_subjects}"
                )

        # y_true debe ser el mismo para un sujeto en todas las repeticiones en
        # las que aparece: es una etiqueta del sujeto, no de la repetición.
        y_true_por_sujeto = pred.assign(y_true=y_true).groupby("subject_id")["y_true"].nunique()
        inconsistentes = sorted(y_true_por_sujeto[y_true_por_sujeto > 1].index.tolist())
        if inconsistentes:
            problems.append(
                f"predictions_val{suffix}.csv: y_true distinto para el mismo sujeto entre "
                f"repeticiones: {inconsistentes}"
            )

        # El conjunto de sujetos predichos OOF debe ser idéntico entre
        # repeticiones (cada repetición cubre a todos los sujetos una vez).
        sujetos_por_repeticion = {r: frozenset(g["subject_id"]) for r, g in pred.groupby("repeat")}
        if len(set(sujetos_por_repeticion.values())) > 1:
            problems.append(
                f"predictions_val{suffix}.csv: el conjunto de sujetos no es idéntico en "
                "todas las repeticiones"
            )

        # Valores de split permitidos. Antes, una fila con split="mystery" no
        # encajaba en ninguno de los tres conjuntos de abajo y simplemente se
        # ignoraba, en vez de señalarse como un valor inválido.
        valores_permitidos = {"fit", "inner_val", "outer_val"}
        desconocidos = sorted(set(folds["split"].unique()) - valores_permitidos)
        if desconocidos:
            problems.append(f"folds{suffix}.csv: valores de split desconocidos {desconocidos}")

        # Particiones fit/inner_val/outer_val disjuntas, su unión cubre
        # exactamente n_subjects, y sus tamaños coinciden con
        # n_fit/n_inner_val/n_outer_val registrados en metrics_val.
        conjuntos_por_grupo: dict[tuple, frozenset] = {}
        for (fold_v, repeat_v), g in folds.groupby(["fold", "repeat"]):
            # Los tres splits deben estar presentes en cada (repeat, fold),
            # independientemente de n_fit/n_inner_val/n_outer_val: un
            # artefacto manipulado puede mover todas las filas de un split a
            # otro y ajustar esos conteos para que sigan coincidiendo, sin
            # que ese split deje de faltar del todo. present_splits/
            # missing_splits se calcula directamente sobre las filas de este
            # grupo, no sobre los tamaños declarados.
            present_splits = set(g["split"])
            missing_splits = valores_permitidos - present_splits
            if missing_splits:
                problems.append(
                    f"f{fold_v}r{repeat_v}: faltan splits {sorted(missing_splits)} "
                    f"(presentes: {sorted(present_splits)})"
                )
            sets = {s: set(g.loc[g["split"] == s, "subject_id"]) for s in valores_permitidos}
            conjuntos_por_grupo[(fold_v, repeat_v)] = frozenset(g["subject_id"])
            if sets["fit"] & sets["inner_val"]:
                problems.append(f"f{fold_v}r{repeat_v}: fit∩inner_val no vacío")
            if sets["fit"] & sets["outer_val"]:
                problems.append(f"f{fold_v}r{repeat_v}: fit∩outer_val no vacío")
            if sets["inner_val"] & sets["outer_val"]:
                problems.append(f"f{fold_v}r{repeat_v}: inner_val∩outer_val no vacío")
            union_total = sets["fit"] | sets["inner_val"] | sets["outer_val"]
            if len(union_total) != n_subjects:
                problems.append(
                    f"f{fold_v}r{repeat_v}: fit ∪ inner_val ∪ outer_val tiene "
                    f"{len(union_total)} sujetos, se esperaban n_subjects={n_subjects}"
                )
            if {"fold", "repeat", "subject_id"} <= set(pred.columns):
                pred_ids = set(pred.loc[(pred["fold"] == fold_v) & (pred["repeat"] == repeat_v), "subject_id"])
                if pred_ids != sets["outer_val"]:
                    problems.append(
                        f"f{fold_v}r{repeat_v}: sujetos de predictions_val{suffix}.csv no "
                        f"coinciden con outer_val de folds{suffix}.csv"
                    )
            fila = val[(val["fold"] == fold_v) & (val["repeat"] == repeat_v)]
            if not fila.empty:
                esperado = fila.iloc[0]
                for split_name, columna in (
                    ("fit", "n_fit"), ("inner_val", "n_inner_val"), ("outer_val", "n_outer_val")
                ):
                    n_esperado = int(esperado[columna])
                    if len(sets[split_name]) != n_esperado:
                        problems.append(
                            f"f{fold_v}r{repeat_v}: {split_name} tiene {len(sets[split_name])} "
                            f"sujetos en folds{suffix}.csv, se esperaban {n_esperado} ({columna})"
                        )

        # El conjunto total de sujetos (fit ∪ inner_val ∪ outer_val) debe ser
        # el mismo en todos los folds y repeticiones — todas parten de la
        # misma cohorte, solo cambia cómo se reparte.
        if len(set(conjuntos_por_grupo.values())) > 1:
            problems.append(
                f"folds{suffix}.csv: el conjunto total de sujetos no es el mismo en todos "
                "los folds/repeticiones"
            )
    except (TypeError, ValueError, KeyError) as exc:
        # Red de seguridad: cualquier corrupción de contenido no anticipada
        # por las comprobaciones anteriores se convierte en un diagnóstico
        # legible en vez de propagarse como una excepción sin contexto.
        problems.append(
            f"{run_dir.name}{suffix}: error inesperado validando el contenido "
            f"({type(exc).__name__}: {exc})"
        )

    return problems


def _find_run_dir(root: str | Path, run_id: str) -> Path:
    """Ubica la carpeta de una corrida por su ``run_id``, soportando tanto el
    layout plano histórico (``root/<run_id>/``) como el layout actual anidado
    por ROI (``root/<roi_set>/<run_id>/`` — ver ``run_experiment.py``, que
    escribe ahí desde que ``results/`` se organiza por tamaño de ROI). Prueba
    primero el layout plano y, si no existe, busca un único candidato bajo
    cualquier subcarpeta de primer nivel.
    """
    root = Path(root)
    plano = root / run_id
    if plano.exists():
        return plano
    candidatos = sorted(root.glob(f"*/{run_id}"))
    if len(candidatos) == 1:
        return candidatos[0]
    if len(candidatos) > 1:
        raise FileNotFoundError(
            f"{run_id}: aparece en más de una subcarpeta bajo {root}: "
            f"{[str(c) for c in candidatos]}"
        )
    raise FileNotFoundError(f"{run_id}: no se encontró bajo {root} (ni layout plano ni por ROI)")


def collect(root: str | Path, *, strict: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    notices: list[str] = []
    seen_schema_notice: set[str] = set()
    root = Path(root)
    # results/ organiza las corridas por subcarpeta de ROI
    # (root/<roi_set>/<run_id>/config.json); root.glob("*/*/config.json")
    # cubre ese layout. root.glob("*/config.json") sigue aceptando el layout
    # plano histórico (root/<run_id>/config.json) para no dejar invisibles
    # corridas ya subidas antes de este cambio. cfg_path.parent.name da el
    # run_id en ambos casos (es el último segmento de la ruta), así que el
    # resto de la función no necesita distinguir un layout del otro.
    cfg_paths = sorted(root.glob("*/config.json")) + sorted(root.glob("*/*/config.json"))
    for cfg_path in cfg_paths:
        try:
            cfg = _read_json(cfg_path)
            run_dir = cfg_path.parent
            val_files = sorted(run_dir.glob("metrics_val*.csv"))
            if not val_files:
                errors.append(f"{run_dir.name}: corrida incompleta, sin metrics_val*.csv")
                continue
            # _parse_schema_version() evita comparar un config_schema_version
            # de tipo inválido (texto, booleano) contra el entero 4 más abajo
            # — esa comparación lanzaba TypeError, que el except genérico de
            # este bucle capturaba igual, pero con el texto crudo de la
            # excepción de Python en vez de un diagnóstico que nombre el campo.
            schema = _parse_schema_version(cfg)
            if schema is None:
                errors.append(
                    f"{run_dir.name}: config.json: config_schema_version debe ser un "
                    f"entero >= 1; se recibió {cfg.get('config_schema_version', 1)!r}"
                )
                continue
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
                    problems = validate_run_artifacts(run_dir, suffix, cfg=cfg)
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
        "site", "roi_set", "model", "arch_json", "representation", "representation_seed",
        "connectivity_mode", "windowing_preset",
        "window_tr", "step_tr", "window_seconds", "step_seconds",
        "requested_window_seconds", "requested_step_seconds", "requested_overlap",
        "effective_overlap", "window_shape", "gaussian_sigma", "fisher_z",
        "constant_policy", "lr", "batch_size", "epochs", "patience", "clipnorm",
        "inner_val_frac", "class_weight", "deterministic", "mixed_precision",
        "start_from_epoch", "random_subset", "n_random_sets", "exclude_roi_set",
        "early_stopping_monitor", "early_stopping_min_delta",
        "seed", "n_splits", "n_repeats", "split_fingerprint",
        "roi_indices_hash", "bold_hash", "atlas_hash", "data_code_hash", "runner_code_hash",
    ]
    # Versiones de software y GPU quedan fuera a propósito: son metadatos del
    # entorno de ejecución, no parte de la identidad de la corrida en
    # run_experiment.py (config_hash no las incluye), así que agrupar por
    # ellas separaría corridas metodológicamente idénticas solo porque
    # corrieron con distinto Keras o distinta GPU.
    return [c for c in candidates if c in df.columns]


def aggregate_table(df: pd.DataFrame, group_by: Iterable[str] | None = None) -> pd.DataFrame:
    groups = list(group_by or methodological_group_columns(df))
    metric_cols = [
        c for c in df.columns
        if c.endswith("_mean") and (c.startswith("val_") or c.startswith("oof_"))
    ]
    if not groups or not metric_cols:
        return pd.DataFrame()

    # methodological_group_columns() cubre los ejes conocidos, pero una lista de
    # columnas siempre puede quedarse corta ante un parámetro nuevo. config_hash es
    # la identidad completa de la corrida (ver run_experiment.py): si un grupo
    # mezcla más de un config_hash, hay una diferencia real que las columnas
    # agrupadas no capturaron, y promediar produciría una media sin sentido entre
    # configuraciones distintas. Se aborta señalando el grupo y los run_id en
    # conflicto en vez de promediar de todos modos.
    if "config_hash" in df.columns:
        conflictos = []
        for key, group in df.groupby(groups, dropna=False):
            hashes = sorted(group["config_hash"].dropna().unique().tolist())
            if len(hashes) > 1:
                key_tuple = key if isinstance(key, tuple) else (key,)
                etiqueta = dict(zip(groups, key_tuple))
                conflictos.append(
                    f"{etiqueta}: config_hash {hashes}, run_id "
                    f"{group['base_run_id'].tolist()}"
                )
        if conflictos:
            raise SystemExit(
                "aggregate_table(): un grupo metodológico contiene más de un "
                "config_hash — las corridas comparten las columnas agrupadas pero no "
                "son la misma configuración. Revise las diferencias antes de "
                "promediar:\n- " + "\n- ".join(conflictos)
            )

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
        frame = pd.read_csv(_find_run_dir(root, run_id) / "metrics_val.csv")
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
