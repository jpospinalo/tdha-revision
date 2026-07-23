#!/usr/bin/env python3
"""
Clasificación TDAH vs. control a partir de conectividad funcional.

La versión 2 del ejecutor mantiene la compatibilidad con los comandos históricos
(``--window`` y ``--step`` en TR) y añade:

- ventanas expresadas en segundos o por solapamiento;
- conectividad estática y representaciones de ablación temporal;
- ventana rectangular o gaussiana y Fisher z opcional;
- diagnósticos de redundancia registrados en ``config.json``;
- pesos de clase calculados exclusivamente con el subconjunto ``fit``;
- huella de las particiones externas e internas;
- hashes del código y de los índices ROI para trazabilidad.

La selección de época sigue realizándose con una partición interna del
entrenamiento. El pliegue externo se utiliza una sola vez para la evaluación.
La comparación de múltiples ventanas, arquitecturas o subconjuntos debe
preespecificarse o realizarse mediante validación anidada; el script no convierte
una búsqueda retrospectiva sobre los pliegues externos en una estimación final
sin sesgo.

Ejemplos
--------
Configuración histórica::

    python run_experiment.py --site NYU --roi-set 12

Ventana física de 100 s y 75 % de solapamiento::

    python run_experiment.py --site NYU --roi-set 12 \
        --window-seconds 100 --overlap 0.75

Ablaciones::

    python run_experiment.py --site NYU --roi-set 12 --representation static
    python run_experiment.py --site NYU --roi-set 12 \
        --window-seconds 100 --overlap 0.75 --representation permuted
    python run_experiment.py --site NYU --roi-set 12 \
        --window-seconds 100 --overlap 0.75 --representation mean_std
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

# Fija el backend de Keras a TensorFlow antes de cualquier import de keras, para evitar
# que el entorno (p. ej. Colab) intente un backend en mal estado, como una instalación
# de JAX a medio inicializar. setdefault respeta un valor explícito del usuario.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

try:  # ejecución habitual: ``cd src && python run_experiment.py``
    import data as tdha_data
    import kerasmodels
except ModuleNotFoundError:  # importación desde pruebas o ``python -m src...``
    from src import data as tdha_data  # type: ignore
    from src import kerasmodels  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "runs"
CONFIG_SCHEMA_VERSION = 2
REPRESENTATIONS = (
    "ordered", "permuted", "mean", "mean_std", "static", "partial", "hybrid", "multiview",
)


# ---------------------------------------------------------------------------
# Entorno, hashes y utilidades
# ---------------------------------------------------------------------------

def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                args,
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD") or "desconocido",
        "clean": (status == "") if status is not None else None,
        "user": run("git", "config", "user.name")
        or os.environ.get("USER", "desconocido"),
    }


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except Exception as exc:  # pragma: no cover - depende del entorno
        info["scikit_learn"] = f"no disponible ({type(exc).__name__})"

    try:
        import tensorflow as tf
        import keras

        info["tensorflow"] = tf.__version__
        info["keras"] = keras.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu"] = [
            tf.config.experimental.get_device_details(gpu).get("device_name", "?")
            for gpu in gpus
        ] or "sin GPU"
    except Exception as exc:  # pragma: no cover - depende del entorno
        info["tensorflow"] = f"no disponible ({type(exc).__name__})"
    return info


def file_hash(path: str | Path, *, length: int = 16) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def indices_hash(indices: Iterable[int], *, length: int = 16) -> str:
    array = np.ascontiguousarray(np.asarray(list(indices), dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()[:length]


def config_hash(identity: Mapping[str, Any], *, length: int = 8) -> str:
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def sanitize_component(value: Any, *, max_length: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return (text or "x")[:max_length]


def parse_model_args(pairs: Sequence[str] | None) -> dict[str, Any]:
    """Convierte ``['units=128', 'dropout=0.2']`` en un diccionario."""

    output: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(
                f"ERROR: --model-arg espera clave=valor, se recibió {pair!r}."
            )
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("ERROR: --model-arg contiene una clave vacía.")
        if key in output:
            raise SystemExit(f"ERROR: --model-arg repite la clave {key!r}.")

        value: Any = raw
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            for caster in (int, float):
                try:
                    value = caster(raw)
                    break
                except ValueError:
                    continue
        output[key] = value
    return output


def _positive_int(name: str, value: int, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SystemExit(f"ERROR: {name} debe ser entero.")
    value = int(value)
    if value < minimum:
        raise SystemExit(f"ERROR: {name} debe ser >= {minimum}; se recibió {value}.")
    return value


def _positive_float(name: str, value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise SystemExit(f"ERROR: {name} debe ser numérico.")
    value = float(value)
    minimum_ok = value >= 0 if allow_zero else value > 0
    if not np.isfinite(value) or not minimum_ok:
        relation = ">= 0" if allow_zero else "> 0"
        raise SystemExit(f"ERROR: {name} debe ser finito y {relation}.")
    return value


# ---------------------------------------------------------------------------
# Validación cruzada y pesos de clase
# ---------------------------------------------------------------------------

def validate_training_args(args: argparse.Namespace, y: np.ndarray) -> None:
    args.n_splits = _positive_int("--n-splits", args.n_splits, minimum=2)
    args.n_repeats = _positive_int("--n-repeats", args.n_repeats)
    args.batch_size = _positive_int("--batch-size", args.batch_size)
    args.epochs = _positive_int("--epochs", args.epochs)
    args.patience = _positive_int("--patience", args.patience, minimum=0)
    args.start_from_epoch = _positive_int(
        "--start-from-epoch", args.start_from_epoch, minimum=0
    )
    args.lr = _positive_float("--lr", args.lr)
    if args.clipnorm is not None:
        args.clipnorm = _positive_float("--clipnorm", args.clipnorm)
    if not 0 < float(args.inner_val_frac) < 1:
        raise SystemExit("ERROR: --inner-val-frac debe pertenecer a (0, 1).")

    labels = np.asarray(y, dtype=np.int64)
    values, counts = np.unique(labels, return_counts=True)
    if set(values.tolist()) != {0, 1}:
        raise SystemExit(
            f"ERROR: se esperaban etiquetas binarias 0/1; se encontraron {values.tolist()}."
        )
    if counts.min() < args.n_splits:
        raise SystemExit(
            "ERROR: --n-splits excede el número de sujetos de la clase minoritaria "
            f"({int(counts.min())})."
        )


def compute_class_weights(y_fit: Sequence[int]) -> dict[int, float]:
    """Pesos balanceados calculados únicamente con el subconjunto de ajuste."""

    labels = np.asarray(y_fit, dtype=np.int64).ravel()
    counts = np.bincount(labels, minlength=2)
    if labels.size == 0 or np.any(counts == 0):
        raise ValueError(
            "El subconjunto fit debe contener al menos un sujeto de cada clase."
        )
    return {
        class_id: float(labels.size / (2.0 * count))
        for class_id, count in enumerate(counts)
    }


def build_split_plan(y: Sequence[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Construye una sola vez los pliegues externos e internos."""

    labels = np.asarray(y, dtype=np.int32).ravel()
    outer = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )
    plan: list[dict[str, Any]] = []

    for fold_index, (outer_train, outer_val) in enumerate(
        outer.split(np.zeros(labels.size, dtype=np.uint8), labels)
    ):
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.inner_val_frac,
            random_state=args.seed + fold_index,
        )
        try:
            fit_rel, inner_rel = next(
                inner.split(
                    np.zeros(outer_train.size, dtype=np.uint8),
                    labels[outer_train],
                )
            )
        except ValueError as exc:
            raise SystemExit(
                "ERROR: no fue posible crear la partición interna estratificada. "
                "Reduzca --n-splits o aumente --inner-val-frac. "
                f"Detalle: {exc}"
            ) from exc

        fit_idx = outer_train[fit_rel]
        inner_val_idx = outer_train[inner_rel]
        if np.unique(labels[fit_idx]).size != 2:
            raise SystemExit(
                f"ERROR: el subconjunto fit del pliegue {fold_index + 1} no contiene ambas clases."
            )
        if np.unique(labels[inner_val_idx]).size != 2:
            raise SystemExit(
                f"ERROR: la validación interna del pliegue {fold_index + 1} no contiene ambas clases."
            )

        plan.append(
            {
                "fold": fold_index + 1,
                "repeat": fold_index // args.n_splits + 1,
                "outer_train": outer_train.astype(np.int64, copy=False),
                "fit": fit_idx.astype(np.int64, copy=False),
                "inner_val": inner_val_idx.astype(np.int64, copy=False),
                "outer_val": outer_val.astype(np.int64, copy=False),
            }
        )
    return plan


def split_fingerprint(plan: Sequence[Mapping[str, Any]], *, length: int = 16) -> str:
    digest = hashlib.sha256()
    for fold in plan:
        for name in ("fit", "inner_val", "outer_val"):
            array = np.ascontiguousarray(np.asarray(fold[name], dtype=np.int64))
            digest.update(name.encode("ascii"))
            digest.update(array.size.to_bytes(8, "little"))
            digest.update(array.tobytes())
    return digest.hexdigest()[:length]


# ---------------------------------------------------------------------------
# Enventanado y representaciones
# ---------------------------------------------------------------------------

def resolve_temporal_spec(
    args: argparse.Namespace,
    *,
    n_timepoints: int,
) -> tdha_data.WindowSpec | None:
    """Resuelve los argumentos del CLI manteniendo el legado 70/2."""

    explicit_temporal = any(
        value is not None
        for value in (
            args.window_tr,
            args.window_seconds,
            args.step_tr,
            args.step_seconds,
            args.overlap,
            args.gaussian_sigma,
        )
    ) or args.window_shape != "rectangular"

    if args.representation in ("static", "partial", "multiview"):
        if explicit_temporal:
            raise SystemExit(
                f"ERROR: la representación '{args.representation}' usa toda la serie y no "
                "acepta parámetros de ventana, paso, solapamiento o forma de ventana."
            )
        args.window = None
        args.step = None
        args.windowing_preset = "static"
        return None

    tr_seconds = (
        _positive_float("--tr-seconds", args.tr_seconds)
        if args.tr_seconds is not None
        else float(tdha_data.SITE_TR_SECONDS[args.site])
    )

    window_tr = args.window_tr
    window_seconds = args.window_seconds
    default_window = window_tr is None and window_seconds is None
    if default_window:
        window_tr = 70

    step_tr = args.step_tr
    step_seconds = args.step_seconds
    overlap = args.overlap
    default_step = step_tr is None and step_seconds is None and overlap is None
    if default_step:
        if window_seconds is not None:
            raise SystemExit(
                "ERROR: con --window-seconds debe especificar --step, "
                "--step-seconds o --overlap."
            )
        step_tr = 2  # compatibilidad con ``--window N`` del ejecutor histórico

    args.windowing_preset = (
        "legacy_70_2" if default_window and default_step else "custom"
    )

    try:
        spec = tdha_data.resolve_window_spec(
            tr_seconds=tr_seconds,
            window_tr=window_tr,
            window_seconds=window_seconds,
            step_tr=step_tr,
            step_seconds=step_seconds,
            overlap=overlap,
            shape=args.window_shape,
            fisher_z=args.fisher_z,
            gaussian_sigma=args.gaussian_sigma,
        )
        tdha_data.n_windows(n_timepoints, spec.window_tr, spec.step_tr)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: configuración temporal inválida: {exc}") from exc

    args.window = spec.window_tr  # claves históricas utilizadas por otros scripts
    args.step = spec.step_tr
    return spec


def static_diagnostics(n_timepoints: int, tr_seconds: float) -> dict[str, Any]:
    return {
        "mode": "static",
        "n_timepoints": int(n_timepoints),
        "n_windows": 1,
        "window_tr": int(n_timepoints),
        "step_tr": int(n_timepoints),
        "window_seconds": float(n_timepoints * tr_seconds),
        "step_seconds": float(n_timepoints * tr_seconds),
        "scan_seconds": float(n_timepoints * tr_seconds),
        "effective_overlap": 0.0,
        "unused_timepoints": 0,
        "window_fraction_of_scan": 1.0,
        "coverage_min": 1,
        "coverage_max": 1,
        "coverage_mean": 1.0,
        "median_adjacent_similarity": None,
    }


def build_representation(
    *,
    site: str,
    bold: np.ndarray,
    labels: np.ndarray,
    subjects: Sequence[Any],
    indices: np.ndarray,
    roi_key: str,
    args: argparse.Namespace,
    spec: tdha_data.WindowSpec | None,
    use_cache: bool,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Construye la entrada del modelo y los diagnósticos del enventanado."""

    del labels  # La transformación no depende de la etiqueta.
    n_timepoints = int(bold.shape[-1])
    tr_seconds = (
        float(args.tr_seconds)
        if args.tr_seconds is not None
        else float(tdha_data.SITE_TR_SECONDS[site])
    )

    if args.representation == "static":
        if use_cache:
            base = tdha_data.build_sequences_cached(
                site,
                bold,
                indices,
                n_timepoints,
                n_timepoints,
                roi_key,
                mode="static",
                fisher_z=args.fisher_z,
                constant_policy=args.constant_policy,
            )
        else:
            base = tdha_data.build_flat_static_connectivity(
                bold,
                indices,
                fisher_z=args.fisher_z,
                constant_policy=args.constant_policy,
            )
        diagnostics = static_diagnostics(n_timepoints, tr_seconds)
        return base, diagnostics, []

    if args.representation == "partial":
        base = tdha_data.build_flat_partial_connectivity(
            bold,
            indices,
            fisher_z=args.fisher_z,
            constant_policy=args.constant_policy,
        )
        diagnostics = static_diagnostics(n_timepoints, tr_seconds)
        diagnostics["connectivity"] = "partial_ledoit_wolf"
        return base, diagnostics, []

    if args.representation == "multiview":
        base = tdha_data.build_flat_multiview(
            bold,
            indices,
            fisher_z=args.fisher_z,
            constant_policy=args.constant_policy,
        )
        diagnostics = static_diagnostics(n_timepoints, tr_seconds)
        diagnostics["connectivity"] = "pearson+partial_multiview"
        diagnostics["n_channels"] = int(base.shape[1])
        return base, diagnostics, []

    if spec is None:  # salvaguarda de programación
        raise RuntimeError("Se requiere WindowSpec para representaciones dinámicas.")

    if use_cache:
        ordered = tdha_data.build_sequences_cached(
            site,
            bold,
            indices,
            spec.window_tr,
            spec.step_tr,
            roi_key,
            mode="dynamic",
            window_shape=spec.shape,
            gaussian_sigma=spec.gaussian_sigma,
            fisher_z=spec.fisher_z,
            constant_policy=args.constant_policy,
        )
    else:
        ordered = tdha_data.build_flat_sequences(
            bold,
            indices,
            spec.window_tr,
            spec.step_tr,
            window_shape=spec.shape,
            gaussian_sigma=spec.gaussian_sigma,
            fisher_z=spec.fisher_z,
            constant_policy=args.constant_policy,
        )

    diagnostics = tdha_data.windowing_diagnostics(
        n_timepoints,
        spec.window_tr,
        spec.step_tr,
        tr_seconds=spec.tr_seconds,
        sequences=ordered,
    )
    diagnostics["mode"] = "dynamic"
    diagnostics["window_shape"] = spec.shape
    diagnostics["fisher_z"] = bool(spec.fisher_z)
    diagnostics["gaussian_sigma"] = spec.gaussian_sigma
    warnings = tdha_data.methodological_warnings(diagnostics)

    if args.representation == "ordered":
        output = ordered
    elif args.representation == "permuted":
        output = tdha_data.permute_windows(
            ordered,
            subject_ids=subjects,
            seed=args.representation_seed,
        )
    elif args.representation == "mean":
        output = tdha_data.summarize_windows(ordered, statistics=("mean",))
    elif args.representation == "mean_std":
        output = tdha_data.summarize_windows(ordered, statistics=("mean", "std"))
    elif args.representation == "hybrid":
        static_flat = tdha_data.build_flat_static_connectivity(
            bold,
            indices,
            fisher_z=spec.fisher_z,
            constant_policy=args.constant_policy,
        )
        output = tdha_data.hybrid_summary(ordered, static_flat)
    else:  # pragma: no cover - argparse limita las opciones
        raise ValueError(f"Representación desconocida: {args.representation!r}.")

    return np.asarray(output, dtype=np.float32), diagnostics, warnings


# ---------------------------------------------------------------------------
# Entrenamiento y métricas
# ---------------------------------------------------------------------------

def evaluate(model: Any, X: np.ndarray, y: Sequence[int]) -> tuple[dict[str, float], np.ndarray]:
    """Calcula métricas con una sola pasada hacia adelante."""

    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    probabilities = np.asarray(model(X, training=False)).ravel().astype(np.float64)
    labels = np.asarray(y, dtype=np.int32).ravel()
    epsilon = 1e-7
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    loss = float(
        -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    )
    prediction = (probabilities >= 0.5).astype(np.int32)

    tp = int(((prediction == 1) & (labels == 1)).sum())
    tn = int(((prediction == 0) & (labels == 0)).sum())
    fp = int(((prediction == 1) & (labels == 0)).sum())
    fn = int(((prediction == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    metrics = {
        "loss": loss,
        "accuracy": float((tp + tn) / labels.size),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "f1_macro": float(
            f1_score(labels, prediction, average="macro", zero_division=0)
        ),
        "auc": float(roc_auc_score(labels, probabilities))
        if np.unique(labels).size > 1
        else float("nan"),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }
    return metrics, probabilities


def compile_model(model: Any, args: argparse.Namespace) -> Any:
    """Compila con las métricas necesarias para el historial por época."""

    import keras
    from keras import ops

    def bce(y_true, y_pred):
        # Entropía cruzada SIN el término de regularización, para que el early stopping
        # no siga a la penalización L2 (que baja al encoger los pesos). Alinea el rango
        # de las etiquetas (batch,) con la salida (batch, 1).
        y_true = ops.reshape(ops.cast(y_true, y_pred.dtype), ops.shape(y_pred))
        return keras.losses.binary_crossentropy(y_true, y_pred)

    optimizer_args: dict[str, Any] = {"learning_rate": args.lr}
    if args.clipnorm is not None:
        optimizer_args["clipnorm"] = args.clipnorm
    model.compile(
        optimizer=keras.optimizers.Adam(**optimizer_args),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), bce],
    )
    return model


def run_config(
    Xf: np.ndarray,
    y: np.ndarray,
    subjects: Sequence[Any],
    args: argparse.Namespace,
    outdir: Path,
    split_plan: Sequence[Mapping[str, Any]],
    subset_id: int | None = None,
) -> dict[str, float]:
    """Ejecuta la validación cruzada sobre una representación ya construida."""

    import keras
    from keras.callbacks import EarlyStopping

    Xf = np.asarray(Xf, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    if Xf.ndim != 3:
        raise ValueError("Xf debe tener forma (sujetos, ventanas, características).")
    if Xf.shape[0] != y.size or len(subjects) != y.size:
        raise ValueError("Xf, y y subjects deben contener el mismo número de sujetos.")
    if not np.isfinite(Xf).all():
        raise ValueError("La representación contiene NaN o infinitos.")

    n_subjects, n_windows, n_features = Xf.shape
    print(
        f" entrada: {n_subjects} sujetos · {n_windows} ventanas · "
        f"{n_features} características"
    )

    rows_train: list[dict[str, Any]] = []
    rows_val: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    start_time = time.time()

    for fold_zero, fold_data in enumerate(split_plan):
        outer_train = np.asarray(fold_data["outer_train"], dtype=np.int64)
        fit_idx = np.asarray(fold_data["fit"], dtype=np.int64)
        inner_val_idx = np.asarray(fold_data["inner_val"], dtype=np.int64)
        outer_val_idx = np.asarray(fold_data["outer_val"], dtype=np.int64)
        fold_number = int(fold_data["fold"])
        repeat = int(fold_data["repeat"])

        # Evita acumulación de grafos entre pliegues. La semilla se fija después
        # de limpiar la sesión para conservar una inicialización reproducible.
        keras.backend.clear_session()
        gc.collect()
        keras.utils.set_random_seed(args.seed * 1000 + fold_zero)

        class_weight = (
            compute_class_weights(y[fit_idx]) if args.class_weight else None
        )
        if args.verbose and class_weight is not None:
            rounded = {key: round(value, 4) for key, value in class_weight.items()}
            print(f" pesos pliegue {fold_number}: {rounded}")

        model = compile_model(
            kerasmodels.build(
                args.model,
                n_windows,
                n_features,
                **args._model_kwargs,
            ),
            args,
        )

        history = model.fit(
            Xf[fit_idx],
            y[fit_idx],
            validation_data=(Xf[inner_val_idx], y[inner_val_idx]),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_bce",
                    mode="min",
                    patience=args.patience,
                    min_delta=1e-5,
                    start_from_epoch=args.start_from_epoch,
                    restore_best_weights=True,
                )
            ],
        )

        n_epochs = len(history.history["loss"])
        best_epoch = int(np.argmin(history.history["val_bce"])) + 1

        # El pliegue externo se utiliza aquí por primera vez.
        train_metrics, _ = evaluate(model, Xf[outer_train], y[outer_train])
        val_metrics, probabilities = evaluate(
            model,
            Xf[outer_val_idx],
            y[outer_val_idx],
        )

        metadata: dict[str, Any] = {
            "fold": fold_number,
            "repeat": repeat,
            "n_epochs": n_epochs,
            "best_epoch": best_epoch,
            "n_fit": int(fit_idx.size),
            "n_inner_val": int(inner_val_idx.size),
            "n_outer_val": int(outer_val_idx.size),
            "class_weight_0": class_weight.get(0) if class_weight else None,
            "class_weight_1": class_weight.get(1) if class_weight else None,
        }
        rows_train.append({**metadata, **train_metrics})
        rows_val.append({**metadata, **val_metrics})

        for epoch, loss in enumerate(history.history["loss"], start=1):
            history_rows.append(
                {
                    "fold": fold_number,
                    "repeat": repeat,
                    "epoch": epoch,
                    "loss": float(loss),
                    "inner_val_loss": float(history.history["val_loss"][epoch - 1]),
                    "accuracy": float(history.history["accuracy"][epoch - 1]),
                    "inner_val_accuracy": float(
                        history.history["val_accuracy"][epoch - 1]
                    ),
                }
            )

        for subject_index, probability in zip(outer_val_idx, probabilities):
            prediction_rows.append(
                {
                    "fold": fold_number,
                    "repeat": repeat,
                    "subject": int(subject_index),  # compatibilidad histórica
                    "subject_id": str(subjects[int(subject_index)]),
                    "y_true": int(y[subject_index]),
                    "y_prob": float(probability),
                }
            )

        for split_name, split_indices in (
            ("fit", fit_idx),
            ("inner_val", inner_val_idx),
            ("outer_val", outer_val_idx),
        ):
            for subject_index in split_indices:
                fold_rows.append(
                    {
                        "fold": fold_number,
                        "repeat": repeat,
                        "subject": int(subject_index),
                        "subject_id": str(subjects[int(subject_index)]),
                        "split": split_name,
                    }
                )

        if args.verbose:
            print(
                f" pliegue {fold_number:3d}/{len(split_plan)} "
                f"train acc={train_metrics['accuracy']:.4f} "
                f"val acc={val_metrics['accuracy']:.4f} "
                f"val F1m={val_metrics['f1_macro']:.4f} "
                f"(época {best_epoch}/{n_epochs})",
                flush=True,
            )

        del model, history
        gc.collect()

    suffix = "" if subset_id is None else f"_set{subset_id:02d}"
    pd.DataFrame(rows_train).to_csv(
        outdir / f"metrics_train{suffix}.csv", index=False
    )
    pd.DataFrame(rows_val).to_csv(outdir / f"metrics_val{suffix}.csv", index=False)
    pd.DataFrame(history_rows).to_csv(outdir / f"history{suffix}.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        outdir / f"predictions_val{suffix}.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(outdir / f"folds{suffix}.csv", index=False)

    train_frame = pd.DataFrame(rows_train)
    val_frame = pd.DataFrame(rows_val)
    elapsed = time.time() - start_time
    print(
        f" train acc {train_frame.accuracy.mean() * 100:.2f} ± "
        f"{train_frame.accuracy.std() * 100:.2f} | "
        f"val acc {val_frame.accuracy.mean() * 100:.2f} ± "
        f"{val_frame.accuracy.std() * 100:.2f} | "
        f"val F1m {val_frame.f1_macro.mean() * 100:.2f} ± "
        f"{val_frame.f1_macro.std() * 100:.2f} | {elapsed:.0f} s"
    )
    return {
        "train_acc": float(train_frame.accuracy.mean()),
        "val_acc": float(val_frame.accuracy.mean()),
        "val_f1_macro": float(val_frame.f1_macro.mean()),
        "val_auc": float(val_frame.auc.mean()),
    }


# ---------------------------------------------------------------------------
# CLI y configuración
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data_group = parser.add_argument_group("datos y representación")
    data_group.add_argument("--site", default="NYU", choices=tdha_data.SITES)
    data_group.add_argument(
        "--roi-set",
        default="12",
        help="subconjunto definido en data/atlas/roi_sets.json: 12, 18, 39, 116",
    )

    window_group = data_group.add_mutually_exclusive_group()
    window_group.add_argument(
        "--window",
        dest="window_tr",
        type=int,
        default=None,
        help="longitud de ventana en TR; sin argumentos se conserva 70 TR",
    )
    window_group.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="longitud física de la ventana en segundos",
    )

    step_group = data_group.add_mutually_exclusive_group()
    step_group.add_argument(
        "--step",
        dest="step_tr",
        type=int,
        default=None,
        help="desplazamiento en TR; con --window y sin paso se conserva 2 TR",
    )
    step_group.add_argument(
        "--step-seconds",
        type=float,
        default=None,
        help="desplazamiento físico entre ventanas, en segundos",
    )
    step_group.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="fracción de solapamiento en [0, 1), por ejemplo 0.75",
    )

    data_group.add_argument(
        "--tr-seconds",
        type=float,
        default=None,
        help="sobrescribe explícitamente el TR documentado del sitio",
    )
    data_group.add_argument(
        "--window-shape",
        choices=("rectangular", "gaussian"),
        default="rectangular",
    )
    data_group.add_argument(
        "--gaussian-sigma",
        type=float,
        default=None,
        help="sigma de la ventana gaussiana en TR; por defecto window/6",
    )
    data_group.add_argument("--fisher-z", action="store_true")
    data_group.add_argument(
        "--constant-policy",
        choices=("zero", "raise"),
        default="zero",
        help="tratamiento de ROIs constantes dentro de una ventana",
    )
    data_group.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="ordered",
        help="ordered, permuted, mean, mean_std o static",
    )
    data_group.add_argument(
        "--representation-seed",
        type=int,
        default=None,
        help="semilla de la permutación temporal; por defecto usa --seed",
    )
    data_group.add_argument("--out", default=str(DEFAULT_OUT_DIR))

    architecture_group = parser.add_argument_group("arquitectura")
    architecture_group.add_argument(
        "--model",
        default="lstm",
        help=f"una de: {', '.join(kerasmodels.available())}",
    )
    architecture_group.add_argument(
        "--model-arg",
        nargs="*",
        metavar="CLAVE=VALOR",
        help="hiperparámetros, p. ej. units=128 dropout=0.2",
    )

    training_group = parser.add_argument_group("entrenamiento")
    training_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="fija particiones e inicialización; use el mismo valor al comparar",
    )
    training_group.add_argument("--n-splits", type=int, default=10)
    training_group.add_argument("--n-repeats", type=int, default=5)
    training_group.add_argument("--lr", type=float, default=1e-4)
    training_group.add_argument("--batch-size", type=int, default=8)
    training_group.add_argument("--epochs", type=int, default=150)
    training_group.add_argument(
        "--patience",
        type=int,
        default=25,
        help="épocas sin mejora antes de parar; restore_best_weights recupera la mejor",
    )
    training_group.add_argument("--clipnorm", type=float, default=None)
    training_group.add_argument("--start-from-epoch", type=int, default=0)
    training_group.add_argument(
        "--inner-val-frac",
        type=float,
        default=0.15,
        help="fracción del entrenamiento externo reservada para seleccionar época",
    )
    training_group.add_argument(
        "--class-weight",
        action="store_true",
        help="calcula pesos por pliegue usando únicamente el subconjunto fit",
    )

    anatomy_group = parser.add_argument_group("control anatómico")
    anatomy_group.add_argument(
        "--random-subset",
        type=int,
        default=None,
        help="muestrea N ROIs dentro de --roi-set; use --roi-set 116 para todo el atlas",
    )
    anatomy_group.add_argument("--n-random-sets", type=int, default=20)
    anatomy_group.add_argument("--exclude-roi-set", default=None)

    execution_group = parser.add_argument_group("ejecución")
    execution_group.add_argument("--deterministic", action="store_true")
    execution_group.add_argument(
        "--mixed-precision",
        action="store_true",
        help="activa mixed_float16 en GPU; acelera las configuraciones grandes "
        "(39/116 ROIs, transformer, cnn1d). Cambia los decimales de las métricas.",
    )
    execution_group.add_argument("--tag", default=None)
    execution_group.add_argument("--overwrite", action="store_true")
    execution_group.add_argument("--dry-run", action="store_true")
    execution_group.add_argument("--list-models", action="store_true")
    execution_group.add_argument("--list-roi-sets", action="store_true")
    execution_group.add_argument("--verbose", action="store_true")
    return parser


def _window_identity(spec: tdha_data.WindowSpec | None) -> dict[str, Any]:
    if spec is None:
        return {
            "mode": "static",
            "window_tr": None,
            "step_tr": None,
            "window_seconds": None,
            "step_seconds": None,
            "requested_window_seconds": None,
            "requested_step_seconds": None,
            "requested_overlap": None,
            "effective_overlap": None,
            "shape": None,
            "gaussian_sigma": None,
        }
    return {"mode": "dynamic", **spec.to_dict()}


def make_run_id(
    args: argparse.Namespace,
    spec: tdha_data.WindowSpec | None,
    digest: str,
) -> str:
    parts = [sanitize_component(args.site), f"rois{sanitize_component(args.roi_set)}"]
    if spec is None:
        # 'static' o 'partial': ambos usan toda la serie (sin ventana).
        parts.append(sanitize_component(args.representation))
    else:
        parts.append(f"w{spec.window_tr}s{spec.step_tr}")
        if spec.shape != "rectangular":
            parts.append(spec.shape)
        if spec.fisher_z:
            parts.append("fisher")
        if args.representation != "ordered":
            parts.append(args.representation)
    parts.append(sanitize_component(args.model))
    if args.random_subset:
        parts.append(f"rand{args.random_subset}")
    if args.tag:
        parts.append(sanitize_component(args.tag))
    return "_".join(parts) + f"_{digest}"


def _code_hash(module: Any) -> str:
    path = getattr(module, "__file__", None)
    return file_hash(path) if path and Path(path).exists() else "desconocido"


def write_config(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _mean_sd_pct(series: "pd.Series") -> str:
    mean = float(series.mean()) * 100
    sd = float(series.std(ddof=1)) * 100 if series.notna().sum() > 1 else float("nan")
    return f"{mean:.2f} ± {sd:.2f} %"


def write_run_summary(
    outdir: Path,
    config: Mapping[str, Any],
    *,
    suffix: str = "",
    is_random: bool = False,
) -> None:
    """Escribe un resumen legible de la corrida, derivado del config.

    Es una vista humana para hojear ``results/runs`` sin leer el ``config.json``
    completo; NO es una fuente de verdad: se regenera a partir de él.
    """

    arch = config.get("arch") or {}
    arch_str = ", ".join(f"{k}={v}" for k, v in arch.items()) or "(por defecto)"
    balance = config.get("class_balance") or {}
    control = balance.get(0, balance.get("0", "?"))
    tdah = balance.get(1, balance.get("1", "?"))

    window = config.get("windowing") or {}
    representation = config.get("representation")
    if representation == "static" or window.get("mode") == "static":
        ventana = "estática — una matriz sobre toda la serie, sin ventanas"
    else:
        overlap = window.get("effective_overlap")
        overlap_str = f"{overlap:.0%}" if isinstance(overlap, (int, float)) else "?"
        ventana = (
            f"{window.get('window_tr')} TR / {window.get('window_seconds')} s · "
            f"paso {window.get('step_tr')} TR / {window.get('step_seconds')} s · "
            f"solape {overlap_str} · {config.get('n_windows')} ventanas · "
            f"{window.get('shape', 'rectangular')}"
        )

    n_eval = int(config.get("n_splits", 0)) * int(config.get("n_repeats", 0))
    lines = [
        f"# {config.get('run_id')}",
        "",
        f"Vista legible generada desde `config.json` (la fuente de verdad). "
        f"Timestamp: {config.get('timestamp')}.",
        "",
        "## Configuración",
        "",
        f"- **Sitio**: {config.get('site')} · **ROIs**: {config.get('roi_set')} "
        f"(n={config.get('n_rois')}) · **Sujetos**: {config.get('n_subjects')} "
        f"(control/TDAH: {control}/{tdah})",
        f"- **Modelo**: `{config.get('model')}` — {arch_str}",
        f"- **Representación**: {representation} · **Ventana**: {ventana}",
        f"- **Fisher z**: {'sí' if config.get('fisher_z') else 'no'} · "
        f"**Precisión mixta**: {'sí' if config.get('mixed_precision') else 'no'}",
        f"- **Validación**: {config.get('n_splits')}×{config.get('n_repeats')} = "
        f"{n_eval} evaluaciones externas · semilla {config.get('seed')} · "
        f"class_weight: {'sí' if config.get('class_weight') else 'no'}",
        f"- **Entrenamiento**: lr={config.get('lr')}, batch={config.get('batch_size')}, "
        f"epochs={config.get('epochs')}, patience={config.get('patience')}",
        "",
    ]

    if is_random:
        lines += [
            "## Resultados",
            "",
            f"Corrida de subconjuntos aleatorios: {config.get('n_random_sets')} conjuntos "
            f"de {config.get('random_subset')} ROIs. Ver `random_subsets_summary.csv`.",
            "",
        ]
    else:
        val_path = outdir / f"metrics_val{suffix}.csv"
        if val_path.exists():
            val = pd.read_csv(val_path)
            lines.append(
                f"## Resultados — validación externa (media ± sd sobre {len(val)} pliegues)"
            )
            lines.append("")
            for label, col in [
                ("Accuracy", "accuracy"),
                ("F1-macro", "f1_macro"),
                ("AUC", "auc"),
                ("Balanced acc.", "balanced_accuracy"),
            ]:
                if col in val:
                    lines.append(f"- **{label}**: {_mean_sd_pct(val[col])}")
            train_path = outdir / f"metrics_train{suffix}.csv"
            if train_path.exists():
                train = pd.read_csv(train_path)
                if "accuracy" in val and "accuracy" in train:
                    gap = (train["accuracy"].mean() - val["accuracy"].mean()) * 100
                    lines.append(f"- **Brecha train−val (accuracy)**: {gap:.2f} pp")
            if "best_epoch" in val:
                lines.append(
                    f"- **Época elegida (mediana)**: {val['best_epoch'].median():.0f}"
                )
            lines.append("")

    lines += [
        "## Reproducir",
        "",
        "```",
        str(config.get("command", "")),
        "```",
        "",
    ]
    (outdir / f"resumen{suffix}.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> str | None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for name in kerasmodels.available():
            print(f" {name:14s} {kerasmodels.defaults(name)}")
        return None
    if args.list_roi_sets:
        for key, value in sorted(
            tdha_data.load_roi_sets().items(), key=lambda item: int(item[0])
        ):
            print(f" {key:>4s} {value['n']:3d} ROIs {value['description']}")
        return None

    if args.model not in kerasmodels.REGISTRY:
        raise SystemExit(
            f"ERROR: arquitectura {args.model!r} desconocida. "
            f"Disponibles: {', '.join(kerasmodels.available())}"
        )
    args._model_kwargs = parse_model_args(args.model_arg)
    kerasmodels.validate_args(args.model, args._model_kwargs)

    args.seed = _positive_int("--seed", args.seed, minimum=0)
    args.representation_seed = (
        args.seed
        if args.representation_seed is None
        else _positive_int("--representation-seed", args.representation_seed, minimum=0)
    )
    if args.random_subset is not None:
        args.random_subset = _positive_int("--random-subset", args.random_subset, minimum=2)
        args.n_random_sets = _positive_int("--n-random-sets", args.n_random_sets)
    elif args.exclude_roi_set is not None:
        raise SystemExit("ERROR: --exclude-roi-set requiere --random-subset.")

    if args.deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            import tensorflow as tf

            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # pragma: no cover - depende del entorno
            print(f" AVISO: no se pudo activar determinismo: {exc}")

    if args.mixed_precision:
        # Debe fijarse antes de construir cualquier modelo. La salida de cada
        # arquitectura ya declara dtype="float32", así que la pérdida y la sigmoide
        # se mantienen estables; solo cambia la acumulación interna en float16.
        try:
            import keras
            import tensorflow as tf

            if tf.config.list_physical_devices("GPU"):
                keras.mixed_precision.set_global_policy("mixed_float16")
                print(" precisión mixta activada (mixed_float16).")
            else:
                print(" AVISO: --mixed-precision ignorado; no hay GPU disponible.")
                args.mixed_precision = False
        except Exception as exc:  # pragma: no cover - depende del entorno
            print(f" AVISO: no se pudo activar precisión mixta: {exc}")
            args.mixed_precision = False

    bold_path = tdha_data.BOLD_DIR / f"{args.site}.joblib"
    payload = tdha_data.load_bold(args.site)
    bold = payload["bold"]
    labels = payload["labels"]
    subjects = payload["subjects"]
    roi_idx = tdha_data.roi_indices(args.roi_set)
    roi_idx = tdha_data.validate_indices(roi_idx, bold.shape[1])

    validate_training_args(args, labels)
    spec = resolve_temporal_spec(args, n_timepoints=bold.shape[-1])
    split_plan = build_split_plan(labels, args)
    split_hash = split_fingerprint(split_plan)

    if spec is None:
        basic_diagnostics = static_diagnostics(
            bold.shape[-1],
            args.tr_seconds
            if args.tr_seconds is not None
            else tdha_data.SITE_TR_SECONDS[args.site],
        )
        basic_warnings: list[str] = []
        # 'multiview' apila 2 matrices (Pearson + parcial) como canales.
        n_model_windows = 2 if args.representation == "multiview" else 1
    else:
        basic_diagnostics = tdha_data.windowing_diagnostics(
            bold.shape[-1],
            spec.window_tr,
            spec.step_tr,
            tr_seconds=spec.tr_seconds,
        )
        basic_diagnostics["mode"] = "dynamic"
        basic_diagnostics["window_shape"] = spec.shape
        basic_diagnostics["fisher_z"] = spec.fisher_z
        basic_diagnostics["gaussian_sigma"] = spec.gaussian_sigma
        basic_warnings = tdha_data.methodological_warnings(basic_diagnostics)
        n_model_windows = (
            1
            if args.representation in {"mean", "mean_std", "hybrid"}
            else basic_diagnostics["n_windows"]
        )

    atlas_path = tdha_data.ATLAS_DIR / "roi_sets.json"
    window_identity = _window_identity(spec)
    identity: dict[str, Any] = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "site": args.site,
        "roi_set": str(args.roi_set),
        "roi_indices_hash": indices_hash(roi_idx),
        "representation": args.representation,
        "representation_seed": args.representation_seed
        if args.representation == "permuted"
        else None,
        "windowing": window_identity,
        "fisher_z": bool(args.fisher_z),
        "constant_policy": args.constant_policy,
        "model": args.model,
        "arch": {**kerasmodels.defaults(args.model), **args._model_kwargs},
        "seed": args.seed,
        "split_fingerprint": split_hash,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "clipnorm": args.clipnorm,
        "inner_val_frac": args.inner_val_frac,
        "class_weight": bool(args.class_weight),
        "start_from_epoch": args.start_from_epoch,
        "random_subset": args.random_subset,
        "n_random_sets": args.n_random_sets if args.random_subset else None,
        "exclude_roi_set": args.exclude_roi_set,
        "deterministic": bool(args.deterministic),
        "mixed_precision": bool(args.mixed_precision),
        "bold_hash": file_hash(bold_path),
        "atlas_hash": file_hash(atlas_path),
        "data_code_hash": _code_hash(tdha_data),
        "runner_code_hash": file_hash(__file__),
    }
    digest = config_hash(identity)
    run_id = make_run_id(args, spec, digest)
    git = git_info()

    config: dict[str, Any] = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "run_id": run_id,
        "config_hash": digest,
        "site": args.site,
        "roi_set": str(args.roi_set),
        "n_subjects": int(bold.shape[0]),
        "n_timepoints": int(bold.shape[-1]),
        "n_rois": int(args.random_subset or roi_idx.size),
        "n_windows": int(n_model_windows),
        # Claves históricas que consume compile_results.py.
        "window": spec.window_tr if spec is not None else None,
        "step": spec.step_tr if spec is not None else None,
        "representation": args.representation,
        "windowing_preset": getattr(args, "windowing_preset", "static"),
        "windowing": window_identity,
        "windowing_diagnostics": basic_diagnostics,
        "methodological_warnings": basic_warnings,
        "class_balance": {
            int(key): int(value)
            for key, value in zip(*np.unique(labels, return_counts=True))
        },
        **identity,
        "git": git,
        "env": env_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": "run_experiment.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
    }

    outdir = Path(args.out) / run_id
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"\n corrida: {run_id}")

    completed = list(outdir.glob("metrics_val*.csv"))
    if completed and not args.overwrite and not args.dry_run:
        raise SystemExit(
            f"\nESTA_CONFIGURACION_YA_SE_EJECUTO: {outdir}\n"
            "Use --overwrite para repetirla o --tag para distinguirla.\n"
        )
    if (outdir / "config.json").exists() and not completed and not args.dry_run:
        print(
            "\n AVISO: existe una corrida incompleta en esta carpeta; "
            "se rehace desde cero.\n"
        )
    if git["clean"] is False:
        print(
            "\n AVISO: el árbol de git tiene cambios sin confirmar. La corrida "
            "queda identificada por hashes de código, pero debe hacer commit para "
            "que otras personas puedan reconstruirla.\n"
        )

    outer_val_approx = len(labels) // args.n_splits
    inner_val_approx = int((len(labels) - outer_val_approx) * args.inner_val_frac)
    if outer_val_approx < 10 or inner_val_approx < 12:
        print(
            f"\n AVISO: pliegues pequeños — validación externa ≈ {outer_val_approx} "
            f"sujetos, validación interna ≈ {inner_val_approx}. Las métricas por "
            "pliegue serán inestables.\n"
        )
    for warning in basic_warnings:
        print(f"\n AVISO DE ENVENTANADO: {warning}\n")

    if args.dry_run:
        print(
            f"\ndry-run correcto: {len(split_plan)} particiones completas, "
            f"huella {split_hash}."
        )
        print(
            "Dos corridas tienen particiones pareadas cuando coinciden sitio, "
            "etiquetas, semilla, n_splits, n_repeats y esta huella."
        )
        return run_id

    outdir.mkdir(parents=True, exist_ok=True)
    # El config inicial permite identificar una corrida interrumpida.
    write_config(outdir / "config.json", config)

    if args.random_subset:
        # El conjunto indicado por --roi-set define explícitamente el universo
        # de muestreo. Con --roi-set 116 se conserva el comportamiento histórico.
        pool = roi_idx.copy()
        if args.exclude_roi_set:
            excluded = tdha_data.roi_indices(args.exclude_roi_set)
            pool = np.setdiff1d(pool, excluded)
            print(
                f" muestreando de {len(pool)} ROIs (excluidos {len(excluded)} del "
                f"subconjunto {args.exclude_roi_set!r})"
            )
        if pool.size < args.random_subset:
            raise SystemExit("ERROR: quedan menos ROIs que los solicitados.")

        rng = np.random.default_rng(args.seed)
        summary: list[dict[str, Any]] = []
        for subset_zero in range(args.n_random_sets):
            subset = np.sort(
                rng.choice(pool, size=args.random_subset, replace=False)
            )
            subset_number = subset_zero + 1
            print(
                f"\nsubconjunto {subset_number}/{args.n_random_sets}: "
                f"{subset.tolist()}",
                flush=True,
            )
            Xf, diagnostics, warnings = build_representation(
                site=args.site,
                bold=bold,
                labels=labels,
                subjects=subjects,
                indices=subset,
                roi_key=f"random_{subset_number}_{indices_hash(subset)}",
                args=args,
                spec=spec,
                use_cache=False,
            )
            for warning in warnings:
                print(f" AVISO DE ENVENTANADO: {warning}")
            result = run_config(
                Xf,
                labels,
                subjects,
                args,
                outdir,
                split_plan,
                subset_id=subset_number,
            )
            summary.append(
                {
                    "set": subset_number,
                    "rois": json.dumps(subset.tolist()),
                    "roi_indices_hash": indices_hash(subset),
                    "n_model_windows": int(Xf.shape[1]),
                    "n_features": int(Xf.shape[2]),
                    "median_adjacent_similarity": diagnostics.get(
                        "median_adjacent_similarity"
                    ),
                    "warnings": " | ".join(warnings),
                    **result,
                }
            )
            del Xf
            gc.collect()

        pd.DataFrame(summary).to_csv(
            outdir / "random_subsets_summary.csv", index=False
        )
        config["random_subsets_summary"] = "random_subsets_summary.csv"
        write_config(outdir / "config.json", config)
        write_run_summary(outdir, config, is_random=True)

        accuracies = [row["val_acc"] for row in summary]
        print(
            f"\n{len(accuracies)} subconjuntos aleatorios de "
            f"{args.random_subset} ROIs: val acc media "
            f"{np.mean(accuracies) * 100:.2f}, rango "
            f"[{min(accuracies) * 100:.2f}, {max(accuracies) * 100:.2f}]"
        )
    else:
        print(" construyendo representación de conectividad…", flush=True)
        Xf, diagnostics, warnings = build_representation(
            site=args.site,
            bold=bold,
            labels=labels,
            subjects=subjects,
            indices=roi_idx,
            roi_key=str(args.roi_set),
            args=args,
            spec=spec,
            use_cache=True,
        )
        config["windowing_diagnostics"] = diagnostics
        config["methodological_warnings"] = warnings
        config["n_windows"] = int(Xf.shape[1])
        config["n_features"] = int(Xf.shape[2])
        config["input_shape"] = [int(Xf.shape[1]), int(Xf.shape[2])]
        write_config(outdir / "config.json", config)

        for warning in warnings:
            print(f" AVISO DE ENVENTANADO: {warning}")
        run_config(Xf, labels, subjects, args, outdir, split_plan)
        write_run_summary(outdir, config)

    print(f"\nResultados en {outdir}")
    return run_id


if __name__ == "__main__":
    main()
