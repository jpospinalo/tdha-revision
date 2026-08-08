#!/usr/bin/env python3
"""Campaña LOSO estática multisitio (``loso_static_v1``).

Ruta aditiva y aislada. NO modifica ``run_experiment.py``, ``run_baseline_ml.py``,
``data.py`` ni ninguna arquitectura de ``kerasmodels/``. Reutiliza sin cambiarlas
las funciones históricas ya validadas de ``run_experiment.py`` (``file_hash``,
``indices_hash``, ``config_hash``, ``split_fingerprint``, ``env_info``,
``git_info``, ``compile_model``, ``evaluate``) y la construcción de conectividad
estática de ``data.py`` (``build_flat_static_connectivity``).

Diseño científico (ver ``PLAN_FINAL_LOSO_STATIC_V1_IA_REVISADO.md``):

- 4 sitios held-out (NYU, Peking, NeuroIMAGE, OHSU) x 2 ROI sets (12, 116) x
  2 familias de modelo (BrainNetCNN, regresión logística L2).
- Solo conectividad estática (``fisher_z=False``, ``constant_policy="zero"``).
- Un único split (fit/inner_val/test) por sitio held-out, idéntico para las
  dos representaciones ROI y los cinco seeds de BrainNetCNN.
- Sin harmonización, sin ponderación de clase/sitio, sin ajuste de hiperparámetros.
- La API de entrenamiento nunca recibe ``X_test``/``y_test``: ``train_brainnetcnn``
  y ``fit_logistic`` solo ven fit/inner_val; la evaluación held-out ocurre en una
  llamada aparte (``evaluate_heldout``), después de que el modelo ya quedó fijo.

Uso
---
    python run_loso.py --held-out-site NYU --roi-set 12 --model brainnetcnn --model-seed 42
    python run_loso.py --held-out-site NYU --roi-set 12 --model logreg
    python run_loso.py --held-out-site NYU --roi-set 12 --model brainnetcnn --model-seed 42 --dry-run
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as tdha_data  # noqa: E402
import kerasmodels  # noqa: E402
from run_experiment import (  # noqa: E402 - reutilización explícita (Sección 21 del plan)
    compile_model as _re_compile_model,
    config_hash,
    env_info,
    evaluate as re_evaluate,
    file_hash,
    git_info,
    indices_hash,
    split_fingerprint as re_split_fingerprint,
)

# ---------------------------------------------------------------------------
# Constantes congeladas de la campaña (Secciones 0, 6, 12, 19, 20, 24, 25)
# ---------------------------------------------------------------------------

CAMPAIGN_ID = "loso_static_v1"
CONFIG_SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent
BOLD_DIR = REPO_ROOT / "data" / "bold"
ATLAS_DIR = REPO_ROOT / "data" / "atlas"
FORMAL_OUTPUT_ROOT = REPO_ROOT / "results" / "loso"
DESIGN_DIR = FORMAL_OUTPUT_ROOT / "_design"
DESIGN_JSON_PATH = DESIGN_DIR / "loso_static_v1_design.json"
SPLIT_MANIFEST_PATH = DESIGN_DIR / "loso_static_v1_splits.csv"

# Orden maestro canónico (Sección 7): fija la concatenación de features y las
# filas de la master participant table. No reordenar.
SITES = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]
ROI_SETS = ["12", "116"]
MODELS = ("brainnetcnn", "logreg")
BNN_SEEDS = [42, 43, 44, 45, 46]

SPLIT_SEED = 42
INNER_VAL_FRAC = 0.15
BOOTSTRAP_SEED = 42  # usado únicamente por el analyzer (Sección 8/24), no aquí.

# Representación: exclusivamente estática (Sección 6.4).
FISHER_Z = False
CONSTANT_POLICY = "zero"
N_WINDOWS = 1

# BrainNetCNN: arquitectura congelada (Sección 19).
BNN_ARCH_KWARGS: dict[str, Any] = {
    "e2e": 4,
    "e2n": 8,
    "dense": 8,
    "dropout": 0.7,
    "leaky": 0.33,
    "l2_reg": 0.05,
    "inter_dropout": 0.6,
}
EXPECTED_BNN_PARAM_COUNT = {"12": 1361, "116": 12177}

# BrainNetCNN: entrenamiento congelado (Sección 20).
BNN_TRAIN_CONFIG: dict[str, Any] = {
    "optimizer": "adam",
    "lr": 0.0001,
    "clipnorm": None,
    "batch_size": 32,
    "epochs": 300,
    "shuffle": True,
    "mixed_precision": False,
    "patience": 25,
    "inner_val_frac": INNER_VAL_FRAC,
    "early_stopping_monitor": "val_loss",
    "early_stopping_min_delta": 1e-5,
    "start_from_epoch": 0,
    "class_weight": False,
}

# Regresión logística: configuración congelada (Sección 25).
LOGREG_CONFIG: dict[str, Any] = {
    "penalty": "l2",
    "C": 1.0,
    "class_weight": None,
    "solver": "lbfgs",
    "max_iter": 2000,
}

# Tamaños obligatorios (Secciones 4, 5, 10.2) — guardarraíles de arranque.
EXPECTED_COHORT = {
    "NYU": {"n": 177, "control": 87, "adhd": 90},
    "Peking": {"n": 183, "control": 109, "adhd": 74},
    "NeuroIMAGE": {"n": 39, "control": 22, "adhd": 17},
    "OHSU": {"n": 66, "control": 38, "adhd": 28},
}
EXPECTED_TOTAL = {"n": 465, "control": 256, "adhd": 209}
EXPECTED_ROTATION_SIZES = {
    "NYU": {"fit": 244, "inner_val": 44, "test": 177},
    "Peking": {"fit": 239, "inner_val": 43, "test": 183},
    "NeuroIMAGE": {"fit": 362, "inner_val": 64, "test": 39},
    "OHSU": {"fit": 339, "inner_val": 60, "test": 66},
}
EXPECTED_FEATURES = {"12": 66, "116": 6670}


def _sha_trunc(data: bytes, *, length: int = 16) -> str:
    """SHA-256 truncado, misma convención que ``file_hash``/``indices_hash``."""

    return hashlib.sha256(data).hexdigest()[:length]


def _sha_file(path: Path, *, length: int = 16) -> str:
    return file_hash(path, length=length)


# ---------------------------------------------------------------------------
# Master participant table y features (Sección 7)
# ---------------------------------------------------------------------------


def load_site_payloads() -> dict[str, dict[str, Any]]:
    """Carga (validada, cacheada por ``data.py``) los cuatro sitios."""

    return {site: tdha_data.load_bold(site) for site in SITES}


def build_master_table(payloads: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Tabla en memoria: global_index, site, site_row_index, subject_id,
    subject_key, y_true (Sección 7.1). Orden: SITES canónico, y dentro de cada
    sitio, el orden de ``payload["subjects"]`` (sin reordenar)."""

    rows: list[dict[str, Any]] = []
    global_index = 0
    for site in SITES:
        payload = payloads[site]
        subjects = payload["subjects"]
        labels = payload["labels"]
        for site_row_index, (subject_id, y_true) in enumerate(zip(subjects, labels)):
            rows.append(
                {
                    "global_index": global_index,
                    "site": site,
                    "site_row_index": site_row_index,
                    "subject_id": str(subject_id),
                    "subject_key": f"{site}::{subject_id}",
                    "y_true": int(y_true),
                }
            )
            global_index += 1

    master = pd.DataFrame(rows)
    _validate_master_table(master)
    return master


def _validate_master_table(master: pd.DataFrame) -> None:
    if len(master) != EXPECTED_TOTAL["n"]:
        raise SystemExit(
            f"STOP: master table tiene {len(master)} filas; se esperaban "
            f"{EXPECTED_TOTAL['n']}."
        )
    if master["subject_key"].nunique() != len(master):
        raise SystemExit("STOP: subject_key no es único en la master table.")
    if list(master["global_index"]) != list(range(len(master))):
        raise SystemExit("STOP: global_index no es 0..N-1 secuencial.")
    control = int((master["y_true"] == 0).sum())
    adhd = int((master["y_true"] == 1).sum())
    if control != EXPECTED_TOTAL["control"] or adhd != EXPECTED_TOTAL["adhd"]:
        raise SystemExit(
            f"STOP: cohorte total {control}/{adhd} (control/adhd) no coincide "
            f"con {EXPECTED_TOTAL['control']}/{EXPECTED_TOTAL['adhd']}."
        )
    for site, expected in EXPECTED_COHORT.items():
        sub = master[master["site"] == site]
        n = len(sub)
        c0 = int((sub["y_true"] == 0).sum())
        c1 = int((sub["y_true"] == 1).sum())
        if (n, c0, c1) != (expected["n"], expected["control"], expected["adhd"]):
            raise SystemExit(
                f"STOP: cohorte de {site} es n={n} control={c0} adhd={c1}; "
                f"se esperaba n={expected['n']} control={expected['control']} "
                f"adhd={expected['adhd']}."
            )


def compute_site_static_fc(
    payload: Mapping[str, Any], roi_idx: np.ndarray
) -> np.ndarray:
    """Conectividad estática de un sitio, ``(n_site, n_features)`` float32.

    Reutiliza exclusivamente ``tdha_data.build_flat_static_connectivity`` con
    ``fisher_z=False`` y ``constant_policy="zero"`` (Sección 6.4). No existe
    otra implementación de Pearson en este módulo.
    """

    fc = tdha_data.build_flat_static_connectivity(
        payload["bold"],
        roi_idx,
        fisher_z=FISHER_Z,
        constant_policy=CONSTANT_POLICY,
    )
    if fc.ndim != 3 or fc.shape[1] != 1:
        raise ValueError(f"forma inesperada de FC estática: {fc.shape}")
    return np.asarray(fc[:, 0, :], dtype=np.float32)


def build_feature_matrix(
    roi_set: str,
    payloads: Mapping[str, Mapping[str, Any]],
    master: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, str]]:
    """Concatena FC estática por sitio, en orden canónico, y devuelve también
    los 8 hashes por sitio (Sección 8). NO concatena tensores BOLD (Sección 7):
    cada sitio se procesa por separado antes de concatenar solo las features.
    """

    roi_idx = tdha_data.roi_indices(roi_set)
    n_expected_features = EXPECTED_FEATURES[roi_set]

    blocks: list[np.ndarray] = []
    feature_hashes: dict[str, str] = {}
    for site in SITES:
        payload = payloads[site]
        roi_idx_validated = tdha_data.validate_indices(roi_idx, payload["bold"].shape[1])
        block = compute_site_static_fc(payload, roi_idx_validated)
        if block.shape[1] != n_expected_features:
            raise SystemExit(
                f"STOP: {site}×{roi_set} produjo {block.shape[1]} features; "
                f"se esperaban {n_expected_features}."
            )
        feature_hashes[f"{site}_{roi_set}"] = _sha_trunc(
            np.ascontiguousarray(block, dtype=np.float32).tobytes()
        )
        blocks.append(block)

    feature_matrix = np.concatenate(blocks, axis=0)
    if feature_matrix.shape[0] != len(master):
        raise SystemExit(
            "STOP: la matriz de features concatenada no tiene el mismo número "
            "de filas que la master table."
        )

    # Verificación defensiva por subject_key (Sección 7.1): los offsets por
    # sitio en 'master' (construida con el mismo orden canónico y el mismo
    # payload["subjects"]) deben coincidir exactamente con los bloques de
    # 'blocks' concatenados aquí.
    offset = 0
    for site in SITES:
        n_site = len(payloads[site]["subjects"])
        sub = master.iloc[offset : offset + n_site]
        if not (sub["site"] == site).all():
            raise SystemExit(
                f"STOP: desalineación detectada entre master table y bloque de "
                f"features en el sitio {site} (offset {offset})."
            )
        offset += n_site

    return np.asarray(feature_matrix, dtype=np.float32), feature_hashes


# ---------------------------------------------------------------------------
# Outer LOSO split + inner validation (Secciones 9, 10, 11)
# ---------------------------------------------------------------------------


def build_rotation_split(held_out_site: str, master: pd.DataFrame) -> dict[str, np.ndarray]:
    """Un split (fit/inner_val/test) para un sitio held-out.

    test = todos y solo los sujetos del sitio held-out.
    train_pool = todos los sujetos de los otros tres sitios.
    Dentro de train_pool: StratifiedShuffleSplit(n_splits=1, test_size=0.15,
    random_state=42), estratificado por site×diagnosis (Sección 10).
    """

    if held_out_site not in SITES:
        raise SystemExit(f"STOP: sitio held-out desconocido: {held_out_site!r}.")

    test_mask = master["site"] == held_out_site
    test_idx = master.loc[test_mask, "global_index"].to_numpy(dtype=np.int64)
    pool_df = master.loc[~test_mask]
    pool_idx = pool_df["global_index"].to_numpy(dtype=np.int64)

    strata = (pool_df["site"] + "|" + pool_df["y_true"].astype(str)).to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=INNER_VAL_FRAC, random_state=SPLIT_SEED
    )
    fit_rel, inner_rel = next(splitter.split(np.zeros(len(pool_idx)), strata))
    fit_idx = np.sort(pool_idx[fit_rel])
    inner_idx = np.sort(pool_idx[inner_rel])
    test_idx = np.sort(test_idx)

    _validate_rotation_split(held_out_site, master, fit_idx, inner_idx, test_idx)
    return {"fit": fit_idx, "inner_val": inner_idx, "test": test_idx}


def _validate_rotation_split(
    held_out_site: str,
    master: pd.DataFrame,
    fit_idx: np.ndarray,
    inner_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    site_by_index = master.set_index("global_index")["site"]

    # Sección 15: assertions de no leakage.
    if (site_by_index.loc[fit_idx] == held_out_site).any():
        raise SystemExit("STOP: held_out_site presente en fit.")
    if (site_by_index.loc[inner_idx] == held_out_site).any():
        raise SystemExit("STOP: held_out_site presente en inner_val.")
    if not set(fit_idx).isdisjoint(inner_idx):
        raise SystemExit("STOP: fit e inner_val se solapan.")
    if not set(fit_idx).isdisjoint(test_idx):
        raise SystemExit("STOP: fit y test se solapan.")
    if not set(inner_idx).isdisjoint(test_idx):
        raise SystemExit("STOP: inner_val y test se solapan.")
    union = set(fit_idx.tolist()) | set(inner_idx.tolist()) | set(test_idx.tolist())
    if len(union) != EXPECTED_TOTAL["n"]:
        raise SystemExit(
            f"STOP: unión fit/inner_val/test tiene {len(union)} elementos; "
            f"se esperaban {EXPECTED_TOTAL['n']}."
        )

    expected = EXPECTED_ROTATION_SIZES[held_out_site]
    sizes = {"fit": len(fit_idx), "inner_val": len(inner_idx), "test": len(test_idx)}
    if sizes != expected:
        raise SystemExit(
            f"STOP: tamaños de rotación para {held_out_site} son {sizes}; "
            f"se esperaban {expected}."
        )

    # Sección 10: los seis estratos site×diagnosis deben estar en fit e inner.
    training_sites = [s for s in SITES if s != held_out_site]
    expected_strata = {f"{s}|{c}" for s in training_sites for c in (0, 1)}
    for name, idx in (("fit", fit_idx), ("inner_val", inner_idx)):
        sub = master.set_index("global_index").loc[idx]
        present = set((sub["site"] + "|" + sub["y_true"].astype(str)).tolist())
        if present != expected_strata:
            raise SystemExit(
                f"STOP: estratos incompletos en {name} para held-out "
                f"{held_out_site}: presentes={sorted(present)}, "
                f"esperados={sorted(expected_strata)}."
            )


def rotation_split_fingerprint(split: Mapping[str, np.ndarray]) -> str:
    """Huella semántica de la rotación, vía ``run_experiment.split_fingerprint``
    (Sección 11.1) sobre un pseudo-fold fit/inner_val/outer_val=test."""

    pseudo_fold = [
        {
            "fit": split["fit"],
            "inner_val": split["inner_val"],
            "outer_val": split["test"],
        }
    ]
    return re_split_fingerprint(pseudo_fold)


def build_all_rotation_splits(master: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    return {site: build_rotation_split(site, master) for site in SITES}


def build_split_manifest(
    master: pd.DataFrame, splits: Mapping[str, Mapping[str, np.ndarray]]
) -> pd.DataFrame:
    """CSV canónico: 4 rotaciones x 465 = 1860 filas (Sección 11)."""

    master_indexed = master.set_index("global_index")
    rows: list[dict[str, Any]] = []
    for held_out_site in SITES:
        split = splits[held_out_site]
        split_of: dict[int, str] = {}
        for name in ("fit", "inner_val", "test"):
            for idx in split[name]:
                split_of[int(idx)] = name
        for _, row in master.iterrows():
            gi = int(row["global_index"])
            rows.append(
                {
                    "held_out_site": held_out_site,
                    "global_index": gi,
                    "site": row["site"],
                    "site_row_index": int(row["site_row_index"]),
                    "subject_id": row["subject_id"],
                    "subject_key": row["subject_key"],
                    "y_true": int(row["y_true"]),
                    "split": split_of[gi],
                    "stratum": f"{row['site']}|{row['y_true']}",
                }
            )
    manifest = pd.DataFrame(rows)
    if len(manifest) != 4 * EXPECTED_TOTAL["n"]:
        raise SystemExit(
            f"STOP: split manifest tiene {len(manifest)} filas; se esperaban "
            f"{4 * EXPECTED_TOTAL['n']}."
        )
    return manifest


# ---------------------------------------------------------------------------
# API de entrenamiento sin acceso a test (Sección 14)
# ---------------------------------------------------------------------------


def train_brainnetcnn(
    *,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_inner: np.ndarray,
    y_inner: np.ndarray,
    model_seed: int,
    n_features: int,
) -> tuple[Any, dict[str, Any]]:
    """Entrena BrainNetCNN. NO recibe X_test/y_test/índices test.

    Devuelve (modelo con pesos restaurados a la mejor época, metadata de
    convergencia). La compuerta de restauración (Sección 23) se verifica aquí,
    ANTES de que el modelo pueda tocar el held-out site.
    """

    import keras
    from keras.callbacks import EarlyStopping

    keras.backend.clear_session()
    gc.collect()
    keras.utils.set_random_seed(model_seed)

    model = kerasmodels.build("brainnetcnn", N_WINDOWS, n_features, **BNN_ARCH_KWARGS)
    compile_args = SimpleNamespace(lr=BNN_TRAIN_CONFIG["lr"], clipnorm=BNN_TRAIN_CONFIG["clipnorm"])
    model = _re_compile_model(model, compile_args)

    early_stopping = EarlyStopping(
        monitor=BNN_TRAIN_CONFIG["early_stopping_monitor"],
        mode="min",
        patience=BNN_TRAIN_CONFIG["patience"],
        min_delta=BNN_TRAIN_CONFIG["early_stopping_min_delta"],
        start_from_epoch=BNN_TRAIN_CONFIG["start_from_epoch"],
        restore_best_weights=True,
    )
    history = model.fit(
        X_fit,
        y_fit.reshape(-1, 1),
        validation_data=(X_inner, y_inner.reshape(-1, 1)),
        epochs=BNN_TRAIN_CONFIG["epochs"],
        batch_size=BNN_TRAIN_CONFIG["batch_size"],
        class_weight=None,
        shuffle=BNN_TRAIN_CONFIG["shuffle"],
        verbose=0,
        callbacks=[early_stopping],
    )

    monitor = BNN_TRAIN_CONFIG["early_stopping_monitor"]
    n_epochs = len(history.history["loss"])
    monitor_values = np.asarray(history.history[monitor], dtype=float)
    if not np.isfinite(monitor_values).all():
        raise RuntimeError(f"history.history[{monitor!r}] contiene valores no finitos.")
    if early_stopping.best is None:
        raise RuntimeError("EarlyStopping no observó ninguna época.")

    best_epoch = int(early_stopping.best_epoch) + 1
    best_monitor_value = float(early_stopping.best)
    recorded_value = float(monitor_values[best_epoch - 1])
    if not np.isclose(recorded_value, best_monitor_value, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            "Inconsistencia interna: history no coincide con early_stopping.best "
            f"({recorded_value} vs {best_monitor_value})."
        )

    restored_eval = model.evaluate(
        X_inner,
        y_inner.reshape(-1, 1),
        batch_size=BNN_TRAIN_CONFIG["batch_size"],
        verbose=0,
        return_dict=True,
    )
    restored_key = "loss" if monitor == "val_loss" else "bce"
    restored_monitor_value = float(restored_eval[restored_key])
    if not np.isfinite(restored_monitor_value):
        raise RuntimeError("La reevaluación de pesos restaurados no es finita.")
    if not np.isclose(restored_monitor_value, best_monitor_value, rtol=1e-4, atol=1e-6):
        raise RuntimeError(
            "Los pesos restaurados no reproducen best_monitor_value: "
            f"restored={restored_monitor_value}, best={best_monitor_value}. "
            "ABORTADO antes de tocar el held-out site."
        )

    metadata = {
        "epochs_ran": n_epochs,
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor_value,
        "restored_monitor_value": restored_monitor_value,
        "stopped_early": bool(n_epochs < BNN_TRAIN_CONFIG["epochs"]),
        "history": history.history,
    }
    return model, metadata


def evaluate_heldout(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Única llamada que toca el held-out site; reutiliza run_experiment.evaluate."""

    return re_evaluate(model, X_test, y_test)


def fit_logistic(
    *, X_fit: np.ndarray, y_fit: np.ndarray
) -> tuple[LogisticRegression, StandardScaler]:
    """Ajusta scaler y modelo logístico exclusivamente sobre FIT (Secciones 25.1/25.2).

    No recibe test. Lanza SystemExit si sklearn emite ConvergenceWarning, si
    n_iter_ >= max_iter, o si hay coeficientes/probabilidades no finitos.
    """

    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(**LOGREG_CONFIG)
        clf.fit(X_fit_scaled, y_fit)
        convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]

    if convergence_warnings:
        raise SystemExit(
            "STOP: sklearn.exceptions.ConvergenceWarning durante el fit logístico. "
            "No se aumenta max_iter tras observar el test; abortar la corrida formal."
        )
    if list(clf.classes_) != [0, 1]:
        raise SystemExit(f"STOP: clf.classes_ != [0, 1]: {list(clf.classes_)}.")
    n_iter = int(np.max(clf.n_iter_))
    if n_iter >= LOGREG_CONFIG["max_iter"]:
        raise SystemExit(f"STOP: n_iter_={n_iter} >= max_iter={LOGREG_CONFIG['max_iter']}.")
    if not np.isfinite(clf.coef_).all() or not np.isfinite(clf.intercept_).all():
        raise SystemExit("STOP: coeficientes logísticos no finitos.")

    return clf, scaler


class _ConstantProbabilityModel:
    """Envoltorio callable que deja reutilizar ``run_experiment.evaluate`` con
    probabilidades ya calculadas (sklearn), sin duplicar las fórmulas de métricas."""

    def __init__(self, probabilities: np.ndarray) -> None:
        self._probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1, 1)

    def __call__(self, X: Any, training: bool = False) -> np.ndarray:  # noqa: D401
        del X, training
        return self._probabilities


def evaluate_logistic_heldout(
    clf: LogisticRegression, scaler: StandardScaler, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    X_test_scaled = scaler.transform(X_test)
    probabilities = clf.predict_proba(X_test_scaled)[:, 1]
    if not np.isfinite(probabilities).all():
        raise SystemExit("STOP: probabilidades logísticas no finitas en held-out.")
    metrics, _ = re_evaluate(_ConstantProbabilityModel(probabilities), X_test_scaled, y_test)
    return metrics, probabilities


# ---------------------------------------------------------------------------
# Identidad de corrida formal (Sección 40)
# ---------------------------------------------------------------------------


def _model_code_hash(model: str) -> str | None:
    if model == "brainnetcnn":
        return _sha_file(Path(kerasmodels.brainnetcnn.__file__))
    return None  # logreg: no hay archivo de modelo local (Sección 41).


def build_identity(
    *,
    held_out_site: str,
    roi_set: str,
    model: str,
    model_seed: int | None,
    rotation_fp: str,
    formal_env_signature: str,
) -> dict[str, Any]:
    """Campos que determinan la identidad de una corrida formal.

    Excluye explícitamente: timestamp, output path, métricas de resultado
    (Sección 40).
    """

    roi_idx = tdha_data.roi_indices(roi_set)
    bold_hashes = {
        site: _sha_file(BOLD_DIR / f"{site}.joblib") for site in SITES
    }
    identity: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "held_out_site": held_out_site,
        "roi_set": str(roi_set),
        "model": model,
        "model_seed": model_seed,
        "representation": "static",
        "n_windows": N_WINDOWS,
        "fisher_z": FISHER_Z,
        "constant_policy": CONSTANT_POLICY,
        "split_seed": SPLIT_SEED,
        "inner_val_frac": INNER_VAL_FRAC,
        "class_weight": False,
        "site_weighting": False,
        "sample_weight": False,
        "harmonization": "none",
        "rotation_split_fingerprint": rotation_fp,
        "training_source_git_sha": git_info()["commit"],
        "runner_sha256": _sha_file(Path(__file__)),
        "data_code_sha256": _sha_file(REPO_ROOT / "src" / "data.py"),
        "model_code_sha256": _model_code_hash(model),
        "bold_sha256": bold_hashes,
        "roi_indices_hash": indices_hash(roi_idx),
        "formal_environment_signature": formal_env_signature,
    }
    if model == "brainnetcnn":
        identity["scientific_hyperparameters"] = {**BNN_ARCH_KWARGS, **BNN_TRAIN_CONFIG}
    else:
        identity["scientific_hyperparameters"] = dict(LOGREG_CONFIG)
    return identity


def formal_environment_signature() -> str:
    """Firma determinista del entorno formal (Sección 29), para congelarla en
    identity y detectar si una corrida posterior cambia de entorno."""

    info = env_info()
    payload = {
        "python": info.get("python"),
        "numpy": info.get("numpy"),
        "pandas": info.get("pandas"),
        "scikit_learn": info.get("scikit_learn"),
        "tensorflow": info.get("tensorflow"),
        "keras": info.get("keras"),
        "platform": info.get("platform"),
        "gpu": info.get("gpu"),
    }
    return _sha_trunc(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )


def make_run_id(
    *, held_out_site: str, roi_set: str, model: str, model_seed: int | None, identity_hash: str
) -> str:
    seed_part = f"seed-{model_seed}" if model == "brainnetcnn" else "seed-deterministic"
    return (
        f"{CAMPAIGN_ID}_holdout-{held_out_site}_roi-{roi_set}_{model}_"
        f"{seed_part}_{identity_hash}"
    )


# ---------------------------------------------------------------------------
# Escritura atómica de corridas formales (Sección 42)
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_config(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )


def atomic_promote(staging_dir: Path, final_dir: Path) -> None:
    """Promoción atómica: solo tras validar todos los artefactos en staging."""

    if final_dir.exists():
        raise SystemExit(f"STOP: {final_dir} ya existe; no se sobrescribe (no hay --overwrite).")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_dir, final_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--held-out-site", required=True, choices=SITES)
    parser.add_argument("--roi-set", required=True, choices=ROI_SETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--model-seed", type=int, default=None, choices=BNN_SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", choices=("synthetic", "preflight"), default=None)
    parser.add_argument("--resume", action="store_true", help="omite la corrida si ya existe un run formal válido")
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.model == "brainnetcnn" and args.model_seed is None:
        raise SystemExit("ERROR: --model brainnetcnn requiere --model-seed (uno de 42..46).")
    if args.model == "logreg" and args.model_seed is not None:
        raise SystemExit("ERROR: --model logreg no admite --model-seed.")


# ---------------------------------------------------------------------------
# --resume: validación real, no solo existencia de directorio (Sección 39)
# ---------------------------------------------------------------------------


def validate_existing_run(
    run_dir: Path,
    *,
    expected_identity_hash: str,
    expected_test_n: int,
    held_out_site: str,
) -> bool:
    """True solo si el run existente pasa TODAS las verificaciones formales."""

    config_path = run_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    if not config.get("formal", False):
        return False
    if config.get("identity_hash") != expected_identity_hash:
        return False
    if config.get("held_out_site") != held_out_site:
        return False

    required_artifacts = ["config.json", "split_membership.csv", "metrics_fit.csv",
                           "metrics_inner_val.csv", "metrics_test.csv", "predictions_test.csv",
                           "resumen.md"]
    if config.get("model") == "brainnetcnn":
        required_artifacts.append("history.csv")
    for name in required_artifacts:
        if not (run_dir / name).exists():
            return False

    try:
        predictions = pd.read_csv(run_dir / "predictions_test.csv")
    except (pd.errors.EmptyDataError, OSError):
        return False
    if len(predictions) != expected_test_n:
        return False
    if not np.isfinite(predictions["y_prob"]).all():
        return False
    if not ((predictions["y_prob"] >= 0) & (predictions["y_prob"] <= 1)).all():
        return False
    if (predictions["site"] != held_out_site).any():
        return False

    return True


# ---------------------------------------------------------------------------
# Ejecución formal (Secciones 42-47)
# ---------------------------------------------------------------------------


def _predictions_test_rows(
    *, master: pd.DataFrame, test_idx: np.ndarray, probabilities: np.ndarray,
    held_out_site: str, model: str, roi_set: str, model_seed: int | None, run_id: str,
) -> list[dict[str, Any]]:
    master_indexed = master.set_index("global_index")
    rows: list[dict[str, Any]] = []
    for idx, prob in zip(test_idx, probabilities):
        row = master_indexed.loc[int(idx)]
        rows.append(
            {
                "held_out_site": held_out_site,
                "site": row["site"],
                "subject_id": row["subject_id"],
                "subject_key": row["subject_key"],
                "y_true": int(row["y_true"]),
                "y_prob": float(prob),
                "model": model,
                "roi_set": roi_set,
                "model_seed": model_seed,
                "run_id": run_id,
            }
        )
    return rows


def _validate_predictions_test(rows: list[dict[str, Any]], *, held_out_site: str, expected_n: int) -> None:
    df = pd.DataFrame(rows)
    if len(df) != expected_n:
        raise SystemExit(f"STOP: predictions_test tiene {len(df)} filas; se esperaban {expected_n}.")
    if (df["site"] != held_out_site).any():
        raise SystemExit("STOP: predictions_test contiene sitios distintos del held-out.")
    if not ((df["y_prob"] >= 0) & (df["y_prob"] <= 1)).all():
        raise SystemExit("STOP: y_prob fuera de [0, 1] en predictions_test.")
    if not np.isfinite(df["y_prob"]).all():
        raise SystemExit("STOP: y_prob no finito en predictions_test.")
    if df["subject_key"].duplicated().any():
        raise SystemExit("STOP: subject_key duplicado en predictions_test.")
    if df["y_true"].nunique() < 2:
        raise SystemExit("STOP: predictions_test no contiene ambas clases.")


def _split_membership_rows(master: pd.DataFrame, split: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    master_indexed = master.set_index("global_index")
    split_of: dict[int, str] = {}
    for name in ("fit", "inner_val", "test"):
        for idx in split[name]:
            split_of[int(idx)] = name
    rows: list[dict[str, Any]] = []
    for gi, name in split_of.items():
        row = master_indexed.loc[gi]
        rows.append(
            {
                "site": row["site"],
                "subject_id": row["subject_id"],
                "subject_key": row["subject_key"],
                "y_true": int(row["y_true"]),
                "split": name,
            }
        )
    return rows


def run_formal(
    args: argparse.Namespace,
    master: pd.DataFrame,
    feature_matrix: np.ndarray,
    split: Mapping[str, np.ndarray],
    identity: Mapping[str, Any],
    identity_hash: str,
    run_id: str,
    rotation_fp: str,
) -> str:
    final_dir = FORMAL_OUTPUT_ROOT / run_id
    if args.resume and final_dir.exists():
        expected_n = EXPECTED_ROTATION_SIZES[args.held_out_site]["test"]
        if validate_existing_run(
            final_dir, expected_identity_hash=identity_hash,
            expected_test_n=expected_n, held_out_site=args.held_out_site,
        ):
            print(f"--resume: {run_id} ya existe y es válido; se omite.")
            return run_id
        raise SystemExit(
            f"STOP: {final_dir} existe pero NO pasa la validación de --resume "
            "(config incompleto, artefactos faltantes o identidad distinta). "
            "Revise y elimine manualmente el directorio corrupto antes de reintentar; "
            "no se sobrescribe automáticamente."
        )
    if final_dir.exists():
        raise SystemExit(f"STOP: {final_dir} ya existe. Use --resume o elimínelo manualmente.")

    if not SPLIT_MANIFEST_PATH.exists() or not DESIGN_JSON_PATH.exists():
        raise SystemExit(
            "STOP: no existe el design/split manifest congelado "
            f"({DESIGN_JSON_PATH}, {SPLIT_MANIFEST_PATH}). Ejecute la Fase de freeze "
            "(CP4) antes de la primera corrida formal."
        )
    split_manifest_hash = _sha_file(SPLIT_MANIFEST_PATH)
    design = json.loads(DESIGN_JSON_PATH.read_text(encoding="utf-8"))
    if design.get("training_source_git_sha") != identity["training_source_git_sha"]:
        raise SystemExit(
            "STOP: training_source_git_sha de esta corrida no coincide con el "
            "design.json congelado; el código cambió después del freeze."
        )
    # Los 8 hashes de feature están indexados por sitio, no por held-out; se
    # valida contra TODOS los sitios de entrenamiento y el held-out para este roi_set.
    for site in SITES:
        key = f"{site}_{args.roi_set}"
        if key not in design.get("feature_matrix_sha256", {}):
            raise SystemExit(f"STOP: falta el hash de feature congelado para {key}.")

    n_features = feature_matrix.shape[1]
    fit_idx, inner_idx, test_idx = split["fit"], split["inner_val"], split["test"]
    y = master.set_index("global_index")["y_true"].to_numpy()

    if args.model == "brainnetcnn":
        X = feature_matrix.reshape(-1, N_WINDOWS, n_features)
        model, meta = train_brainnetcnn(
            X_fit=X[fit_idx], y_fit=y[fit_idx].astype(np.int32),
            X_inner=X[inner_idx], y_inner=y[inner_idx].astype(np.int32),
            model_seed=args.model_seed, n_features=n_features,
        )
        fit_metrics, _ = re_evaluate(model, X[fit_idx], y[fit_idx])
        inner_metrics, _ = re_evaluate(model, X[inner_idx], y[inner_idx])
        test_metrics, probabilities = evaluate_heldout(model, X[test_idx], y[test_idx])
        history_rows = [
            {
                "epoch": e + 1,
                "loss": meta["history"]["loss"][e],
                "inner_val_loss": meta["history"]["val_loss"][e],
                "bce": meta["history"]["bce"][e],
                "inner_val_bce": meta["history"]["val_bce"][e],
                "accuracy": meta["history"]["accuracy"][e],
                "inner_val_accuracy": meta["history"]["val_accuracy"][e],
            }
            for e in range(meta["epochs_ran"])
        ]
        import keras

        keras.backend.clear_session()
        gc.collect()
    else:
        X = feature_matrix
        clf, scaler = fit_logistic(X_fit=X[fit_idx], y_fit=y[fit_idx])
        Xs = scaler.transform(X)
        fit_metrics, _ = re_evaluate(_ConstantProbabilityModel(clf.predict_proba(Xs[fit_idx])[:, 1]), Xs[fit_idx], y[fit_idx])
        inner_metrics, _ = re_evaluate(_ConstantProbabilityModel(clf.predict_proba(Xs[inner_idx])[:, 1]), Xs[inner_idx], y[inner_idx])
        test_metrics, probabilities = evaluate_logistic_heldout(clf, scaler, X[test_idx], y[test_idx])
        meta = {"epochs_ran": None, "best_epoch": None, "best_monitor_value": None,
                "restored_monitor_value": None, "stopped_early": None}
        history_rows = None

    pred_rows = _predictions_test_rows(
        master=master, test_idx=test_idx, probabilities=probabilities,
        held_out_site=args.held_out_site, model=args.model, roi_set=args.roi_set,
        model_seed=args.model_seed, run_id=run_id,
    )
    _validate_predictions_test(
        pred_rows, held_out_site=args.held_out_site,
        expected_n=EXPECTED_ROTATION_SIZES[args.held_out_site]["test"],
    )
    split_rows = _split_membership_rows(master, split)

    config: dict[str, Any] = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "formal": True,
        "run_id": run_id,
        "identity_hash": identity_hash,
        "held_out_site": args.held_out_site,
        "training_sites": [s for s in SITES if s != args.held_out_site],
        "roi_set": args.roi_set,
        "n_rois": int(tdha_data.roi_indices(args.roi_set).size),
        "n_features": int(n_features),
        "representation": "static",
        "n_windows": N_WINDOWS,
        "fisher_z": FISHER_Z,
        "constant_policy": CONSTANT_POLICY,
        "model": args.model,
        "arch": BNN_ARCH_KWARGS if args.model == "brainnetcnn" else None,
        "model_seed": args.model_seed,
        "split_seed": SPLIT_SEED,
        "inner_val_frac": INNER_VAL_FRAC,
        "rotation_split_fingerprint": rotation_fp,
        "class_weight": False,
        "site_weighting": False,
        "sample_weight": False,
        "harmonization": "none",
        "fit_n": int(len(fit_idx)),
        "inner_val_n": int(len(inner_idx)),
        "test_n": int(len(test_idx)),
        "split_manifest_path": str(SPLIT_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "split_manifest_file_sha256": split_manifest_hash,
        "roi_indices_hash": identity["roi_indices_hash"],
        "bold_sha256": identity["bold_sha256"],
        "training_source_git_sha": identity["training_source_git_sha"],
        "runner_sha256": identity["runner_sha256"],
        "data_code_sha256": identity["data_code_sha256"],
        "model_code_sha256": identity["model_code_sha256"],
        "environment": env_info(),
        "environment_signature": identity["formal_environment_signature"],
        "git_head_sha": git_info()["commit"],
        "git_clean_except_results_loso": git_info()["clean"] or _git_clean_except_results_loso(),
        "convergence": meta,
        "command": "run_loso.py " + " ".join(sys.argv[1:]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "identity": identity,
    }

    staging_dir = Path(tempfile.mkdtemp(prefix=f".staging-{run_id}.", dir=str(FORMAL_OUTPUT_ROOT)))
    try:
        _write_config(staging_dir / "config.json", config)
        _write_csv(staging_dir / "split_membership.csv", split_rows)
        _write_csv(staging_dir / "metrics_fit.csv", [fit_metrics])
        _write_csv(staging_dir / "metrics_inner_val.csv", [inner_metrics])
        _write_csv(staging_dir / "metrics_test.csv", [test_metrics])
        _write_csv(staging_dir / "predictions_test.csv", pred_rows)
        if history_rows is not None:
            _write_csv(staging_dir / "history.csv", history_rows)
        (staging_dir / "resumen.md").write_text(
            f"# {run_id}\n\nheld_out_site={args.held_out_site} roi_set={args.roi_set} "
            f"model={args.model} model_seed={args.model_seed}\n\n"
            f"test AUC={test_metrics.get('auc')}\n\n"
            f"rotation_split_fingerprint={rotation_fp}\n",
            encoding="utf-8",
        )
        atomic_promote(staging_dir, final_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"\ncorrida formal escrita en: {final_dir}")
    print(f"test AUC = {test_metrics.get('auc')}")
    return run_id


def _git_clean_except_results_loso() -> bool:
    import subprocess

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True,
        )
    except Exception:
        return False
    dirty_paths = [line[3:] for line in status.splitlines() if line.strip()]
    return all(p.startswith("results/loso/") for p in dirty_paths)


# ---------------------------------------------------------------------------
# Smoke tests pre-freeze (Sección 36): NUNCA calculan performance held-out real
# antes del freeze científico.
# ---------------------------------------------------------------------------


def run_synthetic_smoke(args: argparse.Namespace) -> str:
    """Smoke A: datos 100% sintéticos. Ejercita entrenamiento, early stopping,
    evaluación y promoción atómica sin tocar datos reales."""

    rng = np.random.default_rng(0)
    sizes = EXPECTED_ROTATION_SIZES[args.held_out_site]
    n_features = EXPECTED_FEATURES[args.roi_set]

    def synth(n: int) -> tuple[np.ndarray, np.ndarray]:
        X = rng.normal(size=(n, n_features)).astype(np.float32)
        y = rng.integers(0, 2, size=n).astype(np.int32)
        if y.sum() == 0:
            y[0] = 1
        if y.sum() == n:
            y[0] = 0
        return X, y

    X_fit, y_fit = synth(sizes["fit"])
    X_inner, y_inner = synth(sizes["inner_val"])
    X_test, y_test = synth(sizes["test"])

    if args.model == "brainnetcnn":
        import keras

        model, meta = train_brainnetcnn(
            X_fit=X_fit.reshape(-1, N_WINDOWS, n_features), y_fit=y_fit,
            X_inner=X_inner.reshape(-1, N_WINDOWS, n_features), y_inner=y_inner,
            model_seed=args.model_seed or 42, n_features=n_features,
        )
        test_metrics, _ = evaluate_heldout(model, X_test.reshape(-1, N_WINDOWS, n_features), y_test)
        keras.backend.clear_session()
        gc.collect()
    else:
        clf, scaler = fit_logistic(X_fit=X_fit, y_fit=y_fit)
        test_metrics, _ = evaluate_logistic_heldout(clf, scaler, X_test, y_test)

    tmp_root = Path(tempfile.mkdtemp(prefix="loso_synthetic_smoke_"))
    try:
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(tmp_root)))
        (staging / "predictions_test.csv").write_text("subject_key,y_prob\nsmoke,0.5\n")
        final = tmp_root / "smoke_run"
        atomic_promote(staging, final)
        if not final.exists() or staging.exists():
            raise SystemExit("STOP: la promoción atómica sintética no se comportó como se esperaba.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(
        f"smoke sintético OK: model={args.model} held_out={args.held_out_site} "
        f"roi_set={args.roi_set} test_auc={test_metrics.get('auc')} "
        "(datos sintéticos — NO es un resultado científico)."
    )
    return "smoke-synthetic-ok"


def run_real_preflight(
    args: argparse.Namespace, master: pd.DataFrame, feature_matrix: np.ndarray,
    split: Mapping[str, np.ndarray], *, epochs: int = 3,
) -> str:
    """Smoke B: datos reales, SOLO fit+inner_val, epochs reducidas.
    Prohibido evaluar el held-out site aquí (Sección 36)."""

    if args.model != "brainnetcnn":
        raise SystemExit("STOP: --smoke preflight solo se implementa para brainnetcnn.")

    import keras

    n_features = feature_matrix.shape[1]
    y = master.set_index("global_index")["y_true"].to_numpy()
    X = feature_matrix.reshape(-1, N_WINDOWS, n_features)
    fit_idx, inner_idx = split["fit"], split["inner_val"]

    original = {"epochs": BNN_TRAIN_CONFIG["epochs"], "patience": BNN_TRAIN_CONFIG["patience"]}
    BNN_TRAIN_CONFIG["epochs"] = epochs
    BNN_TRAIN_CONFIG["patience"] = epochs
    try:
        train_brainnetcnn(
            X_fit=X[fit_idx], y_fit=y[fit_idx].astype(np.int32),
            X_inner=X[inner_idx], y_inner=y[inner_idx].astype(np.int32),
            model_seed=args.model_seed or 42, n_features=n_features,
        )
    finally:
        BNN_TRAIN_CONFIG.update(original)
        keras.backend.clear_session()
        gc.collect()

    print(
        f"preflight real OK: held_out={args.held_out_site} roi_set={args.roi_set} "
        f"epochs={epochs} — SIN evaluación held-out (no se tocó test_idx)."
    )
    return "smoke-preflight-ok"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> str | None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(args)

    if args.smoke == "synthetic":
        return run_synthetic_smoke(args)

    payloads = load_site_payloads()
    master = build_master_table(payloads)
    feature_matrix, feature_hashes = build_feature_matrix(args.roi_set, payloads, master)
    split = build_rotation_split(args.held_out_site, master)
    rotation_fp = rotation_split_fingerprint(split)

    n_features = feature_matrix.shape[1]
    param_count = None
    if args.model == "brainnetcnn":
        import keras

        probe = kerasmodels.build("brainnetcnn", N_WINDOWS, n_features, **BNN_ARCH_KWARGS)
        param_count = int(probe.count_params())
        del probe
        keras.backend.clear_session()
        gc.collect()
        if param_count != EXPECTED_BNN_PARAM_COUNT[args.roi_set]:
            raise SystemExit(
                f"STOP: parámetros BrainNetCNN = {param_count}; se esperaban "
                f"{EXPECTED_BNN_PARAM_COUNT[args.roi_set]} para roi_set={args.roi_set}."
            )

    env_sig = formal_environment_signature()
    identity = build_identity(
        held_out_site=args.held_out_site, roi_set=args.roi_set, model=args.model,
        model_seed=args.model_seed, rotation_fp=rotation_fp, formal_env_signature=env_sig,
    )
    identity_hash = config_hash(identity)
    run_id = make_run_id(
        held_out_site=args.held_out_site, roi_set=args.roi_set, model=args.model,
        model_seed=args.model_seed, identity_hash=identity_hash,
    )

    print(f"run_id candidato: {run_id}")
    print(f"rotation_split_fingerprint: {rotation_fp}")
    print(f"fit={len(split['fit'])} inner_val={len(split['inner_val'])} test={len(split['test'])}")
    print(f"feature hashes ({args.roi_set} ROI): {feature_hashes}")
    if param_count is not None:
        print(f"param_count: {param_count}")

    if args.dry_run:
        print("\ndry-run correcto: no se entrenó, no se evaluó held-out, no se creó ningún run formal.")
        return run_id

    if args.smoke == "preflight":
        return run_real_preflight(args, master, feature_matrix, split)

    return run_formal(args, master, feature_matrix, split, identity, identity_hash, run_id, rotation_fp)


if __name__ == "__main__":
    main()
