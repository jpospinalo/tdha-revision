#!/usr/bin/env python3
"""Fase 1 del análisis comparativo de ROIs: auditoría de comparabilidad y
construcción del dataset analítico.

Lee únicamente los artefactos de las 16 corridas listadas en
``run_manifest.csv`` (config.json, folds.csv, predictions_val.csv,
metrics_val.csv, y la presencia de history.csv/metrics_train.csv/resumen.md).
No modifica nada bajo ``results/``. No calcula el estimando principal desde
``metrics_train.csv``, ``history.csv`` ni ``metrics_val.csv``.

Ver ``analysis/roi_comparison/README.md`` y ``analysis_plan.md`` (plan 5.6
congelado, D1-D5 resueltos) para el contexto metodológico completo.

Salidas (solo si todas las validaciones pasan):
  outputs/data/subject_scores.csv
  outputs/data/metrics_by_repeat.csv
  outputs/tables/comparability_audit.csv

Si alguna validación falla, se escribe únicamente
``outputs/tables/comparability_audit.csv`` (con diagnóstico) y el script
termina con código de salida distinto de cero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

REQUIRED_ARTIFACTS = [
    "config.json",
    "folds.csv",
    "history.csv",
    "metrics_train.csv",
    "metrics_val.csv",
    "predictions_val.csv",
    "resumen.md",
]

# roi_indices_hash congelado por tamaño de ROI: debe coincidir entre los
# cuatro sitios para un mismo roi_set. No exige que los cuatro tamaños
# compartan el mismo hash entre sí.
ROI_INDICES_HASH = {
    12: "640bfb198d6edcdd",
    18: "2d67f7d1b693a33a",
    39: "3f7657c228ed0f19",
    116: "b761f1f690ecb0af",
}

# atlas_hash aceptados. Difieren por una corrección textual histórica de
# roi_sets.json y NO se usan para decidir comparabilidad entre tamaños.
ACCEPTED_ATLAS_HASHES = {"104b6c37ad9b7299", "eb7675377cec20c2"}

# (site, roi_set) con git.clean == False y procedencia ya aceptada por el
# equipo (ver results/README.md, "Comparabilidad y procedencia por sitio").
ACCEPTED_DIRTY_TREE = {
    ("NeuroIMAGE", 12),
    ("OHSU", 39),
    ("Peking", 12),
}

EXPECTED_SUBJECT_COUNTS = {"NYU": 177, "Peking": 183, "NeuroIMAGE": 39, "OHSU": 66}
EXPECTED_PRED_COUNTS = {"NYU": 885, "Peking": 915, "NeuroIMAGE": 195, "OHSU": 330}

# Campos escalares de config.json que deben ser IDÉNTICOS entre los cuatro
# tamaños de ROI dentro de un mismo sitio.
WITHIN_SITE_SCALAR_FIELDS = [
    "split_fingerprint",
    "bold_hash",
    "data_code_hash",
    "runner_code_hash",
    "n_subjects",
    "n_timepoints",
    "seed",
    "n_splits",
    "n_repeats",
    "lr",
    "batch_size",
    "epochs",
    "patience",
    "inner_val_frac",
    "early_stopping_monitor",
    "early_stopping_min_delta",
    "start_from_epoch",
    "model",
    "representation",
    "window",
    "step",
    "class_weight",
    # CORRECCIONES_V19 §7.2: campos científicos adicionales.
    "windowing_preset",
    "fisher_z",
    "constant_policy",
    "clipnorm",
    "deterministic",
    "mixed_precision",
]
# Campos anidados (dict) que deben coincidir dentro de un mismo sitio.
WITHIN_SITE_DICT_FIELDS = ["arch", "class_balance", "windowing"]

# Parámetros ajenos a este baseline: deben permanecer null en cada una de las
# 16 corridas aprobadas (CORRECCIONES_V19 §7.2). No es una comparación entre
# tamaños dentro de un sitio; es una condición por corrida individual.
EXPECTED_NULL_FIELDS = ["representation_seed", "random_subset", "n_random_sets", "exclude_roi_set"]

# CORRECCIONES_V19 §7.1: config_hash debe ser exactamente 8 hex minúsculos.
CONFIG_HASH_RE = re.compile(r"^[0-9a-f]{8}$")

# CORRECCIONES_V19 §7.1: único artefacto extra documentado y aceptado, y solo
# para esta corrida puntual. Cualquier otro artefacto extra, en cualquier
# corrida, debe fallar la auditoría.
ACCEPTED_EXTRA_ARTIFACTS = {("Peking", 116): {"peking_dummy.txt"}}

CONFIG_COLUMN_TO_MANIFEST = {"site": "site", "roi_set": "roi_set", "run_id": "run_id"}

# Decisiones científicas congeladas de analysis_config.json (plan 5.6,
# resolución D1-D5 aprobada por el equipo). validate_analysis_config() es el
# único validador de estos campos; run_statistical_analysis.py lo importa y
# reutiliza en vez de duplicar esta lista (CORRECCIONES_V19 §6).
FROZEN_ANALYSIS_CONFIG = {
    "analysis_schema_version": 1,
    "plan_version": "5.6",
    "site_order": ["NYU", "Peking", "NeuroIMAGE", "OHSU"],
    "roi_order": [12, 18, 39, 116],
    "primary_metric": "mean_repeat_oof_auc",
    "repeat_aggregation": "metric_then_mean",
    "secondary_metrics": ["balanced_accuracy", "f1_macro", "sensitivity", "specificity"],
    "audit_metrics": ["accuracy"],
    "classification_threshold": 0.5,
    "positive_label": 1,
    "ci_level": 0.95,
    "noninferiority_margin": None,
    "noninferiority_margin_rationale": None,
    "bootstrap_iterations": 10000,
    "bootstrap_seed": 42,
    "bootstrap_rng": "numpy_pcg64",
    "bootstrap_seed_scope": "reset_per_site",
    "bootstrap_subject_order": "subject_id_ascending",
    "bootstrap_quantile_method": "linear",
    "bootstrap_method": "paired_stratified_percentile",
    "readme_round_decimals": 2,
}


class ValidationError(Exception):
    """Fallo bloqueante de validación: archivo, corrida y regla incumplida."""


def validate_analysis_config(config: dict) -> None:
    """Valida explícitamente las decisiones científicas congeladas de
    ``analysis_config.json`` (CORRECCIONES_V19 §6). Cualquier discrepancia
    detiene la ejecución con el campo, el valor recibido y el valor
    esperado; no corrige el JSON automáticamente ni acepta un valor
    alternativo por línea de comandos.

    No rechaza claves adicionales puramente documentales que no estén en
    ``FROZEN_ANALYSIS_CONFIG`` (por ejemplo, parámetros de una prueba
    exploratoria agregada después de este plan): valida únicamente las
    decisiones científicas ya aprobadas.
    """
    for field, expected in FROZEN_ANALYSIS_CONFIG.items():
        if field not in config:
            raise ValidationError(
                f"analysis_config.json: falta el campo congelado {field!r} "
                f"(se esperaba {expected!r})"
            )
        actual = config[field]
        if actual != expected:
            raise ValidationError(
                f"analysis_config.json: campo {field!r} tiene valor {actual!r}, "
                f"se esperaba el valor congelado {expected!r}"
            )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    expected_cols = ["site", "roi_set", "run_id", "relative_path", "include", "rationale"]
    if list(df.columns) != expected_cols:
        raise ValidationError(
            f"run_manifest.csv: columnas inesperadas {list(df.columns)}, "
            f"se esperaba {expected_cols}"
        )
    df["roi_set"] = df["roi_set"].astype(int)
    df["include"] = df["include"].str.lower().map({"true": True, "false": False})
    return df


def validate_manifest_structure(manifest: pd.DataFrame, site_order: list, roi_order: list) -> None:
    included = manifest[manifest["include"]]
    if len(included) != 16:
        raise ValidationError(
            f"run_manifest.csv: se esperaban 16 filas con include=true, hay {len(included)}"
        )
    sites = set(included["site"])
    if sites != set(site_order):
        raise ValidationError(f"run_manifest.csv: sitios inesperados {sites}, se esperaba {set(site_order)}")
    for site in site_order:
        rois = sorted(included[included["site"] == site]["roi_set"].tolist())
        if rois != sorted(roi_order):
            raise ValidationError(
                f"run_manifest.csv: sitio {site} tiene roi_set {rois}, se esperaba {sorted(roi_order)}"
            )
    dup = included.duplicated(subset=["site", "roi_set"])
    if dup.any():
        bad = included[dup][["site", "roi_set"]].to_dict("records")
        raise ValidationError(f"run_manifest.csv: combinaciones sitio-ROI duplicadas: {bad}")
    for _, row in included.iterrows():
        rel = row["relative_path"]
        if not rel.startswith("results/runs/"):
            raise ValidationError(
                f"run_manifest.csv: {row['run_id']} tiene relative_path fuera de results/runs/: {rel}"
            )
        if "archive" in Path(rel).parts:
            raise ValidationError(f"run_manifest.csv: {row['run_id']} apunta a results/archive/: {rel}")


def validate_artifacts_contract(run_dir: Path) -> tuple[list, list]:
    if not run_dir.is_dir():
        raise ValidationError(f"{run_dir}: carpeta de corrida inexistente")
    present = {p.name for p in run_dir.iterdir() if p.is_file()}
    missing = [a for a in REQUIRED_ARTIFACTS if a not in present]
    extras = sorted(present - set(REQUIRED_ARTIFACTS))
    return missing, extras


def validate_run_config_matches_manifest(cfg: dict, row: pd.Series, run_dir: Path) -> None:
    if cfg.get("site") != row["site"]:
        raise ValidationError(f"{run_dir}: config.json site={cfg.get('site')!r} != manifiesto {row['site']!r}")
    if str(cfg.get("roi_set")) != str(row["roi_set"]):
        raise ValidationError(
            f"{run_dir}: config.json roi_set={cfg.get('roi_set')!r} != manifiesto {row['roi_set']!r}"
        )
    if cfg.get("run_id") != row["run_id"]:
        raise ValidationError(f"{run_dir}: config.json run_id={cfg.get('run_id')!r} != manifiesto {row['run_id']!r}")
    if cfg.get("config_schema_version") != 4:
        raise ValidationError(
            f"{run_dir}: config_schema_version={cfg.get('config_schema_version')!r}, se esperaba 4"
        )
    if cfg.get("n_splits") != 10:
        raise ValidationError(f"{run_dir}: n_splits={cfg.get('n_splits')!r}, se esperaba 10")
    if cfg.get("n_repeats") != 5:
        raise ValidationError(f"{run_dir}: n_repeats={cfg.get('n_repeats')!r}, se esperaba 5")
    if cfg.get("seed") != 42:
        raise ValidationError(f"{run_dir}: seed={cfg.get('seed')!r}, se esperaba 42")

    # CORRECCIONES_V19 §7.1: identidad de la carpeta y formato de config_hash.
    # No se reimplementa el algoritmo histórico que produjo config_hash; solo
    # se valida su forma y su coherencia con run_id.
    if run_dir.name != row["run_id"]:
        raise ValidationError(
            f"{run_dir}: el nombre de la carpeta ({run_dir.name!r}) no coincide con "
            f"run_id ({row['run_id']!r})"
        )
    config_hash = cfg.get("config_hash")
    if not isinstance(config_hash, str) or not CONFIG_HASH_RE.match(config_hash):
        raise ValidationError(
            f"{run_dir}: config_hash={config_hash!r} no es una cadena de exactamente "
            f"ocho caracteres hexadecimales minúsculos"
        )
    if not row["run_id"].endswith("_" + config_hash):
        raise ValidationError(
            f"{run_dir}: run_id ({row['run_id']!r}) no termina en '_' + config_hash "
            f"({config_hash!r})"
        )

    # CORRECCIONES_V19 §7.2: parámetros ajenos a este baseline deben ser null.
    for field in EXPECTED_NULL_FIELDS:
        if cfg.get(field) is not None:
            raise ValidationError(
                f"{run_dir}: campo {field!r} esperado en null, tiene valor {cfg.get(field)!r} "
                f"(parámetro ajeno al baseline aprobado)"
            )


def validate_within_site_comparability(site: str, configs_by_roi: dict[int, dict]) -> None:
    roi_sets = sorted(configs_by_roi)
    ref_roi = roi_sets[0]
    ref_cfg = configs_by_roi[ref_roi]
    for roi in roi_sets[1:]:
        cfg = configs_by_roi[roi]
        for field in WITHIN_SITE_SCALAR_FIELDS:
            if ref_cfg.get(field) != cfg.get(field):
                raise ValidationError(
                    f"sitio {site}: campo científico {field!r} difiere entre ROI {ref_roi} "
                    f"({ref_cfg.get(field)!r}) y ROI {roi} ({cfg.get(field)!r})"
                )
        for field in WITHIN_SITE_DICT_FIELDS:
            if ref_cfg.get(field) != cfg.get(field):
                raise ValidationError(
                    f"sitio {site}: campo científico {field!r} difiere entre ROI {ref_roi} "
                    f"({ref_cfg.get(field)!r}) y ROI {roi} ({cfg.get(field)!r})"
                )
        # atlas_hash: debe estar en el conjunto aceptado, pero no se exige igualdad.
        for candidate_roi, candidate_cfg in ((ref_roi, ref_cfg), (roi, cfg)):
            atlas_hash = candidate_cfg.get("atlas_hash")
            if atlas_hash not in ACCEPTED_ATLAS_HASHES:
                raise ValidationError(
                    f"sitio {site} ROI {candidate_roi}: atlas_hash {atlas_hash!r} no está en el "
                    f"conjunto aceptado {ACCEPTED_ATLAS_HASHES}; las entradas ya no corresponden "
                    f"al conjunto auditado"
                )
        # roi_indices_hash: debe coincidir con el mapa congelado para este tamaño,
        # tanto para la corrida de referencia como para la comparada.
        for candidate_roi, candidate_cfg in ((ref_roi, ref_cfg), (roi, cfg)):
            expected_hash = ROI_INDICES_HASH.get(candidate_roi)
            if candidate_cfg.get("roi_indices_hash") != expected_hash:
                raise ValidationError(
                    f"sitio {site} ROI {candidate_roi}: roi_indices_hash="
                    f"{candidate_cfg.get('roi_indices_hash')!r}, se esperaba {expected_hash!r} "
                    f"(mapa congelado)"
                )


def validate_folds_hash_within_site(site: str, folds_hash_by_roi: dict[int, str]) -> None:
    """CORRECCIONES_V19 §7.3: folds.csv debe ser byte-idéntico (mismo SHA-256)
    entre los cuatro tamaños de ROI de un mismo sitio. Si difieren, no se
    intenta determinar cuál archivo debería reemplazar a los demás: se falla
    y se reportan los hashes encontrados."""
    hashes = set(folds_hash_by_roi.values())
    if len(hashes) > 1:
        raise ValidationError(
            f"sitio {site}: folds.csv difiere entre tamaños de ROI (deben ser idénticos): "
            f"{folds_hash_by_roi}"
        )


def validate_roi_indices_hash_across_sites(all_configs: dict[str, dict[int, dict]], roi_order: list) -> None:
    for roi in roi_order:
        expected = ROI_INDICES_HASH[roi]
        for site, by_roi in all_configs.items():
            actual = by_roi[roi].get("roi_indices_hash")
            if actual != expected:
                raise ValidationError(
                    f"roi_indices_hash para ROI {roi} en sitio {site} es {actual!r}, "
                    f"se esperaba {expected!r} (debe coincidir entre sitios para el mismo tamaño)"
                )


def load_predictions(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "predictions_val.csv")
    required_cols = ["fold", "repeat", "subject", "subject_id", "y_true", "y_prob"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"{run_dir}/predictions_val.csv: faltan columnas {missing_cols}")
    return df


def load_folds(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "folds.csv")
    required_cols = ["fold", "repeat", "subject", "subject_id", "split"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"{run_dir}/folds.csv: faltan columnas {missing_cols}")
    return df


def validate_predictions(run_dir: Path, preds: pd.DataFrame, folds: pd.DataFrame, site: str) -> None:
    if not np.issubdtype(preds["y_prob"].dtype, np.number):
        raise ValidationError(f"{run_dir}: y_prob no es numérico")
    if not np.isfinite(preds["y_prob"]).all():
        raise ValidationError(f"{run_dir}: y_prob contiene valores no finitos (NaN/inf)")
    if (preds["y_prob"] < 0).any() or (preds["y_prob"] > 1).any():
        raise ValidationError(f"{run_dir}: y_prob fuera de [0,1]")
    if not set(preds["y_true"].unique()).issubset({0, 1}):
        raise ValidationError(f"{run_dir}: y_true fuera de {{0,1}}: {sorted(preds['y_true'].unique())}")

    repeats = sorted(preds["repeat"].unique())
    if repeats != [1, 2, 3, 4, 5]:
        raise ValidationError(f"{run_dir}: repeticiones {repeats}, se esperaba 1..5")

    for r in repeats:
        folds_in_repeat = sorted(preds.loc[preds["repeat"] == r, "fold"].unique())
        if len(folds_in_repeat) != 10:
            raise ValidationError(
                f"{run_dir}: repetición {r} tiene {len(folds_in_repeat)} folds distintos, se esperaban 10"
            )
    pairs = preds[["repeat", "fold"]].drop_duplicates()
    if len(pairs) != 50:
        raise ValidationError(f"{run_dir}: {len(pairs)} pares (repeat,fold) distintos, se esperaban 50")

    dup_key = preds.duplicated(subset=["repeat", "subject_id"])
    if dup_key.any():
        bad = preds.loc[dup_key, ["repeat", "subject_id"]].to_dict("records")
        raise ValidationError(f"{run_dir}: predicciones duplicadas (repeat, subject_id): {bad}")

    for r in repeats:
        sub = preds[preds["repeat"] == r]
        n_expected = EXPECTED_SUBJECT_COUNTS[site]
        if len(sub) != n_expected:
            raise ValidationError(
                f"{run_dir}: repetición {r} tiene {len(sub)} predicciones, se esperaban {n_expected}"
                f" (una por sujeto)"
            )
        if set(sub["y_true"].unique()) != {0, 1}:
            raise ValidationError(f"{run_dir}: repetición {r} no contiene ambas clases")

    # CORRECCIONES_V19 §7.4: correspondencia exacta predicción <-> outer_val,
    # en ambos sentidos, por (repeat, fold, subject, subject_id). No basta con
    # que las predicciones sean subconjunto de outer_val: el caso inverso
    # también se valida (ninguna fila outer_val sin predicción).
    if not np.issubdtype(preds["subject"].dtype, np.integer) and not (
        np.issubdtype(preds["subject"].dtype, np.number) and (preds["subject"] % 1 == 0).all()
    ):
        raise ValidationError(f"{run_dir}: la columna subject de predictions_val.csv no es entera")
    if preds["subject_id"].isna().any():
        raise ValidationError(f"{run_dir}: predictions_val.csv tiene subject_id nulo")

    key_cols = ["repeat", "fold", "subject", "subject_id"]
    outer_val = folds[folds["split"] == "outer_val"][key_cols]
    if outer_val.duplicated().any():
        dup = outer_val[outer_val.duplicated()].head(5).to_dict("records")
        raise ValidationError(f"{run_dir}: folds.csv tiene filas outer_val duplicadas: {dup}")
    if preds[key_cols].duplicated().any():
        dup = preds[preds[key_cols].duplicated()].head(5).to_dict("records")
        raise ValidationError(f"{run_dir}: predictions_val.csv tiene claves (repeat,fold,subject,subject_id) duplicadas: {dup}")

    outer_key = set(map(tuple, outer_val.itertuples(index=False, name=None)))
    pred_key = set(map(tuple, preds[key_cols].itertuples(index=False, name=None)))
    if len(outer_val) != len(preds):
        raise ValidationError(
            f"{run_dir}: {len(preds)} filas en predictions_val.csv vs {len(outer_val)} filas "
            f"outer_val en folds.csv; deben tener el mismo número de filas"
        )
    if pred_key - outer_key:
        missing = list(pred_key - outer_key)[:5]
        raise ValidationError(
            f"{run_dir}: {len(pred_key - outer_key)} predicciones sin fila outer_val correspondiente "
            f"en folds.csv (ejemplos: {missing})"
        )
    if outer_key - pred_key:
        missing = list(outer_key - pred_key)[:5]
        raise ValidationError(
            f"{run_dir}: {len(outer_key - pred_key)} filas outer_val de folds.csv sin predicción "
            f"correspondiente (ejemplos: {missing})"
        )

    # Correspondencia estable subject <-> subject_id dentro de esta corrida.
    mapping = preds[["subject", "subject_id"]].drop_duplicates()
    if mapping["subject"].nunique() != mapping["subject_id"].nunique():
        raise ValidationError(f"{run_dir}: subject <-> subject_id no es estable/biyectivo en predictions_val.csv")


def validate_metrics_val_structure(run_dir: Path, metrics_val: pd.DataFrame, preds: pd.DataFrame) -> None:
    """CORRECCIONES_V19 §7.5: integridad estructural de metrics_val.csv.

    Sigue siendo únicamente un control estructural; sus métricas no se usan
    para el estimando. No exige que los folds se reinicien en 1 dentro de
    cada repetición (numeración global 1..50, ver §7 de las instrucciones).
    """
    required_cols = {"repeat", "fold"}
    missing_cols = required_cols - set(metrics_val.columns)
    if missing_cols:
        raise ValidationError(f"{run_dir}: metrics_val.csv: faltan columnas {sorted(missing_cols)}")
    if len(metrics_val) != 50:
        raise ValidationError(f"{run_dir}: metrics_val.csv tiene {len(metrics_val)} filas, se esperaban 50")
    if metrics_val.duplicated(subset=["repeat", "fold"]).any():
        dup = metrics_val[metrics_val.duplicated(subset=["repeat", "fold"])][["repeat", "fold"]].to_dict("records")
        raise ValidationError(f"{run_dir}: metrics_val.csv tiene (repeat,fold) duplicados: {dup}")
    reps = sorted(metrics_val["repeat"].unique())
    if reps != [1, 2, 3, 4, 5]:
        raise ValidationError(f"{run_dir}: metrics_val.csv: repeticiones {reps}, se esperaba 1..5")
    for r in reps:
        n_folds = metrics_val.loc[metrics_val["repeat"] == r, "fold"].nunique()
        if n_folds != 10:
            raise ValidationError(
                f"{run_dir}: metrics_val.csv: repetición {r} tiene {n_folds} folds distintos, se esperaban 10"
            )
    mv_pairs = set(map(tuple, metrics_val[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)))
    if len(mv_pairs) != 50:
        raise ValidationError(f"{run_dir}: metrics_val.csv: {len(mv_pairs)} pares (repeat,fold) distintos, se esperaban 50")
    pred_pairs = set(map(tuple, preds[["repeat", "fold"]].drop_duplicates().itertuples(index=False, name=None)))
    if mv_pairs != pred_pairs:
        raise ValidationError(
            f"{run_dir}: metrics_val.csv y predictions_val.csv no comparten el mismo conjunto de "
            f"pares (repeat,fold): diferencia={mv_pairs.symmetric_difference(pred_pairs)}"
        )


def validate_subject_identity(preds_by_roi: dict[int, pd.DataFrame], site: str) -> None:
    ref_roi = sorted(preds_by_roi)[0]
    ref = preds_by_roi[ref_roi]

    mapping = ref[["subject", "subject_id"]].drop_duplicates()
    if mapping["subject"].nunique() != mapping["subject_id"].nunique():
        raise ValidationError(f"sitio {site}: subject <-> subject_id no es biyectivo (ROI {ref_roi})")

    n_subjects = EXPECTED_SUBJECT_COUNTS[site]
    subjects_sorted = sorted(mapping["subject"].tolist())
    if subjects_sorted != list(range(n_subjects)):
        raise ValidationError(
            f"sitio {site}: los índices subject no forman 0..{n_subjects - 1} (ROI {ref_roi})"
        )

    label_by_subject_id = ref.drop_duplicates("subject_id").set_index("subject_id")["y_true"].to_dict()
    id_by_subject = ref.drop_duplicates("subject").set_index("subject")["subject_id"].to_dict()

    for roi, df in preds_by_roi.items():
        for repeat, sub in df.groupby("repeat"):
            m = sub[["subject", "subject_id"]].drop_duplicates()
            for subj, sid in zip(m["subject"], m["subject_id"]):
                if id_by_subject.get(subj) != sid:
                    raise ValidationError(
                        f"sitio {site} ROI {roi} repeat {repeat}: subject {subj} mapea a "
                        f"subject_id {sid!r}, se esperaba {id_by_subject.get(subj)!r}"
                    )
            labels = sub.set_index("subject_id")["y_true"].to_dict()
            for sid, y in labels.items():
                if label_by_subject_id.get(sid) != y:
                    raise ValidationError(
                        f"sitio {site} ROI {roi} repeat {repeat}: subject_id {sid!r} tiene "
                        f"y_true={y}, se esperaba {label_by_subject_id.get(sid)}"
                    )
        subjects_here = sorted(df["subject"].drop_duplicates().tolist())
        if subjects_here != list(range(n_subjects)):
            raise ValidationError(f"sitio {site} ROI {roi}: los índices subject no forman 0..{n_subjects - 1}")


def metrics_from_arrays(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Definición única de las métricas (sección 7.7). Reutilizada, sin
    redefinir, por el bootstrap del segundo script."""
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    balanced_accuracy = (sensitivity + specificity) / 2
    f1_macro = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    accuracy = float((y_pred == y_true).mean())

    return {
        "auc": auc,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1_macro,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
    }


def compute_repeat_metrics(sub: pd.DataFrame) -> dict:
    y_true = sub["y_true"].to_numpy()
    y_prob = sub["y_prob"].to_numpy()
    m = metrics_from_arrays(y_true, y_prob)
    return {
        "n_subjects": len(sub),
        "n_control": int((y_true == 0).sum()),
        "n_adhd": int((y_true == 1).sum()),
        **m,
    }


def build_metrics_by_repeat(all_preds: dict[tuple[str, int], pd.DataFrame], site_order: list, roi_order: list) -> pd.DataFrame:
    rows = []
    for site in site_order:
        for roi in roi_order:
            df = all_preds[(site, roi)]
            for repeat in sorted(df["repeat"].unique()):
                sub = df[df["repeat"] == repeat]
                m = compute_repeat_metrics(sub)
                rows.append({"site": site, "roi_set": roi, "repeat": int(repeat), **m})
    out = pd.DataFrame(rows)
    col_order = [
        "site", "roi_set", "repeat", "n_subjects", "n_control", "n_adhd",
        "auc", "balanced_accuracy", "f1_macro", "sensitivity", "specificity", "accuracy",
    ]
    return out[col_order]


def build_subject_scores(all_preds: dict[tuple[str, int], pd.DataFrame], site_order: list, roi_order: list) -> pd.DataFrame:
    rows = []
    for site in site_order:
        for roi in roi_order:
            df = all_preds[(site, roi)]
            wide = df.pivot(index="subject_id", columns="repeat", values="y_prob")
            wide = wide.rename(columns={r: f"y_prob_r{r}" for r in range(1, 6)})
            y_true = df.drop_duplicates("subject_id").set_index("subject_id")["y_true"]
            wide = wide.join(y_true)
            wide = wide.sort_index(key=lambda idx: idx.astype(str))
            prob_cols = [f"y_prob_r{r}" for r in range(1, 6)]
            prob_matrix = wide[prob_cols].to_numpy()
            wide["y_prob_mean"] = prob_matrix.mean(axis=1)
            wide["y_prob_sd"] = prob_matrix.std(axis=1, ddof=0)
            wide["n_positive_predictions"] = (prob_matrix >= 0.5).sum(axis=1)
            wide = wide.reset_index()
            wide.insert(0, "roi_set", roi)
            wide.insert(0, "site", site)
            rows.append(wide[["site", "roi_set", "subject_id", "y_true"] + prob_cols +
                              ["y_prob_mean", "y_prob_sd", "n_positive_predictions"]])
    out = pd.concat(rows, ignore_index=True)
    return out


README_TABLE_HEADER_RE = re.compile(r"^\|\s*Sitio\s*\|(.+)\|\s*$")
README_ROI_COL_RE = re.compile(r"(\d+)\s*ROIs")


def parse_readme_main_table(readme_text: str) -> dict[tuple[str, int], tuple[float, float, float]]:
    lines = readme_text.splitlines()
    header_idx = None
    col_roi = {}
    for i, line in enumerate(lines):
        m = README_TABLE_HEADER_RE.match(line)
        if m:
            candidate_col_roi = {}
            headers = [h.strip() for h in line.strip().strip("|").split("|")][1:]
            for col_i, h in enumerate(headers):
                roi_m = README_ROI_COL_RE.search(h)
                if roi_m:
                    candidate_col_roi[col_i] = int(roi_m.group(1))
            # Cada línea candidata se evalúa por separado (no se acumula entre
            # líneas): evita que una línea parcial anterior "complete" sus 4
            # columnas con las de una tabla distinta más abajo en el archivo.
            if len(candidate_col_roi) == 4:
                col_roi = candidate_col_roi
                header_idx = i
                break
    if header_idx is None:
        raise ValidationError("results/README.md: no se encontró la tabla principal de AUC/balanced accuracy/accuracy")

    result: dict[tuple[str, int], tuple[float, float, float]] = {}
    i = header_idx + 1
    if i < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i].strip()):
        i += 1
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        site = cells[0]
        for col_i, roi in col_roi.items():
            cell = cells[1 + col_i]
            parts = [p.strip() for p in cell.split("/")]
            if len(parts) != 3:
                raise ValidationError(f"results/README.md: celda mal formada para {site}/{roi}: {cell!r}")
            result[(site, roi)] = (float(parts[0]), float(parts[1]), float(parts[2]))
        i += 1
    return result


def reconcile_with_readme(metrics_by_repeat: pd.DataFrame, readme_path: Path, decimals: int,
                           site_order: list, roi_order: list) -> list[dict]:
    readme_text = readme_path.read_text(encoding="utf-8")
    published = parse_readme_main_table(readme_text)

    rows = []
    for site in site_order:
        for roi in roi_order:
            key = (site, roi)
            if key not in published:
                raise ValidationError(f"results/README.md: falta la celda para {site} / {roi} ROIs")
            pub_auc, pub_bal, pub_acc = published[key]
            sub = metrics_by_repeat[(metrics_by_repeat["site"] == site) & (metrics_by_repeat["roi_set"] == roi)]
            if len(sub) != 5:
                raise ValidationError(f"{site}/{roi}: se esperaban 5 repeticiones para reconciliar, hay {len(sub)}")
            calc_auc = sub["auc"].mean() * 100
            calc_bal = sub["balanced_accuracy"].mean() * 100
            calc_acc = sub["accuracy"].mean() * 100
            calc_auc_r = round(calc_auc, decimals)
            calc_bal_r = round(calc_bal, decimals)
            calc_acc_r = round(calc_acc, decimals)
            ok = (calc_auc_r == pub_auc) and (calc_bal_r == pub_bal) and (calc_acc_r == pub_acc)
            if not ok:
                raise ValidationError(
                    f"Reconciliación README falló para {site}/{roi}: "
                    f"AUC calc={calc_auc!r} (redondeado {calc_auc_r}) vs publicado {pub_auc}; "
                    f"balanced_accuracy calc={calc_bal!r} (redondeado {calc_bal_r}) vs publicado {pub_bal}; "
                    f"accuracy calc={calc_acc!r} (redondeado {calc_acc_r}) vs publicado {pub_acc}"
                )
            rows.append({
                "site": site, "roi_set": roi,
                "published_auc": pub_auc, "published_balanced_accuracy": pub_bal, "published_accuracy": pub_acc,
                "recalculated_auc": calc_auc_r, "recalculated_balanced_accuracy": calc_bal_r,
                "recalculated_accuracy": calc_acc_r,
                "reconciliation_status": "PASS",
            })
    return rows


def resolve_and_check_output_paths(paths: list[Path], output_dir: Path) -> None:
    """CORRECCIONES_V19 §9.1: cada ruta de salida debe estar dentro del
    directorio de salida autorizado."""
    output_dir_resolved = Path(output_dir).resolve()
    for p in paths:
        resolved = Path(p).resolve()
        try:
            resolved.relative_to(output_dir_resolved)
        except ValueError:
            raise ValidationError(
                f"ruta de salida fuera del directorio autorizado: {resolved} "
                f"(se esperaba bajo {output_dir_resolved})"
            )


def preflight_outputs(paths: list[Path], output_dir: Path, overwrite: bool) -> list[str]:
    """CORRECCIONES_V19 §9.1: preflight completo antes de calcular. Devuelve
    la lista de mensajes de error (vacía si se puede proceder)."""
    resolve_and_check_output_paths(paths, output_dir)
    if overwrite:
        return []
    existing = [str(p) for p in paths if Path(p).exists()]
    return existing


def stage_csv(df: pd.DataFrame, final_path: Path) -> Path:
    """CORRECCIONES_V19 §9.2/§9.3: serializa df a un archivo temporal en el
    mismo directorio que final_path. No lo promueve al nombre final: eso
    queda a cargo del llamador, que decide cuándo hacerlo con os.replace,
    solo después de que todas las serializaciones de un lote terminaron
    correctamente."""
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".tmp", dir=str(final_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    df.to_csv(tmp_path, index=False)
    return tmp_path


def promote_staged(staged: list[tuple[Path, Path]]) -> None:
    """Promueve (os.replace) todos los pares (tmp_path, final_path) de una
    vez; si algo en el llamador falló antes de llegar aquí, ningún archivo
    final se promueve (ver limpieza de temporales en el bloque except del
    llamador)."""
    for tmp_path, final_path in staged:
        os.replace(tmp_path, final_path)


def cleanup_staged(staged: list[tuple[Path, Path]]) -> None:
    for tmp_path, _ in staged:
        tmp_path = Path(tmp_path)
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    tables_dir = output_dir / "tables"

    analysis_config = load_config_file(config_path)
    try:
        validate_analysis_config(analysis_config)
    except ValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    site_order = analysis_config["site_order"]
    roi_order = analysis_config["roi_order"]

    # CORRECCIONES_V19 §9.1: preflight completo antes de calcular nada.
    audit_path = tables_dir / "comparability_audit.csv"
    subject_scores_path = data_dir / "subject_scores.csv"
    metrics_by_repeat_path = data_dir / "metrics_by_repeat.csv"
    all_output_paths = [audit_path, subject_scores_path, metrics_by_repeat_path]
    try:
        existing = preflight_outputs(all_output_paths, output_dir, args.overwrite)
    except ValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if existing:
        print("ERROR: las siguientes salidas ya existen; use --overwrite para reemplazarlas:", file=sys.stderr)
        for p in existing:
            print(f"  - {p}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)

    audit_rows = []
    error_messages = []
    all_configs: dict[str, dict[int, dict]] = {site: {} for site in site_order}
    all_preds: dict[tuple[str, int], pd.DataFrame] = {}
    folds_hash_by_site: dict[str, dict[int, str]] = {site: {} for site in site_order}

    try:
        validate_manifest_structure(manifest, site_order, roi_order)
    except ValidationError as e:
        error_messages.append(str(e))

    included = manifest[manifest["include"]]

    for _, row in included.sort_values(["site", "roi_set"]).iterrows():
        site, roi, run_id, rel_path = row["site"], int(row["roi_set"]), row["run_id"], row["relative_path"]
        run_dir = repo_root / rel_path
        audit = {
            "site": site, "roi_set": roi, "run_id": run_id, "relative_path": rel_path,
            "missing_artifacts": "", "extra_artifacts": "",
            "config_schema_version": None, "config_hash": None,
            "n_predictions": None, "n_metrics_val_rows": None,
            "atlas_hash": None, "roi_indices_hash": None, "roi_indices_hash_expected_ok": None,
            "class_weight": None, "git_clean": None, "git_clean_accepted_exception": False,
            "folds_csv_sha256": None,
            "published_auc": None, "published_balanced_accuracy": None, "published_accuracy": None,
            "recalculated_auc": None, "recalculated_balanced_accuracy": None, "recalculated_accuracy": None,
            "reconciliation_status": "PENDING",
            "status": "PASS", "diagnostic": "",
        }
        run_errors = []
        try:
            missing, extras = validate_artifacts_contract(run_dir)
            audit["missing_artifacts"] = ";".join(missing)
            audit["extra_artifacts"] = ";".join(extras)
            if missing:
                raise ValidationError(f"{run_dir}: faltan artefactos requeridos: {missing}")
            allowed_extras = ACCEPTED_EXTRA_ARTIFACTS.get((site, roi), set())
            unexpected_extras = set(extras) - allowed_extras
            if unexpected_extras:
                raise ValidationError(
                    f"{run_dir}: artefactos extra no documentados: {sorted(unexpected_extras)} "
                    f"(solo se acepta {allowed_extras or '{}'} para (site, roi_set)={(site, roi)})"
                )

            cfg = load_config_file(run_dir / "config.json")
            validate_run_config_matches_manifest(cfg, row, run_dir)
            all_configs[site][roi] = cfg

            audit["config_schema_version"] = cfg.get("config_schema_version")
            audit["config_hash"] = cfg.get("config_hash")
            audit["atlas_hash"] = cfg.get("atlas_hash")
            audit["roi_indices_hash"] = cfg.get("roi_indices_hash")
            audit["roi_indices_hash_expected_ok"] = (cfg.get("roi_indices_hash") == ROI_INDICES_HASH.get(roi))
            audit["class_weight"] = cfg.get("class_weight")
            git_clean = cfg.get("git", {}).get("clean")
            if not isinstance(git_clean, bool):
                raise ValidationError(f"{run_dir}: git.clean={git_clean!r} no es booleano")
            audit["git_clean"] = git_clean
            audit["git_clean_accepted_exception"] = (not git_clean) and ((site, roi) in ACCEPTED_DIRTY_TREE)
            if git_clean is False and (site, roi) not in ACCEPTED_DIRTY_TREE:
                raise ValidationError(f"{run_dir}: git.clean=False sin excepción de procedencia aceptada")

            preds = load_predictions(run_dir)
            folds = load_folds(run_dir)
            audit["n_predictions"] = len(preds)
            if len(preds) != EXPECTED_PRED_COUNTS[site]:
                raise ValidationError(
                    f"{run_dir}: {len(preds)} predicciones, se esperaban {EXPECTED_PRED_COUNTS[site]}"
                )
            validate_predictions(run_dir, preds, folds, site)
            all_preds[(site, roi)] = preds

            folds_hash = sha256_file(run_dir / "folds.csv")
            audit["folds_csv_sha256"] = folds_hash
            folds_hash_by_site[site][roi] = folds_hash

            metrics_val = pd.read_csv(run_dir / "metrics_val.csv")
            audit["n_metrics_val_rows"] = len(metrics_val)
            validate_metrics_val_structure(run_dir, metrics_val, preds)

        except ValidationError as e:
            audit["status"] = "FAIL"
            audit["diagnostic"] = str(e)
            run_errors.append(str(e))

        audit_rows.append(audit)
        error_messages.extend(run_errors)

    # Comparabilidad dentro de sitio y entre sitios (solo si todas las corridas cargaron config).
    if not error_messages:
        for site in site_order:
            if set(all_configs[site]) != set(roi_order):
                error_messages.append(f"sitio {site}: no se cargaron las 4 configuraciones de ROI")
                continue
            try:
                validate_within_site_comparability(site, all_configs[site])
            except ValidationError as e:
                error_messages.append(str(e))
                for a in audit_rows:
                    if a["site"] == site:
                        a["status"] = "FAIL"
                        a["diagnostic"] = (a["diagnostic"] + "; " if a["diagnostic"] else "") + str(e)
            try:
                validate_folds_hash_within_site(site, folds_hash_by_site[site])
            except ValidationError as e:
                error_messages.append(str(e))
                for a in audit_rows:
                    if a["site"] == site:
                        a["status"] = "FAIL"
                        a["diagnostic"] = (a["diagnostic"] + "; " if a["diagnostic"] else "") + str(e)
        if not error_messages:
            try:
                validate_roi_indices_hash_across_sites(all_configs, roi_order)
            except ValidationError as e:
                error_messages.append(str(e))

    # Identidad de sujeto dentro de cada sitio.
    if not error_messages:
        for site in site_order:
            preds_by_roi = {roi: all_preds[(site, roi)] for roi in roi_order}
            try:
                validate_subject_identity(preds_by_roi, site)
            except ValidationError as e:
                error_messages.append(str(e))
                for a in audit_rows:
                    if a["site"] == site:
                        a["status"] = "FAIL"
                        a["diagnostic"] = (a["diagnostic"] + "; " if a["diagnostic"] else "") + str(e)

    metrics_by_repeat = None
    subject_scores = None
    if not error_messages:
        metrics_by_repeat = build_metrics_by_repeat(all_preds, site_order, roi_order)
        if len(metrics_by_repeat) != 80:
            error_messages.append(f"metrics_by_repeat: {len(metrics_by_repeat)} filas, se esperaban 80")

    if not error_messages:
        try:
            readme_path = repo_root / "results" / "README.md"
            recon_rows = reconcile_with_readme(
                metrics_by_repeat, readme_path, analysis_config["readme_round_decimals"], site_order, roi_order
            )
            recon_by_key = {(r["site"], r["roi_set"]): r for r in recon_rows}
            for a in audit_rows:
                key = (a["site"], a["roi_set"])
                if key in recon_by_key:
                    a.update({k: v for k, v in recon_by_key[key].items() if k not in ("site", "roi_set")})
        except ValidationError as e:
            error_messages.append(str(e))
            for a in audit_rows:
                a["reconciliation_status"] = "FAIL"
                a["diagnostic"] = (a["diagnostic"] + "; " if a["diagnostic"] else "") + str(e)

    if not error_messages:
        subject_scores = build_subject_scores(all_preds, site_order, roi_order)
        if len(subject_scores) != 1860:
            error_messages.append(f"subject_scores: {len(subject_scores)} filas, se esperaban 1860")

    # CORRECCIONES_V19 §7.6: ninguna fila puede quedar en PASS si el script
    # termina con un fallo global (por ejemplo, estructura del manifiesto o
    # conteo agregado) que no se haya atribuido ya a un sitio específico.
    if error_messages:
        global_summary = "; ".join(error_messages)
        for a in audit_rows:
            if a["status"] == "PASS":
                a["status"] = "FAIL"
                a["diagnostic"] = (
                    (a["diagnostic"] + "; " if a["diagnostic"] else "")
                    + f"fallo global de la ejecución: {global_summary}"
                )

    audit_df = pd.DataFrame(audit_rows).sort_values(["site", "roi_set"]).reset_index(drop=True)

    if error_messages:
        # CORRECCIONES_V19 §9.2: si la validación científica falla, se
        # permite escribir únicamente comparability_audit.csv, también por
        # staging + promoción.
        tables_dir.mkdir(parents=True, exist_ok=True)
        tmp = stage_csv(audit_df, audit_path)
        os.replace(tmp, audit_path)
        print("VALIDACIÓN FALLIDA. Se escribió solo la auditoría de comparabilidad.", file=sys.stderr)
        for msg in error_messages:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    if args.validate_only:
        tables_dir.mkdir(parents=True, exist_ok=True)
        tmp = stage_csv(audit_df, audit_path)
        os.replace(tmp, audit_path)
        print("Validación completa: 16/16 PASS. (--validate-only: no se construyó el dataset)")
        return 0

    # CORRECCIONES_V19 §9.2: ejecución exitosa. Serializar los tres archivos
    # a temporales dentro del directorio de salida; promoverlos a sus
    # nombres finales solo después de que las tres serializaciones terminen
    # correctamente.
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    to_write = [
        (audit_df, audit_path),
        (subject_scores, subject_scores_path),
        (metrics_by_repeat, metrics_by_repeat_path),
    ]
    staged: list[tuple[Path, Path]] = []
    try:
        for df, final_path in to_write:
            staged.append((stage_csv(df, final_path), final_path))
    except Exception:
        cleanup_staged(staged)
        raise
    promote_staged(staged)

    print("Construcción completa: 16/16 PASS.")
    print(f"  subject_scores.csv: {len(subject_scores)} filas")
    print(f"  metrics_by_repeat.csv: {len(metrics_by_repeat)} filas")
    print(f"  comparability_audit.csv: {len(audit_df)} filas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
