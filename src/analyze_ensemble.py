#!/usr/bin/env python3
"""Ensamble por promedio de probabilidades OOF entre corridas ya entrenadas.

Combina, por **promedio equiponderado simple** de `y_prob`, las predicciones de
validación externa (`predictions_val.csv`) de 2 o 3 corridas ya entrenadas. No
entrena nada, no optimiza pesos ni el umbral de clasificación (fijo en 0.5), y
no es una corrida: no produce `metrics_train.csv`, `history.csv` ni ningún
metadato de `EarlyStopping`. Es un análisis exploratorio y post hoc, separado
del entrenamiento, que vive en su propia carpeta bajo
``results/analyses/ensembles/`` — nunca dentro de ``results/runs/``.

Antes de combinar, exige que las corridas fuente compartan identidad de datos
y de partición (``site``, ``bold_hash``, ``split_fingerprint``, ``seed``,
``n_splits``, ``n_repeats``, ``n_subjects``, y las mismas claves externas
``(repeat, fold, subject_id)`` con el mismo ``y_true``) — pero **no** exige que
compartan ``roi_set``, ``n_rois``, ``n_features`` ni ``config_hash``, porque el
objetivo es precisamente combinar corridas de distinto conjunto de ROIs sobre
la misma partición de sujetos.

Uso::

    python analyze_ensemble.py --root results/runs \\
        --runs RUN_ID_1 RUN_ID_2 \\
        --out results/analyses/ensembles
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compile_results as C
import run_experiment as R

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "results" / "runs"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "results" / "analyses" / "ensembles"

REQUIRED_PRED_COLUMNS = {"fold", "repeat", "subject", "subject_id", "y_true", "y_prob"}

# Clave externa completa de una predicción: `subject` es el índice interno
# (posición del sujeto en los datos del sitio, estable entre corridas del
# mismo sitio con distinto roi_set — verificado empíricamente sobre NYU 12 vs
# 18: coincide en 885/885 claves) y `subject_id` es la identidad legible.
# Exigir ambos —no sustituir uno por el otro— reduce el riesgo de un
# pareamiento silenciosamente incorrecto si algún día dejan de coincidir.
JOIN_KEYS = ("repeat", "fold", "subject", "subject_id")

# Campos que las corridas fuente deben compartir exactamente antes de
# combinarse. roi_set/n_rois/n_features/config_hash quedan fuera a propósito:
# son precisamente lo que varía entre las corridas que se quieren ensamblar.
IDENTITY_FIELDS = (
    "site", "bold_hash", "split_fingerprint", "seed", "n_splits", "n_repeats", "n_subjects",
)

# n_splits/n_repeats/n_subjects deben ser enteros (no booleanos) en o por
# encima de este mínimo — mismos umbrales que exige compile_results.py para
# config.json de corridas de esquema 4, para no inventar un contrato distinto
# para el mismo campo.
STRUCTURAL_INT_FIELDS = (("n_splits", 2), ("n_repeats", 1), ("n_subjects", 1))


class EnsembleError(ValueError):
    """Cualquier incompatibilidad entre corridas fuente o en sus artefactos.

    Se usa una excepción propia (en vez de dejar escapar KeyError/TypeError
    de pandas) para que main() pueda convertirla en un mensaje de CLI legible
    y para que las pruebas puedan distinguir "corrida incompatible" (detectada
    a propósito) de un error real de programación.
    """


def _locate_run_dir(root: Path, run_id: str) -> Path:
    """Ubica la carpeta de una corrida por su ``run_id``.

    A diferencia de ``compile_results._find_run_dir()`` — que prioriza en
    silencio el layout plano histórico para no romper la recolección masiva
    de ``collect()`` —, aquí una corrida presente en más de una ubicación es
    un error: no hay forma segura de saber cuál de las dos copias es la que
    se quiere combinar en un ensamble, así que se rechaza en vez de elegir
    una arbitrariamente. Cubre tanto ``root/<run_id>/`` (histórico) como
    ``root/<roi_set>/<run_id>/`` (actual).
    """
    root = Path(root)
    candidatos: list[Path] = []
    plano = root / run_id
    if plano.exists():
        candidatos.append(plano)
    candidatos.extend(sorted(root.glob(f"*/{run_id}")))
    if not candidatos:
        raise EnsembleError(f"{run_id}: no se encontró bajo {root}")
    if len(candidatos) > 1:
        raise EnsembleError(
            f"{run_id}: aparece en más de una ubicación bajo {root} "
            f"({[str(c) for c in candidatos]}) — no se elige una copia arbitrariamente"
        )
    return candidatos[0]


def _check_structural_fields(run_id: str, cfg: dict[str, Any]) -> None:
    """Campos obligatorios de config.json antes de aceptar la corrida.

    site/bold_hash/split_fingerprint/seed deben estar presentes (la igualdad
    entre corridas se exige aparte, en _check_compatibility()).
    n_splits/n_repeats/n_subjects deben ser enteros — no booleanos — en o por
    encima del mínimo correspondiente; mismos umbrales que
    compile_results.validate_run_artifacts() usa para config_schema_version
    >= 4, para no inventar un contrato distinto para el mismo campo.
    """
    for campo in ("site", "bold_hash", "split_fingerprint", "seed"):
        if cfg.get(campo) is None:
            raise EnsembleError(f"{run_id}: config.json: falta el campo {campo!r}")
    for campo, minimo in STRUCTURAL_INT_FIELDS:
        valor = cfg.get(campo)
        if (
            campo not in cfg
            or isinstance(valor, bool)
            or not isinstance(valor, (int, np.integer))
            or int(valor) < minimo
        ):
            raise EnsembleError(
                f"{run_id}: config.json: {campo} debe ser un entero >= {minimo}; "
                f"se recibió {valor!r}"
            )


def _check_run_identity(run_id: str, run_dir: Path, cfg: dict[str, Any]) -> None:
    """El run_id solicitado, el nombre de la carpeta y config.json['run_id']
    deben ser exactamente el mismo texto. _locate_run_dir() ya garantiza que
    la carpeta encontrada se llama ``run_id`` (así es como la busca), así que
    lo único que falta comprobar es que config.json no declare uno distinto
    — la misma protección que compile_results.validate_run_artifacts() exige
    para las corridas guardadas en resultados/runs, aplicada aquí a las
    corridas fuente de un ensamble.
    """
    declarado = cfg.get("run_id")
    if not isinstance(declarado, str) or not declarado:
        raise EnsembleError(
            f"{run_id}: config.json: run_id debe ser una cadena no vacía; "
            f"se recibió {declarado!r} (carpeta: {run_dir.name})"
        )
    if declarado != run_id or run_dir.name != run_id:
        raise EnsembleError(
            f"identificador solicitado={run_id!r}, carpeta={run_dir.name!r}, "
            f"config.json declara run_id={declarado!r} — no coinciden"
        )


def _check_coverage(run_id: str, pred: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Cobertura completa de repeticiones y folds declarados por config.json.

    Reproduce y corrige el caso reportado (n_repeats=2 en config.json, pero
    predictions_val.csv solo contiene la repetición 1): antes solo se
    validaba el número de sujetos DENTRO de las repeticiones presentes, sin
    comprobar que estuvieran las n_repeats repeticiones ni los n_splits
    folds de cada una.

    Los identificadores de repetición sí son 1..n_repeats en los datos
    reales del proyecto y se exigen así. Los de fold NO: run_experiment.py
    numera los folds de forma corrida a lo largo de toda la validación
    repetida (con n_splits=10, la repetición 2 usa los folds 11-20, no
    1-10 — verificado en corridas reales), así que aquí solo se exige la
    CANTIDAD de folds distintos por repetición, no una etiqueta concreta.
    """
    n_splits = int(cfg["n_splits"])
    n_repeats = int(cfg["n_repeats"])
    n_subjects = int(cfg["n_subjects"])

    repeticiones_presentes = sorted(int(r) for r in pred["repeat"].unique())
    esperadas = list(range(1, n_repeats + 1))
    if repeticiones_presentes != esperadas:
        faltantes = sorted(set(esperadas) - set(repeticiones_presentes))
        sobrantes = sorted(set(repeticiones_presentes) - set(esperadas))
        raise EnsembleError(
            f"{run_id}: cobertura de repeticiones incompleta — se esperaban "
            f"{esperadas}, hay {repeticiones_presentes} (faltan {faltantes}, "
            f"sobran {sobrantes})"
        )

    pares_totales = 0
    for repeat, group in pred.groupby("repeat"):
        folds_repeticion = group["fold"].unique()
        if len(folds_repeticion) != n_splits:
            raise EnsembleError(
                f"{run_id}: repetición {int(repeat)} tiene {len(folds_repeticion)} "
                f"fold(s) distinto(s), se esperaban exactamente {n_splits} "
                "(cobertura de folds incompleta)"
            )
        pares_totales += len(folds_repeticion)
        for columna in ("subject", "subject_id"):
            n_unico = group[columna].nunique()
            if len(group) != n_subjects or n_unico != n_subjects:
                raise EnsembleError(
                    f"{run_id}: repetición {int(repeat)} tiene {len(group)} predicciones "
                    f"({n_unico} valores únicos de {columna}), se esperaban exactamente "
                    f"{n_subjects} (un sujeto faltante, adicional o duplicado)"
                )
    if pares_totales != n_splits * n_repeats:
        raise EnsembleError(
            f"{run_id}: {pares_totales} pares (repeat, fold) distintos en total, "
            f"se esperaban {n_splits * n_repeats} (= n_splits × n_repeats)"
        )


def _check_longitudinal_consistency(run_id: str, pred: pd.DataFrame) -> None:
    """Dentro de UNA misma corrida: (subject, subject_id) debe ser una
    correspondencia estable 1 a 1 entre repeticiones, cada sujeto debe tener
    un único y_true en todo el archivo, y todas las repeticiones deben cubrir
    exactamente el mismo conjunto de sujetos. Se comprueba antes de comparar
    nada contra la otra corrida fuente.
    """
    por_par = pred.groupby(["subject", "subject_id"])["y_true"].nunique()
    inconsistentes = por_par[por_par > 1]
    if not inconsistentes.empty:
        ejemplos = inconsistentes.index.tolist()[:5]
        raise EnsembleError(
            f"{run_id}: y_true no es consistente entre repeticiones para "
            f"(subject, subject_id) {ejemplos}"
        )

    # subject_id -> subject debe ser una función (y viceversa): un subject_id
    # no puede apuntar a más de un subject interno dentro de la misma corrida,
    # ni un subject a más de un subject_id.
    por_subject_id = pred.groupby("subject_id")["subject"].nunique()
    ambiguos = por_subject_id[por_subject_id > 1]
    if not ambiguos.empty:
        raise EnsembleError(
            f"{run_id}: subject_id con más de un subject interno asociado: "
            f"{ambiguos.index.tolist()[:5]}"
        )
    por_subject = pred.groupby("subject")["subject_id"].nunique()
    ambiguos2 = por_subject[por_subject > 1]
    if not ambiguos2.empty:
        raise EnsembleError(
            f"{run_id}: subject con más de un subject_id asociado: "
            f"{ambiguos2.index.tolist()[:5]}"
        )

    # Las comprobaciones anteriores (y _check_coverage(), que ya verificó la
    # CANTIDAD de sujetos por repetición) no bastan para detectar que dos
    # repeticiones tengan el mismo número de sujetos pero conjuntos
    # distintos — p. ej. repetición 1 con sujetos A y B, repetición 2 con A
    # y C. El ensamble exige cobertura longitudinal idéntica: cada
    # repetición debe contener exactamente el mismo conjunto de pares
    # (subject, subject_id) que la primera.
    repeticiones = sorted(int(r) for r in pred["repeat"].unique())
    if repeticiones:
        referencia = repeticiones[0]
        conjunto_referencia = set(
            map(
                tuple,
                pred.loc[pred["repeat"] == referencia, ["subject", "subject_id"]].to_numpy(),
            )
        )
        for repeat in repeticiones[1:]:
            conjunto_actual = set(
                map(
                    tuple,
                    pred.loc[pred["repeat"] == repeat, ["subject", "subject_id"]].to_numpy(),
                )
            )
            if conjunto_actual != conjunto_referencia:
                faltantes = sorted(conjunto_referencia - conjunto_actual, key=str)[:5]
                adicionales = sorted(conjunto_actual - conjunto_referencia, key=str)[:5]
                raise EnsembleError(
                    f"{run_id}: la repetición {repeat} no cubre el mismo conjunto de "
                    f"sujetos (subject, subject_id) que la repetición {referencia} — "
                    f"faltan {faltantes}, sobran {adicionales}"
                )


def _load_run(root: Path, run_id: str) -> dict[str, Any]:
    """Lee y valida una corrida fuente: config.json y predictions_val.csv."""
    run_dir = _locate_run_dir(root, run_id)

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise EnsembleError(f"{run_id}: falta config.json en {run_dir}")
    cfg = C._read_json(cfg_path)

    _check_run_identity(run_id, run_dir, cfg)
    _check_structural_fields(run_id, cfg)

    pred_path = run_dir / "predictions_val.csv"
    if not pred_path.exists():
        raise EnsembleError(f"{run_id}: falta predictions_val.csv en {run_dir}")
    try:
        pred = pd.read_csv(pred_path)
    except Exception as exc:
        raise EnsembleError(f"{run_id}: no se pudo leer predictions_val.csv ({exc})") from exc

    missing = REQUIRED_PRED_COLUMNS - set(pred.columns)
    if missing:
        raise EnsembleError(f"{run_id}: predictions_val.csv: faltan columnas {sorted(missing)}")

    dup = pred.duplicated(list(JOIN_KEYS))
    if dup.any():
        raise EnsembleError(
            f"{run_id}: predictions_val.csv tiene {int(dup.sum())} fila(s) duplicada(s) "
            f"para la misma clave {JOIN_KEYS}"
        )

    y_prob = pd.to_numeric(pred["y_prob"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y_prob).all() or (y_prob < 0).any() or (y_prob > 1).any():
        raise EnsembleError(
            f"{run_id}: predictions_val.csv tiene y_prob no finito o fuera de [0, 1]"
        )

    y_true_num = pd.to_numeric(pred["y_true"], errors="coerce")
    if not np.isfinite(y_true_num.to_numpy(dtype=float)).all() or not set(
        y_true_num.unique()
    ) <= {0, 1}:
        raise EnsembleError(f"{run_id}: predictions_val.csv tiene y_true fuera de {{0, 1}}")

    _check_coverage(run_id, pred, cfg)
    _check_longitudinal_consistency(run_id, pred)

    return {"run_id": run_id, "run_dir": run_dir, "cfg": cfg, "pred": pred}


def _check_compatibility(runs: list[dict[str, Any]]) -> None:
    """Aborta si las corridas fuente no son combinables. No genera ningún
    resultado parcial: se llama antes de tocar disco de salida.
    """
    if len(runs) < 2 or len(runs) > 3:
        raise EnsembleError(f"se admiten entre 2 y 3 corridas fuente, se recibieron {len(runs)}")

    for field in IDENTITY_FIELDS:
        valores = {r["run_id"]: r["cfg"].get(field) for r in runs}
        distintos = {v for v in valores.values()}
        if len(distintos) > 1 or None in distintos:
            raise EnsembleError(f"las corridas no comparten {field!r}: {valores}")

    def _claves_y_verdad(r: dict[str, Any]) -> pd.Series:
        p = r["pred"][list(JOIN_KEYS) + ["y_true"]].copy()
        p["y_true"] = pd.to_numeric(p["y_true"], errors="coerce").astype(int)
        return p.set_index(list(JOIN_KEYS))["y_true"]

    base = runs[0]
    base_series = _claves_y_verdad(base)
    for r in runs[1:]:
        otra_series = _claves_y_verdad(r)
        if set(base_series.index) != set(otra_series.index):
            faltan = sorted(set(base_series.index) - set(otra_series.index))[:5]
            sobran = sorted(set(otra_series.index) - set(base_series.index))[:5]
            raise EnsembleError(
                f"{r['run_id']}: claves {JOIN_KEYS} no coinciden con "
                f"{base['run_id']} — ejemplo de faltantes {faltan}, de sobrantes {sobran}"
            )
        alineada = otra_series.reindex(base_series.index)
        desacuerdo = base_series != alineada
        if desacuerdo.any():
            ejemplos = base_series.index[desacuerdo][:5].tolist()
            raise EnsembleError(
                f"{r['run_id']}: y_true no coincide con {base['run_id']} para las claves "
                f"{ejemplos}"
            )


def _combine(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Promedio equiponderado de y_prob sobre la clave JOIN_KEYS.

    _check_compatibility() ya garantizó que todas las corridas comparten
    exactamente el mismo conjunto de claves y el mismo y_true, así que
    reindexar sobre las claves de la primera corrida no pierde ni introduce
    filas. El orden de `runs` no afecta el resultado (es un promedio), pero
    run_ensemble() ya lo recibe canonicalizado para que la salida también
    sea determinista en nombre y metadatos.
    """
    marcos = []
    for r in runs:
        p = r["pred"][list(JOIN_KEYS) + ["y_true", "y_prob"]].copy()
        p["y_true"] = pd.to_numeric(p["y_true"], errors="coerce").astype(int)
        p["y_prob"] = pd.to_numeric(p["y_prob"], errors="coerce").astype(float)
        marcos.append(p.set_index(list(JOIN_KEYS)))

    indice = marcos[0].index
    matriz_prob = np.column_stack([m.reindex(indice)["y_prob"].to_numpy() for m in marcos])
    y_true = marcos[0]["y_true"].to_numpy()
    y_prob_ensemble = matriz_prob.mean(axis=1)

    salida = pd.DataFrame({
        **{clave: [k[i] for k in indice] for i, clave in enumerate(JOIN_KEYS)},
        "y_true": y_true,
        "y_prob": y_prob_ensemble,
    })
    return salida.sort_values(["repeat", "fold", "subject_id"]).reset_index(drop=True)


def _metrics_oof_by_repeat(predictions: pd.DataFrame) -> pd.DataFrame:
    """Métricas OOF por repetición del ensamble.

    accuracy/auc/f1_macro/balanced_accuracy/log_loss/brier reutilizan
    compile_results.oof_metrics_per_repetition() (misma función que usa el
    resto del proyecto para esta agregación, no una segunda implementación).
    sensibilidad y especificidad no están en esa función porque las corridas
    normales no las reportan hoy; se calculan aquí con el mismo umbral 0.5.
    """
    base = C.oof_metrics_per_repetition(predictions)
    if base is None:
        raise EnsembleError("no se pudieron calcular métricas OOF del ensamble combinado")

    filas_extra = []
    for repeat, group in predictions.groupby("repeat"):
        y = pd.to_numeric(group["y_true"], errors="coerce").to_numpy()
        p = pd.to_numeric(group["y_prob"], errors="coerce").to_numpy()
        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask].astype(int)
        p = p[mask].astype(float)
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        filas_extra.append({
            "repeat": int(repeat),
            "sensitivity": (tp / (tp + fn)) if (tp + fn) else float("nan"),
            "specificity": (tn / (tn + fp)) if (tn + fp) else float("nan"),
        })
    extra = pd.DataFrame(filas_extra)
    return base.merge(extra, on="repeat", how="left")


SUMMARY_METRICS = (
    "accuracy", "auc", "f1_macro", "balanced_accuracy",
    "sensitivity", "specificity", "log_loss", "brier",
)


def _summary_stats(per_repeat: pd.DataFrame) -> dict[str, dict[str, float]]:
    resumen: dict[str, dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        if metric not in per_repeat:
            continue
        serie = per_repeat[metric]
        resumen[metric] = {
            "mean": float(serie.mean()),
            "sd": float(serie.std(ddof=1)) if serie.notna().sum() > 1 else float("nan"),
        }
    return resumen


def _canonical_order(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordena las corridas fuente de forma determinista, independiente del
    orden con el que se hayan pasado en --runs. El promedio no depende del
    orden, pero el nombre de la carpeta de salida, `source_runs` y `weights`
    en config.json sí dependían del orden de entrada antes de esta función:
    ``--runs A B`` y ``--runs B A`` podían producir dos carpetas de salida
    distintas para el mismo análisis. La clave de orden es
    (roi_set numérico si es posible, si no como texto; run_id) para que sea
    estable y reproducible.
    """
    def clave(r: dict[str, Any]) -> tuple[Any, str]:
        roi_set = r["cfg"].get("roi_set")
        try:
            roi_key: Any = (0, int(roi_set))
        except (TypeError, ValueError):
            roi_key = (1, str(roi_set))
        return (roi_key, r["run_id"])

    return sorted(runs, key=clave)


def _ensemble_dir_name(runs: list[dict[str, Any]]) -> str:
    """Asume que `runs` ya está en orden canónico (ver _canonical_order):
    con eso, invertir --runs A B / --runs B A produce el mismo nombre.
    """
    import hashlib

    site = runs[0]["cfg"].get("site", "site")
    rois = "-".join(str(r["cfg"].get("roi_set", "?")) for r in runs)
    ids_ordenados = "|".join(r["run_id"] for r in runs)
    h = hashlib.sha256(ids_ordenados.encode("utf-8")).hexdigest()[:8]
    return f"{site}_ensemble_rois{rois}_{h}"


def _write_config(out_dir: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    base_cfg = runs[0]["cfg"]
    peso = 1.0 / len(runs)
    config = {
        "artifact_type": "oof_probability_ensemble",
        "ensemble_id": out_dir.name,
        "source_runs": [
            {
                "run_id": r["run_id"],
                "config_hash": r["cfg"].get("config_hash"),
                "roi_set": r["cfg"].get("roi_set"),
            }
            for r in runs
        ],
        "weights": {r["run_id"]: peso for r in runs},
        "weights_type": "equal",
        "threshold": 0.5,
        "join_keys": list(JOIN_KEYS),
        "site": base_cfg.get("site"),
        "n_subjects": base_cfg.get("n_subjects"),
        "n_splits": base_cfg.get("n_splits"),
        "n_repeats": base_cfg.get("n_repeats"),
        "split_fingerprint": base_cfg.get("split_fingerprint"),
        "bold_hash": base_cfg.get("bold_hash"),
        "analysis_code_hash": R.file_hash(__file__),
        "git": R.git_info(),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    (out_dir / "config.json").write_text(
        __import__("json").dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return config


def _write_resumen(out_dir: Path, runs: list[dict[str, Any]], summary: dict[str, dict[str, float]]) -> None:
    site = runs[0]["cfg"].get("site")
    rois = ", ".join(str(r["cfg"].get("roi_set")) for r in runs)
    lines = [
        f"# Ensamble OOF — {site}, ROIs {rois}",
        "",
        "**Análisis exploratorio y post hoc.** Combina, por promedio equiponderado "
        "de probabilidad, las predicciones de validación externa (OOF) de corridas "
        "ya entrenadas. No es una corrida entrenada — no hay entrenamiento, "
        "`EarlyStopping` ni selección de hiperparámetros aquí — y este resultado "
        "no debe presentarse como una confirmación independiente de las corridas "
        "fuente.",
        "",
        "## Corridas fuente",
        "",
    ]
    for r in runs:
        cfg = r["cfg"]
        lines.append(
            f"- `{r['run_id']}` (config_hash=`{cfg.get('config_hash')}`, "
            f"roi_set={cfg.get('roi_set')})"
        )
    lines += [
        "",
        f"Pesos: iguales ({1.0/len(runs):.4f} cada una, {len(runs)} corridas). "
        "Umbral de clasificación: 0.5 (fijo, sin optimizar).",
        "",
        "## Métricas OOF por repetición (media ± desviación estándar)",
        "",
        "| Métrica | Media ± DE |",
        "|---|---:|",
    ]
    etiquetas = {
        "accuracy": "Accuracy", "auc": "AUC", "f1_macro": "F1-macro",
        "balanced_accuracy": "Balanced accuracy", "sensitivity": "Sensibilidad",
        "specificity": "Especificidad", "log_loss": "Log-loss", "brier": "Brier",
    }
    for metric, etiqueta in etiquetas.items():
        if metric not in summary:
            continue
        stats = summary[metric]
        escala = 1.0 if metric in ("log_loss", "brier") else 100.0
        unidad = "" if metric in ("log_loss", "brier") else " %"
        lines.append(
            f"| {etiqueta} | {stats['mean'] * escala:.2f} ± {stats['sd'] * escala:.2f}{unidad} |"
        )
    lines += [
        "",
        "Ver `metrics_oof_by_repeat.csv` para el detalle por repetición y "
        "`predictions_val.csv` para las probabilidades combinadas. `config.json` "
        "es la fuente de verdad de esta carpeta de análisis.",
        "",
    ]
    (out_dir / "resumen.md").write_text("\n".join(lines), encoding="utf-8")


def run_ensemble(root: Path, run_ids: list[str], out_root: Path, *, overwrite: bool) -> Path:
    """Punto de entrada reutilizable por main() y por las pruebas."""
    if len(run_ids) != len(set(run_ids)):
        raise EnsembleError("--runs tiene identificadores repetidos")

    runs = [_load_run(root, run_id) for run_id in run_ids]
    _check_compatibility(runs)
    # A partir de aquí el orden es canónico (roi_set, run_id), no el orden en
    # que se pasaron en --runs: --runs A B y --runs B A deben producir el
    # mismo directorio de salida, el mismo source_runs/weights y las mismas
    # probabilidades (el promedio ya era invariante al orden; lo que no lo
    # era es el nombre y los metadatos).
    runs = _canonical_order(runs)

    out_dir = out_root / _ensemble_dir_name(runs)
    if out_dir.exists():
        if not overwrite:
            raise EnsembleError(f"{out_dir} ya existe; use --overwrite para sustituirla")
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble_pred = _combine(runs)
    per_repeat = _metrics_oof_by_repeat(ensemble_pred)
    summary = _summary_stats(per_repeat)

    ensemble_pred.to_csv(out_dir / "predictions_val.csv", index=False)
    per_repeat.to_csv(out_dir / "metrics_oof_by_repeat.csv", index=False)
    _write_config(out_dir, runs)
    _write_resumen(out_dir, runs, summary)

    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="raíz de results/runs")
    parser.add_argument(
        "--runs", nargs="+", required=True, metavar="RUN_ID",
        help="2 o 3 run_id de corridas fuente (promedio equiponderado de y_prob)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="raíz de salida del análisis")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="sustituir una salida existente para la misma combinación de corridas",
    )
    args = parser.parse_args(argv)

    if not (2 <= len(args.runs) <= 3):
        print(f"--runs requiere 2 o 3 identificadores de corrida, se recibieron {len(args.runs)}",
              file=sys.stderr)
        return 1

    try:
        out_dir = run_ensemble(
            Path(args.root), list(args.runs), Path(args.out), overwrite=args.overwrite
        )
    except EnsembleError as exc:
        print(f"ensamble abortado: {exc}", file=sys.stderr)
        return 1

    print(f"ensamble escrito en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
