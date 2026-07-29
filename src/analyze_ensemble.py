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

REQUIRED_PRED_COLUMNS = {"fold", "repeat", "subject_id", "y_true", "y_prob"}

# Campos que las corridas fuente deben compartir exactamente antes de
# combinarse. roi_set/n_rois/n_features/config_hash quedan fuera a propósito:
# son precisamente lo que varía entre las corridas que se quieren ensamblar.
IDENTITY_FIELDS = (
    "site", "bold_hash", "split_fingerprint", "seed", "n_splits", "n_repeats", "n_subjects",
)


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


def _load_run(root: Path, run_id: str) -> dict[str, Any]:
    """Lee y valida una corrida fuente: config.json y predictions_val.csv."""
    run_dir = _locate_run_dir(root, run_id)

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise EnsembleError(f"{run_id}: falta config.json en {run_dir}")
    cfg = C._read_json(cfg_path)

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

    dup = pred.duplicated(["repeat", "fold", "subject_id"])
    if dup.any():
        raise EnsembleError(
            f"{run_id}: predictions_val.csv tiene {int(dup.sum())} fila(s) duplicada(s) "
            "para la misma clave (repeat, fold, subject_id)"
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

    n_subjects = cfg.get("n_subjects")
    if isinstance(n_subjects, (int, np.integer)) and not isinstance(n_subjects, bool):
        for repeat, group in pred.groupby("repeat"):
            n_rows = len(group)
            n_unique = group["subject_id"].nunique()
            if n_rows != n_subjects or n_unique != n_subjects:
                raise EnsembleError(
                    f"{run_id}: repetición {repeat!r} tiene {n_rows} predicciones "
                    f"({n_unique} sujetos distintos), se esperaban exactamente "
                    f"{n_subjects} (un sujeto faltante, adicional o duplicado)"
                )

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
        p = r["pred"][["repeat", "fold", "subject_id", "y_true"]].copy()
        p["y_true"] = pd.to_numeric(p["y_true"], errors="coerce").astype(int)
        return p.set_index(["repeat", "fold", "subject_id"])["y_true"]

    base = runs[0]
    base_series = _claves_y_verdad(base)
    for r in runs[1:]:
        otra_series = _claves_y_verdad(r)
        if set(base_series.index) != set(otra_series.index):
            faltan = sorted(set(base_series.index) - set(otra_series.index))[:5]
            sobran = sorted(set(otra_series.index) - set(base_series.index))[:5]
            raise EnsembleError(
                f"{r['run_id']}: claves (repeat, fold, subject_id) no coinciden con "
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
    """Promedio equiponderado de y_prob sobre la clave (repeat, fold, subject_id).

    _check_compatibility() ya garantizó que todas las corridas comparten
    exactamente el mismo conjunto de claves y el mismo y_true, así que
    reindexar sobre las claves de la primera corrida no pierde ni introduce
    filas.
    """
    marcos = []
    for r in runs:
        p = r["pred"][["repeat", "fold", "subject_id", "y_true", "y_prob"]].copy()
        p["y_true"] = pd.to_numeric(p["y_true"], errors="coerce").astype(int)
        p["y_prob"] = pd.to_numeric(p["y_prob"], errors="coerce").astype(float)
        marcos.append(p.set_index(["repeat", "fold", "subject_id"]))

    indice = marcos[0].index
    matriz_prob = np.column_stack([m.reindex(indice)["y_prob"].to_numpy() for m in marcos])
    y_true = marcos[0]["y_true"].to_numpy()
    y_prob_ensemble = matriz_prob.mean(axis=1)

    salida = pd.DataFrame({
        "repeat": [k[0] for k in indice],
        "fold": [k[1] for k in indice],
        "subject_id": [k[2] for k in indice],
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


def _ensemble_dir_name(runs: list[dict[str, Any]]) -> str:
    import hashlib

    site = runs[0]["cfg"].get("site", "site")
    rois = "-".join(str(r["cfg"].get("roi_set", "?")) for r in runs)
    ids_ordenados = "|".join(sorted(r["run_id"] for r in runs))
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
        "join_keys": ["repeat", "fold", "subject_id"],
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
