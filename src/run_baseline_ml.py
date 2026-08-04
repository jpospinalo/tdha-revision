#!/usr/bin/env python3
"""Baseline de regresión logística sobre conectividad estática.

Implementa la contingencia de baseline lineal prevista en
``docs/PLAN_RESPUESTA_REVISORES.md`` §9.1, activada por la enmienda del 3 de
agosto de 2026. Reutiliza sin modificar la generación de particiones y el
cálculo de pesos de clase de ``run_experiment.py``, de modo que la
comparación con la corrida BrainNetCNN pareada del mismo sitio y grupo de
ROIs es pareada sujeto por sujeto y pliegue por pliegue.

Penalización L2 fija (``C=1.0``, sin búsqueda de hiperparámetro) y
estandarización ajustada exclusivamente con el subconjunto ``fit`` de cada
pliegue, nunca con ``inner_val`` ni ``outer_val``.

Antes de escribir cualquier artefacto, el script busca la corrida
BrainNetCNN ``static`` pareada (mismo sitio, mismo roi-set) bajo
``results/runs/`` y exige que coincidan ``split_fingerprint``, ``bold_hash``
y ``roi_indices_hash``. Si no encuentra una corrida pareada o si alguno de
los tres no coincide, el script se detiene sin escribir nada: no es una
advertencia, es una compuerta real.

Ejemplo
-------
    python run_baseline_ml.py --site OHSU --roi-set 12
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as tdha_data  # noqa: E402
from run_experiment import (  # noqa: E402
    build_split_plan,
    compute_class_weights,
    file_hash,
    indices_hash,
    split_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "runs"

FIXED_C = 1.0  # preespecificado; no se busca (nota de contingencia §9.1)
SITE_USES_CLASS_WEIGHT = {"NYU": False, "Peking": True, "NeuroIMAGE": False, "OHSU": False}
CONFIG_SCHEMA_VERSION = "baseline_ml_v1"


MANIFEST_PATH = REPO_ROOT / "analysis" / "roi_comparison" / "config" / "run_manifest.csv"


def find_paired_run(site: str, roi_set: str) -> tuple[Path, str, bool]:
    """Localiza el comparador BrainNetCNN pareado del mismo sitio y roi-set.

    Devuelve (ruta, representación_del_comparador, hay_confusión_de_representación).

    Solo existe una corrida BrainNetCNN `static` para roi_set=12 (las cuatro
    corridas de §8.1 del plan). Para 18, 39 y 116 no hay comparador `static`
    en el repositorio, y crear uno implicaría reabrir la campaña de diez
    corridas ya cerrada. En esos tres tamaños se empareja en su lugar contra
    la corrida `ordered` (dinámica) que ya es la referencia primaria de la
    Tabla 6 — la misma que usa `run_manifest.csv`. Esto introduce la MISMA
    confusión representación/arquitectura que el manuscrito ya declara y
    acepta para su propia dimensión de sensibilidad «signal representation»
    (§2.6): no aísla el factor arquitectura del factor representación. Se
    marca explícitamente en el config.json de la corrida resultante.
    """

    static_pattern = str(RESULTS_DIR / roi_set / f"{site}_rois{roi_set}_static_brainnetcnn_*")
    static_matches = sorted(glob.glob(static_pattern))
    if len(static_matches) == 1:
        return Path(static_matches[0]), "static", False
    if len(static_matches) > 1:
        raise SystemExit(
            f"ERROR: hay más de una corrida 'static' candidata: {static_matches}. "
            f"Indique una sola corrida pareada sin ambigüedad antes de continuar."
        )

    if not MANIFEST_PATH.exists():
        raise SystemExit(f"ERROR: no existe {MANIFEST_PATH} para buscar el comparador 'ordered'.")
    with open(MANIFEST_PATH, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["site"] == site and r["roi_set"] == roi_set and r["include"].lower() == "true"]
    if not rows:
        raise SystemExit(
            f"ERROR: no se encontró comparador 'static' NI 'ordered' (via run_manifest.csv) para "
            f"site={site} roi_set={roi_set}. No se puede verificar el guardarraíl de partición; "
            f"el baseline no se ejecuta sin comparador pareado."
        )
    if len(rows) > 1:
        raise SystemExit(
            f"ERROR: run_manifest.csv tiene más de una fila incluida para "
            f"site={site} roi_set={roi_set}: {[r['run_id'] for r in rows]}."
        )
    return REPO_ROOT / rows[0]["relative_path"], "ordered", True


def evaluate(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Mismas fórmulas que evaluate() en run_experiment.py: los resultados
    quedan comparables punto por punto con las corridas de BrainNetCNN."""

    labels = np.asarray(labels, dtype=np.int32).ravel()
    prediction = (probabilities >= 0.5).astype(np.int32)
    tp = int(((prediction == 1) & (labels == 1)).sum())
    tn = int(((prediction == 0) & (labels == 0)).sum())
    fp = int(((prediction == 1) & (labels == 0)).sum())
    fn = int(((prediction == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    auc = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size > 1 else float("nan")
    return {
        "accuracy": (tp + tn) / labels.size,
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "f1_macro": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "auc": auc,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", required=True, choices=["NYU", "Peking", "NeuroIMAGE", "OHSU"])
    ap.add_argument("--roi-set", required=True, choices=["12", "18", "39", "116"])
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inner-val-frac", type=float, default=0.15)
    ap.add_argument("--skip-guardrail", action="store_true",
                     help="SOLO para pruebas de humo. No usar en corridas formales.")
    args = ap.parse_args(argv)

    # ------------------------------------------------------------------
    # Guardarraíl: exigido antes de tocar los datos, salvo prueba de humo.
    # ------------------------------------------------------------------
    paired_run_dir = None
    comparator_representation = None
    representation_confound = None
    if not args.skip_guardrail:
        paired_run_dir, comparator_representation, representation_confound = find_paired_run(
            args.site, args.roi_set
        )
        paired_cfg = json.loads((paired_run_dir / "config.json").read_text())

    # ------------------------------------------------------------------
    # Datos, particiones y hashes de procedencia.
    # ------------------------------------------------------------------
    payload = tdha_data.load_bold(args.site)
    labels, subjects = payload["labels"], payload["subjects"]
    roi_idx = tdha_data.roi_indices(args.roi_set)
    roi_idx = tdha_data.validate_indices(roi_idx, payload["bold"].shape[1])

    split_plan = build_split_plan(labels, args)
    split_hash = split_fingerprint(split_plan)
    roi_hash = indices_hash(roi_idx)
    bold_path = tdha_data.BOLD_DIR / f"{args.site}.joblib"
    atlas_path = tdha_data.ATLAS_DIR / "roi_sets.json"
    bold_hash_val = file_hash(bold_path)
    atlas_hash_val = file_hash(atlas_path)

    if not args.skip_guardrail:
        mismatches = []
        if split_hash != paired_cfg["split_fingerprint"]:
            mismatches.append(f"split_fingerprint: {split_hash} != {paired_cfg['split_fingerprint']}")
        if bold_hash_val != paired_cfg["bold_hash"]:
            mismatches.append(f"bold_hash: {bold_hash_val} != {paired_cfg['bold_hash']}")
        if roi_hash != paired_cfg["roi_indices_hash"]:
            mismatches.append(f"roi_indices_hash: {roi_hash} != {paired_cfg['roi_indices_hash']}")
        if mismatches:
            raise SystemExit(
                "ERROR: el guardarraíl de partición falló contra "
                f"{paired_run_dir.name}:\n  " + "\n  ".join(mismatches) +
                "\nLa comparación no sería pareada por sujeto. No se escribe nada."
            )
        aviso_confusion = (
            " (comparador 'ordered': representación y arquitectura cambian a la vez, "
            "no se aísla el factor arquitectura — mismo aviso que la dimensión "
            "'signal representation' del manuscrito)"
            if representation_confound else ""
        )
        print(f"guardarraíl OK contra {paired_run_dir.name} [{comparator_representation}]: "
              f"split_fingerprint, bold_hash y roi_indices_hash coinciden.{aviso_confusion}")

    # ------------------------------------------------------------------
    # Características: conectividad estática, triángulo superior vectorizado.
    # ------------------------------------------------------------------
    X = tdha_data.build_flat_static_connectivity(payload["bold"], roi_idx).squeeze(1)
    n_features = X.shape[1]

    pred_rows, val_rows, train_rows, fold_rows = [], [], [], []
    use_class_weight = SITE_USES_CLASS_WEIGHT[args.site]

    for fold in split_plan:
        fit_idx, inner_idx = fold["fit"], fold["inner_val"]
        outer_train_idx, outer_val_idx = fold["outer_train"], fold["outer_val"]

        for idx, split_name in ((fit_idx, "fit"), (inner_idx, "inner_val"), (outer_val_idx, "outer_val")):
            for s in idx:
                fold_rows.append({
                    "fold": fold["fold"], "repeat": fold["repeat"],
                    "subject": int(s), "subject_id": str(subjects[s]), "split": split_name,
                })

        scaler = StandardScaler().fit(X[fit_idx])
        Xs = scaler.transform(X)

        class_weight = compute_class_weights(labels[fit_idx]) if use_class_weight else None
        clf = LogisticRegression(penalty="l2", C=FIXED_C, class_weight=class_weight,
                                  solver="lbfgs", max_iter=2000)
        clf.fit(Xs[fit_idx], labels[fit_idx])

        p_val = clf.predict_proba(Xs[outer_val_idx])[:, 1]
        p_train = clf.predict_proba(Xs[outer_train_idx])[:, 1]

        for j, subj_idx in enumerate(outer_val_idx):
            pred_rows.append({
                "fold": fold["fold"], "repeat": fold["repeat"],
                "subject": int(subj_idx), "subject_id": str(subjects[subj_idx]),
                "y_true": int(labels[subj_idx]), "y_prob": float(p_val[j]),
            })

        val_metrics = evaluate(p_val, labels[outer_val_idx])
        val_rows.append({
            "fold": fold["fold"], "repeat": fold["repeat"],
            "n_fit": len(fit_idx), "n_inner_val": len(inner_idx), "n_outer_val": len(outer_val_idx),
            **val_metrics,
        })
        train_metrics = evaluate(p_train, labels[outer_train_idx])
        train_rows.append({
            "fold": fold["fold"], "repeat": fold["repeat"],
            "n_fit": len(fit_idx), "n_inner_val": len(inner_idx), "n_outer_val": len(outer_val_idx),
            **train_metrics,
        })

    # ------------------------------------------------------------------
    # Resumen OOF por repetición (la estimación principal, igual que el resto del proyecto).
    # ------------------------------------------------------------------
    n_repeats = args.n_repeats
    oof_auc_by_repeat = []
    for r in range(1, n_repeats + 1):
        rows = [row for row in pred_rows if row["repeat"] == r]
        y_true_r = np.array([row["y_true"] for row in rows])
        y_prob_r = np.array([row["y_prob"] for row in rows])
        oof_auc_by_repeat.append(float(roc_auc_score(y_true_r, y_prob_r)))

    # ------------------------------------------------------------------
    # Escritura de artefactos.
    # ------------------------------------------------------------------
    run_id = f"{args.site}_rois{args.roi_set}_static_logreg_baseline_{split_hash[:8]}"
    run_dir = RESULTS_DIR / args.roi_set / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "run_id": run_id,
        "site": args.site,
        "roi_set": args.roi_set,
        "n_subjects": len(labels),
        "n_features": int(n_features),
        "representation": "static",
        "model": "logreg_baseline",
        "penalty": "l2",
        "C": FIXED_C,
        "hyperparameter_search": False,
        "class_weight": use_class_weight,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "inner_val_frac": args.inner_val_frac,
        "split_fingerprint": split_hash,
        "bold_hash": bold_hash_val,
        "atlas_hash": atlas_hash_val,
        "roi_indices_hash": roi_hash,
        "paired_run": paired_run_dir.name if paired_run_dir else None,
        "comparator_representation": comparator_representation,
        "representation_confound": representation_confound,
        "guardrail_verified": not args.skip_guardrail,
        "oof_auc_by_repeat": oof_auc_by_repeat,
        "oof_auc_mean": float(np.mean(oof_auc_by_repeat)),
        "python": platform.python_version(),
        "sklearn": __import__("sklearn").__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "amendment": "docs/PLAN_RESPUESTA_REVISORES.md §9.1, enmienda del 3 de agosto de 2026",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with open(run_dir / name, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("folds.csv", fold_rows)
    write_csv("predictions_val.csv", pred_rows)
    write_csv("metrics_val.csv", val_rows)
    write_csv("metrics_train.csv", train_rows)

    (run_dir / "resumen.md").write_text(
        f"# {run_id}\n\n"
        f"Baseline de regresión logística (L2, C={FIXED_C}, sin búsqueda de hiperparámetro). "
        f"Conectividad estática, {n_features} características.\n\n"
        f"AUC OOF por repetición: {', '.join(f'{a:.4f}' for a in oof_auc_by_repeat)}\n\n"
        f"AUC OOF media: {np.mean(oof_auc_by_repeat):.4f}\n\n"
        f"Guardarraíl verificado contra: {paired_run_dir.name if paired_run_dir else 'omitido (--skip-guardrail)'} "
        f"[representación del comparador: {comparator_representation}]\n\n"
        + (
            "**Confusión declarada:** el comparador es la corrida `ordered` (dinámica), no `static`. "
            "Este contraste cambia representación y arquitectura a la vez y no aísla el factor "
            "arquitectura — mismo aviso que la dimensión «signal representation» del manuscrito "
            "(§2.6). Interpretar junto con las corridas de roi_set=12, donde sí existe comparador "
            "`static` y el contraste es de un solo factor.\n\n"
            if representation_confound else
            "Comparación de un solo factor: misma representación (`static`) en ambos modelos, "
            "solo cambia la arquitectura.\n\n"
        )
        + f"split_fingerprint: {split_hash}\n\n"
        f"Enmienda: docs/PLAN_RESPUESTA_REVISORES.md §9.1, 3 de agosto de 2026.\n"
    )

    print(f"\ncorrida escrita en: {run_dir}")
    print(f"AUC OOF por repetición: {oof_auc_by_repeat}")
    print(f"AUC OOF media: {np.mean(oof_auc_by_repeat):.4f}")
    return run_id


if __name__ == "__main__":
    main()
