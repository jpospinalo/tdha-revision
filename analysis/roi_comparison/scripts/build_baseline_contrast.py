#!/usr/bin/env python3
"""Contraste bootstrap del baseline de regresión logística (§9.1, enmienda
del 3 de agosto de 2026) frente a su comparador BrainNetCNN pareado.

Deliberadamente SEPARADO de build_analysis_dataset.py / run_statistical_analysis.py:
esos dos scripts están anclados a la comparación primaria de 16 corridas
(analysis_plan.md con hash canónico verificado, EXPECTED_SUBJECT_COUNTS,
ROI_INDICES_HASH y demás constantes específicas de esa campaña). El baseline
es un análisis nuevo, activado por una enmienda posterior, sobre 16 corridas
distintas (logreg vs. comparador BrainNetCNN) que ese pipeline no conoce y no
debe conocer -- extenderlo ahí violaría el propio candado que lo protege.

Lee:
  analysis/roi_comparison/config/baseline_manifest.csv (16 filas, generado a
    partir de los config.json de las corridas logreg_baseline)
  predictions_val.csv de cada corrida baseline y de su comparador pareado

No lee ni modifica nada del pipeline primario (subject_scores.csv,
metrics_by_repeat.csv, comparability_audit.csv, analysis_manifest.json).

Escribe (solo si todas las verificaciones pasan):
  analysis/roi_comparison/outputs/baseline/data/baseline_contrast_results.json
    (mismo esquema que new_contrasts_results.json: ref_run, new_run,
    n_subjects, n_iter, ref_auc_point, ref_ci, ref_reps, new_auc_point,
    new_ci, new_reps, delta, delta_ci, más comparator_representation y
    representation_confound)
  analysis/roi_comparison/outputs/baseline/tables/baseline_contrast.csv
  analysis/roi_comparison/outputs/baseline/baseline_contrast_manifest.json
    (procedencia: hashes de entrada, versiones, parámetros del bootstrap)

Bootstrap: pareado por sujeto, estratificado por clase, PCG64, seed=42,
reset por sitio, 2000 remuestreos -- misma especificación documentada en
docs/Guia_implementacion_baseline_ML.md §6 y ya usada para los diez
contrastes de sensibilidad del manuscrito.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metrics_from_arrays():
    """Importa metrics_from_arrays del script de Fase 1 del pipeline primario,
    sin duplicar la definición de las métricas (misma disciplina que
    run_statistical_analysis.py aplica para el resto del módulo)."""
    primary_scripts_dir = SCRIPT_DIR
    # Cuando este script vive en analysis/roi_comparison/scripts/, el import
    # directo funciona. Si se ejecuta desde otra ubicación, se busca el
    # repositorio hacia arriba.
    candidates = [primary_scripts_dir, primary_scripts_dir.parent / "scripts"]
    for c in candidates:
        if (c / "build_analysis_dataset.py").exists():
            sys.path.insert(0, str(c))
            from build_analysis_dataset import metrics_from_arrays  # noqa
            return metrics_from_arrays
    raise SystemExit("No se encontró build_analysis_dataset.py junto a este script.")


def load_predictions(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "predictions_val.csv", dtype={"subject_id": str})
    required_cols = {"fold", "repeat", "subject_id", "y_true", "y_prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"{run_dir}: predictions_val.csv sin columnas {missing}")
    return df


def per_repeat_auc(preds: pd.DataFrame, metrics_from_arrays) -> tuple[list, list]:
    """Devuelve (subject_ids ordenados, [auc_r1..auc_r5]) agregando OOF por
    repetición -- misma agregación que build_metrics_by_repeat del pipeline
    primario (metric_then_mean: se calcula la métrica sobre todas las
    predicciones OOF de la repetición, no el promedio de métricas por pliegue)."""
    reps = []
    for r in range(1, 6):
        sub = preds[preds["repeat"] == r]
        m = metrics_from_arrays(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy())
        reps.append(float(m["auc"]))
    return reps


def build_subject_tensor(preds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """(n_subjects, 5) de y_prob por repetición, y_true (n_subjects,),
    subject_ids ordenados ascendentemente -- para que el bootstrap remuestree
    sujetos, no filas OOF sueltas."""
    ref = preds[preds["repeat"] == 1].sort_values("subject_id", key=lambda s: s.astype(str))
    subject_ids = ref["subject_id"].tolist()
    y_true = ref["y_true"].to_numpy()
    probs = np.empty((len(subject_ids), 5), dtype=np.float64)
    for r in range(1, 6):
        sub = preds[preds["repeat"] == r].set_index("subject_id").loc[subject_ids]
        if not np.array_equal(sub["y_true"].to_numpy(), y_true):
            raise SystemExit(f"y_true inconsistente entre repeticiones para el mismo sujeto")
        probs[:, r - 1] = sub["y_prob"].to_numpy()
    return probs, y_true, subject_ids


def bilateral_ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def bootstrap_contrast(
    ref_probs: np.ndarray, new_probs: np.ndarray, y_true: np.ndarray,
    n_iter: int, seed: int, metrics_from_arrays=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap pareado estratificado por clase, semilla reiniciada en cada
    llamada (reset_per_site: cada sitio construye su propio
    default_rng(seed)). Devuelve (ref_draws, new_draws, delta_draws), cada
    uno de forma (n_iter,): AUC media de las 5 repeticiones por remuestreo.

    Usa ``roc_auc_score`` directamente en vez de ``metrics_from_arrays``
    (que también calcula balanced_accuracy/f1_macro/sensitivity/specificity,
    ninguno de los cuales se usa aquí): es exactamente la misma fórmula --
    ``metrics_from_arrays`` calcula ``auc = roc_auc_score(y_true, y_prob)``
    sin transformación adicional, verificado leyendo su código fuente -- pero
    sin el costo de las métricas secundarias, que en 16 combinaciones x 2.000
    remuestreos x 2 modelos x 5 repeticiones (320.000 llamadas) es la
    diferencia entre minutos y no terminar dentro del límite de la
    herramienta. El parámetro ``metrics_from_arrays`` se conserva por
    compatibilidad de firma pero no se usa."""
    rng = np.random.default_rng(seed)
    control_idx = np.flatnonzero(y_true == 0)
    adhd_idx = np.flatnonzero(y_true == 1)
    n_control, n_adhd = len(control_idx), len(adhd_idx)

    ref_draws = np.empty(n_iter, dtype=np.float64)
    new_draws = np.empty(n_iter, dtype=np.float64)

    for it in range(n_iter):
        boot_control = control_idx[rng.integers(0, n_control, size=n_control)]
        boot_adhd = adhd_idx[rng.integers(0, n_adhd, size=n_adhd)]
        boot_idx = np.concatenate([boot_control, boot_adhd])
        y_boot = y_true[boot_idx]

        ref_rep_aucs = np.empty(5)
        new_rep_aucs = np.empty(5)
        for r in range(5):
            ref_rep_aucs[r] = roc_auc_score(y_boot, ref_probs[boot_idx, r])
            new_rep_aucs[r] = roc_auc_score(y_boot, new_probs[boot_idx, r])
        ref_draws[it] = ref_rep_aucs.mean()
        new_draws[it] = new_rep_aucs.mean()

    delta_draws = new_draws - ref_draws
    return ref_draws, new_draws, delta_draws


def process_manifest_rows(manifest_subset, repo_root, metrics_from_arrays, n_iter, seed):
    """Procesa un subconjunto de filas del manifiesto (usado tanto para la
    corrida completa como para el modo por sitio --only-site, que existe
    solo para poder terminar cada tramo dentro del límite de tiempo de una
    herramienta de shell en sandbox; el resultado numérico es idéntico en
    ambos modos porque cada sitio reinicia su propio generador de números
    aleatorios -- reset_per_site -- y no depende de los demás sitios)."""
    results = {}
    csv_rows = []
    input_hashes = {}

    for _, row in manifest_subset.sort_values(["site", "roi_set"]).iterrows():
        site = row["site"]
        roi_set = int(row["roi_set"])
        baseline_dir = repo_root / row["baseline_relative_path"]
        comparator_dir = repo_root / row["comparator_relative_path"]

        # ---- Guardarraíl independiente: split_fingerprint pareado ----------
        baseline_cfg = json.loads((baseline_dir / "config.json").read_text())
        comparator_cfg = json.loads((comparator_dir / "config.json").read_text())
        if baseline_cfg["split_fingerprint"] != comparator_cfg["split_fingerprint"]:
            raise SystemExit(
                f"ERROR: split_fingerprint no coincide para {site}/{roi_set}: "
                f"baseline={baseline_cfg['split_fingerprint']} "
                f"comparador={comparator_cfg['split_fingerprint']}. No se escribe nada."
            )
        if baseline_cfg["roi_indices_hash"] != comparator_cfg["roi_indices_hash"]:
            raise SystemExit(f"ERROR: roi_indices_hash no coincide para {site}/{roi_set}. No se escribe nada.")

        baseline_preds = load_predictions(baseline_dir)
        comparator_preds = load_predictions(comparator_dir)

        baseline_probs, baseline_y, baseline_ids = build_subject_tensor(baseline_preds)
        comparator_probs, comparator_y, comparator_ids = build_subject_tensor(comparator_preds)

        if baseline_ids != comparator_ids:
            raise SystemExit(f"ERROR: sujetos distintos entre baseline y comparador en {site}/{roi_set}.")
        if not np.array_equal(baseline_y, comparator_y):
            raise SystemExit(f"ERROR: y_true distinto entre baseline y comparador en {site}/{roi_set}.")

        ref_reps = per_repeat_auc(comparator_preds, metrics_from_arrays)
        new_reps = per_repeat_auc(baseline_preds, metrics_from_arrays)

        ref_draws, new_draws, delta_draws = bootstrap_contrast(
            comparator_probs, baseline_probs, baseline_y, n_iter, seed
        )

        ref_auc_point = float(np.mean(ref_reps))
        new_auc_point = float(np.mean(new_reps))
        ref_ci = bilateral_ci(ref_draws)
        new_ci = bilateral_ci(new_draws)
        delta = new_auc_point - ref_auc_point
        delta_ci = bilateral_ci(delta_draws)

        roi_key = str(roi_set)
        results.setdefault(site, {})[roi_key] = {
            "ref_run": row["comparator_run_id"],
            "new_run": row["baseline_run_id"],
            "n_subjects": len(baseline_ids),
            "n_iter": n_iter,
            "ref_auc_point": ref_auc_point,
            "ref_ci": list(ref_ci),
            "ref_reps": ref_reps,
            "new_auc_point": new_auc_point,
            "new_ci": list(new_ci),
            "new_reps": new_reps,
            "delta": delta,
            "delta_ci": list(delta_ci),
            "comparator_representation": row["comparator_representation"],
            "representation_confound": bool(row["representation_confound"]),
        }
        csv_rows.append({
            "site": site, "roi_set": roi_set,
            "ref_run": row["comparator_run_id"], "new_run": row["baseline_run_id"],
            "comparator_representation": row["comparator_representation"],
            "representation_confound": bool(row["representation_confound"]),
            "n_subjects": len(baseline_ids), "n_iter": n_iter,
            "ref_auc_point": ref_auc_point, "ref_ci_low": ref_ci[0], "ref_ci_high": ref_ci[1],
            "new_auc_point": new_auc_point, "new_ci_low": new_ci[0], "new_ci_high": new_ci[1],
            "delta": delta, "delta_ci_low": delta_ci[0], "delta_ci_high": delta_ci[1],
        })

        input_hashes[f"{site}_{roi_set}"] = {
            "baseline_config_sha256": sha256_file(baseline_dir / "config.json"),
            "baseline_predictions_sha256": sha256_file(baseline_dir / "predictions_val.csv"),
            "comparator_config_sha256": sha256_file(comparator_dir / "config.json"),
            "comparator_predictions_sha256": sha256_file(comparator_dir / "predictions_val.csv"),
        }

        print(f"[contraste] {site}/{roi_set}: ref={ref_auc_point:.4f} new={new_auc_point:.4f} "
              f"delta={delta:+.4f} ({row['comparator_representation']}"
              f"{', confusión declarada' if row['representation_confound'] else ''})")

    return results, csv_rows, input_hashes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-site", default=None, choices=["NYU", "Peking", "NeuroIMAGE", "OHSU"],
        help=(
            "Procesa un solo sitio (4 filas del manifiesto) y escribe un checkpoint parcial "
            "en outputs/baseline/.checkpoints/. Existe solo para poder terminar cada tramo "
            "dentro del limite de tiempo de una herramienta de shell en sandbox; no cambia el "
            "resultado numerico (reset_per_site: cada sitio es independiente). No documentado "
            "como interfaz de linea de comandos normal -- el uso normal es sin argumentos."
        ),
    )
    parser.add_argument(
        "--finalize", action="store_true",
        help="Combina los 4 checkpoints de --only-site en las salidas finales y los borra.",
    )
    args = parser.parse_args(argv)

    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = SCRIPT_DIR.parent.parent.parent  # scripts/ -> roi_comparison/ -> analysis/ -> repo

    manifest_path = repo_root / "analysis" / "roi_comparison" / "config" / "baseline_manifest.csv"
    if not manifest_path.exists():
        print(f"ERROR: no se encontró {manifest_path}", file=sys.stderr)
        return 1

    manifest = pd.read_csv(manifest_path, dtype=str)
    manifest["roi_set"] = manifest["roi_set"].astype(int)
    manifest["representation_confound"] = manifest["representation_confound"].map(
        {"True": True, "False": False}
    )
    if len(manifest) != 16:
        print(f"ERROR: baseline_manifest.csv tiene {len(manifest)} filas, se esperaban 16", file=sys.stderr)
        return 1

    output_dir = repo_root / "analysis" / "roi_comparison" / "outputs" / "baseline"
    data_dir = output_dir / "data"
    tables_dir = output_dir / "tables"
    checkpoints_dir = output_dir / ".checkpoints"
    for d in (data_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    json_out_path = data_dir / "baseline_contrast_results.json"
    csv_out_path = tables_dir / "baseline_contrast.csv"
    manifest_out_path = output_dir / "baseline_contrast_manifest.json"

    if args.only_site:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        metrics_from_arrays = load_metrics_from_arrays()
        subset = manifest[manifest["site"] == args.only_site]
        results, csv_rows, input_hashes = process_manifest_rows(
            subset, repo_root, metrics_from_arrays, args.n_iter, args.seed
        )
        checkpoint_path = checkpoints_dir / f"{args.only_site}.json"
        checkpoint_path.write_text(
            json.dumps({"results": results, "csv_rows": csv_rows, "input_hashes": input_hashes},
                       ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nCheckpoint escrito: {checkpoint_path} ({len(csv_rows)} filas)")
        return 0

    if args.finalize:
        site_order = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]
        results, csv_rows, input_hashes = {}, [], {}
        for site in site_order:
            p = checkpoints_dir / f"{site}.json"
            if not p.exists():
                print(f"ERROR: falta el checkpoint de {site} ({p}). No se finaliza.", file=sys.stderr)
                return 1
            chunk = json.loads(p.read_text(encoding="utf-8"))
            results.update(chunk["results"])
            csv_rows.extend(chunk["csv_rows"])
            input_hashes.update(chunk["input_hashes"])
        if not args.overwrite:
            existing = [p for p in (json_out_path, csv_out_path, manifest_out_path) if p.exists()]
            if existing:
                print("ERROR: ya existen salidas; use --overwrite:", file=sys.stderr)
                for p in existing:
                    print(f"  - {p}", file=sys.stderr)
                return 1
    else:
        if not args.overwrite:
            existing = [p for p in (json_out_path, csv_out_path, manifest_out_path) if p.exists()]
            if existing:
                print("ERROR: ya existen salidas; use --overwrite:", file=sys.stderr)
                for p in existing:
                    print(f"  - {p}", file=sys.stderr)
                return 1
        metrics_from_arrays = load_metrics_from_arrays()
        results, csv_rows, input_hashes = process_manifest_rows(
            manifest, repo_root, metrics_from_arrays, args.n_iter, args.seed
        )

    if len(csv_rows) != 16:
        print(f"ERROR: se procesaron {len(csv_rows)} filas, se esperaban 16. No se escribe nada.", file=sys.stderr)
        return 1

    json_out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    pd.DataFrame(csv_rows).sort_values(["site", "roi_set"]).to_csv(csv_out_path, index=False)

    import sklearn
    manifest_out = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_manifest": str(manifest_path.relative_to(repo_root)),
        "source_manifest_sha256": sha256_file(manifest_path),
        "n_combinations": len(csv_rows),
        "bootstrap": {
            "iterations": args.n_iter, "seed": args.seed, "rng": "numpy_pcg64",
            "seed_scope": "reset_per_site", "method": "paired_stratified_percentile",
            "quantile_method": "linear", "ci_level": 0.95,
        },
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "input_hashes": input_hashes,
        "scope": (
            "Contraste del baseline de regresion logistica (docs/PLAN_RESPUESTA_REVISORES.md "
            "§9.1, enmienda del 3 de agosto de 2026) frente a su comparador BrainNetCNN pareado. "
            "Independiente del pipeline primario de 16 corridas (analysis_plan.md, plan 5.6); "
            "no lo modifica ni depende de sus salidas."
        ),
        "representation_confound_note": (
            "4 de las 16 filas (roi_set=12) comparan static vs static: un solo factor cambia, "
            "la arquitectura. Las otras 12 (roi_set en {18,39,116}) comparan contra el "
            "comparador ordered de la Tabla 6, porque no existe corrida BrainNetCNN static para "
            "esos tamanos; en esas 12 filas representation_confound=true y arquitectura y "
            "representacion cambian a la vez (ver docs/Guia_implementacion_baseline_ML.md §0)."
        ),
        "team_review_status": "pendiente: resultados no revisados por el equipo todavia",
    }
    manifest_out_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.finalize:
        import shutil
        shutil.rmtree(checkpoints_dir, ignore_errors=True)
        print(f"Checkpoints en {checkpoints_dir} eliminados tras finalizar.")

    print(f"\nEscrito: {json_out_path}")
    print(f"Escrito: {csv_out_path}")
    print(f"Escrito: {manifest_out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
