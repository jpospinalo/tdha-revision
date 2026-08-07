#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera el CSV de auditoría del contraste de comparación de algoritmo
DeepSets vs BrainNetCNN a representación estática emparejada (panel de 12 ROI).

Motivo
------
La fila "DeepSets, 12 ROIs (static comparator)" de la Tabla 5 del manuscrito se
calculó durante la ronda de integración de sensibilidad pero **no quedó
persistida en ningún CSV canónico**: no aparece en `manuscript_bootstrap_10k.csv`
(que solo cubre los contrastes de regresión logística `baseline__*`) ni en
`figure4_v6_audit.csv` (que excluye deliberadamente los contrastes de tipo
"comparación de algoritmo", por la decisión v9.7 de mantenerlos solo en tabla).

Eso dejaba cuatro valores del manuscrito sin respaldo auditable, lo que
incumple el requisito de que cada fila de la tabla suplementaria corresponda
íntegramente con un archivo canónico. Este script cierra esa brecha.

Metodología
-----------
Idéntica a la del resto del manuscrito (ver Methods, Statistical Analysis):
  - AUC OOF agregada por repetición: dentro de cada repetición se agrupan las
    predicciones de los diez folds, se calcula el AUC sobre todos los
    participantes del sitio, y se promedian las cinco repeticiones.
  - Intervalos por bootstrap de participantes con reemplazo, estratificado por
    clase, pareado por sujeto entre las dos condiciones y a través de las cinco
    repeticiones. 10.000 remuestreos, NumPy PCG64, semilla 42, reiniciada en
    cada comparación. Límites = percentiles 2.5 y 97.5.
  - No se reentrena ningún modelo: se leen las predicciones ya almacenadas.

Delta reportado: DeepSets − BrainNetCNN, ambos con representación estática y
panel de 12 ROI (representación emparejada; solo cambia el algoritmo).

Provenance (2026-08-07): la corrida DeepSets estática de Peking usada aquí
es la corregida reviewer_sensitivity_weighted_fix (class_weight=True, por
política prespecificada del sitio); la corrida histórica sin weighting
queda solo como provenance. NYU/NeuroIMAGE/OHSU no cambian.

Salida
------
analysis/roi_comparison/outputs/tables/algorithm_comparison_deepsets_audit.csv

con las mismas columnas que figure4_v6_audit.csv más las rutas de las corridas
de entrada, para que la procedencia quede explícita fila por fila.
"""
from pathlib import Path
import glob
import hashlib
import csv

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RUNS = REPO_ROOT / "results" / "runs" / "12"
OUT = REPO_ROOT / "analysis" / "roi_comparison" / "outputs" / "tables" / "algorithm_comparison_deepsets_audit.csv"

SITES = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]

# Corridas estáticas de 12 ROI. Los nombres de las corridas de BrainNetCNN no son
# homogéneos entre sitios (etiquetas históricas distintas); se resuelven por glob
# explícito y se exige coincidencia única.
#
# NOTA (2026-08-07, corrección class_weight Peking): la corrida DeepSets
# estática de Peking fue originalmente ejecutada sin --class-weight,
# violando la política prespecificada del sitio (class_weight=True, Gate
# G2). Fue sustituida por una corrida corregida etiquetada
# reviewer_sensitivity_weighted_fix (mismo split_fingerprint
# =1e9626ad3839ff46, mismas particiones, solo cambia class_weight). Peking
# se fija explícitamente a ese tag para que no pueda coincidir con la
# corrida histórica superseded. NYU, NeuroIMAGE y OHSU no están afectados
# y conservan el patrón original.
DEEPSETS_STATIC = {
    "NYU": "NYU_rois12_static_deepsets_reviewer_sensitivity_*",
    "Peking": "Peking_rois12_static_deepsets_reviewer_sensitivity_weighted_fix_*",
    "NeuroIMAGE": "NeuroIMAGE_rois12_static_deepsets_reviewer_sensitivity_*",
    "OHSU": "OHSU_rois12_static_deepsets_reviewer_sensitivity_*",
}
BNN_STATIC = {
    "NYU": "NYU_rois12_static_brainnetcnn_rev32_static_r12_*",
    "Peking": "Peking_rois12_static_brainnetcnn_rev32_lstm128_ordered_*",
    "NeuroIMAGE": "NeuroIMAGE_rois12_static_brainnetcnn_rev32_static_r12_*",
    "OHSU": "OHSU_rois12_static_brainnetcnn_rev32_lstm128_ordered_*",
}


def find_run(pattern: str) -> Path:
    matches = glob.glob(str(RUNS / pattern))
    assert len(matches) == 1, f"esperaba 1 coincidencia para {pattern}, encontré {matches}"
    return Path(matches[0])


def auc_by_repeat(df: pd.DataFrame) -> float:
    aucs = []
    for _, sub in df.groupby("repeat"):
        yt = sub["y_true"].to_numpy()
        yp = sub["y_prob"].to_numpy()
        pos, neg = yp[yt == 1], yp[yt == 0]
        ranks = rankdata(np.concatenate([pos, neg]))
        u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
        aucs.append(u / (len(pos) * len(neg)))
    return float(np.mean(aucs))


def paired_bootstrap(path_a: Path, path_b: Path, n_iter: int = 10000, seed: int = 42):
    df_a = pd.read_csv(path_a / "predictions_val.csv")
    df_b = pd.read_csv(path_b / "predictions_val.csv")
    reps = sorted(df_a["repeat"].unique())
    assert reps == sorted(df_b["repeat"].unique()), "las repeticiones no coinciden"

    point_a, point_b = auc_by_repeat(df_a), auc_by_repeat(df_b)

    per_rep = []
    for rep in reps:
        sa = df_a[df_a["repeat"] == rep].set_index("subject_id")
        sb = df_b[df_b["repeat"] == rep].set_index("subject_id")
        subs = sorted(sa.index)
        assert subs == sorted(sb.index), f"los sujetos no coinciden en repeat {rep}"
        yt = sa.loc[subs, "y_true"].to_numpy()
        per_rep.append((
            np.where(yt == 1)[0], np.where(yt == 0)[0],
            sa.loc[subs, "y_prob"].to_numpy(), sb.loc[subs, "y_prob"].to_numpy(),
        ))

    rng = np.random.Generator(np.random.PCG64(seed))
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        ra, rb = [], []
        for idx_pos, idx_neg, ypa, ypb in per_rep:
            sp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            sn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            n_pos = len(sp)

            def auc(yp):
                ranks = rankdata(np.concatenate([yp[sp], yp[sn]]))
                u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2
                return u / (n_pos * len(sn))

            ra.append(auc(ypa))
            rb.append(auc(ypb))
        deltas[i] = np.mean(ra) - np.mean(rb)

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point_a, point_b, point_a - point_b, lo, hi


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    rows = []
    for site in SITES:
        a = find_run(DEEPSETS_STATIC[site])
        b = find_run(BNN_STATIC[site])
        auc_a, auc_b, delta, lo, hi = paired_bootstrap(a, b)
        rows.append({
            "figure": "Table5",
            "site": site,
            "condition": "DeepSets, 12 ROIs (static comparator)",
            "point": delta,
            "ci_low": lo,
            "ci_high": hi,
            "status": "evaluated",
            "auc_deepsets": auc_a,
            "auc_brainnetcnn": auc_b,
            "run_deepsets": a.name,
            "run_brainnetcnn": b.name,
            "predictions_sha256_deepsets": sha256(a / "predictions_val.csv"),
            "predictions_sha256_brainnetcnn": sha256(b / "predictions_val.csv"),
        })
        print(f"  {site:11s} DeepSets={auc_a*100:5.1f}%  BrainNetCNN={auc_b*100:5.1f}%  "
              f"Δ={delta*100:+5.1f}pp [{lo*100:+5.1f}, {hi*100:+5.1f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nEscrito: {OUT}")
    print(f"SHA-256 del CSV: {sha256(OUT)}")


if __name__ == "__main__":
    main()
