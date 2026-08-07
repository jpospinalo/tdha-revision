#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoría de convergencia (respaldo para Results Sec. 3.4, "Convergence and
Model Behavior").

Motivo
------
El parrafo sobre early stopping (Sec. 3.4) hace afirmaciones cuantitativas
concretas -- cuantos de los 50 pliegues externos llegaron al techo de 300
epocas, por panel de ROI -- que no tenian ninguna tabla o CSV que las
respaldara: la unica figura citada ahi (Figure 4) es un ROC, no muestra
epocas de entrenamiento. Este script cierra esa brecha, leyendo unicamente
`metrics_val.csv` y `history.csv` ya almacenados (no reentrena nada).

Verificacion cruzada
---------------------
`n_epochs` (metrics_val.csv) se verifica contra el maximo `epoch` por
(fold, repeat) en `history.csv`: deben coincidir exactamente. Si no
coinciden para alguna fila, el script aborta (posible artefacto corrupto).

Hallazgo
--------
Al construir esta tabla se encontro que el rango publicado en el manuscrito
para el panel de 116 ROI ("19 to 28 of 50 folds") no reproduce: el rango
real es [19, 43], por un valor de Peking (43/50) fuera del rango declarado.
El rango del panel de referencia (12 ROI, "44 and 50") si reproduce
exactamente. Ver docs/finalization/f3_terminologia/informe_f3_ronda2.md.

Salida
------
analysis/finalization/convergence_summary.csv
"""
from pathlib import Path
import glob
import hashlib
import csv

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = REPO_ROOT / "results" / "runs"
OUT = REPO_ROOT / "analysis" / "finalization" / "convergence_summary.csv"

# Corridas oficiales de las 16 combinaciones sitio x panel (BrainNetCNN,
# Table 4). Tres de ellas no llevan el tag control_baseline_v13 (ver
# docs/paper_reference_configuration.md Sec. 9); se resuelven por glob
# explicito, igual que en generate_algorithm_comparison_audit.py.
PATTERNS = {
    ("NYU", "12"): "NYU_rois12_w60s6_brainnetcnn_control_baseline_v13_*",
    ("Peking", "12"): "Peking_rois12_w60s6_brainnetcnn_control_baseline_v13_*",
    ("NeuroIMAGE", "12"): "NeuroIMAGE_rois12_w61s6_brainnetcnn_control_baseline_v13_*",
    ("OHSU", "12"): "OHSU_rois12_w48s5_brainnetcnn_control_baseline_v13_*",
    ("NYU", "18"): "NYU_rois18_w60s6_brainnetcnn_control_baseline_v13_*",
    ("Peking", "18"): "Peking_rois18_w60s6_brainnetcnn_control_baseline_v13_*",
    ("NeuroIMAGE", "18"): "NeuroIMAGE_rois18_w61s6_brainnetcnn_control_baseline_v13_*",
    ("OHSU", "18"): "OHSU_rois18_w48s5_brainnetcnn_2ce6c48e",
    ("NYU", "39"): "NYU_rois39_w60s6_brainnetcnn_control_base_line_1521c348",
    ("Peking", "39"): "Peking_rois39_w60s6_brainnetcnn_control_baseline_v13_*",
    ("NeuroIMAGE", "39"): "NeuroIMAGE_rois39_w61s6_brainnetcnn_control_baseline_v13_*",
    ("OHSU", "39"): "OHSU_rois39_w48s5_brainnetcnn_control_baseline_v13_*",
    ("NYU", "116"): "NYU_rois116_w60s6_brainnetcnn_control_baseline_v13_*",
    ("Peking", "116"): "Peking_rois116_w60s6_brainnetcnn_240732d1",
    ("NeuroIMAGE", "116"): "NeuroIMAGE_rois116_w61s6_brainnetcnn_control_baseline_v13_*",
    ("OHSU", "116"): "OHSU_rois116_w48s5_brainnetcnn_control_baseline_v13_*",
}


def find_run(panel: str, pattern: str) -> Path:
    matches = glob.glob(str(RUNS / panel / pattern))
    assert len(matches) == 1, f"esperaba 1 coincidencia para {panel}/{pattern}, encontre {matches}"
    return Path(matches[0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    rows = []
    for (site, panel), pattern in PATTERNS.items():
        run = find_run(panel, pattern)
        mv = pd.read_csv(run / "metrics_val.csv")
        hist = pd.read_csv(run / "history.csv")

        max_hist = hist.groupby(["fold", "repeat"])["epoch"].max().reset_index()
        max_hist.columns = ["fold", "repeat", "max_epoch_history"]
        merged = mv.merge(max_hist, on=["fold", "repeat"], how="left")
        mismatches = merged[merged["n_epochs"] != merged["max_epoch_history"]]
        assert len(mismatches) == 0, (
            f"{site} {panel}: n_epochs no coincide con el maximo epoch de history.csv "
            f"en {len(mismatches)} filas"
        )

        n_folds = len(mv)
        n_ceiling = int((mv["n_epochs"] == 300).sum())
        rows.append({
            "site": site, "roi_panel": panel, "n_folds": n_folds,
            "n_folds_at_epoch_ceiling": n_ceiling,
            "pct_at_epoch_ceiling": round(100 * n_ceiling / n_folds, 1),
            "min_n_epochs": int(mv["n_epochs"].min()),
            "run_id": run.name,
            "predictions_sha256": sha256(run / "predictions_val.csv"),
        })
        print(f"{site:12s} {panel:4s} at_ceiling={n_ceiling}/{n_folds}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print()
    for panel in ["12", "18", "39", "116"]:
        sub = df[df.roi_panel == panel]
        print(f"panel {panel}: rango n_folds_at_epoch_ceiling = "
              f"[{sub.n_folds_at_epoch_ceiling.min()}, {sub.n_folds_at_epoch_ceiling.max()}]")

    print(f"\nEscrito: {OUT}")
    print(f"SHA-256 del CSV: {sha256(OUT)}")


if __name__ == "__main__":
    main()
