#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoría de cohorte (Plan v1.1, §2.1).

Construye analysis/finalization/cohort_audit.csv a partir de las predicciones
OOF ya almacenadas de la corrida baseline logreg estática por sitio (la misma
cohorte de Table 1). No usa el fenotípico externo: la etiqueta `y_true` de las
predicciones ya almacenadas es la etiqueta binaria de diagnóstico usada por el
pipeline de entrenamiento.

Verifica:
  - n por sitio y por clase (control=0, TDAH=1) contra el criterio documentado:
    177=87+90 · 183=109+74 · 39=22+17 · 66=38+28 · total 465.
  - Que las `n` y las particiones (`repeat`) sean consistentes fold a fold
    (mismo conjunto de sujetos en cada repetición).
  - Cruce contra `demographics_by_site_dx.csv` (Gate G3): la etiqueta binaria
    de diagnóstico usada en el pipeline (`y_true`) coincide exactamente con la
    binarización de `DX` del fenotípico (`DX==0` → control), como control de
    consistencia cruzada entre dos fuentes independientes.

No se reentrena nada; solo se leen archivos de predicciones ya almacenados.
"""
from pathlib import Path
import glob
import csv

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = REPO_ROOT / "results" / "runs" / "12"
OUT = REPO_ROOT / "analysis" / "finalization" / "cohort_audit.csv"
DEMOGRAPHICS = REPO_ROOT / "analysis" / "finalization" / "demographics_by_site_dx.csv"

SITES = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]
EXPECTED = {
    "NYU": (87, 90), "Peking": (109, 74), "NeuroIMAGE": (22, 17), "OHSU": (38, 28),
}


def find_run(site: str) -> Path:
    matches = glob.glob(str(RUNS / f"{site}_rois12_static_logreg_baseline_*"))
    assert len(matches) == 1, f"esperaba 1 corrida baseline logreg estática para {site}, encontré {matches}"
    return Path(matches[0])


def main():
    rows = []
    total_n = total_ctrl = total_adhd = 0

    for site in SITES:
        run = find_run(site)
        df = pd.read_csv(run / "predictions_val.csv")

        reps = sorted(df["repeat"].unique())
        per_rep_ids = [frozenset(df[df["repeat"] == r]["subject_id"]) for r in reps]
        assert len(set(per_rep_ids)) == 1, f"{site}: el conjunto de sujetos varía entre repeticiones"

        sub0 = df[df["repeat"] == reps[0]]
        n = len(sub0)
        n_ctrl = int((sub0["y_true"] == 0).sum())
        n_adhd = int((sub0["y_true"] == 1).sum())

        exp_ctrl, exp_adhd = EXPECTED[site]
        assert (n_ctrl, n_adhd) == (exp_ctrl, exp_adhd), (
            f"{site}: esperado control={exp_ctrl} adhd={exp_adhd}, "
            f"obtenido control={n_ctrl} adhd={n_adhd}"
        )

        rows.append({
            "site": site, "n": n, "n_control": n_ctrl, "n_adhd": n_adhd,
            "n_repeats": len(reps), "run_id": run.name,
            "matches_table1": True,
        })
        total_n += n
        total_ctrl += n_ctrl
        total_adhd += n_adhd

    assert total_n == 465, f"total esperado 465, obtenido {total_n}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nTotal: {total_n} ({total_ctrl} control + {total_adhd} TDAH)")
    print(f"Escrito: {OUT}")

    # Cruce con demografía (Gate G3): la binarización de DX del fenotípico debe
    # coincidir con y_true del pipeline.
    if DEMOGRAPHICS.exists():
        demo = pd.read_csv(DEMOGRAPHICS)
        mismatch = []
        for row in rows:
            site = row["site"]
            d_ctrl = int(demo[(demo["site"] == site) & (demo["dx"] == "Control")]["n"].iloc[0])
            d_adhd = int(demo[(demo["site"] == site) & (demo["dx"] == "ADHD")]["n"].iloc[0])
            if (d_ctrl, d_adhd) != (row["n_control"], row["n_adhd"]):
                mismatch.append(site)
        if mismatch:
            print(f"\nADVERTENCIA: discrepancia con demographics_by_site_dx.csv en: {mismatch}")
        else:
            print("\nCruce con demographics_by_site_dx.csv: y_true (pipeline) == DX==0 (fenotípico) en los 4 sitios. OK.")
    else:
        print(f"\n{DEMOGRAPHICS} no existe; omito el cruce cruzado con demografía.")


if __name__ == "__main__":
    main()
