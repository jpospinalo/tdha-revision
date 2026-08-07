#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye la tabla de demografía por sitio y diagnóstico (Gate G3) a partir del
fenotípico consolidado de ADHD-200.

El archivo fuente NO se versiona en este repositorio (ver
docs/data_provenance/adhd200_phenotypics.md para la justificación: datos
individuales sujetos a los términos de acceso de ADHD-200/INDI). Este script
recibe su ruta local, verifica su hash contra el valor documentado, y produce
únicamente una tabla agregada por sitio × diagnóstico (sin identificadores
individuales) como salida versionable.

Uso:
    python analysis/finalization/build_demographics.py /ruta/local/adhd200_preprocessed_phenotypics.tsv

Codificación verificada contra la ADHD-200 Phenotypic Key (ADHD-200 Consortium /
NITRC) y, de forma independiente, contra el cruce con las predicciones OOF ya
almacenadas (ver docs/data_provenance/adhd200_phenotypics.md, §2).
"""
from pathlib import Path
import argparse
import glob
import hashlib
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = REPO_ROOT / "results" / "runs" / "12"
OUT = REPO_ROOT / "analysis" / "finalization" / "demographics_by_site_dx.csv"

EXPECTED_SHA256 = "7a37195f0260b04246b833ff4b8050afc4756b8a8f1622feca52944189f5a898"

SITE_CODE = {"NYU": 5, "Peking": 1, "NeuroIMAGE": 4, "OHSU": 6}
SITES = list(SITE_CODE.keys())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def cohort_ids(site: str) -> list[str]:
    """IDs numéricos (sin prefijo de sitio) de la cohorte de análisis oficial."""
    matches = glob.glob(str(RUNS / f"{site}_rois12_static_logreg_baseline_*" / "predictions_val.csv"))
    assert len(matches) == 1, f"esperaba 1 corrida baseline logreg estática para {site}, encontré {matches}"
    df = pd.read_csv(matches[0])
    subs = sorted(set(df["subject_id"]))
    return [s.split("-", 1)[1] for s in subs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phenotype_path", type=Path,
                     help="Ruta local al fenotípico consolidado de ADHD-200 (no versionado)")
    args = ap.parse_args()

    if not args.phenotype_path.exists():
        sys.exit(f"No existe: {args.phenotype_path}")

    actual_hash = sha256(args.phenotype_path)
    if actual_hash != EXPECTED_SHA256:
        sys.exit(
            f"SHA-256 no coincide con el documentado en "
            f"docs/data_provenance/adhd200_phenotypics.md.\n"
            f"  esperado: {EXPECTED_SHA256}\n"
            f"  obtenido: {actual_hash}\n"
            f"Abortando: no se genera la tabla con un archivo fuente no verificado."
        )

    pheno = pd.read_csv(args.phenotype_path, sep="\t", dtype={"ScanDir ID": str})

    rows = []
    total_n = 0
    for site in SITES:
        ids = cohort_ids(site)
        code = SITE_CODE[site]
        sub = pheno[(pheno["Site"] == code) & (pheno["ScanDir ID"].isin(ids))].copy()
        assert len(sub) == len(ids), (
            f"{site}: cohorte OOF tiene {len(ids)} sujetos, "
            f"cruce con fenotípico dio {len(sub)}. G3 no reproduce; abortando."
        )
        sub["dx_bin"] = sub["DX"].astype(str).map(lambda d: "Control" if d == "0" else "ADHD")
        for dxb in ["Control", "ADHD"]:
            g = sub[sub["dx_bin"] == dxb]
            n = len(g)
            total_n += n
            rows.append({
                "site": site,
                "dx": dxb,
                "n": n,
                "age_mean": round(g["Age"].mean(), 2) if n else None,
                "age_sd": round(g["Age"].std(), 2) if n else None,
                "age_min": round(g["Age"].min(), 2) if n else None,
                "age_max": round(g["Age"].max(), 2) if n else None,
                "n_male": int((g["Gender"] == 1).sum()),
                "n_female": int((g["Gender"] == 0).sum()),
                "pct_male": round(100 * (g["Gender"] == 1).sum() / n, 1) if n else None,
                "age_missing": int(g["Age"].isna().sum()),
                "gender_missing": int(g["Gender"].isna().sum()),
            })

    assert total_n == 465, f"total esperado 465, obtenido {total_n}"

    out_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)

    print(out_df.to_string(index=False))
    print(f"\nEscrito: {OUT}")
    print(f"Total: {total_n}/465")
    print(f"SHA-256 fuente verificado: {actual_hash}")
    print(f"SHA-256 salida: {sha256(OUT)}")


if __name__ == "__main__":
    main()
