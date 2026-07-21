#!/usr/bin/env python3
"""
Compila las corridas de `results/runs/` en una tabla única.

Pensado para el escenario en que varias personas corren experimentos en paralelo y
hacen push al mismo repositorio. Cada corrida vive en su propia carpeta con su
`config.json`, así que compilar es solo recorrer y agregar: no hay archivos
compartidos que fusionar.

Antes de agregar, verifica que las corridas seleccionadas sean **comparables**:

- Mismas señales BOLD por sitio (`bold_hash`). Si alguien regeneró un archivo, avisa
  en lugar de promediar cosas distintas.
- Misma semilla y mismo número de pliegues, sin lo cual la comparación deja de ser
  pareada.
- Mismo enventanado (`window`, `step`).
- Código confirmado en git al momento de correr.

Uso
    python compile_results.py
    python compile_results.py --site NYU --model lstm --stats
    python compile_results.py --out ../results/tabla.csv
"""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "results" / "runs"
METRICS = ["accuracy", "precision", "recall", "auc"]


def specificity(df):
    denom = df.true_negatives + df.false_positives
    return df.true_negatives / denom.replace(0, np.nan)


def summarize(run_dir, cfg, suffix=""):
    val_f = run_dir / f"metrics_val{suffix}.csv"
    tr_f = run_dir / f"metrics_train{suffix}.csv"
    if not (val_f.exists() and tr_f.exists()):
        return None
    va, tr = pd.read_csv(val_f), pd.read_csv(tr_f)
    row = {
        "run_id": cfg.get("run_id", run_dir.name) + suffix,
        "site": cfg.get("site"), "roi_set": cfg.get("roi_set"),
        "window": cfg.get("window"), "step": cfg.get("step"),
        "model": cfg.get("model"), "seed": cfg.get("seed"),
        "n_windows": cfg.get("n_windows"), "n_rois": cfg.get("n_rois"),
        "n_folds": len(va), "random_subset": cfg.get("random_subset"),
        "usuario": cfg.get("git", {}).get("user"),
        "commit": (cfg.get("git", {}).get("commit") or "")[:8],
        "arbol_limpio": cfg.get("git", {}).get("clean"),
        "bold_hash": cfg.get("bold_hash"),
    }
    for m in METRICS:
        if m in va:
            row[f"train_{m}"] = tr[m].mean() * 100
            row[f"val_{m}"] = va[m].mean() * 100
            row[f"val_{m}_sd"] = va[m].std() * 100
    if "true_negatives" in va:
        row["train_sp"] = specificity(tr).mean() * 100
        row["val_sp"] = specificity(va).mean() * 100
    row["start_from_epoch"] = cfg.get("start_from_epoch")
    row["gap_acc"] = row.get("train_accuracy", np.nan) - row.get("val_accuracy", np.nan)
    row["epoca_media"] = va.best_epoch.mean() if "best_epoch" in va else np.nan
    return row


def collect(root):
    rows = []
    for cfg_path in sorted(Path(root).glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        d = cfg_path.parent
        for f in sorted(d.glob("metrics_val*.csv")):
            sfx = f.name.replace("metrics_val", "").replace(".csv", "")
            r = summarize(d, cfg, sfx)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def check_comparability(df):
    """Avisos si el conjunto seleccionado no es directamente comparable."""
    problemas = []
    por_sitio = defaultdict(set)
    for _, r in df.iterrows():
        por_sitio[r.site].add(r.bold_hash)
    for sitio, hashes in por_sitio.items():
        if len(hashes) > 1:
            problemas.append(f"las señales BOLD de {sitio} cambiaron entre corridas "
                             f"({len(hashes)} versiones): alguien regeneró el archivo")
    for col, etiqueta in [("seed", "semillas"), ("n_folds", "número de pliegues"),
                          ("window", "longitud de ventana"), ("step", "paso de ventana"),
                          ("start_from_epoch", "época mínima antes de la parada")]:
        if col in df and df[col].nunique() > 1:
            problemas.append(f"{etiqueta} distintas: {sorted(df[col].dropna().unique())}")
    if "arbol_limpio" in df and df.arbol_limpio.eq(False).any():
        sucias = df.loc[df.arbol_limpio == False, "run_id"].tolist()
        problemas.append(f"corridas hechas con cambios sin confirmar: {sucias}")
    return problemas


def paired_stats(root, runs):
    """ANOVA de medidas repetidas y contrastes pareados con corrección de Holm.

    Se usan pruebas pareadas y no un Tukey clásico porque todas las corridas comparten
    sujetos y particiones: las observaciones no son independientes.
    """
    from scipy import stats
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multitest import multipletests

    acc = {k: pd.read_csv(Path(root) / rid / "metrics_val.csv")
                 .sort_values("fold").accuracy.values
           for k, rid in runs.items()}
    n = {k: len(v) for k, v in acc.items()}
    if len(set(n.values())) > 1:
        print(f"  no se puede: número de pliegues distinto entre corridas {n}")
        return

    long = pd.concat([pd.DataFrame({"acc": v, "grupo": str(k),
                                    "fold": np.arange(1, len(v) + 1)})
                      for k, v in acc.items()])
    print("\nANOVA de medidas repetidas:\n")
    print(AnovaRM(long, "acc", "fold", within=["grupo"]).fit())

    res = []
    for a, b in itertools.combinations(acc, 2):
        t, p_t = stats.ttest_rel(acc[a], acc[b])
        try:
            _, p_w = stats.wilcoxon(acc[a], acc[b])
        except ValueError:
            p_w = np.nan
        res.append({"grupo_1": a, "grupo_2": b,
                    "dif_pp": (acc[b].mean() - acc[a].mean()) * 100,
                    "p_t_pareada": p_t, "p_wilcoxon": p_w})
    res = pd.DataFrame(res)
    res["p_holm"] = multipletests(res.p_t_pareada, method="holm")[1]
    res["significativo"] = res.p_holm < 0.05
    print("\nContrastes pareados con corrección de Holm:\n")
    print(res.round(4).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--site")
    p.add_argument("--model")
    p.add_argument("--roi-set", nargs="*")
    p.add_argument("--out", default=None, help="guardar la tabla en CSV")
    p.add_argument("--stats", action="store_true",
                   help="ANOVA de medidas repetidas y contrastes pareados entre "
                        "subconjuntos de ROIs (requiere un solo sitio y un solo modelo)")
    args = p.parse_args()

    df = collect(args.root)
    if df.empty:
        raise SystemExit(f"No se encontraron corridas en {args.root}")
    if args.site:
        df = df[df.site == args.site]
    if args.model:
        df = df[df.model == args.model]
    if args.roi_set:
        df = df[df.roi_set.astype(str).isin([str(r) for r in args.roi_set])]
    if df.empty:
        raise SystemExit("Ninguna corrida coincide con los filtros")

    cols = ["run_id", "site", "roi_set", "window", "step", "model", "n_folds",
            "n_windows", "train_accuracy", "val_accuracy", "val_accuracy_sd",
            "gap_acc", "val_auc", "epoca_media", "usuario", "commit"]
    print(df[[c for c in cols if c in df.columns]].round(2).to_string(index=False))

    problemas = check_comparability(df)
    if problemas:
        print("\n" + "=" * 70 + "\nAVISOS DE COMPARABILIDAD\n" + "=" * 70)
        for x in problemas:
            print(f"  · {x}")
    else:
        print("\nTodas las corridas seleccionadas son directamente comparables.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nTabla guardada en {args.out}")

    if args.stats:
        base = df[df.random_subset.isna()]
        if base.site.nunique() != 1 or base.model.nunique() != 1:
            raise SystemExit("Para --stats filtre a un solo sitio y un solo modelo "
                             "con --site y --model")
        runs = {r.roi_set: r.run_id for _, r in base.sort_values("n_rois").iterrows()}
        if len(runs) < 2:
            raise SystemExit("Se necesitan al menos dos subconjuntos de ROIs")
        paired_stats(args.root, runs)


if __name__ == "__main__":
    main()