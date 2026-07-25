#!/usr/bin/env python3
"""
Verificación del repositorio y del entorno.

Comprueba que todo lo necesario para correr experimentos esté en su sitio y sea
coherente. Pensado para ejecutarse justo después de clonar, antes de lanzar nada.

    cd src
    python verify_setup.py            # comprobaciones rápidas
    python verify_setup.py --full     # añade una prueba de entrenamiento real

Sin --full no se importa TensorFlow, así que sirve también para revisar el
repositorio en un entorno sin GPU ni Keras instalados.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ok_count = 0
fail_count = 0
warn_count = 0


def ok(msg):
    global ok_count
    ok_count += 1
    print(f"  \033[32m✓\033[0m {msg}")


def fail(msg):
    global fail_count
    fail_count += 1
    print(f"  \033[31m✗\033[0m {msg}")


def warn(msg):
    global warn_count
    warn_count += 1
    print(f"  \033[33m!\033[0m {msg}")


def seccion(t):
    print(f"\n{t}\n" + "-" * len(t))


# --------------------------------------------------------------------------- #

def check_estructura():
    seccion("Estructura de archivos")
    esperados = [
        "README.md", "requirements.txt", "tdha_experimentos.ipynb",
        "src/data.py", "src/run_experiment.py", "src/compile_results.py",
        "src/kerasmodels/__init__.py", "src/kerasmodels/lstm.py",
        "src/kerasmodels/gru.py", "src/kerasmodels/cnn1d.py",
        "src/kerasmodels/transformer.py",
        "data/atlas/aal116.csv", "data/atlas/roi_sets.json",
    ]
    for rel in esperados:
        (ok if (REPO / rel).exists() else fail)(rel)
    if (REPO / "results" / "runs").is_dir():
        ok("results/runs/")
    else:
        fail("results/runs/ no existe (¿falta el .gitkeep?)")


def check_datos():
    seccion("Señales BOLD")
    import numpy as np
    import joblib

    esperado = {"NYU": (177, 116, 172), "Peking": (183, 116, 232),
                "NeuroIMAGE": (39, 116, 257), "OHSU": (66, 116, 74)}
    for site, forma in esperado.items():
        p = REPO / "data" / "bold" / f"{site}.joblib"
        if not p.exists():
            fail(f"{site}: falta {p.name}")
            continue
        if p.open("rb").read(64).startswith(b"version https://git-lfs"):
            fail(f"{site}: es un puntero de Git LFS. Ejecute 'git lfs pull'")
            continue
        try:
            d = joblib.load(p)
        except Exception as e:
            fail(f"{site}: no se pudo leer ({type(e).__name__}: {e})")
            continue
        faltan = {"subjects", "bold", "labels", "roi_names"} - set(d)
        if faltan:
            fail(f"{site}: faltan claves {sorted(faltan)}")
            continue
        b = np.asarray(d["bold"])
        y = np.asarray(d["labels"])
        detalle = f"{b.shape}, clases {dict(zip(*[x.tolist() for x in np.unique(y, return_counts=True)]))}"
        if b.shape != forma:
            warn(f"{site}: forma {b.shape}, se esperaba {forma}")
        elif len(y) != b.shape[0]:
            fail(f"{site}: {b.shape[0]} sujetos pero {len(y)} etiquetas")
        elif len(d["roi_names"]) != b.shape[1]:
            fail(f"{site}: {b.shape[1]} ROIs pero {len(d['roi_names'])} nombres")
        else:
            ok(f"{site}: {detalle}")


def check_roi_sets():
    seccion("Subconjuntos de ROIs")
    import joblib
    import numpy as np

    sets = json.loads((REPO / "data" / "atlas" / "roi_sets.json").read_text("utf-8"))
    nombres = joblib.load(REPO / "data" / "bold" / "NYU.joblib")["roi_names"]
    for k in sorted(sets, key=int):
        v = sets[k]
        idx = np.asarray(v["indices"])
        problemas = []
        if len(idx) != v["n"]:
            problemas.append(f"n={v['n']} pero {len(idx)} índices")
        if len(set(idx.tolist())) != len(idx):
            problemas.append("índices repetidos")
        if idx.min() < 0 or idx.max() >= len(nombres):
            problemas.append("índices fuera de rango")
        if not problemas and [nombres[i] for i in idx] != v["names"]:
            problemas.append("los índices no corresponden a los nombres declarados")
        (fail if problemas else ok)(
            f"{k:>4s} ROIs" + (f": {'; '.join(problemas)}" if problemas else ""))


def check_secuencias():
    seccion("Construcción de secuencias")
    import numpy as np
    sys.path.insert(0, str(REPO / "src"))
    import data as D

    b = D.load_bold("NYU")
    for rs, W, S, nw in [("12", 70, 2, 52), ("18", 70, 2, 52), ("39", 70, 4, 26)]:
        idx = D.roi_indices(rs)
        X = D.build_sequences(b["bold"], idx, W, S)
        r = len(idx)
        esperado = (b["bold"].shape[0], nw, r, r)
        if X.shape != esperado:
            fail(f"{rs} ROIs, ventana {W}/{S}: forma {X.shape}, se esperaba {esperado}")
            continue
        simetrica = np.allclose(X, np.swapaxes(X, -1, -2), atol=1e-5)
        diag = np.allclose(np.diagonal(X, axis1=-2, axis2=-1), 1.0, atol=1e-4)
        rango = float(X.min()) >= -1.0001 and float(X.max()) <= 1.0001
        finito = bool(np.isfinite(X).all())
        prob = [n for n, c in [("no simétrica", simetrica), ("diagonal≠1", diag),
                               ("fuera de [-1,1]", rango), ("valores no finitos", finito)]
                if not c]
        (fail if prob else ok)(
            f"{rs} ROIs, ventana {W}/{S}: {X.shape}" + (f" — {', '.join(prob)}" if prob else ""))
        tri = D.upper_triangle(X)
        exp_tri = (esperado[0], nw, r * (r - 1) // 2)
        (ok if tri.shape == exp_tri else fail)(f"      triángulo superior {tri.shape}")


def check_representaciones():
    seccion("Representaciones adicionales")
    import numpy as np
    sys.path.insert(0, str(REPO / "src"))
    import data as D

    b = D.load_bold("NYU")
    idx = D.roi_indices("18")
    r = len(idx)
    F = r * (r - 1) // 2
    n = b["bold"].shape[0]

    P = D.build_flat_partial_connectivity(b["bold"], idx)
    prob = []
    if P.shape != (n, 1, F):
        prob.append(f"forma {P.shape}")
    if not np.isfinite(P).all():
        prob.append("valores no finitos")
    if float(np.abs(P).max()) > 1.0001:
        prob.append("fuera de [-1, 1]")
    (fail if prob else ok)(
        f"partial (Ledoit-Wolf): {P.shape}" + (f" — {', '.join(prob)}" if prob else ""))

    S = D.build_flat_shrunk_connectivity(b["bold"], idx)
    prob = []
    if S.shape != (n, 1, F):
        prob.append(f"forma {S.shape}")
    if not np.isfinite(S).all():
        prob.append("valores no finitos")
    if float(np.abs(S).max()) > 1.0001:
        prob.append("fuera de [-1, 1]")
    (fail if prob else ok)(
        f"shrunk (Ledoit-Wolf): {S.shape}" + (f" — {', '.join(prob)}" if prob else ""))

    seq = D.build_flat_sequences(b["bold"], idx, 60, 6)
    H = D.hybrid_summary(seq, D.build_flat_static_connectivity(b["bold"], idx))
    okH = H.shape == (n, 1, 4 * F) and bool(np.isfinite(H).all())
    (ok if okH else fail)(f"hybrid: {H.shape} (4×{F})" + ("" if okH else " — forma o valores inválidos"))


def check_representaciones_fold_aware():
    seccion("Representaciones fold-aware (ordered_scaled, permuted_scaled, tangent)")
    import argparse as _argparse
    import numpy as np
    sys.path.insert(0, str(REPO / "src"))
    import data as D
    import run_experiment as R

    b = D.load_bold("NYU")
    idx = D.roi_indices("12")
    r = len(idx)
    F = r * (r - 1) // 2
    n = b["bold"].shape[0]
    rng = np.random.default_rng(0)
    fit_idx = np.sort(rng.choice(n, size=n // 2, replace=False))
    rest_idx = np.setdiff1d(np.arange(n), fit_idx)

    args = _argparse.Namespace(
        site="NYU", representation="ordered_scaled", fisher_z=False,
        constant_policy="zero", tr_seconds=None, window_tr=None, step_tr=None,
        window_seconds=120.0, step_seconds=12.0, overlap=None,
        window_shape="rectangular", gaussian_sigma=None, representation_seed=1,
    )
    spec = R.resolve_temporal_spec(args, n_timepoints=b["bold"].shape[-1])
    Xf, _, _ = R.build_representation(
        site="NYU", bold=b["bold"], labels=b["labels"], subjects=b["subjects"],
        indices=idx, roi_key="12", args=args, spec=spec, use_cache=False,
    )
    transform, out_shape = R.resolve_fold_transform(args, r)
    scaled = transform(Xf, fit_idx)
    fit_vals = scaled[fit_idx].reshape(-1, scaled.shape[-1])
    prob = []
    if out_shape is not None:
        prob.append("output_shape no debería ser None para ordered_scaled")
    if not np.isfinite(scaled).all():
        prob.append("valores no finitos")
    if np.max(np.abs(fit_vals.mean(axis=0))) > 1e-4:
        prob.append("media de fit no queda ~0")
    scaled_perturbed = transform(
        np.where(np.arange(n)[:, None, None] == rest_idx[0], 999.0, Xf), fit_idx
    )
    if not np.allclose(scaled[fit_idx], scaled_perturbed[fit_idx], atol=1e-6):
        prob.append("¡FUGA! perturbar rest_idx cambió la salida de fit_idx")
    (fail if prob else ok)("ordered_scaled: sin fuga, fit centrado en 0" + (f" — {'; '.join(prob)}" if prob else ""))

    args_p = _argparse.Namespace(
        site="NYU", representation="permuted_scaled", fisher_z=False,
        constant_policy="zero", tr_seconds=None, window_tr=None, step_tr=None,
        window_seconds=120.0, step_seconds=12.0, overlap=None,
        window_shape="rectangular", gaussian_sigma=None, representation_seed=1,
    )
    spec_p = R.resolve_temporal_spec(args_p, n_timepoints=b["bold"].shape[-1])
    Xf_p, _, _ = R.build_representation(
        site="NYU", bold=b["bold"], labels=b["labels"], subjects=b["subjects"],
        indices=idx, roi_key="12", args=args_p, spec=spec_p, use_cache=False,
    )
    transform_p, out_shape_p = R.resolve_fold_transform(args_p, r)
    scaled_p = transform_p(Xf_p, fit_idx)
    fit_vals_p = scaled_p[fit_idx].reshape(-1, scaled_p.shape[-1])
    prob = []
    if out_shape_p is not None:
        prob.append("output_shape no debería ser None para permuted_scaled")
    if not np.isfinite(scaled_p).all():
        prob.append("valores no finitos")
    if np.max(np.abs(fit_vals_p.mean(axis=0))) > 1e-4:
        prob.append("media de fit no queda ~0")
    scaled_p_perturbed = transform_p(
        np.where(np.arange(n)[:, None, None] == rest_idx[0], 999.0, Xf_p), fit_idx
    )
    if not np.allclose(scaled_p[fit_idx], scaled_p_perturbed[fit_idx], atol=1e-6):
        prob.append("¡FUGA! perturbar rest_idx cambió la salida de fit_idx")
    (fail if prob else ok)("permuted_scaled: sin fuga, fit centrado en 0" + (f" — {'; '.join(prob)}" if prob else ""))

    if importlib.util.find_spec("nilearn") is None:
        warn("nilearn no está instalado: no se puede probar 'tangent' aquí (pip install nilearn)")
        return

    args_t = _argparse.Namespace(
        site="NYU", representation="tangent", fisher_z=False, constant_policy="zero",
        tr_seconds=None, window_tr=None, step_tr=None, window_seconds=None,
        step_seconds=None, overlap=None, window_shape="rectangular", gaussian_sigma=None,
        representation_seed=None,
    )
    spec_t = R.resolve_temporal_spec(args_t, n_timepoints=b["bold"].shape[-1])
    raw, _, _ = R.build_representation(
        site="NYU", bold=b["bold"], labels=b["labels"], subjects=b["subjects"],
        indices=idx, roi_key="12", args=args_t, spec=spec_t, use_cache=False,
    )
    transform_t, out_shape_t = R.resolve_fold_transform(args_t, r)
    prob = []
    if out_shape_t != (1, F):
        prob.append(f"output_shape {out_shape_t}, se esperaba (1, {F})")
    Xt = transform_t(raw, fit_idx)
    if Xt.shape != (n, 1, F):
        prob.append(f"forma {Xt.shape}")
    if not np.isfinite(Xt).all():
        prob.append("valores no finitos")
    raw_perturbed = raw.copy()
    raw_perturbed[rest_idx[0]] = rng.normal(size=raw_perturbed[rest_idx[0]].shape) * 50
    Xt_perturbed = transform_t(raw_perturbed, fit_idx)
    if not np.allclose(Xt[fit_idx], Xt_perturbed[fit_idx], atol=1e-6):
        prob.append("¡FUGA! perturbar rest_idx cambió la referencia de fit_idx")
    (fail if prob else ok)(f"tangent (nilearn): {Xt.shape}, sin fuga" + (f" — {'; '.join(prob)}" if prob else ""))


def check_particiones():
    seccion("Particiones de validación cruzada")
    import numpy as np
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
    sys.path.insert(0, str(REPO / "src"))
    import data as D

    y = D.load_bold("NYU")["labels"]
    outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    fugas = solapes = 0
    cobertura = []
    for fold, (tr, va) in enumerate(outer.split(np.zeros((len(y), 1)), y)):
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42 + fold)
        fit_r, sel_r = next(inner.split(np.zeros((len(tr), 1)), y[tr]))
        fit, sel = tr[fit_r], tr[sel_r]
        fugas += len(set(fit) & set(va)) + len(set(sel) & set(va))
        solapes += len(set(fit) & set(sel))
        cobertura.extend(va.tolist())
    (ok if fugas == 0 else fail)(f"sin fuga entre validación y entrenamiento ({fugas})")
    (ok if solapes == 0 else fail)(f"sin solape entre fit y selección de época ({solapes})")
    from collections import Counter
    c = Counter(cobertura)
    (ok if set(c.values()) == {5} and len(c) == len(y) else fail)(
        f"cada sujeto validado 5 veces ({len(c)}/{len(y)} sujetos)")


def check_modelos(full):
    seccion("Arquitecturas")
    sys.path.insert(0, str(REPO / "src"))
    import kerasmodels

    disponibles = kerasmodels.available()
    esperadas = {"lstm", "gru", "cnn1d", "transformer", "deepsets", "brainnetcnn"}
    faltan = esperadas - set(disponibles)
    (fail if faltan else ok)(
        f"registradas: {', '.join(disponibles)}" + (f" — faltan {faltan}" if faltan else ""))

    if importlib.util.find_spec("keras") is None:
        warn("Keras no está instalado: no se puede construir ningún modelo aquí")
        return
    for name in disponibles:
        try:
            m = kerasmodels.build(name, 52, 66)
            n_par = m.count_params()
            forma = tuple(m.output.shape)
            if forma[-1] != 1:
                fail(f"{name}: la salida tiene forma {forma}, se esperaba (…, 1)")
            else:
                ok(f"{name}: {n_par:,} parámetros, salida {forma}")
        except Exception as e:
            fail(f"{name}: {type(e).__name__}: {e}")


def check_entrenamiento():
    seccion("Prueba de entrenamiento (2 pliegues, 3 épocas)")
    import subprocess
    r = subprocess.run(
        [sys.executable, "run_experiment.py", "--site", "NYU", "--roi-set", "12",
         "--n-splits", "2", "--n-repeats", "1", "--epochs", "3", "--patience", "2",
         "--out", "/tmp/verify_setup", "--tag", "verify"],
        cwd=REPO / "src", capture_output=True, text=True)
    if r.returncode == 0:
        ok("la corrida completa se ejecuta sin errores")
        for linea in r.stdout.splitlines():
            if "val acc" in linea:
                print(f"      {linea.strip()}")
    else:
        fail("la corrida falló")
        print((r.stdout + r.stderr)[-1500:])

    check_entrenamiento_val_bce()


def check_entrenamiento_val_bce():
    """Corrida corta con --early-stopping-monitor val_bce, auditando los artefactos
    (no solo el código de salida): esquema 3, columnas bce/inner_val_bce, ausencia de
    NaN/inf, best_epoch/best_monitor_value reconstruidos desde inner_val_bce, y
    cobertura completa de la validación externa. Usa un directorio propio bajo /tmp,
    nunca results/runs.
    """
    seccion("Prueba de entrenamiento con --early-stopping-monitor val_bce")
    import shutil
    import subprocess

    import numpy as np
    import pandas as pd

    outdir = Path("/tmp/verify_setup_val_bce")
    if outdir.exists():
        shutil.rmtree(outdir)
    r = subprocess.run(
        [sys.executable, "run_experiment.py", "--site", "NYU", "--roi-set", "12",
         "--n-splits", "2", "--n-repeats", "1", "--epochs", "3", "--patience", "2",
         "--early-stopping-monitor", "val_bce", "--early-stopping-min-delta", "1e-5",
         "--out", str(outdir), "--tag", "verify_bce"],
        cwd=REPO / "src", capture_output=True, text=True)
    if r.returncode != 0:
        fail("la corrida con --early-stopping-monitor val_bce falló")
        print((r.stdout + r.stderr)[-1500:])
        return
    ok("la corrida con --early-stopping-monitor val_bce se ejecuta sin errores")

    run_dirs = [p for p in outdir.iterdir() if p.is_dir()] if outdir.exists() else []
    if len(run_dirs) != 1:
        fail(f"se esperaba exactamente 1 carpeta de corrida en {outdir}, hay {len(run_dirs)}")
        return
    run_dir = run_dirs[0]

    try:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"no se pudo leer config.json: {type(e).__name__}: {e}")
        return
    prob = []
    if cfg.get("config_schema_version") != 3:
        prob.append(f"config_schema_version={cfg.get('config_schema_version')!r}, se esperaba 3")
    if cfg.get("early_stopping_monitor") != "val_bce":
        prob.append(f"early_stopping_monitor={cfg.get('early_stopping_monitor')!r}, se esperaba 'val_bce'")
    (fail if prob else ok)(
        "config.json: config_schema_version=3 y monitor solicitado"
        + (f" — {'; '.join(prob)}" if prob else "")
    )

    try:
        hist = pd.read_csv(run_dir / "history.csv")
        val = pd.read_csv(run_dir / "metrics_val.csv")
        pred = pd.read_csv(run_dir / "predictions_val.csv")
    except Exception as e:
        fail(f"no se pudieron leer los artefactos de la corrida: {type(e).__name__}: {e}")
        return

    prob = [c for c in ("bce", "inner_val_bce") if c not in hist.columns]
    (fail if prob else ok)(
        "history.csv contiene bce/inner_val_bce" + (f" — faltan {prob}" if prob else "")
    )

    numeric_hist = hist.select_dtypes(include=[np.number]).to_numpy()
    if numeric_hist.size == 0 or not np.isfinite(numeric_hist).all():
        fail("history.csv contiene valores no finitos o está vacío")
    else:
        ok("history.csv sin NaN/inf")

    prob = [c for c in ("early_stopping_monitor", "best_monitor_value") if c not in val.columns]
    (fail if prob else ok)(
        "metrics_val.csv contiene early_stopping_monitor/best_monitor_value"
        + (f" — faltan {prob}" if prob else "")
    )

    # best_epoch/best_monitor_value deben coincidir con el mínimo de inner_val_bce
    # de ESE pliegue en history.csv — no con val_loss ni con ningún otro monitor.
    prob = []
    for _, row in val.iterrows():
        fold_hist = hist[
            (hist["fold"] == row["fold"]) & (hist["repeat"] == row["repeat"])
        ].sort_values("epoch")
        if fold_hist.empty:
            prob.append(f"pliegue f{row['fold']}/r{row['repeat']}: sin filas en history.csv")
            continue
        series = fold_hist["inner_val_bce"].to_numpy()
        expected_epoch = int(np.argmin(series)) + 1
        if int(row["best_epoch"]) != expected_epoch:
            prob.append(
                f"pliegue f{row['fold']}/r{row['repeat']}: best_epoch={row['best_epoch']}, "
                f"se esperaba {expected_epoch} (mínimo de inner_val_bce)"
            )
            continue
        expected_value = float(series[expected_epoch - 1])
        if abs(float(row["best_monitor_value"]) - expected_value) > 1e-6:
            prob.append(
                f"pliegue f{row['fold']}/r{row['repeat']}: best_monitor_value="
                f"{row['best_monitor_value']}, se esperaba {expected_value}"
            )
    (fail if prob else ok)(
        "best_epoch/best_monitor_value coinciden con el mínimo de inner_val_bce por pliegue"
        + (f" — {'; '.join(prob)}" if prob else "")
    )

    # Cobertura de outer_val: cada sujeto exactamente una vez por repetición.
    prob = []
    n_subjects = cfg.get("n_subjects")
    for repeat, group in pred.groupby("repeat"):
        n_dup = int(group["subject_id"].duplicated().sum())
        if n_dup:
            prob.append(f"repetición {repeat}: {n_dup} subject_id duplicados en outer_val")
        if n_subjects is not None and group["subject_id"].nunique() != n_subjects:
            prob.append(
                f"repetición {repeat}: {group['subject_id'].nunique()} sujetos evaluados, "
                f"se esperaban {n_subjects}"
            )
    (fail if prob else ok)(
        "validación externa cubre cada sujeto exactamente una vez por repetición"
        + (f" — {'; '.join(prob)}" if prob else "")
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="añade una prueba de entrenamiento real (requiere TensorFlow)")
    args = ap.parse_args()

    print(f"Verificando {REPO}")
    check_estructura()
    try:
        check_datos()
        check_roi_sets()
        check_secuencias()
        check_representaciones()
        check_representaciones_fold_aware()
        check_particiones()
        check_modelos(args.full)
        if args.full:
            check_entrenamiento()
    except Exception as e:
        fail(f"error inesperado: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"{ok_count} correctas · {warn_count} avisos · {fail_count} fallos")
    if fail_count:
        print("\nHay fallos que impiden correr experimentos.")
    elif not args.full:
        print("\nTodo correcto. Ejecute --full para probar además el entrenamiento.")
    else:
        print("\nTodo correcto. El repositorio está listo.")
    sys.exit(1 if fail_count else 0)


if __name__ == "__main__":
    main()