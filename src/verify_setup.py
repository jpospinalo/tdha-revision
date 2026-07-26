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


def _write_schema4_fixture(run_dir: Path) -> None:
    """Escribe los artefactos de una corrida de esquema 4 completa y
    coherente (2 pliegues, 6 sujetos), sin entrenar nada — usada por
    check_schema4_artifact_validation() para mutar copias y probar que
    compile_results.validate_run_artifacts()/collect() rechazan lo que deben
    rechazar. Las particiones son reales — fit/inner_val/outer_val disjuntas,
    outer_val cubre los 6 sujetos exactamente una vez en la repetición — para
    que la corrida "sana" también pase las comprobaciones semánticas de H12
    (conteo de filas, claves compartidas, cobertura OOF, tamaños de
    partición), no solo las de columnas/finitud.
    """
    import pandas as pd

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "run_id": run_dir.name, "config_schema_version": 4, "site": "NYU",
        "n_splits": 2, "n_repeats": 1, "n_subjects": 6,
        "early_stopping_monitor": "val_loss", "early_stopping_ab_hash": "fixture0000000",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    folds_spec = {
        1: {"outer_val": ["s1", "s2", "s3"], "fit": ["s4", "s5"], "inner_val": ["s6"]},
        2: {"outer_val": ["s4", "s5", "s6"], "fit": ["s1", "s2"], "inner_val": ["s3"]},
    }
    metrics_rows, history_rows, prediction_rows, fold_rows = [], [], [], []
    for fold, parts in folds_spec.items():
        metrics_rows.append({
            "fold": fold, "repeat": 1, "n_epochs": 3, "best_epoch": 2,
            "n_fit": len(parts["fit"]), "n_inner_val": len(parts["inner_val"]),
            "n_outer_val": len(parts["outer_val"]),
            "early_stopping_monitor": "val_loss",
            "best_monitor_value": 0.5, "restored_monitor_value": 0.5,
            "accuracy": 0.7, "loss": 0.5,
        })
        for epoch in (1, 2, 3):
            history_rows.append({
                "fold": fold, "repeat": 1, "epoch": epoch,
                "loss": 0.6 - 0.05 * epoch, "inner_val_loss": 0.5,
                "bce": 0.5 - 0.05 * epoch, "inner_val_bce": 0.45,
            })
        for subject in parts["outer_val"]:
            prediction_rows.append({
                "fold": fold, "repeat": 1, "subject_id": subject,
                "y_true": 1 if subject in ("s1", "s4") else 0, "y_prob": 0.6,
            })
        for split_name in ("fit", "inner_val", "outer_val"):
            for subject in parts[split_name]:
                fold_rows.append({"fold": fold, "repeat": 1, "subject_id": subject, "split": split_name})

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics_train.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics_val.csv", index=False)
    pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions_val.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "folds.csv", index=False)


def check_schema4_artifact_validation():
    """Regresiones de compile_results.validate_run_artifacts()/collect() con
    fixtures CSV escritos a mano, sin entrenar ni importar TensorFlow.

    Caso A prueba un fixture sano; B, columnas estructurales ausentes; C,
    NaN en un campo numérico obligatorio; D, compatibilidad con esquemas
    anteriores a 4; E, las comprobaciones semánticas más nuevas (duplicados,
    cobertura OOF incompleta, probabilidad fuera de rango, solape entre
    particiones, y predicciones que no coinciden con el outer_val real —
    ver docs/validation.md).
    """
    seccion("Validación de artefactos de esquema 4 (fixtures, sin entrenar)")
    import shutil
    import pandas as pd
    sys.path.insert(0, str(REPO / "src"))
    import compile_results as C

    root = Path("/tmp/verify_schema4_fixtures")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # --- Caso A: fixture válido ---
    root_a = root / "caso_a"
    run_a = root_a / "run_valido"
    _write_schema4_fixture(run_a)
    problems = C.validate_run_artifacts(run_a, "")
    (ok if not problems else fail)(
        "caso A (válido): validate_run_artifacts() devuelve []"
        + (f" — {problems}" if problems else "")
    )
    try:
        df_a = C.collect(root_a, strict=True)
        prob = []
        if len(df_a) != 1:
            prob.append(f"se esperaba 1 fila, hay {len(df_a)}")
        if df_a.attrs.get("collection_warnings"):
            prob.append(f"collection_warnings no vacío: {df_a.attrs.get('collection_warnings')}")
        (ok if not prob else fail)(
            "caso A: collect(strict=True) acepta el fixture válido, 1 fila, sin avisos"
            + (f" — {'; '.join(prob)}" if prob else "")
        )
    except Exception as e:
        fail(
            "caso A: collect(strict=True) lanzó una excepción inesperada sobre un "
            f"fixture válido: {type(e).__name__}: {e}"
        )

    # --- Caso B: columnas estructurales ausentes en 3 artefactos distintos ---
    root_b = root / "caso_b"
    run_b = root_b / "run_columnas_ausentes"
    _write_schema4_fixture(run_b)
    drops = [("predictions_val.csv", "subject_id"), ("folds.csv", "split"), ("history.csv", "epoch")]
    for filename, column in drops:
        path = run_b / filename
        pd.read_csv(path).drop(columns=[column]).to_csv(path, index=False)

    problems = C.validate_run_artifacts(run_b, "")
    texto = " | ".join(problems)
    detectadas = sum(1 for _, column in drops if column in texto)
    (ok if detectadas == len(drops) else fail)(
        f"caso B: validate_run_artifacts() informa las {len(drops)} columnas ausentes"
        + (f" — solo detectó {detectadas}/{len(drops)}: {problems}" if detectadas != len(drops) else "")
    )
    try:
        C.collect(root_b, strict=True)
        fail("caso B: collect(strict=True) no lanzó ValueError con columnas estructurales ausentes")
    except ValueError as e:
        # No basta con "lanzó ValueError": si la excepción viniera de otra causa,
        # este chequeo pasaría sin haber ejercido de verdad la ruta que estamos
        # probando. Se exige que el mensaje identifique la corrida y las tres
        # columnas ausentes.
        msg = str(e)
        prob = []
        if run_b.name not in msg:
            prob.append(f"el mensaje no identifica la corrida ({run_b.name!r})")
        faltan_en_msg = [column for _, column in drops if column not in msg]
        if faltan_en_msg:
            prob.append(f"el mensaje no menciona: {faltan_en_msg}")
        (ok if not prob else fail)(
            "caso B: collect(strict=True) lanza ValueError identificando la corrida y "
            "las columnas ausentes" + (f" — {'; '.join(prob)}; mensaje={msg!r}" if prob else "")
        )
    df_b = C.collect(root_b, strict=False)
    prob = []
    if not df_b.empty:
        prob.append(f"se esperaban 0 filas, hay {len(df_b)}")
    if not df_b.attrs.get("collection_warnings"):
        prob.append("collection_warnings vacío, se esperaba el diagnóstico de la corrida defectuosa")
    (ok if not prob else fail)(
        "caso B: collect(strict=False) descarta la corrida y registra el diagnóstico"
        + (f" — {'; '.join(prob)}" if prob else "")
    )

    # --- Caso C: NaN en un campo numérico obligatorio (un subcaso por campo) ---
    subcasos = [
        ("metrics_val.csv", "best_monitor_value"),
        ("history.csv", "inner_val_bce"),
        ("predictions_val.csv", "y_prob"),
    ]
    for i, (filename, column) in enumerate(subcasos):
        root_c = root / f"caso_c_{i}"
        run_c = root_c / "run_nan"
        _write_schema4_fixture(run_c)
        path = run_c / filename
        frame = pd.read_csv(path)
        frame.loc[0, column] = float("nan")
        frame.to_csv(path, index=False)

        problems = C.validate_run_artifacts(run_c, "")
        base_name = filename.replace(".csv", "")
        detectado = any(base_name in p and column in p for p in problems)
        (ok if detectado else fail)(
            f"caso C ({filename}/{column}=NaN): validate_run_artifacts() señala "
            f"{filename} y {column}" + ("" if detectado else f" — problems={problems}")
        )
        try:
            C.collect(root_c, strict=True)
            fail(f"caso C ({filename}/{column}=NaN): collect(strict=True) no lanzó ValueError")
        except ValueError as e:
            # Igual que en el caso B: se exige que el mensaje señale la corrida y la
            # columna alterada, no solo que haya llegado *algún* ValueError.
            msg = str(e)
            prob = []
            if run_c.name not in msg:
                prob.append(f"el mensaje no identifica la corrida ({run_c.name!r})")
            if column not in msg:
                prob.append(f"el mensaje no menciona la columna alterada ({column!r})")
            (ok if not prob else fail)(
                f"caso C ({filename}/{column}=NaN): collect(strict=True) lanza ValueError "
                "identificando la corrida y la columna"
                + (f" — {'; '.join(prob)}; mensaje={msg!r}" if prob else "")
            )

    # --- Caso D: compatibilidad histórica (esquema 2, sin contrato de esquema 4) ---
    root_d = root / "caso_d"
    run_d = root_d / "run_esquema2"
    run_d.mkdir(parents=True)
    cfg_d = {"run_id": "run_esquema2", "config_schema_version": 2, "site": "NYU"}
    (run_d / "config.json").write_text(json.dumps(cfg_d), encoding="utf-8")
    minimal = {"accuracy": [0.7], "loss": [0.5]}
    pd.DataFrame(minimal).to_csv(run_d / "metrics_train.csv", index=False)
    pd.DataFrame(minimal).to_csv(run_d / "metrics_val.csv", index=False)

    df_d = C.collect(root_d, strict=True)
    prob = []
    if len(df_d) != 1:
        prob.append(f"se esperaba 1 fila, hay {len(df_d)}")
    if not df_d.attrs.get("collection_notices"):
        prob.append("collection_notices vacío, se esperaba el aviso histórico")
    if df_d.attrs.get("collection_warnings"):
        prob.append(
            f"collection_warnings no vacío para una corrida histórica válida: "
            f"{df_d.attrs.get('collection_warnings')}"
        )
    (ok if not prob else fail)(
        "caso D: una corrida de esquema 2 compila de forma descriptiva (1 fila) con el "
        "aviso histórico, sin pasar por el contrato de esquema 4"
        + (f" — {'; '.join(prob)}" if prob else "")
    )

    # --- Caso E: comprobaciones semánticas de H12, cada una en su propia mutación ---
    def _semantica(nombre: str, mutar) -> None:
        root_e = root / f"caso_e_{nombre}"
        run_e = root_e / "run_mutado"
        _write_schema4_fixture(run_e)
        mutar(run_e)
        problems = C.validate_run_artifacts(run_e, "")
        (ok if problems else fail)(
            f"caso E ({nombre}): validate_run_artifacts() detecta el problema"
            + ("" if problems else " — devolvió []")
        )
        try:
            C.collect(root_e, strict=True)
            fail(f"caso E ({nombre}): collect(strict=True) no lanzó ValueError")
        except ValueError as e:
            msg = str(e)
            (ok if run_e.name in msg else fail)(
                f"caso E ({nombre}): collect(strict=True) lanza ValueError identificando la corrida"
                + ("" if run_e.name in msg else f" — mensaje={msg!r}")
            )

    def _subject_id_duplicado(run_dir: Path) -> None:
        p = run_dir / "predictions_val.csv"
        frame = pd.read_csv(p)
        pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(p, index=False)

    def _cobertura_oof_incompleta(run_dir: Path) -> None:
        p = run_dir / "predictions_val.csv"
        pd.read_csv(p).drop(index=0).to_csv(p, index=False)

    def _probabilidad_fuera_de_rango(run_dir: Path) -> None:
        p = run_dir / "predictions_val.csv"
        frame = pd.read_csv(p)
        frame.loc[0, "y_prob"] = 1.5
        frame.to_csv(p, index=False)

    def _solape_fit_outer_val(run_dir: Path) -> None:
        p = run_dir / "folds.csv"
        frame = pd.read_csv(p)
        fuga = pd.DataFrame([{"fold": 1, "repeat": 1, "subject_id": "s1", "split": "fit"}])
        pd.concat([frame, fuga], ignore_index=True).to_csv(p, index=False)

    def _prediccion_no_coincide_con_outer_val(run_dir: Path) -> None:
        # s6 pertenece a inner_val en el pliegue 1 (ver folds_spec de
        # _write_schema4_fixture); sustituirlo por s1 en predictions_val.csv
        # deja un sujeto de outer_val sin predicción y uno ajeno con predicción.
        p = run_dir / "predictions_val.csv"
        frame = pd.read_csv(p)
        frame.loc[(frame["fold"] == 1) & (frame["subject_id"] == "s1"), "subject_id"] = "s6"
        frame.to_csv(p, index=False)

    _semantica("subject_id_duplicado_oof", _subject_id_duplicado)
    _semantica("cobertura_oof_incompleta", _cobertura_oof_incompleta)
    _semantica("y_prob_fuera_de_rango", _probabilidad_fuera_de_rango)
    _semantica("solape_fit_outer_val", _solape_fit_outer_val)
    _semantica("prediccion_no_coincide_outer_val", _prediccion_no_coincide_con_outer_val)

    shutil.rmtree(root)


def check_aggregate_table_gate():
    """Regresión de la compuerta de aggregate_table() (H11): dos filas
    idénticas en las columnas de methodological_group_columns() pero con
    config_hash distinto no deben promediarse en silencio — señalarían un eje
    metodológico que la lista de columnas no cubrió. Con datos sintéticos
    (sin CSV ni entrenamiento): primero confirma que lr/arch_json/
    deterministic ya están en esa lista; luego, para cada uno de esos tres
    campos por separado, confirma que dos corridas que solo difieren ahí
    caen en grupos distintos de aggregate_table() (no solo que el campo
    está listado); por último confirma que, si dos filas comparten todas
    las columnas agrupadas y aun así traen config_hash distinto,
    aggregate_table() aborta en vez de promediar.
    """
    seccion("Compuerta de aggregate_table() ante config_hash mixto (H11)")
    sys.path.insert(0, str(REPO / "src"))
    import compile_results as C
    import pandas as pd

    base = {
        "site": "NYU", "roi_set": "12", "model": "brainnetcnn",
        "arch_json": '{"dropout":0.7}', "representation": "ordered",
        "connectivity_mode": "dynamic", "lr": 1e-4, "batch_size": 32,
        "epochs": 300, "patience": 25, "clipnorm": None, "inner_val_frac": 0.15,
        "class_weight": False, "deterministic": False, "mixed_precision": False,
        "early_stopping_monitor": "val_loss", "early_stopping_min_delta": 1e-5,
        "val_accuracy_mean": 0.7, "oof_accuracy_mean": 0.7,
    }

    groups = C.methodological_group_columns(pd.DataFrame([base]))
    prob = [c for c in ("lr", "arch_json", "deterministic") if c not in groups]
    (ok if not prob else fail)(
        "methodological_group_columns() incluye lr/arch_json/deterministic"
        + (f" — faltan {prob}" if prob else "")
    )

    # Dos filas que solo difieren en un campo de entrenamiento/arquitectura:
    # con ese campo ya en la lista de columnas, deben quedar en grupos
    # separados, no promediarse como si fueran la misma corrida. Se repite
    # para lr, arch_json y deterministic — los tres que motivaron H11 — no
    # solo se confirma que están en la lista, sino que aggregate_table()
    # realmente los usa para separar los grupos.
    variantes = {
        "lr": 5e-4,
        "arch_json": '{"dropout":0.5}',
        "deterministic": True,
    }
    for campo, valor_distinto in variantes.items():
        fila_a = {**base, "config_hash": f"hashA_{campo}", "base_run_id": f"runA_{campo}"}
        fila_b = {**base, "config_hash": f"hashB_{campo}", "base_run_id": f"runB_{campo}",
                  campo: valor_distinto}
        agregada = C.aggregate_table(pd.DataFrame([fila_a, fila_b]))
        (ok if len(agregada) == 2 else fail)(
            f"dos corridas que solo difieren en {campo} quedan en grupos separados"
            + (f" — se esperaban 2 grupos, hay {len(agregada)}" if len(agregada) != 2 else "")
        )

    # Dos filas idénticas en las columnas agrupadas pero con config_hash
    # distinto: el eje que las diferencia no está cubierto por esa lista, y
    # promediarlas produciría una media entre configuraciones distintas.
    fila_c = {**base, "config_hash": "hashC1", "base_run_id": "runC1"}
    fila_d = {**base, "config_hash": "hashC2", "base_run_id": "runC2"}
    try:
        C.aggregate_table(pd.DataFrame([fila_c, fila_d]))
        fail("aggregate_table() no abortó ante config_hash mixto dentro de un mismo grupo")
    except SystemExit as e:
        msg = str(e)
        prob = []
        if "runC1" not in msg or "runC2" not in msg:
            prob.append("el mensaje no identifica ambos run_id")
        if "hashC1" not in msg or "hashC2" not in msg:
            prob.append("el mensaje no identifica ambos config_hash")
        (ok if not prob else fail)(
            "aggregate_table() aborta ante config_hash mixto, identificando el grupo y "
            "los run_id/config_hash en conflicto" + (f" — {'; '.join(prob)}" if prob else "")
        )


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


def check_default_monitor_parser():
    """Regresión de qué monitor queda activo cuando no se pasa
    --early-stopping-monitor, comprobada leyendo solo el parser (sin
    TensorFlow, sin entrenar) — reemplaza a la antigua corrida de humo
    genérica en /tmp/verify_setup, que además no era repetible sin
    --overwrite (ver docs/validation.md).
    """
    seccion("Valores por defecto del parser (sin entrenar)")
    sys.path.insert(0, str(REPO / "src"))
    import run_experiment as R

    args = R.build_parser().parse_args([])
    prob = []
    if args.early_stopping_monitor != "val_loss":
        prob.append(f"early_stopping_monitor={args.early_stopping_monitor!r}, se esperaba 'val_loss'")
    if args.early_stopping_min_delta != 1e-5:
        prob.append(f"early_stopping_min_delta={args.early_stopping_min_delta!r}, se esperaba 1e-5")
    (fail if prob else ok)(
        "sin --early-stopping-monitor/--early-stopping-min-delta, el parser deja "
        "val_loss / 1e-5" + (f" — {'; '.join(prob)}" if prob else "")
    )


def check_compiler_integration(root: Path, run_loss: Path, run_bce: Path):
    """Comprueba, sobre las mismas dos corridas ya entrenadas por
    check_early_stopping_ab() (sin reentrenar), que compile_results.py las
    lea, acepte el par como A/B válido, y rechace variantes en memoria con
    monitor duplicado o early_stopping_ab_hash alterado — sin lo cual
    _check_early_stopping_ab() quedaría sin probar de verdad (ver
    docs/validation.md).
    """
    seccion("Integración con compile_results.py (sin reentrenar)")
    sys.path.insert(0, str(REPO / "src"))
    import compile_results as C
    import pandas as pd

    try:
        df = C.collect(root, strict=True)
    except Exception as e:
        fail(f"collect(root, strict=True) lanzó una excepción inesperada: {type(e).__name__}: {e}")
        return

    prob = []
    if len(df) != 2:
        prob.append(f"se esperaban 2 filas, hay {len(df)}")
    monitors = sorted(df["early_stopping_monitor"]) if "early_stopping_monitor" in df else []
    if monitors != ["val_bce", "val_loss"]:
        prob.append(f"monitores encontrados {monitors}, se esperaban ['val_bce', 'val_loss']")
    (fail if prob else ok)(
        "collect(root, strict=True) sobre el par val_loss/val_bce da 2 filas, "
        "una por monitor, sin duplicados" + (f" — {'; '.join(prob)}" if prob else "")
    )
    if prob:
        return

    try:
        C._check_early_stopping_ab(df)
        ok("_check_early_stopping_ab() acepta el par real val_loss/val_bce")
    except SystemExit as e:
        fail(f"_check_early_stopping_ab() rechazó el par real: {e}")
        return

    # Negativo 1: monitor duplicado (val_loss repetido, sin val_bce) — debe rechazarse.
    dup = pd.concat([df[df["early_stopping_monitor"] == "val_loss"]] * 2, ignore_index=True)
    try:
        C._check_early_stopping_ab(dup)
        fail("_check_early_stopping_ab() no rechazó un par con monitor duplicado")
    except SystemExit:
        ok("_check_early_stopping_ab() rechaza un par con monitor duplicado (val_loss × 2)")

    # Negativo 2: early_stopping_ab_hash alterado en una fila — debe rechazarse.
    altered = df.copy()
    altered.loc[altered.index[0], "early_stopping_ab_hash"] = "deadbeefdeadbeef"
    try:
        C._check_early_stopping_ab(altered)
        fail("_check_early_stopping_ab() no rechazó un early_stopping_ab_hash alterado")
    except SystemExit:
        ok("_check_early_stopping_ab() rechaza un early_stopping_ab_hash alterado")


def check_entrenamiento():
    seccion("Prueba de entrenamiento (BrainNetCNN, esquema 4)")
    check_default_monitor_parser()
    resultado = check_early_stopping_ab()
    if resultado is not None:
        root, run_loss, run_bce = resultado
        check_compiler_integration(root, run_loss, run_bce)


def _run_early_stopping_smoke(root: Path, monitor: str) -> Path | None:
    """Corrida corta con BrainNetCNN, subproceso (no --in-process), dentro de
    `root` — compartida entre val_loss y val_bce, limpiada una sola vez por
    check_early_stopping_ab() para que --full solo entrene dos veces. La
    carpeta nueva se detecta por diferencia de contenido antes/después de
    correr (no por orden alfabético ni por `run_dirs[-1]`, que se vuelven
    ambiguos con un root compartido). Devuelve la carpeta de la corrida, o
    None si algo falló.
    """
    import subprocess

    antes = {p for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    r = subprocess.run(
        [sys.executable, "run_experiment.py", "--site", "NYU", "--roi-set", "12",
         "--model", "brainnetcnn", "--representation", "ordered",
         "--window-seconds", "120", "--step-seconds", "12",
         "--model-arg", "e2e=4", "e2n=8", "dense=8", "dropout=0.7", "leaky=0.33",
         "l2_reg=0.05", "inter_dropout=0.6",
         "--batch-size", "32", "--lr", "0.0001",
         "--n-splits", "2", "--n-repeats", "1", "--epochs", "4", "--patience", "2",
         "--seed", "42", "--deterministic",
         "--early-stopping-monitor", monitor, "--early-stopping-min-delta", "1e-5",
         "--out", str(root), "--tag", f"verify_{monitor}", "--overwrite"],
        cwd=REPO / "src", capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"la corrida con --early-stopping-monitor {monitor} falló")
        print((r.stdout + r.stderr)[-1500:])
        return None
    ok(f"la corrida con --early-stopping-monitor {monitor} se ejecuta sin errores")

    despues = {p for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    nuevas = sorted(despues - antes)
    if len(nuevas) != 1:
        fail(
            f"se esperaba exactamente 1 carpeta nueva en {root} tras "
            f"--early-stopping-monitor {monitor}, hubo {len(nuevas)}"
        )
        return None
    return nuevas[0]


def _audit_early_stopping_artifacts(run_dir: Path, monitor: str):
    """Audita una corrida de la comparación val_loss/val_bce.

    Los campos que describen qué se le pidió a ESTA corrida en particular
    (el monitor, min_delta) se comprueban aquí. El resto de la validez del
    artefacto — columnas, finitud, consistencia entre best_epoch/history/
    restored_monitor_value, cobertura OOF, particiones disjuntas — se delega
    en compile_results.validate_run_artifacts(), la misma función que usa
    collect(strict=True) y la celda de validación del notebook: no hay una
    segunda implementación de esas reglas (ver H12, docs/validation.md).
    Devuelve el config.json leído, o None si algo esencial faltó.
    """
    sys.path.insert(0, str(REPO / "src"))
    import compile_results as C

    try:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"no se pudo leer config.json: {type(e).__name__}: {e}")
        return None

    prob = []
    if cfg.get("config_schema_version") != 4:
        prob.append(f"config_schema_version={cfg.get('config_schema_version')!r}, se esperaba 4")
    if cfg.get("early_stopping_monitor") != monitor:
        prob.append(f"early_stopping_monitor={cfg.get('early_stopping_monitor')!r}, se esperaba {monitor!r}")
    if cfg.get("early_stopping_min_delta") != 1e-5:
        prob.append(f"early_stopping_min_delta={cfg.get('early_stopping_min_delta')!r}, se esperaba 1e-5")
    if not cfg.get("early_stopping_ab_hash"):
        prob.append("falta early_stopping_ab_hash")
    (fail if prob else ok)(
        f"{monitor}: config.json (esquema 4, monitor, min_delta, ab_hash)"
        + (f" — {'; '.join(prob)}" if prob else "")
    )

    problems = C.validate_run_artifacts(run_dir, cfg=cfg)
    (fail if problems else ok)(
        f"{monitor}: validate_run_artifacts() sin problemas (columnas, finitud, "
        "best_epoch/history/restored_monitor_value, cobertura OOF y particiones "
        "disjuntas)" + (f" — {problems}" if problems else "")
    )

    return cfg


def check_early_stopping_ab():
    """Dos corridas cortas con BrainNetCNN, idénticas salvo el monitor
    (val_loss / val_bce), compartiendo un único directorio raíz (limpiado una
    sola vez aquí, no una vez por corrida) y auditadas simétricamente sin usar
    np.argmin como oráculo de qué época restauró EarlyStopping (ver
    methodology.md). Devuelve (root, run_loss, run_bce) para que la
    comprobación del compilador (check_compiler_integration) pueda reusar las
    mismas dos corridas sin reentrenar, o None si algo falló.
    """
    import shutil

    seccion("Comparación val_loss / val_bce con BrainNetCNN (esquema 4)")

    root = Path("/tmp/verify_setup_early_stopping")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    run_loss = _run_early_stopping_smoke(root, "val_loss")
    run_bce = _run_early_stopping_smoke(root, "val_bce")
    if run_loss is None or run_bce is None:
        return None

    cfg_loss = _audit_early_stopping_artifacts(run_loss, "val_loss")
    cfg_bce = _audit_early_stopping_artifacts(run_bce, "val_bce")
    if cfg_loss is None or cfg_bce is None:
        return None

    prob = []
    if cfg_loss.get("split_fingerprint") != cfg_bce.get("split_fingerprint"):
        prob.append("split_fingerprint distinto entre las dos corridas")
    if cfg_loss.get("config_hash") == cfg_bce.get("config_hash"):
        prob.append("config_hash igual entre las dos corridas (debería diferir por el monitor)")
    if cfg_loss.get("early_stopping_ab_hash") != cfg_bce.get("early_stopping_ab_hash"):
        prob.append(
            "early_stopping_ab_hash distinto: las corridas no quedaron idénticas salvo el monitor"
        )
    (fail if prob else ok)(
        "las dos corridas comparten split_fingerprint/early_stopping_ab_hash y "
        "difieren en config_hash" + (f" — {'; '.join(prob)}" if prob else "")
    )

    return root, run_loss, run_bce


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
        check_schema4_artifact_validation()
        check_aggregate_table_gate()
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