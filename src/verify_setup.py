#!/usr/bin/env python3
"""
Verificación del repositorio y del entorno.

Comprueba que todo lo necesario para correr experimentos esté en su sitio y sea
coherente. Pensado para ejecutarse justo después de clonar, antes de lanzar nada.

    cd src
    python verify_setup.py            # comprobaciones rápidas
    python verify_setup.py --full     # añade una prueba de entrenamiento real

Sin --full no se entrena ningún modelo, pero check_modelos() sí construye cada
arquitectura registrada (para contar parámetros y validar la forma de salida), lo que
importa Keras/TensorFlow si está instalado. Si Keras no está instalado, ese chequeo se
reduce a un aviso en vez de fallar, así que el script sigue siendo útil en un entorno
sin GPU ni Keras.
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

    Caso A prueba un fixture sano; B, columnas estructurales ausentes (incluye
    B2: n_fit/n_inner_val/n_outer_val); C, NaN en un campo numérico
    obligatorio; D, compatibilidad con esquemas anteriores a 4; E, las
    comprobaciones semánticas de sujetos/predicciones (duplicados, cobertura
    OOF incompleta, probabilidad fuera de rango, solape entre particiones,
    predicciones que no coinciden con el outer_val real, filas duplicadas en
    folds.csv/history.csv, un fold completo ausente de un archivo, valores de
    split desconocidos, un fold completo ausente específicamente de
    folds.csv, y una clave (repeat, fold) en predictions_val.csv que no
    existe en metrics_val.csv); F, restored_monitor_value no numérico sin que
    escape TypeError; G, el nombre de la carpeta debe coincidir con
    config["run_id"] (carpeta renombrada a mano — collect(strict=True) debe
    lanzar ValueError, collect(strict=False) debe descartarla y dejar el
    diagnóstico en collection_warnings); H, una corrida bien nombrada sigue
    aceptándose en ambos layouts (plano histórico y anidado por ROI), y
    _find_run_dir() ubica correctamente la del layout anidado. T1-T4 (correcciones v9): T1 exige n_splits/n_repeats/
    n_subjects válidos en config.json (ausentes, booleanos o fuera de rango,
    individualmente y los tres a la vez); T2 exige config_schema_version
    válido (texto, booleano o cero, sin que escape TypeError); T3 reproduce
    un fold al que le falta el split inner_val aunque n_fit/n_inner_val se
    hayan ajustado a juego para que la unión y los tamaños declarados sigan
    cuadrando; T4 cubre un fixture de dos repeticiones (sano, y luego con
    y_true inconsistente para el mismo sujeto entre repeticiones). Ver
    docs/validation.md.
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

    # --- Caso B2: n_fit/n_inner_val/n_outer_val ausentes en metrics_val.csv ---
    # Bug C4 #1: antes de esta revisión estas tres columnas no estaban en el
    # contrato de esquema 4, así que un CSV sin ellas pasaba la validación.
    root_b2 = root / "caso_b2"
    run_b2 = root_b2 / "run_sin_n_fit_inner_outer"
    _write_schema4_fixture(run_b2)
    faltantes_b2 = ["n_fit", "n_inner_val", "n_outer_val"]
    path_b2 = run_b2 / "metrics_val.csv"
    pd.read_csv(path_b2).drop(columns=faltantes_b2).to_csv(path_b2, index=False)
    problems_b2 = C.validate_run_artifacts(run_b2, "")
    texto_b2 = " | ".join(problems_b2)
    detectadas_b2 = sum(1 for c in faltantes_b2 if c in texto_b2)
    (ok if detectadas_b2 == len(faltantes_b2) else fail)(
        "caso B2: validate_run_artifacts() informa n_fit/n_inner_val/n_outer_val "
        "ausentes en metrics_val.csv"
        + (f" — solo detectó {detectadas_b2}/{len(faltantes_b2)}: {problems_b2}"
           if detectadas_b2 != len(faltantes_b2) else "")
    )

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

    # --- Casos C4: los 4 bugs restantes reportados sobre validate_run_artifacts ---

    def _fila_duplicada_folds(run_dir: Path) -> None:
        # Bug C4 #2: una fila repetida en folds.csv (misma clave
        # (repeat, fold, subject_id)) no se detectaba porque no había un
        # chequeo de unicidad de claves por archivo.
        p = run_dir / "folds.csv"
        frame = pd.read_csv(p)
        pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(p, index=False)

    def _fila_duplicada_history(run_dir: Path) -> None:
        # Mismo bug que arriba, pero en history.csv con clave
        # (repeat, fold, epoch).
        p = run_dir / "history.csv"
        frame = pd.read_csv(p)
        pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(p, index=False)

    def _fold_completo_ausente(run_dir: Path) -> None:
        # Bug C4 #3: si un (repeat, fold) completo falta en un archivo (aquí,
        # metrics_train.csv) pero sigue presente en metrics_val.csv, antes no
        # se detectaba porque no había comprobación de cobertura cruzada de
        # pares (repeat, fold) entre archivos.
        p = run_dir / "metrics_train.csv"
        frame = pd.read_csv(p)
        objetivo = frame[["repeat", "fold"]].drop_duplicates().iloc[0]
        mascara = (frame["repeat"] == objetivo["repeat"]) & (frame["fold"] == objetivo["fold"])
        frame[~mascara].to_csv(p, index=False)

    def _split_desconocido(run_dir: Path) -> None:
        # Bug C4 #4: un valor de split fuera de {fit, inner_val, outer_val}
        # se descartaba en silencio en vez de señalarse.
        p = run_dir / "folds.csv"
        frame = pd.read_csv(p)
        frame.loc[0, "split"] = "unknown"
        frame.to_csv(p, index=False)

    _semantica("fila_duplicada_folds", _fila_duplicada_folds)
    _semantica("fila_duplicada_history", _fila_duplicada_history)
    _semantica("fold_completo_ausente", _fold_completo_ausente)
    _semantica("split_desconocido", _split_desconocido)

    # Bug C4 #5: restored_monitor_value no numérico (p. ej. "abc" por una
    # corrida corrupta) hacía que np.isfinite() lanzara TypeError, que se
    # propagaba sin capturar en vez de convertirse en un diagnóstico. Se
    # prueba por separado porque el contrato exigido es más fuerte que "hay
    # un problema en la lista": ni validate_run_artifacts() ni
    # collect(strict=True) deben dejar escapar una excepción de contenido.
    root_f = root / "caso_f_restored_monitor_value_no_numerico"
    run_f = root_f / "run_restored_no_numerico"
    _write_schema4_fixture(run_f)
    p_f = run_f / "metrics_val.csv"
    frame_f = pd.read_csv(p_f)
    frame_f["restored_monitor_value"] = frame_f["restored_monitor_value"].astype(object)
    frame_f.loc[0, "restored_monitor_value"] = "abc"
    frame_f.to_csv(p_f, index=False)
    try:
        problems_f = C.validate_run_artifacts(run_f, "")
        (ok if problems_f else fail)(
            "caso F (restored_monitor_value no numérico): validate_run_artifacts() "
            "devuelve un diagnóstico en vez de lanzar" + ("" if problems_f else " — devolvió []")
        )
    except (TypeError, ValueError) as e:
        fail(
            "caso F (restored_monitor_value no numérico): validate_run_artifacts() "
            f"dejó escapar {type(e).__name__}: {e} en vez de devolver un diagnóstico"
        )
    try:
        C.collect(root_f, strict=True)
        fail("caso F (restored_monitor_value no numérico): collect(strict=True) no lanzó ValueError")
    except ValueError:
        ok(
            "caso F (restored_monitor_value no numérico): collect(strict=True) lanza "
            "ValueError (no TypeError) con el contenido corrupto"
        )
    except TypeError as e:
        fail(
            "caso F (restored_monitor_value no numérico): collect(strict=True) dejó "
            f"escapar TypeError: {e} en vez de ValueError"
        )

    # Bug C4 #6 (v12): el nombre de la carpeta es la identidad operativa de
    # la corrida — collect() y _find_run_dir() la ubican por ese nombre, no
    # por el contenido de config.json. Si alguien la renombra a mano (o
    # queda desincronizada por cualquier otro motivo), los artefactos siguen
    # siendo válidos pero la corrida se vuelve invisible para --stats y
    # comparaciones futuras por run_id. Se prueba por separado de los casos
    # A-F porque no es un problema de contenido de los CSV, sino de la
    # correspondencia entre el nombre de la carpeta y config.json["run_id"].
    root_g = root / "caso_g_carpeta_no_coincide_con_run_id"
    run_g_src = root_g / "run_id_original"
    _write_schema4_fixture(run_g_src)
    run_g = root_g / "carpeta_renombrada_a_mano"
    run_g_src.rename(run_g)
    problems_g = C.validate_run_artifacts(run_g, "")
    ok_g = (
        len(problems_g) == 1
        and "carpeta_renombrada_a_mano" in problems_g[0]
        and "run_id_original" in problems_g[0]
    )
    (ok if ok_g else fail)(
        "caso G (carpeta ≠ run_id): validate_run_artifacts() devuelve un diagnóstico "
        "que menciona el nombre real de la carpeta y el run_id declarado"
        + ("" if ok_g else f" — {problems_g}")
    )
    try:
        C.collect(root_g, strict=True)
        fail("caso G (carpeta ≠ run_id): collect(strict=True) no lanzó ValueError")
    except ValueError:
        ok("caso G (carpeta ≠ run_id): collect(strict=True) lanza ValueError")
    df_g = C.collect(root_g, strict=False)
    prob_g = []
    if len(df_g) != 0:
        prob_g.append(f"se esperaba 0 filas (la corrida mal nombrada se descarta), hay {len(df_g)}")
    avisos_g = df_g.attrs.get("collection_warnings", [])
    if not any("carpeta_renombrada_a_mano" in w and "run_id_original" in w for w in avisos_g):
        prob_g.append(f"collection_warnings no menciona ambos nombres: {avisos_g}")
    (ok if not prob_g else fail)(
        "caso G (carpeta ≠ run_id): collect(strict=False) descarta la corrida y "
        "conserva el diagnóstico en collection_warnings"
        + (f" — {'; '.join(prob_g)}" if prob_g else "")
    )

    # Una corrida correctamente nombrada (carpeta == run_id) debe seguir
    # aceptándose sin cambios — no es una regla nueva sobre corridas sanas,
    # solo un rechazo de las mal nombradas. Se prueba en ambos layouts
    # (plano histórico y anidado por ROI) porque _find_run_dir() y
    # collect() deben aceptar los dos cuando la carpeta final es correcta.
    root_h = root / "caso_h_layouts_compatibles"

    run_h_plano = root_h / "plano" / "NYU_run_layout_plano"
    _write_schema4_fixture(run_h_plano)
    problems_h1 = C.validate_run_artifacts(run_h_plano, "")
    (ok if not problems_h1 else fail)(
        "caso H (layout plano, bien nombrada): validate_run_artifacts() devuelve []"
        + (f" — {problems_h1}" if problems_h1 else "")
    )
    df_h1 = C.collect(root_h / "plano", strict=True)
    (ok if len(df_h1) == 1 else fail)(
        f"caso H (layout plano): collect(strict=True) acepta 1 fila, hay {len(df_h1)}"
    )

    run_id_anidado = "NYU_run_layout_anidado"
    run_h_anidado = root_h / "anidado" / "12" / run_id_anidado
    _write_schema4_fixture(run_h_anidado)
    problems_h2 = C.validate_run_artifacts(run_h_anidado, "")
    (ok if not problems_h2 else fail)(
        "caso H (layout anidado por ROI, bien nombrada): validate_run_artifacts() devuelve []"
        + (f" — {problems_h2}" if problems_h2 else "")
    )
    df_h2 = C.collect(root_h / "anidado", strict=True)
    (ok if len(df_h2) == 1 else fail)(
        f"caso H (layout anidado): collect(strict=True) acepta 1 fila, hay {len(df_h2)}"
    )

    encontrada = C._find_run_dir(root_h / "anidado", run_id_anidado)
    esperada = root_h / "anidado" / "12" / run_id_anidado
    (ok if encontrada == esperada else fail)(
        f"caso H: _find_run_dir(root, run_id) == root/roi_set/run_id — se obtuvo {encontrada}, "
        f"se esperaba {esperada}"
    )

    # --- T1: n_splits/n_repeats/n_subjects ausentes o inválidos en config.json ---
    def _config_case(nombre: str, mutar_cfg, campos_esperados: list[str]) -> None:
        root_t = root / f"caso_t1_{nombre}"
        run_t = root_t / "run_config"
        _write_schema4_fixture(run_t)
        cfg_path = run_t / "config.json"
        cfg = json.loads(cfg_path.read_text())
        mutar_cfg(cfg)
        cfg_path.write_text(json.dumps(cfg))

        problems = C.validate_run_artifacts(run_t, "")
        texto = " | ".join(problems)
        faltan_en_lista = [c for c in campos_esperados if c not in texto]
        (ok if not faltan_en_lista else fail)(
            f"T1 ({nombre}): validate_run_artifacts() menciona {campos_esperados}"
            + (f" — no menciona {faltan_en_lista}: {problems}" if faltan_en_lista else "")
        )
        try:
            C.collect(root_t, strict=True)
            fail(f"T1 ({nombre}): collect(strict=True) no lanzó ValueError")
        except ValueError as e:
            msg = str(e)
            prob = []
            if run_t.name not in msg:
                prob.append(f"el mensaje no identifica la corrida ({run_t.name!r})")
            faltan_en_msg = [c for c in campos_esperados if c not in msg]
            if faltan_en_msg:
                prob.append(f"el mensaje no menciona {faltan_en_msg}")
            (ok if not prob else fail)(
                f"T1 ({nombre}): collect(strict=True) lanza ValueError identificando la "
                "corrida y el/los campo(s)" + (f" — {'; '.join(prob)}; mensaje={msg!r}" if prob else "")
            )

    _config_case("n_splits_ausente", lambda cfg: cfg.pop("n_splits", None), ["n_splits"])
    _config_case("n_repeats_ausente", lambda cfg: cfg.pop("n_repeats", None), ["n_repeats"])
    _config_case("n_subjects_ausente", lambda cfg: cfg.pop("n_subjects", None), ["n_subjects"])
    _config_case(
        "tres_ausentes_a_la_vez",
        lambda cfg: [cfg.pop(k, None) for k in ("n_splits", "n_repeats", "n_subjects")],
        ["n_splits", "n_repeats", "n_subjects"],
    )
    _config_case("n_splits_booleano", lambda cfg: cfg.__setitem__("n_splits", True), ["n_splits"])
    _config_case("n_repeats_fuera_de_rango", lambda cfg: cfg.__setitem__("n_repeats", 0), ["n_repeats"])

    # --- T2: config_schema_version inválido (no debe escapar TypeError) ---
    for valor, nombre in (("abc", "texto"), (True, "booleano"), (0, "cero")):
        root_t2 = root / f"caso_t2_{nombre}"
        run_t2 = root_t2 / "run_schema"
        _write_schema4_fixture(run_t2)
        cfg_path = run_t2 / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["config_schema_version"] = valor
        cfg_path.write_text(json.dumps(cfg))
        try:
            problems = C.validate_run_artifacts(run_t2, "")
            prob = []
            if not problems:
                prob.append("devolvió []")
            elif "config_schema_version" not in " | ".join(problems):
                prob.append(f"no menciona config_schema_version: {problems}")
            (ok if not prob else fail)(
                f"T2 (config_schema_version={valor!r}): validate_run_artifacts() devuelve "
                "una lista mencionando config_schema_version" + (f" — {'; '.join(prob)}" if prob else "")
            )
        except (TypeError, ValueError) as e:
            fail(
                f"T2 (config_schema_version={valor!r}): validate_run_artifacts() dejó "
                f"escapar {type(e).__name__}: {e}"
            )
        try:
            C.collect(root_t2, strict=True)
            fail(f"T2 (config_schema_version={valor!r}): collect(strict=True) no lanzó ValueError")
        except ValueError as e:
            msg = str(e)
            (ok if run_t2.name in msg and "config_schema_version" in msg else fail)(
                f"T2 (config_schema_version={valor!r}): collect(strict=True) lanza ValueError "
                "identificando la corrida y config_schema_version"
                + ("" if run_t2.name in msg and "config_schema_version" in msg else f" — mensaje={msg!r}")
            )
        except TypeError as e:
            fail(f"T2 (config_schema_version={valor!r}): collect(strict=True) dejó escapar TypeError: {e}")

    # --- T3: split ausente de un fold con n_fit/n_inner_val ajustados a juego (P2) ---
    root_t3 = root / "caso_t3_split_ausente_conteos_coordinados"
    run_t3 = root_t3 / "run_split_ausente"
    _write_schema4_fixture(run_t3)
    folds_path = run_t3 / "folds.csv"
    folds_frame = pd.read_csv(folds_path)
    fila0 = folds_frame.iloc[0]
    fold_v, repeat_v = fila0["fold"], fila0["repeat"]
    mask_inner = (
        (folds_frame["fold"] == fold_v) & (folds_frame["repeat"] == repeat_v)
        & (folds_frame["split"] == "inner_val")
    )
    n_movido = int(mask_inner.sum())
    folds_frame.loc[mask_inner, "split"] = "fit"
    folds_frame.to_csv(folds_path, index=False)
    for fname in ("metrics_train.csv", "metrics_val.csv"):
        p = run_t3 / fname
        m = pd.read_csv(p)
        row_mask = (m["fold"] == fold_v) & (m["repeat"] == repeat_v)
        m.loc[row_mask, "n_fit"] = m.loc[row_mask, "n_fit"] + n_movido
        m.loc[row_mask, "n_inner_val"] = m.loc[row_mask, "n_inner_val"] - n_movido
        m.to_csv(p, index=False)

    problems = C.validate_run_artifacts(run_t3, "")
    texto = " | ".join(problems)
    (ok if "inner_val" in texto else fail)(
        "T3 (split inner_val ausente de un fold, n_fit/n_inner_val coordinados): "
        "validate_run_artifacts() menciona 'inner_val' ausente"
        + ("" if "inner_val" in texto else f" — problems={problems}")
    )
    try:
        C.collect(root_t3, strict=True)
        fail("T3: collect(strict=True) no lanzó ValueError")
    except ValueError as e:
        msg = str(e)
        (ok if run_t3.name in msg and "inner_val" in msg else fail)(
            "T3: collect(strict=True) lanza ValueError identificando la corrida y 'inner_val'"
            + ("" if run_t3.name in msg and "inner_val" in msg else f" — mensaje={msg!r}")
        )

    # --- T4: casos pendientes del contrato anterior ---
    def _fold_completo_ausente_en_folds_csv(run_dir: Path) -> None:
        # A diferencia de "fold_completo_ausente" (que lo quita de
        # metrics_train.csv), aquí el (repeat, fold) desaparece de folds.csv
        # específicamente.
        p = run_dir / "folds.csv"
        frame = pd.read_csv(p)
        objetivo = frame[["repeat", "fold"]].drop_duplicates().iloc[0]
        mascara = (frame["repeat"] == objetivo["repeat"]) & (frame["fold"] == objetivo["fold"])
        frame[~mascara].to_csv(p, index=False)

    def _clave_repeat_fold_extra_en_predictions(run_dir: Path) -> None:
        # A diferencia de "subject_id_duplicado_oof" (duplica una fila con la
        # misma clave), esto agrega una clave (repeat, fold) que no existe en
        # metrics_val.csv en absoluto.
        p = run_dir / "predictions_val.csv"
        frame = pd.read_csv(p)
        fila_extra = pd.DataFrame([{
            "fold": 99, "repeat": 1, "subject_id": "s_extra", "y_true": 0, "y_prob": 0.5,
        }])
        pd.concat([frame, fila_extra], ignore_index=True).to_csv(p, index=False)

    _semantica("fold_completo_ausente_en_folds_csv", _fold_completo_ausente_en_folds_csv)
    _semantica("clave_repeat_fold_extra_en_predictions_val", _clave_repeat_fold_extra_en_predictions)

    def _construir_fixture_dos_repeticiones(run_dir: Path) -> None:
        # Duplica el fixture válido de una repetición como su propia segunda
        # repetición (mismos fold/subject_id, repeat=2) y fija n_repeats=2 en
        # config.json — para probar comprobaciones que solo se manifiestan
        # con más de una repetición (consistencia de y_true entre ellas).
        _write_schema4_fixture(run_dir)
        cfg_path = run_dir / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["n_repeats"] = 2
        cfg_path.write_text(json.dumps(cfg))
        for fname in ("metrics_train.csv", "metrics_val.csv", "history.csv",
                      "predictions_val.csv", "folds.csv"):
            p = run_dir / fname
            frame = pd.read_csv(p)
            dup = frame.copy()
            dup["repeat"] = 2
            pd.concat([frame, dup], ignore_index=True).to_csv(p, index=False)

    root_t4_base = root / "caso_t4_dos_repeticiones_sin_mutar"
    run_t4_base = root_t4_base / "run_dos_repeticiones"
    _construir_fixture_dos_repeticiones(run_t4_base)
    problems = C.validate_run_artifacts(run_t4_base, "")
    (ok if not problems else fail)(
        "T4 (fixture de dos repeticiones, sin mutar): validate_run_artifacts() devuelve []"
        + (f" — {problems}" if problems else "")
    )

    root_t4_ytrue = root / "caso_t4_y_true_inconsistente_entre_repeticiones"
    run_t4_ytrue = root_t4_ytrue / "run_y_true_inconsistente"
    _construir_fixture_dos_repeticiones(run_t4_ytrue)
    p_pred = run_t4_ytrue / "predictions_val.csv"
    pred_frame = pd.read_csv(p_pred)
    objetivo = (pred_frame["repeat"] == 2) & (pred_frame["subject_id"] == "s1")
    pred_frame.loc[objetivo, "y_true"] = 1 - pred_frame.loc[objetivo, "y_true"]
    pred_frame.to_csv(p_pred, index=False)
    problems = C.validate_run_artifacts(run_t4_ytrue, "")
    texto = " | ".join(problems)
    (ok if "y_true" in texto and "s1" in texto else fail)(
        "T4 (y_true distinto para 's1' entre repeticiones): validate_run_artifacts() lo detecta"
        + ("" if "y_true" in texto and "s1" in texto else f" — problems={problems}")
    )
    try:
        C.collect(root_t4_ytrue, strict=True)
        fail("T4 (y_true inconsistente entre repeticiones): collect(strict=True) no lanzó ValueError")
    except ValueError as e:
        (ok if run_t4_ytrue.name in str(e) else fail)(
            "T4 (y_true inconsistente entre repeticiones): collect(strict=True) lanza "
            "ValueError identificando la corrida"
            + ("" if run_t4_ytrue.name in str(e) else f" — mensaje={e!r}")
        )

    shutil.rmtree(root)


def _write_ensemble_source_run(
    run_dir: Path,
    *,
    roi_set: str,
    config_hash: str,
    y_prob_by_subject: dict,
    y_true_by_subject: dict,
    fold_by_subject: dict | None = None,
    subject_index_by_subject_id: dict | None = None,
    run_id_override: str | None = None,
    site: str = "NYU",
    bold_hash: str = "bold0000000000",
    split_fingerprint: str = "split00000000",
    seed: int = 42,
    n_splits: int = 2,
    n_repeats: int = 1,
) -> None:
    """Escribe config.json + predictions_val.csv de una corrida fuente mínima
    para analyze_ensemble.py — sin folds.csv/metrics_*.csv/history.csv, que
    esa utilidad no lee. subject_id/y_prob/y_true vienen de diccionarios
    {(repeat, subject_id): valor} para que las pruebas puedan construir
    exactamente el desacuerdo o la coincidencia que necesitan.

    predictions_val.csv incluye la columna `subject` (índice interno), que
    analyze_ensemble.py exige desde la validación reforzada. Por defecto se
    deriva de forma determinista de `subject_id` (los dígitos finales, p. ej.
    "s3" -> 3) para que dos corridas fuente con los mismos subject_id
    compartan el mismo subject sin tener que pasarlo explícitamente en cada
    prueba; `subject_index_by_subject_id` fuerza una asignación distinta
    cuando la prueba necesita justamente esa discrepancia.
    `run_id_override` escribe un run_id distinto del nombre de la carpeta en
    config.json, para las pruebas de identidad carpeta/run_id.
    """
    import pandas as pd

    run_dir.mkdir(parents=True, exist_ok=True)
    n_subjects = len({sid for (_, sid) in y_prob_by_subject})
    cfg = {
        "run_id": run_id_override if run_id_override is not None else run_dir.name,
        "config_schema_version": 4, "config_hash": config_hash,
        "site": site, "roi_set": roi_set, "bold_hash": bold_hash,
        "split_fingerprint": split_fingerprint, "seed": seed,
        "n_splits": n_splits, "n_repeats": n_repeats, "n_subjects": n_subjects,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    def _subject_index(subject_id: str) -> int:
        if subject_index_by_subject_id and subject_id in subject_index_by_subject_id:
            return subject_index_by_subject_id[subject_id]
        digitos = "".join(ch for ch in subject_id if ch.isdigit())
        return int(digitos) if digitos else abs(hash(subject_id)) % 1_000_000

    rows = []
    for (repeat, subject_id), y_prob in y_prob_by_subject.items():
        fold = (fold_by_subject or {}).get((repeat, subject_id), 1)
        rows.append({
            "repeat": repeat, "fold": fold, "subject": _subject_index(subject_id),
            "subject_id": subject_id,
            "y_true": y_true_by_subject[(repeat, subject_id)], "y_prob": y_prob,
        })
    pd.DataFrame(rows).to_csv(run_dir / "predictions_val.csv", index=False)


def check_ensemble_analysis():
    """Regresiones de analyze_ensemble.py con fixtures pequeños (sin TensorFlow).

    Caso 1: dos corridas válidas — la probabilidad resultante es exactamente
    el promedio. Caso 2: split_fingerprint distinto falla. Caso 3: un sujeto
    distinto entre corridas (clave (repeat, fold, subject_id) que no coincide)
    falla. Caso 4: mismo sujeto, y_true distinto entre corridas, falla. Caso
    5: clave duplicada dentro de una misma corrida falla. Caso 6: y_prob
    ausente/infinito/fuera de [0,1] falla. Caso 7: el mismo run_id presente a
    la vez en el layout plano y en el layout por ROI falla por ambigüedad (no
    se elige una copia arbitrariamente). Caso 8: una salida existente sin
    --overwrite falla y no la modifica. Caso 9: las métricas OOF por
    repetición agrupan todos los folds de la repetición antes de calcular la
    métrica, no promedian las métricas de cada fold por separado — se
    construye un caso donde ambos cálculos dan números distintos y se
    confirma cuál de los dos se reportó.

    Casos 10-16 (validación reforzada, v13): 10: config.json declara más
    repeticiones que las presentes en predictions_val.csv, falla mencionando
    la cobertura de repeticiones. 11: dentro de una repetición hay menos
    folds distintos que n_splits, falla mencionando fold. 12: config.json
    declara un run_id distinto del nombre de carpeta, falla señalando la
    discrepancia. 13: mismo subject_id pero distinto subject (índice interno)
    entre las dos corridas, falla por clave JOIN_KEYS incompatible (igual que
    el caso 3, pero variando subject en vez de subject_id). 14: dentro de UNA
    corrida, el mismo sujeto tiene y_true distinto entre dos repeticiones,
    falla antes de comparar contra la otra corrida. 15: --runs A B y --runs
    B A producen el mismo nombre de carpeta de salida, el mismo source_runs/
    weights y las mismas predicciones. 16: campos estructurales inválidos en
    config.json (n_repeats ausente, n_splits booleano, n_subjects=0) fallan
    mencionando el requisito de entero.
    """
    seccion("Ensamble de probabilidades OOF (analyze_ensemble.py, sin entrenar)")
    import shutil
    import pandas as pd
    sys.path.insert(0, str(REPO / "src"))
    import analyze_ensemble as E

    root = Path("/tmp/verify_ensemble_fixtures")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # Partición compartida de referencia: 4 sujetos, 1 repetición, 2 folds
    # (s1/s2 en el fold 1, s3/s4 en el fold 2) — analyze_ensemble.py no lee
    # folds.csv, así que basta con que predictions_val.csv sea internamente
    # consistente.
    y_true = {(1, "s1"): 1, (1, "s2"): 0, (1, "s3"): 1, (1, "s4"): 0}
    folds = {(1, "s1"): 1, (1, "s2"): 1, (1, "s3"): 2, (1, "s4"): 2}

    # --- Caso 1: dos corridas válidas, promedio exacto ---
    root1 = root / "caso1"
    run_a = root1 / "12" / "RUN_A"
    run_b = root1 / "18" / "RUN_B"
    prob_a = {(1, "s1"): 0.8, (1, "s2"): 0.2, (1, "s3"): 0.6, (1, "s4"): 0.3}
    prob_b = {(1, "s1"): 0.6, (1, "s2"): 0.4, (1, "s3"): 0.9, (1, "s4"): 0.1}
    _write_ensemble_source_run(run_a, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_b, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    out_dir = E.run_ensemble(root1, ["RUN_A", "RUN_B"], root1 / "analyses", overwrite=False)
    pred_out = pd.read_csv(out_dir / "predictions_val.csv")
    esperado = {sid: (prob_a[(1, sid)] + prob_b[(1, sid)]) / 2 for (_, sid) in prob_a}
    obtenido = dict(zip(pred_out["subject_id"], pred_out["y_prob"]))
    coincide = all(abs(obtenido[sid] - esperado[sid]) < 1e-9 for sid in esperado)
    (ok if coincide else fail)(
        "caso 1 (dos corridas válidas): y_prob del ensamble es exactamente el promedio"
        + ("" if coincide else f" — obtenido={obtenido}, esperado={esperado}")
    )
    for f in ("config.json", "predictions_val.csv", "metrics_oof_by_repeat.csv", "resumen.md"):
        (ok if (out_dir / f).exists() else fail)(f"caso 1: {f} presente en la carpeta de análisis")
    cfg_out = json.loads((out_dir / "config.json").read_text())
    (ok if cfg_out.get("artifact_type") == "oof_probability_ensemble" else fail)(
        "caso 1: config.json declara artifact_type=oof_probability_ensemble"
    )
    (ok if cfg_out.get("threshold") == 0.5 else fail)("caso 1: config.json declara threshold=0.5")

    # --- Caso 2: split_fingerprint distinto ---
    root2 = root / "caso2"
    run_a2 = root2 / "12" / "RUN_A"
    run_b2 = root2 / "18" / "RUN_B"
    _write_ensemble_source_run(run_a2, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_b2, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds, split_fingerprint="split_DISTINTA")
    try:
        E.run_ensemble(root2, ["RUN_A", "RUN_B"], root2 / "analyses", overwrite=False)
        fail("caso 2 (split_fingerprint distinto): no lanzó EnsembleError")
    except E.EnsembleError as e:
        (ok if "split_fingerprint" in str(e) else fail)(
            "caso 2 (split_fingerprint distinto): EnsembleError menciona split_fingerprint"
            + ("" if "split_fingerprint" in str(e) else f" — mensaje={e!r}")
        )

    # --- Caso 3: sujeto distinto entre corridas ---
    root3 = root / "caso3"
    run_a3 = root3 / "12" / "RUN_A"
    run_b3 = root3 / "18" / "RUN_B"
    y_true_b3 = dict(y_true)
    del y_true_b3[(1, "s4")]
    y_true_b3[(1, "s5")] = 0
    prob_b3 = dict(prob_b)
    del prob_b3[(1, "s4")]
    prob_b3[(1, "s5")] = 0.1
    folds_b3 = dict(folds)
    del folds_b3[(1, "s4")]
    folds_b3[(1, "s5")] = 2
    _write_ensemble_source_run(run_a3, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_b3, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b3, y_true_by_subject=y_true_b3,
                                fold_by_subject=folds_b3)
    try:
        E.run_ensemble(root3, ["RUN_A", "RUN_B"], root3 / "analyses", overwrite=False)
        fail("caso 3 (sujeto distinto entre corridas): no lanzó EnsembleError")
    except E.EnsembleError as e:
        (ok if "claves" in str(e) else fail)(
            "caso 3 (sujeto distinto entre corridas): EnsembleError menciona las claves"
            + ("" if "claves" in str(e) else f" — mensaje={e!r}")
        )

    # --- Caso 4: y_true distinto para la misma clave ---
    root4 = root / "caso4"
    run_a4 = root4 / "12" / "RUN_A"
    run_b4 = root4 / "18" / "RUN_B"
    y_true_b4 = dict(y_true)
    y_true_b4[(1, "s1")] = 0  # era 1 en run_a
    _write_ensemble_source_run(run_a4, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_b4, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true_b4,
                                fold_by_subject=folds)
    try:
        E.run_ensemble(root4, ["RUN_A", "RUN_B"], root4 / "analyses", overwrite=False)
        fail("caso 4 (y_true distinto): no lanzó EnsembleError")
    except E.EnsembleError as e:
        (ok if "y_true" in str(e) else fail)(
            "caso 4 (y_true distinto): EnsembleError menciona y_true"
            + ("" if "y_true" in str(e) else f" — mensaje={e!r}")
        )

    # --- Caso 5: clave duplicada dentro de una corrida ---
    root5 = root / "caso5"
    run_a5 = root5 / "12" / "RUN_A"
    run_b5 = root5 / "18" / "RUN_B"
    _write_ensemble_source_run(run_a5, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_b5, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    pred_b5 = pd.read_csv(run_b5 / "predictions_val.csv")
    pred_b5 = pd.concat([pred_b5, pred_b5.iloc[[0]]], ignore_index=True)
    pred_b5.to_csv(run_b5 / "predictions_val.csv", index=False)
    try:
        E.run_ensemble(root5, ["RUN_A", "RUN_B"], root5 / "analyses", overwrite=False)
        fail("caso 5 (clave duplicada): no lanzó EnsembleError")
    except E.EnsembleError as e:
        (ok if "duplicada" in str(e) else fail)(
            "caso 5 (clave duplicada): EnsembleError menciona la duplicación"
            + ("" if "duplicada" in str(e) else f" — mensaje={e!r}")
        )

    # --- Caso 6: y_prob ausente/infinito/fuera de rango ---
    for nombre, valor_malo in (("nan", float("nan")), ("infinito", float("inf")), ("fuera_de_rango", 1.5)):
        root6 = root / f"caso6_{nombre}"
        run_a6 = root6 / "12" / "RUN_A"
        run_b6 = root6 / "18" / "RUN_B"
        _write_ensemble_source_run(run_a6, roi_set="12", config_hash="aaaa1111",
                                    y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                    fold_by_subject=folds)
        prob_b6 = dict(prob_b)
        prob_b6[(1, "s1")] = valor_malo
        _write_ensemble_source_run(run_b6, roi_set="18", config_hash="bbbb2222",
                                    y_prob_by_subject=prob_b6, y_true_by_subject=y_true,
                                    fold_by_subject=folds)
        try:
            E.run_ensemble(root6, ["RUN_A", "RUN_B"], root6 / "analyses", overwrite=False)
            fail(f"caso 6 ({nombre}): no lanzó EnsembleError")
        except E.EnsembleError as e:
            (ok if "y_prob" in str(e) else fail)(
                f"caso 6 ({nombre}): EnsembleError menciona y_prob"
                + ("" if "y_prob" in str(e) else f" — mensaje={e!r}")
            )

    # --- Caso 7: mismo run_id en layout plano y por ROI a la vez ---
    root7 = root / "caso7"
    run_plano = root7 / "RUN_A"
    run_anidado = root7 / "12" / "RUN_A"
    run_b7 = root7 / "18" / "RUN_B"
    _write_ensemble_source_run(run_plano, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_anidado, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_b7, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    try:
        E.run_ensemble(root7, ["RUN_A", "RUN_B"], root7 / "analyses", overwrite=False)
        fail("caso 7 (run_id en ambos layouts): no lanzó EnsembleError")
    except E.EnsembleError as e:
        (ok if "más de una ubicación" in str(e) else fail)(
            "caso 7 (run_id en ambos layouts): EnsembleError señala la ambigüedad"
            + ("" if "más de una ubicación" in str(e) else f" — mensaje={e!r}")
        )

    # --- Caso 8: salida existente sin --overwrite ---
    root8 = root / "caso8"
    run_a8 = root8 / "12" / "RUN_A"
    run_b8 = root8 / "18" / "RUN_B"
    _write_ensemble_source_run(run_a8, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_b8, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    out_dir8 = E.run_ensemble(root8, ["RUN_A", "RUN_B"], root8 / "analyses", overwrite=False)
    contenido_antes = (out_dir8 / "config.json").read_text()
    try:
        E.run_ensemble(root8, ["RUN_A", "RUN_B"], root8 / "analyses", overwrite=False)
        fail("caso 8 (salida existente sin --overwrite): no lanzó EnsembleError")
    except E.EnsembleError as e:
        contenido_despues = (out_dir8 / "config.json").read_text()
        sin_cambios = contenido_antes == contenido_despues
        (ok if "overwrite" in str(e) and sin_cambios else fail)(
            "caso 8 (salida existente sin --overwrite): EnsembleError menciona --overwrite "
            "y no modifica la salida existente"
            + ("" if "overwrite" in str(e) and sin_cambios else f" — mensaje={e!r}, "
               f"sin_cambios={sin_cambios}")
        )
    # con --overwrite sí debe completarse
    E.run_ensemble(root8, ["RUN_A", "RUN_B"], root8 / "analyses", overwrite=True)
    ok("caso 8: con --overwrite, la misma combinación se puede regenerar")

    # --- Caso 9: OOF por repetición agrupa folds, no promedia sus métricas ---
    # repeat 1: fold 1 con 3 sujetos (2 aciertos), fold 2 con 1 sujeto (0 aciertos).
    # media ingenua de folds = (2/3 + 0/1) / 2 = 1/3; agrupado (correcto) = 2/4 = 0.5.
    df9 = pd.DataFrame([
        {"repeat": 1, "fold": 1, "subject_id": "s1", "y_true": 1, "y_prob": 0.9},  # acierto
        {"repeat": 1, "fold": 1, "subject_id": "s2", "y_true": 0, "y_prob": 0.1},  # acierto
        {"repeat": 1, "fold": 1, "subject_id": "s3", "y_true": 1, "y_prob": 0.4},  # fallo
        {"repeat": 1, "fold": 2, "subject_id": "s4", "y_true": 1, "y_prob": 0.2},  # fallo
    ])
    por_repeticion = E._metrics_oof_by_repeat(df9)
    n_filas_correctas = len(por_repeticion) == 1  # una fila por repetición, no por fold
    acc_agrupada = float(por_repeticion.loc[por_repeticion["repeat"] == 1, "accuracy"].iloc[0])
    (ok if n_filas_correctas else fail)(
        f"caso 9: una fila por repetición en metrics_oof_by_repeat, hay {len(por_repeticion)}"
    )
    (ok if abs(acc_agrupada - 0.5) < 1e-9 else fail)(
        f"caso 9: accuracy agrupada por repetición es 0.5 (2/4), no la media ingenua de folds "
        f"(≈0.333) — se obtuvo {acc_agrupada}"
    )

    # --- Caso 10: repetición ausente (config.json declara n_repeats=2, solo
    # hay datos de la repetición 1) ---
    root10 = root / "caso10"
    run_a10 = root10 / "12" / "RUN_A"
    run_b10 = root10 / "18" / "RUN_B"
    _write_ensemble_source_run(run_a10, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds, n_repeats=2)
    _write_ensemble_source_run(run_b10, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds, n_repeats=2)
    try:
        E.run_ensemble(root10, ["RUN_A", "RUN_B"], root10 / "analyses", overwrite=False)
        fail("caso 10 (repetición ausente): no lanzó EnsembleError")
    except E.EnsembleError as e:
        (ok if "repeticiones" in str(e) else fail)(
            "caso 10 (repetición ausente): EnsembleError menciona la cobertura de repeticiones"
            + ("" if "repeticiones" in str(e) else f" — mensaje={e!r}")
        )

    # --- Caso 11: fold ausente (n_splits=2, pero todos los sujetos caen en
    # el mismo fold dentro de la repetición) ---
    root11 = root / "caso11"
    run_a11 = root11 / "12" / "RUN_A"
    run_b11 = root11 / "18" / "RUN_B"
    folds_un_solo_fold = {clave: 1 for clave in folds}
    _write_ensemble_source_run(run_a11, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds_un_solo_fold, n_splits=2)
    _write_ensemble_source_run(run_b11, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds_un_solo_fold, n_splits=2)
    try:
        E.run_ensemble(root11, ["RUN_A", "RUN_B"], root11 / "analyses", overwrite=False)
        fail("caso 11 (fold ausente): no lanzó EnsembleError")
    except E.EnsembleError as e:
        (ok if "fold" in str(e) else fail)(
            "caso 11 (fold ausente): EnsembleError menciona fold"
            + ("" if "fold" in str(e) else f" — mensaje={e!r}")
        )

    # --- Caso 12: run_id inconsistente entre config.json y la carpeta ---
    root12 = root / "caso12"
    run_a12 = root12 / "12" / "RUN_A"
    run_b12 = root12 / "18" / "RUN_B"
    _write_ensemble_source_run(run_a12, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds, run_id_override="RUN_A_OTRO")
    _write_ensemble_source_run(run_b12, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    try:
        E.run_ensemble(root12, ["RUN_A", "RUN_B"], root12 / "analyses", overwrite=False)
        fail("caso 12 (run_id inconsistente): no lanzó EnsembleError")
    except E.EnsembleError as e:
        (ok if "no coinciden" in str(e) else fail)(
            "caso 12 (run_id inconsistente): EnsembleError señala la discrepancia"
            + ("" if "no coinciden" in str(e) else f" — mensaje={e!r}")
        )

    # --- Caso 13: mismo subject_id, distinto subject (índice interno) entre
    # las dos corridas ---
    root13 = root / "caso13"
    run_a13 = root13 / "12" / "RUN_A"
    run_b13 = root13 / "18" / "RUN_B"
    _write_ensemble_source_run(run_a13, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_b13, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds,
                                subject_index_by_subject_id={"s1": 999})
    try:
        E.run_ensemble(root13, ["RUN_A", "RUN_B"], root13 / "analyses", overwrite=False)
        fail("caso 13 (subject incompatible): no lanzó EnsembleError")
    except E.EnsembleError as e:
        (ok if "claves" in str(e) else fail)(
            "caso 13 (subject incompatible): EnsembleError menciona las claves"
            + ("" if "claves" in str(e) else f" — mensaje={e!r}")
        )

    # --- Caso 14: y_true inconsistente entre repeticiones DENTRO de una
    # misma corrida (no llega a compararse contra la otra) ---
    root14 = root / "caso14"
    run_a14 = root14 / "12" / "RUN_A"
    run_b14 = root14 / "18" / "RUN_B"
    y_true_2rep = {
        (1, "s1"): 1, (1, "s2"): 0, (1, "s3"): 1, (1, "s4"): 0,
        (2, "s1"): 0, (2, "s2"): 0, (2, "s3"): 1, (2, "s4"): 0,  # s1 cambia de 1 a 0
    }
    prob_2rep = {
        (1, "s1"): 0.8, (1, "s2"): 0.2, (1, "s3"): 0.6, (1, "s4"): 0.3,
        (2, "s1"): 0.7, (2, "s2"): 0.25, (2, "s3"): 0.55, (2, "s4"): 0.35,
    }
    folds_2rep = {
        (1, "s1"): 1, (1, "s2"): 1, (1, "s3"): 2, (1, "s4"): 2,
        (2, "s1"): 1, (2, "s2"): 1, (2, "s3"): 2, (2, "s4"): 2,
    }
    _write_ensemble_source_run(run_a14, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_2rep, y_true_by_subject=y_true_2rep,
                                fold_by_subject=folds_2rep, n_repeats=2)
    _write_ensemble_source_run(run_b14, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_2rep, y_true_by_subject=y_true_2rep,
                                fold_by_subject=folds_2rep, n_repeats=2)
    try:
        E.run_ensemble(root14, ["RUN_A", "RUN_B"], root14 / "analyses", overwrite=False)
        fail("caso 14 (y_true inconsistente entre repeticiones): no lanzó EnsembleError")
    except E.EnsembleError as e:
        condicion = "y_true" in str(e) and "repeticiones" in str(e)
        (ok if condicion else fail)(
            "caso 14 (y_true inconsistente entre repeticiones): EnsembleError señala la "
            "inconsistencia dentro de la corrida"
            + ("" if condicion else f" — mensaje={e!r}")
        )

    # --- Caso 15: orden canónico — --runs A B y --runs B A producen la misma
    # combinación ---
    root15 = root / "caso15"
    run_a15 = root15 / "12" / "RUN_A"
    run_b15 = root15 / "18" / "RUN_B"
    _write_ensemble_source_run(run_a15, roi_set="12", config_hash="aaaa1111",
                                y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    _write_ensemble_source_run(run_b15, roi_set="18", config_hash="bbbb2222",
                                y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                fold_by_subject=folds)
    out_ab = E.run_ensemble(root15, ["RUN_A", "RUN_B"], root15 / "analyses_ab", overwrite=False)
    out_ba = E.run_ensemble(root15, ["RUN_B", "RUN_A"], root15 / "analyses_ba", overwrite=False)
    mismo_nombre = out_ab.name == out_ba.name
    (ok if mismo_nombre else fail)(
        "caso 15 (orden canónico): --runs A B y --runs B A producen el mismo nombre de carpeta"
        + ("" if mismo_nombre else f" — {out_ab.name} != {out_ba.name}")
    )
    cfg_ab = json.loads((out_ab / "config.json").read_text())
    cfg_ba = json.loads((out_ba / "config.json").read_text())
    mismo_source_runs = cfg_ab["source_runs"] == cfg_ba["source_runs"]
    mismos_pesos = cfg_ab["weights"] == cfg_ba["weights"]
    (ok if mismo_source_runs and mismos_pesos else fail)(
        "caso 15 (orden canónico): source_runs y weights de config.json son idénticos en "
        "ambos órdenes"
        + ("" if mismo_source_runs and mismos_pesos else
           f" — source_runs_ab={cfg_ab['source_runs']}, source_runs_ba={cfg_ba['source_runs']}")
    )
    pred_ab = pd.read_csv(out_ab / "predictions_val.csv")
    pred_ba = pd.read_csv(out_ba / "predictions_val.csv")
    predicciones_iguales = pred_ab.equals(pred_ba)
    (ok if predicciones_iguales else fail)(
        "caso 15 (orden canónico): predictions_val.csv es idéntico en ambos órdenes"
    )

    # --- Caso 16: campos estructurales inválidos en config.json ---
    for nombre, mutador in (
        ("n_repeats_ausente", lambda cfg: cfg.pop("n_repeats")),
        ("n_splits_booleano", lambda cfg: cfg.__setitem__("n_splits", True)),
        ("n_subjects_cero", lambda cfg: cfg.__setitem__("n_subjects", 0)),
    ):
        root16 = root / f"caso16_{nombre}"
        run_a16 = root16 / "12" / "RUN_A"
        run_b16 = root16 / "18" / "RUN_B"
        _write_ensemble_source_run(run_a16, roi_set="12", config_hash="aaaa1111",
                                    y_prob_by_subject=prob_a, y_true_by_subject=y_true,
                                    fold_by_subject=folds)
        _write_ensemble_source_run(run_b16, roi_set="18", config_hash="bbbb2222",
                                    y_prob_by_subject=prob_b, y_true_by_subject=y_true,
                                    fold_by_subject=folds)
        cfg_path16 = run_a16 / "config.json"
        cfg16 = json.loads(cfg_path16.read_text())
        mutador(cfg16)
        cfg_path16.write_text(json.dumps(cfg16), encoding="utf-8")
        try:
            E.run_ensemble(root16, ["RUN_A", "RUN_B"], root16 / "analyses", overwrite=False)
            fail(f"caso 16 ({nombre}): no lanzó EnsembleError")
        except E.EnsembleError as e:
            (ok if "entero" in str(e) else fail)(
                f"caso 16 ({nombre}): EnsembleError menciona el requisito de entero"
                + ("" if "entero" in str(e) else f" — mensaje={e!r}")
            )

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
    está listado); luego confirma que, si dos filas comparten todas las
    columnas agrupadas y aun así traen config_hash distinto,
    aggregate_table() aborta en vez de promediar. Por último (C5) repite el
    mismo patrón de "está en la lista y de verdad separa grupos" para los
    campos de identidad añadidos en C5 (representation_seed,
    start_from_epoch, random_subset/n_random_sets, seed, runner_code_hash) y
    confirma que las versiones de software/GPU siguen excluidas aunque estén
    presentes en el DataFrame.
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

    # --- C5: campos de identidad añadidos a methodological_group_columns() ---
    # Antes de C5 estos campos no estaban en la lista, así que dos corridas
    # que solo difirieran en, por ejemplo, la semilla de la representación o
    # el hash del código del runner se hubieran promediado juntas como si
    # fueran la misma configuración metodológica.
    base_c5 = {
        **base,
        "representation_seed": 1,
        "start_from_epoch": 0,
        "random_subset": 20,
        "n_random_sets": 5,
        "seed": 42,
        "runner_code_hash": "aaaa1111",
    }
    groups_c5 = C.methodological_group_columns(pd.DataFrame([base_c5]))
    prob_c5 = [c for c in ("representation_seed", "start_from_epoch", "random_subset",
                            "n_random_sets", "seed", "runner_code_hash") if c not in groups_c5]
    (ok if not prob_c5 else fail)(
        "methodological_group_columns() incluye representation_seed/start_from_epoch/"
        "random_subset/n_random_sets/seed/runner_code_hash"
        + (f" — faltan {prob_c5}" if prob_c5 else "")
    )

    variantes_c5 = {
        "start_from_epoch": 5,
        "representation_seed": 2,
        # random_subset y n_random_sets se prueban juntos: cambiar solo el
        # tamaño del subconjunto aleatorio de ROIs ya debe separar los grupos.
        "random_subset": 30,
        "seed": 7,
        "runner_code_hash": "bbbb2222",
    }
    for campo, valor_distinto in variantes_c5.items():
        fila_a = {**base_c5, "config_hash": f"hashA_{campo}", "base_run_id": f"runA_{campo}"}
        fila_b = {**base_c5, "config_hash": f"hashB_{campo}", "base_run_id": f"runB_{campo}",
                  campo: valor_distinto}
        agregada = C.aggregate_table(pd.DataFrame([fila_a, fila_b]))
        (ok if len(agregada) == 2 else fail)(
            f"dos corridas que solo difieren en {campo} quedan en grupos separados (C5)"
            + (f" — se esperaban 2 grupos, hay {len(agregada)}" if len(agregada) != 2 else "")
        )

    # Versiones de software/GPU quedan fuera a propósito (no son parte de
    # config_hash): confirmar que methodological_group_columns() no las
    # incluye aunque estén presentes en el DataFrame.
    base_software = {**base_c5, "tensorflow_version": "2.15.0", "gpu_name": "A100", "python_version": "3.11"}
    groups_software = C.methodological_group_columns(pd.DataFrame([base_software]))
    excluidas_presentes = [c for c in ("tensorflow_version", "gpu_name", "python_version")
                            if c in groups_software]
    (ok if not excluidas_presentes else fail)(
        "methodological_group_columns() excluye versiones de software/GPU aunque estén "
        "presentes en el DataFrame"
        + (f" — incluyó {excluidas_presentes}" if excluidas_presentes else "")
    )


def _extraer_celda_notebook(nb, cell_id):
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell["source"])
    raise KeyError(f"celda {cell_id!r} no encontrada en el notebook")


def check_notebook_state_machine():
    """Máquina de estados de tdha_experimentos.ipynb (C1-C3): preflight(),
    prueba_humo(), ejecutar_corrida(), ejecutar_diagnostico_orden() y
    exigir_corrida_validada(), ejecutadas fuera de Colab con un
    run_experiment.main() simulado (sin entrenar, sin importar TensorFlow).

    Se extrae el código real de las celdas del notebook (por id, para no
    mantener una copia manual que se desincronice) y se ejecuta en un
    namespace con las variables de la celda de configuración fijadas a un
    caso válido conocido — el mismo patrón de docs/guia-experimentacion-
    colaborativa.md. Prueba: preflight() éxito/fracaso; ejecutar_corrida()
    bloqueada sin preflight, bloqueada sin prueba de humo superada, y
    bloqueada si el run_id de la corrida formal no coincide con el que
    predijo preflight(); prueba_humo() fracaso; ejecutar_diagnostico_orden()
    rechazando representación estática y modelo brainnetcnn; y
    exigir_corrida_validada() rechazando corrida no validada, validación
    vieja (RUN_ID_VALIDADO distinto de RUN_ID) y corrida que cambió en disco
    después de validarse. T5 (correcciones v9): ejecutar_corrida() resetea
    CORRIDA_VALIDADA/RUN_ID_VALIDADO aunque partan de un estado previo válido
    de otra corrida (True/"RUN_ANTERIOR"), no solo de su valor inicial; la
    celda de diagnóstico con EJECUTAR_DIAGNOSTICO_ORDEN=False no llama al
    runner simulado ni una vez; y prueba_humo() corrida dos veces en la misma
    sesión tiene éxito ambas veces, con --overwrite y la misma carpeta
    temporal en las dos llamadas. T6 (correcciones v9): construir_argv() real
    del notebook, sin entrenar — ventana en segundos, ventana en TR, OVERLAP
    como alternativa a STEP_SECONDS, ventana gaussiana con GAUSSIAN_SIGMA,
    permuted con REPRESENTATION_SEED, DETERMINISTIC/MIXED_PRECISION/
    CLASS_WEIGHT/CLIPNORM, RANDOM_SUBSET/N_RANDOM_SETS/EXCLUDE_ROI_SET, y el
    rechazo de las combinaciones incompatibles ya documentadas (WINDOW_TR y
    WINDOW_SECONDS a la vez, ninguno de los dos, STEP_SECONDS y OVERLAP a la
    vez, GAUSSIAN_SIGMA sin WINDOW_SHAPE='gaussian').
    """
    seccion("Máquina de estados del notebook (preflight/prueba_humo/corrida/validación)")
    import contextlib
    import io
    import types

    def _silencioso(func, *args, **kwargs):
        # preflight()/prueba_humo() imprimen el "argv simulado" y mensajes de
        # progreso pensados para Colab; aquí solo interesa el resultado o la
        # excepción, así que se silencia stdout durante la llamada.
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args, **kwargs)

    nb_path = REPO / "tdha_experimentos.ipynb"
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    src_builder = _extraer_celda_notebook(nb, "9151337b")
    lineas = src_builder.splitlines()
    while lineas and not lineas[-1].strip():
        lineas.pop()
    if lineas[-1].strip() != "preflight()":
        fail(
            "check_notebook_state_machine(): la celda 9151337b no termina en una "
            "llamada a preflight() como se esperaba — se omite esta prueba para no "
            "correr una llamada real por accidente"
        )
        return
    src_builder_sin_llamada = "\n".join(lineas[:-1])

    src_gate_completo = _extraer_celda_notebook(nb, "4b8c48fb")
    idx = src_gate_completo.find("def exigir_corrida_validada")
    if idx == -1:
        fail("check_notebook_state_machine(): no se encontró def exigir_corrida_validada en la celda 4b8c48fb")
        return
    src_gate = src_gate_completo[idx:]

    config_valida = dict(
        SITIO="NYU", ROI_SET="12", MODELO="lstm", HIPERPARAMS={},
        REPRESENTACION="ordered", REPRESENTATION_SEED=None,
        FISHER_Z=False, CONSTANT_POLICY="zero",
        WINDOW_SECONDS=120, STEP_SECONDS=12, OVERLAP=None,
        WINDOW_TR=None, STEP_TR=None, TR_SECONDS=None,
        WINDOW_SHAPE="rectangular", GAUSSIAN_SIGMA=None,
        LR=1e-4, BATCH_SIZE=8, EPOCHS=150, PATIENCE=25,
        CLIPNORM=None, INNER_VAL_FRAC=0.15, START_FROM_EPOCH=0,
        EARLY_STOPPING_MONITOR="val_loss", EARLY_STOPPING_MIN_DELTA=1e-5,
        N_SPLITS=10, N_REPEATS=5, CLASS_WEIGHT=False, SEED=42,
        DETERMINISTIC=False, MIXED_PRECISION=False,
        RANDOM_SUBSET=None, N_RANDOM_SETS=20, EXCLUDE_ROI_SET=None,
        NOMBRE="Juan", CORREO="juan@ejemplo.com",
        TAG=None, OVERWRITE=False, EJECUTAR_PRUEBA_HUMO=True,
        EJECUTAR_DIAGNOSTICO_ORDEN=False,
    )
    src_diagnostico = _extraer_celda_notebook(nb, "9fe26788")

    def _main_falso(mode):
        def _main(argv):
            # Registra cada argv recibido (T5.3: confirmar cuántas veces se
            # llamó al runner y con qué argumentos, no solo el resultado).
            mode.setdefault("calls", []).append(list(argv))
            if "--dry-run" in argv:
                if mode.get("fail_dry_run"):
                    raise SystemExit(mode.get("dry_run_msg", "preflight simulado: fallo"))
                return mode.get("run_id_dry_run", "RUN_X")
            if "--out" in argv:
                if mode.get("fail_smoke"):
                    raise SystemExit(mode.get("smoke_msg", "prueba de humo simulada: fallo"))
                return "RUN_SMOKE"
            if mode.get("fail_formal"):
                raise SystemExit(mode.get("formal_msg", "corrida formal simulada: fallo"))
            return mode.get("run_id_formal", "RUN_X")
        return _main

    original_run_experiment = sys.modules.get("run_experiment")

    def _preparar_ns(mode, overrides=None):
        fake_module = types.ModuleType("run_experiment")
        fake_module.main = _main_falso(mode)
        sys.modules["run_experiment"] = fake_module
        ns = dict(config_valida)
        if overrides:
            ns.update(overrides)
        exec(compile(src_builder_sin_llamada, "<celda-constructor>", "exec"), ns)
        return ns

    try:
        # --- preflight(): éxito ---
        ns = _preparar_ns({"run_id_dry_run": "RUN_A"})
        run_id = _silencioso(ns["preflight"])
        prob = []
        if run_id != "RUN_A":
            prob.append(f"preflight() devolvió {run_id!r}, se esperaba 'RUN_A'")
        if ns["PREFLIGHT_OK"] is not True:
            prob.append(f"PREFLIGHT_OK quedó en {ns['PREFLIGHT_OK']!r}, se esperaba True")
        if ns["PREFLIGHT_RUN_ID"] != "RUN_A":
            prob.append(f"PREFLIGHT_RUN_ID quedó en {ns['PREFLIGHT_RUN_ID']!r}, se esperaba 'RUN_A'")
        (ok if not prob else fail)(
            "preflight() con --dry-run limpio: PREFLIGHT_OK=True, PREFLIGHT_RUN_ID fijado"
            + (f" — {'; '.join(prob)}" if prob else "")
        )

        # --- preflight(): fracaso ---
        ns = _preparar_ns({"fail_dry_run": True, "dry_run_msg": "AVISO_SIMULADO_PREFLIGHT"})
        try:
            _silencioso(ns["preflight"])
            fail("preflight() con --dry-run fallido no lanzó RuntimeError")
        except RuntimeError as e:
            prob = []
            if "AVISO_SIMULADO_PREFLIGHT" not in str(e):
                prob.append("el mensaje no incluye el aviso original")
            if ns["PREFLIGHT_OK"] is not False:
                prob.append(f"PREFLIGHT_OK quedó en {ns['PREFLIGHT_OK']!r}, se esperaba False")
            (ok if not prob else fail)(
                "preflight() con --dry-run fallido: RuntimeError con el aviso, PREFLIGHT_OK=False"
                + (f" — {'; '.join(prob)}" if prob else "")
            )

        # --- ejecutar_corrida() sin preflight ---
        ns = _preparar_ns({"run_id_dry_run": "RUN_A", "run_id_formal": "RUN_A"})
        try:
            _silencioso(ns["ejecutar_corrida"])
            fail("ejecutar_corrida() sin preflight() no lanzó RuntimeError")
        except RuntimeError as e:
            (ok if "preflight" in str(e) else fail)(
                "ejecutar_corrida() sin preflight() lanza RuntimeError mencionando preflight()"
                + (f" — mensaje={e!r}" if "preflight" not in str(e) else "")
            )

        # --- ejecutar_corrida() con preflight pero sin prueba de humo superada ---
        ns = _preparar_ns({"run_id_dry_run": "RUN_A", "run_id_formal": "RUN_A"})
        _silencioso(ns["preflight"])
        try:
            _silencioso(ns["ejecutar_corrida"])
            fail("ejecutar_corrida() sin prueba de humo superada (EJECUTAR_PRUEBA_HUMO=True) no lanzó RuntimeError")
        except RuntimeError as e:
            (ok if "humo" in str(e) else fail)(
                "ejecutar_corrida() sin prueba de humo superada lanza RuntimeError mencionando la prueba de humo"
                + (f" — mensaje={e!r}" if "humo" not in str(e) else "")
            )

        # --- prueba_humo(): fracaso ---
        ns = _preparar_ns({"fail_smoke": True, "smoke_msg": "AVISO_SIMULADO_SMOKE"})
        try:
            _silencioso(ns["prueba_humo"])
            fail("prueba_humo() con entrenamiento simulado fallido no lanzó RuntimeError")
        except RuntimeError as e:
            prob = []
            if "AVISO_SIMULADO_SMOKE" not in str(e):
                prob.append("el mensaje no incluye el aviso original")
            if ns["SMOKE_OK"] is not False:
                prob.append(f"SMOKE_OK quedó en {ns['SMOKE_OK']!r}, se esperaba False")
            (ok if not prob else fail)(
                "prueba_humo() con entrenamiento simulado fallido: RuntimeError, SMOKE_OK=False"
                + (f" — {'; '.join(prob)}" if prob else "")
            )

        # --- ejecutar_corrida(): camino feliz completo ---
        ns = _preparar_ns({"run_id_dry_run": "RUN_A", "run_id_formal": "RUN_A"})
        _silencioso(ns["preflight"])
        _silencioso(ns["prueba_humo"])
        run_id = _silencioso(ns["ejecutar_corrida"])
        prob = []
        if run_id != "RUN_A":
            prob.append(f"ejecutar_corrida() devolvió {run_id!r}, se esperaba 'RUN_A'")
        if ns["CORRIDA_VALIDADA"] is not False or ns["RUN_ID_VALIDADO"] is not None:
            prob.append("CORRIDA_VALIDADA/RUN_ID_VALIDADO no quedaron reseteados tras la corrida formal")
        (ok if not prob else fail)(
            "ejecutar_corrida() con preflight y prueba de humo superados: corre y "
            "resetea CORRIDA_VALIDADA/RUN_ID_VALIDADO" + (f" — {'; '.join(prob)}" if prob else "")
        )

        # --- T5.1: ejecutar_corrida() resetea un estado de validación previo ---
        # A diferencia del camino feliz de arriba (que parte de
        # CORRIDA_VALIDADA=False/RUN_ID_VALIDADO=None, los valores con los que
        # la celda constructora ya los inicializa), aquí se fuerzan a mano a
        # True/"RUN_ANTERIOR" después de construir el namespace, para
        # comprobar que ejecutar_corrida() los resetea aunque ya hubiera una
        # corrida distinta marcada como válida en la sesión — no solo que los
        # deja en su valor inicial de siempre.
        ns = _preparar_ns({"run_id_dry_run": "RUN_A", "run_id_formal": "RUN_A"})
        ns["CORRIDA_VALIDADA"] = True
        ns["RUN_ID_VALIDADO"] = "RUN_ANTERIOR"
        _silencioso(ns["preflight"])
        _silencioso(ns["prueba_humo"])
        _silencioso(ns["ejecutar_corrida"])
        prob = []
        if ns["CORRIDA_VALIDADA"] is not False:
            prob.append(f"CORRIDA_VALIDADA quedó en {ns['CORRIDA_VALIDADA']!r}, se esperaba False")
        if ns["RUN_ID_VALIDADO"] is not None:
            prob.append(f"RUN_ID_VALIDADO quedó en {ns['RUN_ID_VALIDADO']!r}, se esperaba None")
        (ok if not prob else fail)(
            "ejecutar_corrida() resetea CORRIDA_VALIDADA/RUN_ID_VALIDADO aunque partan de "
            "un estado previo válido (True/'RUN_ANTERIOR') de otra corrida"
            + (f" — {'; '.join(prob)}" if prob else "")
        )

        # --- T5.2: EJECUTAR_DIAGNOSTICO_ORDEN=False no debe llamar al runner ---
        mode_diag = {"run_id_dry_run": "RUN_A"}
        ns = _preparar_ns(mode_diag, overrides={"EJECUTAR_DIAGNOSTICO_ORDEN": False})
        llamadas_antes = len(mode_diag.get("calls", []))
        _silencioso(exec, compile(src_diagnostico, "<celda-diagnostico>", "exec"), ns)
        llamadas_despues = len(mode_diag.get("calls", []))
        (ok if llamadas_despues == llamadas_antes else fail)(
            "celda de diagnóstico con EJECUTAR_DIAGNOSTICO_ORDEN=False: no llama al runner "
            f"simulado (llamadas antes={llamadas_antes}, después={llamadas_despues})"
        )

        # --- T5.3: prueba_humo() dos veces en la misma sesión ---
        mode_humo: dict = {"run_id_dry_run": "RUN_A"}
        ns = _preparar_ns(mode_humo)
        resultado_1 = _silencioso(ns["prueba_humo"])
        resultado_2 = _silencioso(ns["prueba_humo"])
        llamadas_humo = [c for c in mode_humo.get("calls", []) if "--out" in c]
        prob = []
        if resultado_1 is not True or resultado_2 is not True:
            prob.append(f"resultados: primera={resultado_1!r}, segunda={resultado_2!r}, se esperaba True en ambas")
        if len(llamadas_humo) != 2:
            prob.append(f"se esperaban 2 llamadas al runner con --out, hubo {len(llamadas_humo)}")
        else:
            sin_overwrite = [i for i, argv in enumerate(llamadas_humo, start=1) if "--overwrite" not in argv]
            if sin_overwrite:
                prob.append(f"llamada(s) {sin_overwrite} sin --overwrite")
            carpetas = {argv[argv.index("--out") + 1] for argv in llamadas_humo}
            if len(carpetas) != 1:
                prob.append(f"las dos llamadas no usan la misma carpeta --out: {carpetas}")
        (ok if not prob else fail)(
            "prueba_humo() ejecutada dos veces en la misma sesión: dos éxitos, ambas con "
            "--overwrite y la misma carpeta temporal" + (f" — {'; '.join(prob)}" if prob else "")
        )

        # --- Regresión: prueba_humo() con START_FROM_EPOCH grande en la config formal ---
        # Bug reportado: construir_argv() no aceptaba un override de
        # start_from_epoch, así que prueba_humo() (EPOCHS=3) heredaba el
        # START_FROM_EPOCH de la config formal tal cual — con un warm-up
        # largo (p. ej. 150, como en un TAG "bce_warmupNNN") la prueba de
        # humo fallaba con "--start-from-epoch debe ser menor que --epochs"
        # sin haber entrenado nada, incluso con la configuración formal
        # correcta. prueba_humo() debe forzar start_from_epoch=0 en su propia
        # llamada, sin tocar START_FROM_EPOCH de la config formal.
        mode_warmup: dict = {"run_id_dry_run": "RUN_A"}
        ns = _preparar_ns(mode_warmup, overrides={"START_FROM_EPOCH": 150})
        try:
            resultado = _silencioso(ns["prueba_humo"])
            llamadas_smoke = [c for c in mode_warmup.get("calls", []) if "--out" in c]
            prob = []
            if resultado is not True:
                prob.append(f"prueba_humo() devolvió {resultado!r}, se esperaba True")
            if not llamadas_smoke:
                prob.append("no se registró ninguna llamada al runner con --out")
            else:
                argv = llamadas_smoke[0]
                valor_start = argv[argv.index("--start-from-epoch") + 1] if "--start-from-epoch" in argv else None
                if valor_start != "0":
                    prob.append(f"--start-from-epoch en la llamada de humo fue {valor_start!r}, se esperaba '0'")
            (ok if not prob else fail)(
                "prueba_humo() con START_FROM_EPOCH=150 en la config formal: fuerza "
                "--start-from-epoch 0 en su propia llamada y tiene éxito"
                + (f" — {'; '.join(prob)}" if prob else "")
            )
            if ns["CONFIG_NOTEBOOK"]["start_from_epoch"] != 150:
                fail(
                    "prueba_humo() con START_FROM_EPOCH=150: CONFIG_NOTEBOOK quedó "
                    f"alterado a {ns['CONFIG_NOTEBOOK']['start_from_epoch']!r}, se esperaba "
                    "que la config formal (150) no se tocara"
                )
        except RuntimeError as e:
            fail(
                "prueba_humo() con START_FROM_EPOCH=150 en la config formal: lanzó "
                f"RuntimeError en vez de tener éxito — {e}"
            )

        # --- ejecutar_corrida(): run_id no coincide con el de preflight() ---
        ns = _preparar_ns({"run_id_dry_run": "RUN_A", "run_id_formal": "RUN_B"})
        _silencioso(ns["preflight"])
        _silencioso(ns["prueba_humo"])
        try:
            _silencioso(ns["ejecutar_corrida"])
            fail("ejecutar_corrida() con run_id distinto al de preflight() no lanzó RuntimeError")
        except RuntimeError as e:
            (ok if "no coincide" in str(e) else fail)(
                "ejecutar_corrida() con run_id distinto al de preflight() lanza RuntimeError "
                "señalando el desajuste" + (f" — mensaje={e!r}" if "no coincide" not in str(e) else "")
            )

        # --- ejecutar_diagnostico_orden(): rechaza representación estática ---
        ns = _preparar_ns({"run_id_dry_run": "RUN_A"}, overrides={"REPRESENTACION": "static"})
        try:
            _silencioso(ns["ejecutar_diagnostico_orden"])
            fail("ejecutar_diagnostico_orden() con REPRESENTACION='static' no lanzó ValueError")
        except ValueError as e:
            (ok if "static" in str(e) else fail)(
                "ejecutar_diagnostico_orden() con REPRESENTACION='static' lanza ValueError "
                "mencionándola" + (f" — mensaje={e!r}" if "static" not in str(e) else "")
            )

        # --- ejecutar_diagnostico_orden(): rechaza brainnetcnn ---
        ns = _preparar_ns({"run_id_dry_run": "RUN_A"}, overrides={"MODELO": "brainnetcnn"})
        try:
            _silencioso(ns["ejecutar_diagnostico_orden"])
            fail("ejecutar_diagnostico_orden() con MODELO='brainnetcnn' no lanzó ValueError")
        except ValueError as e:
            (ok if "brainnetcnn" in str(e) else fail)(
                "ejecutar_diagnostico_orden() con MODELO='brainnetcnn' lanza ValueError "
                "mencionándolo" + (f" — mensaje={e!r}" if "brainnetcnn" not in str(e) else "")
            )

        # --- T6: construir_argv(), sin entrenar — inspecciona ARGV_CORRIDA ---
        # _preparar_ns() ya construye CONFIG_NOTEBOOK/ARGV_CORRIDA al ejecutar
        # la celda constructora con las variables de config_valida más los
        # overrides de cada caso, así que no hace falta llamar a
        # construir_argv() aparte: alcanza con inspeccionar ns["ARGV_CORRIDA"].
        # Para las combinaciones rechazadas, construir_argv() lanza el
        # ValueError durante ese mismo exec (en la línea que arma
        # ARGV_CORRIDA), así que _preparar_ns() es quien lo deja escapar.
        def _argv_case(nombre: str, overrides: dict, presentes: list, ausentes: list) -> None:
            try:
                ns_t6 = _preparar_ns({}, overrides=overrides)
            except ValueError as e:
                fail(f"T6 ({nombre}): construir_argv() lanzó ValueError inesperado: {e}")
                return
            argv = list(ns_t6["ARGV_CORRIDA"])
            prob = []
            faltan = [flag for flag in presentes if flag not in argv]
            if faltan:
                prob.append(f"faltan en argv: {faltan}")
            sobran = [flag for flag in ausentes if flag in argv]
            if sobran:
                prob.append(f"no deberían estar en argv: {sobran}")
            (ok if not prob else fail)(
                f"T6 ({nombre}): construir_argv() produce el argv esperado"
                + (f" — {'; '.join(prob)}; argv={argv}" if prob else "")
            )

        def _argv_rechazo(nombre: str, overrides: dict, fragmento_esperado: str) -> None:
            try:
                _preparar_ns({}, overrides=overrides)
                fail(f"T6 ({nombre}): construir_argv() no lanzó ValueError para una combinación incompatible")
            except ValueError as e:
                (ok if fragmento_esperado in str(e) else fail)(
                    f"T6 ({nombre}): construir_argv() lanza ValueError mencionando la causa esperada"
                    + (f" — mensaje={e!r}, se esperaba fragmento {fragmento_esperado!r}"
                       if fragmento_esperado not in str(e) else "")
                )

        _argv_case(
            "ventana en segundos (default)", {},
            presentes=["--window-seconds", "120", "--step-seconds", "12"],
            ausentes=["--window", "--overlap"],
        )
        _argv_case(
            "ventana en TR",
            {"WINDOW_TR": 60, "STEP_TR": 6, "WINDOW_SECONDS": None, "STEP_SECONDS": None, "OVERLAP": None},
            presentes=["--window", "60", "--step", "6"],
            ausentes=["--window-seconds", "--step-seconds", "--overlap"],
        )
        _argv_case(
            "OVERLAP en vez de STEP_SECONDS",
            {"STEP_SECONDS": None, "OVERLAP": 0.5},
            presentes=["--window-seconds", "120", "--overlap", "0.5"],
            ausentes=["--step-seconds"],
        )
        _argv_case(
            "ventana gaussiana con GAUSSIAN_SIGMA",
            {"WINDOW_SHAPE": "gaussian", "GAUSSIAN_SIGMA": 10},
            presentes=["--window-shape", "gaussian", "--gaussian-sigma", "10"],
            ausentes=[],
        )
        _argv_case(
            "permuted con REPRESENTATION_SEED",
            {"REPRESENTACION": "permuted", "REPRESENTATION_SEED": 7},
            presentes=["--representation", "permuted", "--representation-seed", "7"],
            ausentes=[],
        )
        _argv_case(
            "DETERMINISTIC/MIXED_PRECISION/CLASS_WEIGHT/CLIPNORM",
            {"DETERMINISTIC": True, "MIXED_PRECISION": True, "CLASS_WEIGHT": True, "CLIPNORM": 1.0},
            presentes=["--deterministic", "--mixed-precision", "--class-weight", "--clipnorm", "1.0"],
            ausentes=[],
        )
        _argv_case(
            "control anatómico: RANDOM_SUBSET/N_RANDOM_SETS/EXCLUDE_ROI_SET",
            {"RANDOM_SUBSET": 20, "N_RANDOM_SETS": 5, "EXCLUDE_ROI_SET": "18"},
            presentes=["--random-subset", "20", "--n-random-sets", "5", "--exclude-roi-set", "18"],
            ausentes=[],
        )

        _argv_rechazo(
            "WINDOW_TR y WINDOW_SECONDS a la vez",
            {"WINDOW_TR": 60, "STEP_TR": 6},
            "modos alternativos",
        )
        _argv_rechazo(
            "ni WINDOW_TR ni WINDOW_SECONDS",
            {"WINDOW_SECONDS": None, "STEP_SECONDS": None, "OVERLAP": None},
            "sin ventana",
        )
        _argv_rechazo(
            "STEP_SECONDS y OVERLAP a la vez",
            {"OVERLAP": 0.5},
            "alternativos",
        )
        _argv_rechazo(
            "GAUSSIAN_SIGMA sin WINDOW_SHAPE='gaussian'",
            {"GAUSSIAN_SIGMA": 10},
            "solo aplica con WINDOW_SHAPE",
        )
    finally:
        if original_run_experiment is not None:
            sys.modules["run_experiment"] = original_run_experiment
        else:
            sys.modules.pop("run_experiment", None)

    # --- exigir_corrida_validada() ---
    def _preparar_ns_gate(cv, run_id_validado, run_id, problemas_export):
        ns_gate = {
            "CORRIDA_VALIDADA": cv, "RUN_ID_VALIDADO": run_id_validado, "RUN_ID": run_id,
            "RUTA": Path("/tmp/no_existe_verify_setup"),
            "validate_run_artifacts": lambda ruta: list(problemas_export),
        }
        exec(compile(src_gate, "<celda-gate>", "exec"), ns_gate)
        return ns_gate

    ns_gate = _preparar_ns_gate(False, None, "RUN_A", [])
    try:
        _silencioso(ns_gate["exigir_corrida_validada"])
        fail("exigir_corrida_validada() con CORRIDA_VALIDADA=False no lanzó RuntimeError")
    except RuntimeError as e:
        (ok if "validada" in str(e) else fail)(
            "exigir_corrida_validada() con CORRIDA_VALIDADA=False lanza RuntimeError"
            + (f" — mensaje={e!r}" if "validada" not in str(e) else "")
        )

    ns_gate = _preparar_ns_gate(True, "RUN_A", "RUN_B", [])
    try:
        _silencioso(ns_gate["exigir_corrida_validada"])
        fail("exigir_corrida_validada() con RUN_ID_VALIDADO distinto de RUN_ID no lanzó RuntimeError")
    except RuntimeError as e:
        (ok if "validada" in str(e) else fail)(
            "exigir_corrida_validada() con RUN_ID_VALIDADO≠RUN_ID (validación vieja) lanza RuntimeError"
            + (f" — mensaje={e!r}" if "validada" not in str(e) else "")
        )

    ns_gate = _preparar_ns_gate(True, "RUN_A", "RUN_A", ["archivo corrupto tras validar"])
    try:
        _silencioso(ns_gate["exigir_corrida_validada"])
        fail("exigir_corrida_validada() con la corrida corrupta en disco no lanzó RuntimeError")
    except RuntimeError as e:
        prob = []
        if "cambió" not in str(e) and "dejó de ser válida" not in str(e):
            prob.append("el mensaje no indica que la corrida cambió/dejó de ser válida")
        if ns_gate["CORRIDA_VALIDADA"] is not False or ns_gate["RUN_ID_VALIDADO"] is not None:
            prob.append("CORRIDA_VALIDADA/RUN_ID_VALIDADO no quedaron reseteados")
        (ok if not prob else fail)(
            "exigir_corrida_validada() revalida contra disco y rechaza una corrida que "
            "cambió después de validarse, reseteando el estado"
            + (f" — {'; '.join(prob)}" if prob else "")
        )

    ns_gate = _preparar_ns_gate(True, "RUN_A", "RUN_A", [])
    try:
        _silencioso(ns_gate["exigir_corrida_validada"])
        ok("exigir_corrida_validada() con validación vigente y sin problemas en disco no lanza nada")
    except RuntimeError as e:
        fail(f"exigir_corrida_validada() con validación vigente lanzó RuntimeError inesperado: {e}")


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
    ambiguos con un root compartido). La comparación se hace sobre los
    `config.json` encontrados con `rglob` (no `root.iterdir()`), porque
    `run_experiment.py` escribe en `root/<roi_set>/<run_id>/`: diferenciar
    solo el primer nivel de `root` detectaría la subcarpeta de ROI (p. ej.
    `root/12`) como "nueva" la primera vez, y ninguna carpeta nueva en
    corridas posteriores con el mismo ROI. Devuelve la carpeta de la
    corrida, o None si algo falló.
    """
    import subprocess

    antes = {p for p in root.rglob("config.json")} if root.exists() else set()
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

    despues = {p for p in root.rglob("config.json")} if root.exists() else set()
    nuevas = sorted(despues - antes)
    if len(nuevas) != 1:
        fail(
            f"se esperaba exactamente 1 config.json nuevo en {root} tras "
            f"--early-stopping-monitor {monitor}, hubo {len(nuevas)}"
        )
        return None
    return nuevas[0].parent


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
        check_ensemble_analysis()
        check_aggregate_table_gate()
        check_notebook_state_machine()
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