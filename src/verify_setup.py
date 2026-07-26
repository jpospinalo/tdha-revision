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
    """Escribe los 5 artefactos mínimos (un pliegue, una fila) de una corrida
    de esquema 4 completa y válida, sin entrenar nada — usado por
    check_schema4_artifact_validation() para mutar copias y probar que
    _validate_schema4_artifacts()/collect() rechazan lo que deben rechazar.
    """
    import pandas as pd

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "run_id": run_dir.name, "config_schema_version": 4, "site": "NYU",
        "early_stopping_monitor": "val_loss", "early_stopping_ab_hash": "fixture0000000",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    common = {
        "fold": [1], "repeat": [1], "n_epochs": [3], "best_epoch": [2],
        "early_stopping_monitor": ["val_loss"],
        "best_monitor_value": [0.5], "restored_monitor_value": [0.5],
        "accuracy": [0.7], "loss": [0.5],
    }
    pd.DataFrame(common).to_csv(run_dir / "metrics_train.csv", index=False)
    pd.DataFrame(common).to_csv(run_dir / "metrics_val.csv", index=False)
    pd.DataFrame({
        "fold": [1], "repeat": [1], "epoch": [1],
        "loss": [0.5], "inner_val_loss": [0.5], "bce": [0.4], "inner_val_bce": [0.4],
    }).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame({
        "fold": [1], "repeat": [1], "subject_id": ["s1"], "y_true": [1], "y_prob": [0.6],
    }).to_csv(run_dir / "predictions_val.csv", index=False)
    pd.DataFrame({
        "fold": [1], "repeat": [1], "subject_id": ["s1"], "split": ["fit"],
    }).to_csv(run_dir / "folds.csv", index=False)


def check_schema4_artifact_validation():
    """Regresiones de _validate_schema4_artifacts()/collect() con fixtures
    CSV escritos a mano, sin entrenar ni importar TensorFlow. Cubre los dos
    defectos de la corrección v11 (ver docs/validation.md): (1)
    predictions_val.csv/folds.csv no tenían ninguna columna obligatoria, así
    que podían perder subject_id/split/epoch sin que nada lo detectara; (2)
    la comprobación de finitud usaba frame[present].stack(), que descarta los
    NaN por defecto (dropna=True) antes de que np.isfinite() los viera, así
    que un best_monitor_value=NaN pasaba inadvertido.
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
    problems = C._validate_schema4_artifacts(run_a, "")
    (ok if not problems else fail)(
        "caso A (válido): _validate_schema4_artifacts() devuelve []"
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

    problems = C._validate_schema4_artifacts(run_b, "")
    texto = " | ".join(problems)
    detectadas = sum(1 for _, column in drops if column in texto)
    (ok if detectadas == len(drops) else fail)(
        f"caso B: _validate_schema4_artifacts() informa las {len(drops)} columnas ausentes"
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

        problems = C._validate_schema4_artifacts(run_c, "")
        base_name = filename.replace(".csv", "")
        detectado = any(base_name in p and column in p for p in problems)
        (ok if detectado else fail)(
            f"caso C ({filename}/{column}=NaN): _validate_schema4_artifacts() señala "
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

    shutil.rmtree(root)


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
    """Audita una corrida (config.json/history.csv/metrics_val.csv/predictions_val.csv/
    folds.csv) SIN volver a asumir que la época correcta es el mínimo global de la
    serie monitoreada — esa era la comprobación circular que este chequeo reemplaza
    (ver docs/methodology.md, 'Early-stopping monitor'). Se limita a verificar que
    los metadatos son internamente consistentes y están respaldados por la
    reevaluación no circular (`restored_monitor_value`). Devuelve el config.json
    leído, o None si algo esencial faltó.
    """
    import numpy as np
    import pandas as pd

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

    try:
        hist = pd.read_csv(run_dir / "history.csv")
        train = pd.read_csv(run_dir / "metrics_train.csv")
        val = pd.read_csv(run_dir / "metrics_val.csv")
        pred = pd.read_csv(run_dir / "predictions_val.csv")
        folds = pd.read_csv(run_dir / "folds.csv")
    except Exception as e:
        fail(f"{monitor}: no se pudieron leer los artefactos: {type(e).__name__}: {e}")
        return cfg

    prob = [c for c in ("early_stopping_monitor", "best_monitor_value", "restored_monitor_value")
            if c not in train.columns]
    if len(train) != len(val):
        prob.append(f"metrics_train.csv tiene {len(train)} filas, metrics_val.csv tiene {len(val)}")
    (fail if prob else ok)(
        f"{monitor}: metrics_train.csv presente, mismo número de filas que "
        "metrics_val.csv y con metadatos de esquema 4" + (f" — {prob}" if prob else "")
    )

    prob = [c for c in ("bce", "inner_val_bce", "loss", "inner_val_loss") if c not in hist.columns]
    if hist.empty:
        prob.append("history.csv está vacío")
    (fail if prob else ok)(
        f"{monitor}: history.csv no vacío, con bce/inner_val_bce/loss/inner_val_loss"
        + (f" — {prob}" if prob else "")
    )

    numeric_hist = hist.select_dtypes(include=[np.number]).to_numpy()
    if numeric_hist.size == 0 or not np.isfinite(numeric_hist).all():
        fail(f"{monitor}: history.csv contiene valores no finitos")
    else:
        ok(f"{monitor}: history.csv sin NaN/inf")

    prob = [c for c in ("early_stopping_monitor", "best_monitor_value", "restored_monitor_value")
            if c not in val.columns]
    (fail if prob else ok)(
        f"{monitor}: metrics_val.csv contiene early_stopping_monitor/best_monitor_value/"
        "restored_monitor_value" + (f" — faltan {prob}" if prob else "")
    )

    # Consistencia interna: best_epoch dentro de rango, valor registrado en
    # history.csv coincide con best_monitor_value, y restored_monitor_value (la
    # reevaluación posterior a fit(), no derivada de best_epoch) también coincide
    # — esta última es la prueba no circular de qué pesos quedaron restaurados.
    prob = []
    inner_col = "inner_val_loss" if monitor == "val_loss" else "inner_val_bce"
    for _, row in val.iterrows():
        fold_v, repeat_v = row["fold"], row["repeat"]
        fold_hist = hist[(hist["fold"] == fold_v) & (hist["repeat"] == repeat_v)].sort_values("epoch")
        if fold_hist.empty:
            prob.append(f"f{fold_v}r{repeat_v}: sin filas en history.csv")
            continue
        n_epochs_fold = int(row["n_epochs"])
        best_epoch = int(row["best_epoch"])
        if not (1 <= best_epoch <= n_epochs_fold):
            prob.append(f"f{fold_v}r{repeat_v}: best_epoch={best_epoch} fuera de [1,{n_epochs_fold}]")
            continue
        recorded = fold_hist.loc[fold_hist["epoch"] == best_epoch, inner_col]
        if recorded.empty:
            prob.append(f"f{fold_v}r{repeat_v}: no hay fila de history.csv para epoch={best_epoch}")
            continue
        if abs(float(recorded.iloc[0]) - float(row["best_monitor_value"])) > 1e-6:
            prob.append(
                f"f{fold_v}r{repeat_v}: history[{inner_col}][{best_epoch}]={float(recorded.iloc[0])} "
                f"!= best_monitor_value={row['best_monitor_value']}"
            )
        if not np.isfinite(row["restored_monitor_value"]):
            prob.append(f"f{fold_v}r{repeat_v}: restored_monitor_value no finito")
        elif abs(float(row["restored_monitor_value"]) - float(row["best_monitor_value"])) > 1e-4:
            prob.append(
                f"f{fold_v}r{repeat_v}: restored_monitor_value={row['restored_monitor_value']} "
                f"se aleja de best_monitor_value={row['best_monitor_value']} (pesos no consistentes)"
            )
    (fail if prob else ok)(
        f"{monitor}: best_epoch en rango, y best_monitor_value/restored_monitor_value consistentes"
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
                f"repetición {repeat}: {group['subject_id'].nunique()} sujetos, "
                f"se esperaban {n_subjects}"
            )
    (fail if prob else ok)(
        f"{monitor}: validación externa cubre cada sujeto exactamente una vez por repetición"
        + (f" — {'; '.join(prob)}" if prob else "")
    )

    # Sin sujetos compartidos entre fit/inner_val/outer_val dentro de un mismo pliegue.
    prob = []
    for (fold_v, repeat_v), g in folds.groupby(["fold", "repeat"]):
        sets = {s: set(g.loc[g["split"] == s, "subject_id"]) for s in ("fit", "inner_val", "outer_val")}
        if sets["fit"] & sets["inner_val"]:
            prob.append(f"f{fold_v}r{repeat_v}: fit∩inner_val no vacío")
        if sets["fit"] & sets["outer_val"]:
            prob.append(f"f{fold_v}r{repeat_v}: fit∩outer_val no vacío")
        if sets["inner_val"] & sets["outer_val"]:
            prob.append(f"f{fold_v}r{repeat_v}: inner_val∩outer_val no vacío")
    (fail if prob else ok)(
        f"{monitor}: sin sujetos compartidos entre fit/inner_val/outer_val"
        + (f" — {'; '.join(prob)}" if prob else "")
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