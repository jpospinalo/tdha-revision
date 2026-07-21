#!/usr/bin/env python3
"""
Clasificación TDAH vs. control a partir de conectividad funcional dinámica.

Una corrida = un sitio + un subconjunto de ROIs + un enventanado + una arquitectura
+ una configuración de entrenamiento. Cada corrida escribe en su propia carpeta, cuyo
nombre se deriva de la configuración, de modo que varias personas pueden ejecutar en
paralelo y hacer push al mismo repositorio sin conflictos: nadie escribe nunca en un
archivo compartido.

Garantías
---------
Particiones
    Con la misma --seed y las mismas etiquetas, todas las configuraciones usan las
    mismas particiones. La comparación entre subconjuntos de ROIs o entre
    arquitecturas es pareada, lo que permite usar contrastes de medidas repetidas.

Selección de época sin fuga
    Dentro de cada pliegue se aparta una fracción del entrenamiento para el early
    stopping. El pliegue de validación externo solo se usa en la evaluación final,
    nunca para tomar decisiones.

Trazabilidad
    El config.json guarda el hash de las señales BOLD, los parámetros de enventanado,
    el commit de git, si el árbol estaba limpio, el usuario y las versiones del
    entorno. Los tensores de conectividad se derivan de las señales en cada corrida,
    así que no pueden quedar desincronizados de los parámetros que los generaron.

Límite conocido
    Las métricas NO son idénticas entre máquinas: los kernels de cuDNN para redes
    recurrentes no son deterministas. Lo garantizado son las particiones y el
    protocolo, no los decimales. Ver --deterministic.

Precisión numérica
    Todas las corridas usan float32. No hay opción de precisión mixta: activarla en
    unas corridas y no en otras produciría una tabla final con configuraciones
    numéricas distintas, y el ahorro de tiempo no compensa ese riesgo. Si en algún
    momento se decide adoptarla, debe ser para el conjunto completo de experimentos
    y relanzando todo.

Ejemplos
--------
    python run_experiment.py --site NYU --roi-set 12
    python run_experiment.py --site NYU --roi-set 12 --model gru --model-arg units=64
    python run_experiment.py --site NYU --roi-set 12 --window 40 --step 2
    python run_experiment.py --site Peking --roi-set 18 --class-weight
    python run_experiment.py --site NYU --roi-set 116 --random-subset 12 \\
                             --exclude-roi-set 12 --n-random-sets 20
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

import data as tdha_data
import kerasmodels

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "runs"


# --------------------------------------------------------------------------- #
# Entorno
# --------------------------------------------------------------------------- #

def git_info():
    def run(*a):
        try:
            return subprocess.check_output(a, cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except Exception:
            return None
    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD") or "desconocido",
        "clean": (status == "") if status is not None else None,
        "user": run("git", "config", "user.name") or os.environ.get("USER", "desconocido"),
    }


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import tensorflow as tf
        import keras
        info["tensorflow"] = tf.__version__
        info["keras"] = keras.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu"] = [tf.config.experimental.get_device_details(g).get("device_name", "?")
                       for g in gpus] or "sin GPU"
    except Exception as e:
        info["tensorflow"] = f"no disponible ({type(e).__name__})"
    return info


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def arch_size(arch):
    """Tamaño principal de la arquitectura, para el nombre de la corrida.

    Toma el primer hiperparámetro numérico en el orden de la firma de la función:
    ``units`` en lstm y gru, ``filters`` en cnn1d, ``d_model`` en transformer.
    """
    for v in arch.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(int(v))
    return "0"


def parse_model_args(pairs):
    """['units=128', 'dropout=0.2'] -> {'units': 128, 'dropout': 0.2}"""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"ERROR: --model-arg espera clave=valor, se recibió '{p}'")
        k, v = p.split("=", 1)
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = {"true": True, "false": False}.get(v.lower(), v)
    return out


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #

def evaluate(model, X, y):
    """Métricas de un modelo sobre un conjunto, con una sola pasada hacia adelante.

    Sustituye a ``model.evaluate`` y ``model.predict``. Hacerlo así tiene tres
    ventajas: evita construir y trazar dos funciones de TensorFlow por pliegue
    —origen de los avisos de *retracing*, porque los pliegues tienen 17 o 18
    sujetos y cada forma distinta obliga a retrazar—, es más rápido con muestras
    de este tamaño, y el AUC pasa a ser exacto en lugar de la aproximación por
    200 umbrales que usa la métrica de Keras.

    Las demás métricas son idénticas a las de Keras: umbral 0,5 y las mismas
    definiciones.
    """
    from sklearn.metrics import roc_auc_score

    p = np.asarray(model(X, training=False)).ravel().astype("float64")
    y = np.asarray(y).ravel()
    eps = 1e-7
    pc = np.clip(p, eps, 1 - eps)
    loss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))

    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    return {
        "loss": loss,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "true_positives": tp, "true_negatives": tn,
        "false_positives": fp, "false_negatives": fn,
    }, p


def compile_model(model, args):
    """Compila con UNA sola métrica.

    Las métricas que se reportan las calcula ``evaluate`` al final de cada pliegue,
    no Keras durante el entrenamiento. Compilar con las ocho obligaba a actualizarlas
    en cada lote de cada época —el AUC es especialmente caro, con 200 umbrales— para
    después descartarlas. Se conserva ``accuracy`` porque alimenta el historial
    opcional; el early stopping solo usa ``val_loss``.
    """
    import keras
    metrics = [keras.metrics.BinaryAccuracy(name="accuracy")]
    opt = {"learning_rate": args.lr}
    if args.clipnorm:
        opt["clipnorm"] = args.clipnorm
    model.compile(optimizer=keras.optimizers.Adam(**opt),
                  loss="binary_crossentropy", metrics=metrics)
    return model


def run_config(Xf, y, args, outdir, subset_id=None):
    """Validación cruzada completa sobre una matriz de secuencias ya construida."""
    import keras
    from keras.callbacks import EarlyStopping

    n_subj, n_win, n_feat = Xf.shape
    print(f"  entrada: {n_subj} sujetos · {n_win} ventanas · {n_feat} características")

    outer = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats,
                                    random_state=args.seed)

    class_weight = None
    if args.class_weight:
        counts = np.bincount(y)
        class_weight = {i: len(y) / (len(counts) * c) for i, c in enumerate(counts)}
        print(f"  pesos de clase: { {k: round(v, 3) for k, v in class_weight.items()} }")

    rows_train, rows_val, hist_rows, pred_rows, fold_rows = [], [], [], [], []
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(outer.split(Xf, y)):
        # Se construye un modelo nuevo por pliegue: sin limpiar, Keras acumula
        # grafos y funciones trazadas de los 50 modelos anteriores. El orden importa:
        # clear_session ANTES de fijar la semilla, para que la inicialización de
        # pesos siga siendo la misma y los resultados no cambien.
        keras.backend.clear_session()
        gc.collect()
        keras.utils.set_random_seed(args.seed * 1000 + fold)
        repeat = fold // args.n_splits + 1

        # Partición interna, SOLO sobre entrenamiento, para decidir la época.
        inner = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_val_frac,
                                       random_state=args.seed + fold)
        # StratifiedShuffleSplit solo necesita el número de muestras y las etiquetas:
        # pasarle Xf[tr_idx] materializaría una copia de ~220 MB con 116 ROIs, en cada
        # uno de los 50 pliegues, para nada.
        fit_rel, sel_rel = next(inner.split(np.zeros(len(tr_idx)), y[tr_idx]))
        fit_idx, sel_idx = tr_idx[fit_rel], tr_idx[sel_rel]
        X_fit, X_sel = Xf[fit_idx], Xf[sel_idx]

        model = compile_model(
            kerasmodels.build(args.model, n_win, n_feat, **args._model_kwargs), args)

        # restore_best_weights deja el modelo en su mejor época según la partición
        # INTERNA. No se usa ModelCheckpoint: sería el mismo criterio con escritura de
        # disco y nombres que pueden colisionar entre subconjuntos aleatorios.
        history = model.fit(
            X_fit, y[fit_idx],
            validation_data=(X_sel, y[sel_idx]),
            epochs=args.epochs, batch_size=args.batch_size,
            class_weight=class_weight, verbose=0,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min",
                                     patience=args.patience, min_delta=1e-5,
                                     start_from_epoch=args.start_from_epoch,
                                     restore_best_weights=True)],
        )
        n_ep = len(history.history["loss"])
        best_ep = int(np.argmin(history.history["val_loss"])) + 1

        # La validación externa se toca aquí por primera vez.
        m_tr, _ = evaluate(model, Xf[tr_idx], y[tr_idx])
        m_va, probs = evaluate(model, Xf[va_idx], y[va_idx])
        meta = {"fold": fold + 1, "repeat": repeat, "n_epochs": n_ep, "best_epoch": best_ep}
        rows_train.append({**meta, **m_tr})
        rows_val.append({**meta, **m_va})

        for ep, loss in enumerate(history.history["loss"]):
            hist_rows.append({"fold": fold + 1, "repeat": repeat, "epoch": ep + 1,
                              "loss": loss,
                              "inner_val_loss": history.history["val_loss"][ep],
                              "accuracy": history.history["accuracy"][ep],
                              "inner_val_accuracy": history.history["val_accuracy"][ep]})

        for s, p in zip(va_idx, probs):
            pred_rows.append({"fold": fold + 1, "repeat": repeat, "subject": int(s),
                              "y_true": int(y[s]), "y_prob": float(p)})
        for name, idxs in [("fit", fit_idx), ("inner_val", sel_idx), ("outer_val", va_idx)]:
            for s in idxs:
                fold_rows.append({"fold": fold + 1, "subject": int(s), "split": name})

        if args.verbose:
            print(f"    pliegue {fold + 1:3d}/{args.n_splits * args.n_repeats}  "
                  f"train acc={m_tr['accuracy']:.4f}  val acc={m_va['accuracy']:.4f}  "
                  f"(época {best_ep}/{n_ep})", flush=True)

        del model, history, X_fit, X_sel
        gc.collect()

    sfx = "" if subset_id is None else f"_set{subset_id:02d}"
    pd.DataFrame(rows_train).to_csv(outdir / f"metrics_train{sfx}.csv", index=False)
    pd.DataFrame(rows_val).to_csv(outdir / f"metrics_val{sfx}.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(outdir / f"predictions_val{sfx}.csv", index=False)
    # history y folds no se guardan por defecto. history sostiene las curvas de
    # convergencia (comentarios 4 de R1 y 11 de R2) y folds es la evidencia de que
    # la partición es por sujeto (comentario 6 de R2): actívelos cuando haga falta
    # producir esas figuras o responder a esa auditoría.
    if args.save_history:
        pd.DataFrame(hist_rows).to_csv(outdir / f"history{sfx}.csv", index=False)
    if args.save_folds:
        pd.DataFrame(fold_rows).to_csv(outdir / f"folds{sfx}.csv", index=False)

    tr, va = pd.DataFrame(rows_train), pd.DataFrame(rows_val)
    print(f"  train acc {tr.accuracy.mean() * 100:.2f} ± {tr.accuracy.std() * 100:.2f}  |  "
          f"val acc {va.accuracy.mean() * 100:.2f} ± {va.accuracy.std() * 100:.2f}  |  "
          f"{time.time() - t0:.0f} s")
    return {"train_acc": float(tr.accuracy.mean()), "val_acc": float(va.accuracy.mean())}


# --------------------------------------------------------------------------- #

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("datos")
    g.add_argument("--site", default="NYU", choices=tdha_data.SITES)
    g.add_argument("--roi-set", default="12",
                   help="subconjunto definido en data/atlas/roi_sets.json: 12, 18, 39, 116")
    g.add_argument("--window", type=int, default=70, help="longitud de ventana, en TR")
    g.add_argument("--step", type=int, default=2, help="desplazamiento entre ventanas, en TR")
    g.add_argument("--out", default=str(DEFAULT_OUT_DIR))

    g = p.add_argument_group("arquitectura")
    g.add_argument("--model", default="lstm",
                   help=f"una de: {', '.join(kerasmodels.available())}")
    g.add_argument("--model-arg", nargs="*", metavar="CLAVE=VALOR",
                   help="hiperparámetros de la arquitectura, p. ej. units=128 dropout=0.2")

    g = p.add_argument_group("entrenamiento")
    g.add_argument("--seed", type=int, default=42,
                   help="fija particiones e inicialización. Usar el MISMO valor en todas "
                        "las configuraciones que se vayan a comparar")
    g.add_argument("--n-splits", type=int, default=10)
    g.add_argument("--n-repeats", type=int, default=5)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--epochs", type=int, default=150)
    g.add_argument("--patience", type=int, default=100)
    g.add_argument("--clipnorm", type=float, default=None)
    g.add_argument("--start-from-epoch", type=int, default=0,
                   help="épocas mínimas antes de empezar a vigilar la parada. Con 0 "
                        "(por defecto) el mínimo de la pérdida interna puede caer en la "
                        "época 1 por ruido, y restore_best_weights devolvería un modelo "
                        "sin entrenar: en el piloto eso ocurrió en 18 de 50 pliegues")
    g.add_argument("--inner-val-frac", type=float, default=0.15,
                   help="fracción del entrenamiento reservada para elegir la época")
    g.add_argument("--class-weight", action="store_true",
                   help="pondera clases por frecuencia; útil en sitios desbalanceados")

    g = p.add_argument_group("control anatómico")
    g.add_argument("--random-subset", type=int, default=None,
                   help="muestrea N ROIs al azar, para separar el efecto de la selección "
                        "anatómica del de la reducción de dimensionalidad")
    g.add_argument("--n-random-sets", type=int, default=20)
    g.add_argument("--exclude-roi-set", default=None,
                   help="excluye del muestreo los ROIs de este subconjunto, p. ej. 12")

    g = p.add_argument_group("ejecución")
    g.add_argument("--deterministic", action="store_true",
                   help="fuerza operaciones deterministas: dos máquinas dan cifras "
                        "idénticas, pero las RNN pierden el camino rápido de cuDNN y la "
                        "corrida puede tardar varias veces más")
    g.add_argument("--tag", default=None, help="sufijo opcional para el nombre de la corrida")
    g.add_argument("--save-history", action="store_true",
                   help="guarda history.csv: pérdida y accuracy por época y pliegue. "
                        "Necesario para las curvas de convergencia que piden los revisores")
    g.add_argument("--save-folds", action="store_true",
                   help="guarda folds.csv: los sujetos de cada partición. Es la evidencia "
                        "auditable de que la división es por sujeto y no por ventana")
    g.add_argument("--dry-run", action="store_true", help="valida sin entrenar")
    g.add_argument("--list-models", action="store_true")
    g.add_argument("--list-roi-sets", action="store_true")
    g.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.list_models:
        for name in kerasmodels.available():
            print(f"  {name:14s} {kerasmodels.defaults(name)}")
        return
    if args.list_roi_sets:
        for k, v in sorted(tdha_data.load_roi_sets().items(), key=lambda x: int(x[0])):
            print(f"  {k:>4s}  {v['n']:3d} ROIs   {v['description']}")
        return

    # Validar la arquitectura ANTES de cargar datos: un nombre mal escrito debe
    # fallar de inmediato, y --dry-run también debe detectarlo.
    if args.model not in kerasmodels.REGISTRY:
        raise SystemExit(f"ERROR: arquitectura '{args.model}' desconocida. "
                         f"Disponibles: {', '.join(kerasmodels.available())}")
    args._model_kwargs = parse_model_args(args.model_arg)
    kerasmodels.validate_args(args.model, args._model_kwargs)

    if args.deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()
        except Exception as e:
            print(f"  AVISO: no se pudo activar determinismo: {e}")

    bold_path = tdha_data.BOLD_DIR / f"{args.site}.joblib"
    b = tdha_data.load_bold(args.site)
    y = b["labels"]
    idx = tdha_data.roi_indices(args.roi_set)
    n_win = tdha_data.n_windows(b["bold"].shape[2], args.window, args.step)

    # La identidad de la corrida se deriva de todo lo que afecta al resultado. Dos
    # personas con la misma configuración obtienen el mismo nombre de carpeta, así que
    # la duplicación se detecta en vez de producir archivos paralelos.
    ident = {
        "site": args.site, "roi_set": args.roi_set,
        "window": args.window, "step": args.step,
        "model": args.model,
        "arch": {**kerasmodels.defaults(args.model), **args._model_kwargs},
        "seed": args.seed, "n_splits": args.n_splits, "n_repeats": args.n_repeats,
        "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
        "patience": args.patience, "clipnorm": args.clipnorm,
        "inner_val_frac": args.inner_val_frac, "class_weight": args.class_weight,
        "start_from_epoch": args.start_from_epoch,
        "random_subset": args.random_subset, "n_random_sets": args.n_random_sets,
        "exclude_roi_set": args.exclude_roi_set, "deterministic": args.deterministic,
        "bold_hash": file_hash(bold_path),
    }
    cfg_hash = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:8]
    # NYU_12_rois_win_70_step_2_lstm_128_ID_2136273e
    parts = [args.site, str(args.roi_set), "rois",
             "win", str(args.window), "step", str(args.step),
             args.model, arch_size(ident["arch"])]
    if args.random_subset:
        parts.append(f"rand{args.random_subset}")
    if args.tag:
        parts.append(args.tag)
    run_id = "_".join(parts) + f"_ID_{cfg_hash}"

    git = git_info()
    cfg = {
        "run_id": run_id, "config_hash": cfg_hash,
        "n_subjects": int(b["bold"].shape[0]), "n_timepoints": int(b["bold"].shape[2]),
        "n_windows": int(n_win), "n_rois": int(len(idx)),
        "class_balance": {int(k): int(v)
                          for k, v in zip(*np.unique(y, return_counts=True))},
        **ident, "git": git, "env": env_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Los argumentos reales, no sys.argv: al invocar main(argv) desde un
        # notebook, sys.argv es el del kernel de Jupyter y no dice nada.
        "command": "run_experiment.py " + " ".join(
            argv if argv is not None else sys.argv[1:]),
    }

    outdir = Path(args.out) / run_id
    # El identificador se imprime ANTES de cualquier salida temprana: si la corrida ya
    # existe, quien la invocó sigue necesitando el identificador para localizar los
    # resultados y publicarlos.
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"\n  corrida: {run_id}")

    # Repetir una configuración REEMPLAZA la anterior: no se conservan dos versiones
    # de lo mismo. Se limpia la carpeta para que no queden archivos sueltos de una
    # corrida previa (por ejemplo, subconjuntos aleatorios de un barrido más largo).
    if outdir.exists() and not args.dry_run:
        previos = sorted(outdir.glob("*.csv")) + sorted(outdir.glob("config.json"))
        if previos:
            print(f"\n  Se reemplaza una corrida anterior de esta misma configuración "
                  f"({len(previos)} archivos).\n")
            for f in previos:
                f.unlink()

    if git["clean"] is False:
        print("\n  AVISO: el árbol de git tiene cambios sin confirmar. Esta corrida no "
              "será reproducible por otras personas hasta que se haga commit.\n")
    n_val = len(y) // args.n_splits
    n_sel = int((len(y) - n_val) * args.inner_val_frac)
    if n_val < 10 or n_sel < 12:
        print(f"\n  AVISO: pliegues pequeños — validación externa ≈ {n_val} sujetos, "
              f"selección de época ≈ {n_sel}. Las métricas por pliegue serán muy "
              f"inestables. Considere reducir --n-splits.\n")
    if n_win < 10:
        print(f"\n  AVISO: solo {n_win} ventanas por sujeto. Con secuencias tan cortas "
              f"apenas hay dinámica temporal que modelar. OHSU tiene 74 TR por sujeto, "
              f"así que una ventana de 70 TR deja 3 ventanas casi idénticas.\n")

    if args.dry_run:
        splits = list(RepeatedStratifiedKFold(
            n_splits=args.n_splits, n_repeats=args.n_repeats,
            random_state=args.seed).split(np.zeros((len(y), 1)), y))
        h = hashlib.sha256(b"".join(v.tobytes() for _, v in splits)).hexdigest()[:12]
        print(f"\ndry-run correcto: {len(splits)} particiones, huella {h}")
        print("Dos corridas son comparables si coinciden su sitio, su semilla y esta huella.")
        return run_id

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    if args.random_subset:
        pool = np.arange(b["bold"].shape[1])
        if args.exclude_roi_set:
            excl = tdha_data.roi_indices(args.exclude_roi_set)
            pool = np.setdiff1d(pool, excl)
            print(f"  muestreando de {len(pool)} ROIs (excluidos los {len(excl)} del "
                  f"subconjunto '{args.exclude_roi_set}')")
        if len(pool) < args.random_subset:
            raise SystemExit("ERROR: quedan menos ROIs disponibles que los solicitados")
        rng = np.random.default_rng(args.seed)
        summary = []
        for k in range(args.n_random_sets):
            sub = np.sort(rng.choice(pool, size=args.random_subset, replace=False))
            print(f"\nsubconjunto {k + 1}/{args.n_random_sets}: {sub.tolist()}", flush=True)
            Xf = tdha_data.build_flat_sequences(b["bold"], sub, args.window, args.step)
            summary.append({"set": k + 1, "rois": sub.tolist(),
                            **run_config(Xf, y, args, outdir, k + 1)})
        pd.DataFrame(summary).to_csv(outdir / "random_subsets_summary.csv", index=False)
        accs = [s["val_acc"] for s in summary]
        print(f"\n{len(accs)} subconjuntos aleatorios de {args.random_subset} ROIs: "
              f"val acc media {np.mean(accs) * 100:.2f}, "
              f"rango [{min(accs) * 100:.2f}, {max(accs) * 100:.2f}]")
        print("Compare esta distribución con la del subconjunto anatómico: si el "
              "anatómico no la supera, la ventaja es de dimensionalidad, no de anatomía.")
    else:
        print("  construyendo secuencias de conectividad…", flush=True)
        Xf = tdha_data.build_sequences_cached(
            args.site, b["bold"], idx, args.window, args.step, args.roi_set)
        run_config(Xf, y, args, outdir)

    print(f"\nResultados en {outdir}")
    return run_id


if __name__ == "__main__":
    main()