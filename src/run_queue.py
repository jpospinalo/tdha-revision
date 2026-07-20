#!/usr/bin/env python3
"""
Ejecuta varias corridas encadenadas en un solo proceso.

Por qué existe
--------------
Lanzar ``run_experiment.py`` N veces desde la consola paga N veces el arranque de
TensorFlow (entre 10 y 20 s), la lectura de las señales BOLD y la construcción de
las secuencias de conectividad. Este ejecutor lo hace una sola vez y reutiliza:

- las señales BOLD del sitio, mientras la cola siga en el mismo sitio;
- las secuencias construidas, mientras no cambien el sitio, el subconjunto de ROIs
  ni el enventanado.

Con 116 ROIs, construir las secuencias cuesta unos 8 s. Encadenar cuatro
arquitecturas sobre la misma configuración de datos las construye una vez en lugar
de cuatro.

**Ordene la cola agrupando por configuración de datos.** La caché guarda una sola
entrada, porque el tensor de 116 ROIs ocupa cerca de 500 MB y conservar varios
agotaría la memoria de Colab. Una cola ordenada aprovecha la caché; una cola
alternada la invalida en cada paso.

Uso
---
Producto cartesiano desde la línea de comandos::

    python run_queue.py --site NYU --roi-set 12 18 39 116 --model lstm
    python run_queue.py --site NYU --roi-set 12 --model lstm gru cnn1d transformer
    python run_queue.py --site NYU --roi-set 12 --window 20 40 70 --model lstm

Cola explícita desde archivo, una línea por corrida con los argumentos de
``run_experiment.py``. Las líneas vacías y las que empiezan por ``#`` se ignoran::

    # cola.txt
    --site NYU --roi-set 12 --model lstm
    --site NYU --roi-set 12 --model gru
    --site Peking --roi-set 18 --model lstm --class-weight

    python run_queue.py --file cola.txt

Comportamiento ante fallos
--------------------------
Una corrida que falle no detiene la cola: se registra y se pasa a la siguiente. Al
final se imprime un resumen. Las corridas ya ejecutadas se saltan sin recalcular,
porque ``run_experiment.py`` detecta la configuración repetida.
"""

import argparse
import itertools
import shlex
import sys
import time
import traceback

import run_experiment


def desde_archivo(path):
    colas = []
    with open(path, encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if linea and not linea.startswith("#"):
                colas.append(shlex.split(linea))
    return colas


def producto(args):
    """Producto cartesiano de las opciones repetibles, en orden de caché."""
    combos = itertools.product(args.site, args.roi_set, args.window, args.step,
                               args.model)
    colas = []
    for site, rs, w, st, model in combos:
        argv = ["--site", site, "--roi-set", str(rs), "--window", str(w),
                "--step", str(st), "--model", model, "--seed", str(args.seed)]
        if args.n_splits:
            argv += ["--n-splits", str(args.n_splits)]
        if args.n_repeats:
            argv += ["--n-repeats", str(args.n_repeats)]
        argv += args.extra
        colas.append(argv)
    return colas


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", help="archivo con una corrida por línea")
    p.add_argument("--site", nargs="*", default=["NYU"])
    p.add_argument("--roi-set", nargs="*", default=["12"])
    p.add_argument("--window", nargs="*", type=int, default=[70])
    p.add_argument("--step", nargs="*", type=int, default=[2])
    p.add_argument("--model", nargs="*", default=["lstm"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int)
    p.add_argument("--n-repeats", type=int)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="argumentos adicionales para todas las corridas; debe ir al final")
    p.add_argument("--dry-run", action="store_true",
                   help="muestra la cola sin ejecutarla")
    args = p.parse_args()

    colas = desde_archivo(args.file) if args.file else producto(args)

    print(f"Cola de {len(colas)} corridas\n" + "=" * 70)
    for i, argv in enumerate(colas, 1):
        print(f"  {i:3d}. {' '.join(argv)}")
    print("=" * 70)
    if args.dry_run:
        return

    resultados = []
    t_total = time.time()
    for i, argv in enumerate(colas, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(colas)}] {' '.join(argv)}\n{'=' * 70}", flush=True)
        t = time.time()
        try:
            run_experiment.main(argv)
            estado = "correcta"
        except SystemExit as e:
            # run_experiment usa SystemExit tanto para errores como para avisar de
            # configuraciones ya ejecutadas, que no son un fallo.
            msg = str(e)
            estado = "ya existía" if "YA_SE_EJECUTO" in msg else f"detenida: {msg[:80]}"
            print(msg)
        except Exception:
            estado = "FALLÓ"
            traceback.print_exc()
        resultados.append((argv, estado, time.time() - t))

    print(f"\n{'=' * 70}\nRESUMEN  ({time.time() - t_total:.0f} s en total)\n{'=' * 70}")
    for argv, estado, dt in resultados:
        print(f"  {estado:14s} {dt:7.0f}s  {' '.join(argv)}")
    fallos = sum(1 for _, e, _ in resultados if e == "FALLÓ")
    print(f"\n{len(resultados) - fallos}/{len(resultados)} sin fallos")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
