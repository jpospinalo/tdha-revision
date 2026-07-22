#!/usr/bin/env python3
"""
Encadena varias corridas de ``run_experiment.py`` en un solo proceso.

Genera el producto cartesiano de las dimensiones que se le pasan (sitios,
subconjuntos de ROIs, arquitecturas, enventanados, etc.) y lanza una corrida por
combinación. Cada corrida se ejecuta como un subproceso independiente, de modo que
un fallo aislado no interrumpe el resto de la cola.

Los argumentos coinciden con los de ``run_experiment.py``: el enventanado se define
en segundos (``--window-seconds`` / ``--step-seconds`` u ``--overlap``) o en TR
(``--window-tr`` / ``--step-tr``), pero no en ambas unidades a la vez. La
representación usa los mismos nombres que el ejecutor (``ordered``, ``permuted``,
``mean``, ``mean_std``, ``static``); con ``static`` se omiten los parámetros de
ventana automáticamente.

Cualquier argumento no reconocido se reenvía tal cual a ``run_experiment.py`` (por
ejemplo ``--n-splits 5`` o ``--class-weight``), aplicándose a todas las corridas.

Ejemplos
--------
Barrido de arquitecturas sobre NYU y varios subconjuntos::

    python run_queue.py --sites NYU --roi-sets 12 18 39 --models lstm gru

Enventanados físicos con solapamiento fijo::

    python run_queue.py --sites NYU --roi-sets 12 \\
        --window-seconds 100 140 --overlap 0.75 --n-splits 5

Ver los comandos sin ejecutarlos::

    python run_queue.py --sites NYU --roi-sets 12 18 --models lstm --dry-run

Lote grande en un solo proceso (arranca TensorFlow una vez y reutiliza la caché de
conectividad entre configuraciones que comparten datos y enventanado)::

    python run_queue.py --in-process --sites NYU Peking \\
        --roi-sets 12 18 39 --models lstm gru
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPRESENTATIONS = ("ordered", "permuted", "mean", "mean_std", "static", "partial", "hybrid")
WINDOW_SHAPES = ("rectangular", "gaussian")


def _expand(values: Sequence[Any] | None) -> list[Any]:
    """Una lista de valores, o ``[None]`` para indicar 'no variar esta dimensión'."""

    return list(values) if values else [None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--python", default=sys.executable, help="intérprete a usar")
    parser.add_argument(
        "--script",
        default=str(Path(__file__).with_name("run_experiment.py")),
        help="ruta a run_experiment.py",
    )
    parser.add_argument("--dry-run", action="store_true", help="imprime los comandos sin ejecutarlos")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="corre todas las configuraciones en este mismo proceso, sin arrancar "
        "TensorFlow una vez por corrida. Más rápido para lotes grandes; sin el "
        "aislamiento del subproceso, así que una caída detiene la cola.",
    )

    parser.add_argument("--sites", nargs="*", help="uno o más sitios")
    parser.add_argument("--roi-sets", nargs="*", help="uno o más subconjuntos de ROIs")
    parser.add_argument("--models", nargs="*", help="una o más arquitecturas")
    parser.add_argument(
        "--representation",
        nargs="*",
        choices=REPRESENTATIONS,
        help=f"una o más de: {', '.join(REPRESENTATIONS)}",
    )

    # Enventanado en segundos (tiempo físico).
    parser.add_argument("--window-seconds", type=float, nargs="*")
    parser.add_argument("--step-seconds", type=float, nargs="*")
    parser.add_argument("--overlap", type=float, nargs="*")
    # Enventanado en TR (alternativa, no combinable con la anterior).
    parser.add_argument("--window-tr", type=int, nargs="*")
    parser.add_argument("--step-tr", type=int, nargs="*")

    parser.add_argument("--window-shape", nargs="*", choices=WINDOW_SHAPES)
    parser.add_argument("--gaussian-sigma", type=float, nargs="*")
    parser.add_argument("--fisher-z", action="store_true", help="aplica Fisher z a todas las corridas")
    return parser


def _check_unique_source(names_values: list[tuple[str, Sequence[Any] | None]], label: str) -> None:
    provided = [name for name, value in names_values if value]
    if len(provided) > 1:
        raise SystemExit(
            f"ERROR: {label} está definido de varias formas a la vez ({', '.join(provided)}). "
            "Elija una sola."
        )


def build_arg_lists(args: argparse.Namespace, passthrough: Sequence[str]) -> list[list[str]]:
    """Genera la lista de argumentos de experimento por combinación.

    Devuelve solo los argumentos que recibe ``run_experiment.py`` (sin el prefijo
    ``python`` / ``script``), de modo que sirvan tanto para ``subprocess`` como para
    la llamada in-process a ``run_experiment.main``.
    """

    # La longitud de la ventana y el desplazamiento admiten una sola unidad cada uno.
    _check_unique_source(
        [("--window-seconds", args.window_seconds), ("--window-tr", args.window_tr)],
        "la longitud de la ventana",
    )
    _check_unique_source(
        [
            ("--step-seconds", args.step_seconds),
            ("--overlap", args.overlap),
            ("--step-tr", args.step_tr),
        ],
        "el desplazamiento entre ventanas",
    )

    combos = itertools.product(
        _expand(args.sites),
        _expand(args.roi_sets),
        _expand(args.models),
        _expand(args.representation),
        _expand(args.window_seconds),
        _expand(args.window_tr),
        _expand(args.step_seconds),
        _expand(args.overlap),
        _expand(args.step_tr),
        _expand(args.window_shape),
        _expand(args.gaussian_sigma),
    )

    arg_lists: list[list[str]] = []
    for site, roi, model, rep, w_s, w_tr, s_s, ov, s_tr, shape, sigma in combos:
        exp: list[str] = []
        if site:
            exp += ["--site", str(site)]
        if roi:
            exp += ["--roi-set", str(roi)]
        if model:
            exp += ["--model", str(model)]

        sin_ventana = rep in ("static", "partial")
        if rep:
            exp += ["--representation", rep]

        # 'static' y 'partial' usan toda la serie: no llevan parámetros de ventana.
        if not sin_ventana:
            if w_s is not None:
                exp += ["--window-seconds", str(w_s)]
            if w_tr is not None:
                exp += ["--window", str(w_tr)]
            if s_s is not None:
                exp += ["--step-seconds", str(s_s)]
            if ov is not None:
                exp += ["--overlap", str(ov)]
            if s_tr is not None:
                exp += ["--step", str(s_tr)]
            if shape:
                exp += ["--window-shape", shape]
            if sigma is not None:
                exp += ["--gaussian-sigma", str(sigma)]
        if args.fisher_z:
            exp.append("--fisher-z")

        exp.extend(passthrough)
        arg_lists.append(exp)
    return arg_lists


def _run_subprocess(args: argparse.Namespace, arg_lists: list[list[str]]) -> list[tuple[str, int]]:
    failures: list[tuple[str, int]] = []
    for i, exp in enumerate(arg_lists, start=1):
        cmd = [args.python, args.script, *exp]
        print(f"[{i}/{len(arg_lists)}] " + " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append((" ".join(exp), result.returncode))
            print(f"  ! terminó con código {result.returncode}")
    return failures


def _run_in_process(args: argparse.Namespace, arg_lists: list[list[str]]) -> list[tuple[str, int]]:
    # run_experiment está junto a este archivo; garantizamos que sea importable.
    sys.path.insert(0, str(Path(args.script).resolve().parent))
    import run_experiment  # noqa: E402

    failures: list[tuple[str, int]] = []
    for i, exp in enumerate(arg_lists, start=1):
        print(f"[{i}/{len(arg_lists)}] run_experiment.py " + " ".join(exp))
        if args.dry_run:
            continue
        try:
            run_experiment.main(exp)
        except SystemExit as exc:  # run_experiment aborta con SystemExit
            code = exc.code if isinstance(exc.code, int) else 1
            if code:
                failures.append((" ".join(exp), code))
                print(f"  ! terminó con código {code}: {exc}")
        except Exception as exc:  # una corrida no debe tumbar el resto de la cola
            failures.append((" ".join(exp), 1))
            print(f"  ! excepción: {type(exc).__name__}: {exc}")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)

    arg_lists = build_arg_lists(args, passthrough)
    modo = "in-process" if args.in_process else "subproceso"
    print(f"Corridas en cola: {len(arg_lists)} ({modo})\n")

    runner = _run_in_process if args.in_process else _run_subprocess
    failures = runner(args, arg_lists)

    if failures and not args.dry_run:
        print(f"\n{len(failures)} de {len(arg_lists)} corridas fallaron:")
        for exp, code in failures:
            print(f"  · (código {code}) {exp}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
