#!/usr/bin/env python3
"""Orquestador de la campaña ``loso_static_v1`` (Sección 37 del plan).

Responsabilidad exclusiva: construir la matriz de 48 comandos formales,
ejecutarlos como subprocesos independientes de ``run_loso.py``, y abortar la
campaña completa ante el primer fallo. NO contiene cálculo de FC, lógica de
modelo, lógica de splits ni análisis estadístico — todo eso vive en
``run_loso.py`` (entrenamiento/evaluación) y ``analysis/loso/`` (análisis).

Por diseño, ``max_parallel=1``: nunca se ejecutan dos redes BrainNetCNN en
paralelo, para evitar acumulación de estado de TensorFlow entre corridas
(Sección 30).

Uso
---
    python run_loso_campaign.py --dry-run
    python run_loso_campaign.py --resume
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_loso import BNN_SEEDS, MODELS, ROI_SETS, SITES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_LOSO = Path(__file__).resolve().parent / "run_loso.py"

EXPECTED_TOTAL = 48
EXPECTED_BRAINNETCNN = 40
EXPECTED_LOGREG = 8


def build_run_matrix() -> list[dict[str, Any]]:
    """Orden determinista: site -> ROI -> BrainNet seeds -> logistic (Sección 37)."""

    matrix: list[dict[str, Any]] = []
    for held_out_site in SITES:
        for roi_set in ROI_SETS:
            for seed in BNN_SEEDS:
                matrix.append(
                    {
                        "held_out_site": held_out_site,
                        "roi_set": roi_set,
                        "model": "brainnetcnn",
                        "model_seed": seed,
                    }
                )
            matrix.append(
                {
                    "held_out_site": held_out_site,
                    "roi_set": roi_set,
                    "model": "logreg",
                    "model_seed": None,
                }
            )
    _validate_run_matrix(matrix)
    return matrix


def _validate_run_matrix(matrix: list[dict[str, Any]]) -> None:
    if len(matrix) != EXPECTED_TOTAL:
        raise SystemExit(f"STOP: la matriz tiene {len(matrix)} corridas; se esperaban {EXPECTED_TOTAL}.")
    n_bnn = sum(1 for row in matrix if row["model"] == "brainnetcnn")
    n_log = sum(1 for row in matrix if row["model"] == "logreg")
    if n_bnn != EXPECTED_BRAINNETCNN:
        raise SystemExit(f"STOP: {n_bnn} corridas brainnetcnn; se esperaban {EXPECTED_BRAINNETCNN}.")
    if n_log != EXPECTED_LOGREG:
        raise SystemExit(f"STOP: {n_log} corridas logreg; se esperaban {EXPECTED_LOGREG}.")
    identities = {
        (row["held_out_site"], row["roi_set"], row["model"], row["model_seed"])
        for row in matrix
    }
    if len(identities) != EXPECTED_TOTAL:
        raise SystemExit("STOP: hay identidades de corrida duplicadas en la matriz.")


def command_for(row: dict[str, Any], *, resume: bool, dry_run: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_LOSO),
        "--held-out-site", row["held_out_site"],
        "--roi-set", row["roi_set"],
        "--model", row["model"],
    ]
    if row["model_seed"] is not None:
        cmd += ["--model-seed", str(row["model_seed"])]
    if resume:
        cmd.append("--resume")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_campaign(*, resume: bool, dry_run: bool, limit: int | None = None) -> int:
    matrix = build_run_matrix()
    if limit is not None:
        matrix = matrix[:limit]

    print(f"Matriz de campaña: {len(matrix)} corridas (max_parallel=1, secuencial).")
    for position, row in enumerate(matrix, start=1):
        cmd = command_for(row, resume=resume, dry_run=dry_run)
        label = (
            f"[{position}/{len(matrix)}] held_out={row['held_out_site']} "
            f"roi={row['roi_set']} model={row['model']} seed={row['model_seed']}"
        )
        print(f"\n=== {label} ===")
        print(" ".join(cmd))
        result = subprocess.run(cmd, cwd=str(RUN_LOSO.parent))
        if result.returncode != 0:
            print(
                f"\nSTOP CAMPAÑA: la corrida {label} terminó con código "
                f"{result.returncode}. No se continúa con el resto de la matriz; "
                "no se mezclan formal runs generados bajo configuraciones "
                "científicas distintas (Sección 73)."
            )
            return result.returncode

    print(f"\nCampaña completa: {len(matrix)}/{len(matrix)} corridas terminaron con código 0.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true", help="omite corridas formales ya válidas")
    parser.add_argument("--dry-run", action="store_true", help="propaga --dry-run a cada subproceso; no entrena nada")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="ejecuta solo las primeras N filas de la matriz (diagnóstico/CP2, no para campaña formal)",
    )
    parser.add_argument("--print-matrix", action="store_true", help="imprime la matriz de 48 corridas y termina")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.print_matrix:
        matrix = build_run_matrix()
        for position, row in enumerate(matrix, start=1):
            print(f"{position:2d}. {row}")
        print(f"\nTotal: {len(matrix)} (brainnetcnn={sum(1 for r in matrix if r['model']=='brainnetcnn')}, "
              f"logreg={sum(1 for r in matrix if r['model']=='logreg')})")
        return 0

    return run_campaign(resume=args.resume, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
