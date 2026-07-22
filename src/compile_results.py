#!/usr/bin/env python3
"""Compila corridas de ``results/runs`` del proyecto TDHA.

Compatible con configuraciones históricas y con el esquema v2 de
``run_experiment.py``: representación estática/dinámica, ventanas en TR o
segundos, solapamiento, Fisher z y ventanas gaussianas.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "results" / "runs"
COUNT_COLUMNS = {"true_positives", "true_negatives", "false_positives", "false_negatives"}
META_COLUMNS = {
    "fold", "repeat", "n_epochs", "best_epoch", "n_fit", "n_inner_val",
    "n_outer_val", "class_weight_0", "class_weight_1",
}
PREFERRED_METRICS = [
    "loss", "accuracy", "balanced_accuracy", "precision", "recall",
    "specificity", "f1", "f1_macro", "auc",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se pudo leer {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path} no contiene un objeto JSON.")
    return obj


def _windowing(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("windowing")
    return value if isinstance(value, dict) else {}


def _diagnostics(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("windowing_diagnostics")
    return value if isinstance(value, dict) else {}


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _suffix_from_metrics(path: Path) -> str:
    return path.stem.removeprefix("metrics_val")


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=[np.number]).columns
    discovered = [c for c in numeric if c not in META_COLUMNS and c not in COUNT_COLUMNS]
    ordered = [c for c in PREFERRED_METRICS if c in discovered]
    ordered.extend(c for c in discovered if c not in ordered)
    return ordered


def _safe_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if series.notna().sum() > 1 else float("nan")


def _pct(value: float, metric: str) -> float:
    return value if metric == "loss" else value * 100.0


def summarize(run_dir: Path, cfg: dict[str, Any], suffix: str = "") -> dict[str, Any] | None:
    val_path = run_dir / f"metrics_val{suffix}.csv"
    train_path = run_dir / f"metrics_train{suffix}.csv"
    if not val_path.exists() or not train_path.exists():
        return None

    val = pd.read_csv(val_path)
    train = pd.read_csv(train_path)
    if val.empty or train.empty:
        return None

    w = _windowing(cfg)
    d = _diagnostics(cfg)
    git = cfg.get("git") if isinstance(cfg.get("git"), dict) else {}
    representation = cfg.get("representation") or ("static" if cfg.get("window") is None else "ordered")
    mode = "static" if representation in ("static", "partial") else "dynamic"

    row: dict[str, Any] = {
        "run_id": str(cfg.get("run_id", run_dir.name)) + suffix,
        "base_run_id": cfg.get("run_id", run_dir.name),
        "subset_suffix": suffix or None,
        "config_schema_version": cfg.get("config_schema_version", 1),
        "config_hash": cfg.get("config_hash"),
        "site": cfg.get("site"),
        "roi_set": cfg.get("roi_set"),
        "n_subjects": cfg.get("n_subjects"),
        "n_rois": cfg.get("n_rois"),
        "n_timepoints": cfg.get("n_timepoints"),
        "n_windows": cfg.get("n_windows", d.get("n_windows")),
        "model": cfg.get("model"),
        "seed": cfg.get("seed"),
        "n_splits": cfg.get("n_splits"),
        "n_repeats": cfg.get("n_repeats"),
        "n_folds": int(len(val)),
        "split_fingerprint": cfg.get("split_fingerprint"),
        "representation": representation,
        "representation_seed": cfg.get("representation_seed"),
        "connectivity_mode": mode,
        "windowing_preset": cfg.get("windowing_preset"),
        "window_tr": _first(w.get("window_tr"), cfg.get("window")),
        "step_tr": _first(w.get("step_tr"), cfg.get("step")),
        "window_seconds": w.get("window_seconds"),
        "step_seconds": w.get("step_seconds"),
        "requested_window_seconds": w.get("requested_window_seconds"),
        "requested_step_seconds": w.get("requested_step_seconds"),
        "requested_overlap": w.get("requested_overlap"),
        "effective_overlap": _first(w.get("effective_overlap"), d.get("effective_overlap")),
        "window_shape": _first(w.get("shape"), cfg.get("window_shape"), "rectangular"),
        "gaussian_sigma": _first(w.get("gaussian_sigma"), cfg.get("gaussian_sigma")),
        "fisher_z": bool(cfg.get("fisher_z", w.get("fisher_z", False))),
        "constant_policy": cfg.get("constant_policy", "zero"),
        "random_subset": cfg.get("random_subset"),
        "start_from_epoch": cfg.get("start_from_epoch"),
        "bold_hash": cfg.get("bold_hash"),
        "atlas_hash": cfg.get("atlas_hash"),
        "roi_indices_hash": cfg.get("roi_indices_hash"),
        "data_code_hash": cfg.get("data_code_hash"),
        "runner_code_hash": cfg.get("runner_code_hash"),
        "usuario": git.get("user"),
        "commit": (git.get("commit") or "")[:8],
        "arbol_limpio": git.get("clean"),
        "timestamp": cfg.get("timestamp"),
        "methodological_warnings": " | ".join(cfg.get("methodological_warnings", []) or []),
    }

    metrics = sorted(set(_metric_columns(val)).intersection(_metric_columns(train)),
                     key=lambda x: PREFERRED_METRICS.index(x) if x in PREFERRED_METRICS else 999)
    for metric in metrics:
        va = pd.to_numeric(val[metric], errors="coerce")
        tr = pd.to_numeric(train[metric], errors="coerce")
        row[f"train_{metric}_mean"] = _pct(float(tr.mean()), metric)
        row[f"val_{metric}_mean"] = _pct(float(va.mean()), metric)
        row[f"val_{metric}_sd"] = _pct(_safe_std(va), metric)
        row[f"val_{metric}_median"] = _pct(float(va.median()), metric)
        row[f"val_{metric}_min"] = _pct(float(va.min()), metric)
        row[f"val_{metric}_max"] = _pct(float(va.max()), metric)

    # Alias históricos.
    for metric in metrics:
        row[f"train_{metric}"] = row[f"train_{metric}_mean"]
        row[f"val_{metric}"] = row[f"val_{metric}_mean"]
    if "accuracy" in metrics:
        row["gap_acc"] = row["train_accuracy_mean"] - row["val_accuracy_mean"]
    row["epoca_media"] = float(val["best_epoch"].mean()) if "best_epoch" in val else np.nan
    row["epoca_sd"] = _safe_std(pd.to_numeric(val["best_epoch"], errors="coerce")) if "best_epoch" in val else np.nan
    return row


def collect(root: str | Path, *, strict: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    root = Path(root)
    for cfg_path in sorted(root.glob("*/config.json")):
        try:
            cfg = _read_json(cfg_path)
            run_dir = cfg_path.parent
            val_files = sorted(run_dir.glob("metrics_val*.csv"))
            if not val_files:
                errors.append(f"{run_dir.name}: corrida incompleta, sin metrics_val*.csv")
                continue
            for path in val_files:
                row = summarize(run_dir, cfg, _suffix_from_metrics(path))
                if row is not None:
                    rows.append(row)
        except Exception as exc:  # informa el archivo defectuoso sin perder las demás corridas
            errors.append(f"{cfg_path}: {exc}")
    if strict and errors:
        raise ValueError("Errores durante la compilación:\n- " + "\n- ".join(errors))
    frame = pd.DataFrame(rows)
    frame.attrs["collection_warnings"] = errors
    return frame


def check_comparability(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if df.empty:
        return problems

    for site, group in df.groupby("site", dropna=False):
        hashes = set(group["bold_hash"].dropna()) if "bold_hash" in group else set()
        if len(hashes) > 1:
            problems.append(f"las señales BOLD de {site} difieren entre corridas ({len(hashes)} hashes)")

    hard = [
        ("seed", "semilla"), ("n_splits", "n_splits"), ("n_repeats", "n_repeats"),
        ("split_fingerprint", "particiones externas/internas"),
    ]
    for column, label in hard:
        if column in df and df[column].dropna().nunique() > 1:
            problems.append(f"{label} distintos; la comparación deja de ser estrictamente pareada")

    if "arbol_limpio" in df and df["arbol_limpio"].eq(False).any():
        dirty = df.loc[df["arbol_limpio"].eq(False), "run_id"].astype(str).tolist()
        problems.append(f"corridas realizadas con cambios sin confirmar: {dirty}")
    if df.duplicated(subset=[c for c in ["config_hash", "subset_suffix"] if c in df], keep=False).any():
        problems.append("se detectaron configuraciones duplicadas")
    return problems


def methodological_group_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "site", "roi_set", "model", "representation", "connectivity_mode",
        "window_tr", "step_tr", "window_seconds", "step_seconds",
        "effective_overlap", "window_shape", "gaussian_sigma", "fisher_z",
        "constant_policy",
    ]
    return [c for c in candidates if c in df.columns]


def aggregate_table(df: pd.DataFrame, group_by: Iterable[str] | None = None) -> pd.DataFrame:
    groups = list(group_by or methodological_group_columns(df))
    metric_cols = [c for c in df.columns if c.startswith("val_") and c.endswith("_mean")]
    if not groups or not metric_cols:
        return pd.DataFrame()
    aggregations: dict[str, list[str]] = {c: ["mean", "std", "min", "max", "median"] for c in metric_cols}
    result = df.groupby(groups, dropna=False).agg(aggregations)
    result.columns = [f"{a}_{b}" for a, b in result.columns]
    result = result.reset_index()
    counts = df.groupby(groups, dropna=False).size().rename("n_runs").reset_index()
    return counts.merge(result, on=groups, how="left")


def paired_stats(root: str | Path, runs: dict[Any, str], metric: str = "accuracy") -> None:
    from scipy import stats
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.multitest import multipletests

    values: dict[Any, np.ndarray] = {}
    for key, run_id in runs.items():
        frame = pd.read_csv(Path(root) / run_id / "metrics_val.csv").sort_values("fold")
        if metric not in frame:
            raise ValueError(f"La métrica {metric!r} no existe en {run_id}.")
        values[key] = frame[metric].to_numpy(dtype=float)
    lengths = {k: len(v) for k, v in values.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Número de pliegues distinto entre corridas: {lengths}")

    long = pd.concat([
        pd.DataFrame({"value": v, "group": str(k), "fold": np.arange(1, len(v) + 1)})
        for k, v in values.items()
    ], ignore_index=True)
    print("\nANOVA de medidas repetidas:\n")
    print(AnovaRM(long, "value", "fold", within=["group"]).fit())

    rows = []
    for a, b in itertools.combinations(values, 2):
        _, p_t = stats.ttest_rel(values[a], values[b])
        try:
            _, p_w = stats.wilcoxon(values[a], values[b])
        except ValueError:
            p_w = np.nan
        rows.append({"grupo_1": a, "grupo_2": b,
                     "dif_pp": (values[b].mean() - values[a].mean()) * 100,
                     "p_t_pareada": p_t, "p_wilcoxon": p_w})
    result = pd.DataFrame(rows)
    result["p_holm"] = multipletests(result["p_t_pareada"], method="holm")[1]
    result["significativo"] = result["p_holm"] < 0.05
    print("\nContrastes pareados con corrección de Holm:\n")
    print(result.round(4).to_string(index=False))


def _filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    for attr, column in [("site", "site"), ("model", "model"),
                         ("representation", "representation")]:
        value = getattr(args, attr, None)
        if value is not None:
            df = df[df[column].astype(str) == str(value)]
    if args.roi_set:
        allowed = {str(x) for x in args.roi_set}
        df = df[df["roi_set"].astype(str).isin(allowed)]
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--site")
    parser.add_argument("--model")
    parser.add_argument("--roi-set", nargs="*")
    parser.add_argument("--representation")
    parser.add_argument("--out", help="CSV detallado por corrida")
    parser.add_argument("--aggregate-out", help="CSV agregado por configuración metodológica")
    parser.add_argument("--strict", action="store_true", help="fallar ante corridas incompletas o archivos inválidos")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--stats-metric", default="accuracy")
    parser.add_argument(
        "--stats-by",
        choices=["roi_set", "representation", "model"],
        default="roi_set",
        help="dimensión a comparar de forma pareada: roi_set (por defecto), "
        "representation o model; el resto de dimensiones deben quedar fijas por filtro",
    )
    args = parser.parse_args(argv)

    df = collect(args.root, strict=args.strict)
    warnings = df.attrs.get("collection_warnings", [])
    if df.empty:
        raise SystemExit(f"No se encontraron corridas completas en {args.root}")
    df = _filter(df, args)
    if df.empty:
        raise SystemExit("Ninguna corrida coincide con los filtros.")

    display_cols = [
        "run_id", "site", "roi_set", "model", "representation", "window_seconds",
        "window_tr", "step_tr", "effective_overlap", "window_shape", "fisher_z",
        "n_folds", "n_windows", "val_accuracy_mean", "val_accuracy_sd",
        "val_f1_macro_mean", "val_auc_mean", "gap_acc", "commit",
    ]
    print(df[[c for c in display_cols if c in df]].round(4).to_string(index=False))

    if warnings:
        print("\nAVISOS DE RECOLECCIÓN")
        for warning in warnings:
            print(f"  · {warning}")
    problems = check_comparability(df)
    if problems:
        print("\nAVISOS DE COMPARABILIDAD")
        for problem in problems:
            print(f"  · {problem}")
    else:
        print("\nLas corridas seleccionadas son compatibles para comparación pareada.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nTabla detallada guardada en {args.out}")
    if args.aggregate_out:
        aggregated = aggregate_table(df)
        Path(args.aggregate_out).parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(args.aggregate_out, index=False)
        print(f"Tabla agregada guardada en {args.aggregate_out}")

    if args.stats:
        base = df[df["random_subset"].isna()] if "random_subset" in df else df
        group_col = args.stats_by
        # Todo lo que no sea la dimensión a comparar debe quedar fijo (site siempre).
        fixed = [d for d in ("site", "model", "roi_set", "representation") if d != group_col]
        for column in fixed:
            if column in base and base[column].dropna().nunique() > 1:
                raise SystemExit(
                    f"Para --stats por {group_col}, filtre un solo {column} "
                    f"(hay {base[column].nunique()}: use --{column.replace('_', '-')})."
                )
        if "split_fingerprint" in base and base["split_fingerprint"].dropna().nunique() > 1:
            raise SystemExit("Las corridas no comparten la misma huella de particiones.")
        order = "n_rois" if group_col == "roi_set" else group_col
        runs = {
            row[group_col]: row.base_run_id
            for _, row in base.sort_values(order).iterrows()
        }
        if len(runs) < 2:
            raise SystemExit(f"Se requieren al menos dos valores de {group_col}.")
        paired_stats(args.root, runs, args.stats_metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
