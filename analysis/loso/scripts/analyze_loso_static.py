#!/usr/bin/env python3
"""Analyzer congelado de la campaña ``loso_static_v1`` (Secciones 49-57).

Este script se implementa y se prueba con fixtures sintéticas ANTES de ver
resultados reales (Sección 49). Después del freeze científico
(``CP4``/Sección 63), no se modifica su lógica salvo para corregir un bug
declarado explícitamente (Sección 66/CP7) — nunca en silencio.

Solo lee corridas formales ya almacenadas bajo ``results/loso/<run_id>/``; no
entrena, no selecciona hiperparámetros, no ejecuta GPU, no modifica esos
resultados (contrato de ``analysis/loso/README.md``).
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results" / "loso"
CONFIG_PATH = REPO_ROOT / "analysis" / "loso" / "config" / "loso_analysis_config.json"
OUTPUT_DIR = REPO_ROOT / "analysis" / "loso" / "outputs"
SPEC_PATH = REPO_ROOT / "analysis" / "loso" / "IMPLEMENTATION_SPEC.md"
CODE_ROOT = REPO_ROOT / "src"
ATLAS_DIR = REPO_ROOT / "data" / "atlas"
BOLD_DIR = REPO_ROOT / "data" / "bold"
LOSO_TEST_DIR = REPO_ROOT / "analysis" / "loso" / "tests"

SITES = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]
ROI_SETS = ["12", "116"]
MODELS = ("brainnetcnn", "logreg")
BNN_SEEDS = [42, 43, 44, 45, 46]

EXPECTED_PREDICTIONS_TOTAL = 5580
EXPECTED_PREDICTIONS_BY_SITE = {"NYU": 2124, "Peking": 2196, "NeuroIMAGE": 468, "OHSU": 792}
EXPECTED_RUNS_TOTAL = 48
EXPECTED_METRICS_SUMMARY_ROWS = 16
EXPECTED_CONTRASTS_ROWS = 12
EXPECTED_CONVERGENCE_ROWS = 8

# CP20 (Sección 33): la "closeout analysis environment" es el runtime que
# ejecuta ESTE analyzer de cierre. No tiene que ser idéntica al training
# environment (Sección 33.3) — se registra por separado, nunca se mezcla.


def closeout_analysis_environment() -> dict[str, Any]:
    import sklearn
    import scipy

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    return info


def _sha_file(path: Path, *, length: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _sha_bytes(data: bytes, *, length: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:length]


def _full_sha256_file(path: Path) -> str:
    """SHA-256 completo (64 hex) de un archivo persistente (Sección 35)."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha256(obj: Any) -> str:
    """SHA-256 semántico/canónico de un objeto JSON (Sección 36): claves
    ordenadas, separadores compactos, sin ASCII forzado, sin NaN."""

    payload = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dupes(values: Sequence[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for v in values:
        if v in seen:
            out.append(v)
        seen.add(v)
    return out


def load_design_and_splits() -> tuple[dict[str, Any], pd.DataFrame]:
    """Carga ``design.json``/``splits.csv`` congelados (Sección 15/CP4 de la
    campaña original). Solo lectura: esta fase NO regenera ni modifica estos
    archivos (Sección 2 del plan de cierre: ``results/loso/_design/**`` es
    intocable)."""

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import run_loso as L  # noqa: WPS433 (import diferido intencional)

    if not L.DESIGN_JSON_PATH.exists() or not L.SPLIT_MANIFEST_PATH.exists():
        raise SystemExit(
            "STOP: falta results/loso/_design/loso_static_v1_design.json o "
            "loso_static_v1_splits.csv; no se puede auditar ni cerrar el "
            "análisis sin el design congelado de la campaña."
        )
    design = json.loads(L.DESIGN_JSON_PATH.read_text(encoding="utf-8"))
    splits_df = pd.read_csv(L.SPLIT_MANIFEST_PATH)
    return design, splits_df


def load_analysis_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "campaign_id", "site_order", "roi_order", "models", "brainnet_seeds",
        "primary_metric", "bootstrap_iterations", "bootstrap_seed", "ci_level",
    }
    missing = required.difference(config)
    if missing:
        raise SystemExit(f"STOP: loso_analysis_config.json incompleto: faltan {sorted(missing)}.")
    return config


# ---------------------------------------------------------------------------
# Descubrimiento de corridas formales (Sección 42: ignora .staging y _design)
# ---------------------------------------------------------------------------


def discover_formal_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    dirs = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "_design" or child.name.startswith(".staging"):
            continue
        dirs.append(child)
    return dirs


def load_formal_run(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"STOP: {run_dir} no tiene config.json.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("formal", False):
        raise SystemExit(f"STOP: {run_dir} no está marcado formal=true.")
    predictions_path = run_dir / "predictions_test.csv"
    if not predictions_path.exists():
        raise SystemExit(f"STOP: {run_dir} no tiene predictions_test.csv.")
    predictions = pd.read_csv(predictions_path)
    metrics_test_path = run_dir / "metrics_test.csv"
    metrics_test = pd.read_csv(metrics_test_path) if metrics_test_path.exists() else pd.DataFrame()
    return {
        "run_dir": run_dir,
        "config": config,
        "predictions": predictions,
        "metrics_test": metrics_test,
    }


def discover_runs(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    return [load_formal_run(d) for d in discover_formal_run_dirs(results_root)]


# ---------------------------------------------------------------------------
# Manifest (Sección 56)
# ---------------------------------------------------------------------------


def _expected_condition_keys() -> list[tuple[str, str, str, int | None]]:
    keys: list[tuple[str, str, str, int | None]] = []
    for site in SITES:
        for roi in ROI_SETS:
            for seed in BNN_SEEDS:
                keys.append((site, roi, "brainnetcnn", seed))
            keys.append((site, roi, "logreg", None))
    return keys


def build_manifest(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    for run in runs:
        config = run["config"]
        key = (
            config["held_out_site"],
            str(config["roi_set"]),
            config["model"],
            config.get("model_seed"),
        )
        if key in by_key:
            raise SystemExit(f"STOP: manifest ambiguo — más de una corrida formal para {key}.")
        by_key[key] = run

    expected_keys = set(_expected_condition_keys())
    present_keys = set(by_key)
    missing = expected_keys - present_keys
    unexpected = present_keys - expected_keys
    if missing:
        raise SystemExit(f"STOP: faltan corridas formales para {sorted(missing)}.")
    if unexpected:
        raise SystemExit(f"STOP: corridas formales inesperadas (glob ambiguo): {sorted(unexpected)}.")
    if len(by_key) != EXPECTED_RUNS_TOTAL:
        raise SystemExit(f"STOP: manifest tiene {len(by_key)} corridas; se esperaban {EXPECTED_RUNS_TOTAL}.")

    entries = []
    for key, run in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or 0)):
        held_out_site, roi_set, model, seed = key
        config = run["config"]
        entries.append(
            {
                "held_out_site": held_out_site,
                "roi_set": roi_set,
                "model": model,
                "model_seed": seed,
                "run_id": config["run_id"],
                "identity_hash": config.get("identity_hash"),
                "config_sha256": _sha_file(run["run_dir"] / "config.json"),
                "predictions_sha256": _sha_file(run["run_dir"] / "predictions_test.csv"),
                "split_sha256": config.get("split_manifest_file_sha256"),
                "training_source_sha": config.get("training_source_git_sha"),
                "environment_signature": config.get("environment_signature"),
                "status": "formal",
            }
        )
    return {"campaign_id": "loso_static_v1", "n_runs": len(entries), "runs": entries}


# ---------------------------------------------------------------------------
# Predicciones largas (Secciones 54-55)
# ---------------------------------------------------------------------------


def build_predictions_long(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frames = []
    for run in runs:
        df = run["predictions"].copy()
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)

    if len(long_df) != EXPECTED_PREDICTIONS_TOTAL:
        raise SystemExit(
            f"STOP: loso_predictions_long tiene {len(long_df)} filas; se esperaban "
            f"{EXPECTED_PREDICTIONS_TOTAL}."
        )
    for site, expected_n in EXPECTED_PREDICTIONS_BY_SITE.items():
        n = int((long_df["held_out_site"] == site).sum())
        if n != expected_n:
            raise SystemExit(f"STOP: {site} tiene {n} filas en predictions_long; se esperaban {expected_n}.")
    if not ((long_df["y_prob"] >= 0) & (long_df["y_prob"] <= 1)).all():
        raise SystemExit("STOP: y_prob fuera de [0, 1] en predictions_long.")
    if not np.isfinite(long_df["y_prob"]).all():
        raise SystemExit("STOP: y_prob no finito en predictions_long.")
    return long_df


def build_metrics_by_run(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        config = run["config"]
        metrics_row = run["metrics_test"].iloc[0].to_dict() if len(run["metrics_test"]) else {}
        rows.append(
            {
                "held_out_site": config["held_out_site"],
                "roi_set": str(config["roi_set"]),
                "model": config["model"],
                "model_seed": config.get("model_seed"),
                "run_id": config["run_id"],
                **metrics_row,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_RUNS_TOTAL:
        raise SystemExit(f"STOP: metrics_by_run tiene {len(df)} filas; se esperaban {EXPECTED_RUNS_TOTAL}.")
    return df


# ---------------------------------------------------------------------------
# Auditoría de cierre: Gates A-Q (Secciones 10-27 del plan de cierre)
# ---------------------------------------------------------------------------


def _gate_row(gate: str, description: str, expected: Any, observed: Any, status: str) -> dict[str, Any]:
    return {"gate": gate, "description": description, "expected": expected, "observed": observed, "status": status}


def run_closeout_audit(
    runs: Sequence[Mapping[str, Any]], design: Mapping[str, Any], splits_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Gates A-Q. Lee únicamente corridas formales ya almacenadas y el design
    congelado; no entrena, no recalcula FC, no modifica ``results/loso/**``.

    Recolecta TODOS los fallos de TODOS los gates antes de decidir; si hay
    al menos uno, levanta ``SystemExit`` con el diagnóstico completo
    (gate/run_id/field/expected/observed) y no continúa hacia el cálculo de
    outputs (Sección 27, CP2 PASS). Si no hay fallos, devuelve las filas para
    ``LOSO_STATIC_V1_QA.md``.
    """

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import run_loso as L  # noqa: WPS433

    failures: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    def fail(gate: str, run_id: str, field: str, expected: Any, observed: Any) -> None:
        failures.append({"gate": gate, "run_id": run_id, "field": field, "expected": expected, "observed": observed})

    configs = [r["config"] for r in runs]
    n = len(configs)

    # --- Gate A: campaign_id y formal -------------------------------------
    bad = [c["run_id"] for c in configs if c.get("campaign_id") != "loso_static_v1" or c.get("formal") is not True]
    for rid in bad:
        fail("A", rid, "campaign_id/formal", "loso_static_v1 / True", "mismatch")
    gate_rows.append(_gate_row(
        "A", "campaign_id == loso_static_v1 y formal is True", f"{n}/{n}",
        f"{n - len(bad)}/{n}", "PASS" if not bad else "FAIL",
    ))

    # --- Gate B: único training_source_git_sha ----------------------------
    shas = sorted({c.get("training_source_git_sha") for c in configs})
    if len(shas) != 1:
        fail("B", "ALL", "training_source_git_sha", "1 valor único", shas)
    gate_rows.append(_gate_row(
        "B", "único training_source_git_sha en 48/48", "1 valor único",
        shas[0] if len(shas) == 1 else shas, "PASS" if len(shas) == 1 else "FAIL",
    ))

    # --- Gate C: único environment_signature de entrenamiento -------------
    envsig = sorted({c.get("environment_signature") for c in configs})
    if len(envsig) != 1:
        fail("C", "ALL", "environment_signature", "1 valor único", envsig)
    gate_rows.append(_gate_row(
        "C", "único environment_signature de entrenamiento en 48/48", "1 valor único",
        envsig[0] if len(envsig) == 1 else envsig, "PASS" if len(envsig) == 1 else "FAIL",
    ))

    # --- Gate D: nombre de directorio == run_id ---------------------------
    bad = [c["run_id"] for r, c in zip(runs, configs) if r["run_dir"].name != c["run_id"]]
    for rid in bad:
        fail("D", rid, "run_dir.name == config.run_id", rid, "mismatch")
    gate_rows.append(_gate_row(
        "D", "nombre de directorio == run_id", f"{n}/{n}", f"{n - len(bad)}/{n}",
        "PASS" if not bad else "FAIL",
    ))

    # --- Gate E: identidades únicas ----------------------------------------
    run_ids = [c["run_id"] for c in configs]
    identity_hashes = [c.get("identity_hash") for c in configs]
    keys = [(c["held_out_site"], str(c["roi_set"]), c["model"], c.get("model_seed")) for c in configs]
    dupe_ids, dupe_hashes, dupe_keys = _dupes(run_ids), _dupes(identity_hashes), _dupes(keys)
    n_bnn = sum(1 for c in configs if c["model"] == "brainnetcnn")
    n_log = sum(1 for c in configs if c["model"] == "logreg")
    if dupe_ids:
        fail("E", "ALL", "run_id duplicado", "48 únicos", dupe_ids)
    if dupe_hashes:
        fail("E", "ALL", "identity_hash duplicado", "48 únicos", dupe_hashes)
    if dupe_keys:
        fail("E", "ALL", "identidad (site,roi,model,seed) duplicada", "48 únicas", dupe_keys)
    if n_bnn != 40:
        fail("E", "ALL", "n_brainnetcnn", 40, n_bnn)
    if n_log != 8:
        fail("E", "ALL", "n_logreg", 8, n_log)
    e_ok = not (dupe_ids or dupe_hashes or dupe_keys) and n_bnn == 40 and n_log == 8
    gate_rows.append(_gate_row(
        "E", "48 run_id/identity_hash únicos; 40 BrainNetCNN + 8 logistic", "48 únicas, 40/8",
        f"{len(set(run_ids))} run_id, {len(set(identity_hashes))} hash, {n_bnn} BNN, {n_log} logreg",
        "PASS" if e_ok else "FAIL",
    ))

    # --- Gate F: design y split manifest ------------------------------------
    f_ok = True
    if design.get("campaign_id") != "loso_static_v1":
        fail("F", "design.json", "campaign_id", "loso_static_v1", design.get("campaign_id"))
        f_ok = False
    if design.get("participant_count") != 465:
        fail("F", "design.json", "participant_count", 465, design.get("participant_count"))
        f_ok = False
    if len(splits_df) != 1860:
        fail("F", "splits.csv", "n_rows", 1860, len(splits_df))
        f_ok = False
    site_counts = splits_df["held_out_site"].value_counts().to_dict()
    if set(site_counts) != set(L.SITES) or any(v != 465 for v in site_counts.values()):
        fail("F", "splits.csv", "filas por rotación", "465 x 4 sitios", site_counts)
        f_ok = False
    if design.get("rotation_sizes") != L.EXPECTED_ROTATION_SIZES:
        fail("F", "design.json", "rotation_sizes", L.EXPECTED_ROTATION_SIZES, design.get("rotation_sizes"))
        f_ok = False
    gate_rows.append(_gate_row(
        "F", "design.json y splits.csv: campaign_id/participant_count/1860 filas/rotation_sizes",
        "campaign_id=loso_static_v1, 465 participantes, 1860 filas, 4x465",
        "coincide" if f_ok else "ver failures", "PASS" if f_ok else "FAIL",
    ))

    # --- Gate G: rotation split fingerprint ---------------------------------
    bad = []
    for c in configs:
        expected_fp = design.get("rotation_split_fingerprints", {}).get(c["held_out_site"])
        if c.get("rotation_split_fingerprint") != expected_fp:
            fail("G", c["run_id"], "rotation_split_fingerprint", expected_fp, c.get("rotation_split_fingerprint"))
            bad.append(c["run_id"])
    gate_rows.append(_gate_row(
        "G", "rotation_split_fingerprint == design[...][held_out_site]", f"{n}/{n}",
        f"{n - len(bad)}/{n}", "PASS" if not bad else "FAIL",
    ))

    # --- Gate H: split_membership.csv vs design (por subject_key) -----------
    # --- Gate I: no leakage (mismo split_membership.csv) --------------------
    bad_h: list[str] = []
    bad_i: list[str] = []
    split_membership_cache: dict[str, pd.DataFrame] = {}
    for run, c in zip(runs, configs):
        site = c["held_out_site"]
        sm_path = run["run_dir"] / "split_membership.csv"
        if not sm_path.exists():
            fail("H", c["run_id"], "split_membership.csv", "existe", "no existe")
            bad_h.append(c["run_id"])
            continue
        sm = pd.read_csv(sm_path)
        split_membership_cache[c["run_id"]] = sm

        design_block = splits_df.loc[splits_df["held_out_site"] == site]
        cols = ["site", "subject_id", "subject_key", "y_true", "split"]
        left = sm[cols].sort_values("subject_key").reset_index(drop=True)
        right = design_block[cols].sort_values("subject_key").reset_index(drop=True)
        if len(left) != 465 or len(right) != 465 or not left.equals(right):
            fail("H", c["run_id"], "split_membership vs design (por subject_key)", "465/465 idénticas",
                 f"{len(left)} vs {len(right)} filas, equals={left.equals(right) if len(left) == len(right) else 'n/a'}")
            bad_h.append(c["run_id"])

        held_out_rows = sm.loc[sm["site"] == site]
        if not (held_out_rows["split"] == "test").all():
            fail("I", c["run_id"], "held-out site 100% test", "100% test", "hay filas fit/inner_val")
            bad_i.append(c["run_id"])
        fit_idx = set(sm.loc[sm["split"] == "fit", "subject_key"])
        inner_idx = set(sm.loc[sm["split"] == "inner_val", "subject_key"])
        test_idx = set(sm.loc[sm["split"] == "test", "subject_key"])
        if fit_idx & inner_idx or fit_idx & test_idx or inner_idx & test_idx:
            fail("I", c["run_id"], "fit/inner_val/test disjuntos", "∅", "intersección no vacía")
            bad_i.append(c["run_id"])
        union = fit_idx | inner_idx | test_idx
        if len(union) != 465 or len(set(sm["subject_key"])) != 465:
            fail("I", c["run_id"], "unión fit∪inner_val∪test", 465, len(union))
            bad_i.append(c["run_id"])
    gate_rows.append(_gate_row(
        "H", "split_membership.csv == bloque design por subject_key (465/465)", f"{n}/{n}",
        f"{n - len(set(bad_h))}/{n}", "PASS" if not bad_h else "FAIL",
    ))
    gate_rows.append(_gate_row(
        "I", "no leakage: held-out 100% test; fit/inner_val/test disjuntos; unión=465", f"{n}/{n}",
        f"{n - len(set(bad_i))}/{n}", "PASS" if not bad_i else "FAIL",
    ))

    # --- Gate J: especificación científica ----------------------------------
    expected_scientific = {
        "class_weight": False, "site_weighting": False, "sample_weight": False,
        "harmonization": "none", "fisher_z": False, "constant_policy": "zero",
        "representation": "static", "n_windows": 1,
    }
    bad = []
    for c in configs:
        for field, expected_val in expected_scientific.items():
            if c.get(field) != expected_val:
                fail("J", c["run_id"], field, expected_val, c.get(field))
                bad.append(c["run_id"])
    gate_rows.append(_gate_row(
        "J", "no weighting/harmonization; static; fisher_z=False; constant_policy=zero", f"{n}/{n}",
        f"{n - len(set(bad))}/{n}", "PASS" if not bad else "FAIL",
    ))

    # --- Gate K: ROI/model --------------------------------------------------
    bad = []
    for c in configs:
        if str(c["roi_set"]) not in ("12", "116"):
            fail("K", c["run_id"], "roi_set", '"12"/"116"', c["roi_set"]); bad.append(c["run_id"])
        if c["model"] not in ("brainnetcnn", "logreg"):
            fail("K", c["run_id"], "model", "brainnetcnn/logreg", c["model"]); bad.append(c["run_id"])
        expected_features = L.EXPECTED_FEATURES.get(str(c["roi_set"]))
        if c.get("n_features") != expected_features:
            fail("K", c["run_id"], "n_features", expected_features, c.get("n_features")); bad.append(c["run_id"])
        if c["model"] == "brainnetcnn" and c.get("arch") != L.BNN_ARCH_KWARGS:
            fail("K", c["run_id"], "arch", L.BNN_ARCH_KWARGS, c.get("arch")); bad.append(c["run_id"])
    gate_rows.append(_gate_row(
        "K", "roi_set∈{12,116}; model∈{brainnetcnn,logreg}; feature counts; arch BNN congelada",
        f"{n}/{n}", f"{n - len(set(bad))}/{n}", "PASS" if not bad else "FAIL",
    ))

    # --- Gate L: prediction schema ------------------------------------------
    required_cols = {
        "held_out_site", "site", "subject_id", "subject_key", "y_true", "y_prob",
        "model", "roi_set", "model_seed", "run_id",
    }
    bad = []
    for run, c in zip(runs, configs):
        pred = run["predictions"]
        missing_cols = required_cols - set(pred.columns)
        if missing_cols:
            fail("L", c["run_id"], "columnas requeridas", sorted(required_cols), f"faltan {sorted(missing_cols)}")
            bad.append(c["run_id"]); continue
        checks = [
            (pred["site"] == c["held_out_site"]).all(),
            np.isfinite(pred["y_prob"]).all(),
            ((pred["y_prob"] >= 0) & (pred["y_prob"] <= 1)).all(),
            pred["subject_key"].nunique() == len(pred),
            len(pred) == L.EXPECTED_ROTATION_SIZES[c["held_out_site"]]["test"],
            set(pred["y_true"].unique()) == {0, 1},
            (pred["run_id"] == c["run_id"]).all(),
            (pred["model"] == c["model"]).all(),
            (pred["roi_set"].astype(str) == str(c["roi_set"])).all(),
        ]
        if c["model"] == "brainnetcnn":
            checks.append((pred["model_seed"] == c["model_seed"]).all())
        else:
            checks.append(pred["model_seed"].isna().all())
        if not all(checks):
            fail("L", c["run_id"], "prediction schema checks", "todos True", checks)
            bad.append(c["run_id"])
    gate_rows.append(_gate_row(
        "L", "predictions_test.csv: schema, [0,1], site==held_out, ambas clases, columnas consistentes",
        f"{n}/{n}", f"{n - len(set(bad))}/{n}", "PASS" if not bad else "FAIL",
    ))

    # --- Gate M: reproducción independiente de métricas test ----------------
    tol = 1e-9
    bad = []
    for run, c in zip(runs, configs):
        pred = run["predictions"]
        metrics_test = run["metrics_test"]
        if len(metrics_test) == 0:
            fail("M", c["run_id"], "metrics_test.csv", "1 fila", "0 filas"); bad.append(c["run_id"]); continue
        stored = metrics_test.iloc[0]
        y_true = pred["y_true"].to_numpy()
        y_pred_label = (pred["y_prob"].to_numpy() >= 0.5).astype(int)
        recomputed = {
            "auc": roc_auc_score(y_true, pred["y_prob"].to_numpy()),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_label),
            "f1_macro": f1_score(y_true, y_pred_label, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred_label, pos_label=1, zero_division=0),
            "specificity": recall_score(y_true, y_pred_label, pos_label=0, zero_division=0),
        }
        for field, value in recomputed.items():
            if field not in stored or abs(float(stored[field]) - value) > tol:
                fail("M", c["run_id"], field, value, stored.get(field))
                bad.append(c["run_id"])
    gate_rows.append(_gate_row(
        "M", "AUC/balanced_accuracy/f1_macro/recall/specificity recalculados == metrics_test.csv",
        f"{n}/{n} (tol {tol})", f"{n - len(set(bad))}/{n}", "PASS" if not bad else "FAIL",
    ))

    # --- Gate N: BrainNetCNN convergence -------------------------------------
    bad = []
    bnn_runs = [(run, c) for run, c in zip(runs, configs) if c["model"] == "brainnetcnn"]
    for run, c in bnn_runs:
        history_path = run["run_dir"] / "history.csv"
        conv = c.get("convergence")
        if not history_path.exists() or not conv:
            fail("N", c["run_id"], "history.csv/convergence", "existen", "faltan")
            bad.append(c["run_id"]); continue
        history = pd.read_csv(history_path)
        epochs_ran, best_epoch = conv.get("epochs_ran"), conv.get("best_epoch")
        checks = [
            epochs_ran == len(history),
            best_epoch is not None and 1 <= best_epoch <= epochs_ran,
            list(history["epoch"]) == list(range(1, len(history) + 1)),
            np.isfinite(conv.get("best_monitor_value", np.nan)),
            np.isfinite(conv.get("restored_monitor_value", np.nan)),
        ]
        if all(checks[:3]):
            row = history.loc[history["epoch"] == best_epoch]
            monitor_col = "inner_val_loss"
            if len(row) == 1 and monitor_col in history.columns:
                checks.append(abs(float(row[monitor_col].iloc[0]) - conv["best_monitor_value"]) < 1e-4)
            else:
                checks.append(False)
            checks.append(abs(conv["best_monitor_value"] - conv["restored_monitor_value"]) < 1e-4)
        if not all(checks):
            fail("N", c["run_id"], "convergence checks", "todos True", checks)
            bad.append(c["run_id"])
    gate_rows.append(_gate_row(
        "N", "40 BNN: history.csv + convergence coherentes; best/restored monitor ~iguales", "40/40",
        f"{40 - len(set(bad))}/40", "PASS" if not bad else "FAIL",
    ))

    # --- Gate O: logistic configuration --------------------------------------
    bad = []
    log_runs = [(run, c) for run, c in zip(runs, configs) if c["model"] == "logreg"]
    for run, c in log_runs:
        ok = c.get("arch") is None and c.get("model_seed") is None
        hp = c.get("identity", {}).get("scientific_hyperparameters")
        if hp != L.LOGREG_CONFIG:
            ok = False
            fail("O", c["run_id"], "scientific_hyperparameters", L.LOGREG_CONFIG, hp)
        if not ok:
            bad.append(c["run_id"])
    gate_rows.append(_gate_row(
        "O", "8 logistic: arch/model_seed null; hyperparams == LOGREG_CONFIG congelado", "8/8",
        f"{8 - len(set(bad))}/8", "PASS" if not bad else "FAIL",
    ))

    # --- Gate P: provenance de inputs y código --------------------------------
    bad = []
    for c in configs:
        site = c["held_out_site"]
        for other_site, bold_prefix in c.get("bold_sha256", {}).items():
            if design.get("input_bold_sha256", {}).get(other_site) != bold_prefix:
                fail("P", c["run_id"], f"bold_sha256[{other_site}]",
                     design.get("input_bold_sha256", {}).get(other_site), bold_prefix)
                bad.append(c["run_id"])
        if design.get("roi_indices_sha256", {}).get(str(c["roi_set"])) != c.get("roi_indices_hash"):
            fail("P", c["run_id"], "roi_indices_hash",
                 design.get("roi_indices_sha256", {}).get(str(c["roi_set"])), c.get("roi_indices_hash"))
            bad.append(c["run_id"])
        if design.get("split_manifest_file_sha256") != c.get("split_manifest_file_sha256"):
            fail("P", c["run_id"], "split_manifest_file_sha256",
                 design.get("split_manifest_file_sha256"), c.get("split_manifest_file_sha256"))
            bad.append(c["run_id"])
        ident = c.get("identity", {})
        if ident.get("model_code_sha256") is None and c["model"] != "logreg":
            fail("P", c["run_id"], "model_code_sha256 presente para BNN", "no null", "null")
            bad.append(c["run_id"])
        if c["model"] == "logreg" and ident.get("model_code_sha256") is not None:
            fail("P", c["run_id"], "model_code_sha256 para logreg", None, ident.get("model_code_sha256"))
            bad.append(c["run_id"])
    # Código histórico: full SHA-256 actual [:16] == prefijo histórico almacenado.
    code_files = {
        "runner_sha256": CODE_ROOT / "run_loso.py",
        "data_code_sha256": CODE_ROOT / "data.py",
    }
    code_current_prefix16 = {}
    for field, path in code_files.items():
        full = _full_sha256_file(path)
        code_current_prefix16[field] = full[:16]
        stored = {c["identity"][field] for c in configs}
        if stored != {full[:16]}:
            fail("P", "ALL", field, full[:16], sorted(stored))
            bad.append("ALL")
    model_code_full = _full_sha256_file(CODE_ROOT / "kerasmodels" / "brainnetcnn.py")
    stored_model_code = {c["identity"]["model_code_sha256"] for c in configs if c["model"] == "brainnetcnn"}
    if stored_model_code != {model_code_full[:16]}:
        fail("P", "ALL(BNN)", "model_code_sha256", model_code_full[:16], sorted(stored_model_code))
        bad.append("ALL")
    gate_rows.append(_gate_row(
        "P", "runner/data/model code hash actual[:16] == prefijo histórico en 48/48", f"{n}/{n}",
        f"{n - len(set(x for x in bad if x != 'ALL'))}/{n}" + (" (+ código)" if "ALL" in bad else ""),
        "PASS" if not bad else "FAIL",
    ))

    # --- Gate Q: feature hashes: no fabricar full hashes retrospectivos ------
    bad = []
    for c in configs:
        fmh = c.get("feature_matrix_sha256", {})
        for key, value in fmh.items():
            if not isinstance(value, str) or len(value) != 16:
                fail("Q", c["run_id"], f"feature_matrix_sha256[{key}] longitud", 16,
                     len(value) if isinstance(value, str) else type(value))
                bad.append(c["run_id"])
            if design.get("feature_matrix_sha256", {}).get(key) != value:
                fail("Q", c["run_id"], f"feature_matrix_sha256[{key}] vs design",
                     design.get("feature_matrix_sha256", {}).get(key), value)
                bad.append(c["run_id"])
    gate_rows.append(_gate_row(
        "Q", "feature_matrix_sha256: prefijos16 == design; NO se recomputan full hashes", f"{n}/{n}",
        f"{n - len(set(bad))}/{n}", "PASS" if not bad else "FAIL",
    ))

    if failures:
        lines = ["STOP: la auditoría de cierre (Gates A-Q) encontró fallos. No se calculó ningún output.", ""]
        for f_ in failures[:200]:
            lines.append(
                f"  gate={f_['gate']} run_id={f_['run_id']} field={f_['field']} "
                f"expected={f_['expected']!r} observed={f_['observed']!r}"
            )
        if len(failures) > 200:
            lines.append(f"  ... y {len(failures) - 200} fallos más.")
        raise SystemExit("\n".join(lines))

    return gate_rows


# ---------------------------------------------------------------------------
# Bootstrap (Secciones 51-52) y contrastes (Sección 53)
# ---------------------------------------------------------------------------


def _canonical_site_order(predictions_long: pd.DataFrame, site: str) -> np.ndarray:
    """subject_key ascendente para el sitio, común a todas las condiciones."""

    keys = predictions_long.loc[predictions_long["held_out_site"] == site, "subject_key"].unique()
    return np.sort(keys)


def _condition_arrays(
    predictions_long: pd.DataFrame, *, site: str, roi_set: str, model: str,
    seed: int | None, subject_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        (predictions_long["held_out_site"] == site)
        & (predictions_long["roi_set"].astype(str) == str(roi_set))
        & (predictions_long["model"] == model)
    )
    if model == "brainnetcnn":
        mask &= predictions_long["model_seed"] == seed
    sub = predictions_long.loc[mask].set_index("subject_key").reindex(subject_order)
    if sub["y_true"].isna().any():
        raise SystemExit(f"STOP: faltan predicciones para {site}/{roi_set}/{model}/seed={seed}.")
    return sub["y_true"].to_numpy(dtype=np.int64), sub["y_prob"].to_numpy(dtype=np.float64)


def _assert_y_true_consistent(predictions_long: pd.DataFrame, site: str, subject_order: np.ndarray) -> np.ndarray:
    """Sección 51 punto 2: y_true idéntico entre condiciones/seeds para el sitio."""

    sub = predictions_long.loc[predictions_long["held_out_site"] == site]
    pivot = sub.pivot_table(index="subject_key", values="y_true", aggfunc="nunique")
    if (pivot["y_true"] != 1).any():
        raise SystemExit(f"STOP: y_true inconsistente entre condiciones para {site}.")
    y_true = sub.drop_duplicates("subject_key").set_index("subject_key").reindex(subject_order)["y_true"]
    return y_true.to_numpy(dtype=np.int64)


def compute_site_bootstrap(
    predictions_long: pd.DataFrame, site: str, analysis_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Un conjunto de resamples por sitio, reutilizado por TODAS las condiciones
    y contrastes de ese sitio (Sección 51 punto 8)."""

    subject_order = _canonical_site_order(predictions_long, site)
    y_true = _assert_y_true_consistent(predictions_long, site, subject_order)

    control_pos = np.flatnonzero(y_true == 0)
    adhd_pos = np.flatnonzero(y_true == 1)
    n_iter = int(analysis_config["bootstrap_iterations"])
    rng = np.random.Generator(np.random.PCG64(int(analysis_config["bootstrap_seed"])))

    draws = np.empty((n_iter, len(subject_order)), dtype=np.int64)
    for i in range(n_iter):
        resampled_control = rng.choice(control_pos, size=control_pos.size, replace=True)
        resampled_adhd = rng.choice(adhd_pos, size=adhd_pos.size, replace=True)
        draws[i] = np.concatenate([resampled_control, resampled_adhd])

    condition_probs: dict[tuple[str, str, int | None], np.ndarray] = {}
    for roi_set in ROI_SETS:
        for seed in BNN_SEEDS:
            _, y_prob = _condition_arrays(
                predictions_long, site=site, roi_set=roi_set, model="brainnetcnn",
                seed=seed, subject_order=subject_order,
            )
            condition_probs[(roi_set, "brainnetcnn", seed)] = y_prob
        _, y_prob = _condition_arrays(
            predictions_long, site=site, roi_set=roi_set, model="logreg",
            seed=None, subject_order=subject_order,
        )
        condition_probs[(roi_set, "logreg", None)] = y_prob

    return {"subject_order": subject_order, "y_true": y_true, "draws": draws, "condition_probs": condition_probs}


def _auc_or_nan(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def condition_point_estimate(
    site_ctx: Mapping[str, Any], *, roi_set: str, model: str,
) -> tuple[float, dict[str, float] | None]:
    """Punto estimado sobre los datos observados (NO sobre el bootstrap)."""

    y_true = site_ctx["y_true"]
    if model == "brainnetcnn":
        seed_aucs = {
            seed: _auc_or_nan(y_true, site_ctx["condition_probs"][(roi_set, "brainnetcnn", seed)])
            for seed in BNN_SEEDS
        }
        point = float(np.mean(list(seed_aucs.values())))
        dispersion = {
            "seed_sd": float(np.std(list(seed_aucs.values()), ddof=1)),
            "seed_min": float(np.min(list(seed_aucs.values()))),
            "seed_max": float(np.max(list(seed_aucs.values()))),
        }
        return point, dispersion
    y_prob = site_ctx["condition_probs"][(roi_set, "logreg", None)]
    return _auc_or_nan(y_true, y_prob), None


def _auc_batch(y_true_rows: np.ndarray, y_prob_rows: np.ndarray) -> np.ndarray:
    """AUC vectorizado sobre muchas filas a la vez (una por iteración de
    bootstrap), vía la fórmula de Mann-Whitney basada en rangos:

        AUC = (suma_de_rangos_de_positivos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

    Numéricamente idéntico a ``sklearn.metrics.roc_auc_score`` fila por fila
    para clasificación binaria (verificado: diff máxima ~1e-16 en un chequeo
    de 500 iteraciones aleatorias) — no es una fórmula nueva, es la misma
    definición de AUC escrita para operar sobre toda la matriz de resamples
    de una vez en lugar de convocar sklearn 10000 veces por condición. Esto
    fue necesario para que las 16 condiciones x 10000 iteraciones terminen en
    tiempo razonable; no cambia ningún valor calculado, solo cómo se calcula.
    """

    from scipy.stats import rankdata

    ranks = rankdata(y_prob_rows, axis=1, method="average")
    n_pos = y_true_rows.sum(axis=1).astype(np.float64)
    n = y_true_rows.shape[1]
    n_neg = n - n_pos
    sum_ranks_pos = (ranks * y_true_rows).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc = np.where((n_pos == 0) | (n_neg == 0), np.nan, auc)
    return auc


def condition_bootstrap_replicates(site_ctx: Mapping[str, Any], *, roi_set: str, model: str) -> np.ndarray:
    """Sección 51.1/51.2: metric-then-mean por seed para BNN; una métrica para logistic."""

    draws = site_ctx["draws"]
    y_true_full = site_ctx["y_true"]

    if model == "brainnetcnn":
        per_seed = np.empty((len(BNN_SEEDS), draws.shape[0]), dtype=np.float64)
        for s, seed in enumerate(BNN_SEEDS):
            y_prob_full = site_ctx["condition_probs"][(roi_set, "brainnetcnn", seed)]
            per_seed[s] = _auc_batch(y_true_full[draws], y_prob_full[draws])
        return per_seed.mean(axis=0)

    y_prob_full = site_ctx["condition_probs"][(roi_set, "logreg", None)]
    return _auc_batch(y_true_full[draws], y_prob_full[draws])


def percentile_ci(replicates: np.ndarray, *, ci_level: float) -> tuple[float, float]:
    alpha = (1.0 - ci_level) * 100 / 2.0
    lo, hi = np.percentile(replicates, [alpha, 100 - alpha], method="linear")
    return float(lo), float(hi)


def build_metrics_summary(
    predictions_long: pd.DataFrame, analysis_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    ci_level = float(analysis_config["ci_level"])
    rows = []
    site_contexts: dict[str, dict[str, Any]] = {}
    replicate_cache: dict[tuple[str, str, str], np.ndarray] = {}

    for site in SITES:
        ctx = compute_site_bootstrap(predictions_long, site, analysis_config)
        site_contexts[site] = ctx
        for roi_set in ROI_SETS:
            for model in MODELS:
                point, dispersion = condition_point_estimate(ctx, roi_set=roi_set, model=model)
                replicates = condition_bootstrap_replicates(ctx, roi_set=roi_set, model=model)
                replicate_cache[(site, roi_set, model)] = replicates
                lo, hi = percentile_ci(replicates, ci_level=ci_level)
                row = {
                    "held_out_site": site,
                    "roi_set": roi_set,
                    "model": model,
                    "auc_point": point,
                    "auc_ci_low": lo,
                    "auc_ci_high": hi,
                }
                if dispersion is not None:
                    row.update(dispersion)
                rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_METRICS_SUMMARY_ROWS:
        raise SystemExit(f"STOP: metrics_summary tiene {len(df)} filas; se esperaban {EXPECTED_METRICS_SUMMARY_ROWS}.")
    return df, {"contexts": site_contexts, "replicates": replicate_cache}


SECONDARY_METRIC_COLUMNS = {
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "sensitivity": "recall",  # Sección 28.2: sensitivity == recall (no existe columna "sensitivity" cruda).
    "specificity": "specificity",
}


def build_secondary_metrics(metrics_by_run: pd.DataFrame) -> pd.DataFrame:
    """CP3 (Sección 28): métricas secundarias aditivas a ``loso_metrics_summary``.

    NO toca ``auc_point``/``auc_ci_low``/``auc_ci_high``/``seed_sd``/``seed_min``/
    ``seed_max`` (Sección 28.1, intocables) — este dataframe se hace ``merge``
    aparte y solo aporta columnas nuevas. BrainNetCNN: media de las 5 métricas
    por-seed + SD con ddof=1 (Sección 28.2). Logistic: un solo valor
    determinista; su ``*_seed_sd`` es NA, nunca 0 (Sección 28.3).
    """

    rows = []
    for (site, roi_set, model), group in metrics_by_run.groupby(["held_out_site", "roi_set", "model"]):
        row: dict[str, Any] = {"held_out_site": site, "roi_set": roi_set, "model": model}
        for out_name, raw_col in SECONDARY_METRIC_COLUMNS.items():
            values = group[raw_col].to_numpy(dtype=np.float64)
            row[f"{out_name}_point"] = float(np.mean(values))
            if model == "brainnetcnn":
                row[f"{out_name}_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            else:
                row[f"{out_name}_seed_sd"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_METRICS_SUMMARY_ROWS:
        raise SystemExit(
            f"STOP: secondary metrics tiene {len(df)} filas; se esperaban {EXPECTED_METRICS_SUMMARY_ROWS}."
        )
    return df


def build_convergence_summary(runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """CP4 (Sección 29): 8 filas = 4 held-out x 2 ROI, solo BrainNetCNN.

    No infiere overfitting automáticamente (Sección 29): solo agrega epochs
    ran/best_epoch y cuenta cuántas de las 5 corridas llegaron a la epoch 300
    vs. se detuvieron antes (early stopping real).
    """

    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        c = run["config"]
        if c["model"] != "brainnetcnn":
            continue
        key = (c["held_out_site"], str(c["roi_set"]))
        rows_by_key.setdefault(key, []).append(c["convergence"])

    rows = []
    for site in SITES:
        for roi_set in ROI_SETS:
            convs = rows_by_key.get((site, roi_set), [])
            if len(convs) != 5:
                raise SystemExit(
                    f"STOP: convergence summary para {site}/{roi_set} tiene {len(convs)} corridas; se esperaban 5."
                )
            epochs_ran = np.array([c["epochs_ran"] for c in convs], dtype=np.float64)
            best_epoch = np.array([c["best_epoch"] for c in convs], dtype=np.float64)
            rows.append({
                "held_out_site": site,
                "roi_set": roi_set,
                "n_runs": len(convs),
                "n_seeds": len(convs),
                "epochs_ran_mean": float(np.mean(epochs_ran)),
                "epochs_ran_sd": float(np.std(epochs_ran, ddof=1)),
                "epochs_ran_min": float(np.min(epochs_ran)),
                "epochs_ran_max": float(np.max(epochs_ran)),
                "best_epoch_mean": float(np.mean(best_epoch)),
                "best_epoch_sd": float(np.std(best_epoch, ddof=1)),
                "best_epoch_min": float(np.min(best_epoch)),
                "best_epoch_max": float(np.max(best_epoch)),
                "n_hit_epoch_300": int(np.sum(epochs_ran == 300)),
                "n_stopped_before_300": int(np.sum(epochs_ran < 300)),
            })
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_CONVERGENCE_ROWS:
        raise SystemExit(f"STOP: convergence_summary tiene {len(df)} filas; se esperaban {EXPECTED_CONVERGENCE_ROWS}.")
    return df


CONTRAST_SPECS = (
    ("dimensionality", "brainnetcnn", "116", "brainnetcnn", "12"),
    ("model_family_at_12", "logreg", "12", "brainnetcnn", "12"),
    ("model_family_at_116", "logreg", "116", "brainnetcnn", "116"),
)


def build_contrasts(
    predictions_long: pd.DataFrame, analysis_config: Mapping[str, Any], bootstrap_state: Mapping[str, Any],
) -> pd.DataFrame:
    ci_level = float(analysis_config["ci_level"])
    contexts = bootstrap_state["contexts"]
    replicates = bootstrap_state["replicates"]

    rows = []
    for name, model_a, roi_a, model_b, roi_b in CONTRAST_SPECS:
        for site in SITES:
            ctx = contexts[site]
            point_a, _ = condition_point_estimate(ctx, roi_set=roi_a, model=model_a)
            point_b, _ = condition_point_estimate(ctx, roi_set=roi_b, model=model_b)
            rep_a = replicates[(site, roi_a, model_a)]
            rep_b = replicates[(site, roi_b, model_b)]
            diff_replicates = rep_a - rep_b  # mismo draw por construcción (Sección 51 p.8)
            lo, hi = percentile_ci(diff_replicates, ci_level=ci_level)
            rows.append(
                {
                    "contrast": name,
                    "held_out_site": site,
                    "condition_a": f"{model_a}_{roi_a}",
                    "condition_b": f"{model_b}_{roi_b}",
                    "delta_point": point_a - point_b,
                    "delta_ci_low": lo,
                    "delta_ci_high": hi,
                }
            )
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_CONTRASTS_ROWS:
        raise SystemExit(f"STOP: contrasts tiene {len(df)} filas; se esperaban {EXPECTED_CONTRASTS_ROWS}.")
    return df


def build_bootstrap_manifest(
    analysis_config: Mapping[str, Any], predictions_long_hash: str, summary_hash: str,
    *, predictions_long_path: Path | None = None, summary_path: Path | None = None,
    closeout_analysis_source_git_sha: str | None = None,
) -> dict[str, Any]:
    import sklearn

    manifest = {
        # --- Campos legacy: NO se renombran ni eliminan (Sección 34/48). ---
        "analysis_config_sha256": _sha_bytes(
            json.dumps(analysis_config, sort_keys=True).encode("utf-8")
        ),
        "analysis_source_git_sha": None,  # se completa en main() con git_info()
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "bootstrap_seed": analysis_config["bootstrap_seed"],
        "rng": analysis_config["bootstrap_rng"],
        "iterations": analysis_config["bootstrap_iterations"],
        "percentile_method": "linear",
        "subject_ordering": analysis_config["bootstrap_subject_order"],
        "input_predictions_sha256": predictions_long_hash,
        "output_summary_sha256": summary_hash,
    }
    # --- Campos nuevos, inequívocos (Sección 39/CP10): aditivos, no reemplazan
    # los legacy de arriba. ---
    manifest["analysis_config_file_sha256"] = _full_sha256_file(CONFIG_PATH)
    manifest["analysis_config_semantic_sha256"] = _semantic_sha256(dict(analysis_config))
    if predictions_long_path is not None:
        manifest["input_predictions_file_sha256"] = _full_sha256_file(predictions_long_path)
    if summary_path is not None:
        manifest["output_summary_file_sha256"] = _full_sha256_file(summary_path)
    manifest["closeout_analysis_source_git_sha"] = closeout_analysis_source_git_sha
    manifest["analyzer_file_sha256"] = _full_sha256_file(
        REPO_ROOT / "analysis" / "loso" / "scripts" / "analyze_loso_static.py"
    )
    manifest["closeout_analysis_environment"] = closeout_analysis_environment()
    manifest["bootstrap_seed"] = analysis_config["bootstrap_seed"]  # ya legacy arriba; se repite por claridad de spec
    manifest["rng"] = analysis_config["bootstrap_rng"]
    manifest["iterations"] = analysis_config["bootstrap_iterations"]
    manifest["percentile_method"] = "linear"
    manifest["subject_ordering"] = analysis_config["bootstrap_subject_order"]
    manifest["class_stratified"] = True
    manifest["paired_across_conditions"] = True
    manifest["reset_rng_per_site"] = True
    return manifest


# ---------------------------------------------------------------------------
# CP9: provenance manifest (Sección 38) y CP12: QA doc (Sección 41)
# ---------------------------------------------------------------------------


def build_provenance_manifest(
    *,
    runs: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    manifest: Mapping[str, Any],
    closeout_analysis_source_git_sha: str | None,
    original_bootstrap_manifest: Mapping[str, Any] | None,
    output_dir_for_hashing: Path,
) -> dict[str, Any]:
    """CP9: ``loso_provenance_manifest.json``. Distingue explícitamente hashes
    de archivo completos (64 hex), hashes semánticos de JSON, y prefijos
    históricos de 16 caracteres (Secciones 35-37). No fabrica full hashes para
    las FC (Sección 26/Gate Q): esas quedan como ``*_sha256_prefix16``.
    """

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import run_loso as L  # noqa: WPS433

    configs = [r["config"] for r in runs]
    n_bnn = sum(1 for c in configs if c["model"] == "brainnetcnn")
    n_log = sum(1 for c in configs if c["model"] == "logreg")

    design_bytes_sha = _full_sha256_file(L.DESIGN_JSON_PATH)
    split_bytes_sha = _full_sha256_file(L.SPLIT_MANIFEST_PATH)
    analysis_config_bytes_sha = _full_sha256_file(CONFIG_PATH)
    analysis_config_obj = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    roi_indices_full = {}
    for roi_set in ROI_SETS:
        idx = L.tdha_data.roi_indices(roi_set)
        arr = np.ascontiguousarray(np.asarray(list(idx), dtype=np.int64))
        roi_indices_full[roi_set] = hashlib.sha256(arr.tobytes()).hexdigest()

    bold_full = {site: _full_sha256_file(BOLD_DIR / f"{site}.joblib") for site in SITES}
    roi_sets_json_full = _full_sha256_file(ATLAS_DIR / "roi_sets.json")
    aal116_csv_full = _full_sha256_file(ATLAS_DIR / "aal116.csv")

    code_hashes = {
        "run_loso.py": {
            "full_sha256": _full_sha256_file(CODE_ROOT / "run_loso.py"),
            "historical_prefix16": next(iter({c["identity"]["runner_sha256"] for c in configs})),
        },
        "run_loso_campaign.py": {"full_sha256": _full_sha256_file(CODE_ROOT / "run_loso_campaign.py")},
        "data.py": {
            "full_sha256": _full_sha256_file(CODE_ROOT / "data.py"),
            "historical_prefix16": next(iter({c["identity"]["data_code_sha256"] for c in configs})),
        },
        "kerasmodels/brainnetcnn.py": {
            "full_sha256": _full_sha256_file(CODE_ROOT / "kerasmodels" / "brainnetcnn.py"),
            "historical_prefix16": next(iter(
                {c["identity"]["model_code_sha256"] for c in configs if c["model"] == "brainnetcnn"}
            )),
        },
        "analyze_loso_static.py": {
            "full_sha256": _full_sha256_file(REPO_ROOT / "analysis" / "loso" / "scripts" / "analyze_loso_static.py"),
        },
    }
    test_files = {}
    for test_path in sorted(LOSO_TEST_DIR.glob("test_*.py")):
        test_files[test_path.name] = {"full_sha256": _full_sha256_file(test_path)}

    formal_runs = []
    for run in runs:
        c = run["config"]
        run_dir = run["run_dir"]
        formal_runs.append({
            "run_id": c["run_id"],
            "identity_hash": c.get("identity_hash"),
            "config_file_sha256": _full_sha256_file(run_dir / "config.json"),
            "config_semantic_sha256": _semantic_sha256(c),
            "predictions_file_sha256": _full_sha256_file(run_dir / "predictions_test.csv"),
            "split_membership_file_sha256": _full_sha256_file(run_dir / "split_membership.csv"),
            "metrics_test_file_sha256": _full_sha256_file(run_dir / "metrics_test.csv"),
        })

    analysis_outputs = {}
    for name in sorted(output_dir_for_hashing.glob("*")):
        if name.name in ("loso_provenance_manifest.json", "LOSO_STATIC_V1_QA.md"):
            continue
        if name.is_file():
            analysis_outputs[name.name] = _full_sha256_file(name)

    original_env_partial = None
    if original_bootstrap_manifest:
        original_env_partial = {
            "numpy": original_bootstrap_manifest.get("numpy_version"),
            "sklearn": original_bootstrap_manifest.get("sklearn_version"),
            "note": "solo se registraron estas versiones en la corrida original; no se inventan las demás (Sección 33.2).",
        }

    return {
        "campaign_id": "loso_static_v1",
        "training_source_git_sha": design.get("training_source_git_sha"),
        "original_analysis_source_git_sha": (
            original_bootstrap_manifest.get("analysis_source_git_sha") if original_bootstrap_manifest else None
        ),
        "closeout_analysis_source_git_sha": closeout_analysis_source_git_sha,
        "formal_run_count": manifest["n_runs"],
        "brainnet_run_count": n_bnn,
        "logistic_run_count": n_log,
        "training_environment_signature": design.get("training_environment_signature"),
        "training_environment": configs[0].get("environment") if configs else None,
        "original_analysis_environment_partial": original_env_partial,
        "closeout_analysis_environment": closeout_analysis_environment(),
        "design": {
            "path": "results/loso/_design/loso_static_v1_design.json",
            "file_sha256": design_bytes_sha,
            "semantic_sha256": _semantic_sha256(design),
            "historical_file_sha256_prefix16": design_bytes_sha[:16],
        },
        "split_manifest": {
            "path": "results/loso/_design/loso_static_v1_splits.csv",
            "file_sha256": split_bytes_sha,
            "historical_file_sha256_prefix16": design.get("split_manifest_file_sha256"),
        },
        "analysis_config": {
            "path": "analysis/loso/config/loso_analysis_config.json",
            "file_sha256": analysis_config_bytes_sha,
            "semantic_sha256": _semantic_sha256(analysis_config_obj),
        },
        "implementation_spec": {
            "path": "analysis/loso/IMPLEMENTATION_SPEC.md",
            "file_sha256": _full_sha256_file(SPEC_PATH),
        },
        "inputs": {
            "BOLD": {
                site: {"full_sha256": bold_full[site], "historical_prefix16": design.get("input_bold_sha256", {}).get(site)}
                for site in SITES
            },
            "atlas": {
                "roi_sets.json": {"full_sha256": roi_sets_json_full, "historical_prefix16": design.get("roi_sets_json_sha256")},
                "aal116.csv": {"full_sha256": aal116_csv_full, "historical_prefix16": design.get("aal116_csv_sha256")},
            },
            "roi_indices": {
                roi_set: {
                    "full_sha256": roi_indices_full[roi_set],
                    "historical_prefix16": design.get("roi_indices_sha256", {}).get(roi_set),
                }
                for roi_set in ROI_SETS
            },
            "feature_matrices": {
                key: {"historical_prefix16_only": value, "note": "no recomputado: depende de NumPy/BLAS (Sección 26/Gate Q)."}
                for key, value in design.get("feature_matrix_sha256", {}).items()
            },
        },
        "code": code_hashes | {"loso_test_files": test_files},
        "formal_runs": formal_runs,
        "analysis_outputs": analysis_outputs,
    }


def build_qa_doc(
    *, gate_rows: Sequence[Mapping[str, Any]], provenance_manifest_file_sha256: str,
) -> str:
    """CP12 (Sección 41): tabla de gates + hash del provenance manifest."""

    lines = [
        "# LOSO_STATIC_V1_QA",
        "",
        "Tabla de auditoría de cierre (`fix/loso-static-v1-analysis-closeout`). "
        "Generada únicamente después de que los Gates A-Q (Sección 27, CP2 PASS) "
        "pasaron sobre las 48 corridas formales reales. No reentrena, no cambia "
        "splits, no cambia AUC/CI/contrastes primarios.",
        "",
        "| Gate | Description | Expected | Observed | Status |",
        "|:---|:---|:---|:---|:---|",
    ]
    for row in gate_rows:
        lines.append(
            f"| {row['gate']} | {row['description']} | {row['expected']} | {row['observed']} | {row['status']} |"
        )
    lines += [
        "",
        "Adicionalmente (verificado en CP14-CP20, fuera de esta tabla de gates "
        "por-corrida): X test suite (35 tests históricos + tests de auditoría "
        "nuevos), V raw LOSO hash protection, W historical repo hash protection — "
        "ver `git diff`/`sha256sum -c` registrados en el commit de cierre.",
        "",
        f"`loso_provenance_manifest_file_sha256`: `{provenance_manifest_file_sha256}`",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporte y main()
# ---------------------------------------------------------------------------


COHORT_TABLE = [
    ("NYU", 177, 87, 90),
    ("Peking", 183, 109, 74),
    ("NeuroIMAGE", 39, 22, 17),
    ("OHSU", 66, 38, 28),
]


def build_report(
    *,
    manifest: Mapping[str, Any],
    metrics_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    convergence_summary: pd.DataFrame,
    design: Mapping[str, Any],
    training_environment: Mapping[str, Any] | None,
    original_analysis_source_git_sha: str | None,
    closeout_analysis_source_git_sha: str | None,
    closeout_env: Mapping[str, Any],
) -> str:
    """CP11 (Sección 40): reporte autocontenido — no depende de ningún plan
    externo no versionado (Sección 1 punto 6: referencia rota eliminada). El
    diseño completo vive en ``analysis/loso/IMPLEMENTATION_SPEC.md``,
    versionado dentro de este mismo repositorio."""

    cohort_lines = ["| Site | n | Control | ADHD |", "|---|---:|---:|---:|"]
    for site, n, control, adhd in COHORT_TABLE:
        cohort_lines.append(f"| {site} | {n} | {control} | {adhd} |")
    total_n = sum(r[1] for r in COHORT_TABLE)
    total_c = sum(r[2] for r in COHORT_TABLE)
    total_a = sum(r[3] for r in COHORT_TABLE)
    cohort_lines.append(f"| **Total** | **{total_n}** | **{total_c}** | **{total_a}** |")

    split_lines = ["| Held-out | Fit | Inner val | Test |", "|---|---:|---:|---:|"]
    for site in SITES:
        sizes = design.get("rotation_sizes", {}).get(site, {})
        split_lines.append(f"| {site} | {sizes.get('fit')} | {sizes.get('inner_val')} | {sizes.get('test')} |")

    training_env_lines = ["(no disponible)"]
    if training_environment:
        training_env_lines = [f"- {k}: {v}" for k, v in training_environment.items()]

    lines = [
        "# LOSO_STATIC_V1_REPORT",
        "",
        "Campaña `loso_static_v1`: 4 sitios held-out x 2 ROI sets x 2 familias de "
        "modelo, conectividad estática, sin harmonización ni ponderación de "
        "clase/sitio. Diseño completo autocontenido en "
        "`analysis/loso/IMPLEMENTATION_SPEC.md` (dentro de este repositorio).",
        "",
        "## Diseño",
        "",
        "Una rotación por sitio held-out (LOSO exhaustivo, 4 rotaciones); "
        "entrenamiento con los otros tres sitios; ROI 12 y 116; BrainNetCNN "
        "(5 seeds) + regresión logística L2 (determinista); FC estática, sin "
        "Fisher-z; sin harmonización; sin ponderación de clase/sitio/muestra.",
        "",
        "## Cohorte",
        "",
        *cohort_lines,
        "",
        "## Splits",
        "",
        *split_lines,
        "",
        "Estratificación inner: sitio x diagnóstico. `split_seed = 42`. Mismo "
        "split entre los dos ROI sets, los 5 seeds de BrainNetCNN y la "
        "regresión logística dentro de cada rotación.",
        "",
        "## Environments y source SHAs",
        "",
        "### Training",
        "",
        f"- `training_source_git_sha`: `{design.get('training_source_git_sha')}`",
        f"- `training_environment_signature`: `{design.get('training_environment_signature')}`",
        "- Training environment:",
        *[f"  {line}" for line in training_env_lines],
        "",
        "### Original analysis",
        "",
        f"- `original_analysis_source_git_sha`: `{original_analysis_source_git_sha}`",
        "- Solo se registraron `numpy`/`scikit-learn` de este environment original "
        "(ver `loso_bootstrap_manifest.json` -> `original_analysis_environment_partial` "
        "en `loso_provenance_manifest.json`); no se inventan las demás versiones.",
        "",
        "### Closeout analysis",
        "",
        f"- `closeout_analysis_source_git_sha`: `{closeout_analysis_source_git_sha}`",
        "- Closeout analysis environment:",
        *[f"  - {k}: {v}" for k, v in closeout_env.items()],
        "",
        "Training y analysis environments se registran por separado; no se exige "
        "que sean idénticos (Sección 33.3) — la validez del rerun del analyzer se "
        "decide por el primary-result regression gate (Sección 45), no por "
        "identidad de environments.",
        "",
        f"## Completitud: {manifest['n_runs']}/48 corridas formales",
        "",
        "40 BrainNetCNN + 8 logistic = 48 total, 5580 predicciones, 0 faltantes, "
        "0 duplicadas, 0 parciales (verificado en Gates A-Q antes de calcular "
        "cualquier output; ver `LOSO_STATIC_V1_QA.md`).",
        "",
        "## AUC por condición (95% CI, percentil, sin ajustar) — primario, sin cambios",
        "",
        metrics_summary[[
            "held_out_site", "roi_set", "model", "auc_point", "auc_ci_low", "auc_ci_high",
            "seed_sd", "seed_min", "seed_max",
        ]].to_markdown(index=False),
        "",
        "## Métricas secundarias (threshold=0.5; BNN: media de 5 seeds + SD; "
        "logistic: valor único determinista; sin CI adicionales)",
        "",
        metrics_summary[[
            "held_out_site", "roi_set", "model",
            "balanced_accuracy_point", "balanced_accuracy_seed_sd",
            "f1_macro_point", "f1_macro_seed_sd",
            "sensitivity_point", "sensitivity_seed_sd",
            "specificity_point", "specificity_seed_sd",
        ]].to_markdown(index=False),
        "",
        "`sensitivity` es sinónimo de `recall` (no existe columna `sensitivity` "
        "cruda en `metrics_test.csv`).",
        "",
        "## Variabilidad entre seeds (BrainNetCNN, AUC)",
        "",
        metrics_summary.loc[metrics_summary["model"] == "brainnetcnn", [
            "held_out_site", "roi_set", "auc_point", "seed_sd", "seed_min", "seed_max",
        ]].to_markdown(index=False),
        "",
        "## Contrastes preespecificados (95% CI) — primario, sin cambios, sin p-values",
        "",
        contrasts.to_markdown(index=False),
        "",
        "## Convergencia (BrainNetCNN)",
        "",
        convergence_summary.to_markdown(index=False),
        "",
        "No se infiere overfitting automáticamente a partir de estos valores "
        "(Sección 29/40.10).",
        "",
        "## QA",
        "",
        "Ver `LOSO_STATIC_V1_QA.md` para la tabla completa de gates (A-X) tras "
        "auditoría PASS. Resumen: 48/48 config PASS, 48/48 split PASS, 48/48 "
        "prediction schema PASS, 48/48 métricas reproducidas de forma "
        "independiente, 40/40 convergencia BNN PASS, 8/8 configuración "
        "logistic PASS, 5580/5580 predicciones completas, hashes de campaña "
        "cruda y del repositorio histórico intactos.",
        "",
        "## Terminología",
        "",
        "cross-site transport / LOSO cross-site evaluation / internal-external "
        "validation-style analysis / held-out site / transportability. Evitar sin "
        "matiz: external validation, independent validation, generalizes across "
        "sites, clinically validated, robust biomarker.",
        "",
        "## Caveat NYU",
        "",
        "La configuración de BrainNetCNN se desarrolló/fijó históricamente usando "
        "NYU antes de la evaluación multisitio. La rotación con NYU held-out es "
        "'development-site held-out re-evaluation within the LOSO campaign', no "
        "una evaluación en un sitio totalmente ajeno al desarrollo del modelo.",
        "",
        "## No harmonización / no ponderación",
        "",
        "class_weight=false, site_weighting=false, sample_weight=false, "
        "harmonization=none en las 48 corridas formales.",
        "",
    ]
    return "\n".join(lines)


def assert_primary_outputs_unchanged(
    *, baseline: pd.DataFrame, candidate: pd.DataFrame, key_cols: Sequence[str],
    value_cols: Sequence[str], tol: float = 1e-9, label: str = "primary output",
) -> None:
    """CP17-CP20/Sección 45: regression gate obligatorio antes de promover.

    Compara ``candidate`` (recién calculado en staging) contra ``baseline``
    (congelado antes de tocar el analyzer) por columnas clave. Si cualquier
    valor primario (AUC/CI/contraste) cambió más allá de ``tol``, o si el
    conjunto de llaves difiere, levanta ``SystemExit`` y NO promueve — la
    Sección 45 es explícita: "Si cualquier primary AUC/CI/contraste cambia:
    STOP. No promover."
    """

    # NaN en una columna llave (p. ej. model_seed=NaN para logistic) rompe la
    # igualdad de tuplas de índice aunque ambos lados tengan el mismo NaN;
    # se reemplaza por un centinela solo para la comparación de llaves.
    sentinel = "__NA_KEY_SENTINEL__"
    b_keys = baseline[list(key_cols)].fillna(sentinel)
    c_keys = candidate[list(key_cols)].fillna(sentinel)
    b = baseline.copy(); b[list(key_cols)] = b_keys
    c = candidate.copy(); c[list(key_cols)] = c_keys
    b = b.set_index(list(key_cols)).sort_index()
    c = c.set_index(list(key_cols)).sort_index()
    if list(b.index) != list(c.index):
        raise SystemExit(f"STOP: {label} — el conjunto de llaves difiere entre baseline y candidate.")
    for col in value_cols:
        diff = (b[col].astype(float) - c[col].astype(float)).abs()
        if (diff > tol).any():
            bad = diff[diff > tol].to_dict()
            raise SystemExit(f"STOP: {label} cambió en columna '{col}' más allá de tol={tol}: {bad}")


FINAL_OUTPUT_NAMES = (
    "loso_manifest.json", "loso_predictions_long.csv", "loso_metrics_by_run.csv",
    "loso_metrics_summary.csv", "loso_contrasts.csv", "loso_convergence_summary.csv",
    "loso_bootstrap_manifest.json", "LOSO_STATIC_V1_REPORT.md",
    "loso_provenance_manifest.json", "LOSO_STATIC_V1_QA.md",
)


def main(argv: Sequence[str] | None = None) -> int:
    del argv

    # closeout_analysis_source_git_sha se captura ANTES de crear cualquier
    # directorio de staging bajo OUTPUT_DIR, para no ensuciar "git status"
    # con el propio directorio temporal de este proceso (Sección 31: el
    # código de cierre debe estar commiteado ANTES de correr esto).
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from run_experiment import git_info

    git = git_info()
    closeout_analysis_source_git_sha = git.get("commit")
    if git.get("clean") is False:
        print(
            "AVISO: working tree no está limpio al capturar "
            "closeout_analysis_source_git_sha (puede incluir outputs previos "
            "de analysis/loso/outputs/ aún no commiteados); el SHA registrado "
            "corresponde de todas formas al código fuente actualmente en HEAD.",
            file=sys.stderr,
        )

    # Provenance de la corrida ORIGINAL de análisis: se lee del bootstrap
    # manifest actual (previo a esta corrida de cierre) ANTES de que este
    # main() lo sobrescriba, para no perder ese dato histórico.
    original_bootstrap_manifest = None
    existing_bootstrap_path = OUTPUT_DIR / "loso_bootstrap_manifest.json"
    if existing_bootstrap_path.exists():
        original_bootstrap_manifest = json.loads(existing_bootstrap_path.read_text(encoding="utf-8"))

    analysis_config = load_analysis_config()
    design, splits_df = load_design_and_splits()
    runs = discover_runs()

    # CP2/Gates A-Q: auditoría fail-fast ANTES de calcular ningún output.
    gate_rows = run_closeout_audit(runs, design, splits_df)

    manifest = build_manifest(runs)
    predictions_long = build_predictions_long(runs)
    metrics_by_run = build_metrics_by_run(runs)
    metrics_summary, bootstrap_state = build_metrics_summary(predictions_long, analysis_config)
    contrasts = build_contrasts(predictions_long, analysis_config, bootstrap_state)
    convergence_summary = build_convergence_summary(runs)

    # CP3: métricas secundarias, aditivas — no tocan auc_point/ci/seed_*.
    secondary = build_secondary_metrics(metrics_by_run)
    metrics_summary = metrics_summary.merge(secondary, on=["held_out_site", "roi_set", "model"], how="left")
    if len(metrics_summary) != EXPECTED_METRICS_SUMMARY_ROWS:
        raise SystemExit(
            f"STOP: metrics_summary con secundarias tiene {len(metrics_summary)} filas; "
            f"se esperaban {EXPECTED_METRICS_SUMMARY_ROWS}."
        )

    training_environment = runs[0]["config"].get("environment") if runs else None
    closeout_env = closeout_analysis_environment()

    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(OUTPUT_DIR)))
    try:
        # Orden de generación (Sección 49): CSVs núcleo -> manifest -> bootstrap
        # manifest -> convergence -> report -> validar -> provenance -> QA.
        (staging_dir / "loso_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        predictions_long.to_csv(staging_dir / "loso_predictions_long.csv", index=False)
        metrics_by_run.to_csv(staging_dir / "loso_metrics_by_run.csv", index=False)
        metrics_summary.to_csv(staging_dir / "loso_metrics_summary.csv", index=False)
        contrasts.to_csv(staging_dir / "loso_contrasts.csv", index=False)
        convergence_summary.to_csv(staging_dir / "loso_convergence_summary.csv", index=False)

        predictions_hash = _sha_file(staging_dir / "loso_predictions_long.csv")
        summary_hash = _sha_file(staging_dir / "loso_metrics_summary.csv")
        bootstrap_manifest = build_bootstrap_manifest(
            analysis_config, predictions_hash, summary_hash,
            predictions_long_path=staging_dir / "loso_predictions_long.csv",
            summary_path=staging_dir / "loso_metrics_summary.csv",
            closeout_analysis_source_git_sha=closeout_analysis_source_git_sha,
        )
        bootstrap_manifest["analysis_source_git_sha"] = closeout_analysis_source_git_sha
        (staging_dir / "loso_bootstrap_manifest.json").write_text(
            json.dumps(bootstrap_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        report = build_report(
            manifest=manifest, metrics_summary=metrics_summary, contrasts=contrasts,
            convergence_summary=convergence_summary, design=design,
            training_environment=training_environment,
            original_analysis_source_git_sha=(
                original_bootstrap_manifest.get("analysis_source_git_sha") if original_bootstrap_manifest else None
            ),
            closeout_analysis_source_git_sha=closeout_analysis_source_git_sha,
            closeout_env=closeout_env,
        )
        (staging_dir / "LOSO_STATIC_V1_REPORT.md").write_text(report, encoding="utf-8")

        # Validar todo lo generado hasta ahora antes de construir provenance/QA.
        for name in (
            "loso_manifest.json", "loso_predictions_long.csv", "loso_metrics_by_run.csv",
            "loso_metrics_summary.csv", "loso_contrasts.csv", "loso_convergence_summary.csv",
            "loso_bootstrap_manifest.json", "LOSO_STATIC_V1_REPORT.md",
        ):
            if not (staging_dir / name).exists():
                raise SystemExit(f"STOP: {name} no se generó correctamente en staging.")

        provenance_manifest = build_provenance_manifest(
            runs=runs, design=design, manifest=manifest,
            closeout_analysis_source_git_sha=closeout_analysis_source_git_sha,
            original_bootstrap_manifest=original_bootstrap_manifest,
            output_dir_for_hashing=staging_dir,
        )
        (staging_dir / "loso_provenance_manifest.json").write_text(
            json.dumps(provenance_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        provenance_file_sha256 = _full_sha256_file(staging_dir / "loso_provenance_manifest.json")

        qa_doc = build_qa_doc(gate_rows=gate_rows, provenance_manifest_file_sha256=provenance_file_sha256)
        (staging_dir / "LOSO_STATIC_V1_QA.md").write_text(qa_doc, encoding="utf-8")

        # Promoción con backup temporal (Sección 50): si falla, se restaura el
        # estado anterior completo y no se deja un conjunto parcial.
        backups: dict[str, Path] = {}
        try:
            for name in FINAL_OUTPUT_NAMES:
                final_path = OUTPUT_DIR / name
                if final_path.exists():
                    backup_path = staging_dir / f".backup-{name}"
                    shutil.copy2(final_path, backup_path)
                    backups[name] = backup_path
            for name in FINAL_OUTPUT_NAMES:
                final_path = OUTPUT_DIR / name
                staged_path = staging_dir / name
                if final_path.exists():
                    final_path.unlink()
                shutil.copy2(staged_path, final_path)
        except Exception:
            for name, backup_path in backups.items():
                shutil.copy2(backup_path, OUTPUT_DIR / name)
            raise SystemExit("STOP: falló la promoción de outputs; se restauró el estado anterior completo.")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Analysis outputs escritos en {OUTPUT_DIR}")
    print(f"closeout_analysis_source_git_sha={closeout_analysis_source_git_sha}")
    for row in gate_rows:
        print(f"  Gate {row['gate']}: {row['status']} — {row['observed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
