#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recalcula, a 10,000 remuestreos, los 26 contrastes de sensibilidad citados
en Manuscript_Methods_Results_Rebuilt (representacion estatica: 4 sitios;
arquitectura LSTM-128: NYU/Peking; enventanado 140s/12s y 120s/24s:
NYU/Peking; y los 16 contrastes del baseline de regresion logistica: 4
sitios x 4 paneles de ROI).

Metodologia: bootstrap pareado por sujeto, estratificado por clase, generador
NumPy PCG64, semilla 42 reiniciada por contraste, metodo percentil bilateral
2.5/97.5. Identica especificacion que build_baseline_contrast.py, salvo
n_iter=10000 en lugar de 2000.

Calculo de AUC (funcion batch_auc_mannwhitney): formula de Mann-Whitney U
vectorizada -- para cada iteracion de remuestreo, cuenta pares (positivo,
negativo) con score_pos > score_neg como 1 y score_pos == score_neg como 0.5,
dividido por n_pos*n_neg. Esta es la definicion misma de la AUC (equivalente
al area bajo la curva ROC por trapecios que usa sklearn.roc_auc_score) e
incluye credito de 0.5 en empates por construccion, sin depender de que la
implementacion evite necesariamente casos empatados.

Se prefirio esta formulacion sobre una basada en rangos (np.argsort) porque
esta ultima, verificada en la ronda anterior (v3), NO asigna rango promedio a
valores empatados y por tanto puede sesgar el resultado si dos sujetos de
distinta clase tienen exactamente el mismo score dentro de una iteracion de
remuestreo. La formulacion Mann-Whitney no tiene ese problema: fue verificada
contra sklearn.metrics.roc_auc_score con probabilidades sinteticas
fuertemente empatadas (diferencia maxima 1.11e-16, muy por debajo de la
tolerancia exigida de 1e-12) y contra una muestra de remuestreos reales
tomados de los datos del proyecto (diferencia maxima 2.22e-16). Ver
recalculate_manuscript_bootstrap_10k_tie_tests.json para la evidencia cruda
de ambas pruebas.

No reentrena ningun modelo: solo remuestrea predictions_val.csv ya
existentes en resultados/runs/12.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import hashlib
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.1"  # 1.0.1: fixed RNG draw order (interleaved per-iteration
                          # control-then-adhd, matching build_baseline_contrast.py)
                          # after reconciliation against v3 found CI-limit drift
                          # caused by a batched draw order in 1.0.0.
N_ITER = 10000
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]  # analysis/roi_comparison/scripts -> repo root
OUT_TABLES = SCRIPT_DIR.parents[0] / "outputs" / "tables"
OUT_LOGS = SCRIPT_DIR.parents[0] / "outputs" / "logs"

R = "results/runs/12"
REF = {
    "NYU": f"{R}/NYU_rois12_w60s6_brainnetcnn_control_baseline_v13_3e220e5c",
    "Peking": f"{R}/Peking_rois12_w60s6_brainnetcnn_control_baseline_v13_bc841110",
    "NeuroIMAGE": f"{R}/NeuroIMAGE_rois12_w61s6_brainnetcnn_control_baseline_v13_2b729a8c",
    "OHSU": f"{R}/OHSU_rois12_w48s5_brainnetcnn_control_baseline_v13_1a7c37ce",
}

TEN_CONTRASTS = {
    ("NYU", "static"): (REF["NYU"], f"{R}/NYU_rois12_static_brainnetcnn_rev32_static_r12_f78fe9f1"),
    ("Peking", "static"): (REF["Peking"], f"{R}/Peking_rois12_static_brainnetcnn_rev32_lstm128_ordered_61d3f2af"),
    ("NeuroIMAGE", "static"): (REF["NeuroIMAGE"], f"{R}/NeuroIMAGE_rois12_static_brainnetcnn_rev32_static_r12_2dcba12c"),
    ("OHSU", "static"): (REF["OHSU"], f"{R}/OHSU_rois12_static_brainnetcnn_rev32_lstm128_ordered_85129048"),
    ("NYU", "lstm128"): (REF["NYU"], f"{R}/NYU_rois12_w60s6_lstm_rev32_lstm128_ordered_0209fc13"),
    ("Peking", "lstm128"): (REF["Peking"], f"{R}/Peking_rois12_w60s6_lstm_rev32_lstm128_ordered_38b74bdf"),
    ("NYU", "window140_12"): (REF["NYU"], f"{R}/NYU_rois12_w70s6_brainnetcnn_rev32_window140_step12_d236544e"),
    ("Peking", "window140_12"): (REF["Peking"], f"{R}/Peking_rois12_w70s6_brainnetcnn_rev32_window140_step12_24b0ab4f"),
    ("NYU", "window120_24"): (REF["NYU"], f"{R}/NYU_rois12_w60s12_brainnetcnn_rev32_window120_step24_84fa43da"),
    ("Peking", "window120_24"): (REF["Peking"], f"{R}/Peking_rois12_w60s12_brainnetcnn_rev32_window120_step24_43775a9b"),
}


def load_baseline_contrasts():
    manifest = pd.read_csv(REPO / "analysis/roi_comparison/config/baseline_manifest.csv", dtype=str)
    manifest["roi_set"] = manifest["roi_set"].astype(int)
    out = {}
    for _, row in manifest.iterrows():
        key = ("baseline", row["site"], f"roi{row['roi_set']}")
        out[key] = (str(REPO / row["comparator_relative_path"]), str(REPO / row["baseline_relative_path"]),
                     row["comparator_representation"], row["representation_confound"] == "True")
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(run_dir: str):
    p = Path(run_dir)
    if not p.is_absolute():
        p = REPO / run_dir
    return pd.read_csv(p / "predictions_val.csv", dtype={"subject_id": str}), p


def build_subject_tensor(preds: pd.DataFrame):
    ref = preds[preds["repeat"] == 1].sort_values("subject_id", key=lambda s: s.astype(str))
    subject_ids = ref["subject_id"].tolist()
    y_true = ref["y_true"].to_numpy()
    probs = np.empty((len(subject_ids), 5), dtype=np.float64)
    for r in range(1, 6):
        sub = preds[preds["repeat"] == r].set_index("subject_id").loc[subject_ids]
        if not np.array_equal(sub["y_true"].to_numpy(), y_true):
            raise SystemExit("y_true inconsistente entre repeticiones")
        probs[:, r - 1] = sub["y_prob"].to_numpy()
    return probs, y_true, subject_ids


def per_repeat_auc(preds: pd.DataFrame):
    from sklearn.metrics import roc_auc_score
    reps = []
    for r in range(1, 6):
        sub = preds[preds["repeat"] == r]
        reps.append(float(roc_auc_score(sub["y_true"].to_numpy(), sub["y_prob"].to_numpy())))
    return reps


def batch_auc_mannwhitney(pos_mat: np.ndarray, neg_mat: np.ndarray, batch: int = 2000) -> np.ndarray:
    """pos_mat: (n_iter, n_pos) scores of resampled positive-class subjects.
    neg_mat: (n_iter, n_neg) scores of resampled negative-class subjects.
    Returns AUC per iteration via the Mann-Whitney U definition, with 0.5
    credit for exact ties -- correct by construction, verified against
    sklearn.metrics.roc_auc_score (see module docstring and
    recalculate_manuscript_bootstrap_10k_tie_tests.json)."""
    n_iter = pos_mat.shape[0]
    n_pos = pos_mat.shape[1]
    n_neg = neg_mat.shape[1]
    out = np.empty(n_iter, dtype=np.float64)
    for s in range(0, n_iter, batch):
        e = min(s + batch, n_iter)
        p = pos_mat[s:e][:, :, None]
        q = neg_mat[s:e][:, None, :]
        gt = (p > q).sum(axis=(1, 2))
        eq = (p == q).sum(axis=(1, 2))
        out[s:e] = (gt + 0.5 * eq) / (n_pos * n_neg)
    return out


def bilateral_ci(draws):
    lo, hi = np.quantile(draws, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def bootstrap_pair_10k(ref_dir, new_dir, seed=SEED, n_iter=N_ITER):
    ref_preds, ref_path = load_predictions(ref_dir)
    new_preds, new_path = load_predictions(new_dir)
    ref_probs, ref_y, ref_ids = build_subject_tensor(ref_preds)
    new_probs, new_y, new_ids = build_subject_tensor(new_preds)
    if ref_ids != new_ids:
        raise SystemExit(f"ERROR sujetos distintos: {ref_dir} vs {new_dir}")
    if not np.array_equal(ref_y, new_y):
        raise SystemExit(f"ERROR y_true distinto: {ref_dir} vs {new_dir}")
    y_true = ref_y
    n_subj = len(ref_ids)

    rng = np.random.default_rng(seed)
    control_idx = np.flatnonzero(y_true == 0)
    adhd_idx = np.flatnonzero(y_true == 1)
    n_control, n_adhd = len(control_idx), len(adhd_idx)

    # Bootstrap indices resampled SEPARATELY per class (stratified), kept
    # separate (not concatenated) so the Mann-Whitney formula can consume
    # them directly without needing a combined y_true per iteration.
    #
    # IMPORTANT: drawn with a per-iteration loop (control draw, then adhd
    # draw, repeated n_iter times) rather than two batched calls
    # (rng.integers(..., size=(n_iter, n))). Both consume the same seed and
    # generator, but a batched call consumes the underlying PCG64 stream in
    # a different order (all control draws for all iterations, then all
    # adhd draws for all iterations) than the interleaved per-iteration loop
    # used by build_baseline_contrast.py and by the original v3 script
    # (bootstrap10k_v2.py). That difference was caught by the mandatory
    # reconciliation step (section 3.4 of the v4 instructions): a first
    # version of this script used the batched form and reproduced the exact
    # same point estimates as v3 (as expected, since those don't depend on
    # resampling) but different confidence-interval limits, including a
    # changed displayed value for NYU window140_12. The loop below restores
    # bit-for-bit the same draw order as v3/build_baseline_contrast.py.
    boot_control = np.empty((n_iter, n_control), dtype=np.int64)
    boot_adhd = np.empty((n_iter, n_adhd), dtype=np.int64)
    for it in range(n_iter):
        boot_control[it] = control_idx[rng.integers(0, n_control, size=n_control)]
        boot_adhd[it] = adhd_idx[rng.integers(0, n_adhd, size=n_adhd)]

    ref_rep = np.empty((n_iter, 5))
    new_rep = np.empty((n_iter, 5))
    for r in range(5):
        ref_pos = ref_probs[boot_adhd, r]
        ref_neg = ref_probs[boot_control, r]
        new_pos = new_probs[boot_adhd, r]
        new_neg = new_probs[boot_control, r]
        ref_rep[:, r] = batch_auc_mannwhitney(ref_pos, ref_neg)
        new_rep[:, r] = batch_auc_mannwhitney(new_pos, new_neg)
    ref_draws = ref_rep.mean(axis=1)
    new_draws = new_rep.mean(axis=1)
    delta_draws = new_draws - ref_draws

    ref_reps_point = per_repeat_auc(ref_preds)
    new_reps_point = per_repeat_auc(new_preds)
    ref_point = float(np.mean(ref_reps_point))
    new_point = float(np.mean(new_reps_point))
    delta_point = new_point - ref_point

    return {
        "ref_dir": str(ref_dir), "new_dir": str(new_dir),
        "ref_predictions_sha256": sha256_file(ref_path / "predictions_val.csv"),
        "new_predictions_sha256": sha256_file(new_path / "predictions_val.csv"),
        "ref_config_sha256": sha256_file(ref_path / "config.json"),
        "new_config_sha256": sha256_file(new_path / "config.json"),
        "n_subjects": n_subj, "n_control": int(n_control), "n_adhd": int(n_adhd),
        "n_iter": n_iter, "seed": seed, "generator": "PCG64",
        "resampling": "paired_by_subject_stratified_by_class",
        "ci_method": "percentile_2.5_97.5",
        "ref_auc_point": ref_point, "ref_ci": list(bilateral_ci(ref_draws)), "ref_reps": ref_reps_point,
        "new_auc_point": new_point, "new_ci": list(bilateral_ci(new_draws)), "new_reps": new_reps_point,
        "delta": delta_point, "delta_ci": list(bilateral_ci(delta_draws)),
    }


CACHE_PATH = OUT_LOGS / ".manuscript_bootstrap_10k_partial_cache.json"


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(results):
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Procesar solo contrastes cuya clave contenga este substring")
    ap.add_argument("--finalize", action="store_true", help="Escribir CSV y manifiesto finales a partir de la cache acumulada (requiere 26/26)")
    args = ap.parse_args()

    t0 = time.time()
    results = load_cache()
    print(f"Cache cargada: {len(results)}/26 contrastes ya calculados en llamadas previas.")

    all_contrasts = {}
    for key, (ref_dir, new_dir) in TEN_CONTRASTS.items():
        all_contrasts["__".join(key)] = (ref_dir, new_dir, None, None)
    for key, (ref_dir, new_dir, comparator_repr, confound) in load_baseline_contrasts().items():
        all_contrasts["__".join(key)] = (ref_dir, new_dir, comparator_repr, confound)

    if not args.finalize:
        pending = [k for k in all_contrasts if k not in results]
        if args.only:
            pending = [k for k in pending if args.only in k]
        for cid in pending:
            if time.time() - t0 > 38:
                print(f"Presupuesto de tiempo agotado; {len(pending) - pending.index(cid)} contrastes quedan pendientes en esta llamada.")
                break
            ref_dir, new_dir, comparator_repr, confound = all_contrasts[cid]
            r = bootstrap_pair_10k(ref_dir, new_dir)
            if comparator_repr is not None:
                r["comparator_representation"] = comparator_repr
                r["representation_confound"] = confound
            results[cid] = r
            save_cache(results)
            print(f"{cid}: ref={r['ref_auc_point']:.4f} new={r['new_auc_point']:.4f} "
                  f"delta={r['delta']:+.4f} CI={r['delta_ci']}  ({time.time()-t0:.1f}s elapsed, "
                  f"{len(results)}/26 total)")
        print(f"\nEstado: {len(results)}/26 contrastes calculados y guardados en cache ({CACHE_PATH}).")
        if len(results) < len(all_contrasts):
            print("INCOMPLETE_RESUME_NEEDED")
            return
        print("ALL_COMPLETE -- ejecutar con --finalize para escribir CSV y manifiesto finales.")
        return

    missing = [k for k in all_contrasts if k not in results]
    if missing:
        raise SystemExit(f"No se puede finalizar: faltan {len(missing)} contrastes: {missing}")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)

    # tabla CSV plana para consumo trazable
    rows = []
    for cid, r in results.items():
        rows.append({
            "contrast_id": cid,
            "ref_dir": r["ref_dir"], "new_dir": r["new_dir"],
            "n_subjects": r["n_subjects"], "n_control": r["n_control"], "n_adhd": r["n_adhd"],
            "n_iter": r["n_iter"], "seed": r["seed"], "generator": r["generator"],
            "ref_auc_point": r["ref_auc_point"], "ref_ci_low": r["ref_ci"][0], "ref_ci_high": r["ref_ci"][1],
            "new_auc_point": r["new_auc_point"], "new_ci_low": r["new_ci"][0], "new_ci_high": r["new_ci"][1],
            "delta": r["delta"], "delta_ci_low": r["delta_ci"][0], "delta_ci_high": r["delta_ci"][1],
            "ref_predictions_sha256": r["ref_predictions_sha256"], "new_predictions_sha256": r["new_predictions_sha256"],
            "ref_config_sha256": r["ref_config_sha256"], "new_config_sha256": r["new_config_sha256"],
        })
    df = pd.DataFrame(rows).sort_values("contrast_id")
    csv_path = OUT_TABLES / "manuscript_bootstrap_10k.csv"
    df.to_csv(csv_path, index=False)

    manifest_path = OUT_LOGS / "manuscript_bootstrap_10k_manifest.json"
    manifest = {
        "script_version": SCRIPT_VERSION,
        "script_path": "analysis/roi_comparison/scripts/recalculate_manuscript_bootstrap_10k.py",
        "executed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_contrasts": len(results),
        "n_iter": N_ITER, "seed": SEED, "generator": "PCG64",
        "resampling": "paired_by_subject_stratified_by_class",
        "ci_method": "percentile_2.5_97.5",
        "auc_formula": "mann_whitney_u_vectorized_tie_aware",
        "retraining": False,
        "elapsed_seconds": round(time.time() - t0, 1),
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"\nEscrito: {csv_path}")
    print(f"Escrito: {manifest_path}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print(f"Contrastes calculados: {len(results)} (esperado 26)")


if __name__ == "__main__":
    main()
