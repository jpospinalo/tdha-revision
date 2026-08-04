#!/usr/bin/env python3
"""Genera la evidencia de las pruebas de empates exigidas por las
instrucciones v4 (seccion 3.3): (1) prueba sintetica de empates fuertes
comparando la formula previa basada en rangos (v3, incorrecta para empates)
y la formula Mann-Whitney (v4, correcta) contra sklearn.roc_auc_score; (2)
prueba sobre una muestra real de remuestreos tomados de datos del proyecto;
(3) barrido exhaustivo de las 42 corridas involucradas en los 26 contrastes
buscando empates cruzados de clase (score positivo == score negativo) en los
datos originales, para siete todas las repeticiones."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

REPO = Path("/sessions/laughing-wizardly-archimedes/mnt/tdha-revision")


def batch_auc_rank_v3(y_mat, score_mat):
    order = np.argsort(score_mat, axis=1)
    y_sorted = np.take_along_axis(y_mat, order, axis=1)
    n = score_mat.shape[1]
    ranks = np.arange(1, n + 1)
    n_pos = y_sorted.sum(axis=1)
    n_neg = n - n_pos
    sum_ranks_pos = (y_sorted * ranks[None, :]).sum(axis=1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def batch_auc_mannwhitney_v4(pos_mat, neg_mat):
    p = pos_mat[:, :, None]
    q = neg_mat[:, None, :]
    gt = (p > q).sum(axis=(1, 2))
    eq = (p == q).sum(axis=(1, 2))
    n_pos = pos_mat.shape[1]
    n_neg = neg_mat.shape[1]
    return (gt + 0.5 * eq) / (n_pos * n_neg)


results = {}

# --- Test 1: empate sintetico simple ---
y = np.array([0, 0, 1, 1, 0, 1])
scores = np.array([0.3, 0.5, 0.5, 0.5, 0.2, 0.9])
sk = float(roc_auc_score(y, scores))
v3 = float(batch_auc_rank_v3(y[None, :], scores[None, :])[0])
pos = scores[y == 1][None, :]
neg = scores[y == 0][None, :]
v4 = float(batch_auc_mannwhitney_v4(pos, neg)[0])
results["test1_synthetic_simple_tie"] = {
    "y": y.tolist(), "scores": scores.tolist(),
    "sklearn_auc": sk, "v3_rank_based_auc": v3, "v4_mannwhitney_auc": v4,
    "v3_diff_vs_sklearn": abs(sk - v3), "v4_diff_vs_sklearn": abs(sk - v4),
    "v3_within_1e-12": abs(sk - v3) < 1e-12, "v4_within_1e-12": abs(sk - v4) < 1e-12,
}

# --- Test 1b: empates adversariales (5 sitios distintos, muchos empates) ---
rng = np.random.default_rng(0)
adversarial = []
for trial in range(20):
    n = 40
    y_t = rng.integers(0, 2, size=n)
    if y_t.sum() == 0 or y_t.sum() == n:
        continue
    sc = rng.integers(0, 5, size=n).astype(float)
    sk_t = float(roc_auc_score(y_t, sc))
    v3_t = float(batch_auc_rank_v3(y_t[None, :], sc[None, :])[0])
    pos_t = sc[y_t == 1][None, :]; neg_t = sc[y_t == 0][None, :]
    v4_t = float(batch_auc_mannwhitney_v4(pos_t, neg_t)[0])
    adversarial.append({
        "trial": trial, "sklearn_auc": sk_t, "v3_diff": abs(sk_t - v3_t), "v4_diff": abs(sk_t - v4_t),
    })
results["test1b_adversarial_heavy_ties_20_trials"] = {
    "n_trials": len(adversarial),
    "max_v3_diff": max(d["v3_diff"] for d in adversarial),
    "max_v4_diff": max(d["v4_diff"] for d in adversarial),
    "n_v3_within_1e-12": sum(1 for d in adversarial if d["v3_diff"] < 1e-12),
    "n_v4_within_1e-12": sum(1 for d in adversarial if d["v4_diff"] < 1e-12),
    "detail": adversarial,
}

# --- Test 2: remuestreos reales (bootstrap con reemplazo genera duplicados intra-clase) ---
run = REPO / "results/runs/12/NYU_rois12_w60s6_brainnetcnn_control_baseline_v13_3e220e5c"
preds = pd.read_csv(run / "predictions_val.csv", dtype={"subject_id": str})
ref = preds[preds["repeat"] == 1].sort_values("subject_id", key=lambda s: s.astype(str))
y_true = ref["y_true"].to_numpy()
probs = ref["y_prob"].to_numpy()
rng2 = np.random.default_rng(42)
control_idx = np.flatnonzero(y_true == 0)
adhd_idx = np.flatnonzero(y_true == 1)
n_iter = 2000
boot_c = control_idx[rng2.integers(0, len(control_idx), size=(n_iter, len(control_idx)))]
boot_a = adhd_idx[rng2.integers(0, len(adhd_idx), size=(n_iter, len(adhd_idx)))]
pos_scores = probs[boot_a]
neg_scores = probs[boot_c]
v4_all = batch_auc_mannwhitney_v4(pos_scores, neg_scores)
sample_diffs_v3 = []
sample_diffs_v4 = []
for it in range(0, n_iter, 20):
    idx = np.concatenate([boot_c[it], boot_a[it]])
    yb = y_true[idx]; sb = probs[idx]
    sk_i = roc_auc_score(yb, sb)
    v3_i = batch_auc_rank_v3(yb[None, :], sb[None, :])[0]
    sample_diffs_v3.append(abs(sk_i - v3_i))
    sample_diffs_v4.append(abs(sk_i - v4_all[it]))
mean_dup = float(np.mean([len(x) - len(np.unique(x)) for x in np.concatenate([boot_c[:200], boot_a[:200]], axis=1)]))
results["test2_real_bootstrap_resamples"] = {
    "source_run": str(run.relative_to(REPO)), "n_iter_generated": n_iter, "n_iter_sampled_for_check": len(sample_diffs_v3),
    "mean_within_class_duplicate_subjects_per_iter": mean_dup,
    "max_v3_diff_vs_sklearn": float(max(sample_diffs_v3)),
    "max_v4_diff_vs_sklearn": float(max(sample_diffs_v4)),
    "v3_within_1e-12": bool(max(sample_diffs_v3) < 1e-12),
    "v4_within_1e-12": bool(max(sample_diffs_v4) < 1e-12),
}

# --- Test 3: barrido de empates cruzados de clase en los 42 corridas reales ---
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
manifest = pd.read_csv(REPO / "analysis/roi_comparison/config/baseline_manifest.csv", dtype=str)
baseline_paths = set()
for _, row in manifest.iterrows():
    baseline_paths.add(str(REPO / row["comparator_relative_path"]))
    baseline_paths.add(str(REPO / row["baseline_relative_path"]))
all_runs = set()
for k, (a, b) in TEN_CONTRASTS.items():
    all_runs.add(str(REPO / a)); all_runs.add(str(REPO / b))
all_runs |= baseline_paths

cross_class_ties = []
n_checked = 0
for run_path in sorted(all_runs):
    p = Path(run_path) / "predictions_val.csv"
    df = pd.read_csv(p, dtype={"subject_id": str})
    for r in range(1, 6):
        sub = df[df["repeat"] == r]
        pos = sub[sub["y_true"] == 1]["y_prob"].to_numpy()
        neg = sub[sub["y_true"] == 0]["y_prob"].to_numpy()
        common = np.intersect1d(pos, neg)
        n_checked += 1
        if len(common) > 0:
            cross_class_ties.append({"run": str(Path(run_path).relative_to(REPO)), "repeat": r, "n_ties": int(len(common))})
results["test3_cross_class_tie_sweep"] = {
    "n_unique_runs_checked": len(all_runs),
    "n_run_repeat_combinations_checked": n_checked,
    "n_combinations_with_cross_class_ties": len(cross_class_ties),
    "detail": cross_class_ties,
}

out = Path("/sessions/laughing-wizardly-archimedes/mnt/outputs/rebuild_v4/recalculate_manuscript_bootstrap_10k_tie_tests.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "detail"} for k, v in results.items()}, indent=2))
print("\nWritten:", out)
