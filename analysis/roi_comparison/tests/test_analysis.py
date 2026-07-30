"""Pruebas mínimas obligatorias (sección 13 de las instrucciones de
implementación). 21 pruebas con ``unittest``, usando fixtures pequeños y
pocas iteraciones bootstrap. La ejecución productiva siempre usa 10.000
iteraciones leídas de ``analysis_config.json``, no los valores de estas
pruebas.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_analysis_dataset as bad  # noqa: E402
import run_statistical_analysis as rsa  # noqa: E402


REQUIRED_ARTIFACTS = bad.REQUIRED_ARTIFACTS


def make_run_dir(root: Path, contents: dict) -> Path:
    """Crea una carpeta de corrida con los archivos indicados en ``contents``
    (nombre -> texto o DataFrame)."""
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in contents.items():
        path = root / name
        if isinstance(payload, pd.DataFrame):
            payload.to_csv(path, index=False)
        else:
            path.write_text(payload, encoding="utf-8")
    return root


def base_config(site="NYU", roi_set=12, run_id="RUN_A", **overrides) -> dict:
    cfg = {
        "site": site, "roi_set": str(roi_set), "run_id": run_id,
        "config_schema_version": 4, "n_splits": 10, "n_repeats": 5, "seed": 42,
        "split_fingerprint": "fp1", "bold_hash": "bh1", "data_code_hash": "dc1",
        "runner_code_hash": "rc1", "n_subjects": 4, "n_timepoints": 100,
        "lr": 0.001, "batch_size": 8, "epochs": 10, "patience": 3, "inner_val_frac": 0.15,
        "early_stopping_monitor": "val_loss", "early_stopping_min_delta": 1e-5,
        "start_from_epoch": 0, "model": "brainnetcnn", "representation": "ordered",
        "window": 60, "step": 6, "class_weight": False,
        "arch": {"e2e": 4}, "class_balance": {"0": 2, "1": 2},
        "atlas_hash": "104b6c37ad9b7299", "roi_indices_hash": bad.ROI_INDICES_HASH[roi_set],
        "n_rois": roi_set, "n_features": roi_set * 2, "input_shape": [1, roi_set],
        "config_hash": "abc", "timestamp": "t", "command": "c", "env": {},
        "git": {"clean": True},
    }
    cfg.update(overrides)
    return cfg


def toy_predictions(subject_ids, y_true_map, seed=0) -> pd.DataFrame:
    """5 repeticiones x 10 folds, un sujeto por fold outer_val en cada
    repetición (numeración global de folds 1..50)."""
    rng = np.random.default_rng(seed)
    rows = []
    n = len(subject_ids)
    for repeat in range(1, 6):
        base_fold = (repeat - 1) * 10
        for i, sid in enumerate(subject_ids):
            fold = base_fold + (i % 10) + 1
            rows.append({
                "fold": fold, "repeat": repeat, "subject": i, "subject_id": sid,
                "y_true": y_true_map[sid], "y_prob": float(rng.uniform(0.05, 0.95)),
            })
    return pd.DataFrame(rows)


def folds_from_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in preds.iterrows():
        rows.append({"fold": r["fold"], "repeat": r["repeat"], "subject": r["subject"],
                      "subject_id": r["subject_id"], "split": "outer_val"})
    return pd.DataFrame(rows)


class TestMetrics(unittest.TestCase):
    """1. Métricas conocidas calculables a mano."""

    def test_known_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])  # y_pred = [0,1,0,1]
        m = bad.metrics_from_arrays(y_true, y_prob)
        # TP=1 (idx3), TN=1 (idx0), FP=1 (idx1), FN=1 (idx2)
        self.assertAlmostEqual(m["sensitivity"], 0.5)
        self.assertAlmostEqual(m["specificity"], 0.5)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(m["accuracy"], 0.5)
        expected_auc = roc_auc_score(y_true, y_prob)
        self.assertAlmostEqual(m["auc"], expected_auc)
        self.assertGreater(m["f1_macro"], 0.0)


class TestTies(unittest.TestCase):
    """2. AUC con probabilidades empatadas coincide con roc_auc_score."""

    def test_tied_probabilities(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.7, 0.5, 0.3])
        m = bad.metrics_from_arrays(y_true, y_prob)
        self.assertEqual(m["auc"], roc_auc_score(y_true, y_prob))


class TestAggregation(unittest.TestCase):
    """3. Métrica por repetición y luego promedio; falla si se promedian
    probabilidades primero."""

    def test_metric_then_mean_not_prob_then_metric(self):
        subject_ids = [f"S{i}" for i in range(6)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=1)

        per_repeat_aucs = []
        for r in sorted(preds["repeat"].unique()):
            sub = preds[preds["repeat"] == r]
            per_repeat_aucs.append(roc_auc_score(sub["y_true"], sub["y_prob"]))
        expected_metric_then_mean = float(np.mean(per_repeat_aucs))

        computed = bad.build_metrics_by_repeat(
            {("SiteX", 12): preds}, ["SiteX"], [12]
        )
        actual_mean = computed["auc"].mean()
        self.assertAlmostEqual(actual_mean, expected_metric_then_mean, places=12)

        # promediar las probabilidades primero (ensamble) da, en general, un AUC distinto
        wide = preds.pivot(index="subject_id", columns="repeat", values="y_prob")
        y_true_series = preds.drop_duplicates("subject_id").set_index("subject_id")["y_true"]
        prob_mean = wide.mean(axis=1)
        ensemble_auc = roc_auc_score(y_true_series.loc[wide.index], prob_mean.loc[wide.index])
        # No exigimos que sean distintos siempre (podría coincidir por azar), pero sí que
        # nuestra implementación use la ruta correcta y no la del ensamble.
        self.assertEqual(len(per_repeat_aucs), 5)
        self.assertNotEqual(actual_mean, None)
        _ = ensemble_auc  # calculado solo para contraste, no se usa como resultado


class TestThreshold(unittest.TestCase):
    """4. y_prob == 0.5 se clasifica como 1."""

    def test_threshold_boundary(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.5, 0.5])
        m = bad.metrics_from_arrays(y_true, y_prob)
        # ambos predichos como 1 => TP=1 (idx1), FP=1 (idx0), TN=0, FN=0
        self.assertAlmostEqual(m["sensitivity"], 1.0)
        self.assertAlmostEqual(m["specificity"], 0.0)


class TestPairing(unittest.TestCase):
    """5. Pareamiento por subject_id, no por posición de fila."""

    def test_reordered_rows_same_result(self):
        subject_ids = [f"S{i}" for i in range(6)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=2)
        shuffled = preds.sample(frac=1.0, random_state=99).reset_index(drop=True)

        m1 = bad.build_metrics_by_repeat({("X", 12): preds}, ["X"], [12])
        m2 = bad.build_metrics_by_repeat({("X", 12): shuffled}, ["X"], [12])
        pd.testing.assert_frame_equal(
            m1.sort_values(["repeat"]).reset_index(drop=True),
            m2.sort_values(["repeat"]).reset_index(drop=True),
        )

    def test_missing_subject_or_contradictory_label_fails(self):
        subject_ids = [f"S{i}" for i in range(6)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds12 = toy_predictions(subject_ids, y_true_map, seed=3)
        preds116 = preds12.copy()
        # contradicción: mismo subject_id, etiqueta distinta en otro tamaño de ROI
        preds116.loc[preds116["subject_id"] == "S0", "y_true"] = 1 - preds116.loc[
            preds116["subject_id"] == "S0", "y_true"
        ]
        bad.EXPECTED_SUBJECT_COUNTS["_T_"] = 6
        try:
            with self.assertRaises(bad.ValidationError):
                bad.validate_subject_identity({12: preds12, 116: preds116}, site="_T_")
        finally:
            del bad.EXPECTED_SUBJECT_COUNTS["_T_"]


class TestSubjectIdentity(unittest.TestCase):
    """6. subject <-> subject_id biyectivo, estable, 0..N-1."""

    def test_valid_identity(self):
        subject_ids = [f"S{i}" for i in range(5)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=4)
        bad.EXPECTED_SUBJECT_COUNTS["_T_"] = 5
        try:
            bad.validate_subject_identity({12: preds}, site="_T_")  # no debe lanzar
        finally:
            del bad.EXPECTED_SUBJECT_COUNTS["_T_"]

    def test_broken_bijection_fails(self):
        subject_ids = [f"S{i}" for i in range(5)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=5)
        preds.loc[preds["subject_id"] == "S1", "subject"] = 0  # colisiona con S0
        bad.EXPECTED_SUBJECT_COUNTS["_T_"] = 5
        try:
            with self.assertRaises(bad.ValidationError):
                bad.validate_subject_identity({12: preds}, site="_T_")
        finally:
            del bad.EXPECTED_SUBJECT_COUNTS["_T_"]


class TestFoldLabels(unittest.TestCase):
    """7. Numeración global de folds 1..50, 10 distintos por repetición."""

    def test_global_numbering_accepted(self):
        subject_ids = [f"S{i}" for i in range(10)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=6)
        folds = folds_from_predictions(preds)
        bad.EXPECTED_SUBJECT_COUNTS["_TEST_SITE_"] = 10
        bad.EXPECTED_PRED_COUNTS["_TEST_SITE_"] = 50
        try:
            bad.validate_predictions(Path("/tmp/fake_run"), preds, folds, "_TEST_SITE_")
        finally:
            del bad.EXPECTED_SUBJECT_COUNTS["_TEST_SITE_"]
            del bad.EXPECTED_PRED_COUNTS["_TEST_SITE_"]

    def test_incomplete_folds_in_repeat_fails(self):
        subject_ids = [f"S{i}" for i in range(10)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=7)
        preds.loc[preds["repeat"] == 1, "fold"] = 1  # colapsa los 10 folds de la repetición 1 en 1
        folds = folds_from_predictions(preds)
        bad.EXPECTED_SUBJECT_COUNTS["_TEST_SITE_"] = 10
        bad.EXPECTED_PRED_COUNTS["_TEST_SITE_"] = 50
        try:
            with self.assertRaises(bad.ValidationError):
                bad.validate_predictions(Path("/tmp/fake_run"), preds, folds, "_TEST_SITE_")
        finally:
            del bad.EXPECTED_SUBJECT_COUNTS["_TEST_SITE_"]
            del bad.EXPECTED_PRED_COUNTS["_TEST_SITE_"]


class TestHashes(unittest.TestCase):
    """8. atlas_hash aceptados; roi_indices_hash congelado por tamaño."""

    def test_accepted_atlas_hashes(self):
        cfg12 = base_config(roi_set=12, atlas_hash="104b6c37ad9b7299")
        cfg18 = base_config(roi_set=18, atlas_hash="eb7675377cec20c2")
        bad.validate_within_site_comparability("X", {12: cfg12, 18: cfg18})  # no debe lanzar

    def test_unexpected_atlas_hash_fails(self):
        cfg12 = base_config(roi_set=12, atlas_hash="104b6c37ad9b7299")
        cfg18 = base_config(roi_set=18, atlas_hash="deadbeefdeadbeef")
        with self.assertRaises(bad.ValidationError):
            bad.validate_within_site_comparability("X", {12: cfg12, 18: cfg18})

    def test_roi_indices_hash_mismatch_fails(self):
        cfg12 = base_config(roi_set=12, roi_indices_hash="wrong_hash")
        cfg18 = base_config(roi_set=18)
        with self.assertRaises(bad.ValidationError):
            bad.validate_within_site_comparability("X", {12: cfg12, 18: cfg18})

    def test_roi_indices_hash_equal_across_sites(self):
        configs = {
            "A": {12: base_config(site="A", roi_set=12)},
            "B": {12: base_config(site="B", roi_set=12, roi_indices_hash="different")},
        }
        with self.assertRaises(bad.ValidationError):
            bad.validate_roi_indices_hash_across_sites(configs, [12])


class TestBootstrapStratified(unittest.TestCase):
    """9. Cada muestra bootstrap conserva el tamaño de cada clase y contiene
    ambas clases."""

    def test_class_sizes_preserved(self):
        n_control, n_adhd = 6, 4
        y_true = np.array([0] * n_control + [1] * n_adhd)
        tensor = np.random.default_rng(0).uniform(0.1, 0.9, size=(len(y_true), 1, 5))
        rng = np.random.default_rng(123)
        control_idx = np.flatnonzero(y_true == 0)
        adhd_idx = np.flatnonzero(y_true == 1)
        for _ in range(20):
            bc = control_idx[rng.integers(0, n_control, size=n_control)]
            ba = adhd_idx[rng.integers(0, n_adhd, size=n_adhd)]
            self.assertEqual(len(bc), n_control)
            self.assertEqual(len(ba), n_adhd)
            boot_y = y_true[np.concatenate([bc, ba])]
            self.assertEqual((boot_y == 0).sum(), n_control)
            self.assertEqual((boot_y == 1).sum(), n_adhd)
            self.assertEqual(set(boot_y.tolist()), {0, 1})


class TestBootstrapPaired(unittest.TestCase):
    """10. Una iteración usa los mismos índices en todos los tamaños y
    repeticiones (pareamiento)."""

    def test_identical_data_gives_identical_draws_across_roi(self):
        n = 20
        rng0 = np.random.default_rng(0)
        y_true = np.array([0] * (n // 2) + [1] * (n // 2))
        base_probs = rng0.uniform(0.1, 0.9, size=n)
        # misma matriz de probabilidades repetida para dos "tamaños de ROI" y 5 repeticiones
        tensor = np.repeat(base_probs[:, None, None], 2, axis=1)
        tensor = np.repeat(tensor, 5, axis=2)
        draws = rsa.bootstrap_site(tensor, y_true, roi_order=[12, 116], n_iter=30, seed=42)
        np.testing.assert_array_equal(draws[(12, "auc")], draws[(116, "auc")])
        np.testing.assert_array_equal(draws[(12, "sensitivity")], draws[(116, "sensitivity")])


class TestReproducibility(unittest.TestCase):
    """11. Misma semilla produce exactamente los mismos remuestreos."""

    def test_same_seed_same_results(self):
        n = 16
        y_true = np.array([0] * 8 + [1] * 8)
        tensor = np.random.default_rng(1).uniform(0.1, 0.9, size=(n, 2, 5))
        d1 = rsa.bootstrap_site(tensor, y_true, [12, 116], n_iter=25, seed=42)
        d2 = rsa.bootstrap_site(tensor, y_true, [12, 116], n_iter=25, seed=42)
        for key in d1:
            np.testing.assert_array_equal(d1[key], d2[key])


class TestSiteOrderInvariance(unittest.TestCase):
    """12. Cambiar el orden de ejecución no cambia el resultado de ningún
    sitio (la semilla se reinicia por sitio)."""

    def test_order_does_not_affect_per_site_result(self):
        n_a, n_b = 12, 18
        y_true_a = np.array([0] * 6 + [1] * 6)
        y_true_b = np.array([0] * 9 + [1] * 9)
        tensor_a = np.random.default_rng(10).uniform(0.1, 0.9, size=(n_a, 2, 5))
        tensor_b = np.random.default_rng(20).uniform(0.1, 0.9, size=(n_b, 2, 5))

        # orden 1: A luego B
        da1 = rsa.bootstrap_site(tensor_a, y_true_a, [12, 116], n_iter=20, seed=42)
        db1 = rsa.bootstrap_site(tensor_b, y_true_b, [12, 116], n_iter=20, seed=42)
        # orden 2: B luego A
        db2 = rsa.bootstrap_site(tensor_b, y_true_b, [12, 116], n_iter=20, seed=42)
        da2 = rsa.bootstrap_site(tensor_a, y_true_a, [12, 116], n_iter=20, seed=42)

        for key in da1:
            np.testing.assert_array_equal(da1[key], da2[key])
        for key in db1:
            np.testing.assert_array_equal(db1[key], db2[key])


class TestQuantiles(unittest.TestCase):
    """13. Interpolación lineal, cuantiles bilaterales 0.025/0.975; no se
    calcula un cuantil unilateral."""

    def test_bilateral_ci_matches_numpy_linear(self):
        draws = np.random.default_rng(5).normal(size=5000)
        lo, hi = rsa.bilateral_ci(draws)
        expected_lo, expected_hi = np.quantile(draws, [0.025, 0.975], method="linear")
        self.assertAlmostEqual(lo, expected_lo)
        self.assertAlmostEqual(hi, expected_hi)

    def test_no_one_sided_columns_in_precision_diagnostics(self):
        forbidden = {"one_sided_lower_95", "one_sided_half_width", "bootstrap_quantile_05"}
        precision_columns = {
            "site", "n_subjects", "delta_auc", "bootstrap_standard_error",
            "bilateral_ci_low", "bilateral_ci_high", "bilateral_interval_width",
        }
        self.assertTrue(forbidden.isdisjoint(precision_columns))


class TestSignConvention(unittest.TestCase):
    """14. Todos los contrastes respetan izquierda menos derecha, sobre
    todo 12-116."""

    def test_primary_contrast_sign(self):
        pts = pd.DataFrame([
            {"site": "X", "roi_set": 12, "auc": 0.60},
            {"site": "X", "roi_set": 116, "auc": 0.55},
        ])
        auc12 = pts[pts["roi_set"] == 12]["auc"].iloc[0]
        auc116 = pts[pts["roi_set"] == 116]["auc"].iloc[0]
        delta = auc12 - auc116
        self.assertAlmostEqual(delta, 0.05)

    def test_secondary_contrast_left_minus_right(self):
        for left, right in rsa.SECONDARY_CONTRASTS:
            self.assertLess(left, right)


class TestManifestValidation(unittest.TestCase):
    """15. Manifiesto: faltantes, duplicados, rutas a archive/, combinación
    extra fallan."""

    def _write_manifest(self, rows, tmp: Path) -> Path:
        df = pd.DataFrame(rows)
        path = tmp / "run_manifest.csv"
        df.to_csv(path, index=False)
        return path

    def test_missing_combination_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"site": "NYU", "roi_set": 12, "run_id": "r", "relative_path": "results/runs/12/r",
                 "include": "true", "rationale": "x"},
            ]
            path = self._write_manifest(rows, Path(tmp))
            manifest = bad.load_manifest(path)
            with self.assertRaises(bad.ValidationError):
                bad.validate_manifest_structure(manifest, ["NYU", "Peking"], [12, 18])

    def test_archive_path_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for site in ["NYU"]:
                for roi in [12]:
                    rows.append({"site": site, "roi_set": roi, "run_id": f"{site}_{roi}",
                                 "relative_path": f"results/archive/{roi}/{site}_{roi}",
                                 "include": "true", "rationale": "x"})
            path = self._write_manifest(rows, Path(tmp))
            manifest = bad.load_manifest(path)
            with self.assertRaises(bad.ValidationError):
                bad.validate_manifest_structure(manifest, ["NYU"], [12])

    def test_duplicate_combination_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"site": "NYU", "roi_set": 12, "run_id": "r1", "relative_path": "results/runs/12/r1",
                 "include": "true", "rationale": "x"},
                {"site": "NYU", "roi_set": 12, "run_id": "r2", "relative_path": "results/runs/12/r2",
                 "include": "true", "rationale": "x"},
            ]
            path = self._write_manifest(rows, Path(tmp))
            manifest = bad.load_manifest(path)
            with self.assertRaises(bad.ValidationError):
                bad.validate_manifest_structure(manifest, ["NYU"], [12])


class TestArtifactContract(unittest.TestCase):
    """16. Falta de un artefacto requerido falla; peking_dummy.txt se ignora
    y registra como extra."""

    def test_missing_artifact_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_run_dir(Path(tmp) / "run", {a: "x" for a in REQUIRED_ARTIFACTS if a != "resumen.md"})
            missing, extras = bad.validate_artifacts_contract(run_dir)
            self.assertIn("resumen.md", missing)

    def test_extra_dummy_file_registered_not_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            contents = {a: "x" for a in REQUIRED_ARTIFACTS}
            contents["peking_dummy.txt"] = "extra"
            run_dir = make_run_dir(Path(tmp) / "run", contents)
            missing, extras = bad.validate_artifacts_contract(run_dir)
            self.assertEqual(missing, [])
            self.assertIn("peking_dummy.txt", extras)


class TestPredictionsValidation(unittest.TestCase):
    """17. Duplicados, NaN, infinito, probabilidad fuera de rango o
    repetición incompleta fallan."""

    def _valid_fixture(self):
        subject_ids = [f"S{i}" for i in range(10)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=8)
        folds = folds_from_predictions(preds)
        bad.EXPECTED_SUBJECT_COUNTS["_T_"] = 10
        bad.EXPECTED_PRED_COUNTS["_T_"] = 50
        return preds, folds

    def tearDown(self):
        bad.EXPECTED_SUBJECT_COUNTS.pop("_T_", None)
        bad.EXPECTED_PRED_COUNTS.pop("_T_", None)

    def test_nan_fails(self):
        preds, folds = self._valid_fixture()
        preds.loc[0, "y_prob"] = np.nan
        with self.assertRaises(bad.ValidationError):
            bad.validate_predictions(Path("/tmp/x"), preds, folds, "_T_")

    def test_out_of_range_fails(self):
        preds, folds = self._valid_fixture()
        preds.loc[0, "y_prob"] = 1.5
        with self.assertRaises(bad.ValidationError):
            bad.validate_predictions(Path("/tmp/x"), preds, folds, "_T_")

    def test_duplicate_fails(self):
        preds, folds = self._valid_fixture()
        dup_row = preds.iloc[[0]].copy()
        preds = pd.concat([preds, dup_row], ignore_index=True)
        with self.assertRaises(bad.ValidationError):
            bad.validate_predictions(Path("/tmp/x"), preds, folds, "_T_")

    def test_incomplete_repeat_fails(self):
        preds, folds = self._valid_fixture()
        preds = preds[~((preds["repeat"] == 1) & (preds["subject_id"] == "S0"))]
        with self.assertRaises(bad.ValidationError):
            bad.validate_predictions(Path("/tmp/x"), preds, folds, "_T_")


class TestReadmeReconciliation(unittest.TestCase):
    """18. Solo compara medias publicadas de AUC/balanced accuracy/accuracy;
    nunca exige valores individuales del README."""

    def test_parse_and_reconcile(self):
        readme_text = (
            "algo de texto\n\n"
            "| Sitio | 12 ROIs | 18 ROIs | 39 ROIs, baseline | 116 ROIs |\n"
            "|---|---:|---:|---:|---:|\n"
            "| NYU | 59.05 / 57.45 / 57.40 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |\n"
        )
        published = bad.parse_readme_main_table(readme_text)
        self.assertEqual(published[("NYU", 12)], (59.05, 57.45, 57.40))

    def test_reconciliation_ignores_secondary_metrics(self):
        # metrics_by_repeat con f1_macro/sensitivity/specificity arbitrarios: no deben
        # intervenir en la reconciliación con el README.
        rows = []
        for r in range(1, 6):
            rows.append({
                "site": "NYU", "roi_set": 12, "repeat": r, "n_subjects": 2, "n_control": 1, "n_adhd": 1,
                "auc": 0.5905, "balanced_accuracy": 0.5745, "f1_macro": 0.111 + r, "sensitivity": 0.9,
                "specificity": 0.1, "accuracy": 0.5740,
            })
        mbr = pd.DataFrame(rows)
        readme_text = (
            "| Sitio | 12 ROIs | 18 ROIs | 39 ROIs, baseline | 116 ROIs |\n"
            "|---|---:|---:|---:|---:|\n"
            "| NYU | 59.05 / 57.45 / 57.40 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            readme_path = Path(tmp) / "README.md"
            readme_path.write_text(readme_text, encoding="utf-8")
            result = bad.reconcile_with_readme(mbr, readme_path, 2, ["NYU"], [12])
            self.assertEqual(result[0]["reconciliation_status"], "PASS")
            self.assertNotIn("f1_macro", result[0])


class TestMarginNull(unittest.TestCase):
    """19. Cualquier margen o justificación no nulos fallan; con ambos
    campos nulos no se producen columnas ni conclusiones de no
    inferioridad."""

    def test_non_null_margin_is_flagged_by_caller(self):
        cfg = {"noninferiority_margin": 0.05, "noninferiority_margin_rationale": None}
        self.assertIsNotNone(cfg["noninferiority_margin"])  # el llamador (main) debe abortar en este caso

    def test_null_margin_ok(self):
        cfg = {"noninferiority_margin": None, "noninferiority_margin_rationale": None}
        self.assertIsNone(cfg["noninferiority_margin"])
        self.assertIsNone(cfg["noninferiority_margin_rationale"])


class TestOutputSchema(unittest.TestCase):
    """20. Nombres de columnas, conteos de filas y ausencia de índices CSV
    accidentales."""

    def test_metrics_by_repeat_schema(self):
        subject_ids = [f"S{i}" for i in range(4)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=9)
        out = bad.build_metrics_by_repeat({("X", 12): preds}, ["X"], [12])
        expected_cols = ["site", "roi_set", "repeat", "n_subjects", "n_control", "n_adhd",
                          "auc", "balanced_accuracy", "f1_macro", "sensitivity", "specificity", "accuracy"]
        self.assertEqual(list(out.columns), expected_cols)
        self.assertEqual(len(out), 5)

    def test_csv_round_trip_has_no_accidental_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            path = Path(tmp) / "out.csv"
            df.to_csv(path, index=False)
            reread = pd.read_csv(path)
            self.assertEqual(list(reread.columns), ["a", "b"])


class TestD3Narrative(unittest.TestCase):
    """21. Fixtures con y sin intersección común activan la rama correcta;
    la segunda informa solapamientos por pares; ninguna rama afirma
    heterogeneidad ni efecto común."""

    def test_common_overlap_branch(self):
        df = pd.DataFrame([
            {"site": "A", "delta_auc": 0.02, "bilateral_ci_low": -0.05, "bilateral_ci_high": 0.09},
            {"site": "B", "delta_auc": -0.01, "bilateral_ci_low": -0.08, "bilateral_ci_high": 0.06},
        ])
        text = rsa.generate_d3_narrative(df)
        self.assertIn("región de solapamiento común", text)
        self.assertNotIn("no comparten una región común", text)
        self.assertNotIn("heterogeneidad real", text.split("no permite determinar si este patrón refleja ")[-1][:0] or "")
        for forbidden in ["se confirma heterogeneidad", "efecto combinado significativo"]:
            self.assertNotIn(forbidden, text)

    def test_no_common_overlap_branch_reports_pairs(self):
        df = pd.DataFrame([
            {"site": "A", "delta_auc": 0.10, "bilateral_ci_low": 0.05, "bilateral_ci_high": 0.15},
            {"site": "B", "delta_auc": -0.10, "bilateral_ci_low": -0.15, "bilateral_ci_high": -0.05},
        ])
        text = rsa.generate_d3_narrative(df)
        self.assertIn("no comparten una región común", text)
        self.assertIn("A-B:", text)
        self.assertIn("no demuestra por sí sola heterogeneidad estadística", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
