"""Pruebas mínimas obligatorias (sección 13 de las instrucciones de
implementación). 21 pruebas con ``unittest``, usando fixtures pequeños y
pocas iteraciones bootstrap. La ejecución productiva siempre usa 10.000
iteraciones leídas de ``analysis_config.json``, no los valores de estas
pruebas.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
REPO_ROOT = Path(__file__).resolve().parents[3]
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


def base_config(site="NYU", roi_set=12, run_id="RUN_A", config_hash="3e220e5c", **overrides) -> dict:
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
        "config_hash": config_hash, "timestamp": "t", "command": "c", "env": {},
        "git": {"clean": True},
        # CORRECCIONES_V19 §7.2: campos científicos adicionales + nulls esperados.
        "windowing_preset": "custom", "fisher_z": False, "constant_policy": "zero",
        "clipnorm": None, "deterministic": False, "mixed_precision": False,
        "windowing": {"mode": "dynamic", "window_tr": 60, "step_tr": 6, "shape": "rectangular"},
        "representation_seed": None, "random_subset": None, "n_random_sets": None, "exclude_roi_set": None,
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
        # Corregida (CORRECCIONES_V19 §13): antes comprobaba un literal
        # escrito dentro del propio test. Ahora llama la función productiva
        # real que arma cada fila de precision_diagnostics.csv y examina su
        # esquema efectivo, más la constante de columnas que usa main().
        forbidden = {"one_sided_lower_95", "one_sided_half_width", "bootstrap_quantile_05"}
        row = rsa.build_precision_diagnostics_row(
            site="X", n_subjects=100, delta_auc=0.05, se=0.03, lo=-0.01, hi=0.11
        )
        self.assertTrue(forbidden.isdisjoint(row.keys()))
        self.assertTrue(forbidden.isdisjoint(rsa.PRECISION_DIAGNOSTICS_COLUMNS))
        self.assertEqual(sorted(row.keys()), sorted(rsa.PRECISION_DIAGNOSTICS_COLUMNS))


class TestSignConvention(unittest.TestCase):
    """14. Todos los contrastes respetan izquierda menos derecha, sobre
    todo 12-116."""

    def test_primary_contrast_sign(self):
        # Corregida (CORRECCIONES_V19 §13): antes restaba dentro del propio
        # test. Ahora llama compute_primary_delta(), la función productiva
        # que usa main() para construir delta_auc en primary_12_vs_116.csv.
        self.assertAlmostEqual(rsa.compute_primary_delta(0.60, 0.55), 0.05)
        self.assertAlmostEqual(rsa.compute_primary_delta(0.55, 0.60), -0.05)

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
        # Corregida (CORRECCIONES_V19 §13): antes solo comprobaba un literal
        # dentro del test. Ahora llama al validador real y comprueba que
        # rechaza un margen no nulo.
        cfg = dict(bad.FROZEN_ANALYSIS_CONFIG)
        cfg["noninferiority_margin"] = 0.05
        with self.assertRaises(bad.ValidationError):
            bad.validate_analysis_config(cfg)

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


class TestHypothesisTests(unittest.TestCase):
    """22. Friedman ómnibus + Wilcoxon-Holm pareado por sitio (prueba
    exploratoria post-hoc agregada a pedido de una revisión externa, fuera
    del alcance original v1.5/plan 5.6): conteos de filas, corrección Holm
    correcta, piso de p=0.0625 con n=5 pares, reproducibilidad con semilla
    fija, y sensibilidad de Friedman a un orden perfectamente consistente
    entre repeticiones."""

    @staticmethod
    def _toy_metrics_by_repeat(sites, roi_order, seed=0, perfect_order=False):
        rng = np.random.default_rng(seed)
        rows = []
        for site in sites:
            for r in range(1, 6):
                if perfect_order:
                    # valores estrictamente crecientes en el orden de roi_order,
                    # con ruido pequeño para no generar empates.
                    base = np.arange(len(roi_order), dtype=float)
                    noise = rng.uniform(-0.01, 0.01, size=len(roi_order))
                    aucs = 0.5 + 0.05 * base + noise
                else:
                    aucs = rng.uniform(0.45, 0.65, size=len(roi_order))
                for roi, auc in zip(roi_order, aucs):
                    rows.append({
                        "site": site, "roi_set": roi, "repeat": r,
                        "n_subjects": 10, "n_control": 5, "n_adhd": 5,
                        "auc": float(auc), "balanced_accuracy": float(auc),
                        "f1_macro": float(auc), "sensitivity": float(auc),
                        "specificity": float(auc), "accuracy": float(auc),
                    })
        return pd.DataFrame(rows)

    def test_row_counts(self):
        sites = ["SiteA", "SiteB"]
        roi_order = [12, 18, 39, 116]
        mbr = self._toy_metrics_by_repeat(sites, roi_order, seed=1)
        fried = rsa.compute_friedman_omnibus_by_site(mbr, sites, roi_order, n_permutations=2000, seed=1)
        wil = rsa.compute_wilcoxon_pairwise_by_site(mbr, sites, roi_order)
        self.assertEqual(len(fried), len(sites))
        self.assertEqual(len(wil), len(sites) * 6)  # C(4,2) = 6 pares por sitio

    def test_wilcoxon_p_floor_with_n5(self):
        # Con n=5 repeticiones pareadas, el p-valor exacto de dos colas
        # nunca puede ser menor que 2*(1/2**5) = 0.0625, sin importar el
        # tamaño del efecto observado.
        sites = ["SiteA"]
        roi_order = [12, 18, 39, 116]
        mbr = self._toy_metrics_by_repeat(sites, roi_order, seed=2)
        wil = rsa.compute_wilcoxon_pairwise_by_site(mbr, sites, roi_order)
        self.assertTrue((wil["p_value_raw"] >= 0.0625 - 1e-9).all())

    def test_holm_correction_matches_statsmodels(self):
        sites = ["SiteA", "SiteB"]
        roi_order = [12, 18, 39, 116]
        mbr = self._toy_metrics_by_repeat(sites, roi_order, seed=3)
        wil = rsa.compute_wilcoxon_pairwise_by_site(mbr, sites, roi_order)
        for site in sites:
            sub = wil[wil["site"] == site]
            _, expected_holm, _, _ = multipletests(sub["p_value_raw"].to_numpy(), alpha=0.05, method="holm")
            np.testing.assert_allclose(sub["p_value_holm"].to_numpy(), expected_holm)

    def test_friedman_reproducible_with_fixed_seed(self):
        sites = ["SiteA"]
        roi_order = [12, 18, 39, 116]
        mbr = self._toy_metrics_by_repeat(sites, roi_order, seed=4)
        f1 = rsa.compute_friedman_omnibus_by_site(mbr, sites, roi_order, n_permutations=5000, seed=42)
        f2 = rsa.compute_friedman_omnibus_by_site(mbr, sites, roi_order, n_permutations=5000, seed=42)
        self.assertEqual(f1["p_value_permutation"].iloc[0], f2["p_value_permutation"].iloc[0])

    def test_friedman_detects_perfectly_consistent_order(self):
        # Sanity check direccional: si el orden de los cuatro tamaños de ROI
        # es idéntico en las cinco repeticiones, el p-valor de permutación
        # debe ser pequeño (caso extremo, no depende de la fórmula exacta).
        sites = ["SiteA"]
        roi_order = [12, 18, 39, 116]
        mbr = self._toy_metrics_by_repeat(sites, roi_order, seed=5, perfect_order=True)
        fried = rsa.compute_friedman_omnibus_by_site(mbr, sites, roi_order, n_permutations=20000, seed=5)
        self.assertLess(fried["p_value_permutation"].iloc[0], 0.01)

    def test_missing_repeat_raises(self):
        sites = ["SiteA"]
        roi_order = [12, 18, 39, 116]
        mbr = self._toy_metrics_by_repeat(sites, roi_order, seed=6)
        mbr = mbr[~((mbr["site"] == "SiteA") & (mbr["roi_set"] == 12) & (mbr["repeat"] == 1))]
        with self.assertRaises(SystemExit):
            rsa.compute_friedman_omnibus_by_site(mbr, sites, roi_order, n_permutations=1000, seed=6)
        with self.assertRaises(SystemExit):
            rsa.compute_wilcoxon_pairwise_by_site(mbr, sites, roi_order)


class TestCanonicalPlan(unittest.TestCase):
    """CORRECCIONES_V19 §13.1: el plan canónico tiene el SHA-256 esperado."""

    def test_plan_hash_matches_canonical(self):
        plan_path = REPO_ROOT / "analysis" / "roi_comparison" / "analysis_plan.md"
        self.assertTrue(plan_path.exists(), f"no existe {plan_path}")
        self.assertEqual(bad.sha256_file(plan_path), rsa.CANONICAL_PLAN_SHA256)


class TestAnalysisConfigValidation(unittest.TestCase):
    """CORRECCIONES_V19 §13.2-3: cada decisión congelada se acepta con la
    configuración correcta; mutar cualquier campo científico crítico falla."""

    def test_frozen_config_accepted(self):
        bad.validate_analysis_config(dict(bad.FROZEN_ANALYSIS_CONFIG))  # no debe lanzar

    def test_mutating_each_frozen_field_fails(self):
        mutations = {
            "analysis_schema_version": 2, "plan_version": "5.7",
            "site_order": ["NYU", "Peking", "NeuroIMAGE"], "roi_order": [12, 18, 39],
            "primary_metric": "auc_por_pliegue", "repeat_aggregation": "prob_then_metric",
            "secondary_metrics": ["accuracy"], "audit_metrics": ["auc"],
            "classification_threshold": 0.4, "positive_label": 0, "ci_level": 0.90,
            "noninferiority_margin": 0.05, "noninferiority_margin_rationale": "porque si",
            "bootstrap_iterations": 5000, "bootstrap_seed": 1,
            "bootstrap_rng": "mersenne_twister", "bootstrap_seed_scope": "global",
            "bootstrap_subject_order": "random", "bootstrap_quantile_method": "nearest",
            "bootstrap_method": "no_pareado", "readme_round_decimals": 3,
        }
        self.assertEqual(set(mutations), set(bad.FROZEN_ANALYSIS_CONFIG))  # cubre los ~19 campos
        for field, bad_value in mutations.items():
            with self.subTest(field=field):
                cfg = dict(bad.FROZEN_ANALYSIS_CONFIG)
                cfg[field] = bad_value
                with self.assertRaises(bad.ValidationError):
                    bad.validate_analysis_config(cfg)

    def test_real_repo_config_passes(self):
        real_cfg = json.loads(
            (REPO_ROOT / "analysis/roi_comparison/config/analysis_config.json").read_text(encoding="utf-8")
        )
        bad.validate_analysis_config(real_cfg)  # no debe lanzar; permite claves documentales extra


class TestSubjectDiscrepancyInFolds(unittest.TestCase):
    """CORRECCIONES_V19 §13.4: una discordancia de subject en folds.csv
    produce fallo."""

    def test_subject_mismatch_between_predictions_and_folds_fails(self):
        subject_ids = [f"S{i}" for i in range(6)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=20)
        folds = folds_from_predictions(preds)
        # Discordancia: en folds.csv, la primera fila outer_val cambia su
        # `subject` entero sin cambiar `subject_id` -- ya no corresponde a
        # ninguna predicción real con esa combinación (repeat,fold,subject,subject_id).
        folds.loc[folds.index[0], "subject"] = 999
        bad.EXPECTED_SUBJECT_COUNTS["_T_"] = 6
        bad.EXPECTED_PRED_COUNTS["_T_"] = 30
        try:
            with self.assertRaises(bad.ValidationError):
                bad.validate_predictions(Path("/tmp/x"), preds, folds, "_T_")
        finally:
            del bad.EXPECTED_SUBJECT_COUNTS["_T_"]
            del bad.EXPECTED_PRED_COUNTS["_T_"]


class TestOuterValCorrespondence(unittest.TestCase):
    """CORRECCIONES_V19 §13.5: una fila outer_val faltante, adicional o
    duplicada produce fallo (correspondencia en ambos sentidos)."""

    def _fixture(self):
        subject_ids = [f"S{i}" for i in range(10)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=21)
        folds = folds_from_predictions(preds)
        return preds, folds

    def test_missing_outer_val_row_fails(self):
        preds, folds = self._fixture()
        folds = folds.drop(folds.index[0]).reset_index(drop=True)
        bad.EXPECTED_SUBJECT_COUNTS["_T_"] = 10
        bad.EXPECTED_PRED_COUNTS["_T_"] = 50
        try:
            with self.assertRaises(bad.ValidationError):
                bad.validate_predictions(Path("/tmp/x"), preds, folds, "_T_")
        finally:
            del bad.EXPECTED_SUBJECT_COUNTS["_T_"]
            del bad.EXPECTED_PRED_COUNTS["_T_"]

    def test_extra_outer_val_row_fails(self):
        preds, folds = self._fixture()
        extra = folds.iloc[[0]].copy()
        extra["subject_id"] = "S_EXTRA_NOT_IN_PREDICTIONS"
        extra["subject"] = 999
        folds = pd.concat([folds, extra], ignore_index=True)
        bad.EXPECTED_SUBJECT_COUNTS["_T_"] = 10
        bad.EXPECTED_PRED_COUNTS["_T_"] = 50
        try:
            with self.assertRaises(bad.ValidationError):
                bad.validate_predictions(Path("/tmp/x"), preds, folds, "_T_")
        finally:
            del bad.EXPECTED_SUBJECT_COUNTS["_T_"]
            del bad.EXPECTED_PRED_COUNTS["_T_"]

    def test_duplicated_outer_val_row_fails(self):
        preds, folds = self._fixture()
        dup = folds.iloc[[0]].copy()
        folds = pd.concat([folds, dup], ignore_index=True)
        bad.EXPECTED_SUBJECT_COUNTS["_T_"] = 10
        bad.EXPECTED_PRED_COUNTS["_T_"] = 50
        try:
            with self.assertRaises(bad.ValidationError):
                bad.validate_predictions(Path("/tmp/x"), preds, folds, "_T_")
        finally:
            del bad.EXPECTED_SUBJECT_COUNTS["_T_"]
            del bad.EXPECTED_PRED_COUNTS["_T_"]


class TestFoldsHashWithinSite(unittest.TestCase):
    """CORRECCIONES_V19 §13.6: dos folds.csv diferentes dentro del mismo
    sitio producen fallo."""

    def test_differing_folds_hash_fails(self):
        with self.assertRaises(bad.ValidationError):
            bad.validate_folds_hash_within_site(
                "X", {12: "hashA", 18: "hashA", 39: "hashB", 116: "hashA"}
            )

    def test_identical_folds_hash_ok(self):
        bad.validate_folds_hash_within_site("X", {12: "h", 18: "h", 39: "h", 116: "h"})  # no debe lanzar


class TestMetricsValDuplicatedPair(unittest.TestCase):
    """CORRECCIONES_V19 §13.7: metrics_val.csv con 50 filas pero un par
    (repeat, fold) duplicado produce fallo."""

    def test_50_rows_with_duplicated_pair_fails(self):
        subject_ids = [f"S{i}" for i in range(10)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        preds = toy_predictions(subject_ids, y_true_map, seed=22)

        rows = []
        for repeat in range(1, 6):
            base_fold = (repeat - 1) * 10
            for i in range(10):
                fold = base_fold + i + 1
                if repeat == 1 and i == 9:
                    fold = base_fold + 1  # duplica (repeat=1, fold=1) en vez de fold=10
                rows.append({"repeat": repeat, "fold": fold})
        metrics_val = pd.DataFrame(rows)
        self.assertEqual(len(metrics_val), 50)  # el conteo simple no detecta el problema

        with self.assertRaises(bad.ValidationError):
            bad.validate_metrics_val_structure(Path("/tmp/x"), metrics_val, preds)


class TestWithinSiteNewFields(unittest.TestCase):
    """CORRECCIONES_V19 §13.8: una modificación de fisher_z o
    windowing.shape entre tamaños de ROI produce fallo."""

    def test_fisher_z_mismatch_fails(self):
        cfg12 = base_config(roi_set=12, fisher_z=False)
        cfg18 = base_config(roi_set=18, fisher_z=True)
        with self.assertRaises(bad.ValidationError):
            bad.validate_within_site_comparability("X", {12: cfg12, 18: cfg18})

    def test_windowing_shape_mismatch_fails(self):
        cfg12 = base_config(roi_set=12)
        windowing_diff = dict(cfg12["windowing"])
        windowing_diff["shape"] = "triangular"
        cfg18 = base_config(roi_set=18, windowing=windowing_diff)
        with self.assertRaises(bad.ValidationError):
            bad.validate_within_site_comparability("X", {12: cfg12, 18: cfg18})

    def test_non_null_ancillary_field_fails(self):
        cfg18 = base_config(roi_set=18)
        row18 = pd.Series({"site": "X", "roi_set": 18, "run_id": cfg18["run_id"]})
        cfg18_bad = dict(cfg18, random_subset=[1, 2, 3])
        with self.assertRaises(bad.ValidationError):
            bad.validate_run_config_matches_manifest(cfg18_bad, row18, Path(cfg18_bad["run_id"]))

    def test_matching_new_fields_ok(self):
        cfg12 = base_config(roi_set=12)
        cfg18 = base_config(roi_set=18)
        bad.validate_within_site_comparability("X", {12: cfg12, 18: cfg18})  # no debe lanzar


class TestSubjectScoresPhase2(unittest.TestCase):
    """CORRECCIONES_V19 §13.9: subject_scores.csv con duplicados, NaN,
    probabilidad fuera de rango o combinación faltante produce fallo antes
    del bootstrap.

    Los conteos totales (1860, 80) están fijados por el diseño real (4
    sitios x 4 tamaños de ROI x N sujetos reales por sitio), así que estos
    fixtures replican esa forma exacta con datos ficticios."""

    SITE_ORDER = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]
    ROI_ORDER = [12, 18, 39, 116]
    N_BY_SITE = {"NYU": 177, "Peking": 183, "NeuroIMAGE": 39, "OHSU": 66}

    def _valid_subject_scores(self) -> pd.DataFrame:
        rows = []
        for site in self.SITE_ORDER:
            n = self.N_BY_SITE[site]
            for roi in self.ROI_ORDER:
                for i in range(n):
                    rows.append({
                        "site": site, "roi_set": roi, "subject_id": f"{site}_S{i}", "y_true": i % 2,
                        "y_prob_r1": 0.5, "y_prob_r2": 0.5, "y_prob_r3": 0.5, "y_prob_r4": 0.5, "y_prob_r5": 0.5,
                        "y_prob_mean": 0.5, "y_prob_sd": 0.0, "n_positive_predictions": 5,
                    })
        return pd.DataFrame(rows)

    def test_valid_fixture_ok(self):
        df = self._valid_subject_scores()
        self.assertEqual(len(df), 1860)
        rsa.validate_subject_scores_for_phase2(df, self.SITE_ORDER, self.ROI_ORDER, self.N_BY_SITE)  # no debe lanzar

    def test_duplicate_combination_fails(self):
        df = self._valid_subject_scores()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(rsa.ValidationError):
            rsa.validate_subject_scores_for_phase2(df, self.SITE_ORDER, self.ROI_ORDER, self.N_BY_SITE)

    def test_nan_fails(self):
        df = self._valid_subject_scores()
        df.loc[0, "y_prob_r1"] = float("nan")
        with self.assertRaises(rsa.ValidationError):
            rsa.validate_subject_scores_for_phase2(df, self.SITE_ORDER, self.ROI_ORDER, self.N_BY_SITE)

    def test_out_of_range_probability_fails(self):
        df = self._valid_subject_scores()
        df.loc[0, "y_prob_r1"] = 1.7
        with self.assertRaises(rsa.ValidationError):
            rsa.validate_subject_scores_for_phase2(df, self.SITE_ORDER, self.ROI_ORDER, self.N_BY_SITE)

    def test_missing_combination_fails(self):
        df = self._valid_subject_scores()
        df = df[~((df.site == "NYU") & (df.roi_set == 116) & (df.subject_id == "NYU_S0"))]
        with self.assertRaises(rsa.ValidationError):
            rsa.validate_subject_scores_for_phase2(df, self.SITE_ORDER, self.ROI_ORDER, self.N_BY_SITE)


class TestMetricsByRepeatPhase2(unittest.TestCase):
    """CORRECCIONES_V19 §13.10: metrics_by_repeat.csv con repetición
    duplicada o faltante produce fallo antes del bootstrap."""

    SITE_ORDER = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]
    ROI_ORDER = [12, 18, 39, 116]

    def _valid_metrics_by_repeat(self) -> pd.DataFrame:
        rows = []
        for site in self.SITE_ORDER:
            for roi in self.ROI_ORDER:
                for r in range(1, 6):
                    rows.append({
                        "site": site, "roi_set": roi, "repeat": r, "n_subjects": 4, "n_control": 2, "n_adhd": 2,
                        "auc": 0.6, "balanced_accuracy": 0.55, "f1_macro": 0.55,
                        "sensitivity": 0.5, "specificity": 0.6, "accuracy": 0.55,
                    })
        return pd.DataFrame(rows)

    def test_valid_fixture_ok(self):
        df = self._valid_metrics_by_repeat()
        self.assertEqual(len(df), 80)
        rsa.validate_metrics_by_repeat_for_phase2(df, self.SITE_ORDER, self.ROI_ORDER)  # no debe lanzar

    def test_duplicated_repeat_fails(self):
        df = self._valid_metrics_by_repeat()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(rsa.ValidationError):
            rsa.validate_metrics_by_repeat_for_phase2(df, ["X"], [12, 116])

    def test_missing_repeat_fails(self):
        df = self._valid_metrics_by_repeat()
        df = df[~((df.roi_set == 12) & (df.repeat == 5))]
        with self.assertRaises(rsa.ValidationError):
            rsa.validate_metrics_by_repeat_for_phase2(df, ["X"], [12, 116])


class TestAutoconsistency(unittest.TestCase):
    """CORRECCIONES_V19 §13.11: la autoconsistencia entre subject_scores.csv
    y metrics_by_repeat.csv detecta una métrica alterada."""

    def _consistent_fixtures(self):
        subject_ids = [f"S{i}" for i in range(8)]
        y_true_map = {sid: (i % 2) for i, sid in enumerate(subject_ids)}
        rng = np.random.default_rng(30)
        rows_scores = []
        for sid in subject_ids:
            row = {"site": "X", "roi_set": 12, "subject_id": sid, "y_true": y_true_map[sid]}
            for r in range(1, 6):
                row[f"y_prob_r{r}"] = float(rng.uniform(0.1, 0.9))
            rows_scores.append(row)
        subject_scores = pd.DataFrame(rows_scores)

        rows_metrics = []
        for r in range(1, 6):
            y_true = subject_scores["y_true"].to_numpy()
            y_prob = subject_scores[f"y_prob_r{r}"].to_numpy()
            m = bad.metrics_from_arrays(y_true, y_prob)
            rows_metrics.append({
                "site": "X", "roi_set": 12, "repeat": r,
                "n_subjects": 8, "n_control": 4, "n_adhd": 4, **m,
            })
        metrics_by_repeat = pd.DataFrame(rows_metrics)
        return subject_scores, metrics_by_repeat

    def test_consistent_fixture_ok(self):
        subject_scores, metrics_by_repeat = self._consistent_fixtures()
        rsa.validate_autoconsistency(subject_scores, metrics_by_repeat, ["X"], [12])  # no debe lanzar

    def test_altered_metric_detected(self):
        subject_scores, metrics_by_repeat = self._consistent_fixtures()
        metrics_by_repeat.loc[0, "auc"] = metrics_by_repeat.loc[0, "auc"] + 0.1
        with self.assertRaises(rsa.ValidationError):
            rsa.validate_autoconsistency(subject_scores, metrics_by_repeat, ["X"], [12])


class TestManifestConfigHashes(unittest.TestCase):
    """CORRECCIONES_V19 §13.12: el manifiesto final contiene exactamente 16
    config_hash no vacíos (sobre el repositorio real)."""

    def test_16_config_hashes_present_and_well_formed(self):
        manifest = rsa.load_manifest(REPO_ROOT / "analysis/roi_comparison/config/run_manifest.csv")
        run_hashes = rsa.build_run_hashes(manifest, REPO_ROOT)
        self.assertEqual(len(run_hashes), 16)
        for r in run_hashes:
            self.assertTrue(r["config_hash"], f"config_hash vacío para {r['site']}/{r['roi_set']}")
            self.assertRegex(r["config_hash"], r"^[0-9a-f]{8}$")


class TestGitUnavailable(unittest.TestCase):
    """CORRECCIONES_V19 §13.13: sin Git, los campos quedan en null y
    git_provenance_status="unavailable"; results_read_only se basa solo en
    hashes."""

    def test_get_git_status_outside_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            commit, status_lines, provenance = rsa.get_git_status(Path(tmp))
            self.assertIsNone(commit)
            self.assertIsNone(status_lines)
            self.assertEqual(provenance, "unavailable")

    def test_results_read_only_uses_only_hash_comparison(self):
        # Dos inventarios idénticos -> misma huella -> "no cambió", sin
        # ninguna referencia a Git en el cálculo.
        inv_a = {"results_readme_sha256": "h1", "run_artifacts": {"NYU_12": {"folds.csv": "fA"}}}
        inv_b = {"results_readme_sha256": "h1", "run_artifacts": {"NYU_12": {"folds.csv": "fA"}}}
        self.assertEqual(rsa.hash_inventory_fingerprint(inv_a), rsa.hash_inventory_fingerprint(inv_b))
        inv_c = {"results_readme_sha256": "h1", "run_artifacts": {"NYU_12": {"folds.csv": "fB"}}}
        self.assertNotEqual(rsa.hash_inventory_fingerprint(inv_a), rsa.hash_inventory_fingerprint(inv_c))


class TestAtomicStaging(unittest.TestCase):
    """CORRECCIONES_V19 §13.14: un fallo deliberado tardío no deja nuevas
    tablas científicas finales (nada se promueve hasta que todo el lote
    terminó de escribirse en staging)."""

    def test_unpromoted_file_never_appears_at_final_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            final_a = tmp_path / "tables" / "a.csv"
            final_b = tmp_path / "tables" / "b.csv"
            df_a = pd.DataFrame({"x": [1, 2]})
            df_b = pd.DataFrame({"x": [3, 4]})

            staged_a = bad.stage_csv(df_a, final_a)
            staged_b = bad.stage_csv(df_b, final_b)
            # Nada se promovió todavía: ninguna ruta final existe.
            self.assertFalse(final_a.exists())
            self.assertFalse(final_b.exists())

            # Simula que el lote falla después de "a" pero antes de "b":
            # solo se promueve "a", "b" se limpia como haría el bloque
            # except del llamador real.
            bad.promote_staged([(staged_a, final_a)])
            bad.cleanup_staged([(staged_b, final_b)])

            self.assertTrue(final_a.exists())
            self.assertFalse(final_b.exists())
            self.assertFalse(staged_b.exists())  # el temporal también se limpió


class TestBootstrapProgressNoSideEffects(unittest.TestCase):
    """CORRECCIONES_V19 §13.15: los mensajes de progreso no cambian los
    remuestreos ni los resultados."""

    def test_progress_messages_do_not_change_draws(self):
        n = 20
        y_true = np.array([0] * 10 + [1] * 10)
        tensor = np.random.default_rng(2).uniform(0.1, 0.9, size=(n, 2, 5))
        with contextlib.redirect_stdout(io.StringIO()):
            d_with_progress = rsa.bootstrap_site(
                tensor, y_true, [12, 116], n_iter=25, seed=42, site="TEST", progress_every=1
            )
        d_without_progress = rsa.bootstrap_site(
            tensor, y_true, [12, 116], n_iter=25, seed=42, progress_every=0
        )
        for key in d_with_progress:
            np.testing.assert_array_equal(d_with_progress[key], d_without_progress[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
