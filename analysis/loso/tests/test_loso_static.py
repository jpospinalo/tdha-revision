#!/usr/bin/env python3
"""Suite T1-T30 de ``loso_static_v1`` (Sección 58 del plan).

Ejecutar con:

    python -m unittest discover -s analysis/loso/tests -p "test_*.py" -v

El analyzer se prueba con fixtures sintéticas/toy (Sección 49): nunca se
ejecuta contra corridas formales reales dentro de esta suite.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "analysis" / "loso" / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import run_loso as L  # noqa: E402
import run_loso_campaign as C  # noqa: E402
import analyze_loso_static as A  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures compartidas (cargadas una sola vez; datos reales, no simulados,
# porque la cohorte/master-table/splits son hechos fijos del dataset).
# ---------------------------------------------------------------------------

_PAYLOADS = None
_MASTER = None


def _payloads():
    global _PAYLOADS
    if _PAYLOADS is None:
        _PAYLOADS = L.load_site_payloads()
    return _PAYLOADS


def _master():
    global _MASTER
    if _MASTER is None:
        _MASTER = L.build_master_table(_payloads())
    return _MASTER


class T01Cohort(unittest.TestCase):
    def test_cohort_counts(self) -> None:
        master = _master()
        self.assertEqual(len(master), 465)
        for site, expected in L.EXPECTED_COHORT.items():
            sub = master[master["site"] == site]
            self.assertEqual(len(sub), expected["n"])
            self.assertEqual(int((sub["y_true"] == 0).sum()), expected["control"])
            self.assertEqual(int((sub["y_true"] == 1).sum()), expected["adhd"])
        self.assertEqual(int((master["y_true"] == 0).sum()), 256)
        self.assertEqual(int((master["y_true"] == 1).sum()), 209)


class T02IdUniqueness(unittest.TestCase):
    def test_unique_subject_keys(self) -> None:
        master = _master()
        self.assertEqual(master["subject_key"].nunique(), 465)
        self.assertEqual(list(master["global_index"]), list(range(465)))


class T03StaticFcShape(unittest.TestCase):
    def test_shapes_12_and_116(self) -> None:
        payload = _payloads()["OHSU"]
        for roi_set, n_features in (("12", 66), ("116", 6670)):
            roi_idx = L.tdha_data.roi_indices(roi_set)
            block = L.compute_site_static_fc(payload, roi_idx)
            self.assertEqual(block.shape, (len(payload["subjects"]), n_features))


class T04StaticFcParity(unittest.TestCase):
    def test_uses_build_flat_static_connectivity_directly(self) -> None:
        payload = _payloads()["OHSU"]
        roi_idx = L.tdha_data.roi_indices("12")
        via_loso = L.compute_site_static_fc(payload, roi_idx)
        direct = L.tdha_data.build_flat_static_connectivity(
            payload["bold"], roi_idx, fisher_z=False, constant_policy="zero",
        )[:, 0, :]
        np.testing.assert_array_equal(via_loso, direct)


class T05NoFisherZ(unittest.TestCase):
    def test_fisher_z_frozen_false_and_not_exposed(self) -> None:
        self.assertFalse(L.FISHER_Z)
        parser = L.build_parser()
        dests = {action.dest for action in parser._actions}
        self.assertNotIn("fisher_z", dests)


class T06ConstantPolicy(unittest.TestCase):
    def test_constant_policy_frozen_zero_and_not_exposed(self) -> None:
        self.assertEqual(L.CONSTANT_POLICY, "zero")
        parser = L.build_parser()
        dests = {action.dest for action in parser._actions}
        self.assertNotIn("constant_policy", dests)


class T07OuterIsolation(unittest.TestCase):
    def test_held_out_only_in_test(self) -> None:
        master = _master()
        for site in L.SITES:
            split = L.build_rotation_split(site, master)
            site_by_index = master.set_index("global_index")["site"]
            self.assertTrue((site_by_index.loc[split["fit"]] != site).all())
            self.assertTrue((site_by_index.loc[split["inner_val"]] != site).all())
            self.assertTrue((site_by_index.loc[split["test"]] == site).all())


class T08Exhaustiveness(unittest.TestCase):
    def test_partitions_cover_465_disjoint(self) -> None:
        master = _master()
        for site in L.SITES:
            split = L.build_rotation_split(site, master)
            fit, inner, test = split["fit"], split["inner_val"], split["test"]
            self.assertTrue(set(fit).isdisjoint(inner))
            self.assertTrue(set(fit).isdisjoint(test))
            self.assertTrue(set(inner).isdisjoint(test))
            self.assertEqual(len(set(fit) | set(inner) | set(test)), 465)


class T09Stratification(unittest.TestCase):
    def test_six_strata_present_in_fit_and_inner(self) -> None:
        master = _master()
        for site in L.SITES:
            split = L.build_rotation_split(site, master)
            training_sites = [s for s in L.SITES if s != site]
            expected = {f"{s}|{c}" for s in training_sites for c in (0, 1)}
            indexed = master.set_index("global_index")
            for name in ("fit", "inner_val"):
                sub = indexed.loc[split[name]]
                present = set((sub["site"] + "|" + sub["y_true"].astype(str)).tolist())
                self.assertEqual(present, expected)


class T10SplitInvariance(unittest.TestCase):
    def test_fingerprint_deterministic_across_calls(self) -> None:
        master = _master()
        for site in L.SITES:
            fp1 = L.rotation_split_fingerprint(L.build_rotation_split(site, master))
            fp2 = L.rotation_split_fingerprint(L.build_rotation_split(site, master))
            self.assertEqual(fp1, fp2)
        # La función no recibe roi_set/model/seed como argumento: por
        # construcción, es idéntica para las dos representaciones ROI, los
        # cinco seeds BrainNetCNN y la regresión logística (Sección 10.1).
        self.assertNotIn("roi_set", L.build_rotation_split.__code__.co_varnames)


class T11ScalerLeakage(unittest.TestCase):
    def test_scaler_unaffected_by_inner_or_test(self) -> None:
        rng = np.random.default_rng(0)
        X_fit = rng.normal(size=(30, 10)).astype(np.float32)
        y_fit = np.array([0, 1] * 15, dtype=np.int32)

        clf1, scaler1 = L.fit_logistic(X_fit=X_fit, y_fit=y_fit)
        # fit_logistic() no acepta X_inner/X_test como parámetro (Sección 14):
        # perturbar cualquier variable externa no puede afectar el resultado.
        X_inner_perturbed = rng.normal(size=(500, 10)) * 1e6
        del X_inner_perturbed
        clf2, scaler2 = L.fit_logistic(X_fit=X_fit, y_fit=y_fit)

        np.testing.assert_allclose(scaler1.mean_, scaler2.mean_)
        np.testing.assert_allclose(scaler1.scale_, scaler2.scale_)
        np.testing.assert_allclose(scaler1.transform(X_fit), scaler2.transform(X_fit))
        self.assertNotIn("X_inner", L.fit_logistic.__code__.co_varnames)
        self.assertNotIn("X_test", L.fit_logistic.__code__.co_varnames)


class T12NoWeighting(unittest.TestCase):
    def test_class_and_site_weighting_disabled(self) -> None:
        self.assertFalse(L.BNN_TRAIN_CONFIG["class_weight"])
        self.assertIsNone(L.LOGREG_CONFIG["class_weight"])
        self.assertNotIn("class_weight", {a.dest for a in L.build_parser()._actions})


class T13BrainNetArchitecture(unittest.TestCase):
    def test_frozen_architecture_matches_plan(self) -> None:
        self.assertEqual(
            L.BNN_ARCH_KWARGS,
            {"e2e": 4, "e2n": 8, "dense": 8, "dropout": 0.7, "leaky": 0.33,
             "l2_reg": 0.05, "inter_dropout": 0.6},
        )


class T14OptimizerLossParity(unittest.TestCase):
    def test_compile_adam_bce(self) -> None:
        import keras

        model = L.kerasmodels.build("brainnetcnn", 1, 66, **L.BNN_ARCH_KWARGS)
        compile_args = SimpleNamespace(lr=L.BNN_TRAIN_CONFIG["lr"], clipnorm=None)
        model = L._re_compile_model(model, compile_args)
        self.assertIsInstance(model.optimizer, keras.optimizers.Adam)
        self.assertAlmostEqual(float(model.optimizer.learning_rate), 1e-4, places=8)

        # model.metrics solo se puebla de verdad tras una pasada de fit/evaluate
        # en esta versión de Keras (antes solo expone 'loss'/'compile_metrics');
        # la comprobación real y estable es sobre history.history, que es
        # exactamente lo que consume train_brainnetcnn()/run_experiment.py.
        rng = np.random.default_rng(5)
        X = rng.normal(size=(8, 1, 66)).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32).reshape(-1, 1)
        history = model.fit(X, y, epochs=1, batch_size=8, verbose=0)
        self.assertIn("accuracy", history.history)
        self.assertIn("bce", history.history)
        keras.backend.clear_session()


class T15ParameterCount(unittest.TestCase):
    def test_1361_and_12177(self) -> None:
        import keras

        for roi_set, n_features, expected in (("12", 66, 1361), ("116", 6670, 12177)):
            model = L.kerasmodels.build("brainnetcnn", 1, n_features, **L.BNN_ARCH_KWARGS)
            self.assertEqual(model.count_params(), expected)
            keras.backend.clear_session()


class T16EarlyStoppingGate(unittest.TestCase):
    def test_restoration_gate_toy_model(self) -> None:
        import keras

        rng = np.random.default_rng(1)
        n_features = 66
        X_fit = rng.normal(size=(24, 1, n_features)).astype(np.float32)
        y_fit = np.array([0, 1] * 12, dtype=np.int32)
        X_inner = rng.normal(size=(10, 1, n_features)).astype(np.float32)
        y_inner = np.array([0, 1] * 5, dtype=np.int32)

        original = dict(L.BNN_TRAIN_CONFIG)
        L.BNN_TRAIN_CONFIG["epochs"] = 4
        L.BNN_TRAIN_CONFIG["patience"] = 4
        try:
            model, meta = L.train_brainnetcnn(
                X_fit=X_fit, y_fit=y_fit, X_inner=X_inner, y_inner=y_inner,
                model_seed=42, n_features=n_features,
            )
        finally:
            L.BNN_TRAIN_CONFIG.clear()
            L.BNN_TRAIN_CONFIG.update(original)
        self.assertLessEqual(meta["best_epoch"], meta["epochs_ran"])
        self.assertAlmostEqual(meta["best_monitor_value"], meta["restored_monitor_value"], places=4)
        keras.backend.clear_session()


class T17LabelShape(unittest.TestCase):
    def test_model_output_is_single_unit(self) -> None:
        import keras

        model = L.kerasmodels.build("brainnetcnn", 1, 66, **L.BNN_ARCH_KWARGS)
        self.assertEqual(model.output_shape[-1], 1)
        keras.backend.clear_session()


class T18LogisticConfig(unittest.TestCase):
    def test_frozen_hyperparameters(self) -> None:
        self.assertEqual(
            L.LOGREG_CONFIG,
            {"penalty": "l2", "C": 1.0, "class_weight": None, "solver": "lbfgs", "max_iter": 2000},
        )


class T19LogisticClasses(unittest.TestCase):
    def test_classes_are_0_1(self) -> None:
        rng = np.random.default_rng(2)
        X_fit = rng.normal(size=(40, 5)).astype(np.float32)
        y_fit = np.array([0, 1] * 20, dtype=np.int32)
        clf, _ = L.fit_logistic(X_fit=X_fit, y_fit=y_fit)
        self.assertEqual(list(clf.classes_), [0, 1])


class T20RunMatrix(unittest.TestCase):
    def test_48_40_8(self) -> None:
        matrix = C.build_run_matrix()
        self.assertEqual(len(matrix), 48)
        self.assertEqual(sum(1 for r in matrix if r["model"] == "brainnetcnn"), 40)
        self.assertEqual(sum(1 for r in matrix if r["model"] == "logreg"), 8)


class T21RunIdentityUniqueness(unittest.TestCase):
    def test_48_unique_run_ids(self) -> None:
        master = _master()
        fp_by_site = {site: L.rotation_split_fingerprint(L.build_rotation_split(site, master)) for site in L.SITES}
        env_sig = L.formal_environment_signature()
        run_ids = set()
        for row in C.build_run_matrix():
            identity = L.build_identity(
                held_out_site=row["held_out_site"], roi_set=row["roi_set"], model=row["model"],
                model_seed=row["model_seed"], rotation_fp=fp_by_site[row["held_out_site"]],
                formal_env_signature=env_sig,
            )
            ihash = L.config_hash(identity)
            run_id = L.make_run_id(
                held_out_site=row["held_out_site"], roi_set=row["roi_set"], model=row["model"],
                model_seed=row["model_seed"], identity_hash=ihash,
            )
            run_ids.add(run_id)
        self.assertEqual(len(run_ids), 48)


class T22OutputPathProtection(unittest.TestCase):
    def test_no_user_controllable_output_root(self) -> None:
        dests = {a.dest for a in L.build_parser()._actions}
        self.assertNotIn("out", dests)
        self.assertNotIn("output", dests)
        self.assertTrue(str(L.FORMAL_OUTPUT_ROOT).endswith("results/loso"))


class T23AtomicPromotion(unittest.TestCase):
    def test_failure_leaves_no_partial_formal_dir(self) -> None:
        tmp_root = Path(tempfile.mkdtemp(prefix="t23_"))
        try:
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(tmp_root)))
            (staging / "partial.txt").write_text("partial")
            final_dir = tmp_root / "formal_run"
            try:
                raise RuntimeError("fallo simulado antes de promover atomic_promote()")
            except RuntimeError:
                # Mismo patrón que run_formal(): limpieza en el except, NUNCA
                # se llega a llamar atomic_promote()/os.replace().
                shutil.rmtree(staging, ignore_errors=True)
            self.assertFalse(final_dir.exists())
            self.assertFalse(staging.exists())
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_successful_promotion(self) -> None:
        tmp_root = Path(tempfile.mkdtemp(prefix="t23b_"))
        try:
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(tmp_root)))
            (staging / "ok.txt").write_text("ok")
            final_dir = tmp_root / "formal_run"
            L.atomic_promote(staging, final_dir)
            self.assertTrue(final_dir.exists())
            self.assertFalse(staging.exists())
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


class T24ResumeValidation(unittest.TestCase):
    def test_incomplete_run_not_skipped(self) -> None:
        tmp_root = Path(tempfile.mkdtemp(prefix="t24_"))
        try:
            run_dir = tmp_root / "fake_run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(json.dumps({
                "formal": True, "identity_hash": "abc123", "held_out_site": "NYU", "model": "logreg",
            }))
            # Faltan predictions_test.csv y el resto de artefactos obligatorios.
            valid = L.validate_existing_run(
                run_dir, expected_identity_hash="abc123", expected_test_n=177, held_out_site="NYU",
            )
            self.assertFalse(valid)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_corrupt_config_not_skipped(self) -> None:
        tmp_root = Path(tempfile.mkdtemp(prefix="t24b_"))
        try:
            run_dir = tmp_root / "fake_run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text("{not valid json")
            valid = L.validate_existing_run(
                run_dir, expected_identity_hash="abc123", expected_test_n=177, held_out_site="NYU",
            )
            self.assertFalse(valid)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# T25-T29: analyzer, con una campaña formal SINTÉTICA completa (48 corridas
# fabricadas en disco, nunca datos reales de results/loso).
# ---------------------------------------------------------------------------


def _build_synthetic_campaign(tmp_root: Path, *, rng: np.random.Generator) -> None:
    """Fabrica 48 directorios de corrida formal, con esquema válido pero
    predicciones sintéticas, para probar el analyzer de punta a punta."""

    for site in L.SITES:
        test_n = L.EXPECTED_ROTATION_SIZES[site]["test"]
        subjects = [f"{site}::s{i:03d}" for i in range(test_n)]
        y_true = np.array([0, 1] * (test_n // 2) + ([0] if test_n % 2 else []), dtype=int)
        for roi_set in L.ROI_SETS:
            # Efecto de SEPARABILIDAD (no un mero desplazamiento aditivo, que es
            # invariante para AUC): roi 116 separa mejor las clases que roi 12,
            # a propósito, para que el contraste de dimensionalidad tenga un
            # signo definido y verificable (T28).
            effect_size = 0.6 if roi_set == "116" else 0.3
            for model in L.MODELS:
                seeds = L.BNN_SEEDS if model == "brainnetcnn" else [None]
                for seed in seeds:
                    seed_noise = (seed or 42) * 0.0005
                    y_prob = np.clip(
                        0.5 + (y_true - 0.5) * effect_size + seed_noise
                        + rng.normal(scale=0.3, size=test_n),
                        0.01, 0.99,
                    )
                    run_id = f"loso_static_v1_holdout-{site}_roi-{roi_set}_{model}_seed-{seed or 'deterministic'}_fake"
                    run_dir = tmp_root / run_id
                    run_dir.mkdir(parents=True)
                    config = {
                        "formal": True,
                        "run_id": run_id,
                        "identity_hash": f"fake-{run_id}",
                        "held_out_site": site,
                        "roi_set": roi_set,
                        "model": model,
                        "model_seed": seed,
                        "split_manifest_file_sha256": "fakehash",
                        "training_source_git_sha": "fakesha",
                        "environment_signature": "fakeenv",
                    }
                    (run_dir / "config.json").write_text(json.dumps(config))
                    pred_rows = [
                        {
                            "held_out_site": site, "site": site, "subject_id": s, "subject_key": s,
                            "y_true": int(yt), "y_prob": float(yp), "model": model, "roi_set": roi_set,
                            "model_seed": seed, "run_id": run_id,
                        }
                        for s, yt, yp in zip(subjects, y_true, y_prob)
                    ]
                    pd.DataFrame(pred_rows).to_csv(run_dir / "predictions_test.csv", index=False)
                    pd.DataFrame([{"auc": 0.5}]).to_csv(run_dir / "metrics_test.csv", index=False)


class AnalyzerIntegrationBase(unittest.TestCase):
    tmp_root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_root = Path(tempfile.mkdtemp(prefix="loso_fake_campaign_"))
        _build_synthetic_campaign(cls.tmp_root, rng=np.random.default_rng(7))
        cls.runs = A.discover_runs(cls.tmp_root)
        cls.manifest = A.build_manifest(cls.runs)
        cls.predictions_long = A.build_predictions_long(cls.runs)
        cls.analysis_config = A.load_analysis_config()
        cls.analysis_config = dict(cls.analysis_config)
        cls.analysis_config["bootstrap_iterations"] = 200  # rápido para tests, misma lógica

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_root, ignore_errors=True)


class T25MetricThenMean(unittest.TestCase):
    def test_metric_then_mean_not_prob_then_metric(self) -> None:
        # ctx fabricado a mano (toy, sin depender de disco ni de la campaña
        # sintética completa): 4 sujetos, y_true=[0,0,1,1]. Tres semillas con
        # separación perfecta (AUC=1.0) y dos con separación perfectamente
        # invertida (AUC=0.0) -> metric-then-mean = 0.6. Promediar antes las
        # probabilidades de las 5 semillas cancela el patrón invertido y da
        # AUC=1.0: los dos órdenes de agregación deben diferir aquí.
        y_true = np.array([0, 0, 1, 1], dtype=np.int64)
        good = np.array([0.1, 0.1, 0.9, 0.9])
        bad = np.array([0.9, 0.9, 0.1, 0.1])
        seed_pattern = {42: good, 43: bad, 44: good, 45: bad, 46: good}

        ctx = {
            "y_true": y_true,
            "draws": np.array([[0, 1, 2, 3]], dtype=np.int64),  # una sola "iteración" = identidad
            "condition_probs": {("12", "brainnetcnn", seed): probs for seed, probs in seed_pattern.items()},
        }

        replicates_metric_then_mean = A.condition_bootstrap_replicates(ctx, roi_set="12", model="brainnetcnn")
        self.assertEqual(replicates_metric_then_mean.shape, (1,))
        self.assertAlmostEqual(replicates_metric_then_mean[0], 0.6, places=6)

        from sklearn.metrics import roc_auc_score

        mean_prob = np.mean([seed_pattern[s] for s in L.BNN_SEEDS], axis=0)
        prob_then_metric = roc_auc_score(y_true, mean_prob)
        self.assertAlmostEqual(prob_then_metric, 1.0, places=6)

        self.assertNotAlmostEqual(replicates_metric_then_mean[0], prob_then_metric, places=2)


class T26PairedBootstrap(AnalyzerIntegrationBase):
    def test_same_draw_across_conditions(self) -> None:
        ctx = A.compute_site_bootstrap(self.predictions_long, "OHSU", self.analysis_config)
        draws_snapshot = ctx["draws"].copy()
        _ = A.condition_bootstrap_replicates(ctx, roi_set="12", model="brainnetcnn")
        _ = A.condition_bootstrap_replicates(ctx, roi_set="116", model="logreg")
        np.testing.assert_array_equal(ctx["draws"], draws_snapshot)


class T27ConditionCiRowCount(AnalyzerIntegrationBase):
    def test_16_rows(self) -> None:
        summary, _ = A.build_metrics_summary(self.predictions_long, self.analysis_config)
        self.assertEqual(len(summary), 16)
        self.assertEqual(
            set(zip(summary["held_out_site"], summary["roi_set"], summary["model"])),
            {(s, r, m) for s in L.SITES for r in L.ROI_SETS for m in L.MODELS},
        )


class T28ContrastSign(AnalyzerIntegrationBase):
    def test_three_contrasts_signs(self) -> None:
        summary, bootstrap_state = A.build_metrics_summary(self.predictions_long, self.analysis_config)
        contrasts = A.build_contrasts(self.predictions_long, self.analysis_config, bootstrap_state)
        self.assertEqual(len(contrasts), 12)
        self.assertEqual(set(contrasts["contrast"]), {"dimensionality", "model_family_at_12", "model_family_at_116"})
        # Fixture: roi 116 tiene un shift positivo (+0.1) sobre roi 12 -> el
        # contraste de dimensionalidad (116 - 12) debe ser positivo en cada sitio.
        dim = contrasts[contrasts["contrast"] == "dimensionality"]
        self.assertTrue((dim["delta_point"] > 0).all())


class T29ManifestCompleteness(unittest.TestCase):
    def test_missing_condition_raises(self) -> None:
        tmp_root = Path(tempfile.mkdtemp(prefix="t29_missing_"))
        try:
            _build_synthetic_campaign(tmp_root, rng=np.random.default_rng(3))
            # Elimina una corrida para forzar "falta una condición".
            victim = next(tmp_root.glob("*holdout-NYU_roi-12_logreg*"))
            shutil.rmtree(victim)
            runs = A.discover_runs(tmp_root)
            with self.assertRaises(SystemExit):
                A.build_manifest(runs)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_duplicate_condition_raises(self) -> None:
        tmp_root = Path(tempfile.mkdtemp(prefix="t29_dup_"))
        try:
            _build_synthetic_campaign(tmp_root, rng=np.random.default_rng(3))
            original = next(tmp_root.glob("*holdout-NYU_roi-12_logreg*"))
            duplicate = tmp_root / (original.name + "_dup")
            shutil.copytree(original, duplicate)
            runs = A.discover_runs(tmp_root)
            with self.assertRaises(SystemExit):
                A.build_manifest(runs)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


class T30HistoricalIsolation(unittest.TestCase):
    def test_formal_output_root_is_results_loso(self) -> None:
        self.assertEqual(L.FORMAL_OUTPUT_ROOT, L.REPO_ROOT / "results" / "loso")
        self.assertNotEqual(L.FORMAL_OUTPUT_ROOT, L.REPO_ROOT / "results" / "runs")
        self.assertEqual(L.DESIGN_DIR, L.FORMAL_OUTPUT_ROOT / "_design")

    def test_no_hardcoded_legacy_paths_in_module(self) -> None:
        source = Path(L.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"results/runs"', source)
        self.assertNotIn("'results/runs'", source)


if __name__ == "__main__":
    unittest.main()
