#!/usr/bin/env python3
"""Tests A1-A20 del plan de cierre de análisis (Sección 42).

Ejercitan ``run_closeout_audit()`` (Gates A-Q) y las utilidades de provenance
del analyzer de cierre CONTRA LOS DATOS REALES de la campaña ``loso_static_v1``
ya almacenada en ``results/loso/`` — nunca los modifican: cada test que
necesita provocar un fallo copia una corrida a un directorio temporal, aplica
UNA mutación puntual ahí, y reconstruye la lista ``runs`` en memoria con esa
única copia sustituida. El resto de la campaña (47/48 corridas + design +
splits) permanece siendo el dato real, sin tocar disco de ``results/loso/``.

Ejecutar junto con la suite histórica:

    python -m unittest discover -s analysis/loso/tests -p "test_*.py" -v
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "analysis" / "loso" / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import run_loso as L  # noqa: E402
import analyze_loso_static as A  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures compartidas: campaña real, cargada una sola vez (solo lectura).
# ---------------------------------------------------------------------------

_RUNS = None
_DESIGN = None
_SPLITS = None


def _real_runs() -> list[dict[str, Any]]:
    global _RUNS
    if _RUNS is None:
        _RUNS = A.discover_runs()
    return _RUNS


def _real_design_and_splits() -> tuple[dict[str, Any], pd.DataFrame]:
    global _DESIGN, _SPLITS
    if _DESIGN is None:
        _DESIGN, _SPLITS = A.load_design_and_splits()
    return _DESIGN, _SPLITS


def _pick_run(runs: list[dict[str, Any]], needle: str) -> int:
    for i, run in enumerate(runs):
        if needle in run["config"]["run_id"]:
            return i
    raise AssertionError(f"no se encontró ninguna corrida con {needle!r} en run_id")


def _mutate_copy(
    run: dict[str, Any],
    tmp_root: Path,
    *,
    config_patch: dict[str, Any] | None = None,
    predictions_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    metrics_test_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    split_membership_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Copia UNA corrida real a un directorio temporal y aplica una mutación
    puntual a uno de sus archivos, sin tocar jamás ``results/loso/`` real."""

    dst = tmp_root / run["run_dir"].name
    shutil.copytree(run["run_dir"], dst)

    if config_patch:
        config_path = dst / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(config_patch)
        config_path.write_text(json.dumps(config), encoding="utf-8")

    if predictions_fn:
        p = dst / "predictions_test.csv"
        df = pd.read_csv(p)
        predictions_fn(df).to_csv(p, index=False)

    if metrics_test_fn:
        p = dst / "metrics_test.csv"
        df = pd.read_csv(p)
        metrics_test_fn(df).to_csv(p, index=False)

    if split_membership_fn:
        p = dst / "split_membership.csv"
        df = pd.read_csv(p)
        split_membership_fn(df).to_csv(p, index=False)

    return A.load_formal_run(dst)


class AuditGateTestBase(unittest.TestCase):
    tmp_root: Path

    def setUp(self) -> None:
        self.tmp_root = Path(tempfile.mkdtemp(prefix="loso_audit_mut_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _runs_with_one_mutated(self, needle: str, **mutation_kwargs: Any) -> list[dict[str, Any]]:
        runs = list(_real_runs())
        idx = _pick_run(runs, needle)
        runs[idx] = _mutate_copy(runs[idx], self.tmp_root, **mutation_kwargs)
        return runs

    def assertAuditFails(self, runs: list[dict[str, Any]], *, gate_substring: str) -> None:
        design, splits = _real_design_and_splits()
        with self.assertRaises(SystemExit) as ctx:
            A.run_closeout_audit(runs, design, splits)
        self.assertIn(gate_substring, str(ctx.exception))


class A1MixedTrainingSourceFails(AuditGateTestBase):
    def test_mixed_training_source_git_sha_fails(self) -> None:
        runs = self._runs_with_one_mutated(
            "holdout-NYU_roi-12_brainnetcnn_seed-42",
            config_patch={"training_source_git_sha": "0" * 40},
        )
        self.assertAuditFails(runs, gate_substring="gate=B")


class A2MixedTrainingEnvironmentFails(AuditGateTestBase):
    def test_mixed_environment_signature_fails(self) -> None:
        runs = self._runs_with_one_mutated(
            "holdout-Peking_roi-116_logreg",
            config_patch={"environment_signature": "deadbeef00000000"},
        )
        self.assertAuditFails(runs, gate_substring="gate=C")


class A3DuplicateIdentityFails(AuditGateTestBase):
    def test_duplicate_run_id_fails(self) -> None:
        runs = list(_real_runs())
        runs.append(runs[0])  # misma identidad, duplicada
        self.assertAuditFails(runs, gate_substring="gate=E")


class A4WrongSplitFingerprintFails(AuditGateTestBase):
    def test_wrong_rotation_split_fingerprint_fails(self) -> None:
        runs = self._runs_with_one_mutated(
            "holdout-OHSU_roi-12_brainnetcnn_seed-44",
            config_patch={"rotation_split_fingerprint": "0000000000000000"},
        )
        self.assertAuditFails(runs, gate_substring="gate=G")


class A5SplitMembershipMismatchFails(AuditGateTestBase):
    def test_split_membership_mismatch_fails(self) -> None:
        def corrupt(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.loc[df.index[0], "y_true"] = 1 - int(df.loc[df.index[0], "y_true"])
            return df

        runs = self._runs_with_one_mutated(
            "holdout-NeuroIMAGE_roi-116_logreg",
            split_membership_fn=corrupt,
        )
        self.assertAuditFails(runs, gate_substring="gate=H")


class A6WeightingEnabledFails(AuditGateTestBase):
    def test_class_weight_true_fails(self) -> None:
        runs = self._runs_with_one_mutated(
            "holdout-NYU_roi-116_brainnetcnn_seed-43",
            config_patch={"class_weight": True},
        )
        self.assertAuditFails(runs, gate_substring="gate=J")


class A7WrongRepresentationOrFisherZFails(AuditGateTestBase):
    def test_fisher_z_true_fails(self) -> None:
        runs = self._runs_with_one_mutated(
            "holdout-Peking_roi-12_brainnetcnn_seed-45",
            config_patch={"fisher_z": True},
        )
        self.assertAuditFails(runs, gate_substring="gate=J")

    def test_non_static_representation_fails(self) -> None:
        runs = self._runs_with_one_mutated(
            "holdout-OHSU_roi-116_logreg",
            config_patch={"representation": "windowed"},
        )
        self.assertAuditFails(runs, gate_substring="gate=J")


class A8MetricMismatchFails(AuditGateTestBase):
    def test_stored_auc_does_not_match_recomputed_fails(self) -> None:
        def corrupt(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.loc[0, "auc"] = 0.0123456789
            return df

        runs = self._runs_with_one_mutated(
            "holdout-NYU_roi-12_logreg",
            metrics_test_fn=corrupt,
        )
        self.assertAuditFails(runs, gate_substring="gate=M")


class A9MissingPredictionRowFails(AuditGateTestBase):
    def test_missing_prediction_row_fails(self) -> None:
        def drop_one(df: pd.DataFrame) -> pd.DataFrame:
            return df.iloc[1:].reset_index(drop=True)

        runs = self._runs_with_one_mutated(
            "holdout-NeuroIMAGE_roi-12_brainnetcnn_seed-46",
            predictions_fn=drop_one,
        )
        self.assertAuditFails(runs, gate_substring="gate=L")


class A10DuplicatePredictionSubjectFails(AuditGateTestBase):
    def test_duplicate_subject_key_fails(self) -> None:
        def duplicate_row(df: pd.DataFrame) -> pd.DataFrame:
            return pd.concat([df, df.iloc[[0]]], ignore_index=True)

        runs = self._runs_with_one_mutated(
            "holdout-OHSU_roi-12_logreg",
            predictions_fn=duplicate_row,
        )
        self.assertAuditFails(runs, gate_substring="gate=L")


class A11InvalidProbabilityFails(AuditGateTestBase):
    def test_y_prob_out_of_range_fails(self) -> None:
        def corrupt(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df.loc[0, "y_prob"] = 1.5
            return df

        runs = self._runs_with_one_mutated(
            "holdout-Peking_roi-116_brainnetcnn_seed-42",
            predictions_fn=corrupt,
        )
        self.assertAuditFails(runs, gate_substring="gate=L")


class A12WrongModelRoiMatrixFails(AuditGateTestBase):
    def test_arch_mismatch_fails(self) -> None:
        wrong_arch = dict(L.BNN_ARCH_KWARGS)
        wrong_arch["dropout"] = 0.1234
        runs = self._runs_with_one_mutated(
            "holdout-NYU_roi-12_brainnetcnn_seed-44",
            config_patch={"arch": wrong_arch},
        )
        self.assertAuditFails(runs, gate_substring="gate=K")


class A13WrongConditionSummaryRowCountFails(unittest.TestCase):
    def test_secondary_metrics_wrong_row_count_raises(self) -> None:
        runs = _real_runs()
        metrics_by_run = A.build_metrics_by_run(runs)
        truncated = metrics_by_run[~(
            (metrics_by_run["held_out_site"] == "NYU")
            & (metrics_by_run["roi_set"] == "12")
            & (metrics_by_run["model"] == "logreg")
        )]
        with self.assertRaises(SystemExit):
            A.build_secondary_metrics(truncated)


class A14WrongContrastRowCountFails(unittest.TestCase):
    def test_truncated_contrast_specs_raises(self) -> None:
        runs = _real_runs()
        analysis_config = dict(A.load_analysis_config())
        analysis_config["bootstrap_iterations"] = 50  # rápido, misma lógica
        predictions_long = A.build_predictions_long(runs)
        _, bootstrap_state = A.build_metrics_summary(predictions_long, analysis_config)

        original_specs = A.CONTRAST_SPECS
        try:
            A.CONTRAST_SPECS = original_specs[:2]  # 2 tipos x 4 sitios = 8, se esperan 12
            with self.assertRaises(SystemExit):
                A.build_contrasts(predictions_long, analysis_config, bootstrap_state)
        finally:
            A.CONTRAST_SPECS = original_specs


class A15FileVsSemanticHashesDistinct(unittest.TestCase):
    def test_file_hash_sensitive_to_formatting_semantic_hash_is_not(self) -> None:
        tmp_root = Path(tempfile.mkdtemp(prefix="a15_"))
        try:
            obj = {"b": 1, "a": 2}
            p1 = tmp_root / "one.json"
            p2 = tmp_root / "two.json"
            p1.write_text(json.dumps(obj, indent=2), encoding="utf-8")  # formato "bonito"
            p2.write_text(json.dumps(obj), encoding="utf-8")  # formato compacto, mismo objeto

            file_hash_1 = A._full_sha256_file(p1)
            file_hash_2 = A._full_sha256_file(p2)
            self.assertNotEqual(file_hash_1, file_hash_2, "distinto byte-a-byte -> file hash debe diferir")

            semantic_1 = A._semantic_sha256(json.loads(p1.read_text()))
            semantic_2 = A._semantic_sha256(json.loads(p2.read_text()))
            self.assertEqual(semantic_1, semantic_2, "mismo contenido semántico -> semantic hash debe coincidir")
            self.assertNotEqual(semantic_1, file_hash_1, "semantic y file hash son conceptos distintos")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


class A16NewFullShaFieldsAre64Hex(unittest.TestCase):
    def test_full_sha256_fields_are_64_hex_chars(self) -> None:
        runs = _real_runs()
        design, _ = _real_design_and_splits()
        manifest = A.build_manifest(runs)
        provenance = A.build_provenance_manifest(
            runs=runs, design=design, manifest=manifest,
            closeout_analysis_source_git_sha="f" * 40,
            original_analysis_source_git_sha="e" * 40,
            original_bootstrap_manifest=None,
            output_dir_for_hashing=A.OUTPUT_DIR,
        )

        def is_64_hex(value: Any) -> bool:
            return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)

        self.assertTrue(is_64_hex(provenance["design"]["file_sha256"]))
        self.assertTrue(is_64_hex(provenance["design"]["semantic_sha256"]))
        self.assertTrue(is_64_hex(provenance["split_manifest"]["file_sha256"]))
        self.assertTrue(is_64_hex(provenance["analysis_config"]["file_sha256"]))
        self.assertTrue(is_64_hex(provenance["implementation_spec"]["file_sha256"]))
        self.assertTrue(is_64_hex(provenance["code"]["run_loso.py"]["full_sha256"]))
        for entry in provenance["formal_runs"][:5]:
            self.assertTrue(is_64_hex(entry["config_file_sha256"]))
            self.assertTrue(is_64_hex(entry["config_semantic_sha256"]))
            self.assertTrue(is_64_hex(entry["predictions_file_sha256"]))


class A17LegacyPrefix16Preserved(unittest.TestCase):
    def test_historical_prefix16_fields_are_16_hex_chars(self) -> None:
        runs = _real_runs()
        design, _ = _real_design_and_splits()
        manifest = A.build_manifest(runs)
        provenance = A.build_provenance_manifest(
            runs=runs, design=design, manifest=manifest,
            closeout_analysis_source_git_sha="f" * 40,
            original_analysis_source_git_sha="e" * 40,
            original_bootstrap_manifest=None,
            output_dir_for_hashing=A.OUTPUT_DIR,
        )
        prefix16 = provenance["design"]["historical_file_sha256_prefix16"]
        self.assertEqual(len(prefix16), 16)
        for site in L.SITES:
            self.assertEqual(len(provenance["inputs"]["BOLD"][site]["historical_prefix16"]), 16)
        for roi_set in A.ROI_SETS:
            self.assertEqual(len(provenance["inputs"]["roi_indices"][roi_set]["historical_prefix16"]), 16)


class A18FeatureFullHashIsNotFabricated(unittest.TestCase):
    def test_feature_matrices_only_carry_prefix16(self) -> None:
        runs = _real_runs()
        design, _ = _real_design_and_splits()
        manifest = A.build_manifest(runs)
        provenance = A.build_provenance_manifest(
            runs=runs, design=design, manifest=manifest,
            closeout_analysis_source_git_sha="f" * 40,
            original_analysis_source_git_sha="e" * 40,
            original_bootstrap_manifest=None,
            output_dir_for_hashing=A.OUTPUT_DIR,
        )
        for key, entry in provenance["inputs"]["feature_matrices"].items():
            self.assertIn("historical_prefix16_only", entry)
            self.assertEqual(len(entry["historical_prefix16_only"]), 16)
            self.assertNotIn("full_sha256", entry)


class A19SensitivityMapsToRawRecall(unittest.TestCase):
    def test_sensitivity_is_an_alias_of_recall(self) -> None:
        self.assertEqual(A.SECONDARY_METRIC_COLUMNS["sensitivity"], "recall")
        runs = _real_runs()
        one_run = next(r for r in runs if "roi-12_logreg" in r["config"]["run_id"] and "NYU" in r["config"]["run_id"])
        self.assertNotIn("sensitivity", one_run["metrics_test"].columns)
        self.assertIn("recall", one_run["metrics_test"].columns)


class A20AucRegressionGuardCatchesChangedValue(unittest.TestCase):
    def test_regression_guard_raises_on_changed_auc(self) -> None:
        baseline = pd.DataFrame([
            {"held_out_site": "NYU", "roi_set": "12", "model": "brainnetcnn", "auc_point": 0.5757},
            {"held_out_site": "OHSU", "roi_set": "12", "model": "brainnetcnn", "auc_point": 0.6310},
        ])
        unchanged = baseline.copy()
        A.assert_primary_outputs_unchanged(
            baseline=baseline, candidate=unchanged,
            key_cols=["held_out_site", "roi_set", "model"], value_cols=["auc_point"],
        )  # no debe levantar nada

        changed = baseline.copy()
        changed.loc[0, "auc_point"] = 0.999
        with self.assertRaises(SystemExit):
            A.assert_primary_outputs_unchanged(
                baseline=baseline, candidate=changed,
                key_cols=["held_out_site", "roi_set", "model"], value_cols=["auc_point"],
            )


if __name__ == "__main__":
    unittest.main()
