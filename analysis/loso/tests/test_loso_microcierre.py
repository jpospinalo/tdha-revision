#!/usr/bin/env python3
"""Tests del microcierre v31->v32 (``INSTRUCCIONES_MICROCIERRE_LOSO_STATIC_V1_V31_IA_VERIFICADO.md``).

Cubre los cuatro defectos corregidos:

  D1 — Gate U (regression gate) conectado obligatoriamente a ``main()``.
  D2 — lineage estable de ``original_analysis_source_git_sha`` (tag Git
       inmutable + referencia versionada, nunca el bootstrap manifest mutable).
  D3 — promoción con backup verificado por hash, ``os.replace()``, rollback
       transaccional a nivel de proceso, y journal de recuperación.
  D4 — QA A-X completo con evidencia real (R/S/T/U calculados; V/W/X vía
       ``--finalize-qa`` determinista).

Todos los tests usan directorios temporales para outputs/backups/journals;
ninguno toca ``analysis/loso/outputs/`` real ni ``results/loso/``.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
# Fixtures reales compartidas (solo lectura).
# ---------------------------------------------------------------------------

_CLOSEOUT_REFERENCE = None


def _closeout_reference() -> dict:
    global _CLOSEOUT_REFERENCE
    if _CLOSEOUT_REFERENCE is None:
        _CLOSEOUT_REFERENCE = A.load_closeout_reference()
    return _CLOSEOUT_REFERENCE


def _real_scientific_frames() -> dict[str, pd.DataFrame]:
    return {
        "metrics_by_run": pd.read_csv(A.OUTPUT_DIR / "loso_metrics_by_run.csv"),
        "metrics_summary": pd.read_csv(A.OUTPUT_DIR / "loso_metrics_summary.csv"),
        "contrasts": pd.read_csv(A.OUTPUT_DIR / "loso_contrasts.csv"),
        "convergence_summary": pd.read_csv(A.OUTPUT_DIR / "loso_convergence_summary.csv"),
    }


# ---------------------------------------------------------------------------
# D2 — lineage estable.
# ---------------------------------------------------------------------------


class D2LineageStability(unittest.TestCase):
    def test_closeout_reference_has_required_fields_and_valid_hashes(self) -> None:
        ref = _closeout_reference()
        self.assertEqual(ref["campaign_id"], "loso_static_v1")
        self.assertEqual(ref["primary_baseline_git_ref"], "loso-static-v1-complete")
        self.assertEqual(len(ref["primary_baseline_git_commit"]), 40)
        self.assertEqual(len(ref["v31_baseline_git_commit"]), 40)
        self.assertEqual(ref["original_analysis_source_git_sha"], "428cbc18f9b7e099d56bed91acd2fbc4f18ee6e8")

    def test_original_analysis_source_recovered_from_immutable_tag(self) -> None:
        ref = _closeout_reference()
        manifest = A.load_original_bootstrap_manifest_from_tag(ref)
        self.assertEqual(manifest["analysis_source_git_sha"], ref["original_analysis_source_git_sha"])

    def test_original_analysis_source_is_stable_across_closeout_reruns(self) -> None:
        """Sección 16: simular original=A, closeout_1=B, closeout_2=C ->
        original permanece A, sin importar qué closeout mutable haya
        sobrescrito el bootstrap manifest actualmente en disco."""

        ref = _closeout_reference()
        original_a = ref["original_analysis_source_git_sha"]

        # "closeout_1" produce B: nada en la fuente de lineage debe cambiar
        # por esto, porque load_original_bootstrap_manifest_from_tag() NUNCA
        # lee el archivo mutable en OUTPUT_DIR — solo el tag Git inmutable.
        manifest_rerun_1 = A.load_original_bootstrap_manifest_from_tag(ref)
        self.assertEqual(manifest_rerun_1["analysis_source_git_sha"], original_a)

        # Simular que un closeout previo (B) dejó un bootstrap_manifest.json
        # mutable con analysis_source_git_sha = B (el closeout, no el
        # original) — si la función leyera de ahí, "contaminaría" el lineage.
        tmp_root = Path(tempfile.mkdtemp(prefix="d2_stability_"))
        try:
            fake_output_dir = tmp_root
            fake_output_dir.mkdir(exist_ok=True)
            (fake_output_dir / "loso_bootstrap_manifest.json").write_text(
                json.dumps({"analysis_source_git_sha": "B" * 40}), encoding="utf-8"
            )
            original_output_dir = A.OUTPUT_DIR
            A.OUTPUT_DIR = fake_output_dir
            try:
                manifest_rerun_2 = A.load_original_bootstrap_manifest_from_tag(ref)
            finally:
                A.OUTPUT_DIR = original_output_dir
            # "closeout_2" (C) tampoco debe afectar el original recuperado.
            self.assertEqual(manifest_rerun_2["analysis_source_git_sha"], original_a)
            self.assertNotEqual(manifest_rerun_2["analysis_source_git_sha"], "B" * 40)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_closeout_reference_rejects_tampered_fixture(self) -> None:
        tmp_root = Path(tempfile.mkdtemp(prefix="d2_tamper_"))
        try:
            ref = _closeout_reference()
            # Copiar el reference JSON y los 5 fixtures a un árbol temporal,
            # luego corromper uno para verificar la autovalidación de hashes.
            (tmp_root / "analysis" / "loso" / "config" / "loso_primary_regression_reference").mkdir(parents=True)
            (tmp_root / "analysis" / "loso" / "config" / "loso_v31_regression_reference").mkdir(parents=True)
            for rel in (
                ref["primary_regression_reference"]["metrics_by_run"],
                ref["primary_regression_reference"]["metrics_summary"],
                ref["primary_regression_reference"]["contrasts"],
                ref["v31_regression_reference"]["metrics_summary"],
                ref["v31_regression_reference"]["convergence_summary"],
            ):
                shutil.copy2(REPO_ROOT / rel, tmp_root / rel)
            # Corromper un fixture DESPUÉS de haber congelado sus hashes.
            tampered = tmp_root / ref["primary_regression_reference"]["contrasts"]
            tampered.write_text(tampered.read_text() + "\n# tampered\n")

            tampered_ref_path = tmp_root / "loso_closeout_reference.json"
            tampered_ref_path.write_text(json.dumps(ref), encoding="utf-8")

            original_repo_root = A.REPO_ROOT
            A.REPO_ROOT = tmp_root
            try:
                with self.assertRaises(SystemExit) as ctx:
                    A.load_closeout_reference(tampered_ref_path)
                self.assertIn("modificado desde que se congeló su hash", str(ctx.exception))
            finally:
                A.REPO_ROOT = original_repo_root
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def test_closeout_reference_missing_file_raises(self) -> None:
        with self.assertRaises(SystemExit):
            A.load_closeout_reference(Path("/nonexistent/loso_closeout_reference.json"))


# ---------------------------------------------------------------------------
# D1 — Gate U (regression gate) conectado obligatoriamente.
# ---------------------------------------------------------------------------


class D1GateUPassesOnRealData(unittest.TestCase):
    def test_gate_u_passes_against_unmodified_real_outputs(self) -> None:
        frames = _real_scientific_frames()
        result = A.run_regression_gate_u(closeout_reference=_closeout_reference(), **frames)
        self.assertEqual(result, {"u1": "48/48", "u2": "16/16", "u3": "12/12", "u4": "16/16", "u5": "8/8"})


class D1GateUBlocksChanges(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = {k: v.copy() for k, v in _real_scientific_frames().items()}
        self.ref = _closeout_reference()

    def test_changed_auc_blocks(self) -> None:
        self.frames["metrics_summary"].loc[0, "auc_point"] = 0.999999
        with self.assertRaises(SystemExit) as ctx:
            A.run_regression_gate_u(closeout_reference=self.ref, **self.frames)
        self.assertIn("U2", str(ctx.exception))

    def test_changed_contrast_blocks(self) -> None:
        self.frames["contrasts"].loc[0, "delta_point"] = 0.5
        with self.assertRaises(SystemExit) as ctx:
            A.run_regression_gate_u(closeout_reference=self.ref, **self.frames)
        self.assertIn("U3", str(ctx.exception))

    def test_changed_secondary_metric_blocks(self) -> None:
        self.frames["metrics_summary"].loc[0, "balanced_accuracy_point"] = 0.123456
        with self.assertRaises(SystemExit) as ctx:
            A.run_regression_gate_u(closeout_reference=self.ref, **self.frames)
        self.assertIn("U4", str(ctx.exception))

    def test_changed_convergence_blocks(self) -> None:
        self.frames["convergence_summary"].loc[0, "epochs_ran_mean"] = 1.0
        with self.assertRaises(SystemExit) as ctx:
            A.run_regression_gate_u(closeout_reference=self.ref, **self.frames)
        self.assertIn("U5", str(ctx.exception))

    def test_changed_metrics_by_run_blocks(self) -> None:
        self.frames["metrics_by_run"].loc[0, "auc"] = 0.0101
        with self.assertRaises(SystemExit) as ctx:
            A.run_regression_gate_u(closeout_reference=self.ref, **self.frames)
        self.assertIn("U1", str(ctx.exception))

    def test_missing_key_blocks(self) -> None:
        self.frames["metrics_summary"] = self.frames["metrics_summary"].iloc[1:].reset_index(drop=True)
        with self.assertRaises(SystemExit):
            A.run_regression_gate_u(closeout_reference=self.ref, **self.frames)

    def test_duplicate_key_blocks(self) -> None:
        dup = pd.concat([self.frames["metrics_summary"], self.frames["metrics_summary"].iloc[[0]]], ignore_index=True)
        self.frames["metrics_summary"] = dup
        with self.assertRaises(SystemExit):
            A.run_regression_gate_u(closeout_reference=self.ref, **self.frames)


class D1RegressionFailureCausesZeroPromotion(unittest.TestCase):
    """Test conductual (Sección 21): si Gate U falla, no se genera ni
    promueve ningún output — el directorio de outputs queda exactamente como
    estaba antes de correr el analyzer."""

    def test_zero_promotion_on_regression_failure(self) -> None:
        tmp_out = Path(tempfile.mkdtemp(prefix="d1_zero_promotion_"))
        try:
            for f in A.OUTPUT_DIR.glob("*"):
                if f.is_file():
                    shutil.copy2(f, tmp_out / f.name)
            before = {f.name: f.stat().st_size for f in tmp_out.glob("*")}
            before_hashes = {f.name: A._full_sha256_file(f) for f in tmp_out.glob("*")}

            original_output_dir = A.OUTPUT_DIR
            A.OUTPUT_DIR = tmp_out

            original_gate_u = A.run_regression_gate_u

            def tampered_gate_u(**kwargs):
                kwargs["metrics_summary"] = kwargs["metrics_summary"].copy()
                kwargs["metrics_summary"].loc[0, "auc_point"] = 0.4242
                return original_gate_u(**kwargs)

            with mock.patch.object(A, "run_regression_gate_u", side_effect=tampered_gate_u):
                try:
                    with self.assertRaises(SystemExit):
                        A.run_analyzer()
                finally:
                    A.OUTPUT_DIR = original_output_dir

            after = {f.name: f.stat().st_size for f in tmp_out.glob("*") if not f.name.startswith(".")}
            after_hashes = {f.name: A._full_sha256_file(f) for f in tmp_out.glob("*") if not f.name.startswith(".")}
            self.assertEqual(before, after, "el tamaño de los outputs cambió pese al fallo de Gate U")
            self.assertEqual(before_hashes, after_hashes, "el contenido de los outputs cambió pese al fallo de Gate U")
            # Ningún residuo de staging/journal/backup debe quedar.
            self.assertEqual(list(tmp_out.glob(".staging-*")), [])
            self.assertEqual(list(tmp_out.glob(".backup-*")), [])
            self.assertEqual(list(tmp_out.glob(A.PROMOTION_JOURNAL_GLOB)), [])
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)


# ---------------------------------------------------------------------------
# D3 — promoción segura.
# ---------------------------------------------------------------------------


class D3PromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = Path(tempfile.mkdtemp(prefix="d3_"))
        self.out = self.tmp_root / "out"
        self.staging = self.tmp_root / "staging"
        self.out.mkdir()
        self.staging.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_successful_promotion(self) -> None:
        (self.out / "a.txt").write_text("old-a")
        (self.staging / "a.txt").write_text("new-a")
        (self.staging / "b.txt").write_text("new-b")
        A._promote_outputs_with_rollback(staging_dir=self.staging, output_dir=self.out, final_names=("a.txt", "b.txt"))
        self.assertEqual((self.out / "a.txt").read_text(), "new-a")
        self.assertEqual((self.out / "b.txt").read_text(), "new-b")
        self.assertEqual(list(self.out.glob(".backup-*")), [])
        self.assertEqual(list(self.out.glob(A.PROMOTION_JOURNAL_GLOB)), [])

    def test_failure_before_first_replace_leaves_canonical_untouched(self) -> None:
        (self.out / "a.txt").write_text("old-a")
        (self.staging / "a.txt").write_text("new-a")
        with mock.patch.object(A, "_full_sha256_file", side_effect=RuntimeError("boom during backup")):
            with self.assertRaises(SystemExit):
                A._promote_outputs_with_rollback(staging_dir=self.staging, output_dir=self.out, final_names=("a.txt",))
        self.assertEqual((self.out / "a.txt").read_text(), "old-a")
        self.assertEqual(list(self.out.glob(".backup-*")), [])
        self.assertEqual(list(self.out.glob(A.PROMOTION_JOURNAL_GLOB)), [])

    def test_failure_mid_promotion_rolls_back_all_replaced_files(self) -> None:
        (self.out / "a.txt").write_text("old-a")
        (self.out / "b.txt").write_text("old-b")
        (self.staging / "a.txt").write_text("new-a")
        (self.staging / "b.txt").write_text("new-b")
        # c.txt no existe en staging -> falla al llegar a c.txt, DESPUÉS de
        # haber reemplazado a.txt y b.txt.
        with self.assertRaises(SystemExit):
            A._promote_outputs_with_rollback(
                staging_dir=self.staging, output_dir=self.out, final_names=("a.txt", "b.txt", "c.txt"),
            )
        self.assertEqual((self.out / "a.txt").read_text(), "old-a")
        self.assertEqual((self.out / "b.txt").read_text(), "old-b")
        self.assertFalse((self.out / "c.txt").exists())
        self.assertEqual(list(self.out.glob(".backup-*")), [])
        self.assertEqual(list(self.out.glob(A.PROMOTION_JOURNAL_GLOB)), [])

    def test_new_file_rollback_removes_created_file(self) -> None:
        (self.staging / "new1.txt").write_text("n1")
        with self.assertRaises(SystemExit):
            A._promote_outputs_with_rollback(
                staging_dir=self.staging, output_dir=self.out, final_names=("new1.txt", "missing.txt"),
            )
        self.assertFalse((self.out / "new1.txt").exists())

    def test_restored_hash_equals_original_after_rollback(self) -> None:
        (self.out / "a.txt").write_text("original content, exactly this")
        original_hash = A._full_sha256_file(self.out / "a.txt")
        (self.staging / "a.txt").write_text("candidate content")
        with self.assertRaises(SystemExit):
            A._promote_outputs_with_rollback(
                staging_dir=self.staging, output_dir=self.out, final_names=("a.txt", "missing.txt"),
            )
        self.assertEqual(A._full_sha256_file(self.out / "a.txt"), original_hash)

    def test_stale_journal_causes_stop(self) -> None:
        (self.out / A.PROMOTION_JOURNAL_GLOB.replace("*", "deadbeef")).write_text("{}")
        with self.assertRaises(SystemExit) as ctx:
            A.check_no_stale_promotion_journal(self.out)
        self.assertIn("promotion journal previo", str(ctx.exception))

    def test_no_stale_journal_is_silent(self) -> None:
        A.check_no_stale_promotion_journal(self.out)  # no debe levantar nada


# ---------------------------------------------------------------------------
# D4 — QA A-X y finalización determinista.
# ---------------------------------------------------------------------------


class D4QaDocStructure(unittest.TestCase):
    def test_qa_doc_contains_all_gates_a_through_x(self) -> None:
        gate_rows = [
            A._gate_row(letter, f"desc {letter}", "exp", "obs", "PASS")
            for letter in "ABCDEFGHIJKLMNOPQRSTU"
        ]
        doc = A.build_qa_doc(gate_rows=gate_rows, provenance_manifest_file_sha256="f" * 64)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWX":
            self.assertIn(f"| {letter} |", doc, f"falta la fila del gate {letter}")
        for letter in "VWX":
            self.assertIn("PENDING", doc)


class D4FinalizeQa(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = Path(tempfile.mkdtemp(prefix="d4_"))
        self.out = self.tmp_root / "out"
        self.out.mkdir()
        gate_rows = [A._gate_row(letter, f"desc {letter}", "exp", "obs", "PASS") for letter in "ABCDEFGHIJKLMNOPQRSTU"]
        doc = A.build_qa_doc(gate_rows=gate_rows, provenance_manifest_file_sha256="f" * 64)
        (self.out / "LOSO_STATIC_V1_QA.md").write_text(doc, encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_finalize_qa_all_pass(self) -> None:
        raw_log = self.tmp_root / "raw.log"
        raw_log.write_text("f1: OK\nf2: OK\n")
        hist_log = self.tmp_root / "hist.log"
        hist_log.write_text("f1: OK\n")
        tests_log = self.tmp_root / "tests.log"
        tests_log.write_text("....\n\n----------------------------------------------------------------------\nRan 56 tests in 18.0s\n\nOK\n")

        A.finalize_qa(raw_integrity_log=raw_log, historical_integrity_log=hist_log, tests_log=tests_log, output_dir=self.out)
        text = (self.out / "LOSO_STATIC_V1_QA.md").read_text()
        self.assertIn("| V | ", text)
        self.assertNotIn("| V | Raw LOSO integrity", "")  # sanity no-op
        self.assertIn("2/2 OK", text)
        self.assertIn("tests_run=56, failures=0, errors=0, skipped=0", text)
        # El texto narrativo de cabecera menciona la palabra "PENDING" al
        # explicar el mecanismo; lo que debe desaparecer es el ESTADO
        # PENDING en las filas V/W/X de la tabla.
        self.assertNotRegex(text, r"\| (V|W|X) \|.*\| PENDING \| PENDING \|")

    def test_finalize_qa_detects_failed_hash_check(self) -> None:
        raw_log = self.tmp_root / "raw.log"
        raw_log.write_text("f1: OK\nf2: FAILED\n")
        hist_log = self.tmp_root / "hist.log"
        hist_log.write_text("f1: OK\n")
        tests_log = self.tmp_root / "tests.log"
        tests_log.write_text("Ran 56 tests in 18.0s\n\nOK\n")

        A.finalize_qa(raw_integrity_log=raw_log, historical_integrity_log=hist_log, tests_log=tests_log, output_dir=self.out)
        text = (self.out / "LOSO_STATIC_V1_QA.md").read_text()
        self.assertIn("| V | ", text)
        self.assertRegex(text, r"\| V \|.*\| FAIL \|")

    def test_finalize_qa_detects_test_failures(self) -> None:
        raw_log = self.tmp_root / "raw.log"
        raw_log.write_text("f1: OK\n")
        hist_log = self.tmp_root / "hist.log"
        hist_log.write_text("f1: OK\n")
        tests_log = self.tmp_root / "tests.log"
        tests_log.write_text(
            "test_x ... FAIL\n\n----------------------------------------------------------------------\n"
            "Ran 56 tests in 18.0s\n\nFAILED (failures=1, errors=0)\n"
        )
        A.finalize_qa(raw_integrity_log=raw_log, historical_integrity_log=hist_log, tests_log=tests_log, output_dir=self.out)
        text = (self.out / "LOSO_STATIC_V1_QA.md").read_text()
        self.assertRegex(text, r"\| X \|.*failures=1.*\| FAIL \|")

    def test_finalize_qa_does_not_recompute_any_science(self) -> None:
        """No debe tocar A-U ni ningún otro output científico: solo V/W/X."""

        (self.out / "loso_metrics_summary.csv").write_text("sentinel-untouched")
        raw_log = self.tmp_root / "raw.log"; raw_log.write_text("f1: OK\n")
        hist_log = self.tmp_root / "hist.log"; hist_log.write_text("f1: OK\n")
        tests_log = self.tmp_root / "tests.log"; tests_log.write_text("Ran 1 tests in 0.0s\n\nOK\n")
        A.finalize_qa(raw_integrity_log=raw_log, historical_integrity_log=hist_log, tests_log=tests_log, output_dir=self.out)
        self.assertEqual((self.out / "loso_metrics_summary.csv").read_text(), "sentinel-untouched")
        text = (self.out / "LOSO_STATIC_V1_QA.md").read_text()
        for letter in "ABCDEFGHIJKLMNOPQRSTU":
            self.assertIn(f"| {letter} | desc {letter} | exp | obs | PASS |", text)

    def test_finalize_qa_missing_gate_row_raises(self) -> None:
        (self.out / "LOSO_STATIC_V1_QA.md").write_text("# no table here\n", encoding="utf-8")
        raw_log = self.tmp_root / "raw.log"; raw_log.write_text("f1: OK\n")
        hist_log = self.tmp_root / "hist.log"; hist_log.write_text("f1: OK\n")
        tests_log = self.tmp_root / "tests.log"; tests_log.write_text("Ran 1 tests in 0.0s\n\nOK\n")
        with self.assertRaises(SystemExit):
            A.finalize_qa(raw_integrity_log=raw_log, historical_integrity_log=hist_log, tests_log=tests_log, output_dir=self.out)

    def test_finalize_qa_requires_existing_qa_doc(self) -> None:
        empty_out = self.tmp_root / "empty_out"
        empty_out.mkdir()
        raw_log = self.tmp_root / "raw.log"; raw_log.write_text("f1: OK\n")
        hist_log = self.tmp_root / "hist.log"; hist_log.write_text("f1: OK\n")
        tests_log = self.tmp_root / "tests.log"; tests_log.write_text("Ran 1 tests in 0.0s\n\nOK\n")
        with self.assertRaises(SystemExit):
            A.finalize_qa(raw_integrity_log=raw_log, historical_integrity_log=hist_log, tests_log=tests_log, output_dir=empty_out)


class D4CliDispatch(unittest.TestCase):
    def test_main_finalize_qa_requires_all_three_logs(self) -> None:
        with self.assertRaises(SystemExit):
            A.main(["--finalize-qa"])

    def test_build_parser_has_finalize_qa_flags(self) -> None:
        parser = A.build_parser()
        dests = {a.dest for a in parser._actions}
        self.assertIn("finalize_qa", dests)
        self.assertIn("raw_integrity_log", dests)
        self.assertIn("historical_integrity_log", dests)
        self.assertIn("tests_log", dests)


if __name__ == "__main__":
    unittest.main()
