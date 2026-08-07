# Informe de verificación — primera tanda de ejecución del Plan v1.1

**Rama:** `paper/finalization-2026-08` (desde `4c15d39`)
**Fecha:** 2026-08-06/07

---

## 1. Qué se ejecutó

Solo las fases que el plan marca como iniciables sin depender de ningún gate,
más las que quedaron desbloqueadas al resolverlas dentro de esta tanda:

| Fase | Estado | Commit |
|---|---|---|
| F0 — Congelar baseline | Cerrado | `bd7765d` |
| F1.2 — Procedencia de hiperparámetros BrainNetCNN | Cerrado (documentado) | `070e5bb` |
| F1.3 — Gate G2 (`class_weight` Peking) | **PASS** | `070e5bb` |
| F2.1 — Auditoría de cohorte | Cerrado | `8d83231` |
| F2.2 — Demografía (Gate G3) | **PASS** | `8d83231` |
| F4-bis — Paquete de traspaso | Cerrado | `78a1df6` |
| F6 — Verificación de referencias | Cerrado | `2129fa2` |
| F7.1/F7.4 — Configuración de referencia y entornos | Cerrado | `41bb6e4` |
| F7.2/F7.3 — Documentación del repositorio | Cerrado | `94fdea4` |

## 2. Qué NO se ejecutó, y por qué

| Fase | Motivo |
|---|---|
| F1.1 — Gate G1 (procedencia de paneles ROI) | Requiere evidencia **fuera** del repositorio o autorización explícita para G1-C. No se activa unilateralmente. |
| F2.3 — Movimiento | Solo existe métrica comparable para NYU (G4). No se construye tabla parcial; queda como limitación en el paquete de traspaso. |
| F2.4 — Sensibilidad de `class_weight` | No aplica: G2 = PASS. |
| F3 — Edición de terminología ROI en Methods/Results | Bloqueada por G1. |
| F5 — Reparto main/suplemento | Bloqueada: falta revista y límite de páginas. |
| F8 — QA final | Fase posterior a F3/F5; no tiene sentido ejecutarla mientras esas sigan bloqueadas. |
| F9 — Tag `paper-submission-v1` | Solo tras pasar todos los checkpoints; no aplica aún. |

## 3. Verificación de integridad

```
ALL_IMMUTABLE_ARTIFACTS_UNCHANGED = TRUE
```

Los 13 artefactos declarados inmutables en el plan (§3) —incluidos los cuatro
`.joblib` de BOLD, `roi_sets.json`, los CSV canónicos de auditoría y
`analysis_manifest.json`— tienen exactamente el mismo SHA-256 que en el
congelamiento F0. `analysis_manifest.json` sigue en `PASS (16/16)`.

**Único archivo de resultados/manuscrito modificado:** el `.docx` oficial, y
solo por F6 (retiro de 2 citas no verificables, verificación de 1). Se
comprobó antes y después: 85 párrafos (antes 87, por las dos entradas de
referencia eliminadas), 5 tablas intactas, 4 imágenes intactas. Ningún AUC,
intervalo de confianza, métrica secundaria, tabla o figura fue alterado.

## 4. Nuevos artefactos versionados en esta tanda

```
docs/finalization/f0_freeze/baseline_hashes.md
docs/finalization/f1_gates.md
docs/finalization/limitations_handoff.md
docs/finalization/f6_refs/{PREVIO_antes_de_F6.docx, f6_referencias.md}
docs/data_provenance/adhd200_phenotypics.md
docs/paper_reference_configuration.md
docs/paper_environment.md
analysis/finalization/{build_cohort_audit.py, cohort_audit.csv,
                        build_demographics.py, demographics_by_site_dx.csv}
```

Ningún archivo bruto de fenotípico se versionó (se mantiene fuera del
repositorio, con procedencia + hash + script, conforme a lo acordado con el
equipo sobre licencia de datos ADHD-200).

## 5. Hallazgos que requieren decisión del equipo (sin cambiar nada más)

1. **G1 sigue abierto.** Se necesita evidencia externa de procedencia de los
   paneles ROI, o luz verde para G1-C.
2. **Naming inconsistente en 3 de las 16 corridas oficiales BrainNetCNN**
   (`Peking_rois116_w60s6_brainnetcnn_240732d1`,
   `NYU_rois39_w60s6_brainnetcnn_control_base_line_1521c348`,
   `OHSU_rois18_w48s5_brainnetcnn_2ce6c48e`): no llevan el tag
   `control_baseline_v13`, aunque sus hiperparámetros y resultados son
   consistentes con el resto de la familia. Documentado en
   `docs/paper_reference_configuration.md` §9; no se renombraron (el plan lo
   prohíbe).
3. **G4 (movimiento):** solo hay métrica para NYU. Si el equipo tiene una
   métrica comparable para Peking/NeuroIMAGE/OHSU, avisar antes de cerrar
   Limitations.
4. **Revista y límite de páginas:** sigue pendiente, bloquea F5.
5. **Destinatario del paquete de traspaso (F4-bis):** quién tiene Discussion,
   Limitations, Abstract e Introduction.

## 6. Estado del repositorio

```
HEAD: 94fdea4 (rama paper/finalization-2026-08)
```

Árbol limpio. 7 commits nuevos desde el freeze, cada uno de una sola fase,
conforme a la secuencia de F9. Ningún push realizado todavía — pendiente de
credenciales o de que el usuario decida fusionar/subir la rama.
